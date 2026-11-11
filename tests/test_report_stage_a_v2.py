from copy import deepcopy
import json

import pytest

from scripts import report_stage_a_v2 as report


def arm(value=100, *, status="ok", accepted=True):
    return {"status": status, "fresh_generation": True, "cross_arm_candidate_reuse": False,
            "verification": {"accepted": accepted, "formal": False, "proof": "differential"},
            "measurement": {"status": status, "samples_ms": [value] * 3 if status == "ok" else [],
                            "runs": 3, "warmups": 1, "scope": "final_evaluation"}}


def record(number, benchmark="tpch", **values):
    return {"query_id": f"{benchmark}:{number}", "benchmark": benchmark,
            "original_sql": f"SELECT {number};", "status": "ok", "verification_mode": "hybrid",
            "study_variant": {"version": 2, "feedback_scope": "transductive_stage_a_followup",
                              "historical_stage_a_records_modified": False},
            "arms": {name: arm(values.get(name, 80 if name == "mcts" else 100)) for name in report.ARMS},
            "model_call_audit": {"calls": [], "total_invocations": 0, "invocations_by_arm": {}}}


@pytest.fixture
def data(tmp_path):
    rows = [record(number, benchmark) for benchmark in report.BENCHMARKS for number in range(10)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"id": row["query_id"], "benchmark": row["benchmark"],
                                          "sql": row["original_sql"]}) + "\n" for row in rows))
    raw = tmp_path / "raw.jsonl"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"timeout_ms": 300}))
    def build():
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return report.build_report(raw, manifest, config)
    return rows, raw, manifest, config, build


def test_same_four_system_cohort_statistics_and_baseline_comparisons(data):
    rows, _, _, _, build = data
    rows[0]["arms"]["rbot"] = arm(50)
    rows[0]["arms"]["mcts"] = arm(90)
    result = build()
    for name in ("strict_finite_only", "timeout_capped_T_max"):
        cohort = result[name]
        assert cohort["paired_queries"] == 30
        assert all(value["queries"] == 30 for value in cohort["systems"].values())
        assert cohort["systems"]["mcts"]["mean_ms"] == pytest.approx((29 * 80 + 90) / 30)
        assert cohort["systems"]["mcts"]["median_ms"] == 80
        assert cohort["systems"]["mcts"]["p90_ms"] == 80
        assert cohort["systems"]["mcts"]["improvement_ge_10pct_vs_original_count"] == 30
        assert cohort["mcts_vs_baselines"]["rbot"]["regression_ids"] == [rows[0]["query_id"]]
        assert all(value["paired_query_ids"] == cohort["paired_query_ids"]
                   for value in cohort["mcts_vs_baselines"].values())
    assert result["per_benchmark"]["tpch"]["strict_finite_only"]["paired_queries"] == 10


def test_genuine_timing_timeout_only_enters_capped_cohort(data):
    rows, _, _, _, build = data
    rows[0]["status"] = "original_timeout"
    rows[0]["arms"]["original"] = arm(status="timeout")
    result = build()
    assert result["strict_finite_only"]["paired_queries"] == 29
    capped = result["timeout_capped_T_max"]
    assert capped["paired_queries"] == 30
    assert capped["systems"]["original"]["mean_ms"] == pytest.approx((300 + 29 * 100) / 30)
    assert capped["systems"]["original"]["timing_timeout_count"] == 1
    assert result["uncapped_raw_metrics"]["original"]["queries"] == 29


@pytest.mark.parametrize("status", ["generation_error", "rejected_verification", "skipped", "error", "running"])
def test_other_arm_statuses_never_receive_timeout_value(data, status):
    rows, _, _, _, build = data
    rows[0]["status"] = "partial_error"
    rows[0]["arms"]["mcts"] = arm(status="timeout")
    rows[0]["arms"]["mcts"]["status"] = status
    result = build()
    assert result["timeout_capped_T_max"]["paired_queries"] == 30
    detail = next(detail for detail in result["query_system_details"]
                  if detail["query_id"] == rows[0]["query_id"] and detail["system"] == "mcts")
    assert detail["timeout_capped_value"] is not None


@pytest.mark.parametrize("mutation", [
    lambda row: row["arms"]["mcts"].update(generation_errors=[{"error": "worker launch"}]),
    lambda row: row["arms"]["mcts"].update(generation_error={"error": "worker launch"}),
    lambda row: row["arms"]["mcts"].update(terminal_generation_failures=[{"failure_code": "model_capacity"}]),
    lambda row: row["arms"]["mcts"]["verification"].update(accepted=False),
    lambda row: row["arms"]["mcts"].update(fresh_generation=False),
    lambda row: row["arms"]["mcts"].update(cross_arm_candidate_reuse=True),
    lambda row: row["arms"]["mcts"]["measurement"].update(scope="search"),
    lambda row: row["arms"]["mcts"]["measurement"].update(runs=2),
    lambda row: row["arms"]["mcts"]["measurement"].update(warmups=0),
])
def test_timeout_does_not_mask_invalid_evidence(data, mutation):
    rows, _, _, _, build = data
    rows[0]["arms"]["mcts"] = arm(status="timeout")
    rows[0]["status"] = "partial_error"
    mutation(rows[0])
    # Transport diagnostics and retry provenance do not invalidate a valid
    # final timeout measurement; semantic/protocol evidence still does.
    result = build()
    invalid_evidence = (rows[0]["arms"]["mcts"]["verification"].get("accepted") is False
                        or rows[0]["arms"]["mcts"]["measurement"].get("scope") != "final_evaluation"
                        or rows[0]["arms"]["mcts"]["measurement"].get("runs") != 3
                        or rows[0]["arms"]["mcts"]["measurement"].get("warmups") != 1)
    assert result["timeout_capped_T_max"]["paired_queries"] == (29 if invalid_evidence else 30)


def test_uncapped_raw_timing_is_retained_for_generation_failure_but_not_comparison(data):
    rows, _, _, _, build = data
    rows[0]["arms"]["mcts"].update(status="generation_error", generation_errors=[{"error": "failed"}])
    rows[0]["status"] = "partial_error"
    result = build()
    assert result["strict_finite_only"]["paired_queries"] == 30
    assert result["timeout_capped_T_max"]["paired_queries"] == 30
    assert result["uncapped_raw_metrics"]["mcts"]["queries"] == 30


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), 0, -1, True, None])
def test_nonfinite_or_invalid_samples_are_excluded_from_both_cohorts(data, sample, tmp_path):
    rows, _, _, _, build = data
    rows[0]["arms"]["mcts"]["measurement"]["samples_ms"] = [100, sample, 100]
    result = build()
    assert result["strict_finite_only"]["paired_queries"] == 29
    assert result["timeout_capped_T_max"]["paired_queries"] == 29
    report.write_report(tmp_path / "nonfinite_report", result)
    assert "NaN" not in (tmp_path / "nonfinite_report/summary.json").read_text()


def test_capped_success_samples_and_uncapped_metrics_are_separate(data):
    rows, _, _, _, build = data
    rows[0]["arms"]["mcts"]["measurement"]["samples_ms"] = [250, 400, 600]
    result = build()
    assert result["strict_finite_only"]["systems"]["mcts"]["mean_ms"] == pytest.approx((29 * 80 + 1250 / 3) / 30)
    assert result["timeout_capped_T_max"]["systems"]["mcts"]["mean_ms"] == pytest.approx((29 * 80 + 850 / 3) / 30)
    assert result["timeout_capped_T_max"]["systems"]["mcts"]["clipped_sample_count"] == 2
    assert result["uncapped_raw_metrics"]["mcts"]["p90_ms"] == 80


def test_failed_model_calls_and_invalid_budgets_exclude_entire_query(data):
    rows, _, _, _, build = data
    rows[0]["model_call_audit"] = {"calls": [{"arm": "mcts", "status": "error"}], "total_invocations": 1}
    rows[1]["model_call_audit"]["total_invocations"] = 9
    result = build()
    assert result["strict_finite_only"]["paired_queries"] == 30
    assert result["timeout_capped_T_max"]["paired_queries"] == 30


def test_partial_input_has_nulls_and_writes_markdown_without_mutating_raw(data, tmp_path):
    rows, raw, _, _, build = data
    rows.clear()
    result = build()
    before = raw.read_bytes()
    assert result["status"] == "incomplete_records"
    assert result["strict_finite_only"]["systems"]["mcts"]["mean_ms"] is None
    destination = tmp_path / "report"
    report.write_report(destination, result)
    assert raw.read_bytes() == before
    assert (destination / "REPORT.md").exists()
    assert (destination / "uncapped_raw_metrics.json").exists()
    assert json.loads((destination / "summary.json").read_text())["timeout_capped_T_max"]["paired_queries"] == 0
    with pytest.raises(ValueError, match="separate report"):
        report.write_report(raw.parent, result)


def test_timeout_configuration_must_be_explicit_positive_and_finite(data):
    _, raw, manifest, config, build = data
    build()
    for value in (None, 0, -1, True, float("inf")):
        config.write_text(json.dumps({"timeout_ms": value}))
        with pytest.raises(ValueError, match="configured SQL timeout"):
            report.build_report(raw, manifest, config)


def test_duplicate_and_out_of_manifest_records_rejected(data):
    rows, _, _, _, build = data
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="Duplicate raw"):
        build()
    rows[-1]["query_id"] = "unknown"
    with pytest.raises(ValueError, match="outside the expected manifest"):
        build()


def test_historical_records_are_rejected(data):
    rows, _, _, _, build = data
    rows[0].pop("study_variant")
    with pytest.raises(ValueError, match="labeled Stage A v2"):
        build()


def test_p90_interpolation_and_ten_percent_boundary():
    assert report.p90([1, 2, 3, 4]) == pytest.approx(3.7)
    assert report._at_least_ten_percent(90, 100)
    assert not report._at_least_ten_percent(90.01, 100)
    assert not report._at_least_ten_percent(100 / 1.1, 100)
