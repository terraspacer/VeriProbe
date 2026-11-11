import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_stage_a", ROOT / "scripts/report_stage_a.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def arm(samples=None, *, proof="differential", formal=False, status="ok", fallback=False, accepted=True):
    return {"status": status, "fallback": fallback,
            "verification": {"accepted": accepted, "proof": proof, "formal": formal},
            "measurement": {"status": "ok" if samples else "timeout", "samples_ms": samples or [], "runs": 3, "warmups": 1}}


def call_ledger(counts):
    calls = []
    for system, count in counts.items():
        for index in range(count):
            calls.append({"arm": system, "status": "ok", "arm_call_index": index + 1,
                          "input_call_index": len(calls) + 1, "transport_invoked": True,
                          "provenance": {"actual_model": "gpt-6.1-sol", "session_transport": {
                              "actual_serving_model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "request_id": "req_test"}}})
    return {"calls": calls, "total_invocations": len(calls), "invocations_by_arm": counts,
            "blocked_calls": [{"transport_invoked": False}]}


@pytest.fixture
def synthetic(tmp_path):
    manifest = tmp_path / "inputs/manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("".join(json.dumps({"id": f"q{number}", "benchmark": "tpch", "sql": f"SELECT {number};",
                                          "source_path": f"vendor/q{number}.sql", "source_commit": "abc123"}) + "\n"
                                 for number in range(1, 5)))
    rows = [
        {"query_id": "q1", "benchmark": "tpch", "original_sql": "SELECT 1;", "status": "ok",
         "arms": {"original": arm([90, 100, 110], proof="identity", formal=True),
                  "rbot": arm([70, 80, 90]), "quite": arm([90, 100, 110], proof="ast", formal=True),
                  "mcts": arm([65, 70, 90])}, "model_call_audit": call_ledger({"rbot": 2, "quite": 3, "mcts": 3})},
        {"query_id": "q2", "benchmark": "tpch", "original_sql": "SELECT 2;", "status": "ok",
         "arms": {"original": arm([190, 200, 210], proof="identity", formal=True),
                  "rbot": arm([210, 220, 230], proof="sqlsolver", formal=True), "quite": arm([170, 180, 190]),
                  "mcts": arm([140, 150, 190])}, "model_call_audit": call_ledger({"rbot": 2, "quite": 3, "mcts": 3})},
        {"query_id": "q3", "benchmark": "tpch", "original_sql": "SELECT 3;", "status": "original_timeout",
         "arms": {"original": arm(proof="identity", formal=True, status="timeout"),
                  "rbot": arm([170, 180, 190]), "quite": arm(status="timeout"),
                  "mcts": arm(status="timeout", proof="identity", formal=True, fallback=True)},
         "model_call_audit": call_ledger({"rbot": 3, "quite": 3, "mcts": 3})},
    ]
    rows[2]["model_call_audit"]["calls"][0]["status"] = "error"
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return raw, manifest, rows


def metric(report, system, cohort="available", benchmark="overall"):
    return next(row for row in report["metrics"] if row["system"] == system and row["cohort"] == cohort and row["benchmark"] == benchmark)


def test_hand_calculated_available_and_common_cohorts(synthetic):
    raw, manifest, _ = synthetic
    report = reporter.build_report(raw, manifest)
    rbot = metric(report, "rbot")
    assert rbot["expected_queries"] == 4
    assert rbot["completed_valid_final_queries"] == 3
    assert rbot["completed_coverage"] == .75
    assert rbot["arithmetic_average_query_mean_ms"] == 160  # (80 + 220 + 180) / 3
    assert rbot["median_query_median_ms"] == 180
    assert rbot["p90_query_median_ms"] == pytest.approx(212)  # 180 + .8 * (220 - 180)
    assert rbot["paired_original_queries"] == 2
    assert rbot["improved_at_least_10pct_latency_reduction_count"] == 1
    assert rbot["improvement_rate_paired"] == .5
    assert rbot["improvement_rate_expected"] == .25
    assert rbot["strict_regression_count"] == 1
    assert rbot["strict_regression_rate_paired"] == .5
    common = metric(report, "rbot", "common_complete")
    assert common["timing_cohort_query_ids"] == ["q1", "q2"]
    assert common["arithmetic_average_query_mean_ms"] == 150
    assert common["median_query_median_ms"] == 150
    assert common["p90_query_median_ms"] == 206
    mcts = metric(report, "mcts")
    assert mcts["arithmetic_average_query_mean_ms"] == 117.5  # query means 75 and 160; do not average medians 70 and 150
    assert mcts["median_query_median_ms"] == 110
    assert mcts["p90_query_median_ms"] == 142
    assert report["scope"]["common_complete_queries"] == 2
    assert report["scope"]["full_suite_performance_estimate"] is False
    assert "0.90" in report["definitions"]["improvement"]
    assert "1.10" in report["definitions"]["additional_speedup_factor"]


def test_equivalence_counts_do_not_turn_finite_evidence_into_formal(synthetic):
    report = reporter.build_report(*synthetic[:2])
    rbot = metric(report, "rbot")
    assert rbot["final_accepted_equivalence_count"] == 3
    assert rbot["final_accepted_equivalence_rate_expected"] == .75
    assert rbot["final_accepted_equivalence_rate_evaluated"] == 1
    assert rbot["finite_differential_count"] == 2
    assert rbot["formal_nonidentity_count"] == 1
    assert rbot["identity_or_conservative_ast_count"] == 0
    assert metric(report, "original")["identity_or_conservative_ast_count"] == 3
    assert metric(report, "quite")["identity_or_conservative_ast_count"] == 1
    assert metric(report, "mcts")["fallback_count_all_present"] == 1


def test_censored_and_missing_timings_stay_null(synthetic):
    report = reporter.build_report(*synthetic[:2])
    q3 = next(row for row in report["query_system_details"] if row["query_id"] == "q3" and row["system"] == "original")
    assert q3["mean_ms"] is q3["median_ms"] is q3["median_speedup_vs_original"] is None
    q3_rbot = next(row for row in report["query_system_details"] if row["query_id"] == "q3" and row["system"] == "rbot")
    assert q3_rbot["median_ms"] == 180
    assert q3_rbot["median_speedup_vs_original"] is None
    assert q3_rbot["strict_regression_vs_original"] is None
    q4 = next(row for row in report["query_model_audits"] if row["query_id"] == "q4")
    assert q4["total_calls"] is q4["rbot_calls"] is None
    assert metric(report, "original")["missing_timing_reason_counts"] == {"measurement_timeout": 1, "measurement_missing": 1}


def test_model_call_caps_failures_and_source_receipts(synthetic):
    raw, manifest, _ = synthetic
    report = reporter.build_report(raw, manifest)
    q3 = next(row for row in report["query_model_audits"] if row["query_id"] == "q3")
    assert q3["total_calls"] == 9
    assert q3["rbot_calls"] == 3
    assert q3["failed_calls"] == 1
    assert q3["blocked_calls"] == 1
    assert q3["cap_violated"] is True
    assert "actual_total_exceeds_8" in q3["cap_violation_reasons"]
    assert "actual_rbot_exceeds_2" in q3["cap_violation_reasons"]
    first = report["model_call_details"][0]
    assert first["actual_model"] == "gpt-6.1-sol"
    assert first["reasoning_effort"] == "xhigh"
    assert first["raw_source"] == str(raw.resolve()) + ":1"
    assert first["raw_sha256"] == reporter.sha256(raw.read_bytes())
    assert first["record_line_sha256"] == reporter.sha256(raw.read_bytes().splitlines()[0])
    assert report["source"]["manifest_sha256"] == reporter.sha256(manifest.read_bytes())


def test_helper_writes_only_report_directory_and_no_runtime_calls(synthetic):
    raw, manifest, _ = synthetic
    before_raw, before_manifest = raw.read_bytes(), manifest.read_bytes()
    assert reporter.main(["--raw", str(raw)]) == 0
    output = raw.parent / "report"
    assert {path.name for path in output.iterdir()} == {
        "summary.json", "system_benchmark_metrics.csv", "query_system_details.csv", "query_model_audits.csv",
        "model_call_details.csv", "source_and_definitions.json"}
    assert raw.read_bytes() == before_raw and manifest.read_bytes() == before_manifest
    assert "batched standalone variant" in (output / "system_benchmark_metrics.csv").read_text()
    assert "native_baseline_reproduction" in (output / "summary.json").read_text()
    with pytest.raises(ValueError, match="separate report"):
        reporter.write_report(raw.parent, reporter.build_report(raw, manifest))


def test_duplicate_or_nonmanifest_inputs_fail_closed(synthetic):
    raw, manifest, rows = synthetic
    raw.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate raw"):
        reporter.build_report(raw, manifest)
    rows[0]["original_sql"] = "SELECT wrong;"
    raw.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="SQL differs"):
        reporter.build_report(raw, manifest)


def test_partial_and_empty_reports_have_no_fabricated_zero_latency(synthetic):
    raw, manifest, _ = synthetic
    raw.write_text("")
    report = reporter.build_report(raw, manifest)
    for row in report["metrics"]:
        assert row["arithmetic_average_query_mean_ms"] is None
        assert row["median_query_median_ms"] is None
        assert row["p90_query_median_ms"] is None
        assert row["improvement_rate_paired"] is None
    assert report["scope"]["raw_records_present"] == 0


def test_cap_violations_cannot_enter_error_free_common_cohort(synthetic):
    raw, manifest, rows = synthetic
    rows[1]["model_call_audit"] = call_ledger({"rbot": 3, "quite": 3, "mcts": 3})
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = reporter.build_report(raw, manifest)
    assert report["scope"]["common_complete_query_ids"] == ["q1"]
    assert metric(report, "rbot")["formal_accepted_total_including_identity_count"] == 1
    assert metric(report, "original")["formal_accepted_total_including_identity_count"] == 3
    assert metric(report, "rbot")["finite_differential_rate_expected"] == .5


def test_filtered_noninvocations_preserve_actual_raw_call_index(synthetic):
    raw, manifest, rows = synthetic
    rows[0]["model_call_audit"]["calls"].insert(0, {"transport_invoked": False, "arm": "rbot"})
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = reporter.build_report(raw, manifest)
    assert report["model_call_details"][0]["raw_json_pointer"] == "/model_call_audit/calls/1"
    assert report["query_model_audits"][0]["total_calls"] == 8


def test_primary_improvement_is_ten_percent_latency_reduction_boundary(tmp_path):
    rows, manifest_rows = [], []
    for index, system_ms in enumerate((91, 90.5, 90)):
        query_id = f"boundary{index}"
        manifest_rows.append({"id": query_id, "benchmark": "tpch", "sql": "SELECT 1;"})
        rows.append({"query_id": query_id, "benchmark": "tpch", "original_sql": "SELECT 1;", "status": "ok",
                     "arms": {"original": arm([100, 100, 100], proof="identity", formal=True),
                              **{system: arm([system_ms] * 3) for system in ("rbot", "quite", "mcts")}},
                     "model_call_audit": call_ledger({"rbot": 0, "quite": 0, "mcts": 0})})
    raw, manifest = tmp_path / "raw.jsonl", tmp_path / "manifest.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    report = reporter.build_report(raw, manifest)
    rbot = metric(report, "rbot")
    assert rbot["improved_at_least_10pct_latency_reduction_count"] == 1
    assert rbot["improvement_rate_paired"] == pytest.approx(1 / 3)
    assert rbot["speedup_factor_ge_1_10_count"] == 2
    query_rows = {row["query_id"]: row for row in report["query_system_details"] if row["system"] == "rbot"}
    assert query_rows["boundary0"]["improves_by_at_least_10pct_latency_reduction"] is False  # 91/100
    assert query_rows["boundary1"]["improves_by_at_least_10pct_latency_reduction"] is False  # 90.5/100
    assert query_rows["boundary1"]["speedup_factor_ge_1_10"] is True
    assert query_rows["boundary2"]["improves_by_at_least_10pct_latency_reduction"] is True  # 90/100
