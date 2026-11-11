import json
import math

import pytest

from benchmark_runner import evaluate_holdout, make_tasks, planned_groups, read_raw, run_task
from mcts_rewrite.reporting import build_report, latency_summary, summarize_group, verified, write_reports


def proof(formal=True):
    comparison = {"equivalent": True, "status": "ok", "same_snapshot": True, "original": {"sha256": "a"}, "candidate": {"sha256": "a"}}
    result = {"accepted": True, "formal": formal, "proof": "identity" if formal else "differential", "execution": comparison}
    if not formal:
        result["miniatures"] = [{"name": name, "comparison": comparison} for name in ("empty", "null", "duplicate", "skew", "unmatched")]
    return result


def record(query="q", original=100, rbot=90, quite=80, mcts=60):
    return {"record_type": "query_result", "query_id": query, "source": "rbot", "benchmark": "tpch",
            "dataset": "tpch_sf1", "ablation": "full", "budget": 16, "status": "ok",
            "evaluation": {arm: {"status": "ok", "median_ms": value} for arm, value in
                           (("original", original), ("rbot", rbot), ("quite", quite), ("mcts", mcts))},
            "verification": {arm: proof() for arm in ("original", "rbot", "quite", "mcts")}}


def test_common_cohort_excludes_timeout_and_preserves_all_query_denominator():
    good, timeout = record("good"), record("timeout")
    timeout["status"] = "incomplete"
    timeout["evaluation"]["rbot"] = {"status": "timeout", "median_ms": None, "samples_ms": [10]}
    report = summarize_group([good, timeout], 3)
    assert report["paired_complete_queries"] == 1
    assert all(report["paired_latency"][arm]["n"] == 1 for arm in report["paired_latency"])
    assert report["expected_queries"] == 3
    assert report["equivalence_rate_all_queries"] == pytest.approx(2 / 3)
    assert report["arm_status_counts"]["rbot"]["timeout"] == 1
    assert report["acceptance_status"] == "incomplete"
    assert report["targets"]["mean_latency_reduction_at_least_15_percent_vs_both"]["status"] == "incomplete"


def test_missing_baselines_never_become_success_or_zero_latency():
    row = record()
    row["status"] = "incomplete"
    row["evaluation"]["quite"] = {"status": "unavailable", "median_ms": None}
    report = summarize_group([row], 1)
    assert report["paired_latency"]["mcts"]["average_ms"] is None
    assert report["comparisons"]["quite"]["aggregate_latency_reduction"] is None
    assert report["acceptance_status"] == "incomplete"


def test_speedup_definition_and_strict_regression_target():
    # >=1.10x speedup is the requested improvement boundary, distinct from a
    # 10% reduction in elapsed time. The target remains a 15% latency reduction.
    result = summarize_group([record(rbot=110, quite=110, mcts=100)], 1)
    assert result["comparisons"]["rbot"]["improvement_rate"] == 1
    assert result["comparisons"]["rbot"]["aggregate_latency_reduction"] == pytest.approx(1 / 11)
    assert result["acceptance_status"] == "fail"
    rows = [record(str(i), mcts=60) for i in range(99)] + [record("regress", mcts=101)]
    result = summarize_group(rows, 100)
    assert result["comparisons"]["rbot"]["regression_rate"] == .01
    assert result["comparison_vs_original"]["regression_rate"] == .01
    assert result["targets"]["regression_below_1_percent_vs_original"]["status"] == "fail"


def test_weighted_latency_and_distribution_are_not_mean_query_speedups():
    rows = [record("big", rbot=100, quite=100, mcts=70), record("small", rbot=1, quite=1, mcts=2)]
    result = summarize_group(rows, 2)
    assert result["comparisons"]["rbot"]["aggregate_latency_reduction"] == pytest.approx(1 - 72 / 101)
    assert result["comparisons"]["rbot"]["regression_rate"] == .5
    distribution = latency_summary([1, 2, 10, None, math.inf, math.nan])
    assert distribution["average_ms"] == pytest.approx(13 / 3)
    assert distribution["median_ms"] == 2
    assert distribution["p90_ms"] == pytest.approx(8.4)


def test_formal_unknown_never_counted_as_verified():
    evidence = {"accepted": True, "formal": False, "proof": "none", "execution": {"equivalent": True}}
    assert not verified(evidence)
    row = record()
    row["verification"]["mcts"] = evidence
    report = summarize_group([row], 1)
    assert report["final_equivalent_queries"] == 0
    assert report["paired_complete_queries"] == 0
    assert not verified(proof(False))
    assert verified(proof(False), "exploratory")


def test_p90_target_detects_a_tail_regression_despite_large_mean_gain():
    rows = [record(str(i), rbot=100, quite=100, mcts=1 if i < 8 else 95) for i in range(10)]
    report = summarize_group(rows, 10)
    assert report["targets"]["mean_latency_reduction_at_least_15_percent_vs_both"]["status"] == "pass"
    assert report["targets"]["p90_latency_reduction_at_least_15_percent_vs_both"]["status"] == "fail"
    assert report["acceptance_status"] == "fail"


def test_required_regression_compares_to_original_even_when_baselines_are_slower():
    report = summarize_group([record(original=50, rbot=100, quite=100, mcts=70)], 1)
    assert report["comparisons"]["rbot"]["regression_rate"] == 0
    assert report["comparisons"]["quite"]["regression_rate"] == 0
    assert report["comparison_vs_original"]["regression_rate"] == 1
    assert report["targets"]["regression_below_1_percent_vs_original"]["status"] == "fail"
    assert report["acceptance_status"] == "fail"


def test_missing_baseline_does_not_hide_original_mcts_regression_evidence():
    row = record(original=50, mcts=70)
    row["status"] = "incomplete"
    row["evaluation"]["quite"] = {"status": "unavailable", "median_ms": None}
    result = summarize_group([row], 1)
    assert result["original_mcts_paired_queries"] == 1
    assert result["comparison_vs_original"]["regression_rate"] == 1
    assert result["targets"]["regression_below_1_percent_vs_original"]["status"] == "fail"
    assert result["acceptance_status"] == "incomplete"


def test_only_full_m16_is_required_for_overall_acceptance():
    full = record()
    ablation = record(mcts=120)
    ablation["ablation"] = "no_safety"
    groups = [{"source": "rbot", "benchmark": "tpch", "dataset": "tpch_sf1", "ablation": variant,
               "budget": 16, "expected_queries": 1} for variant in ("full", "no_safety")]
    report = build_report([full, ablation], groups)
    assert report["groups"][1]["acceptance_status"] == "fail"
    assert report["acceptance_status"] == "pass"
    full["budget"] = 1
    groups[0]["budget"] = 1
    assert build_report([full, ablation], groups)["acceptance_status"] == "incomplete"


def test_hybrid_reports_finite_evidence_separately_and_rejects_missing_hashes():
    row = record()
    row["verification"]["mcts"] = proof(False)
    report = summarize_group([row], 1, "hybrid")
    assert report["verified_equivalence_rate_all_queries"] == 1
    assert report["formal_equivalence_rate_all_queries"] == 0
    assert report["final_finite_only_queries"] == 1
    evidence = proof(False)
    evidence["miniatures"] = evidence["miniatures"][:1]
    assert not verified(evidence, "hybrid")
    evidence = proof()
    evidence["execution"].pop("original")
    assert not verified(evidence, "strict")


class MeasuringDB:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def measure(self, sql, runs, warmups):
        self.calls.append(sql)
        assert runs == 1 and warmups == 0
        if len(self.calls) == self.fail_on:
            return {"status": "timeout", "median_ms": None, "samples_ms": [], "error": "timeout"}
        return {"status": "ok", "median_ms": float(len(self.calls)), "samples_ms": [float(len(self.calls))], "plan": {"Plan": {}}}


def test_holdout_is_interleaved_three_repetitions_and_identity_sharing_is_explicit():
    db = MeasuringDB()
    arms, schedule = evaluate_holdout(db, {"original": "select 1", "mcts": "select 1", "rbot": "select 2"}, seed=42)
    assert len(db.calls) == 8  # Two distinct SQL statements, one warmup + 3 samples.
    assert arms["original"]["samples_ms"] == arms["mcts"]["samples_ms"]
    assert arms["original"]["common_execution_identity_pairing"] is True
    assert arms["rbot"]["common_execution_identity_pairing"] is False
    assert all(len(arms[arm]["samples_ms"]) == 3 for arm in arms)
    for repetition in range(3):
        events = [event for event in schedule if event["phase"] == "holdout" and event["repetition"] == repetition]
        assert len(events) == 2
    assert all(arm["independent_of_search_measurements"] for arm in arms.values())


def test_censored_timeout_preserves_partial_samples_without_median():
    db = MeasuringDB(fail_on=3)
    arms, schedule = evaluate_holdout(db, {"mcts": "select 1"}, seed=0)
    assert arms["mcts"]["status"] == "timeout"
    assert arms["mcts"]["samples_ms"] == [2.0]
    assert arms["mcts"]["median_ms"] is None
    assert arms["mcts"]["censored"] is True
    assert len(schedule) == 3


def test_dataset_variants_and_missing_benchmarks_preserve_tasks():
    queries = [{"id": "q1", "source": "rbot", "benchmark": "tpch"}, {"id": "q2", "source": "quite", "benchmark": "dsb"}]
    databases = {"tpch1": {"benchmark": "tpch", "scale": 1}, "tpch10": {"benchmark": "tpch", "scale": 10}}
    tasks = make_tasks(queries, databases, [1, 4], ["full", "no_safety"])
    assert len(tasks) == 12
    assert sum(task["database"] is None for task in tasks) == 4
    assert {group["scale"] for group in planned_groups(tasks)} == {None, 1, 10}


def test_raw_resume_rejects_duplicate_changed_hash_and_truncated_records(tmp_path):
    path = tmp_path / "raw.jsonl"
    row = {"task_id": "t1", "run_hash": "abc"}
    path.write_text(json.dumps(row) + "\n")
    assert read_raw(path, "abc") == [row]
    with pytest.raises(ValueError, match="different run hash"):
        read_raw(path, "def")
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_raw(path, "abc")
    path.write_text('{"task_id":')
    with pytest.raises(ValueError, match="truncated/corrupt"):
        read_raw(path)


def test_reports_emit_machine_targets_nulls_and_partial_run_caveat(tmp_path):
    groups = [{"source": "rbot", "benchmark": "tpch", "dataset": "tpch_sf1", "ablation": "full", "budget": 16, "expected_queries": 1}]
    report = build_report([record()], groups, metadata={"partial_run": True})
    assert report["acceptance_status"] == "pass"
    write_reports(tmp_path, report)
    acceptance = json.loads((tmp_path / "acceptance.json").read_text())
    assert acceptance["full_benchmark_status"] == "incomplete"
    assert (tmp_path / "scaling.csv").exists()
    assert (tmp_path / "ablation.csv").exists()
    empty = build_report([], groups)
    write_reports(tmp_path, empty)
    assert "NaN" not in (tmp_path / "summary.json").read_text()
    assert empty["groups"][0]["paired_latency"]["mcts"]["average_ms"] is None


def test_runner_keeps_budget_selection_and_holdout_separate_and_verifies_no_safety(tmp_path):
    from mcts_rewrite.search import SearchConfig

    events = []

    class DB:
        def __init__(self, *_args, **_kwargs):
            self.count = 0

        def measure(self, sql, runs=3, warmups=1):
            events.append(("measure", sql, runs, warmups))
            self.count += 1
            value = 9999.0 if runs == 3 else 10.0 + self.count
            return {"status": "ok", "median_ms": value, "samples_ms": [value], "plan": {}}

        def close(self):
            events.append(("close",))

    class Gate:
        def __init__(self, *_args, **_kwargs):
            pass

        def verify(self, original, candidate):
            events.append(("verify", original, candidate))
            return proof() if candidate == original else {"accepted": False, "proof": "none", "formal": False,
                                                          "execution": {"equivalent": False}, "reason": "wrong result"}

    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

    class Controller:
        def __init__(self, db, gate, generator, config):
            self.db, self.config = db, config

        def run(self, original, schema):
            assert self.config.budget == 8
            assert not self.config.use_safety
            result = self.db.measure(original)
            events.append(("selection_frozen",))
            return {"sql": "select 999", "measurement": result, "expanded": 8, "trace": []}

    class Registry:
        def lookup(self, system, original, benchmark):
            return {"sql": original if system == "rbot" else "select 999", "provenance": {"mode": "live"}}

    schema = tmp_path / "schema.sql"
    schema.write_text("create table t (id int);")
    query = {"id": "q", "source": "rbot", "benchmark": "tpch", "sql": "select 1", "schema_path": str(schema),
             "source_path": "q.sql", "sha256": "source", "source_commit": "commit"}
    task = {"query": query, "dataset": "tpch", "database": {"dsn": "ignored"}, "budget": 8, "ablation": "no_safety"}
    result = run_task(task, {}, "run", Registry(), (DB, Gate, Generator, SearchConfig, Controller))
    assert result["search"]["measurement"]["median_ms"] == 9999
    assert result["evaluation"]["mcts"]["median_ms"] != 9999
    assert result["final_sql"] == "select 1"
    assert result["rejected_search_selection"]["sql"] == "select 999"
    assert result["evaluation"]["quite"]["status"] == "rejected_verification"
    assert "sql" not in result["evaluation"]["quite"]
    assert result["baselines"]["rbot"]["mode"] == "replay"
    frozen = events.index(("selection_frozen",))
    assert all(event[2:] == (1, 0) for event in events[frozen + 1:] if event[0] == "measure")
    assert ("verify", "select 1", "select 999") in events[frozen + 1:]


def model_context(roles=None):
    roles = roles or {"macro": "gpt-6.1-sol"}
    return {"transport": "configured_upstream_api", "requested_role_models": roles,
            "actual_role_models": dict(roles), "actual_serving_model": next(iter(roles.values())) if len(set(roles.values())) == 1 else None,
            "reasoning_effort": None, "model_substitution": False,
            "published_model_configuration_reproduced": None, "sampling_controls_reproduced": None,
            "requested_sampling_options_not_enforced": [], "usage_known": None,
            "embedding_source": {"provider": "local_bm25", "substituted": False}}


def publication_report(row, context=None):
    groups = [{"source": "rbot", "benchmark": "tpch", "dataset": "tpch_sf1", "ablation": "full",
               "budget": 16, "expected_queries": 1}]
    return build_report([row], groups, metadata={"publication_scale_run": True, "partial_run": False,
                        "verification_mode": "strict", "model_context": context or model_context()})


def fresh_record():
    row = record()
    row["baselines"] = {
        "rbot": {"mode": "fresh_standalone", "provenance": {"model_context": model_context({"rewrite": "gpt-4o"})}},
        "quite": {"mode": "fresh_standalone", "provenance": {"model_context": model_context({
            "reasoning": "model-a", "decision": "model-b", "assistant": "model-c"})}},
    }
    return row


def test_replayed_baseline_targets_remain_visible_but_publication_is_incomplete(tmp_path):
    row = fresh_record()
    row["baselines"]["rbot"]["mode"] = "replay"
    report = publication_report(row)
    assert report["acceptance_status"] == "pass"
    assert report["groups"][0]["comparisons"]["rbot"]["mean_latency_reduction"] > .15
    assert not report["publication_eligibility"]["eligible"]
    assert "replay" in report["definitions"]["baseline_mode"]
    write_reports(tmp_path, report)
    acceptance = json.loads((tmp_path / "acceptance.json").read_text())
    assert acceptance["status"] == "pass" and acceptance["full_benchmark_status"] == "incomplete"


def test_fresh_comparison_preserves_distinct_algorithm_model_roles(tmp_path):
    report = publication_report(fresh_record())
    assert report["publication_eligibility"]["eligible"]
    assert len(report["publication_eligibility"]["baseline_model_contexts"]) == 2
    write_reports(tmp_path, report)
    assert json.loads((tmp_path / "acceptance.json").read_text())["full_benchmark_status"] == "pass"


def test_missing_model_context_does_not_erase_metrics_but_blocks_publication():
    row = fresh_record()
    row["baselines"]["rbot"]["provenance"].pop("model_context")
    report = publication_report(row)
    assert report["acceptance_status"] == "pass"
    assert not report["publication_eligibility"]["eligible"]
    assert report["publication_eligibility"]["baseline_model_context_failures"]


def test_session_model_substitution_must_be_disclosed_consistently():
    context = model_context()
    context.update(transport="codex_session_file_spool", reasoning_effort="xhigh",
                   requested_role_models={"macro": "gpt-4o"}, model_substitution=False,
                   sampling_controls_reproduced=False, published_model_configuration_reproduced=False, usage_known=False)
    invalid = publication_report(fresh_record(), context)
    assert not invalid["publication_eligibility"]["eligible"]
    context["model_substitution"] = True
    valid = publication_report(fresh_record(), context)
    assert valid["publication_eligibility"]["eligible"]


def test_midtask_host_pressure_is_reported_without_changing_semantic_or_numerical_results(tmp_path):
    row = fresh_record()
    row["host_conditions"] = {"status": "pressure_observed", "quiet_at_observed_samples": False}
    report = publication_report(row)
    assert report["acceptance_status"] == "pass"
    assert report["groups"][0]["final_equivalent_queries"] == 1
    assert report["publication_eligibility"]["host_condition_status_counts"] == {"pressure_observed": 1}
    assert report["publication_eligibility"]["quiet_at_all_observed_task_samples"] is False
    write_reports(tmp_path, report)
    assert json.loads((tmp_path / "acceptance.json").read_text())["publication_eligibility"]["quiet_at_all_observed_task_samples"] is False
