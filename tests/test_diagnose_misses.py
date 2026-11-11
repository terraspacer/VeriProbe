"""Synthetic immutable runs exercise feedback boundaries without a database."""
from copy import deepcopy
import json

import pytest

from scripts.diagnose_misses import (
    classify, diagnose, digest, inspect_plan, main, merge_patterns, pattern_kinds,
    promote, read_jsonl, sql_sha256, template_sha256, valid_search_measurement,
    write_jsonl,
)


def proof(formal=True):
    comparison = {"equivalent": True, "status": "ok", "same_snapshot": True,
                  "original": {"sha256": "same-results"}, "candidate": {"sha256": "same-results"}}
    evidence = {"accepted": True, "formal": formal, "proof": "ast" if formal else "differential",
                "execution": comparison}
    if not formal:
        evidence["miniatures"] = [{"name": name, "comparison": comparison}
                                   for name in ("empty", "null", "duplicate", "skew", "unmatched")]
    return evidence


def measured_plan(timing=False):
    scan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t", "Actual Loops": 10,
            "Actual Rows": 20, "Rows Removed by Filter": 4, "Filter": "(a > 1)",
            "Shared Read Blocks": 3, "Plan Rows": 999, "Total Cost": 123456}
    inner = {"Node Type": "Aggregate", "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1",
             "Actual Loops": 10, "Actual Rows": 1, "HashAgg Batches": 2, "Plans": [scan]}
    root = {"Node Type": "Result", "Actual Loops": 1, "Actual Rows": 1, "Total Cost": 987654,
            "Plans": [inner]}
    if timing:
        scan["Actual Total Time"] = 0.3
        inner["Actual Total Time"] = 0.4
        root["Actual Total Time"] = 5
    return {"Plan": root, "Execution Time": 5.5}


def measurement(ms, plan=None):
    return {"status": "ok", "median_ms": ms, "samples_ms": [ms - 1, ms, ms + 1],
            "censored": False, "plan": deepcopy(plan)}


def row(task="task", mcts=12, rbot=10, quite=11):
    original = "SELECT a FROM t WHERE a > 1;"
    candidate = "SELECT a FROM t WHERE a > 1 AND TRUE;"
    return {"record_type": "query_result", "task_id": task, "query_id": task,
            "benchmark": "synthetic", "dataset": "tiny", "budget": 1, "ablation": "full",
            "original_sql": original, "query_sha256": sql_sha256(original),
            "schema_sha256": sql_sha256("CREATE TABLE t (a INTEGER);"),
            "manifest_provenance": {"source_commit": "f" * 40},
            "finished_at": "2026-10-26T00:00:00Z", "verification_mode": "strict",
            "baselines": {arm: {"status": "found"} for arm in ("rbot", "quite")},
            "verification": {arm: proof() for arm in ("mcts", "rbot", "quite")},
            "evaluation": {arm: measurement(ms, measured_plan()) for arm, ms in
                           (("mcts", mcts), ("rbot", rbot), ("quite", quite))},
            "search": {"measurement": measurement(8, measured_plan()), "trace": [
                {"event": "root", "sql": original, "verification": proof(),
                 "measurement": measurement(20, measured_plan())},
                {"event": "candidate", "sql": candidate, "status": "accepted", "action_class": "B",
                 "action": "reviewed_decorrelation", "verification_against": "original_root",
                 "verification": proof(), "measurement": measurement(8, measured_plan())}]}}


def write_run(tmp_path, records=None, profile="development"):
    records = deepcopy(records if records is not None else [row()])
    fingerprint = {"code_sha256": sql_sha256("code-snapshot"), "upstreams": {"optimizer": sql_sha256("upstream-snapshot")},
                   "source_fingerprints": {item["query_id"]: {"query": item["query_sha256"],
                                                              "schema": item["schema_sha256"]} for item in records}}
    metadata = {"profile": profile, "run_hash": digest(fingerprint), "fingerprint": fingerprint,
                "planned_tasks": len(records), "publication_scale_run": profile == "published"}
    for item in records:
        item["run_hash"] = metadata["run_hash"]
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, records)
    path.with_name("run.json").write_text(json.dumps(metadata))
    return path, metadata, records


def drafts_and_receipt(tmp_path, records=None, scope="development", profile="development"):
    path, metadata, records = write_run(tmp_path, records, profile)
    _, _, drafts = diagnose(read_jsonl(path), path, metadata, scope)
    pattern = next(item for item in drafts if item["trigger"]["kind"] == "repeated_subplan")
    item, event = records[0], records[0]["search"]["trace"][1]
    receipt = {"pattern_id": pattern["id"], "status": "validated", "phase": "search",
               "reviewer": "test reviewer", "conditions_reviewed": True, "safety_checks_reviewed": True,
               "matched_pattern_kind": "repeated_subplan", "raw_path": str(path),
               "raw_sha256": digest(path.read_bytes()), "run_hash": metadata["run_hash"],
               "task_id": item["task_id"], "root_query_sha256": item["query_sha256"],
               "schema_sha256": item["schema_sha256"], "source_commit": item["manifest_provenance"]["source_commit"],
               "candidate_trace_index": 1, "candidate_sql_sha256": sql_sha256(event["sql"]),
               "applied_action": event["action"], "feedback_scope": scope}
    return pattern, receipt, path


def test_misses_include_ties_and_only_successful_verified_baselines():
    item = row(mcts=10)
    assert classify(item)["comparison_status"] == "latency_miss"
    item["evaluation"]["rbot"] = {"status": "timeout", "median_ms": 0.01}
    result = classify(item)
    assert result["comparison_status"] == "beat_available_baselines"
    assert result["available_valid_baselines"] == ["quite"]
    assert result["best_available_valid_baseline"] == "quite"
    item["verification"]["quite"]["accepted"] = False
    result = classify(item)
    assert result["comparison_status"] == "no_valid_baseline"
    assert not result["miss"]
    assert result["best_available_valid_baseline"] is None


@pytest.mark.parametrize("change", ["censored", "failed_registry", "nonfinite", "invalid_proof"])
def test_failed_baseline_never_substitutes_a_latency(change):
    item = row(mcts=12, rbot=1, quite=20)
    if change == "censored":
        item["evaluation"]["rbot"]["censored"] = True
    elif change == "failed_registry":
        item["baselines"]["rbot"]["status"] = "unavailable"
    elif change == "nonfinite":
        item["evaluation"]["rbot"]["median_ms"] = float("nan")
    else:
        item["verification"]["rbot"]["formal"] = False
    result = classify(item)
    assert result["available_valid_baselines"] == ["quite"]
    assert result["comparison_status"] == "beat_available_baselines"


def test_invalid_mcts_is_a_miss_when_a_baseline_is_valid():
    item = row()
    item["evaluation"]["mcts"]["status"] = "timeout"
    assert classify(item)["comparison_status"] == "mcts_invalid_or_incomplete"


def test_timing_off_keeps_measured_subplans_spills_reads_and_scans():
    plan = inspect_plan(measured_plan())
    assert plan["analyze_available"] and not plan["node_timing_available"]
    assert plan["server_execution_time_ms"] == 5.5
    assert plan["planner_total_cost_estimate"] == 987654
    assert plan["hotspot_ranking"] == "actual_row_visits"
    assert all(item["actual_inclusive_work_ms"] is None for item in plan["hotspots"])
    assert plan["repeated_work"][0]["kind"] == "repeated_subplan"
    assert plan["spills"][0]["recorded_spill_fields"]["HashAgg Batches"] == 2
    assert plan["scans"][0]["observed_rows_examined_heuristic"] == 240
    assert {"repeated_subplan", "repeated_scan", "scan_input", "spill", "io_reads"} <= set(pattern_kinds(plan))


def test_estimated_plan_produces_no_measured_diagnosis():
    plan = inspect_plan({"Plan": {"Node Type": "Seq Scan", "Relation Name": "t", "Plan Rows": 500,
                                 "Total Cost": 1000000, "Hash Batches": 10}})
    assert not plan["analyze_available"]
    assert plan["planner_total_cost_estimate"] == 1000000
    assert plan["server_execution_time_ms"] is None
    assert not plan["spills"] and not plan["hotspots"] and not pattern_kinds(plan)


def test_timed_hotspot_subtraction_is_explicitly_a_heuristic():
    plan = inspect_plan(measured_plan(timing=True))
    assert plan["node_timing_available"]
    assert plan["hotspot_ranking"] == "exclusive_work_heuristic_ms"
    assert plan["hotspots"][0]["actual_inclusive_work_ms"] == 3
    assert "Planner cost is an estimate" in plan["interpretation"]


def test_diagnosis_is_deterministic_deduplicated_and_plan_supported(tmp_path):
    path, metadata, _ = write_run(tmp_path, [row("one"), row("two")])
    result = diagnose(read_jsonl(path), path, metadata)
    assert result == diagnose(read_jsonl(path), path, metadata)
    misses, audit, drafts = result
    assert len(misses) == len(audit) == 2
    assert len({item["id"] for item in drafts}) == len(drafts) > 0
    assert all(item["status"] == "draft" and item["validation"]["status"] == "unvalidated" for item in drafts)
    for pattern in drafts:
        assert len(pattern["provenance"]["source_records"]) == 2
        assert pattern["conditions"] and pattern["transformations"] and pattern["safety_checks"]
        assert pattern["leakage"]["allow_schema_sha256"]
        assert all(source["plan_evidence"]["json_pointer"] == "/evaluation/mcts/plan"
                   for source in pattern["provenance"]["source_records"])
    assert not any(item["unbiased_final_evaluation"] for item in audit)


def test_no_measured_plan_does_not_invent_a_pattern(tmp_path):
    item = row()
    item["evaluation"]["mcts"]["plan"] = {"Plan": {"Node Type": "Result", "Total Cost": 9999}}
    path, metadata, _ = write_run(tmp_path, [item])
    misses, _, drafts = diagnose(read_jsonl(path), path, metadata)
    assert not drafts
    assert misses[0]["diagnosis_status"] == "needs_review_no_supported_rewrite_pattern"


def test_published_feedback_requires_explicit_transductive_reclassification(tmp_path):
    path, metadata, _ = write_run(tmp_path, profile="published")
    with pytest.raises(ValueError, match="declared development/tuning"):
        diagnose(read_jsonl(path), path, metadata)
    misses, _, drafts = diagnose(read_jsonl(path), path, metadata, "transductive")
    assert misses[0]["evaluation_role"] == "transductive_tuning_feedback"
    assert all(item["leakage"]["scope"] == "transductive_query_only" and item["leakage"]["allow_query_sha256"]
               for item in drafts)


def test_completed_and_immutable_run_required(tmp_path):
    path, metadata, _ = write_run(tmp_path)
    metadata["planned_tasks"] = 2
    with pytest.raises(ValueError, match="completed raw run"):
        diagnose(read_jsonl(path), path, metadata)
    metadata["planned_tasks"] = 1
    metadata["fingerprint"]["code_sha256"] = "changed"
    with pytest.raises(ValueError, match="fingerprint hash mismatch"):
        diagnose(read_jsonl(path), path, metadata)


def test_promotion_requires_search_evidence_and_deduplicates_receipts(tmp_path):
    pattern, receipt, _ = drafts_and_receipt(tmp_path)
    approved, rejected = promote([pattern], [receipt, deepcopy(receipt)])
    assert not rejected and len(approved) == 1
    entry = approved[0]
    assert entry["status"] == "validated"
    assert entry["validation"]["semantics_scope"] == "formal"
    assert entry["validation"]["timing_scope"] == "search_training_only"
    assert entry["validation"]["requires_root_equivalence_gate"]
    assert len(entry["validation"]["receipts"]) == 1
    evidence = entry["validation"]["receipts"][0]
    assert evidence["phase"] == "search" and evidence["root_median_ms"] == 20
    assert evidence["candidate_median_ms"] == 8 and evidence["schema_sha256"] == receipt["schema_sha256"]
    assert evidence["source_commit"] == receipt["source_commit"]
    again, rejected = promote([pattern], [receipt], approved)
    assert not rejected and again == approved


@pytest.mark.parametrize("field,value", [
    ("phase", "holdout"), ("conditions_reviewed", False), ("safety_checks_reviewed", False),
    ("matched_pattern_kind", "spill"), ("raw_sha256", "tampered"), ("run_hash", "other"),
    ("root_query_sha256", "other"), ("schema_sha256", "other"), ("source_commit", "other"),
    ("candidate_trace_index", True), ("candidate_trace_index", 0), ("candidate_sql_sha256", "other"),
    ("applied_action", "unrelated"), ("feedback_scope", "transductive"),
])
def test_forged_or_unreviewed_promotion_receipts_are_rejected(tmp_path, field, value):
    pattern, receipt, _ = drafts_and_receipt(tmp_path)
    receipt[field] = value
    approved, rejected = promote([pattern], [receipt])
    assert not approved and rejected


@pytest.mark.parametrize("change", ["one_sample", "censored", "median", "slower", "root_sql", "root_proof", "candidate_proof", "estimated_root_plan"])
def test_final_samples_cannot_replace_missing_search_validation(tmp_path, change):
    item = row()
    root, candidate = item["search"]["trace"]
    if change == "one_sample":
        candidate["measurement"]["samples_ms"] = [8]
    elif change == "censored":
        candidate["measurement"]["censored"] = True
    elif change == "median":
        candidate["measurement"]["median_ms"] = 1
    elif change == "slower":
        candidate["measurement"] = measurement(30)
    elif change == "root_sql":
        root["sql"] = "SELECT a FROM another_table;"
    elif change == "root_proof":
        root["verification"]["accepted"] = False
    elif change == "estimated_root_plan":
        root["measurement"]["plan"] = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 999}}
    else:
        candidate["verification"]["formal"] = False
    pattern, receipt, _ = drafts_and_receipt(tmp_path, [item])
    approved, rejected = promote([pattern], [receipt])
    assert not approved and rejected


def test_draft_source_or_definition_cannot_be_swapped(tmp_path):
    pattern, receipt, _ = drafts_and_receipt(tmp_path)
    pattern["transformations"] = ["An unrelated transformation"]
    approved, rejected = promote([pattern], [receipt])
    assert not approved and "definition" in rejected[0]["reason"]
    pattern, receipt, _ = drafts_and_receipt(tmp_path)
    pattern["provenance"]["source_records"][0]["schema_sha256"] = "another schema"
    approved, rejected = promote([pattern], [receipt])
    assert not approved and "provenance mismatch" in rejected[0]["reason"]


def test_finite_data_receipt_keeps_its_limited_scope(tmp_path):
    item = row()
    item["verification_mode"] = "hybrid"
    item["search"]["trace"][1]["verification"] = proof(formal=False)
    pattern, receipt, _ = drafts_and_receipt(tmp_path, [item])
    approved, rejected = promote([pattern], [receipt])
    assert not rejected
    assert approved[0]["validation"]["semantics_scope"] == "finite_data"
    assert approved[0]["validation"]["requires_root_equivalence_gate"]


def test_template_identity_removes_literals_but_retains_relations():
    assert template_sha256("SELECT a FROM t WHERE a=1") == template_sha256("select a from t where a=9")
    assert template_sha256("SELECT a FROM t WHERE a=1") != template_sha256("SELECT a FROM u WHERE a=1")


def test_diagnosis_cli_outputs_drafts_without_modifying_approved_library(tmp_path):
    path, _, _ = write_run(tmp_path)
    library = tmp_path / "library.jsonl"
    library.write_text("")
    output = tmp_path / "feedback"
    drafts = output / "drafts.jsonl"
    assert main(["--raw", str(path), "--output", str(output), "--propose-patterns", str(drafts),
                 "--library", str(library)]) == 0
    summary = json.loads((output / "feedback_summary.json").read_text())
    assert summary["misses"] == 1 and summary["draft_patterns"] > 0
    assert not summary["approved_library_modified"] and library.read_text() == ""
    assert read_jsonl(output / "missed_opportunities.jsonl") and read_jsonl(drafts)


def test_search_measurement_rejects_nonpositive_or_nonfinite_samples():
    for samples in ([0, 1, 2], [1, float("inf"), 2], [True, 1, 2]):
        assert not valid_search_measurement({"status": "ok", "median_ms": 1, "samples_ms": samples})


def test_development_and_transductive_scopes_never_merge(tmp_path):
    path, metadata, _ = write_run(tmp_path)
    development = diagnose(read_jsonl(path), path, metadata)[2]
    transductive = diagnose(read_jsonl(path), path, metadata, "transductive")[2]
    assert len(merge_patterns(development + transductive)) == len(development) + len(transductive)


def test_merge_is_order_independent_for_sources_and_receipts(tmp_path):
    pattern, receipt, _ = drafts_and_receipt(tmp_path)
    approved, _ = promote([pattern], [receipt])
    repeated = deepcopy(approved[0])
    repeated["validation"]["receipts"] *= 2
    repeated["provenance"]["source_records"] *= 2
    repeated["leakage"]["exclude_query_sha256"] *= 2
    assert merge_patterns([repeated]) == approved
    assert merge_patterns([pattern, approved[0]]) == merge_patterns([approved[0], pattern])
