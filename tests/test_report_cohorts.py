"""Supplemental cohort projections use synthetic immutable files only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.report_cohorts import digest, main, report_cohorts


def _proof(formal=True):
    comparison = {"status": "ok", "equivalent": True, "same_snapshot": True,
                  "original": {"sha256": "same"}, "candidate": {"sha256": "same"}}
    result = {"accepted": True, "formal": formal, "proof": "identity" if formal else "differential",
              "execution": comparison, "audit_marker": "preserve proof provenance"}
    if not formal:
        result["miniatures"] = [{"name": n, "comparison": comparison} for n in
                                ("empty", "null", "duplicate", "skew", "unmatched")]
    return result


def _context():
    return {"transport": "synthetic", "requested_role_models": {"assistant": "fixture"},
            "actual_role_models": {"assistant": "fixture"}, "model_substitution": False,
            "actual_serving_model": "fixture", "reasoning_effort": "fixture",
            "published_model_configuration_reproduced": False, "sampling_controls_reproduced": False,
            "requested_sampling_options_not_enforced": [], "usage_known": False,
            "embedding_source": {"provider": "fixture", "substituted": False}}


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _save_lines(path, rows):
    # Deliberate whitespace/order makes byte-for-byte provenance checks useful.
    path.write_text("".join(json.dumps(r, separators=(", ", ": ")) + "\n" for r in rows))


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    queries = {"rbot:tpch:query1_0": "SELECT 1 AS a", "rbot:tpch:query1_1": "SELECT 2 AS a",
               "quite:tpch:1:entry001": "SELECT 3 AS a"}
    classification, groups, records = [], [], []
    for query_id, original in queries.items():
        source = query_id.split(":")[0]
        for dataset, scale in (("tpch10", 10), ("tpch50", 50)):
            kind = ("requested_quite_cross_scale_native_binding_unestablished" if source == "quite"
                    else "expanded_rbot_sf50_seed1" if dataset == "tpch50" and query_id.endswith("_1")
                    else "native_released_evaluation_cohort")
            classification.append({"query_id": query_id, "source": source, "benchmark": "tpch",
                                   "dataset": dataset, "scale": scale, "query_sha256": digest(original.encode()),
                                   "classification": kind, "native_released_dataset_membership":
                                   None if source == "quite" else kind == "native_released_evaluation_cohort"})
            for budget in (1, 16):
                for ablation in ("full", "class_a_only"):
                    records.append({"record_type": "query_result", "task_id": digest([query_id, dataset, budget, ablation]),
                                    "query_id": query_id, "source": source, "benchmark": "tpch", "dataset": dataset,
                                    "budget": budget, "ablation": ablation, "status": "ok", "original_sql": original,
                                    "query_sha256": digest(original.encode()), "verification_mode": "hybrid",
                                    "verification": {arm: _proof(not query_id.endswith("_1")) for arm in
                                                     ("original", "rbot", "quite", "mcts")},
                                    "evaluation": {arm: {"status": "ok", "median_ms": ms} for arm, ms in
                                                   (("original", 100), ("rbot", 90), ("quite", 80), ("mcts", 60))},
                                    "baselines": {arm: {"mode": "fresh_standalone", "status": "found", "provenance":
                                                        {"model_context": _context(), "native_marker": arm}}
                                                  for arm in ("rbot", "quite")}})
    for source, n in (("rbot", 2), ("quite", 1)):
        for dataset, scale in (("tpch10", 10), ("tpch50", 50)):
            for budget in (1, 16):
                for ablation in ("full", "class_a_only"):
                    groups.append({"source": source, "benchmark": "tpch", "dataset": dataset,
                                   "scale": scale, "budget": budget, "ablation": ablation, "expected_queries": n})
    fingerprint = {"source_fingerprints": {q: {"query": digest(s.encode())} for q, s in queries.items()},
                   "selected_query_ids": list(queries), "budgets": [1, 16], "ablations": ["full", "class_a_only"]}
    metadata = {"fingerprint": fingerprint, "run_hash": digest(fingerprint), "selected_queries": 3,
                "planned_tasks": len(records), "planned_groups": groups, "verification_mode": "hybrid",
                "partial_run": False, "publication_scale_run": True, "model_context": _context()}
    for record in records:
        record["run_hash"] = metadata["run_hash"]
    _save(run / "run.json", metadata)
    _save_lines(run / "raw.jsonl", records)
    classes = tmp_path / "classifications.jsonl"
    _save_lines(classes, classification)
    return {"run": run, "classes": classes, "metadata": metadata, "records": records,
            "classification": classification, "output": tmp_path / "supplemental"}


def _report(source):
    return report_cohorts(source["run"], source["output"], source["classes"])


def test_complete_views_keep_groups_counts_provenance_and_source_bytes(source):
    inputs = [source["run"] / "run.json", source["run"] / "raw.jsonl", source["classes"]]
    before = {p: p.read_bytes() for p in inputs}
    result = _report(source)
    expected = {"native_rbot": (3, 12, 8), "expanded_rbot_sf50": (1, 4, 4), "quite_requested_grid": (2, 8, 8)}
    for name, (memberships, tasks, groups) in expected.items():
        receipt = result["views"][name]
        assert receipt["expected_query_dataset_memberships"] == memberships
        assert receipt["expected_tasks"] == receipt["recorded_tasks"] == tasks
        assert receipt["planned_groups"] == groups and receipt["missing_tasks"] == 0
        destination = source["output"] / name
        acceptance = json.loads((destination / "acceptance.json").read_text())
        assert acceptance["status"] == "pass"
        assert acceptance["partial_run"] is True and acceptance["full_benchmark_status"] == "incomplete"
        assert acceptance["scope"]["can_establish_original_full_task_acceptance"] is False
        assert (destination / "source_run.json").read_bytes() == before[inputs[0]]
        chosen = {(r["query_id"], r["dataset"]) for r in receipt["expected_memberships"]}
        original_lines = before[inputs[1]].splitlines(keepends=True)
        expected_raw = b"".join(line for line, r in zip(original_lines, source["records"])
                                if (r["query_id"], r["dataset"]) in chosen)
        assert (destination / "raw.jsonl").read_bytes() == expected_raw
        summary = json.loads((destination / "summary.json").read_text())
        assert {g["budget"] for g in summary["groups"]} == {1, 16}
        assert {g["ablation"] for g in summary["groups"]} == {"full", "class_a_only"}
        if name == "expanded_rbot_sf50":
            assert all(g["final_finite_only_queries"] == 1 for g in summary["groups"])
    assert result["views"]["quite_requested_grid"]["native_dataset_binding_known"] is False
    assert {p: p.read_bytes() for p in inputs} == before
    assert result["input_files"]["raw"]["sha256"] == digest(before[inputs[1]])


def test_missing_records_and_rejected_arms_remain_missing(source):
    records = deepcopy(source["records"])
    records = [r for r in records if not (r["query_id"].endswith("_0") and r["dataset"] == "tpch50"
                                        and r["budget"] == 16 and r["ablation"] == "full")]
    rejected = next(r for r in records if r["query_id"].endswith("_1") and r["dataset"] == "tpch50"
                    and r["budget"] == 16 and r["ablation"] == "full")
    rejected["status"] = "incomplete"
    rejected["baselines"]["rbot"]["status"] = "rejected_verification"
    rejected["verification"]["rbot"]["accepted"] = False
    rejected["evaluation"]["rbot"] = {"status": "rejected_verification", "median_ms": None}
    _save_lines(source["run"] / "raw.jsonl", records)
    result = _report(source)
    assert result["views"]["native_rbot"]["missing_tasks"] == 1
    native = json.loads((source["output"] / "native_rbot/summary.json").read_text())
    missing = next(g for g in native["groups"] if g["dataset"] == "tpch50" and g["budget"] == 16 and g["ablation"] == "full")
    assert missing["expected_queries"] == 1 and missing["recorded_queries"] == 0
    assert missing["acceptance_status"] == "incomplete"
    expanded = json.loads((source["output"] / "expanded_rbot_sf50/summary.json").read_text())
    group = next(g for g in expanded["groups"] if g["budget"] == 16 and g["ablation"] == "full")
    assert group["expected_queries"] == group["recorded_queries"] == 1
    assert group["rejection_counts"]["baseline_rejected"] == 1
    assert group["paired_complete_queries"] == 0 and group["acceptance_status"] == "incomplete"


def test_selected_ids_define_denominators_and_empty_groups_remain(source):
    metadata = deepcopy(source["metadata"])
    metadata["fingerprint"]["selected_query_ids"].remove("rbot:tpch:query1_1")
    metadata["selected_queries"] = 2
    for group in metadata["planned_groups"]:
        if group["source"] == "rbot":
            group["expected_queries"] = 1
    metadata["planned_tasks"] = sum(g["expected_queries"] for g in metadata["planned_groups"])
    metadata["run_hash"] = digest(metadata["fingerprint"])
    records = [deepcopy(r) for r in source["records"] if r["query_id"] != "rbot:tpch:query1_1"]
    for record in records:
        record["run_hash"] = metadata["run_hash"]
    _save(source["run"] / "run.json", metadata)
    _save_lines(source["run"] / "raw.jsonl", records)
    result = _report(source)
    assert result["views"]["native_rbot"]["expected_query_dataset_memberships"] == 2
    assert result["views"]["expanded_rbot_sf50"]["expected_tasks"] == 0
    assert result["views"]["expanded_rbot_sf50"]["planned_groups"] == 4
    summary = json.loads((source["output"] / "expanded_rbot_sf50/summary.json").read_text())
    assert all(g["expected_queries"] == 0 for g in summary["groups"])
    assert summary["acceptance_status"] == "incomplete"


@pytest.mark.parametrize("mutation,error", [
    (lambda rows: rows[0].update(unrecognized_field=True), "unknown keys"),
    (lambda rows: rows[0].update(query_id="rbot:tpch:unknown"), "unknown query key"),
    (lambda rows: rows[0].update(dataset="tpch_typo"), "unknown dataset key"),
    (lambda rows: rows[0].update(query_sha256="0" * 64), "SQL hash mismatch"),
    (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate classification"),
    (lambda rows: rows[0].update(classification="make_it_pass"), "Unknown classification"),
    (lambda rows: rows[0].update(native_released_dataset_membership=False), "Contradictory"),
    (lambda rows: rows[0].update(scale=50), "scale disagrees"),
    (lambda rows: rows[0].update(distribution="zipf"), "distribution disagrees"),
    (lambda rows: rows.pop(1), "Missing classification"),
])
def test_bad_classifications_fail_before_output(source, mutation, error):
    rows = deepcopy(source["classification"])
    mutation(rows)
    _save_lines(source["classes"], rows)
    with pytest.raises(ValueError, match=error):
        _report(source)
    assert not source["output"].exists()


@pytest.mark.parametrize("mutation,error", [
    (lambda rows: rows[0].update(original_sql="SELECT 999"), "Raw SQL hash mismatch"),
    (lambda rows: rows[0].update(run_hash="0" * 64), "run_hash"),
    (lambda rows: rows[0].update(budget=8), "Unknown/unplanned"),
    (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate/invalid raw task"),
])
def test_bad_raw_evidence_fails_before_output(source, mutation, error):
    records = deepcopy(source["records"])
    mutation(records)
    _save_lines(source["run"] / "raw.jsonl", records)
    with pytest.raises(ValueError, match=error):
        _report(source)
    assert not source["output"].exists()


def test_unregistered_source_dataset_is_not_inferred(source):
    metadata = deepcopy(source["metadata"])
    metadata["planned_groups"][0]["dataset"] = "devcalcite"
    _save(source["run"] / "run.json", metadata)
    with pytest.raises(ValueError, match="devcalcite.*not registered"):
        _report(source)
    assert not source["output"].exists()


def test_source_denominator_and_fingerprint_are_checked(source):
    metadata = deepcopy(source["metadata"])
    metadata["planned_groups"][0]["expected_queries"] += 1
    _save(source["run"] / "run.json", metadata)
    with pytest.raises(ValueError, match="denominator"):
        _report(source)
    metadata = deepcopy(source["metadata"])
    metadata["fingerprint"]["selected_query_ids"].pop()
    _save(source["run"] / "run.json", metadata)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        _report(source)


def test_output_cannot_change_source_or_overwrite_existing_results(source):
    with pytest.raises(ValueError, match="outside the source"):
        report_cohorts(source["run"], source["run"] / "cohorts", source["classes"])
    source["output"].mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _report(source)


def test_cli_uses_supplied_classification_and_reports_counts(source, capsys):
    assert main(["--run", str(source["run"]), "--classification", str(source["classes"]),
                 "--output", str(source["output"])]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["views"]["native_rbot"]["expected_query_dataset_memberships"] == 3


def test_empty_raw_file_retains_every_planned_denominator(source):
    (source["run"] / "raw.jsonl").write_bytes(b"")
    result = _report(source)
    assert result["source_expected_query_dataset_memberships"] == 6
    for name, receipt in result["views"].items():
        assert receipt["recorded_tasks"] == 0
        assert receipt["missing_tasks"] == receipt["expected_tasks"]
        acceptance = json.loads((source["output"] / name / "acceptance.json").read_text())
        assert acceptance["status"] == acceptance["full_benchmark_status"] == "incomplete"
        assert sum(g["expected_queries"] for g in acceptance["groups"]) == receipt["expected_tasks"]


def test_changing_input_snapshot_is_rejected_before_output(source, monkeypatch):
    read_bytes = Path.read_bytes
    raw_path = source["run"] / "raw.jsonl"
    reads = 0

    def changing_read(path):
        nonlocal reads
        value = read_bytes(path)
        if path == raw_path:
            reads += 1
            if reads > 1:
                return value + b"\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="Input evidence changed"):
        _report(source)
    assert not source["output"].exists()
