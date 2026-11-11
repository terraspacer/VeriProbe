#!/usr/bin/env python3
"""Auditable development feedback, with no database calls or automatic proof.

Draft feedback from a completed development run::

    .venv/bin/python scripts/diagnose_misses.py --raw results/dev/raw.jsonl \
        --output results/dev_feedback --propose-patterns results/dev_feedback/drafts.jsonl

Promote reviewed drafts using receipts anchored to immutable search events::

    .venv/bin/python scripts/diagnose_misses.py --patterns results/dev_feedback/drafts.jsonl \
        --promote-validated receipts.jsonl --library prompts/action_patterns.jsonl

A receipt contains pattern_id, status="validated", phase="search", reviewer,
conditions_reviewed=true, safety_checks_reviewed=true, matched_pattern_kind,
raw_path, raw_sha256, run_hash, task_id, root_query_sha256, schema_sha256,
source_commit, candidate_trace_index, candidate_sql_sha256, applied_action,
and feedback_scope (development or transductive). Its referenced candidate/root events must have successful median
of >=3 search measurements and accepted root equivalence. Final evaluation
timings cannot serve as promotion evidence. Search timing is observational
training evidence, not proof of causality or an unbiased final result.

Default development mode requires a development/tuning profile in run metadata
or its configuration. --tuning-scope transductive explicitly reclassifies the
input as per-query tuning: resulting patterns are restricted to those queries
and all consumed final evaluation samples lose their unbiased evaluation role.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlglot
from sqlglot import exp

from mcts_rewrite.reporting import complete_arm, verified

BASELINES = ("rbot", "quite")
SCOPES = ("development", "transductive")


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sql_sha256(sql):
    return hashlib.sha256(sql.encode()).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def template_sha256(sql):
    """Literal-insensitive template identity, retaining tables and column names."""
    tree = sqlglot.parse_one(sql, read="postgres")
    for node in list(tree.find_all(exp.Literal)):
        node.replace(exp.Placeholder())
    for node in tree.find_all(exp.Identifier):
        if not node.args.get("quoted"):
            node.set("this", node.this.lower())
    return sql_sha256(tree.sql(dialect="postgres"))


def read_jsonl(path):
    rows = []
    for line, text in enumerate(Path(path).read_text().splitlines(), 1):
        if text.strip():
            try:
                row = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: corrupt/truncated JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line}: expected a JSON object")
            rows.append((line, row))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def run_profile(metadata):
    explicit = metadata.get("profile", metadata.get("purpose"))
    if explicit:
        return explicit
    # Prefer the run's recorded snapshot over a configuration edited afterwards.
    profiles = {v.get("provenance", {}).get("profile") for v in metadata.get("dataset_provenance", {}).values()}
    if len(profiles) == 1 and None not in profiles:
        return next(iter(profiles))
    config = metadata.get("config")
    if isinstance(config, str) and Path(config).is_file():
        configured = json.loads(Path(config).read_text())
        if digest(configured) == metadata.get("fingerprint", {}).get("config_sha256"):
            return configured.get("profile")
    return None


def validate_run(rows, metadata, scope):
    if scope not in SCOPES:
        raise ValueError(f"Unknown tuning scope: {scope}")
    if scope == "development" and (metadata.get("publication_scale_run") is True or run_profile(metadata) not in {"development", "tuning"}):
        raise ValueError("Only a declared development/tuning run is accepted by default; explicitly use --tuning-scope transductive to reclassify per-query feedback and forfeit unbiased final evaluation.")
    if not rows or type(metadata.get("planned_tasks")) is not int or len(rows) != metadata["planned_tasks"]:
        raise ValueError("Feedback requires a completed raw run matching metadata planned_tasks")
    if metadata.get("fingerprint") and digest(metadata["fingerprint"]) != metadata.get("run_hash"):
        raise ValueError("Run metadata fingerprint hash mismatch")
    tasks = set()
    for _, row in rows:
        if row.get("record_type") != "query_result" or not row.get("finished_at"):
            raise ValueError("Raw records must be completed query_result records")
        if not row.get("task_id") or row["task_id"] in tasks:
            raise ValueError("Missing or duplicate raw task_id")
        tasks.add(row["task_id"])
        if not metadata.get("run_hash") or row.get("run_hash") != metadata["run_hash"]:
            raise ValueError("Raw record/run metadata hash mismatch")
        original = row.get("original_sql")
        if not isinstance(original, str) or not original.strip():
            raise ValueError("Raw records must retain their original SQL")
        if row.get("query_sha256") and row["query_sha256"] != sql_sha256(original):
            raise ValueError("Raw query hash does not match original SQL")


def arm_state(row, arm):
    item = row.get("evaluation", {}).get(arm, {})
    evidence = row.get("verification", {}).get(arm, {})
    mode = row.get("verification_mode", "strict")
    baseline_status = row.get("baselines", {}).get(arm, {}).get("status")
    valid = (complete_arm(item) and not item.get("censored") and not item.get("failures")
             and verified(evidence, mode)
             and baseline_status not in {"unavailable", "rejected_verification", "timeout", "error", "failed"})
    return {"status": item.get("status", "missing"), "valid_for_comparison": valid,
            "median_ms": item.get("median_ms"), "samples_ms": item.get("samples_ms", []),
            "verification_accepted": evidence.get("accepted"), "verification_valid": verified(evidence, mode),
            "proof": evidence.get("proof"), "formal": evidence.get("formal"),
            "baseline_status": baseline_status, "censored": bool(item.get("censored")),
            "error": item.get("error")}


def classify(row):
    arms = {arm: arm_state(row, arm) for arm in ("mcts", *BASELINES)}
    valid = [arm for arm in BASELINES if arms[arm]["valid_for_comparison"]]
    best = min(valid, key=lambda arm: (arms[arm]["median_ms"], arm)) if valid else None
    if best is None:
        status = "no_valid_baseline"
    elif not arms["mcts"]["valid_for_comparison"]:
        status = "mcts_invalid_or_incomplete"
    elif arms["mcts"]["median_ms"] >= arms[best]["median_ms"]:
        status = "latency_miss"
    else:
        status = "beat_available_baselines"
    return {"comparison_status": status, "best_available_valid_baseline": best,
            "available_valid_baselines": valid, "arms": arms,
            "miss": status in {"latency_miss", "mcts_invalid_or_incomplete"},
            "comparison_rule": "MCTS median >= minimum successful, verified available R-Bot/QUITE median; failures have no substitute latency"}


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def plan_root(plan):
    while isinstance(plan, list):
        plan = plan[0] if plan else {}
    return plan.get("Plan", plan) if isinstance(plan, dict) else {}


def inspect_plan(plan):
    """Report recorded actual work; estimates never become observed latency."""
    root = plan_root(plan)
    nodes, repeated, spills, reads, scans = [], [], [], [], []

    def visit(node, path, parent=None, child_index=None):
        loops, elapsed = number(node.get("Actual Loops")), number(node.get("Actual Total Time"))
        rows = number(node.get("Actual Rows"))
        # TIMING OFF is still EXPLAIN ANALYZE: actual cardinalities, loops,
        # spill and buffer counters remain measurements without node timings.
        measured = loops is not None and rows is not None
        work = loops * elapsed if measured and elapsed is not None else None
        child_work = 0
        children_timed = True
        for child in node.get("Plans", []):
            a, b = number(child.get("Actual Loops")), number(child.get("Actual Total Time"))
            children_timed = children_timed and a is not None and b is not None
            child_work += a * b if a is not None and b is not None else 0
        item = {"path": path, "node_type": node.get("Node Type"), "relation": node.get("Relation Name"),
                "schema": node.get("Schema"), "alias": node.get("Alias"), "subplan_name": node.get("Subplan Name"),
                "actual_loops": loops, "actual_rows_per_loop": rows,
                "actual_row_visits": loops * rows if measured else None,
                "actual_inclusive_work_ms": work,
                "exclusive_work_heuristic_ms": max(0, work - child_work) if work is not None and children_timed else None,
                "plan_rows_estimate": number(node.get("Plan Rows")),
                "planner_total_cost_estimate": number(node.get("Total Cost")),
                "conditions": {k: node[k] for k in ("Filter", "Join Filter", "Hash Cond", "Index Cond", "Merge Cond") if k in node}}
        if measured:
            nodes.append(item)
        is_subplan = node.get("Parent Relationship") == "SubPlan" or str(node.get("Subplan Name", "")).startswith("SubPlan")
        is_inner = parent and parent.get("Node Type") == "Nested Loop" and (node.get("Parent Relationship") == "Inner" or child_index == 1)
        if measured and loops > 1 and (is_subplan or is_inner):
            repeated.append({**item, "kind": "repeated_subplan" if is_subplan else "repeated_nested_loop_inner"})
        spill_fields = {k: node[k] for k in ("Sort Method", "Sort Space Type", "Sort Space Used", "Temp Read Blocks", "Temp Written Blocks", "Hash Batches", "HashAgg Batches", "Disk Usage") if k in node}
        spilled = (str(node.get("Sort Space Type", "")).lower() == "disk"
                   or "external" in str(node.get("Sort Method", "")).lower()
                   or (number(node.get("Hash Batches")) or 0) > 1
                   or (number(node.get("HashAgg Batches")) or 0) > 1
                   or any((number(node.get(k)) or 0) > 0 for k in ("Temp Read Blocks", "Temp Written Blocks", "Disk Usage")))
        if measured and spilled:
            spills.append({**item, "recorded_spill_fields": spill_fields})
        io = {k: node[k] for k in ("Shared Hit Blocks", "Shared Read Blocks", "Shared Written Blocks", "Local Read Blocks", "Temp Read Blocks", "Temp Written Blocks", "I/O Read Time", "I/O Write Time") if k in node}
        if measured and any((number(node.get(k)) or 0) > 0 for k in ("Shared Read Blocks", "Local Read Blocks", "Temp Read Blocks", "I/O Read Time")):
            reads.append({**item, "recorded_io_fields": io})
        if measured and "Scan" in str(node.get("Node Type", "")) and node.get("Relation Name"):
            removed = {k: number(node[k]) for k in ("Rows Removed by Filter", "Rows Removed by Index Recheck") if k in node}
            scans.append({**item, "rows_removed_per_loop": removed,
                          "observed_rows_examined_heuristic": loops * (rows + sum(value or 0 for value in removed.values()))})
        for index, child in enumerate(node.get("Plans", [])):
            visit(child, path + [index], node, index)

    if root:
        visit(root, [])
    timed = bool(nodes) and all(item["exclusive_work_heuristic_ms"] is not None for item in nodes)
    ranking = "exclusive_work_heuristic_ms" if timed else "actual_row_visits"
    nodes.sort(key=lambda x: (-(x[ranking] or 0), x["path"]))
    relations = Counter((item["schema"], item["relation"]) for item in scans)
    repeated_scans = [item for item in scans if relations[(item["schema"], item["relation"])] > 1 or item["actual_loops"] > 1]
    document = plan
    while isinstance(document, list):
        document = document[0] if document else {}
    return {"analyze_available": bool(nodes), "node_timing_available": any(item["actual_inclusive_work_ms"] is not None for item in nodes),
            "server_execution_time_ms": number(document.get("Execution Time")) if isinstance(document, dict) else None,
            "planner_total_cost_estimate": number(root.get("Total Cost")),
            "hotspot_ranking": ranking, "hotspots": nodes[:8], "repeated_work": repeated,
            "spills": spills, "io_reads": reads, "scans": scans, "repeated_scans": repeated_scans,
            "interpretation": "Actual cardinalities, loops, buffers and node times are recorded observations. TIMING OFF has no per-node latency. Row-visit ranking and time multiplication/subtraction are review heuristics, not additive CPU attribution. Parallel work and inclusive counters overlap. Planner cost is an estimate, never milliseconds. No causal benefit or equivalence is inferred."}


def plan_for(row, arm):
    plan = row.get("evaluation", {}).get(arm, {}).get("plan")
    pointer = f"/evaluation/{arm}/plan"
    if not plan and arm == "mcts":
        plan = row.get("search", {}).get("measurement", {}).get("plan")
        pointer = "/search/measurement/plan"
    return {"json_pointer": pointer, "sha256": digest(plan or {}), **inspect_plan(plan or {})}


_PATTERNS = {
    "repeated_subplan": {
        "name": "decorrelate_measured_repeated_subplan",
        "conditions": ["A stored EXPLAIN ANALYZE shows a SubPlan actually executing more than once.", "Identify every correlation key and the scalar/existence NULL and duplicate semantics before rewriting."],
        "transformations": ["Consider computing aggregate values once per correlation key and joining the grouped relation, or exposing a duplicate-preserving semi/anti join."],
        "safety_checks": ["Preserve empty scalar aggregate NULL behavior and aggregate result types.", "Never replace IN/EXISTS with an ordinary many-match join without controlling duplicate multiplication.", "Prove NOT IN/NOT EXISTS NULL equivalence from explicit constraints or retain NULL-aware logic."]},
    "repeated_nested_loop_inner": {
        "name": "reconsider_measured_repeated_join_inner",
        "conditions": ["A nested-loop inner input has measured Actual Loops > 1.", "Confirm that repeated inner work is material and distinguish indexed probes from repeated scans."],
        "transformations": ["Consider preaggregation or a reusable filtered relation, or a supported statement-local join-strategy hint as an alternative for measurement."],
        "safety_checks": ["Retain original join keys, filters and bag multiplicities.", "Use hints only when the server supports them; validate again for the current database scale."]},
    "spill": {
        "name": "reduce_input_to_measured_spilling_operator",
        "conditions": ["An actual plan records external/disk sort, temporary blocks, multiple hash batches, or nonzero disk usage."],
        "transformations": ["Consider safe predicate pushdown, narrowing projected columns, or duplicate-preserving preaggregation before the spilling sort/hash; test supported statement-local planner settings separately."],
        "safety_checks": ["Preserve outer-join null extension, grouping keys, distinctness and ORDER BY/LIMIT semantics.", "Spill evidence alone does not establish that additional memory or a rewrite improves elapsed time."]},
    "io_reads": {
        "name": "reduce_measured_read_input_or_reuse_scan",
        "conditions": ["Actual plan buffer counters record physical/local/temp reads or positive recorded I/O read time.", "Check selectivity and repeated table scans before choosing a transformation."],
        "transformations": ["Consider pushing safe filters before fact-table work or sharing an identical filtered input through materialization; compare against inlining."],
        "safety_checks": ["Keep duplicate rows and correlation predicates intact.", "Inclusive buffer counters cannot be summed across ancestors; cache state can change the result."]},
    "aggregate_hotspot": {
        "name": "consider_duplicate_preserving_preaggregation",
        "conditions": ["A measured Aggregate operator appears among the actual plan hotspots.", "A join or repeated aggregation presents a concrete opportunity to reduce aggregate input."],
        "transformations": ["Consider preaggregating by all required join/group keys and retaining an outer aggregate when duplicate dimension rows can multiply input."],
        "safety_checks": ["Use only distributive/algebraic decompositions with correct types, NULL and empty-input results.", "Do not assume primary or foreign key uniqueness absent declared constraints."]},
    "scan_input": {
        "name": "review_measured_scan_filter_and_input",
        "conditions": ["An actual plan records rows and loops for a sequential scan with a filter or more than one execution.", "Inspect the recorded predicate, rows removed and table reuse; a sequential scan can be the best available plan."],
        "transformations": ["Consider moving an equivalent filter below a safe join/aggregate boundary, making a supported predicate sargable, or reusing an identical filtered relation when it is consumed repeatedly."],
        "safety_checks": ["Preserve NULL, casts, collation, outer joins and bag multiplicities.", "Do not infer that a missing index, scan type, or observed row count causes the latency miss."]},
    "repeated_scan": {
        "name": "review_reuse_of_measured_repeated_scan",
        "conditions": ["An actual plan records multiple scan nodes for one relation or a scan with Actual Loops > 1.", "Compare the full projections, predicates and correlations; scans of one table need not represent identical reusable work."],
        "transformations": ["Consider sharing an identical filtered relation with a materialized CTE, or replacing independent aggregations with a duplicate-preserving grouped/window computation; compare with inlining."],
        "safety_checks": ["Keep distinct correlation keys, duplicate rows, empty aggregate behavior and statement-local visibility.", "Materialization can increase work; require fresh root equivalence verification and search timings."]},
}


def pattern_kinds(plan):
    if not plan["analyze_available"]:
        return []
    kinds = {"repeated_subplan" if item["kind"] == "repeated_subplan" else "repeated_nested_loop_inner" for item in plan["repeated_work"]}
    if plan["spills"]:
        kinds.add("spill")
    if plan["io_reads"]:
        kinds.add("io_reads")
    if plan["repeated_scans"]:
        kinds.add("repeated_scan")
    if any("Seq Scan" in str(item["node_type"]) and (item["conditions"].get("Filter") or item["actual_loops"] > 1) for item in plan["scans"]):
        kinds.add("scan_input")
    if any("Aggregate" in str(item["node_type"]) for item in plan["hotspots"]):
        kinds.add("aggregate_hotspot")
    return sorted(kinds)


DEFINITION_FIELDS = ("action_class", "trigger", "name", "conditions", "transformations", "safety_checks")


def pattern_identity(pattern):
    return "action_pattern:" + digest({key: pattern[key] for key in DEFINITION_FIELDS})


def draft_pattern(kind, source, scope):
    definition = {"action_class": "B", "trigger": {"kind": kind}, **deepcopy(_PATTERNS[kind])}
    identity = pattern_identity(definition)
    query = source["query_sha256"]
    template = source["template_sha256"]
    return {"id": identity, "status": "draft", **definition,
            "validation": {"status": "unvalidated", "receipts": [], "semantics_scope": None},
            "leakage": {"scope": "transductive_query_only" if scope == "transductive" else "template_disjoint",
                        "exclude_query_sha256": [query], "exclude_template_sha256": [template],
                        "allow_schema_sha256": [source["schema_sha256"]] if source.get("schema_sha256") else [],
                        "allow_query_sha256": [query] if scope == "transductive" else []},
            "provenance": {"corpus_role": "draft_action_pattern", "feedback_scope": scope,
                           "source_records": [source], "claims": "Unvalidated rewrite hypothesis; measured symptoms do not establish causal improvement or proof."}}


def merge_patterns(patterns):
    merged = {}
    for original in patterns:
        item = deepcopy(original)
        for field in ("exclude_query_sha256", "exclude_template_sha256", "allow_query_sha256", "allow_schema_sha256"):
            item["leakage"][field] = sorted(set(item["leakage"][field]))
        for field, container in (("source_records", "provenance"), ("receipts", "validation")):
            unique = {digest(value): value for value in item[container][field]}
            item[container][field] = [unique[key] for key in sorted(unique)]
        # Different leakage scopes must never become one reusable pattern.
        key = (item["id"], item["leakage"]["scope"])
        if key not in merged:
            merged[key] = deepcopy(item)
            continue
        target = merged[key]
        if any(target.get(field) != item.get(field) for field in DEFINITION_FIELDS):
            raise ValueError("One pattern id cannot name different transformation definitions")
        if target["provenance"]["feedback_scope"] != item["provenance"]["feedback_scope"]:
            raise ValueError("Cannot merge different feedback scopes")
        for field in ("exclude_query_sha256", "exclude_template_sha256", "allow_query_sha256", "allow_schema_sha256"):
            target["leakage"][field] = sorted(set(target["leakage"][field]) | set(item["leakage"][field]))
        for field, container in (("source_records", "provenance"), ("receipts", "validation")):
            unique = {digest(value): value for value in target[container][field] + item[container][field]}
            target[container][field] = [unique[key] for key in sorted(unique)]
        if item.get("status") == "validated":
            target["status"] = "validated"
            target["validation"].update({key: value for key, value in item["validation"].items() if key != "receipts"})
            target["provenance"]["corpus_role"] = "validated_action_pattern"
            target["provenance"]["claims"] = item["provenance"]["claims"]
        receipts = target["validation"]["receipts"]
        if receipts:
            target["validation"]["semantics_scope"] = "formal" if all(value.get("formal") for value in receipts) else "finite_data"
    return sorted(merged.values(), key=lambda x: (x["id"], x["leakage"]["scope"]))


def diagnose(rows, raw_path, metadata, scope="development"):
    validate_run(rows, metadata, scope)
    raw_hash = digest(Path(raw_path).read_bytes())
    misses, audit, drafts = [], [], []
    for line, row in rows:
        result = classify(row)
        original = row.get("original_sql", "")
        source = {"raw_path": str(Path(raw_path).resolve()), "raw_sha256": raw_hash, "line": line,
                  "run_hash": row["run_hash"], "task_id": row["task_id"], "query_id": row.get("query_id"),
                  "query_sha256": sql_sha256(original), "template_sha256": template_sha256(original),
                  "schema_sha256": row.get("schema_sha256"),
                  "source_commit": row.get("manifest_provenance", {}).get("source_commit"),
                  "code_sha256": metadata.get("fingerprint", {}).get("code_sha256"),
                  "upstream_fingerprints": metadata.get("fingerprint", {}).get("upstreams", {}),
                  "benchmark": row.get("benchmark"), "dataset": row.get("dataset"),
                  "budget": row.get("budget"), "ablation": row.get("ablation")}
        result.update(provenance=source, feedback_scope=scope,
                      evaluation_role="transductive_tuning_feedback" if scope == "transductive" else "development_feedback",
                      unbiased_final_evaluation=False)
        audit.append(deepcopy(result))
        if not result["miss"]:
            continue
        arms = ["mcts", result["best_available_valid_baseline"]]
        result["plans"] = {arm: plan_for(row, arm) for arm in arms}
        kinds = pattern_kinds(result["plans"]["mcts"])
        supported_source = {**source, "plan_evidence": {key: result["plans"]["mcts"][key] for key in ("json_pointer", "sha256")}}
        proposed = [draft_pattern(kind, supported_source, scope) for kind in kinds]
        result["proposed_pattern_ids"] = [p["id"] for p in proposed]
        result["diagnosis_status"] = "measured_plan_hypotheses" if kinds else "needs_review_no_supported_rewrite_pattern"
        result["claims"] = "Descriptive measured-plan symptoms and hypotheses only; a tie/miss can reflect noise, identity pairing, search coverage, verification restrictions, or a real missed rewrite. No automatic causal or proof claim."
        misses.append(result)
        drafts.extend(proposed)
    return misses, audit, merge_patterns(drafts)


def valid_search_measurement(measurement):
    samples = measurement.get("samples_ms", [])
    return (complete_arm(measurement) and not measurement.get("censored") and not measurement.get("failures")
            and isinstance(samples, list) and len(samples) >= 3
            and all(number(value) is not None and value > 0 for value in samples)
            and math.isclose(measurement["median_ms"], statistics.median(samples), rel_tol=1e-9, abs_tol=1e-9))


def load_evidence_run(path, raw_hash, scope, cache):
    path = Path(path).resolve()
    key = (str(path), raw_hash, scope)
    if key not in cache:
        if digest(path.read_bytes()) != raw_hash:
            raise ValueError("Receipt/source raw file hash mismatch")
        metadata = json.loads(path.with_name("run.json").read_text())
        rows = read_jsonl(path)
        validate_run(rows, metadata, scope)
        cache[key] = (rows, metadata)
    return cache[key]


def validate_pattern(pattern, cache):
    if pattern.get("id") != pattern_identity(pattern):
        raise ValueError("Pattern id does not match its transformation definition")
    kind = pattern["trigger"].get("kind")
    if kind not in _PATTERNS or pattern.get("action_class") != "B":
        raise ValueError("Unknown plan-backed transformation pattern")
    if not all(isinstance(pattern.get(key), list) and pattern[key] and all(isinstance(item, str) and item.strip() for item in pattern[key])
               for key in ("conditions", "transformations", "safety_checks")):
        raise ValueError("Patterns require explicit conditions, transformations and safety checks")
    scope = pattern["provenance"].get("feedback_scope")
    expected_scope = "template_disjoint" if scope == "development" else "transductive_query_only"
    if scope not in SCOPES or pattern["leakage"].get("scope") != expected_scope:
        raise ValueError("Pattern feedback/leakage scopes differ")
    sources = pattern["provenance"].get("source_records", [])
    if not sources:
        raise ValueError("Pattern requires an actual measured-plan source")
    for source in sources:
        rows, metadata = load_evidence_run(source["raw_path"], source["raw_sha256"], scope, cache)
        found = [(line, row) for line, row in rows if row["task_id"] == source["task_id"]]
        if len(found) != 1:
            raise ValueError("Pattern source task was not found exactly once")
        line, row = found[0]
        expected = {"line": line, "run_hash": row["run_hash"], "query_sha256": sql_sha256(row["original_sql"]),
                    "template_sha256": template_sha256(row["original_sql"]), "schema_sha256": row.get("schema_sha256"),
                    "source_commit": row.get("manifest_provenance", {}).get("source_commit"),
                    "code_sha256": metadata.get("fingerprint", {}).get("code_sha256"),
                    "upstream_fingerprints": metadata.get("fingerprint", {}).get("upstreams", {})}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("Pattern source query/schema/commit provenance mismatch")
        plan = plan_for(row, "mcts")
        if (not classify(row)["miss"] or kind not in pattern_kinds(plan)
                or source.get("plan_evidence") != {key: plan[key] for key in ("json_pointer", "sha256")}):
            raise ValueError("Pattern source has no matching measured miss/plan symptom")
        for key, value in (("exclude_query_sha256", expected["query_sha256"]),
                           ("exclude_template_sha256", expected["template_sha256"])):
            if value not in pattern["leakage"].get(key, []):
                raise ValueError("Pattern source is missing from leakage exclusions")
        schema = expected["schema_sha256"]
        if not is_sha256(schema) or schema not in pattern["leakage"].get("allow_schema_sha256", []):
            raise ValueError("Pattern requires recorded schema identity and a matching schema allowlist")
        if not is_sha256(expected["code_sha256"]) or not expected["source_commit"]:
            raise ValueError("Pattern requires recorded code snapshot and source commit")
        if scope == "transductive" and expected["query_sha256"] not in pattern["leakage"].get("allow_query_sha256", []):
            raise ValueError("Transductive pattern source is missing from its query allowlist")


def validate_receipt(receipt, pattern, cache=None):
    cache = {} if cache is None else cache
    if (receipt.get("pattern_id") != pattern["id"] or receipt.get("status") != "validated"
            or receipt.get("phase") != "search" or receipt.get("conditions_reviewed") is not True
            or receipt.get("safety_checks_reviewed") is not True
            or receipt.get("matched_pattern_kind") != pattern["trigger"]["kind"]
            or not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip()):
        raise ValueError("A reviewed validated search-phase receipt is required")
    path = Path(receipt["raw_path"])
    scope = receipt.get("feedback_scope")
    if scope not in SCOPES or scope != pattern["provenance"]["feedback_scope"]:
        raise ValueError("Receipt/draft feedback scopes differ")
    rows, metadata = load_evidence_run(path, receipt.get("raw_sha256"), scope, cache)
    candidates = [row for _, row in rows if row.get("task_id") == receipt.get("task_id")]
    if len(candidates) != 1:
        raise ValueError("Receipt task was not found exactly once")
    row = candidates[0]
    if row.get("run_hash") != receipt.get("run_hash"):
        raise ValueError("Receipt run hash mismatch")
    trace = row.get("search", {}).get("trace", [])
    index = receipt.get("candidate_trace_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(trace):
        raise ValueError("Receipt must reference an actual search candidate trace index")
    event = trace[index]
    roots = [item for item in trace if item.get("event") == "root"]
    if len(roots) != 1 or roots[0].get("sql") != row["original_sql"]:
        raise ValueError("Receipt needs exactly one search root matching original SQL")
    root = roots[0]
    if (event.get("event") != "candidate" or event.get("status") != "accepted"
            or not isinstance(event.get("sql"), str)
            or not event["sql"].strip() or event["sql"] == row["original_sql"]
            or event.get("action_class") != pattern["action_class"]
            or not receipt.get("applied_action") or event.get("action") != receipt["applied_action"]
            or event.get("verification_against") != "original_root"):
        raise ValueError("Receipt does not reference the claimed accepted root-verified search action")
    evidence = event.get("verification", {})
    mode = row.get("verification_mode", "strict")
    if not verified(evidence, mode) or not verified(root.get("verification", {}), mode):
        raise ValueError("Receipt candidate lacks accepted equivalence evidence")
    query_hash = sql_sha256(row["original_sql"])
    schema_hash = row.get("schema_sha256")
    source_commit = row.get("manifest_provenance", {}).get("source_commit")
    if (not schema_hash or not source_commit or receipt.get("root_query_sha256") != query_hash
            or receipt.get("schema_sha256") != schema_hash or receipt.get("source_commit") != source_commit
            or receipt.get("candidate_sql_sha256") != sql_sha256(event["sql"])):
        raise ValueError("Receipt root/query/schema/source commit identity mismatch")
    fingerprints = metadata.get("fingerprint", {}).get("source_fingerprints", {}).get(row.get("query_id"))
    if fingerprints and (fingerprints.get("query") != query_hash or fingerprints.get("schema") != schema_hash):
        raise ValueError("Receipt query/schema differs from immutable run fingerprint")
    candidate_ms, root_ms = event.get("measurement", {}), root.get("measurement", {})
    if not valid_search_measurement(candidate_ms) or not valid_search_measurement(root_ms):
        raise ValueError("Receipt needs successful median-of-three search measurements for candidate and root")
    if candidate_ms["median_ms"] >= root_ms["median_ms"]:
        raise ValueError("Receipt has no observed search improvement over its original root")
    if pattern["trigger"]["kind"] not in pattern_kinds(inspect_plan(root_ms.get("plan", {}))):
        raise ValueError("Receipt search root has no actual plan symptom matching the pattern")
    template_hash = template_sha256(row["original_sql"])
    return {"receipt_sha256": digest(receipt), "raw_path": str(path.resolve()), "raw_sha256": receipt["raw_sha256"],
            "run_hash": row["run_hash"], "task_id": row["task_id"], "candidate_trace_index": index,
            "candidate_sql_sha256": sql_sha256(event["sql"]), "query_sha256": query_hash,
            "root_query_sha256": query_hash, "schema_sha256": schema_hash, "source_commit": source_commit,
            "code_sha256": metadata.get("fingerprint", {}).get("code_sha256"),
            "upstream_fingerprints": metadata.get("fingerprint", {}).get("upstreams", {}),
            "template_sha256": template_hash, "reviewer": receipt["reviewer"], "applied_action": receipt["applied_action"],
            "matched_pattern_kind": receipt["matched_pattern_kind"], "conditions_reviewed": True, "safety_checks_reviewed": True,
            "search_root_plan_sha256": digest(root_ms.get("plan", {})),
            "search_candidate_plan_sha256": digest(candidate_ms.get("plan", {})),
            "phase": "search", "root_median_ms": root_ms["median_ms"], "candidate_median_ms": candidate_ms["median_ms"],
            "verification_mode": row.get("verification_mode", "strict"), "formal": evidence.get("formal") is True,
            "proof": evidence.get("proof"), "claim_scope": "observed training improvement vs original root; no baseline-win or causal claim"}


def promote(patterns, receipts, existing=()):
    approved = []
    rejections = []
    cache = {}
    for pattern in merge_patterns(patterns):
        try:
            validate_pattern(pattern, cache)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejections.append({"pattern_id": pattern.get("id"), "reason": str(exc)})
            continue
        matching = [r for r in receipts if r.get("pattern_id") == pattern["id"]]
        validated = []
        for receipt in matching:
            try:
                validated.append(validate_receipt(receipt, pattern, cache))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejections.append({"pattern_id": pattern["id"], "receipt_sha256": digest(receipt), "reason": str(exc)})
        if not validated:
            if not matching:
                rejections.append({"pattern_id": pattern["id"], "reason": "No validation receipt"})
            continue
        validated = list({value["receipt_sha256"]: value for value in validated}.values())
        validated.sort(key=lambda value: value["receipt_sha256"])
        entry = deepcopy(pattern)
        entry["status"] = "validated"
        entry["validation"] = {"status": "validated", "receipts": validated,
                               "semantics_scope": "formal" if all(v["formal"] for v in validated) else "finite_data",
                               "requires_root_equivalence_gate": True,
                               "timing_scope": "search_training_only"}
        entry["provenance"]["corpus_role"] = "validated_action_pattern"
        entry["provenance"]["claims"] = "Reviewed hypothesis supported by root verification and observed search-phase timings; transferability and causal benefit remain unestablished."
        entry["leakage"]["exclude_query_sha256"] = sorted(set(entry["leakage"]["exclude_query_sha256"]) | {v["query_sha256"] for v in validated})
        entry["leakage"]["exclude_template_sha256"] = sorted(set(entry["leakage"]["exclude_template_sha256"]) | {v["template_sha256"] for v in validated})
        entry["leakage"]["allow_schema_sha256"] = sorted(set(entry["leakage"]["allow_schema_sha256"]) | {v["schema_sha256"] for v in validated})
        if entry["leakage"]["scope"] == "transductive_query_only":
            entry["leakage"]["allow_query_sha256"] = sorted(set(entry["leakage"]["allow_query_sha256"]) | {v["query_sha256"] for v in validated})
        approved.append(entry)
    for entry in existing:
        validation = entry.get("validation", {})
        if (entry.get("status") != "validated" or validation.get("status") != "validated"
                or validation.get("timing_scope") != "search_training_only"
                or validation.get("requires_root_equivalence_gate") is not True
                or not validation.get("receipts") or entry.get("id") != pattern_identity(entry)):
            raise ValueError("Approved library contains an unvalidated entry")
    return merge_patterns([*existing, *approved]), rejections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tuning-scope", choices=SCOPES, default="development")
    parser.add_argument("--propose-patterns", type=Path)
    parser.add_argument("--promote-validated", type=Path, help="Reviewed receipts JSONL, referencing search events only")
    parser.add_argument("--patterns", type=Path, help="Draft patterns JSONL for promotion")
    parser.add_argument("--library", type=Path, default=ROOT / "prompts/action_patterns.jsonl")
    args = parser.parse_args(argv)
    try:
        if args.promote_validated:
            if not args.patterns or args.raw:
                raise ValueError("Promotion requires --patterns and --promote-validated, without --raw")
            patterns = [row for _, row in read_jsonl(args.patterns)]
            receipts = [row for _, row in read_jsonl(args.promote_validated)]
            existing = [row for _, row in read_jsonl(args.library)] if args.library.exists() else []
            library, rejections = promote(patterns, receipts, existing)
            write_jsonl(args.library, library)
            report = {"library": str(args.library), "validated_patterns": len(library), "rejected_receipts": rejections}
            args.library.with_suffix(".promotion.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
            return 1 if rejections else 0
        if not args.raw:
            raise ValueError("Diagnosis requires --raw")
        metadata_path = args.run_metadata or args.raw.with_name("run.json")
        metadata = json.loads(metadata_path.read_text())
        misses, audit, drafts = diagnose(read_jsonl(args.raw), args.raw, metadata, args.tuning_scope)
        output = args.output or args.raw.parent / "feedback"
        write_jsonl(output / "missed_opportunities.jsonl", misses)
        write_jsonl(output / "feedback_audit.jsonl", audit)
        if args.propose_patterns:
            write_jsonl(args.propose_patterns, drafts)
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_hash": metadata["run_hash"],
                  "feedback_scope": args.tuning_scope, "raw_sha256": digest(args.raw.read_bytes()),
                  "records": len(audit), "misses": len(misses), "draft_patterns": len(drafts),
                  "comparison_statuses": dict(Counter(item["comparison_status"] for item in audit)),
                  "approved_library_modified": False, "unbiased_final_evaluation": False,
                  "evaluation_requirement": "Freeze the library and candidates before a new, untouched final evaluation. Transductive tuning remains a separately labeled result."}
        (output / "feedback_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
