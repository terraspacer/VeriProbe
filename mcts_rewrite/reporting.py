"""Auditable, denominator-preserving summaries of independent evaluation runs.

Failed or censored executions are never converted to latency numbers. Comparisons
use one complete, verified cohort shared by all four arms; completion is reported
separately so dropping difficult queries cannot satisfy the acceptance targets.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("original", "rbot", "quite", "mcts")
BASELINES = ("rbot", "quite")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, fraction):
    """Linear interpolation, including the singleton case (no numpy dependency)."""
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def latency_summary(values):
    values = [float(value) for value in values if finite_number(value) and value > 0]
    return {
        "n": len(values),
        "average_ms": statistics.mean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p90_ms": percentile(values, 0.9),
        "total_ms": sum(values) if values else None,
    }


def verified(evidence, mode="strict"):
    if mode not in {"strict", "hybrid", "differential", "exploratory"}:
        return False
    if not evidence or evidence.get("accepted") is not True:
        return False
    if not _matched_snapshot(evidence.get("execution", {})):
        return False
    if mode == "strict":
        return evidence.get("formal") is True and evidence.get("proof") in {"identity", "ast", "sqlsolver"}
    if evidence.get("formal") is True and evidence.get("proof") in {"identity", "ast", "sqlsolver"}:
        return True
    miniature_cases = evidence.get("miniatures", [])
    return (evidence.get("proof") == "differential" and evidence.get("formal") is False
            and len(miniature_cases) >= 5 and all(_matched_snapshot(case.get("comparison", {})) for case in miniature_cases))


def _matched_snapshot(execution):
    left_hash = (execution.get("original") or {}).get("sha256")
    right_hash = (execution.get("candidate") or {}).get("sha256")
    return (execution.get("equivalent") is True and execution.get("status") == "ok"
            and execution.get("same_snapshot") is True and isinstance(left_hash, str)
            and bool(left_hash) and left_hash == right_hash)


def complete_arm(arm):
    value = arm.get("median_ms")
    return arm.get("status") == "ok" and finite_number(value) and value > 0


def _comparison(rows, baseline):
    pairs = [(r["evaluation"][baseline]["median_ms"], r["evaluation"]["mcts"]["median_ms"]) for r in rows]
    improved = sum(base / candidate >= 1.10 for base, candidate in pairs)
    regressed = sum(candidate > base for base, candidate in pairs)
    total_base = sum(base for base, _ in pairs)
    total_candidate = sum(candidate for _, candidate in pairs)
    return {
        "n": len(pairs),
        "aggregate_latency_reduction": 1 - total_candidate / total_base if pairs else None,
        "mean_latency_reduction": 1 - total_candidate / total_base if pairs else None,
        "p90_latency_reduction": 1 - percentile([candidate for _, candidate in pairs], .9) / percentile([base for base, _ in pairs], .9) if pairs else None,
        "geometric_mean_speedup": math.exp(statistics.mean(math.log(base / candidate) for base, candidate in pairs)) if pairs else None,
        "median_speedup": statistics.median(base / candidate for base, candidate in pairs) if pairs else None,
        "improved_count": improved,
        "improvement_rate": improved / len(pairs) if pairs else None,
        "regression_count": regressed,
        "regression_rate": regressed / len(pairs) if pairs else None,
    }


def summarize_group(rows, expected, mode="strict"):
    """Summarize one source/benchmark/budget/ablation, with explicit denominators."""
    complete = [row for row in rows if row.get("status") == "ok" and all(complete_arm(row.get("evaluation", {}).get(arm, {})) for arm in ARMS)
                and all(verified(row.get("verification", {}).get(arm), mode) for arm in ARMS)]
    completed_ids = {row["query_id"] for row in complete}
    final_equivalent = sum(verified(row.get("verification", {}).get("mcts"), mode) for row in rows)
    final_formal = sum(verified(row.get("verification", {}).get("mcts"), "strict") for row in rows)
    wrong_final = sum(row.get("verification", {}).get("mcts", {}).get("execution", {}).get("status") == "ok"
                      and row["verification"]["mcts"]["execution"].get("equivalent") is False for row in rows)
    arm_statuses = {arm: dict(Counter(row.get("evaluation", {}).get(arm, {}).get("status", "missing") for row in rows)) for arm in ARMS}
    final_failures = dict(Counter(row.get("status", "unknown") for row in rows if row.get("status") != "ok"))
    rejections = Counter()
    for row in rows:
        for event in row.get("search", {}).get("trace", []):
            if event.get("status") == "rejected":
                rejections["search_candidate_rejected"] += 1
                if event.get("verification", {}).get("execution", {}).get("status") == "ok" and event["verification"]["execution"].get("equivalent") is False:
                    rejections["search_candidate_wrong_result"] += 1
        if row.get("rejected_search_selection"):
            rejections["final_selection_rejected_and_fallback"] += 1
        for baseline in row.get("baselines", {}).values():
            if baseline.get("status") == "rejected_verification":
                rejections["baseline_rejected"] += 1
    latency = {arm: latency_summary(row["evaluation"][arm]["median_ms"] for row in complete) for arm in ARMS}
    available_latency = {arm: latency_summary(row["evaluation"][arm]["median_ms"] for row in rows
                                             if complete_arm(row.get("evaluation", {}).get(arm, {}))
                                             and verified(row.get("verification", {}).get(arm), mode)) for arm in ARMS}
    search_wall = latency_summary(row.get("search_wall_ms") for row in rows)
    expanded = [row.get("search", {}).get("expanded") for row in rows if finite_number(row.get("search", {}).get("expanded"))]
    fallback_count = sum(bool(row.get("forced_verified_fallback") or row.get("search", {}).get("fallback")) for row in rows)
    comparisons = {base: _comparison(complete, base) for base in BASELINES}
    original_paired = [row for row in rows if row.get("status") in {"ok", "incomplete"}
                       and all(complete_arm(row.get("evaluation", {}).get(arm, {}))
                               and verified(row.get("verification", {}).get(arm), mode) for arm in ("original", "mcts"))]
    original_comparison = _comparison(original_paired, "original")
    original_all_complete = len(rows) == expected and len(original_paired) == expected
    all_complete = len(rows) == expected and len(completed_ids) == expected
    equivalence_rate = final_equivalent / expected if expected else None
    improvement_vs_original = {}
    latency_reduction_vs_original = {}
    for arm in ("mcts", *BASELINES):
        improved = sum(row["evaluation"]["original"]["median_ms"] / row["evaluation"][arm]["median_ms"] >= 1.10 for row in complete)
        improvement_vs_original[arm] = {"count": improved, "rate": improved / len(complete) if complete else None}
        latency_reduction_vs_original[arm] = {
            "mean": 1 - latency[arm]["average_ms"] / latency["original"]["average_ms"] if complete else None,
            "p90": 1 - latency[arm]["p90_ms"] / latency["original"]["p90_ms"] if complete else None,
        }
    point_gains = {base: (improvement_vs_original["mcts"]["rate"] - improvement_vs_original[base]["rate"])
                   if complete else None for base in BASELINES}
    latency_point_gains = {base: {stat: latency_reduction_vs_original["mcts"][stat] - latency_reduction_vs_original[base][stat]
                                 if complete else None for stat in ("mean", "p90")} for base in BASELINES}
    compared_finals = sum(row.get("verification", {}).get("mcts", {}).get("execution", {}).get("status") == "ok" for row in rows)
    test_database_finals = sum(row.get("verification", {}).get("mcts", {}).get("execution", {}).get("status") == "ok"
                              and row["verification"]["mcts"].get("evidence_scope", {}).get("scope") == "separate_verification_database"
                              for row in rows)
    targets = {
        "equivalence_at_least_99_percent": _target(equivalence_rate, lambda x: x >= .99, len(rows) == expected),
        "mean_latency_reduction_at_least_15_percent_vs_both": _target(
            min((x["aggregate_latency_reduction"] for x in comparisons.values()), default=None) if complete else None,
            lambda x: x >= .15, all_complete),
        "p90_latency_reduction_at_least_15_percent_vs_both": _target(
            min((x["p90_latency_reduction"] for x in comparisons.values()), default=None) if complete else None,
            lambda x: x >= .15, all_complete),
        "regression_below_1_percent_vs_original": _target(original_comparison["regression_rate"], lambda x: x < .01, original_all_complete),
        "zero_observed_wrong_final_results": _target(wrong_final, lambda x: x == 0, compared_finals == expected),
    }
    required = [target["status"] for target in targets.values()]
    status = "incomplete" if not all_complete or "incomplete" in required else ("pass" if all(value == "pass" for value in required) else "fail")
    return {
        "expected_queries": expected, "recorded_queries": len(rows),
        "paired_complete_queries": len(complete), "strict_all_query_completion": all_complete,
        "paired_query_ids": sorted(completed_ids), "final_equivalent_queries": final_equivalent,
        "equivalence_rate_all_queries": equivalence_rate, "verification_mode": mode,
        "verified_equivalence_rate_all_queries": equivalence_rate,
        "final_formally_verified_queries": final_formal,
        "final_finite_only_queries": sum(verified(row.get("verification", {}).get("mcts"), mode)
                                          and not verified(row.get("verification", {}).get("mcts"), "strict") for row in rows),
        "observed_wrong_final_results": wrong_final,
        "final_test_database_compared_queries": test_database_finals,
        "final_performance_database_compared_queries": compared_finals - test_database_finals,
        "formal_equivalence_rate_all_queries": final_formal / expected if expected else None,
        "arm_status_counts": arm_statuses, "query_failure_counts": final_failures,
        "rejection_counts": dict(rejections),
        "paired_latency": latency, "available_latency": available_latency, "comparisons": comparisons,
        "comparison_vs_original": original_comparison, "original_mcts_paired_queries": len(original_paired),
        "original_mcts_all_query_completion": original_all_complete,
        "search_wall_latency": search_wall, "mean_expansions": statistics.mean(expanded) if expanded else None,
        "verified_fallback_count": fallback_count, "fallback_rate_recorded_queries": fallback_count / len(rows) if rows else None,
        "baseline_modes": {base: dict(Counter(row.get("baselines", {}).get(base, {}).get("mode", "unavailable") for row in rows)) for base in BASELINES},
        "improvement_vs_original": improvement_vs_original,
        "improvement_rate_percentage_point_gain": point_gains,
        "latency_reduction_vs_original": latency_reduction_vs_original,
        "latency_reduction_percentage_point_gain": latency_point_gains,
        "targets": targets, "acceptance_status": status,
    }


def _target(value, predicate, complete):
    return {"value": value, "status": "incomplete" if value is None or not complete else "pass" if predicate(value) else "fail"}


def _model_context_issue(context):
    if not isinstance(context, dict) or not isinstance(context.get("transport"), str) or not context["transport"]:
        return "missing model transport"
    requested, actual = context.get("requested_role_models"), context.get("actual_role_models")
    if (not isinstance(requested, dict) or not isinstance(actual, dict) or not requested or requested.keys() != actual.keys()
            or not all(isinstance(value, str) and value.strip() for value in [*requested.values(), *actual.values()])):
        return "missing requested or actual model roles"
    substituted = any(requested[role] != actual[role] for role in requested)
    if not isinstance(context.get("model_substitution"), bool) or context["model_substitution"] != substituted:
        return "missing or inconsistent model substitution disclosure"
    if not all(key in context for key in ("actual_serving_model", "reasoning_effort", "published_model_configuration_reproduced",
                                          "sampling_controls_reproduced", "requested_sampling_options_not_enforced", "usage_known")):
        return "incomplete model sampling/effort disclosure"
    embedding = context.get("embedding_source")
    if not isinstance(embedding, dict) or not embedding.get("provider") or not isinstance(embedding.get("substituted"), bool):
        return "embedding source not disclosed separately"
    if context["transport"] == "codex_session_file_spool":
        if (context.get("reasoning_effort") != "xhigh" or any(name != "gpt-6.1-sol" for name in actual.values())
                or context.get("published_model_configuration_reproduced") is not False
                or context.get("sampling_controls_reproduced") is not False
                or context.get("usage_known") is not False):
            return "session model substitution or sampling limitations not fully disclosed"
    return None


def publication_eligibility(records, metadata):
    """Keep publication provenance separate from diagnostic numerical targets."""
    primary = [row for row in records if row.get("record_type") == "query_result"
               and row.get("ablation") == "full" and row.get("budget") == 16]
    reasons = set()
    if not primary:
        reasons.add("no full-method M=16 task records")
    modes, context_failures, contexts, calls = Counter(), Counter(), {}, Counter()
    for row in primary:
        for system in BASELINES:
            baseline = row.get("baselines", {}).get(system, {})
            mode = baseline.get("mode", "unavailable")
            modes[f"{system}:{mode}"] += 1
            if mode != "fresh_standalone":
                reasons.add("every publication baseline must be generated by the fresh standalone upstream method")
            provenance = baseline.get("provenance", {})
            context = provenance.get("model_context")
            issue = _model_context_issue(context)
            if issue:
                context_failures[f"{system}: {issue}"] += 1
                reasons.add("baseline model context is missing or incomplete")
            else:
                key = json.dumps(context, sort_keys=True)
                contexts[key] = context
            count = provenance.get("session_completion_count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                calls[system] += count
    run_context = (metadata or {}).get("model_context")
    run_issue = _model_context_issue(run_context)
    if run_issue:
        reasons.add("MCTS run model context is missing or incomplete")
    conditions = Counter(row.get("host_conditions", {}).get("status", "unobserved") for row in primary)
    quiet_values = [row.get("host_conditions", {}).get("quiet_at_observed_samples") for row in primary]
    quiet = False if any(value is False for value in quiet_values) else None if not quiet_values or any(value is None for value in quiet_values) else True
    return {"eligible": not reasons, "status": "eligible" if not reasons else "incomplete",
            "reasons": sorted(reasons), "baseline_modes": dict(modes),
            "baseline_model_context_failures": dict(context_failures),
            "baseline_model_contexts": [contexts[key] for key in sorted(contexts)],
            "mcts_model_context_issue": run_issue, "configured_mcts_model_context": run_context,
            "baseline_catalog_session_completion_references": dict(calls),
            "session_completion_scope": "Catalog completion references may repeat across query memberships; they are not new model calls during timed evaluation.",
            "model_context_scope": "Declared role configuration and substitutions; this alone is not evidence of successful per-task model calls.",
            "host_condition_status_counts": dict(conditions), "quiet_at_all_observed_task_samples": quiet,
            "host_conditions_scope": "Configured quiet criteria are checked at whole-task samples and include benchmark load. See each task's explicit thresholds; no quiet-between-samples or external-causality claim is supported. Missing samples cannot support a quiet-host claim. Pressure does not invalidate semantic verification or remove numerical results."}


def build_report(records, planned_groups, mode="strict", metadata=None):
    by_group = defaultdict(list)
    for record in records:
        if record.get("record_type") == "query_result":
            by_group[(record["source"], record["benchmark"], record.get("dataset", record["benchmark"]), record["ablation"], record["budget"])].append(record)
    groups = []
    for group in planned_groups:
        key = (group["source"], group["benchmark"], group.get("dataset", group["benchmark"]), group["ablation"], group["budget"])
        groups.append({**group, **summarize_group(by_group[key], group["expected_queries"], mode)})
    primary = [group for group in groups if group["ablation"] == "full" and group["budget"] == 16]
    dimensions = {(g["source"], g["benchmark"], g.get("dataset", g["benchmark"])) for g in groups}
    primary_dimensions = {(g["source"], g["benchmark"], g.get("dataset", g["benchmark"])) for g in primary}
    states = [group["acceptance_status"] for group in primary]
    acceptance = "incomplete" if not primary or dimensions != primary_dimensions or "incomplete" in states else "pass" if all(state == "pass" for state in states) else "fail"
    eligibility = publication_eligibility(records, metadata)
    mode_definitions = {"fresh_standalone": "Actual standalone upstream output generated for the bound input/schema/dataset, then measured locally from its audited catalog.",
                        "replay": "Historical saved upstream SQL measured locally; numerical comparisons are diagnostic for fresh-baseline publication claims.",
                        "unavailable": "No usable baseline output is available."}
    present_modes = sorted({baseline.get("mode", "unavailable") for row in records
                            for baseline in row.get("baselines", {}).values()})
    return {
        "metadata": metadata or {},
        "definitions": {
            "latency": "Independent holdout, median of three successful server execution times after one warmup; failed/censored arms have no median.",
            "cohort": "Per group, queries with all four arms verified and successfully timed. All-query completion is separately required for acceptance.",
            "speedup": "baseline latency / candidate latency; an improved query has speedup >= 1.10.",
            "regression": "Candidate median latency strictly exceeds original SQL median latency; required rate is <1% on verified original/MCTS pairs with all-query coverage. Regressions against standalone baselines are secondary diagnostics.",
            "primary_15_percent_target": "Both mean and p90 query latency must be reduced by at least 15% versus each standalone baseline on the common complete verified cohort.",
            "alternative_percentage_point_interpretation": "Difference between MCTS and each baseline in mean/p90 latency reduction relative to original SQL; reported separately and not substituted for the relative targets.",
            "acceptance_scope": "Full method at expansion budget M=16 for every requested source/benchmark/dataset; smaller budgets and ablations are diagnostic and need not meet performance targets.",
            "missing_or_timeout": "Preserved in denominators and statuses; no substituted zero or timeout-valued latency.",
            "baseline_mode": {name: mode_definitions.get(name, "See recorded baseline provenance.") for name in present_modes},
            "identity_pairing": "Arms with byte-identical SQL may share one explicitly annotated common execution; this is not an independent measurement of arm differences.",
            "partial_run": "A filtered/limited manifest cannot establish full upstream benchmark acceptance.",
            "verification": "Strict requires formal evidence plus a same-snapshot database result comparison. Hybrid accepts that standard or nonempty miniature differential checks plus a database result comparison. Finite testing is not a proof or a universal zero-error guarantee.",
            "verification_database_scope": "By default result comparisons execute on the performance database. An explicit separate verification database must pass catalog/settings compatibility checks; its rows and miniature rows supply result evidence, while performance rows are only planned and timed. Counts separately identify these scopes; no test-data comparison is described as hashing published-size rows.",
            "available_latency": "Each arm's successful verified executions, shown for diagnostics; these differing cohorts are not used for baseline acceptance comparisons.",
        },
        "groups": groups, "acceptance_status": acceptance, "publication_eligibility": eligibility,
        "acceptance_scope": {"ablation": "full", "budget": 16, "required_groups": len(dimensions), "present_groups": len(primary)},
    }


def write_reports(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    acceptance = {"status": report["acceptance_status"], "partial_run": report.get("metadata", {}).get("partial_run", False),
                  "scope": report.get("acceptance_scope"),
                  "groups": [{key: group.get(key) for key in ("source", "benchmark", "dataset", "scale", "ablation", "budget", "expected_queries", "recorded_queries", "paired_complete_queries", "strict_all_query_completion", "verified_equivalence_rate_all_queries", "formal_equivalence_rate_all_queries", "final_finite_only_queries", "observed_wrong_final_results", "original_mcts_paired_queries", "targets", "acceptance_status")} for group in report["groups"]]}
    acceptance["publication_scale_run"] = report.get("metadata", {}).get("publication_scale_run", False)
    acceptance["publication_eligibility"] = report.get("publication_eligibility", {"eligible": False})
    if (acceptance["partial_run"] or not acceptance["publication_scale_run"]
            or report.get("metadata", {}).get("verification_mode", "strict") not in {"strict", "hybrid", "differential"}
            or not acceptance["publication_eligibility"].get("eligible")):
        acceptance["full_benchmark_status"] = "incomplete"
    else:
        acceptance["full_benchmark_status"] = report["acceptance_status"]
    (output_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2, allow_nan=False) + "\n")
    columns = ["source", "benchmark", "dataset", "scale", "ablation", "budget", "expected_queries", "recorded_queries", "paired_complete_queries", "equivalence_rate_all_queries", "verified_equivalence_rate_all_queries", "formal_equivalence_rate_all_queries", "final_finite_only_queries", "observed_wrong_final_results", "acceptance_status"]
    columns += [f"{arm}_{stat}" for arm in ARMS for stat in ("average_ms", "median_ms", "p90_ms")]
    columns += [f"{arm}_available_{stat}" for arm in ARMS for stat in ("average_ms", "median_ms", "p90_ms")]
    columns += [f"vs_{base}_{metric}" for base in BASELINES for metric in ("aggregate_latency_reduction", "p90_latency_reduction", "improvement_rate", "regression_rate")]
    columns += ["vs_original_regression_rate", "vs_original_improvement_rate", "original_mcts_paired_queries"]
    columns += ["search_average_ms", "search_median_ms", "search_p90_ms", "mean_expansions", "fallback_rate_recorded_queries"]
    columns += ["final_test_database_compared_queries", "final_performance_database_compared_queries"]
    flattened = []
    for group in report["groups"]:
        row = {key: group[key] for key in columns if key in group}
        row.update({f"{arm}_{stat}": group["paired_latency"][arm][stat] for arm in ARMS for stat in ("average_ms", "median_ms", "p90_ms")})
        row.update({f"{arm}_available_{stat}": group["available_latency"][arm][stat] for arm in ARMS for stat in ("average_ms", "median_ms", "p90_ms")})
        row.update({f"vs_{base}_{metric}": group["comparisons"][base][metric] for base in BASELINES for metric in ("aggregate_latency_reduction", "p90_latency_reduction", "improvement_rate", "regression_rate")})
        row.update({"vs_original_regression_rate": group["comparison_vs_original"]["regression_rate"], "vs_original_improvement_rate": group["comparison_vs_original"]["improvement_rate"]})
        row.update({f"search_{stat}": group["search_wall_latency"][stat] for stat in ("average_ms", "median_ms", "p90_ms")})
        flattened.append(row)
    for name, rows in (("summary.csv", flattened), ("scaling.csv", [r for r in flattened if r["ablation"] == "full"]), ("ablation.csv", [r for r in flattened if r["ablation"] != "full"])):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
