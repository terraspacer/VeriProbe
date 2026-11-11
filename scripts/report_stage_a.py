#!/usr/bin/env python3
"""Build supplementary Stage A tables without executing SQL or model requests.

Default input: results/stage_a_sf10/raw.jsonl. The input run's saved manifest
supplies expected coverage. Outputs default to a separate report/ subdirectory.
Available timings and an error-free common four-system cohort are reported
separately. Missing, censored and failed timings remain null, never zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("original", "rbot", "quite", "mcts")
BENCHMARKS = ("tpch", "dsb", "calcite")
LIMITS = {"rbot": 2, "quite": 3, "mcts": 3}
LABELS = {
    "original": "Original PostgreSQL input",
    "rbot": "R-Bot batched standalone variant; lexical BM25 retrieval variant",
    "quite": "QUITE batched standalone variant; EXPLAIN and critic selection",
    "mcts": "Batched hybrid MCTS variant; at most two target canaries",
}


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def p90(values):
    if not values:
        return None
    values = sorted(values)
    location = 0.9 * (len(values) - 1)
    lower, upper = math.floor(location), math.ceil(location)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def json_lines(path):
    body = Path(path).read_bytes()
    rows = []
    for line_number, line in enumerate(body.splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append((row, line_number, sha256(line)))
    return rows, sha256(body)


def measured(arm):
    """Derive final query statistics from exactly three successful raw samples."""
    measurement = arm.get("measurement", {})
    samples = measurement.get("samples_ms", [])
    if measurement.get("status") != "ok":
        return None, "measurement_" + str(measurement.get("status", "missing"))
    if not isinstance(samples, list) or len(samples) != 3 or not all(finite_positive(value) for value in samples):
        return None, "invalid_three_run_samples"
    if measurement.get("runs", 3) != 3 or measurement.get("warmups", 1) != 1:
        return None, "invalid_final_timing_protocol"
    if arm.get("verification", {}).get("accepted") is not True:
        return None, "final_equivalence_not_accepted"
    return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples)}, None


def evidence_category(arm):
    evidence = arm.get("verification", {})
    if evidence.get("accepted") is not True:
        return "rejected" if evidence.get("accepted") is False else "missing"
    if evidence.get("proof") in {"identity", "ast"}:
        return "identity_or_conservative_ast"
    if evidence.get("formal") is True:
        return "formal_nonidentity"
    if evidence.get("proof") == "differential":
        return "finite_differential"
    return "accepted_other_nonformal"


def model_audit(record):
    ledger = record.get("model_call_audit", {})
    calls = [{**call, "_raw_call_index": index} for index, call in enumerate(ledger.get("calls", []))
             if isinstance(call, dict) and call.get("transport_invoked") is not False]
    # Retries are physical transport attempts, but not additional logical
    # generation actions. They remain visible in the audit while cap checks
    # apply to the non-retry actions that the study budget controls.
    logical_calls = [call for call in calls if call.get("retry") is not True]
    counts = Counter(call.get("arm", "unknown") for call in logical_calls)
    violations = []
    # Preserve the historical receipt field for compatibility: physical retry
    # attempts are visible here, while Stage A v2 cohort admission deliberately
    # ignores this diagnostic and uses logical-call caps.
    if len(calls) > 8:
        violations.append("actual_total_exceeds_8")
    if len(logical_calls) > 8:
        violations.append("actual_logical_total_exceeds_8")
    reported = ledger.get("total_invocations")
    reported_logical = ledger.get("logical_invocations")
    if isinstance(reported_logical, (int, float)) and not isinstance(reported_logical, bool) and reported_logical > 8:
        violations.append("reported_logical_total_exceeds_8")
    for arm, limit in LIMITS.items():
        if counts[arm] > limit:
            violations.append("actual_" + arm + "_exceeds_" + str(limit))
        reported_arm = ledger.get("logical_invocations_by_arm", {}).get(arm)
        if isinstance(reported_arm, (int, float)) and not isinstance(reported_arm, bool) and reported_arm > limit:
            violations.append("reported_" + arm + "_exceeds_" + str(limit))
    if set(counts) - set(LIMITS):
        violations.append("unknown_call_arm")
    return {"actual_total_calls": len(calls), "actual_logical_calls": len(logical_calls), "reported_total_calls": reported,
            "audit_total_mismatch": reported is not None and reported != len(calls),
            "calls_by_arm": {arm: counts[arm] for arm in LIMITS},
            "failed_calls": sum(call.get("status") in {"error", "timeout"} for call in calls),
            "blocked_calls": len(ledger.get("blocked_calls", [])),
            "cap_violations": violations, "cap_violated": bool(violations), "calls": calls}


def load_inputs(raw, manifest):
    source_rows, raw_hash = json_lines(raw)
    manifest_rows, manifest_hash = json_lines(manifest)
    expected = {}
    for row, line, _ in manifest_rows:
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id or query_id in expected:
            raise ValueError(f"Manifest has a missing or repeated query ID at line {line}")
        if row.get("benchmark") not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark at manifest line {line}")
        expected[query_id] = row
    if not expected:
        raise ValueError("Expected-cohort manifest is empty")
    records, locations = {}, {}
    for record, line, line_hash in source_rows:
        query_id = record.get("query_id")
        if query_id not in expected:
            raise ValueError(f"Raw line {line} query is outside the expected manifest: {query_id}")
        if query_id in records:
            raise ValueError(f"Duplicate raw query ID at line {line}: {query_id}")
        if record.get("benchmark") != expected[query_id]["benchmark"]:
            raise ValueError(f"Raw benchmark differs from manifest at line {line}")
        manifest_sql = expected[query_id].get("sql")
        if manifest_sql is not None and record.get("original_sql") != manifest_sql:
            raise ValueError(f"Raw input SQL differs from manifest at line {line}")
        records[query_id] = record
        locations[query_id] = {"raw_line": line, "record_line_sha256": line_hash}
    return expected, records, locations, raw_hash, manifest_hash


def build_report(raw, manifest):
    raw, manifest = Path(raw).resolve(), Path(manifest).resolve()
    expected, records, locations, raw_hash, manifest_hash = load_inputs(raw, manifest)
    valid, missing_reasons, audits = {}, {}, {}
    for query_id, row in expected.items():
        record = records.get(query_id, {})
        audits[query_id] = model_audit(record)
        for arm in ARMS:
            value, reason = measured(record.get("arms", {}).get(arm, {}))
            valid[query_id, arm] = value
            missing_reasons[query_id, arm] = reason
    common = {query_id for query_id, record in records.items()
              if record.get("status") == "ok" and not audits[query_id]["cap_violated"] and
                  not audits[query_id]["audit_total_mismatch"] and all(valid[query_id, arm] is not None and
                  record.get("arms", {}).get(arm, {}).get("status") == "ok" for arm in ARMS)}
    source = {"raw_path": str(raw), "raw_sha256": raw_hash,
              "manifest_path": str(manifest), "manifest_sha256": manifest_hash}
    metrics, query_details, call_details, audit_details = [], [], [], []
    for query_id, manifest_row in expected.items():
        record = records.get(query_id, {})
        location = locations.get(query_id, {"raw_line": None, "record_line_sha256": None})
        locator = str(raw) + ":" + str(location["raw_line"]) if location["raw_line"] else None
        audit = audits[query_id]
        audit_details.append({"query_id": query_id, "benchmark": manifest_row["benchmark"],
            "record_present": query_id in records, "query_status": record.get("status", "missing"),
            "total_calls": audit["actual_total_calls"] if query_id in records else None,
            "reported_total_calls": audit["reported_total_calls"], "audit_total_mismatch": audit["audit_total_mismatch"],
            **{arm + "_calls": audit["calls_by_arm"][arm] if query_id in records else None for arm in LIMITS},
            "failed_calls": audit["failed_calls"] if query_id in records else None,
            "blocked_calls": audit["blocked_calls"] if query_id in records else None,
            "cap_violated": audit["cap_violated"], "cap_violation_reasons": ";".join(audit["cap_violations"]),
            "raw_source": locator, **location, "raw_sha256": raw_hash})
        for arm in ARMS:
            detail = record.get("arms", {}).get(arm, {})
            value, original = valid[query_id, arm], valid[query_id, "original"]
            paired = value is not None and original is not None
            speedup = original["median_ms"] / value["median_ms"] if paired else None
            query_details.append({"query_id": query_id, "benchmark": manifest_row["benchmark"],
                "system": arm, "system_label": LABELS[arm], "record_present": query_id in records,
                "query_status": record.get("status", "missing"), "system_status": detail.get("status", "missing"),
                "measurement_status": detail.get("measurement", {}).get("status", "missing"),
                "valid_final_timing": value is not None, "missing_or_censored_reason": missing_reasons[query_id, arm],
                "mean_ms": value["mean_ms"] if value else None, "median_ms": value["median_ms"] if value else None,
                "shared_identical_sql_with": detail.get("measurement", {}).get("shared_identical_sql_with"),
                "final_equivalence_accepted": detail.get("verification", {}).get("accepted"),
                "equivalence_category": evidence_category(detail), "proof": detail.get("verification", {}).get("proof"),
                "formal": detail.get("verification", {}).get("formal"),
                "median_speedup_vs_original": speedup,
                "median_latency_reduction_vs_original": 1 - value["median_ms"] / original["median_ms"] if paired else None,
                "improves_by_at_least_10pct_latency_reduction": value["median_ms"] <= 0.90 * original["median_ms"] and detail.get("status") == "ok" if paired else None,
                "speedup_factor_ge_1_10": speedup >= 1.10 and detail.get("status") == "ok" if paired else None,
                "strict_regression_vs_original": value["median_ms"] > original["median_ms"] if paired else None,
                "fallback": detail.get("fallback", False) if detail else None,
                "model_calls": 0 if arm == "original" and query_id in records else audit["calls_by_arm"].get(arm) if query_id in records else None,
                "input_cap_violated": audit["cap_violated"], "common_complete_cohort": query_id in common,
                "raw_source": locator, "raw_json_pointer": "/arms/" + arm, **location,
                "raw_sha256": raw_hash, "query_sha256": record.get("query_sha256", manifest_row.get("sha256") or
                    sha256(manifest_row["sql"].encode()) if isinstance(manifest_row.get("sql"), str) else manifest_row.get("sha256")),
                "final_sql_sha256": detail.get("sql_sha256") or (sha256(detail["sql"].encode()) if isinstance(detail.get("sql"), str) else None), "schema_sha256": record.get("schema_sha256", manifest_row.get("schema_sha256")),
                "original_source_path": manifest_row.get("source_path"), "original_source_commit": manifest_row.get("source_commit")})
        for index, call in enumerate(audit["calls"]):
            provenance = call.get("provenance", {})
            session = provenance.get("session_transport", provenance.get("session_provenance", {}))
            call_details.append({"query_id": query_id, "benchmark": manifest_row["benchmark"], "arm": call.get("arm"),
                "input_call_index": call.get("input_call_index", index + 1), "arm_call_index": call.get("arm_call_index"),
                "task": call.get("task"), "status": call.get("status"), "retry": call.get("retry"),
                "payload_sha256": call.get("payload_sha256"), "actual_model": provenance.get("actual_model", session.get("actual_serving_model")),
                "reasoning_effort": session.get("reasoning_effort"), "request_id": session.get("request_id"),
                "request_sha256": session.get("request_sha256"), "worker_id": session.get("worker_id"),
                "session_id": session.get("session_id"), "session_run_id": session.get("run_id"),
                "raw_source": locator, "raw_json_pointer": "/model_call_audit/calls/" + str(call["_raw_call_index"]),
                "raw_sha256": raw_hash, **location})
    for benchmark in (*BENCHMARKS, "overall"):
        expected_ids = [query_id for query_id, row in expected.items() if benchmark == "overall" or row["benchmark"] == benchmark]
        if not expected_ids:
            continue
        for cohort in ("available", "common_complete"):
            for arm in ARMS:
                present = [query_id for query_id in expected_ids if arm in records.get(query_id, {}).get("arms", {})]
                measured_ids = [query_id for query_id in expected_ids if valid[query_id, arm] is not None]
                selected = measured_ids if cohort == "available" else [query_id for query_id in expected_ids if query_id in common]
                paired = [query_id for query_id in selected if valid[query_id, "original"] is not None]
                means = [valid[query_id, arm]["mean_ms"] for query_id in selected]
                medians = [valid[query_id, arm]["median_ms"] for query_id in selected]
                evidence = Counter(evidence_category(records.get(query_id, {}).get("arms", {}).get(arm, {})) for query_id in present)
                accepted_count = sum(count for category, count in evidence.items() if category not in {"missing", "rejected"})
                evaluated = len(present) - evidence["missing"]
                formal_count = sum(records[query_id]["arms"][arm].get("verification", {}).get("accepted") is True and
                                   records[query_id]["arms"][arm]["verification"].get("formal") is True for query_id in present)
                improved = sum(valid[query_id, arm]["median_ms"] <= 0.90 * valid[query_id, "original"]["median_ms"] and
                               records[query_id]["arms"][arm].get("status") == "ok" for query_id in paired)
                speedup_factor_count = sum(valid[query_id, "original"]["median_ms"] / valid[query_id, arm]["median_ms"] >= 1.10 and
                               records[query_id]["arms"][arm].get("status") == "ok" for query_id in paired)
                regressed = sum(valid[query_id, arm]["median_ms"] > valid[query_id, "original"]["median_ms"] for query_id in paired)
                fallback_count = sum(records.get(query_id, {}).get("arms", {}).get(arm, {}).get("fallback") is True for query_id in present)
                metrics.append({"benchmark": benchmark, "cohort": cohort, "system": arm, "system_label": LABELS[arm],
                    "expected_queries": len(expected_ids), "records_present": sum(query_id in records for query_id in expected_ids),
                    "system_records_present": len(present), "completed_valid_final_queries": len(measured_ids),
                    "completed_coverage": ratio(len(measured_ids), len(expected_ids)),
                    "error_free_system_queries": sum(records[query_id]["arms"][arm].get("status") == "ok" for query_id in measured_ids),
                    "common_complete_queries": sum(query_id in common for query_id in expected_ids),
                    "timing_cohort_queries": len(selected), "timing_cohort_coverage": ratio(len(selected), len(expected_ids)),
                    "arithmetic_average_query_mean_ms": statistics.mean(means) if means else None,
                    "median_query_median_ms": statistics.median(medians) if medians else None,
                    "p90_query_median_ms": p90(medians),
                    "final_equivalence_evaluated_count": evaluated, "final_accepted_equivalence_count": accepted_count,
                    "final_accepted_equivalence_rate_expected": ratio(accepted_count, len(expected_ids)),
                    "final_accepted_equivalence_rate_evaluated": ratio(accepted_count, evaluated),
                    "identity_or_conservative_ast_count": evidence["identity_or_conservative_ast"],
                    "formal_nonidentity_count": evidence["formal_nonidentity"],
                    "formal_accepted_total_including_identity_count": formal_count,
                    "formal_accepted_rate_expected": ratio(formal_count, len(expected_ids)),
                    "finite_differential_count": evidence["finite_differential"],
                    "identity_or_ast_rate_expected": ratio(evidence["identity_or_conservative_ast"], len(expected_ids)),
                    "formal_nonidentity_rate_expected": ratio(evidence["formal_nonidentity"], len(expected_ids)),
                    "finite_differential_rate_expected": ratio(evidence["finite_differential"], len(expected_ids)),
                    "other_nonformal_accepted_count": evidence["accepted_other_nonformal"],
                    "paired_original_queries": len(paired), "improved_at_least_10pct_latency_reduction_count": improved,
                    "improvement_rate_paired": ratio(improved, len(paired)), "improvement_rate_expected": ratio(improved, len(expected_ids)),
                    "speedup_factor_ge_1_10_count": speedup_factor_count,
                    "speedup_factor_ge_1_10_rate_paired": ratio(speedup_factor_count, len(paired)),
                    "strict_regression_count": regressed, "strict_regression_rate_paired": ratio(regressed, len(paired)),
                    "strict_regression_rate_expected": ratio(regressed, len(expected_ids)),
                    "fallback_count_all_present": fallback_count,
                    "fallback_count_timing_cohort": sum(records[query_id]["arms"][arm].get("fallback") is True for query_id in selected),
                    "model_calls_all_present": sum(audits[query_id]["calls_by_arm"].get(arm, 0) for query_id in expected_ids),
                    "input_cap_violation_count_all_present": sum(audits[query_id]["cap_violated"] for query_id in expected_ids),
                    "missing_timing_reason_counts": dict(Counter(missing_reasons[query_id, arm] for query_id in expected_ids if valid[query_id, arm] is None)),
                    "timing_cohort_query_ids": selected, "paired_original_query_ids": paired,
                    "raw_path": str(raw), "raw_sha256": raw_hash})
    benchmark_counts = Counter(row["benchmark"] for row in expected.values())
    return {"format_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
        "scope": {"expected_queries": len(expected), "raw_records_present": len(records),
                  "expected_by_benchmark": dict(benchmark_counts), "common_complete_queries": len(common),
                  "common_complete_query_ids": [query_id for query_id in expected if query_id in common],
                  "matches_stage_a_30_query_contract": len(expected) == 30 and benchmark_counts == Counter({name: 10 for name in BENCHMARKS}),
                  "selected_nonrandom_cohort": True, "full_suite_performance_estimate": False,
                  "no_additional_benchmark_or_model_calls": True, "native_baseline_reproduction": False},
        "definitions": {
            "available": "Per-system accepted final equivalence plus exactly three successful final execution samples. Generation errors remain visible and cannot count as improvements.",
            "common_complete": "Same query IDs for all four systems; query and every system status ok, final equivalence accepted, exactly three successful final samples for every system, and no model-call cap violation or counter mismatch.",
            "coverage": "Completed valid final queries divided by the manifest's expected query count; pending, failed, censored, rejected or invalid timings stay missing.",
            "mean": "Arithmetic average of per-query arithmetic means recomputed from the three final samples; conditional on timing_cohort_queries.",
            "median_and_p90": "Median and linearly interpolated p90 at (n-1)*0.9 of per-query medians recomputed from three final samples; conditional on timing_cohort_queries.",
            "equivalence": "Accepted/evaluated and accepted/expected are both shown. Counts describe all present final arm evidence and do not change between timing cohort rows. Identity/AST, formal nonidentity, and finite differential evidence are separate. Finite differential acceptance is not a universal proof.",
            "improvement": "Primary Improvement Rate requires system median <= 0.90 * Original median, equivalently latency reduction >= 10%, with valid paired timings and system status ok. The paired and expected denominators are both reported.",
            "additional_speedup_factor": "Separate supplemental criterion: Original median / system median >= 1.10. This implies approximately 9.09% latency reduction and is not the primary Improvement Rate.",
            "strict_regression": "System median strictly greater than Original median, without a noise tolerance; includes every valid timing pair, including generation-error fallbacks. Paired and expected denominators are both reported.",
            "fallback": "Explicit raw arm fallback flag. Both all-present and selected timing-cohort counts are shown; Original has no fallback flag.",
            "model_calls": "Count actual attempted transport invocations in the per-input ledger, including failed requests. Blocked calls do not consume an invocation. Limits: R-Bot 2, QUITE 3, MCTS 3, total 8 per input. Both actual and reported counters are checked.",
            "comparison_caution": "Available-system timing cohorts can differ. Use the common_complete rows for comparisons with identical error-free query coverage. Censored Original timing yields no speedup or regression pair.",
            "source_links": "raw_source is an absolute path with a one-based raw JSONL line. raw_json_pointer locates arm/call evidence; raw_sha256 hashes the complete input file and record_line_sha256 hashes the exact line excluding its line ending."
        }, "metrics": metrics, "query_system_details": query_details, "query_model_audits": audit_details, "model_call_details": call_details}


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows, empty_fields):
    fields = list(rows[0]) if rows else empty_fields
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    os.replace(temporary, path)


def write_report(output, report):
    output = Path(output).resolve()
    if output == Path(report["source"]["raw_path"]).parent:
        raise ValueError("Use a separate report subdirectory; active run files must remain untouched")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", report)
    write_csv(output / "system_benchmark_metrics.csv", report["metrics"], ["benchmark", "cohort", "system"])
    write_csv(output / "query_system_details.csv", report["query_system_details"], ["query_id", "system"])
    write_csv(output / "query_model_audits.csv", report["query_model_audits"], ["query_id", "total_calls"])
    write_csv(output / "model_call_details.csv", report["model_call_details"], ["query_id", "arm", "status"])
    atomic_json(output / "source_and_definitions.json", {key: report[key] for key in ("format_version", "generated_at", "source", "scope", "definitions")})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/stage_a_sf10/raw.jsonl")
    parser.add_argument("--manifest", type=Path, help="Defaults to the raw run's inputs/manifest.jsonl")
    parser.add_argument("--output", type=Path, help="Defaults to the raw run's report/ subdirectory")
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.raw.parent / "inputs/manifest.jsonl"
    args.output = args.output or args.raw.parent / "report"
    return args


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.raw, args.manifest)
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "expected_queries": report["scope"]["expected_queries"],
                      "records_present": report["scope"]["raw_records_present"],
                      "common_complete_queries": report["scope"]["common_complete_queries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
