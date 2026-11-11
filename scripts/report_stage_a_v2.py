#!/usr/bin/env python3
"""Report a fresh Stage A v2 study without executing SQL or model requests.

Both comparative summaries use one common four-system paired cohort. The strict
cohort contains only successful finite final timings. The T_max cohort also
admits genuine final timing timeouts at the configured SQL timeout. Generation
errors are diagnostic metadata once a valid measurement exists: failed or
rejected rewrites fall back to Original and remain in the cohort. Raw,
uncapped timing observations are retained separately as evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import report_stage_a as legacy

ARMS = legacy.ARMS
LABELS = {"original": "Original", "rbot": "RBot", "quite": "QUITE", "mcts": "MCTS"}
BENCHMARKS = legacy.BENCHMARKS
finite_positive = legacy.finite_positive
p90 = legacy.p90


def _generation_failed(record, name, arm):
    measurement = arm.get("measurement", {})
    # Standard fallback accounting: a valid final measurement (including a
    # genuine SQL timeout represented by T_max) must not be invalidated by a
    # transient model error recorded alongside it.
    if measurement.get("status") in {"ok", "timeout"}:
        return False
    return bool(arm.get("generation_error") or arm.get("generation_errors")
                or arm.get("terminal_generation_failures")
                or arm.get("status") == "generation_error"
                or any(call.get("arm") == name and call.get("status") != "ok"
                       for call in record.get("model_call_audit", {}).get("calls", [])
                       if isinstance(call, dict) and call.get("transport_invoked") is not False))


def timing_value(record, name, timeout_ms, *, capped=False):
    """Return comparable statistics, or a precise exclusion reason.

    Generation diagnostics are ignored after a valid final timing is present;
    invalid or missing measurements still receive precise exclusion reasons.
    """
    arm = record.get("arms", {}).get(name, {})
    measurement = arm.get("measurement", {})
    # A completed timing is the primary experimental observation. Transport
    # diagnostics, retries, and a failed sub-call are retained in the record
    # but never discard a valid measurement. This also admits a standard
    # Original fallback, whose final timing is explicitly measured/shared.
    measured_status = measurement.get("status")
    if measured_status not in {"ok", "timeout"}:
        if _generation_failed(record, name, arm):
            return None, "generation_error"
        return None, "non_timing_error:" + str(arm.get("status", "missing"))
    evidence = arm.get("verification", {})
    fallback = bool(arm.get("fallback") or arm.get("sql") == record.get("original_sql"))
    # Failed/rejected rewrite generation is represented by the measured
    # Original fallback. A changed SQL still needs accepted equivalence.
    if not fallback and evidence.get("accepted") is not True:
        return None, "final_verification_not_accepted"
    if (not fallback and record.get("verification_mode") == "strict"
            and evidence.get("formal") is not True):
        return None, "strict_final_verification_not_formal"
    if measurement.get("runs") != 3 or measurement.get("warmups") != 1:
        return None, "invalid_final_timing_protocol"
    if measurement.get("scope") != "final_evaluation":
        return None, "final_timing_scope_unproven"
    if measurement.get("status") == "timeout":
        if capped:
            return {"mean_ms": timeout_ms, "median_ms": timeout_ms,
                    "timing_timeout": True, "clipped_samples": 0}, None
        return None, "timing_timeout"
    if measurement.get("status") != "ok":
        return None, "non_timing_error:" + str(arm.get("status", "missing"))
    samples = measurement.get("samples_ms", [])
    if not isinstance(samples, list) or len(samples) != 3 or not all(finite_positive(v) for v in samples):
        return None, "invalid_three_finite_final_samples"
    values = [min(value, timeout_ms) for value in samples] if capped else samples
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "timing_timeout": False,
            "clipped_samples": sum(value > timeout_ms for value in samples) if capped else 0}, None


def _stats(values):
    means = [value["mean_ms"] for value in values]
    medians = [value["median_ms"] for value in values]
    return {"queries": len(values),
            "mean_ms": statistics.mean(means) if means else None,
            "median_ms": statistics.median(medians) if medians else None,
            "p90_ms": p90(medians), "p90_query_mean_ms": p90(means)}


def _reduction(value, baseline):
    return 1 - value / baseline if finite_positive(value) and finite_positive(baseline) else None


def _at_least_ten_percent(value, baseline):
    # isclose includes the exact 10% boundary despite binary float arithmetic.
    return value < baseline and (value < .9 * baseline or math.isclose(value, .9 * baseline, rel_tol=1e-12))


def cohort_summary(expected, records, timeout_ms, *, capped):
    values, excluded, common = {}, {}, []
    audits = {query_id: legacy.model_audit(records.get(query_id, {})) for query_id in expected}
    for query_id in expected:
        record = records.get(query_id, {})
        reasons = {}
        for arm in ARMS:
            value, reason = timing_value(record, arm, timeout_ms, capped=capped)
            values[query_id, arm] = value
            if reason:
                reasons[arm] = reason
        audit = audits[query_id]
        # Failed/retried model calls are diagnostic metadata. They do not
        # invalidate a measured arm or a completed paired query.
        if not record:
            reasons["query"] = "missing_record"
        elif record.get("status") not in ({"ok", "original_timeout", "partial_error"} if capped else {"ok", "partial_error"}):
            reasons["query"] = "query_status:" + str(record.get("status", "missing"))
        if reasons:
            excluded[query_id] = reasons
        else:
            common.append(query_id)
    systems = {}
    for arm in ARMS:
        selected = [values[query_id, arm] for query_id in common]
        improvements, regressions = [], []
        for query_id in common:
            current = values[query_id, arm]["median_ms"]
            original = values[query_id, "original"]["median_ms"]
            if _at_least_ten_percent(current, original):
                improvements.append(query_id)
            if current > original:
                regressions.append(query_id)
        systems[arm] = {"label": LABELS[arm], **_stats(selected),
                        "timing_timeout_count": sum(value["timing_timeout"] for value in selected),
                        "clipped_sample_count": sum(value["clipped_samples"] for value in selected),
                        "improvement_ge_10pct_vs_original_count": len(improvements),
                        "improvement_ge_10pct_vs_original_ids": improvements,
                        "regression_vs_original_count": len(regressions),
                        "regression_vs_original_ids": regressions}
    comparisons = {}
    for baseline in ("original", "rbot", "quite"):
        improvements, regressions, faster, equal = [], [], [], []
        for query_id in common:
            current = values[query_id, "mcts"]["median_ms"]
            base = values[query_id, baseline]["median_ms"]
            if _at_least_ten_percent(current, base):
                improvements.append(query_id)
            if current > base:
                regressions.append(query_id)
            elif current < base:
                faster.append(query_id)
            else:
                equal.append(query_id)
        mcts, base = systems["mcts"], systems[baseline]
        comparisons[baseline] = {"paired_queries": len(common), "paired_query_ids": list(common),
            "mean_latency_reduction": _reduction(mcts["mean_ms"], base["mean_ms"]),
            "median_latency_reduction": _reduction(mcts["median_ms"], base["median_ms"]),
            "p90_latency_reduction": _reduction(mcts["p90_ms"], base["p90_ms"]),
            "improvement_ge_10pct_count": len(improvements), "improvement_ge_10pct_ids": improvements,
            "regression_count": len(regressions), "regression_ids": regressions,
            "faster_count": len(faster), "equal_count": len(equal)}
    return {"expected_queries": len(expected), "paired_queries": len(common), "paired_query_ids": common,
            "excluded_queries": excluded, "systems": systems, "mcts_vs_baselines": comparisons,
            "coverage_complete": len(common) == len(expected)}


def build_report(raw, manifest, config=None):
    raw, manifest = Path(raw).resolve(), Path(manifest).resolve()
    config = Path(config or raw.parent / "inputs/config.public.json").resolve()
    configuration = json.loads(config.read_text())
    timeout_ms = configuration.get("timeout_ms")
    if not finite_positive(timeout_ms):
        raise ValueError("A finite positive configured SQL timeout_ms is required")
    expected, records, locations, raw_hash, manifest_hash = legacy.load_inputs(raw, manifest)
    if len(expected) != 30 or Counter(row["benchmark"] for row in expected.values()) != Counter({b: 10 for b in BENCHMARKS}):
        raise ValueError("Stage A v2 reporting requires the exact 30-query, ten-per-benchmark manifest")
    for query_id, record in records.items():
        variant = record.get("study_variant", {})
        allowed_scope = {"transductive_stage_a_followup", "paper_backfill"}
        if (variant.get("version") != 2
                or variant.get("feedback_scope") not in allowed_scope
                or variant.get("historical_stage_a_records_modified") is not False):
            raise ValueError("Only labeled Stage A v2 or paper-backfill records are accepted: " + query_id)
    details, raw_metrics = [], {}
    for query_id, row in expected.items():
        record = records.get(query_id, {})
        for arm in ARMS:
            detail = record.get("arms", {}).get(arm, {})
            measurement = detail.get("measurement", {})
            samples = measurement.get("samples_ms", [])
            raw_value, raw_reason = legacy.measured(detail)
            strict, strict_reason = timing_value(record, arm, timeout_ms)
            capped, capped_reason = timing_value(record, arm, timeout_ms, capped=True)
            details.append({"query_id": query_id, "benchmark": row["benchmark"], "system": arm,
                "arm_status": detail.get("status", "missing"),
                "diagnostic_generation_status": detail.get("diagnostic_generation_status"),
                "measurement_status": measurement.get("status", "missing"),
                "uncapped_raw_samples_ms": samples, "uncapped_raw_statistics": raw_value,
                "uncapped_raw_exclusion": raw_reason, "strict_value": strict, "strict_exclusion": strict_reason,
                "timeout_capped_value": capped, "timeout_capped_exclusion": capped_reason,
                "source": locations.get(query_id)})
    for arm in ARMS:
        selected = [detail["uncapped_raw_statistics"] for detail in details
                    if detail["system"] == arm and detail["uncapped_raw_statistics"] is not None]
        raw_metrics[arm] = _stats(selected)
    summary = {"format_version": "stage-a-v2-report-1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_records" if len(records) == len(expected) else "incomplete_records",
        "scope": {"expected_queries": len(expected), "raw_records_present": len(records),
                  "missing_query_ids": [query_id for query_id in expected if query_id not in records],
                  "fixed_nonrandom_cohort": True, "historical_records_included": False},
        "source": {"raw_path": str(raw), "raw_sha256": raw_hash, "manifest_path": str(manifest),
                   "manifest_sha256": manifest_hash, "config_path": str(config),
                   "config_sha256": legacy.sha256(config.read_bytes())},
        "definitions": {"timeout_ms": timeout_ms,
            "strict_finite_only": "One common four-system cohort with valid accepted final measurements and three finite positive final samples after one warmup. Generation diagnostics do not invalidate a measured arm; failed rewrites are represented by an Original fallback.",
            "timeout_capped_T_max": "One common four-system cohort using min(sample,T_max) for successful final samples, and T_max for genuine final SQL timeouts. A valid measured Original fallback remains included; missing, skipped, setup and other measurement errors remain excluded.",
            "paired_comparisons": "Each summary and each benchmark uses the same four-system common query IDs for all system statistics and MCTS comparisons. Cohorts may differ between strict and capped summaries; their IDs and exclusions are explicit.",
            "mean": "Arithmetic mean of per-query means of three final samples.",
            "median": "Median of per-query medians of three final samples.",
            "p90": "Linearly interpolated 90th percentile at (n-1)*0.9 of per-query medians.",
            "improvement_ge_10pct": "Per-query median latency at most 0.9 times the comparator median (latency reduction >=10%).",
            "regression": "Per-query median latency strictly greater than comparator median.",
            "censoring": "Timeout-capped values are censored policy values, not measured latencies; ratios involving them describe the T_max policy.",
            "uncapped_raw_metrics": "Descriptive available raw final timings, including timings retained on generation-error arms; independent per-system denominators. Never used for paired comparisons."},
        "strict_finite_only": cohort_summary(expected, records, timeout_ms, capped=False),
        "timeout_capped_T_max": cohort_summary(expected, records, timeout_ms, capped=True),
        "uncapped_raw_metrics": raw_metrics, "query_system_details": details,
        "model_call_audits": {query_id: legacy.model_audit(records[query_id]) for query_id in records}}
    summary["per_benchmark"] = {}
    for benchmark in BENCHMARKS:
        selected = {query_id: row for query_id, row in expected.items() if row["benchmark"] == benchmark}
        summary["per_benchmark"][benchmark] = {
            "strict_finite_only": cohort_summary(selected, records, timeout_ms, capped=False),
            "timeout_capped_T_max": cohort_summary(selected, records, timeout_ms, capped=True)}
    return summary


def _show(value):
    return "n/a" if value is None else f"{value:.3f}"


def _json_safe(value):
    """Retain invalid raw evidence without emitting nonstandard JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_raw_value": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def markdown_report(report):
    lines = [f"# Stage A v2 report — {report['generated_at'][:10]}", "",
        f"Status: **{report['status']}**. Present records: {report['scope']['raw_records_present']}/30.", "",
        "This fixed cohort is a transductive follow-up. The report does not estimate full-suite performance.", ""]
    for key, title in (("strict_finite_only", "Strict finite-only"), ("timeout_capped_T_max", "Timeout-capped T_max")):
        cohort = report[key]
        lines += [f"## {title}", "", f"Common paired cohort: **{cohort['paired_queries']}/30** queries.", ""]
        for arm in ARMS:
            value = cohort["systems"][arm]
            lines.append(f"- {LABELS[arm]}: mean {_show(value['mean_ms'])} ms; median {_show(value['median_ms'])} ms; p90 {_show(value['p90_ms'])} ms; improvements ≥10% vs Original {value['improvement_ge_10pct_vs_original_count']}; regressions {value['regression_vs_original_count']}; timing timeouts {value['timing_timeout_count']}.")
        lines.append("")
        for baseline, value in cohort["mcts_vs_baselines"].items():
            lines.append(f"- MCTS vs {LABELS[baseline]}: mean/median/p90 latency reductions " +
                         "/".join("n/a" if value[name] is None else f"{100 * value[name]:.2f}%" for name in
                                  ("mean_latency_reduction", "median_latency_reduction", "p90_latency_reduction")) +
                         f"; improvements ≥10% {value['improvement_ge_10pct_count']}; regressions {value['regression_count']}.")
        lines.append("")
    lines += ["## Interpretation and evidence", "",
        f"T_max is the configured SQL timeout: {report['definitions']['timeout_ms']} ms. Capped values are censored policy values, not measured latencies. Generation errors remain diagnostic; a failed rewrite is counted as an Original fallback when its final measurement is valid.", "",
        "All systems and comparisons within each summary use exactly the same four-system paired cohort. Strict and capped cohorts may differ. Exact IDs, exclusions, per-benchmark summaries and source hashes are in summary.json. Uncapped available raw metrics are separate descriptive evidence with independent denominators.", "",
        f"Raw evidence: `{report['source']['raw_path']}`. Manifest: `{report['source']['manifest_path']}`.", ""]
    return "\n".join(lines)


def write_report(output, report):
    output = Path(output).resolve()
    if output == Path(report["source"]["raw_path"]).parent:
        raise ValueError("Use a separate report subdirectory; raw study records are immutable")
    output.mkdir(parents=True, exist_ok=True)
    legacy.atomic_json(output / "summary.json", _json_safe(report))
    legacy.write_csv(output / "query_system_details.csv", _json_safe(report["query_system_details"]), ["query_id", "system"])
    legacy.atomic_json(output / "uncapped_raw_metrics.json", report["uncapped_raw_metrics"])
    (output / "REPORT.md").write_text(markdown_report(report))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.raw.parent / "inputs/manifest.jsonl"
    args.config = args.config or args.raw.parent / "inputs/config.public.json"
    args.output = args.output or args.raw.parent / "report"
    return args


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.raw, args.manifest, args.config)
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
                      "strict_paired_queries": report["strict_finite_only"]["paired_queries"],
                      "timeout_capped_paired_queries": report["timeout_capped_T_max"]["paired_queries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
