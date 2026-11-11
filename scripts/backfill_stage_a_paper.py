#!/usr/bin/env python3
"""Merge the completed Stage A runs into a paper-production input.

The merge is deterministic: newer v2 runs win when they have the same valid
measurement coverage, then older completed runs fill missing queries. It also
builds a FileModel-compatible candidate cache from every accepted rewrite plus
the existing model-candidate artifact, without executing SQL or calling a
model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ARMS = ("original", "rbot", "quite", "mcts")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid_arm(arm):
    measurement = arm.get("measurement", {})
    return (measurement.get("status") in {"ok", "timeout"}
            and arm.get("verification", {}).get("accepted") is True
            and (measurement.get("status") == "timeout"
                 or isinstance(measurement.get("samples_ms"), list)
                 and len(measurement["samples_ms"]) == 3))


def quality(record):
    return sum(valid_arm(record.get("arms", {}).get(arm, {})) for arm in ARMS)


def measured_sum(record):
    """Tie-break only among runs with the same arm coverage.

    This keeps a complete, reproducible record while preferring the cleaner
    lower-latency SF10 observation when several immutable runs cover the same
    query. The selected source and all alternatives remain in provenance.
    """
    values = []
    for arm in ARMS:
        measurement = record.get("arms", {}).get(arm, {}).get("measurement", {})
        value = measurement.get("median_ms")
        if isinstance(value, (int, float)) and value > 0:
            values.append(float(value))
        elif measurement.get("status") == "timeout":
            values.append(300000.0)
    return sum(values) if len(values) == len(ARMS) else float("inf")


def candidate_cache(source_rows, artifact, output):
    grouped = {}
    def add(original, candidate, *, task, provenance):
        if not isinstance(original, str) or not original.strip() or not isinstance(candidate, str) or not candidate.strip() or candidate.strip() == original.strip():
            return
        grouped.setdefault(original, {})[candidate] = {"sql": candidate, "action": task,
            "provenance": provenance, "verified": True}
    for source, records in source_rows.items():
        for record in records:
            original = record.get("original_sql")
            for arm in ("rbot", "quite", "mcts"):
                detail = record.get("arms", {}).get(arm, {})
                if valid_arm(detail):
                    add(original, detail.get("sql"), task=arm, provenance={"source_run": source, "query_id": record.get("query_id"), "verification": detail.get("verification", {})})
    if artifact and Path(artifact).exists():
        for line in Path(artifact).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            original = row.get("original_sql")
            for candidate in row.get("candidates", []):
                if isinstance(candidate, dict):
                    add(original, candidate.get("sql"), task=candidate.get("action", "cached_candidate"), provenance={"source_artifact": str(artifact), **row.get("provenance", {})})
    with Path(output).open("w") as stream:
        for original in sorted(grouped):
            stream.write(json.dumps({"original_sql": original, "task": None,
                "candidates": list(grouped[original].values()),
                "provenance": {"cache": "stage_a_dual_action_backfill", "candidate_count": len(grouped[original])}}, sort_keys=True) + "\n")


def merge(inputs, output, manifest, cache, artifact):
    source_rows = {str(path): rows(path) for path in inputs}
    by_id = {}
    for source_index, (source, records) in enumerate(source_rows.items()):
        for record in records:
            qid = record.get("query_id")
            if not qid:
                continue
            # Prefer the latest completed follow-up at equal arm coverage.
            # This keeps the explicitly audited r3 observations (including
            # Query 1's verified MCTS rewrite) instead of silently selecting a
            # lower-latency cache-warm observation from an older run. Source
            # order is deterministic: local resume < r2 < r3.
            key = (quality(record), int(record.get("status") in {"ok", "original_timeout", "partial_error"}), source_index)
            previous = by_id.get(qid)
            if previous is None or key >= previous[0]:
                by_id[qid] = (key, record, source)
    chosen = []
    for qid in sorted(by_id, key=lambda value: (value.split(":", 2)[0], value)):
        key, record, source = by_id[qid]
        record = json.loads(json.dumps(record))
        # Normalize legacy v2 labels to the paper-production invariant: once a
        # final measurement exists, its arm is measured regardless of a
        # diagnostic model timeout. Preserve the historical label separately
        # so the transport event remains auditable.
        for arm_name, arm_detail in record.get("arms", {}).items():
            measurement = arm_detail.get("measurement", {}) if isinstance(arm_detail, dict) else {}
            measured_status = measurement.get("status")
            if measured_status in {"ok", "timeout"} and arm_detail.get("status") != measured_status:
                arm_detail["diagnostic_generation_status"] = arm_detail.get("status")
                arm_detail["status"] = measured_status
        record["study_variant"] = {"version": 2, "feedback_scope": "paper_backfill",
            "historical_stage_a_records_modified": False, "source_run": source,
            "selection_policy": "max_valid_arm_coverage_then_latest_source_run",
            "source_runs": [str(path) for path in inputs]}
        chosen.append(record)
    if len(chosen) != 30:
        raise SystemExit(f"expected 30 merged queries, got {len(chosen)}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as stream:
        for record in chosen:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    candidate_cache(source_rows, artifact, cache)
    Path(manifest).write_text(Path(inputs[0]).parent.joinpath("inputs/manifest.jsonl").read_text())
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--artifact")
    args = parser.parse_args()
    chosen = merge(args.input, args.output, args.manifest, args.cache, args.artifact)
    print(json.dumps({"records": len(chosen), "output": args.output, "candidate_cache": args.cache,
                      "fallback_arms": sum(detail.get("fallback") is True for record in chosen for detail in record.get("arms", {}).values())}))


if __name__ == "__main__":
    main()
