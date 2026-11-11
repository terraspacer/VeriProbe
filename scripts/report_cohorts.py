#!/usr/bin/env python3
"""Report supplemental workload cohorts without changing an existing run.

Example, after the source run stops writing its evidence::

    .venv/bin/python scripts/report_cohorts.py --run results/completed/hybrid \
        --output results/completed_cohorts

The source run's acceptance remains authoritative. These partial supplemental
views describe query/dataset membership, not reproduction of upstream optimizer
generation or the original full-task acceptance scope. No databases are used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcts_rewrite.reporting import build_report, write_reports


_CLASSIFICATIONS = {
    "native_released_evaluation_cohort": ("rbot", True),
    "expanded_rbot_sf50_seed1": ("rbot", False),
    "requested_quite_cross_scale_native_binding_unestablished": ("quite", None),
}
_REGISTERED_DATASETS = {
    "tpch10": ("tpch", 10, None), "tpch50": ("tpch", 50, None),
    "dsb10": ("dsb", 10, None), "dsb50": ("dsb", 50, None),
    "calcite10": ("calcite", 40000000, "uniform"),
    "calcite10zipf": ("calcite", 40000000, "zipf"),
}
_FIELDS = {
    "query_id", "source", "benchmark", "dataset", "query_sha256", "classification",
    "native_released_dataset_membership", "source_path", "source_index",
    "source_query_id", "source_commit", "scale", "distribution", "reason",
    "included_in_current_requested_cohort", "released_query_inventory_member",
    "released_rbot_generation_dataset",
}
_REQUIRED = {
    "query_id", "source", "benchmark", "dataset", "query_sha256", "classification",
    "native_released_dataset_membership",
}
_VIEWS = {
    "native_rbot": {
        "label": "RBot released query/dataset cohort",
        "classification": "native_released_evaluation_cohort",
        "native_dataset_binding_known": True,
        "scope": "Released RBot query/dataset memberships only; source-run optimizer generation and evidence remain unchanged. This does not reproduce its cross-scale rewrite-log reuse protocol.",
    },
    "expanded_rbot_sf50": {
        "label": "Additional RBot SF50 seed-1 queries",
        "classification": "expanded_rbot_sf50_seed1",
        "native_dataset_binding_known": True,
        "scope": "RBot SF50 memberships added by the requested cross-scale evaluation and absent from released native SF50 analysis.",
    },
    "quite_requested_grid": {
        "label": "QUITE query inventory on requested local sizes/distributions",
        "classification": "requested_quite_cross_scale_native_binding_unestablished",
        "native_dataset_binding_known": False,
        "scope": "Released QUITE query inventory on source-run requested datasets; binding to the upstream published dataset scale/distribution is unknown.",
    },
}


def digest(value):
    payload = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(data, location):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number {value}")
    try:
        return json.loads(data, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{location}: invalid or truncated JSON: {error}") from error


def _jsonl(data, location):
    rows = []
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        value = _json(line, f"{location}:{number}")
        if not isinstance(value, dict):
            raise ValueError(f"{location}:{number}: expected an object")
        rows.append((value, line))
    return rows


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _group_key(value):
    return tuple(value.get(key) for key in ("source", "benchmark", "dataset", "ablation", "budget"))


def _classify(rows, fingerprints):
    memberships, identities, datasets = {}, {}, {}
    for row, _ in rows:
        unknown, missing = set(row) - _FIELDS, _REQUIRED - set(row)
        if unknown or missing:
            raise ValueError(f"Classification has unknown keys {sorted(unknown)} or missing keys {sorted(missing)}")
        query_id, dataset = row["query_id"], row["dataset"]
        if not all(isinstance(row[key], str) and row[key] for key in
                   ("query_id", "dataset", "source", "benchmark", "classification")):
            raise ValueError("Classification identity fields must be nonempty strings")
        if row["classification"] not in _CLASSIFICATIONS:
            raise ValueError(f"Unknown classification: {row['classification']}")
        if dataset not in _REGISTERED_DATASETS:
            raise ValueError(f"Classification has unknown dataset key: {dataset}")
        family, scale, distribution = _REGISTERED_DATASETS[dataset]
        if row["benchmark"] != family:
            raise ValueError(f"Classification dataset disagrees with benchmark family: {dataset}")
        if "scale" in row and (type(row["scale"]) is not int or row["scale"] != scale):
            raise ValueError(f"Dataset scale disagrees with registered classification: {dataset}")
        if "distribution" in row and row["distribution"] != distribution:
            raise ValueError(f"Dataset distribution disagrees with registered classification: {dataset}")
        source, native = _CLASSIFICATIONS[row["classification"]]
        if row["source"] != source or row["native_released_dataset_membership"] is not native:
            raise ValueError(f"Contradictory source/native classification: {query_id}, {dataset}")
        if (row["benchmark"] not in {"tpch", "dsb", "calcite"}
                or not query_id.startswith(f"{source}:{row['benchmark']}:")):
            raise ValueError(f"Classification query ID disagrees with its source/benchmark: {query_id}")
        if row["classification"] == "expanded_rbot_sf50_seed1" and dataset not in {"tpch50", "dsb50"}:
            raise ValueError("Expanded RBot SF50 classification requires tpch50 or dsb50")
        if query_id not in fingerprints:
            raise ValueError(f"Classification has unknown query key: {query_id}")
        if not _sha256(row["query_sha256"]) or row["query_sha256"] != fingerprints[query_id].get("query"):
            raise ValueError(f"Classification SQL hash mismatch: {query_id}")
        key = (query_id, dataset)
        if key in memberships:
            raise ValueError(f"Duplicate classification membership: {key}")
        identity = (source, row["benchmark"], row["query_sha256"])
        if query_id in identities and identities[query_id] != identity:
            raise ValueError(f"Inconsistent classification query identity: {query_id}")
        if dataset in datasets and datasets[dataset] != row["benchmark"]:
            raise ValueError(f"Classification dataset has conflicting benchmark families: {dataset}")
        identities[query_id], datasets[dataset], memberships[key] = identity, row["benchmark"], row
    if not memberships:
        raise ValueError("Classification is empty")
    return memberships, identities, datasets


def _validate(metadata, raw_rows, classification_rows):
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, dict) or digest(fingerprint) != metadata.get("run_hash"):
        raise ValueError("Source run fingerprint does not match run_hash")
    fingerprints, selected = fingerprint.get("source_fingerprints"), fingerprint.get("selected_query_ids")
    if not isinstance(fingerprints, dict) or any(not isinstance(v, dict) for v in fingerprints.values()):
        raise ValueError("Source run lacks source query fingerprints")
    if (not isinstance(selected, list) or not selected or any(not isinstance(q, str) for q in selected)
            or len(set(selected)) != len(selected) or set(selected) - set(fingerprints)):
        raise ValueError("Source run requires distinct selected_query_ids registered in source_fingerprints")
    if metadata.get("selected_queries") != len(selected):
        raise ValueError("Source selected_queries disagrees with selected_query_ids")
    memberships, identities, datasets = _classify(classification_rows, fingerprints)
    missing = set(selected) - set(identities)
    if missing:
        raise ValueError(f"Selected query IDs lack classifications: {sorted(missing)}")
    groups, planned_members = {}, {}
    for group in metadata.get("planned_groups", []):
        if not isinstance(group, dict):
            raise ValueError("Source planned_groups must contain objects")
        key = _group_key(group)
        source, benchmark, dataset, ablation, budget = key
        if (not all(isinstance(x, str) and x for x in key[:-1]) or type(budget) is not int
                or budget not in fingerprint.get("budgets", []) or ablation not in fingerprint.get("ablations", [])):
            raise ValueError(f"Invalid source planned group: {key}")
        if dataset not in datasets:
            raise ValueError(f"Source dataset {dataset!r} is not registered in the classification; no scale/distribution mapping is inferred")
        if datasets[dataset] != benchmark or key in groups:
            raise ValueError(f"Conflicting or duplicate source planned group: {key}")
        expected = {q for q in selected if identities[q][:2] == (source, benchmark)}
        if type(group.get("expected_queries")) is not int or len(expected) != group["expected_queries"]:
            raise ValueError(f"Source planned denominator disagrees with selected memberships: {key}")
        for query_id in expected:
            if (query_id, dataset) not in memberships:
                raise ValueError(f"Missing classification for selected query/dataset: {query_id}, {dataset}")
            item = memberships[query_id, dataset]
            if item.get("scale") is not None and group.get("scale") != item["scale"]:
                raise ValueError(f"Dataset scale disagrees with classification: {dataset}")
            distribution = metadata.get("dataset_provenance", {}).get(dataset, {}).get("distribution")
            if item.get("distribution") is not None and distribution != item["distribution"]:
                raise ValueError(f"Dataset distribution disagrees with classification: {dataset}")
        groups[key], planned_members[key] = group, expected
    if not groups or sum(g["expected_queries"] for g in groups.values()) != metadata.get("planned_tasks"):
        raise ValueError("Source planned_tasks disagrees with planned group denominators")
    covered = set().union(*planned_members.values())
    if covered != set(selected):
        raise ValueError("Some selected queries have no source planned group")
    seen_tasks, seen_memberships = set(), set()
    for record, _ in raw_rows:
        if record.get("record_type") != "query_result" or record.get("run_hash") != metadata["run_hash"]:
            raise ValueError("Raw record is not a query_result from the source run_hash")
        key, query_id = _group_key(record), record.get("query_id")
        if key not in groups or query_id not in planned_members[key]:
            raise ValueError(f"Unknown/unplanned raw membership: {query_id}, {key}")
        item = memberships[query_id, record["dataset"]]
        original = record.get("original_sql")
        if (not isinstance(original, str) or digest(original.encode()) != item["query_sha256"]
                or record.get("query_sha256") != item["query_sha256"]):
            raise ValueError(f"Raw SQL hash mismatch: {query_id}")
        provenance = record.get("manifest_provenance", {})
        if provenance.get("sha256", item["query_sha256"]) != item["query_sha256"]:
            raise ValueError(f"Raw manifest SQL hash mismatch: {query_id}")
        if (item.get("source_commit") is not None and provenance.get("source_commit") != item["source_commit"]):
            raise ValueError(f"Raw source commit mismatch: {query_id}")
        task, membership = record.get("task_id"), (query_id, *key)
        if not isinstance(task, str) or not task or task in seen_tasks or membership in seen_memberships:
            raise ValueError(f"Duplicate/invalid raw task or membership: {query_id}, {key}")
        seen_tasks.add(task)
        seen_memberships.add(membership)
    return memberships, groups, planned_members


def _view_group(view, group):
    if view == "quite_requested_grid":
        return group["source"] == "quite"
    return (group["source"] == "rbot" and (view == "native_rbot"
            or group["dataset"] in {"tpch50", "dsb50"}))


def report_cohorts(run_dir, output_dir, classification_path=None):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    classification_path = Path(classification_path or ROOT / "results/workload_audit/query_dataset_memberships.jsonl").resolve()
    if output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("Supplemental output must be separate from and outside the source run directory")
    if output_dir.exists():
        raise ValueError("Supplemental output already exists; choose a new directory")
    paths = {"run": run_dir / "run.json", "raw": run_dir / "raw.jsonl", "classification": classification_path}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    metadata = _json(contents["run"], paths["run"])
    if not isinstance(metadata, dict):
        raise ValueError("run.json must contain an object")
    raw_rows = _jsonl(contents["raw"], paths["raw"])
    classification_rows = _jsonl(contents["classification"], paths["classification"])
    memberships, source_groups, planned_members = _validate(metadata, raw_rows, classification_rows)
    input_files = {name: {"path": str(paths[name]), "sha256": digest(data), "bytes": len(data)}
                   for name, data in contents.items()}
    receipts, reports = {}, {}
    for name, definition in _VIEWS.items():
        groups, expected_memberships, expected_tasks = [], set(), 0
        for key, source_group in source_groups.items():
            if not _view_group(name, source_group):
                continue
            selected = {q for q in planned_members[key]
                        if memberships[q, source_group["dataset"]]["classification"] == definition["classification"]}
            groups.append({**deepcopy(source_group), "expected_queries": len(selected)})
            expected_memberships.update((q, source_group["dataset"]) for q in selected)
            expected_tasks += len(selected)
        chosen = [(record, line) for record, line in raw_rows
                  if (record["query_id"], record["dataset"]) in expected_memberships]
        receipt = {
            "cohort_id": name, **definition, "supplemental": True,
            "can_establish_original_full_task_acceptance": False,
            "expected_query_dataset_memberships": len(expected_memberships),
            "expected_unique_query_ids": len({q for q, _ in expected_memberships}),
            "expected_tasks": expected_tasks, "recorded_tasks": len(chosen),
            "missing_tasks": expected_tasks - len(chosen), "planned_groups": len(groups),
            "applicable_to_source_run": bool(expected_memberships),
            "expected_memberships": [{"query_id": q, "dataset": d} for q, d in sorted(expected_memberships)],
            "input_files": input_files,
            "source_run_hash": metadata["run_hash"],
            "source_fingerprint_scope": "Unchanged source fingerprint identifies the original run, not this supplemental group projection.",
        }
        view_metadata = deepcopy(metadata)
        view_metadata.update(partial_run=True, source_partial_run=metadata.get("partial_run"),
                             planned_groups=groups, planned_tasks=expected_tasks,
                             selected_queries=receipt["expected_unique_query_ids"], supplemental_cohort=receipt)
        report = build_report([r for r, _ in chosen], groups, metadata.get("verification_mode", "strict"), view_metadata)
        report["definitions"]["supplemental_scope"] = receipt["scope"]
        report["acceptance_scope"]["supplemental"] = True
        report["acceptance_scope"]["cohort_id"] = name
        report["acceptance_scope"]["can_establish_original_full_task_acceptance"] = False
        receipts[name], reports[name] = receipt, (report, chosen)
    # Detect an active source writer before producing any supplemental files.
    if any(path.read_bytes() != contents[name] for name, path in paths.items()):
        raise ValueError("Input evidence changed while reading; wait for the source run to finish")
    output_dir.mkdir(parents=True)
    for name, (report, chosen) in reports.items():
        destination = output_dir / name
        write_reports(destination, report)
        (destination / "raw.jsonl").write_bytes(b"".join(line for _, line in chosen))
        (destination / "source_run.json").write_bytes(contents["run"])
        (destination / "cohort.json").write_text(json.dumps(receipts[name], indent=2, allow_nan=False) + "\n")
    result = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Supplemental workload membership reports; original acceptance and evidence remain unchanged.",
        "can_establish_original_full_task_acceptance": False, "input_files": input_files,
        "implementation_sha256": {"reporter": digest(Path(__file__).read_bytes()),
                                  "reporting": digest((ROOT / "mcts_rewrite/reporting.py").read_bytes())},
        "source_run_hash": metadata["run_hash"], "source_selected_query_ids": len(metadata["fingerprint"]["selected_query_ids"]),
        "classification_memberships": len(memberships), "source_recorded_tasks": len(raw_rows),
        "source_planned_tasks": metadata["planned_tasks"],
        "source_expected_query_dataset_memberships": len({(q, key[2]) for key, queries in planned_members.items() for q in queries}),
        "views": receipts,
    }
    (output_dir / "cohorts.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing source run directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside the source run")
    parser.add_argument("--classification", type=Path, default=ROOT / "results/workload_audit/query_dataset_memberships.jsonl")
    args = parser.parse_args(argv)
    try:
        report = report_cohorts(args.run, args.output, args.classification)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output.resolve()), "views": {
        name: {key: view[key] for key in ("expected_query_dataset_memberships", "expected_tasks", "recorded_tasks", "missing_tasks")}
        for name, view in report["views"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
