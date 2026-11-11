"""Opt-in integration for a separate, transductive Stage A follow-up.

This module starts no service or bridge. The caller must supply its supervised
model factory and a fresh output lifecycle. Offline fixture plans are not
runtime bindings, and the original Stage A runner/configuration stay unchanged.
"""
from copy import deepcopy
from pathlib import Path

from scripts import run_stage_a as stage

from .batched_operators_v2 import BatchedOperatorsV2
from .structural_empty_v2 import CONTEXT_KEY, VERSION, ScopedStructuralEmptyGateV2
from .verification_db import _tables_from_schema


CASES = {"empty", "null", "duplicate", "skew", "unmatched"}


def configuration_for_query(config, row, fixture_bindings=None):
    prepared = deepcopy(config)
    if fixture_bindings is None:
        return prepared
    if (fixture_bindings.get("version") != "query-witness-v2"
            or fixture_bindings.get("status") != "postgresql_validated"):
        raise ValueError("A PostgreSQL-validated fixture binding is required; offline plans are not executable fixtures")
    binding = fixture_bindings.get("queries", {}).get(row["id"])
    if binding is None:
        return prepared
    path = Path(row["schema_path"])
    schema = (path if path.is_absolute() else stage.ROOT / path).read_bytes()
    if (binding.get("query_sha256") != stage.digest(row["sql"].encode())
            or binding.get("schema_sha256") != stage.digest(schema)):
        raise ValueError("Query-specific fixture binding does not match this SQL and source schema")
    selected = binding.get("verification_database", {})
    specs = binding.get("mini_dbs", [])
    validation = binding.get("validation", {})
    if (validation.get("postgresql_executed") is not True
            or validation.get("original_expectations_passed") is not True
            or validation.get("saved_candidate_checks_passed") is not True
            or validation.get("nonvacuous") is not True):
        raise ValueError("PostgreSQL fixture coverage, expectations and differential checks must pass")
    if (len(specs) != 5 or {spec.get("name") for spec in specs} != CASES
            or not isinstance(selected.get("dsn"), str) or not selected["dsn"]
            or not selected.get("search_path")):
        raise ValueError("Five distinct measured cases and an explicit primary verification schema are required")
    expected_tables = set(_tables_from_schema(schema.decode()))
    if set(binding.get("source_tables", [])) != expected_tables:
        raise ValueError("Full source-table coverage must match the exact source DDL")
    nonempty = []
    for spec in specs:
        counts = spec.get("rows_by_table", {})
        if (spec.get("coverage_complete") is not True or spec.get("schema_sha256") != binding["schema_sha256"]
                or set(counts) != expected_tables or spec.get("constraint_rejections") != 0
                or not spec.get("dsn") or not spec.get("search_path")
                or any(type(v) is not int or v < 0 for v in counts.values())):
            raise ValueError("Every fixture needs measured full-schema coverage and an unchanged source schema")
        if spec["name"] == "empty":
            if any(counts.values()):
                raise ValueError("The empty case must be empty")
        elif not all(counts.values()):
            raise ValueError("A nonempty case must populate every declared source table")
        else:
            nonempty.append(spec)
    if not any(spec["dsn"] == selected["dsn"] and spec["search_path"] == selected["search_path"]
               for spec in nonempty):
        raise ValueError("Primary verification must use one of this query's nonempty witness schemas")
    dataset = stage.DATASETS[row["benchmark"]]
    verification = deepcopy(selected)
    primary = next(spec for spec in nonempty if spec["dsn"] == selected["dsn"]
                   and spec["search_path"] == selected["search_path"])
    # This bound context enables only the independently proven empty fragment.
    # Live table existence and existing catalog locks are still checked by the gate.
    verification[CONTEXT_KEY] = {"version": VERSION, "postgresql_validated": True,
        "query_sha256": binding["query_sha256"], "schema_sha256": binding["schema_sha256"],
        "source_tables": sorted(expected_tables), "rows_by_table": deepcopy(primary["rows_by_table"]),
        "search_path": selected["search_path"]}
    prepared["databases"][dataset].update(verification_database=verification, mini_dbs=deepcopy(specs))
    return prepared


def default_components():
    components = stage.default_components()
    components.Operators = BatchedOperatorsV2
    components.ScopedGate = ScopedStructuralEmptyGateV2
    return components


def run_query(row, config, model_factory, *, fixture_bindings=None, components=None, checkpoint=None):
    """Use all original budget, equivalence, canary and final-timing guards."""
    prepared = configuration_for_query(config, row, fixture_bindings)
    metadata = {"version": 2, "feedback_scope": "transductive_stage_a_followup",
                "historical_stage_a_records_modified": False,
                "source_guarded_mcts_count_normalization": True,
                "fixture_override": fixture_bindings is not None and row["id"] in fixture_bindings.get("queries", {})}
    def save(record):
        record["study_variant"] = metadata
        if checkpoint:
            checkpoint(record)
    result = stage.run_query(row, prepared, model_factory, components=components or default_components(), checkpoint=save)
    result["study_variant"] = metadata
    return result
