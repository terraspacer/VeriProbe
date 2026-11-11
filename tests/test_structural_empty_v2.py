"""Offline proof and fail-closed context tests; no PostgreSQL service is used."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcts_rewrite.structural_empty_v2 import (
    VERSION, ScopedStructuralEmptyGateV2, StructuralEmptyEquivalenceGateV2,
    prove_tautological_empty,
)
from mcts_rewrite.verify import EquivalenceGate


SCHEMA = "CREATE TABLE emp(empno INT, deptno INT, mgr INT); CREATE TABLE dept(deptno INT);"
ORIGINAL = """SELECT empno, deptno, w_count FROM (
    SELECT empno, deptno, COUNT(empno) OVER (w) AS w_count
    FROM emp WINDOW w AS (PARTITION BY deptno ORDER BY empno)
) sub_query WHERE w_count IS NULL"""
CANDIDATE = "SELECT empno, deptno, 0::bigint AS w_count FROM emp WHERE FALSE"
COLUMNS = [{"name": "empno", "type_oid": 23}, {"name": "deptno", "type_oid": 23},
           {"name": "w_count", "type_oid": 20}]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def context(original=ORIGINAL):
    return {"version": VERSION, "postgresql_validated": True, "query_sha256": sha(original),
            "schema_sha256": sha(SCHEMA), "source_tables": ["dept", "emp"],
            "rows_by_table": {"dept": 1, "emp": 3}, "search_path": "fixture_skew"}


class EmptyDB:
    def __init__(self, change=None):
        self.change = change
        self.calls = []

    def compare(self, original, candidate):
        self.calls.append((original, candidate))
        result = {"equivalent": True, "same_snapshot": True, "status": "ok",
                  "comparison_profile": "positional_numeric", "float_rel_tol": 1e-5, "float_abs_tol": 1e-5,
                  "original": {"row_count": 0, "nonnull_cells": 0, "columns": deepcopy(COLUMNS), "sha256": "empty"},
                  "candidate": {"row_count": 0, "nonnull_cells": 0, "columns": deepcopy(COLUMNS), "sha256": "empty"}}
        if self.change:
            self.change(result)
        return result


def gate(*, db=None, bound=None, validator=None, original=ORIGINAL):
    return StructuralEmptyEquivalenceGateV2(db or EmptyDB(), SCHEMA, mode="hybrid",
        mini_dbs=[EmptyDB() for _ in range(5)], structural_context=bound if bound is not None else context(original),
        context_validator=validator or (lambda *_: {"supported": True, "nonempty_by_table": {"emp": True, "dept": True}}))


def test_exact_frozen_calcite_original_and_saved_candidate_prove_independently():
    root = Path(__file__).resolve().parents[1]
    row = next(row for row in map(json.loads, (root / "benchmarks/stage_a_30.jsonl").read_text().splitlines())
               if row["id"] == "rbot:calcite:testIsNullPushDown")
    schema = (root / row["schema_path"]).read_text()
    original = prove_tautological_empty(row["sql"], schema)
    assert original["status"] == "proven"
    assert original["proof_kind"] == "count_window_is_null"
    assert original["query_sha256"] == sha(row["sql"])
    assert original["schema_sha256"] == row["schema_sha256"]
    assert original["source_tables"] == ["emp"]
    assert original["steps"] == [
        {"path": "root.from", "rule": "count_window_is_nonnull", "output_column": "w_count", "count_argument": "empno"},
        {"path": "root", "rule": "is_null_of_nonnull_count_window", "input_column": "w_count"},
    ]
    candidate = prove_tautological_empty(CANDIDATE, schema)
    assert candidate["status"] == "proven" and candidate["proof_kind"] == "where_false"


@pytest.mark.parametrize("sql", [
    ORIGINAL,
    ORIGINAL.replace("COUNT(empno) OVER (w)", "COUNT(mgr) OVER (w)"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "COUNT(*) OVER (ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)"),
    "WITH c AS (SELECT empno, COUNT(empno) OVER () AS n FROM emp) SELECT c.empno,c.n FROM c WHERE c.n IS NULL",
    "WITH c(x,n) AS (SELECT empno,COUNT(empno) OVER () FROM emp) SELECT x,n FROM c WHERE n IS NULL",
    "SELECT e,n FROM (SELECT empno AS e,w_count AS n FROM (SELECT empno,COUNT(empno) OVER () AS w_count FROM emp) a) b WHERE b.n IS NULL",
    "SELECT x,n FROM (SELECT empno,COUNT(empno) OVER () AS w_count FROM emp) s(x,n) WHERE s.n IS NULL",
    "SELECT empno,w_count FROM (SELECT empno,COUNT(*) OVER () AS w_count FROM emp) s WHERE (w_count IS NULL)",
    "SELECT empno,n FROM (SELECT empno,COUNT(*) OVER () AS n FROM emp WHERE FALSE) s",
    CANDIDATE,
])
def test_safe_single_source_lineage_and_empty_frames(sql):
    proof = prove_tautological_empty(sql, SCHEMA)
    assert proof["status"] == "proven", proof
    assert proof["root_empty"] is True


@pytest.mark.parametrize("sql", [
    # SUM/MIN/AVG can be NULL. NULLIF/CASE can make even COUNT nullable.
    ORIGINAL.replace("COUNT(empno)", "SUM(empno)"),
    ORIGINAL.replace("COUNT(empno)", "MIN(empno)"),
    ORIGINAL.replace("COUNT(empno)", "AVG(empno)"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "NULLIF(COUNT(empno) OVER (w), 1)"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "CASE WHEN empno=1 THEN NULL ELSE COUNT(empno) OVER (w) END"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "COUNT(empno)"),
    ORIGINAL.replace("COUNT(empno)", "COUNT(DISTINCT empno)"),
    ORIGINAL.replace("COUNT(empno)", "COUNT(empno + 1)"),
    ORIGINAL.replace("COUNT(empno)", "COUNT(empno) FILTER (WHERE empno=1)"),
    ORIGINAL.replace("COUNT(empno)", "ROW_NUMBER()"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE w_count IS NULL OR empno=1"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE NOT (w_count IS NOT NULL)"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE w_count IS NULL AND empno=1"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE w_count IS NOT NULL"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE deptno IS NULL"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE missing.w_count IS NULL"),
    ORIGINAL.replace("WHERE w_count IS NULL", "WHERE sub_query.missing IS NULL"),
    # A SELECT-list alias cannot replace the bound input of this WHERE.
    "SELECT COUNT(*) OVER () AS w_count FROM emp WHERE w_count IS NULL",
    "SELECT 1 AS w_count FROM emp WHERE w_count IS NULL",
    "SELECT mgr AS w_count FROM emp WHERE mgr IS NULL",
    "SELECT empno,w_count FROM (SELECT empno,mgr AS w_count FROM emp) s WHERE w_count IS NULL",
    # Outer joins can NULL-extend an otherwise non-NULL window result.
    "SELECT s.n FROM dept d LEFT JOIN (SELECT deptno,COUNT(*) OVER () n FROM emp) s ON d.deptno=s.deptno WHERE s.n IS NULL",
    ORIGINAL.replace("FROM emp WINDOW", "FROM emp JOIN dept ON emp.deptno=dept.deptno WINDOW"),
    ORIGINAL.replace("FROM emp WINDOW", "FROM emp,dept WINDOW"),
    ORIGINAL.replace("FROM emp WINDOW", "FROM missing WINDOW"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "COUNT(empno) OVER (missing)"),
    ORIGINAL.replace("COUNT(empno) OVER (w)", "COUNT(missing) OVER (w)"),
    ORIGINAL.replace("PARTITION BY deptno", "PARTITION BY missing"),
    ORIGINAL.replace("ORDER BY empno", "ORDER BY empno + 1"),
    ORIGINAL + " UNION SELECT empno,deptno,mgr FROM emp",
    "SELECT count(*) AS n FROM emp WHERE FALSE",  # aggregate emits one row
    "WITH c(n) AS (SELECT mgr FROM emp) SELECT n FROM c WHERE n IS NULL",
    "WITH RECURSIVE c(n) AS (SELECT mgr FROM emp) SELECT n FROM c WHERE FALSE",
    "WITH emp AS (SELECT mgr FROM dept) SELECT mgr FROM emp WHERE FALSE",
    "WITH c AS (SELECT COUNT(*) OVER () n FROM emp), c AS (SELECT mgr n FROM emp) SELECT n FROM c WHERE n IS NULL",
    "SELECT x.n FROM (SELECT COUNT(*) OVER () n,mgr n FROM emp) x WHERE x.n IS NULL",
    "SELECT x.n FROM (SELECT COUNT(*) OVER () n FROM emp) x(n,z) WHERE x.n IS NULL",
    "SELECT empno,mgr AS n FROM emp WHERE FALSE ORDER BY empno",
    "SELECT empno,1/0 AS n FROM emp WHERE FALSE",
    'SELECT empno,mgr AS n FROM "emp" WHERE FALSE',
    "/*+ Set(search_path hidden) */ " + CANDIDATE,
    "SELECT empno,mgr AS n FROM public.emp WHERE FALSE",
    "SELECT 1 AS n WHERE FALSE",  # no bound base table
])
def test_counterexamples_and_unmodeled_sql_never_receive_empty_proof(sql):
    proof = prove_tautological_empty(sql, SCHEMA)
    assert proof["status"] == "unsupported", proof
    assert proof["root_empty"] is False


def test_gate_requires_independent_proofs_and_keeps_legacy_vacuity_rejection():
    result = gate().verify(ORIGINAL, CANDIDATE)
    assert result["accepted"] and result["formal"] and result["proof"] == "structural_empty_v2"
    assert result["structural_empty"]["proofs"]["original"]["proof_kind"] == "count_window_is_null"
    assert result["structural_empty"]["proofs"]["candidate"]["proof_kind"] == "where_false"
    bad_candidate = "SELECT empno, deptno, mgr::bigint AS w_count FROM emp WHERE empno=999"
    result = gate().verify(ORIGINAL, bad_candidate)
    assert not result["accepted"] and "empty" in result["reason"]
    legacy = EquivalenceGate(EmptyDB(), SCHEMA, mode="hybrid", mini_dbs=[EmptyDB() for _ in range(5)])
    assert not legacy.verify(ORIGINAL, CANDIDATE)["accepted"]


@pytest.mark.parametrize("change", [
    lambda c: c.update(version="other"),
    lambda c: c.update(postgresql_validated=False),
    lambda c: c.update(query_sha256="other"),
    lambda c: c.update(schema_sha256="other"),
    lambda c: c.update(source_tables=["emp"]),
    lambda c: c.update(rows_by_table={"emp": 3}),
    lambda c: c["rows_by_table"].update(emp=0),
    lambda c: c["rows_by_table"].update(dept=0),
    lambda c: c["rows_by_table"].update(emp=True),
    lambda c: c["rows_by_table"].update(emp=-1),
])
def test_missing_empty_or_mismatched_measured_context_cannot_accept(change):
    bound = context()
    change(bound)
    db = EmptyDB()
    result = gate(db=db, bound=bound).verify(ORIGINAL, CANDIDATE)
    assert not result["accepted"] and not result["formal"]
    assert not db.calls


@pytest.mark.parametrize("change", [
    lambda r: r.update(equivalent=False),
    lambda r: r.update(same_snapshot=False),
    lambda r: r.update(status="timeout"),
    lambda r: r.update(comparison_profile="strict_metadata"),
    lambda r: r["original"].update(row_count=1),
    lambda r: r["candidate"].update(row_count=1),
    lambda r: r["original"].update(row_count=False),
    lambda r: r["candidate"]["columns"].pop(),
    lambda r: r["original"].update(columns=[]),
    lambda r: r["candidate"].update(columns=[]),
    lambda r: r["original"].pop("columns"),
])
def test_failed_execution_profile_or_arity_drift_cannot_accept(change):
    result = gate(db=EmptyDB(change)).verify(ORIGINAL, CANDIDATE)
    assert not result["accepted"] and not result["formal"]


def test_positional_numeric_accepts_alias_difference_and_numeric_width_compatibility():
    candidate = "SELECT empno, deptno, 0 AS changed_alias FROM emp WHERE FALSE"
    def numeric_width_and_alias(result):
        result["candidate"]["columns"][2].update(name="changed_alias", type_oid=23)
        result["numeric_relaxation_used"] = True
    result = gate(db=EmptyDB(numeric_width_and_alias)).verify(ORIGINAL, candidate)
    assert result["accepted"] and result["formal"] and result["proof"] == "structural_empty_v2"
    assert result["execution"]["numeric_relaxation_used"] is True
    assert result["execution"]["original"]["columns"] != result["execution"]["candidate"]["columns"]


def test_compare_non_numeric_incompatibility_overrides_structural_empty_proofs():
    def incompatible(result):
        result["candidate"]["columns"][2].update(type_oid=25)
        result["equivalent"] = False
    result = gate(db=EmptyDB(incompatible)).verify(ORIGINAL, CANDIDATE)
    assert not result["accepted"] and not result["formal"]


def test_no_context_or_live_empty_tables_cannot_accept():
    bare = StructuralEmptyEquivalenceGateV2(EmptyDB(), SCHEMA)
    assert not bare.verify(ORIGINAL, CANDIDATE)["accepted"]
    unsupported = gate(validator=lambda *_: {"supported": False, "nonempty_by_table": {"emp": False}})
    result = unsupported.verify(ORIGINAL, CANDIDATE)
    assert not result["accepted"] and not result["formal"]


def test_scoped_context_checks_exact_binding_and_live_verification_table_existence():
    class Cursor:
        def __init__(self): self.seen = []
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def execute(self, statement): self.seen.append(statement)
        def fetchone(self): return [self.nonempty]
        nonempty = True

    cursor = Cursor()
    db = SimpleNamespace(search_path="fixture_skew", _configure=lambda c: None,
                         connection=SimpleNamespace(transaction=nullcontext, cursor=lambda: cursor))
    scoped = ScopedStructuralEmptyGateV2.__new__(ScopedStructuralEmptyGateV2)
    scoped.verification_db = db
    scoped.schema_sql = SCHEMA
    scoped.validation = {"schema_source_sha256": sha(SCHEMA), "catalog_sha256": "catalog",
        "verification": {"resolved_relations": {name: {"schema": "fixture_skew", "relation": name}
                                                     for name in ("emp", "dept")}}}
    checked = []
    scoped._query_scope = lambda query, current: checked.append(query)
    result = scoped._structural_context(ORIGINAL, CANDIDATE, context())
    assert result["supported"] and result["nonempty_by_table"] == {"dept": True, "emp": True}
    assert checked == [ORIGINAL, CANDIDATE] and len(cursor.seen) == 2
    assert result["performance_database_rows_executed_or_hashed"] is False
    cursor.nonempty = False
    assert not scoped._structural_context(ORIGINAL, CANDIDATE, context())["supported"]
    bound = context()
    bound["search_path"] = "other"
    assert not scoped._structural_context(ORIGINAL, CANDIDATE, bound)["supported"]
