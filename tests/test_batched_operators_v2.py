import sqlite3
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

from mcts_rewrite.batched_operators import BatchedOperators
from mcts_rewrite.batched_operators_v2 import BatchedOperatorsV2, canonicalize_not_null_counts
from mcts_rewrite.fast_search import FastSearchConfig, FastSearchController
from tests.test_batched_operators import Calcite, Retriever
from tests.test_fast_search import Database, Gate


SCHEMA = "CREATE TABLE t(a INT PRIMARY KEY, b INT NOT NULL, c INT);"


def execute(connection, sql):
    translated = sqlglot.transpile(sql, read="postgres", write="sqlite")[0]
    return connection.execute(translated).fetchall()


@pytest.mark.parametrize("rows", [[], [(1, 4, None)], [(1, 4, None), (2, 4, 8), (3, -2, 8)]])
@pytest.mark.parametrize("query", [
    "SELECT COUNT(b)::numeric, COUNT(c)::numeric FROM t",
    "SELECT COUNT(x.a), COUNT(x.b), COUNT(x.c) FROM t AS x WHERE b >= 0",
    "SELECT b, COUNT(b) FROM t GROUP BY b ORDER BY b LIMIT 2",
    "SELECT COUNT(b) FILTER (WHERE c IS NULL) FROM t",
])
def test_count_normalization_preserves_empty_null_filter_duplicate_and_group_results(rows, query):
    receipt = canonicalize_not_null_counts(query, SCHEMA)
    assert receipt and receipt["verified"] is False and receipt["requires_catalog_compatibility"]
    with sqlite3.connect(":memory:") as db:
        db.execute(SCHEMA)
        db.executemany("INSERT INTO t VALUES(?,?,?)", rows)
        assert execute(db, query) == execute(db, receipt["sql"])
    # This is finite SQLite development evidence, not PostgreSQL validation.


@pytest.mark.parametrize("query", [
    "SELECT COUNT(c) FROM t",
    "SELECT COUNT(DISTINCT b) FROM t",
    "SELECT COUNT(b + 0) FROM t",
    "SELECT COUNT(x.b) FROM t",
    "SELECT COUNT(t.b) FROM t AS x",
    "SELECT COUNT(t.b) FROM u LEFT JOIN t ON u.a=t.a",
    "SELECT COUNT(b) FROM (SELECT b FROM t) x",
    "WITH t AS (SELECT NULL AS b) SELECT COUNT(b) FROM t",
    "SELECT COUNT(b) OVER () FROM t",
    "SELECT COUNT(b) FROM public.t",
    "SELECT COUNT(b) FROM t UNION SELECT 1",
    "SELECT COUNT(b) FROM t; SELECT 2",
    "SELECT COUNT(b) FROM t AS x(a,b,c)",
])
def test_ambiguous_nullable_derived_or_complex_count_is_unchanged(query):
    assert canonicalize_not_null_counts(query, SCHEMA) is None


def test_outer_join_counterexample_and_quoted_identifiers():
    with sqlite3.connect(":memory:") as db:
        db.execute(SCHEMA)
        db.execute("CREATE TABLE u(a INT)")
        db.execute("INSERT INTO u VALUES(9)")
        q = "SELECT COUNT(t.b) FROM u LEFT JOIN t ON u.a=t.a"
        assert db.execute(q).fetchall() == [(0,)]
        assert db.execute(q.replace("COUNT(t.b)", "COUNT(*)")).fetchall() == [(1,)]
        assert canonicalize_not_null_counts(q, SCHEMA) is None
    quoted = 'CREATE TABLE "T" ("B" INT NOT NULL);'
    assert canonicalize_not_null_counts('SELECT COUNT("B") FROM "T"', quoted)
    assert canonicalize_not_null_counts('SELECT COUNT(b) FROM "T"', quoted) is None
    assert canonicalize_not_null_counts('SELECT COUNT("B") FROM t', quoted) is None


def test_composite_primary_key_and_nullable_unique_are_distinct():
    schema = "CREATE TABLE t(a INT,b INT,c INT UNIQUE,PRIMARY KEY(a,b));"
    receipt = canonicalize_not_null_counts("SELECT COUNT(a),COUNT(b),COUNT(c) FROM t", schema)
    assert receipt["columns"] == ["a", "b"]
    assert "COUNT(c)" in receipt["sql"]


def test_v2_only_normalizes_mcts_and_does_not_add_model_calls(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"batches": [{"parent_id": "root", "candidates": [
            {"sql": "SELECT COUNT(b)::numeric FROM t", "action": "collapse_aggregate"}]}]}
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=Calcite())
    parent = [{"id": "root", "sql": "SELECT SUM(n) FROM (SELECT COUNT(b) n FROM t GROUP BY b) s"}]
    candidate = ops.generate_batch(parent, SCHEMA)[0]
    assert len(calls) == ops.model_requests == 1
    assert "COUNT(*)" in candidate["sql"]
    assert candidate["provenance"]["verified"] is False
    assert candidate["provenance"]["count_normalization"]["source_sql_sha256"]
    assert calls[-1]["feedback_scope"] == "transductive_stage_a_followup"
    assert "reviewed_action_patterns" in calls[-1]
    standalone = ops.macro_batch(parent, SCHEMA, operator="quite")[0]
    assert len(calls) == ops.model_requests == 2
    assert "COUNT(b)" in standalone["sql"]
    assert "reviewed_action_patterns" not in calls[-1]


def test_v2_same_count_canonical_form_consumes_one_candidate_slot(tmp_path):
    def model(payload):
        return {"batches": [{"parent_id": "root", "candidates": [
            {"sql": "SELECT count(b) FROM t"}, {"sql": "SELECT COUNT(a) FROM t"}]}]}
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=Calcite())
    result = ops.generate_batch([{"id": "root", "sql": "SELECT SUM(n) FROM t"}], SCHEMA)
    assert len(result) == 1 and ops.model_requests == 1


def test_nullable_count_counterexample_is_not_rewritten():
    with sqlite3.connect(":memory:") as db:
        db.execute(SCHEMA)
        db.execute("INSERT INTO t VALUES(1,1,NULL)")
        assert execute(db, "SELECT COUNT(c) FROM t") != execute(db, "SELECT COUNT(*) FROM t")
    assert canonicalize_not_null_counts("SELECT COUNT(c) FROM t", SCHEMA) is None


def join_query_proposals(query_name):
    """Read pinned SQL only; these are structural development proposals."""
    root = Path(__file__).resolve().parents[1]
    source = root / f"vendor/LLM4Rewrite/tpch/{query_name}/{query_name}_0.sql"
    original = sqlglot.parse_one(source.read_text(), read="postgres").sql(dialect="postgres", comments=False)
    inline = sqlglot.parse_one(original, read="postgres")
    for select in inline.find_all(exp.Select):
        for join in select.args.get("joins", []):
            join.set("kind", "INNER")
            join.set("on", exp.true())
    dimension, columns, predicate = (("part", "p_partkey, p_name", "p_name LIKE '%cyan%'")
        if query_name == "query9" else ("region", "r_regionkey, r_name", "r_name = 'EUROPE'"))
    materialized = sqlglot.parse_one(original, read="postgres")
    for table in materialized.find_all(exp.Table):
        if table.name == dimension:
            table.set("this", exp.to_identifier("selected_dimension"))
    materialized = (f"WITH selected_dimension AS MATERIALIZED (SELECT {columns} FROM {dimension} WHERE {predicate}) "
                    + materialized.sql(dialect="postgres"))
    schema = (root / "vendor/LLM4Rewrite/tpch/create_tables.sql").read_text()
    return original, materialized, inline.sql(dialect="postgres"), schema


class OverfullCalcite(Calcite):
    def match(self, sql, schema):
        return {"status": "ok", "matches": [{"name": f"RULE_{i}"} for i in range(3)]}

    def rewrite(self, sql, schema, sequences):
        self.applied.append((sql, sequences))
        return {"status": "ok", "rewrites": [{"sql": f"SELECT {100 + i}", "rules": sequence}
            for i, sequence in enumerate(sequences)]}


def overfull_join_response(parent_ids, materialized, inline):
    return {"batches": [{"parent_id": parent_id,
        "rule_sequences": [[f"RULE_{i}"] for i in range(3)],
        # Misleading action labels must not influence structural coverage.
        "candidates": [{"sql": "SELECT 11", "action": "selective_dimension_materialization"},
                       {"sql": "SELECT 12", "action": "inline_join"},
                       {"sql": materialized, "action": "macro"},
                       {"sql": inline, "action": "macro"}]}
        for parent_id in parent_ids]}


@pytest.mark.parametrize("query_name", ["query9", "query5"])
def test_tpch_multi_join_requests_and_preserves_both_families_before_three_slot_truncation(tmp_path, query_name):
    original, materialized, inline, schema = join_query_proposals(query_name)
    calls = []
    def model(payload):
        calls.append(payload)
        return overfull_join_response(["root"], materialized, inline)
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=OverfullCalcite())
    rows = ops.generate_batch([{"id": "root", "sql": original}], schema, limit_per_parent=3)
    assert len(calls) == ops.model_requests == 1
    assert calls[0]["parents"][0]["required_strategy_families"] == [
        "selective_dimension_materialization", "inline_join"]
    assert "WITH ... AS MATERIALIZED" in calls[0]["instruction"]
    assert {pattern["id"] for pattern in calls[0]["reviewed_action_patterns"]} == {
        "not_null_count_star", "nested_aggregate_collapse", "selective_dimension_materialization"}
    assert len(rows) == 3 and {row["action_class"] for row in rows} == {"A", "B"}
    assert {row["sql"] for row in rows} == {"SELECT 100", materialized, inline}
    assert {row["provenance"].get("join_strategy_family") for row in rows} >= {
        "selective_dimension_materialization", "inline_join"}
    assert all(row["provenance"]["verified"] is False for row in rows)
    assert any(row["status"] == "join_strategy_pair_preserved" for row in ops.last_diagnostics)


def test_three_parent_beam_preserves_each_strategy_pair_and_parent_attribution_in_one_call(tmp_path):
    original, materialized, inline, schema = join_query_proposals("query5")
    calls = []
    def model(payload):
        calls.append(payload)
        return overfull_join_response([0, 1, 2], materialized, inline)
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=OverfullCalcite())
    parents = [{"id": i, "sql": original} for i in range(3)]
    rows = ops.generate_batch(parents, schema, depth=2, limit_per_parent=3)
    assert len(calls) == ops.model_requests == 1 and calls[0]["beam_width"] == 3
    for parent in parents:
        proposals = [row for row in rows if row["parent_id"] == parent["id"]]
        assert len(proposals) == 3
        assert {row["sql"] for row in proposals} == {"SELECT 100", materialized, inline}
        assert all(row["parent_sql"] == parent["sql"] for row in proposals)


def test_join_strategy_changes_do_not_affect_standalone_quite_or_single_table_queries(tmp_path):
    original, materialized, inline, schema = join_query_proposals("query5")
    calls = []
    def model(payload):
        calls.append(payload)
        return overfull_join_response(["root"], materialized, inline)
    parent = [{"id": "root", "sql": original}]
    v2 = BatchedOperatorsV2(tmp_path, model=model, calcite=OverfullCalcite())
    v1 = BatchedOperators(tmp_path, model=model, calcite=OverfullCalcite())
    assert v2.macro_batch(parent, schema) == v1.macro_batch(parent, schema)
    assert all("required_strategy_families" not in call["parents"][0] for call in calls)
    assert all("reviewed_action_patterns" not in call for call in calls)
    calls.clear()
    v2.generate_batch([{"id": "root", "sql": "SELECT COUNT(b) FROM t"}], SCHEMA)
    assert "required_strategy_families" not in calls[0]["parents"][0]
    assert "required_strategy_families" not in calls[0]["instruction"]


def test_v2_leaves_standalone_rbot_payloads_candidates_and_two_call_budget_unchanged(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"rule_sequences": [["FILTER_REDUCE_EXPRESSIONS"]]}
    v2 = BatchedOperatorsV2(tmp_path, model=model, calcite=Calcite(), retriever=Retriever())
    v1 = BatchedOperators(tmp_path, model=model, calcite=Calcite(), retriever=Retriever())
    assert v2.rbot("SELECT 1 WHERE TRUE", "") == v1.rbot("SELECT 1 WHERE TRUE", "")
    assert v2.model_requests == v1.model_requests == 2
    assert all("reviewed_action_patterns" not in call for call in calls)
    assert all("required_strategy_families" not in call for call in calls)


def test_missing_family_is_diagnosed_without_retry_or_automatic_sql_synthesis(tmp_path):
    original, materialized, inline, schema = join_query_proposals("query9")
    # A comment and an unreferenced MATERIALIZED CTE do not supply that family.
    unused = "WITH unused AS MATERIALIZED (SELECT 1 WHERE TRUE) " + inline
    def model(payload):
        return {"batches": [{"parent_id": "root", "candidates": [
            {"sql": unused, "action": "selective_dimension_materialization"}, {"sql": inline}]}]}
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=Calcite())
    rows = ops.generate_batch([{"id": "root", "sql": original}], schema)
    assert ops.model_requests == 1 and len(rows) == 2
    assert all(row["sql"] != materialized for row in rows)
    coverage = [row for row in ops.last_diagnostics if row["status"] == "join_strategy_pair_incomplete"]
    assert coverage[0]["missing_strategy_families"] == ["selective_dimension_materialization"]
    assert coverage[0]["verified"] is False and coverage[0]["additional_model_calls"] == 0


@pytest.mark.parametrize("query_name", ["query9", "query5"])
def test_strategy_pair_reaches_existing_root_gate_explain_and_canary_ranking(tmp_path, query_name):
    original, materialized, inline, schema = join_query_proposals(query_name)
    calls, log = [], []
    def model(payload):
        calls.append(payload)
        return overfull_join_response([0], materialized, inline) if payload["task"] == "mcts_batched_generation" else {}
    ops = BatchedOperatorsV2(tmp_path, model=model, calcite=OverfullCalcite())
    # Synthetic plan/time doubles test routing and ordering; no PostgreSQL or
    # benchmark execution occurs and these numbers are not performance evidence.
    db = Database(costs={original: 100, "SELECT 100": 200, materialized: 30, inline: 20},
                  times={materialized: 6, inline: 8}, log=log)
    gate = Gate(log=log)
    config = FastSearchConfig(beam_width=3, max_depth=1, use_rag=False)
    result = FastSearchController(db, gate, ops, config).run(original, schema, original_ms=10)
    assert result["sql"] == materialized
    assert [call["task"] for call in calls] == ["mcts_batched_generation", "mcts_batched_cost_critic"]
    assert ops.model_requests == 2
    assert [measurement[0] for measurement in db.measurements] == [inline, materialized]
    for candidate in (materialized, inline):
        assert log.index(("verify", candidate)) < log.index(("explain", candidate)) < log.index(("canary", candidate))
    assert all(root == original for root, _ in gate.calls)
