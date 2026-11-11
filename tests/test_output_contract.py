import os

import pytest

from mcts_rewrite.db import Postgres, parse_read_query
from mcts_rewrite.output_contract import restore_output_contract


@pytest.fixture(scope="module")
def live():
    dsn = os.environ.get("MCTS_TEST_DSN")
    if not dsn:
        pytest.skip("set MCTS_TEST_DSN for live PostgreSQL integration")
    with Postgres(dsn) as database:
        yield database


class _NeverConnect:
    @property
    def connection(self):
        raise AssertionError("unsafe SQL must fail before connecting")


@pytest.mark.parametrize("unsafe", [
    "SELECT 1; SELECT 2", "SELECT pg_sleep(1)", "SELECT nextval('sequence')",
    "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
    "SELECT public.abs(1)", "SELECT * INTO t FROM (VALUES (1)) x(a)",
])
def test_unsafe_proposals_are_rejected_before_description(unsafe):
    for original, candidate in (("SELECT 1", unsafe), (unsafe, "SELECT 1")):
        result = restore_output_contract(_NeverConnect(), original, candidate)
        assert result["status"] == "rejected" and result["sql"] is None
        assert result["raw_sql"] == candidate
        assert result["requires_verification"] and not result["formal"]


def test_live_matching_contract_returns_exact_raw_sql(live):
    candidate = "SELECT 2 AS a; -- retain original spelling\n"
    result = restore_output_contract(live, "SELECT 1 AS a", candidate)
    assert result["status"] == "unchanged", result
    assert result["sql"] == result["raw_sql"] == candidate
    assert not result["restoration"] and result["requires_verification"]
    # Compatible metadata does not establish equal values.
    assert not live.compare("SELECT 1 AS a", result["sql"])["equivalent"]


@pytest.mark.parametrize("source,target", [
    ("smallint", "integer"), ("smallint", "bigint"), ("integer", "bigint"),
])
def test_live_lossless_integer_widening(live, source, target):
    original = f"SELECT 32767::{target} AS a, NULL::{target} AS b"
    candidate = f"SELECT 32767::{source} AS x, NULL::{source} AS y"
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "restored", result
    assert [step["cast"] for step in result["restoration"]] == [target.upper()] * 2
    assert result["original_columns"] == result["restored_columns"]
    assert live.compare(original, result["sql"])["equivalent"]


@pytest.mark.parametrize("original,candidate", [
    ("SELECT 1::smallint AS a", "SELECT 1::integer AS a"),
    ("SELECT 1::integer AS a", "SELECT 1::bigint AS a"),
    ("SELECT 1::numeric AS a", "SELECT 1::integer AS a"),
    ("SELECT 1::integer AS a", "SELECT 1::numeric AS a"),
    ("SELECT 1::numeric(12,2) AS a", "SELECT 1::numeric(14,2) AS a"),
    ("SELECT 'x'::varchar(4) AS a", "SELECT 'x'::varchar(8) AS a"),
    ("SELECT 'x'::varchar AS a", "SELECT 'x'::text AS a"),
    ("SELECT 1 AS a, 2 AS b", "SELECT 1 AS a"),
])
def test_live_narrowing_coercion_typmod_and_count_changes_rejected(live, original, candidate):
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "rejected" and result["sql"] is None, result
    assert result["raw_sql"] == candidate


def test_live_duplicate_quoted_names_and_semicolon_literal(live):
    original = 'SELECT 1 AS "a\"\"b", 2::bigint AS "a\"\"b", \';\'::text AS "semi;colon"'
    candidate = "SELECT 1 AS x, 2 AS x, ';'::text AS x;; -- trailing comment\n"
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "restored", result
    assert result["raw_sql"] == candidate
    assert [column["name"] for column in result["restored_columns"]] == ['a"b', 'a"b', 'semi;colon']
    parse_read_query(result["sql"])
    assert live.compare(original, result["sql"])["equivalent"]


def test_live_identical_typmods_are_preserved_during_renaming(live):
    original = "SELECT 1.25::numeric(12,2) AS amount, 'x'::varchar(4) AS label"
    result = restore_output_contract(live, original,
                                     "SELECT 1.25::numeric(12,2) AS x, 'x'::varchar(4) AS y")
    assert result["status"] == "restored", result
    assert all(step["cast"] is None for step in result["restoration"])
    assert result["original_columns"] == result["restored_columns"]
    assert live.compare(original, result["sql"])["equivalent"]


@pytest.mark.parametrize("original,candidate", [
    ("SELECT 1 AS a ORDER BY a", "SELECT 1 AS x ORDER BY x"),
    ("SELECT 1 AS a ORDER BY a", "SELECT 1 AS x"),
    ("SELECT 1 AS a", "SELECT 1 AS x ORDER BY x"),
])
def test_live_restoration_rejects_root_ordering(live, original, candidate):
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "rejected" and "ordered" in result["reason"], result
    unchanged = restore_output_contract(live, original, original)
    assert unchanged["status"] == "unchanged", unchanged


def test_live_metadata_description_does_not_scan_rows(live):
    # Each projected expression raises if evaluated. LIMIT 0 must not reach it.
    # A recursive source prevents PostgreSQL from folding a VALUES constant at
    # planning time, which LIMIT 0 deliberately does not promise to suppress.
    source = "WITH RECURSIVE t(x) AS (VALUES (0) UNION ALL SELECT x+1 FROM t WHERE x<10) "
    original = source + "SELECT (1 / x)::bigint AS quotient FROM t"
    candidate = source + "SELECT 1 / x AS x FROM t"
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "restored", result
    assert live.compare(original, result["sql"])["status"] == "error"
    assert result["requires_verification"] and not result["formal"]


def test_live_ctes_set_operations_and_values_remain_positional(live):
    original = "WITH x AS (VALUES (1::bigint), (NULL::bigint)) SELECT * FROM x"
    candidate = "SELECT 1 AS different UNION ALL SELECT NULL"
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "restored", result
    assert live.compare(original, result["sql"])["equivalent"]


def test_live_description_errors_do_not_poison_next_transaction(live):
    result = restore_output_contract(live, "SELECT 1", "SELECT * FROM __mcts_missing_relation")
    assert result["status"] == "rejected" and result["sql"] is None, result
    next_result = restore_output_contract(live, "SELECT 1 AS a", "SELECT 1 AS x")
    assert next_result["status"] == "restored", next_result


def test_live_quoted_builtin_base_relation_is_described_without_scanning(live):
    original = 'SELECT relnatts::integer AS columns FROM "pg_catalog"."pg_class"'
    candidate = 'SELECT relnatts AS x FROM "pg_catalog"."pg_class"'
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "restored", result
    assert result["restoration"][0]["cast"] == "INTEGER"
    assert live.compare(original, result["sql"])["equivalent"]


def test_live_views_are_rejected_before_hidden_expressions_are_planned(live):
    original = "SELECT schemaname AS schema FROM pg_catalog.pg_tables"
    candidate = "SELECT schemaname AS x FROM pg_catalog.pg_tables"
    result = restore_output_contract(live, original, candidate)
    assert result["status"] == "rejected" and "ordinary base tables" in result["reason"], result
    assert not result["original_columns"] and not result["candidate_columns"]
