"""New recipe tests only: SQLite execution is not PostgreSQL validation.

The test adapter translates the emitted PostgreSQL SQL with sqlglot, expands
the historical generate_series/VALUES construct, and stores DECIMAL as REAL
because SQLite otherwise performs integer division on integral decimal values.
This catches recipe/join mistakes; it establishes neither PG equivalence nor
benchmark performance. Production output always retains the exact source DDL.
"""
from collections import Counter
from functools import reduce
import json
import sqlite3

import pytest
import sqlglot
from sqlglot import ErrorLevel, exp

from scripts import build_query_witness_fixtures_v2 as builder
from scripts.build_query_witness_fixtures_v2 import (
    CASES, RECIPES, ROOT, build_plan, query_plan, sha256,
)
from scripts.query_fixture_recipes_v2 import PROOF_QUERY, QUERY_HASHES

LEGACY_RECIPES = {key: value for key, value in RECIPES.items() if not value.get('generated')}


def sqlite_sql(statement):
    statement = statement.copy()
    # SQLite forbids parentheses around simple EXCEPT operands. Removing these
    # particular wrappers preserves the original left-associative set tree.
    for operation in statement.find_all(exp.Except):
        for side in ('this', 'expression'):
            operand = operation.args.get(side)
            if isinstance(operand, exp.Subquery) and not operand.args.get('alias'):
                assert isinstance(operand.this, exp.Select)
                assert not operand.this.args.get('order') and not operand.this.args.get('limit')
                operation.set(side, operand.this.copy())
    # Test-only translation of PostgreSQL's date + INTERVAL syntax.
    for addition in list(statement.find_all(exp.Add)):
        interval, start = addition.expression, addition.this
        if (isinstance(interval, exp.Interval) and isinstance(start, exp.Cast)
                and start.args['to'].this == exp.DataType.Type.DATE):
            assert interval.args['unit'].name.upper() == 'YEAR'
            addition.replace(sqlglot.parse_one(
                f"DATE('{start.this.this}', '+{interval.this.this} year')", read='sqlite'))
    # sqlglot carries PostgreSQL's default NULLS LAST into key expressions;
    # SQLite forbids NULLS LAST in DDL. These pinned key columns are NOT NULL.
    for key in statement.find_all(exp.PrimaryKey):
        key.set("expressions", [column.this.copy() if isinstance(column, exp.Ordered) else column
                                for column in key.expressions])
    # Test-engine type mapping only; never written into a recipe or source DDL.
    for kind in list(statement.find_all(exp.DataType)):
        if kind.this == exp.DataType.Type.DECIMAL:
            kind.replace(exp.DataType.build("REAL"))
    for values in list(statement.find_all(exp.Values)):
        alias = values.args.get("alias")
        if not alias or not alias.columns:
            continue
        selects = [exp.select(*[exp.alias_(v.copy(), c.name) for v, c in zip(row.expressions, alias.columns)])
                   for row in values.expressions]
        union = reduce(lambda left, right: exp.union(left, right, distinct=False), selects)
        values.replace(exp.Subquery(this=union, alias=exp.TableAlias(this=alias.this.copy())))
    for table in list(statement.find_all(exp.Table)):
        if not isinstance(table.this, exp.ExplodingGenerateSeries):
            continue
        series = table.this
        name = table.alias
        selects = [exp.select(exp.alias_(exp.Literal.number(n), name))
                   for n in range(int(series.args["start"].this), int(series.args["end"].this) + 1)]
        union = reduce(lambda left, right: exp.union(left, right, distinct=False), selects)
        table.replace(exp.Subquery(this=union, alias=exp.TableAlias(this=exp.to_identifier(name))))
    return statement.sql(dialect="sqlite")


def execute_pg_script(connection, source):
    for statement in sqlglot.parse(source, read="postgres", error_level=ErrorLevel.RAISE):
        if statement is not None:
            connection.execute(sqlite_sql(statement))


def load_case(query_id, case):
    metadata, files = query_plan(query_id)
    spec = next(item for item in metadata["cases"] if item["name"] == case)
    connection = sqlite3.connect(":memory:")
    execute_pg_script(connection, files["schema.sql"].decode())
    execute_pg_script(connection, files[spec["data_file"]].decode())
    return connection, metadata, spec, files


def normalized_rows(query_id, rows):
    # Match the manifest's display encoding. Strict production comparison is
    # still performed by EquivalenceGate, not by this expectation formatter.
    result = [[value.rstrip() if isinstance(value, str) else value for value in row] for row in rows]
    if query_id == "rbot:tpch:query18_0":
        for row in result:
            row[4], row[5] = f"{row[4]:.2f}", f"{row[5]:.2f}"
    decimal_columns = {'rbot:dsb:query081_0': {13: 2, 15: 2},
                       'rbot:dsb:query014_0': {4: 2, 10: 2},
                       'rbot:dsb:query065_0': {2: 2, 3: 2, 4: 2},
                       'rbot:tpch:query19_0': {0: 4}}
    for row in result:
        for column, scale in decimal_columns.get(query_id, {}).items():
            if row[column] is not None:
                row[column] = f'{row[column]:.{scale}f}'
    return result


def original_rows(connection, query_id, files):
    tree = sqlglot.parse_one(files["original.sql"].decode(), read="postgres")
    return normalized_rows(query_id, connection.execute(sqlite_sql(tree)).fetchall())


@pytest.mark.parametrize("query_id", RECIPES)
def test_new_five_case_plan_executes_offline_with_exact_source_ddl_and_full_table_counts(query_id):
    fingerprints = set()
    metadata, files = query_plan(query_id)
    source = ROOT / "results/stage_a_sf10/inputs/schemas" / (RECIPES[query_id]["schema_sha256"] + ".sql")
    assert files["schema.sql"] == source.read_bytes()
    assert len(metadata["source_tables"]) == {'dsb': 25, 'tpch': 8, 'calcite': 6}[metadata['benchmark']]
    assert set(metadata["query_lane_tables"]) < set(metadata["source_tables"])
    assert metadata["validation"]["postgresql_executed"] is False
    assert metadata["validation"]["benchmark_executed"] is False
    assert metadata["historical_validation"]["status"] == ('new_recipe_no_historical_validation' if RECIPES[query_id].get('generated')
                                                               else 'historical_reference_only_not_rerun')
    assert metadata["materialization_contract"]["do_not_pass_plan_as_mini_dbs"] is True
    assert "coverage_complete" not in metadata
    assert [case["name"] for case in metadata["cases"]] == list(CASES)
    for case in CASES:
        connection, _, spec, emitted = load_case(query_id, case)
        try:
            actual_counts = {table: connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                             for table in metadata["source_tables"]}
            assert actual_counts == spec["planned_rows_by_table"]
            assert sum(actual_counts.values()) == spec["planned_rows_inserted"]
            assert all(actual_counts.values()) if case != "empty" else not any(actual_counts.values())
            assert sha256(emitted[spec["data_file"]]) == spec["data_sha256"]
            actual = original_rows(connection, query_id, emitted)
            assert actual == spec["expected_original_rows"]
            if case != 'empty' and actual and any(v is not None for row in actual for v in row):
                fingerprints.add(json.dumps(actual, sort_keys=True))
        finally:
            connection.close()
    if query_id == PROOF_QUERY:
        assert not fingerprints
        assert metadata['witness_status'] == 'proof_required'
        assert metadata['output_classification'] == 'tautological_empty'
        assert metadata['materialization_contract']['proof_required'] is True
    else:
        assert len(fingerprints) >= 2


@pytest.mark.parametrize("query_id", LEGACY_RECIPES)
def test_saved_candidates_match_each_new_case_in_offline_engine_only(query_id):
    witness = ROOT / "results/stage_a_followup" / RECIPES[query_id]["witness"]
    provenance = json.loads((witness / "provenance.json").read_text())
    for case in CASES:
        connection, _, _, files = load_case(query_id, case)
        try:
            expected = original_rows(connection, query_id, files)
            for candidate in provenance["candidates"]:
                text = (witness / candidate["file"]).read_text()
                assert sha256(text.rstrip().encode()) == candidate["sha256"]
                tree = sqlglot.parse_one(text, read="postgres")
                actual = normalized_rows(query_id, connection.execute(sqlite_sql(tree)).fetchall())
                assert actual == expected, (case, candidate["file"])
        finally:
            connection.close()


def test_dsb_null_price_is_excluded_and_repairing_it_activates_the_customer():
    query_id = "rbot:dsb:query010_0"
    connection, _, spec, files = load_case(query_id, "null")
    try:
        assert any(value is None for row in spec["expected_original_rows"] for value in row)
        assert connection.execute("SELECT ss_sales_price FROM store_sales WHERE ss_customer_sk=5").fetchone() == (None,)
        connection.execute("UPDATE store_sales SET ss_sales_price=82 WHERE ss_customer_sk=5")
        rows = original_rows(connection, query_id, files)
        assert rows[1][3] == 4
        connection.execute("UPDATE store_sales SET ss_customer_sk=NULL WHERE ss_customer_sk=5")
        assert original_rows(connection, query_id, files) == spec["expected_original_rows"]
    finally:
        connection.close()


@pytest.mark.parametrize("mutation", [
    "UPDATE customer_address SET ca_county='outside'",
    "UPDATE customer SET c_birth_month=6",
    "UPDATE customer_demographics SET cd_education_status='College'",
    "UPDATE date_dim SET d_year=2001",
    "UPDATE date_dim SET d_moy=8",
    "UPDATE item SET i_manager_id=101",
    "UPDATE item SET i_category='Home'",
    "UPDATE store_sales SET ss_sales_price=76",
    "UPDATE web_sales SET ws_sales_price=88; UPDATE catalog_sales SET cs_sales_price=88",
])
def test_dsb_recipe_requires_coordinated_county_customer_date_item_and_all_channel_ratios(mutation):
    query_id = "rbot:dsb:query010_0"
    connection, _, _, files = load_case(query_id, "duplicate")
    try:
        connection.executescript(mutation)
        assert original_rows(connection, query_id, files) == []
    finally:
        connection.close()


def test_dsb_both_or_branches_overlap_and_duplicates_preserve_exists_semantics():
    query_id = "rbot:dsb:query010_0"
    connection, _, spec, files = load_case(query_id, "duplicate")
    try:
        assert connection.execute("SELECT count(*) FROM store_sales WHERE ss_customer_sk=1").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM web_sales WHERE ws_bill_customer_sk=1").fetchone()[0] == 3
        assert original_rows(connection, query_id, files) == spec["expected_original_rows"]
        # C3 exists in both secondary channels, C1 only web, C2 only catalog.
        connection.execute("DELETE FROM web_sales")
        rows = original_rows(connection, query_id, files)
        assert len(rows) == 1 and rows[0][3] == 2
    finally:
        connection.close()


@pytest.mark.parametrize("price", [77, 87])
def test_dsb_every_channel_accepts_inclusive_ratio_bounds(price):
    query_id = "rbot:dsb:query010_0"
    connection, _, spec, files = load_case(query_id, "duplicate")
    try:
        for table, prefix in (("store_sales", "ss"), ("web_sales", "ws"), ("catalog_sales", "cs")):
            connection.execute(f"UPDATE {table} SET {prefix}_sales_price=?", (price,))
        assert original_rows(connection, query_id, files) == spec["expected_original_rows"]
    finally:
        connection.close()


@pytest.mark.parametrize("correct,incorrect", [
    ("c.c_customer_sk = ws_bill_customer_sk", "c.c_customer_sk = ws_ship_customer_sk"),
    ("c.c_customer_sk = cs_ship_customer_sk", "c.c_customer_sk = cs_bill_customer_sk"),
])
def test_dsb_bill_and_ship_roles_cannot_be_interchanged(correct, incorrect):
    query_id = "rbot:dsb:query010_0"
    connection, _, spec, files = load_case(query_id, "duplicate")
    try:
        original = files["original.sql"].decode()
        assert correct in original
        mutant = sqlglot.parse_one(original.replace(correct, incorrect), read="postgres")
        assert normalized_rows(query_id, connection.execute(sqlite_sql(mutant)).fetchall()) != spec["expected_original_rows"]
    finally:
        connection.close()


def test_tpch_threshold_depends_on_multirow_sum_and_nullable_query_values_are_illegal():
    query_id = "rbot:tpch:query18_0"
    connection, metadata, _, files = load_case(query_id, "null")
    try:
        assert metadata["query_lane_nullable_columns"] == []
        assert connection.execute("SELECT max(l_quantity) FROM lineitem").fetchone()[0] == 50
        totals = dict(connection.execute("SELECT l_orderkey,sum(l_quantity) FROM lineitem GROUP BY l_orderkey"))
        assert {key: totals[key] for key in (103, 104, 105)} == {103: 312, 104: 313, 105: 313.01}
        assert connection.execute("SELECT n_comment FROM nation").fetchone() == (None,)
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            connection.execute("UPDATE lineitem SET l_quantity=NULL")
        # Replacing strict >313 with >=313 would wrongly include order104.
        original = files["original.sql"].decode()
        changed = sqlglot.parse_one(original.replace("> 313", ">= 313"), read="postgres")
        rows = normalized_rows(query_id, connection.execute(sqlite_sql(changed)).fetchall())
        assert 104 in {row[2] for row in rows}
        assert rows != original_rows(connection, query_id, files)
    finally:
        connection.close()


def test_tpch_duplicate_case_retains_parent_multiplicity_and_duplicate_line_threshold_crossing():
    query_id = "rbot:tpch:query18_0"
    connection, _, _, files = load_case(query_id, "duplicate")
    try:
        rows = {row[2]: row[5] for row in original_rows(connection, query_id, files)}
        assert rows[101] == "1256.00"  # 314 * 2 identical customers * 2 orders.
        assert rows[102] == "630.00"   # 315 * 2 identical customers.
        assert rows[103] == "362.00"   # 312 + exact duplicate 50-quantity line.
        line_counts = Counter(connection.execute("SELECT * FROM lineitem").fetchall())
        assert any(count == 2 for count in line_counts.values())
    finally:
        connection.close()


@pytest.mark.parametrize("query_id", LEGACY_RECIPES)
def test_unmatched_case_keeps_coverage_but_orphans_do_not_contribute(query_id):
    connection, _, spec, files = load_case(query_id, "unmatched")
    try:
        assert original_rows(connection, query_id, files) == spec["expected_original_rows"]
        if query_id == "rbot:tpch:query18_0":
            assert connection.execute("SELECT count(*) FROM lineitem WHERE l_orderkey=888").fetchone()[0] == 7
            connection.execute("DELETE FROM lineitem WHERE l_orderkey IN (888,999,106)")
        else:
            assert connection.execute("SELECT count(*) FROM store_sales WHERE ss_sold_date_sk=999").fetchone()[0] == 2
            connection.execute("DELETE FROM store_sales WHERE ss_sold_date_sk=999")
            connection.execute("DELETE FROM web_sales WHERE ws_item_sk=999")
            connection.execute("DELETE FROM catalog_sales WHERE cs_sold_date_sk=999")
        assert original_rows(connection, query_id, files) == spec["expected_original_rows"]
    finally:
        connection.close()


def test_writer_is_deterministic_refuses_overwrite_and_never_marks_plan_ready(tmp_path):
    left, right = tmp_path / "left-v2", tmp_path / "right-v2"
    manifest = build_plan(left)
    assert manifest == build_plan(right)
    left_files = {str(path.relative_to(left)): path.read_bytes() for path in left.rglob("*") if path.is_file()}
    right_files = {str(path.relative_to(right)): path.read_bytes() for path in right.rglob("*") if path.is_file()}
    assert left_files == right_files
    assert manifest["status"] == "offline_plan_not_postgresql_validated"
    with pytest.raises(FileExistsError):
        build_plan(left)
    for metadata_path in left.glob("*/manifest.json"):
        metadata = json.loads(metadata_path.read_text())
        assert not any("coverage_complete" in case for case in metadata["cases"])
        for historical in metadata["historical_inputs"]:
            assert sha256((ROOT / historical["path"]).read_bytes()) == historical["sha256"]
    with pytest.raises(ValueError, match="distinct"):
        build_plan(tmp_path / "duplicate", ["rbot:dsb:query010_0"] * 2)
    with pytest.raises(ValueError, match="No pinned recipe"):
        build_plan(tmp_path / "unknown", ["unknown"])
    assert not (tmp_path / "duplicate").exists() and not (tmp_path / "unknown").exists()


@pytest.mark.parametrize("changed,reason", [
    ("schema", "source schema hash mismatch"),
    ("original.sql", "query/provenance mismatch"),
    ("data.sql", "witness data hash mismatch"),
])
def test_pinned_input_drift_fails_before_creating_output(tmp_path, monkeypatch, changed, reason):
    recipe = RECIPES["rbot:dsb:query010_0"]
    relative_schema = "results/stage_a_sf10/inputs/schemas/" + recipe["schema_sha256"] + ".sql"
    relative_witness = "results/stage_a_followup/" + recipe["witness"] + "/"
    inputs = [relative_schema] + [relative_witness + name for name in ("original.sql", "data.sql", "provenance.json")]
    fake_root = tmp_path / "input-copy"
    for relative in inputs:
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    changed_path = fake_root / (relative_schema if changed == "schema" else relative_witness + changed)
    changed_path.write_bytes(changed_path.read_bytes() + b"\n-- changed copied input\n")
    monkeypatch.setattr(builder, "ROOT", fake_root)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=reason):
        builder.build_plan(output, ["rbot:dsb:query010_0"])
    assert not output.exists()


@pytest.mark.parametrize('query_id', QUERY_HASHES)
def test_generated_original_is_exact_checkpoint_sql_and_hash(query_id):
    records = [json.loads(line) for line in (ROOT / 'benchmarks/stage_a_30.jsonl').read_text().splitlines()]
    checkpoint = next(record for record in records if record['id'] == query_id)
    metadata, files = query_plan(query_id)
    assert files['original.sql'] == checkpoint['sql'].encode()
    assert metadata['original_sha256'] == checkpoint['sha256'] == QUERY_HASHES[query_id]
    assert metadata['original_checkpoint_sha256'] == checkpoint['sha256']
    assert not metadata['validation']['postgresql_executed']
    # No runtime benchmark or synthetic-performance claims appear in a plan.
    assert not metadata['validation']['benchmark_executed']


@pytest.mark.parametrize('query_id,mutation', [
    ('rbot:dsb:query081_0', 'UPDATE date_dim SET d_year=2001'),
    ('rbot:dsb:query081_0', "UPDATE customer_address SET ca_state='NY' WHERE ca_address_sk=1"),
    ('rbot:dsb:query038_0', 'DELETE FROM store_sales'),
    ('rbot:dsb:query038_0', 'DELETE FROM catalog_sales'),
    ('rbot:dsb:query038_0', 'DELETE FROM web_sales'),
    ('rbot:dsb:query038_0', 'UPDATE customer SET c_birth_month=1'),
    ('rbot:dsb:query038_0', 'UPDATE store_sales SET ss_wholesale_cost=72'),
    ('rbot:dsb:query069_0', 'DELETE FROM web_sales'),
    ('rbot:dsb:query069_0', 'DELETE FROM catalog_sales'),
    ('rbot:dsb:query069_0', 'UPDATE store_sales SET ss_list_price=138'),
    ('rbot:dsb:query069_0', "UPDATE customer_address SET ca_state='MI'"),
    ('rbot:dsb:query014_0', 'UPDATE web_sales SET ws_wholesale_cost=72'),
    ('rbot:dsb:query014_0', 'UPDATE catalog_sales SET cs_wholesale_cost=94'),
    ('rbot:dsb:query014_0', 'UPDATE date_dim SET d_dom=12 WHERE d_year=1999'),
    ('rbot:dsb:query014_0', "UPDATE item SET i_category='Home'"),
    ('rbot:dsb:query065_0', 'UPDATE store_sales SET ss_list_price=100 WHERE ss_item_sk=1'),
    ('rbot:dsb:query065_0', 'UPDATE date_dim SET d_month_seq=1206'),
    ('rbot:dsb:query065_0', 'UPDATE item SET i_manager_id=53'),
    ('rbot:dsb:query087_0', 'DELETE FROM catalog_sales'),
    ('rbot:dsb:query087_0', 'DELETE FROM web_sales'),
    ('rbot:dsb:query087_0', 'UPDATE customer SET c_birth_year=1971'),
    ('rbot:dsb:query100_0', "UPDATE item SET i_category='Home' WHERE i_item_sk=1"),
    ('rbot:dsb:query100_0', 'UPDATE item SET i_manager_id=47 WHERE i_item_sk=2'),
    ('rbot:dsb:query100_0', "UPDATE customer_demographics SET cd_marital_status='S'"),
    ('rbot:dsb:query100_0', 'UPDATE store_sales SET ss_list_price=79 WHERE ss_item_sk=1'),
    ('rbot:dsb:query100_0', 'UPDATE store_sales SET ss_list_price=95 WHERE ss_item_sk=2'),
    ('rbot:calcite:testDecorrelateTwoExists', 'UPDATE emp SET sal=35 WHERE empno=2'),
    ('rbot:tpch:query12_0', "UPDATE lineitem SET l_shipmode='TRUCK' WHERE l_shipmode='RAIL'"),
    ('rbot:tpch:query12_0', "UPDATE lineitem SET l_shipdate=l_commitdate WHERE l_shipmode='RAIL'"),
    ('rbot:tpch:query19_0', "UPDATE part SET p_brand='Brand#00' WHERE p_partkey=1"),
    ('rbot:tpch:query19_0', "UPDATE part SET p_container='SM CASE' WHERE p_partkey=2"),
    ('rbot:tpch:query19_0', 'UPDATE part SET p_size=16 WHERE p_partkey=3'),
    ('rbot:tpch:query19_0', "UPDATE lineitem SET l_shipinstruct='NONE'"),
])
def test_new_recipes_detect_predicate_and_join_mutations_offline(query_id, mutation):
    connection, _, spec, files = load_case(query_id, 'duplicate')
    try:
        before = original_rows(connection, query_id, files)
        assert before == spec['expected_original_rows']
        connection.executescript(mutation)
        assert original_rows(connection, query_id, files) != before
    finally:
        connection.close()


@pytest.mark.parametrize('query_id,correct,incorrect', [
    ('rbot:dsb:query069_0', 'c.c_customer_sk = ws_bill_customer_sk', 'c.c_customer_sk = ws_ship_customer_sk'),
    ('rbot:dsb:query069_0', 'c.c_customer_sk = cs_ship_customer_sk', 'c.c_customer_sk = cs_bill_customer_sk'),
    ('rbot:dsb:query081_0', 'where ctr1.ctr_state = ctr2.ctr_state', ''),
    ('rbot:dsb:query014_0', 'having sum(ss_quantity*ss_list_price) >', 'having sum(ss_quantity*ss_list_price) <'),
    ('rbot:dsb:query065_0', 'sc.revenue <= 0.1 * sb.ave', 'sc.revenue > 0.1 * sb.ave'),
    ('rbot:tpch:query12_0', "l_receiptdate < date '1995-01-01'", "l_receiptdate <= date '1995-01-01'"),
    ('rbot:tpch:query19_0', 'l_quantity >= 5', 'l_quantity > 5'),
    ('rbot:tpch:query19_0', 'l_quantity <= 19 + 10', 'l_quantity < 19 + 10'),
    ('rbot:tpch:query19_0', 'l_quantity <= 29 + 10', 'l_quantity < 29 + 10'),
])
def test_new_recipes_detect_incorrect_sql_variants(query_id, correct, incorrect):
    connection, _, spec, files = load_case(query_id, 'duplicate')
    try:
        original = files['original.sql'].decode()
        assert correct in original
        mutant = sqlglot.parse_one(original.replace(correct, incorrect), read='postgres')
        assert normalized_rows(query_id, connection.execute(sqlite_sql(mutant)).fetchall()) != spec['expected_original_rows']
    finally:
        connection.close()


def test_return_average_is_correlated_and_strict_and_null_totals_do_not_participate():
    query_id = 'rbot:dsb:query081_0'
    connection, _, spec, files = load_case(query_id, 'null')
    try:
        assert connection.execute('SELECT cr_return_amt_inc_tax FROM catalog_returns WHERE cr_returning_customer_sk=5').fetchone() == (None,)
        assert original_rows(connection, query_id, files) == spec['expected_original_rows']
        # MI group totals 150 and 100: the high total equals 1.2 * their mean.
        connection.execute('UPDATE catalog_returns SET cr_return_amt_inc_tax=150 WHERE cr_returning_customer_sk=1')
        assert original_rows(connection, query_id, files) == []
        mutant = sqlglot.parse_one(files['original.sql'].decode().replace('ctr1.ctr_total_return >', 'ctr1.ctr_total_return >='), read='postgres')
        assert connection.execute(sqlite_sql(mutant)).fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize('query_id', ['rbot:dsb:query038_0', 'rbot:dsb:query087_0'])
def test_null_set_payloads_remain_shared_across_channels_and_not_null_join_comparisons(query_id):
    connection, _, spec, files = load_case(query_id, 'null')
    try:
        assert connection.execute('SELECT count(*) FROM customer WHERE c_last_name IS NULL').fetchone()[0] >= 2
        assert original_rows(connection, query_id, files) == spec['expected_original_rows']
        # These are set operations: matching tuple NULLs are equal for duplicate
        # elimination, unlike a rewrite using ordinary column equality joins.
        branch = '''SELECT c_last_name,c_first_name,d_date FROM store_sales
                    JOIN customer ON ss_customer_sk=c_customer_sk
                    JOIN date_dim ON ss_sold_date_sk=d_date_sk'''
        store_payloads = set(connection.execute(branch))
        catalog_payloads = set(connection.execute(branch.replace('store_sales','catalog_sales').replace('ss_customer_sk','cs_bill_customer_sk').replace('ss_sold_date_sk','cs_sold_date_sk')))
        web_payloads = set(connection.execute(branch.replace('store_sales','web_sales').replace('ss_customer_sk','ws_bill_customer_sk').replace('ss_sold_date_sk','ws_sold_date_sk')))
        actual = store_payloads & catalog_payloads & web_payloads if '038' in query_id else store_payloads - catalog_payloads - web_payloads
        assert [[len(actual)]] == spec['expected_original_rows']
    finally:
        connection.close()


def test_calcite_exists_self_match_does_not_require_a_department_row_and_same_job_is_blocked():
    query_id = 'rbot:calcite:testDecorrelateTwoExists'
    connection, _, spec, files = load_case(query_id, 'unmatched')
    try:
        assert connection.execute('SELECT count(*) FROM dept WHERE deptno=999').fetchone() == (0,)
        assert original_rows(connection, query_id, files) == spec['expected_original_rows']
        assert 4 in {row[0] for row in spec['expected_original_rows']}
        assert 3 not in {row[0] for row in spec['expected_original_rows']}
        connection.execute('UPDATE emp SET sal=35 WHERE empno=2')
        assert {row[0] for row in original_rows(connection, query_id, files)} == {1, 2, 3, 4}
    finally:
        connection.close()


def test_calcite_window_null_predicate_is_explicitly_proof_required_even_with_all_tables_populated():
    metadata, files = query_plan(PROOF_QUERY)
    assert metadata['witness_status'] == 'proof_required'
    assert metadata['output_classification'] == 'tautological_empty'
    assert 'never NULL' in metadata['structural_empty_reason']
    assert not any('two distinct' in item for item in metadata['materialization_contract']['required_before_ready'])
    assert any('structural proof' in item for item in metadata['materialization_contract']['required_before_ready'])
    for name in CASES[1:]:
        connection, _, spec, emitted = load_case(PROOF_QUERY, name)
        try:
            assert all(spec['planned_rows_by_table'].values())
            assert original_rows(connection, PROOF_QUERY, emitted) == []
            # Removing only the impossible outer filter exposes positive counts.
            unfiltered = files['original.sql'].decode().replace('where w_count is null', '')
            rows = connection.execute(sqlite_sql(sqlglot.parse_one(unfiltered, read='postgres'))).fetchall()
            assert rows and all(row[2] >= 1 for row in rows)
            with pytest.raises(sqlite3.IntegrityError, match='NOT NULL'):
                connection.execute('UPDATE emp SET deptno=NULL')
        finally:
            connection.close()


def test_generated_query_input_drift_fails_before_creating_output(tmp_path, monkeypatch):
    query_id = 'rbot:dsb:query038_0'
    root = tmp_path / 'copied-inputs'
    for path in ('benchmarks/stage_a_30.jsonl', 'results/stage_a_sf10/inputs/schemas/' + RECIPES[query_id]['schema_sha256'] + '.sql'):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    checkpoint = root / 'benchmarks/stage_a_30.jsonl'
    records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    next(record for record in records if record['id'] == query_id)['sql'] += '\n-- drift'
    checkpoint.write_text('\n'.join(json.dumps(record) for record in records))
    monkeypatch.setattr(builder, 'ROOT', root)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='checkpoint query hash mismatch'):
        builder.build_plan(output, [query_id])
    assert not output.exists()
