#!/usr/bin/env python3
"""Emit deterministic query-specific PostgreSQL fixture plans without DB access.

These plans combine pinned historical witness inputs and explicit compact row
recipes. They are five-case, full-source-schema plans, not executed miniature specifications.
The materializer must check the live catalog, counts and exact query outputs
before producing coverage_complete=True specs for EquivalenceGate.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import sqlglot
from sqlglot import ErrorLevel, exp

try:
    from scripts.query_fixture_recipes_v2 import PROOF_QUERY, case_rows, generated_recipes
except ModuleNotFoundError:  # Direct invocation as scripts/build_query_witness_fixtures_v2.py.
    from query_fixture_recipes_v2 import PROOF_QUERY, case_rows, generated_recipes

ROOT = Path(__file__).resolve().parents[1]
VERSION = "query-witness-v2"
CASES = ("empty", "null", "duplicate", "skew", "unmatched")
RECIPES = {
    "rbot:dsb:query010_0": {
        "benchmark": "dsb", "directory": "dsb_query010_0", "witness": "dsb_witness",
        "schema_sha256": "805a1b08edf45fee0326efca6056ca881d92a00969e102fe731cf0781d24c0ea",
        "base_sha256": "dcc268c3569193a5a9ab4fdd1b72549329b7ec0001ea36f9a13e6b372bf17f93",
        "base_counts": {"customer_address": 1, "customer_demographics": 2,
                        "customer": 4, "date_dim": 2, "item": 3,
                        "store_sales": 5, "web_sales": 4, "catalog_sales": 2},
    },
    "rbot:tpch:query18_0": {
        "benchmark": "tpch", "directory": "tpch_query18_0", "witness": "tpch18_witness",
        "schema_sha256": "4fc33b97f09b573d748179f6a0a4d049f8d5057c7919ce2c163bb80e1af23eef",
        "base_sha256": "869946c64d7c2a8a216ff9f11671a866ed43b37f00d53e44ab8460cf838259a2",
        "base_counts": {"customer": 2, "orders": 6, "lineitem": 49},
    },
}
RECIPES.update(generated_recipes())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _table_definitions(source: str) -> OrderedDict:
    tables = OrderedDict()
    for statement in sqlglot.parse(source, read="postgres", error_level=ErrorLevel.RAISE):
        if not isinstance(statement, exp.Create) or statement.kind != "TABLE" or not isinstance(statement.this, exp.Schema):
            raise ValueError("Recipe source must contain only exact CREATE TABLE DDL")
        table = statement.this.this
        if table.db or table.catalog:
            raise ValueError("Qualified source table is unsupported")
        name = table.name.lower()
        if name in tables:
            raise ValueError(f"Duplicate source table {name}")
        tables[name] = statement
    return tables


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _inert_dimension_rows(tables: dict, query_tables: set[str]) -> str:
    """One source-valid row per unqueried table, never injected into the lane.

    The pinned schemas have no foreign keys or CHECK constraints. Do not infer
    new relationships or remove authoritative source constraints to fill tables.
    """
    if any(t.find(exp.ForeignKey) or t.find(exp.Reference) or t.find(exp.CheckColumnConstraint)
           for t in tables.values()):
        raise ValueError("Pinned recipes require schemas without FK/CHECK constraints")
    statements = []
    for table, statement in tables.items():
        if table in query_tables:
            continue
        primary = {c.name.lower() for key in statement.find_all(exp.PrimaryKey) for c in key.expressions}
        columns, values = [], []
        for column in statement.this.expressions:
            if not isinstance(column, exp.ColumnDef):
                continue
            name = column.name.lower()
            required = name in primary or column.find(exp.NotNullColumnConstraint) or column.find(exp.PrimaryKeyColumnConstraint)
            kind = column.args["kind"].this.value
            if not required:
                value = "NULL"
            elif kind in {"INT", "BIGINT", "SMALLINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL"}:
                value = "1"
            elif kind in {"CHAR", "VARCHAR", "TEXT"}:
                value = "'X'"
            elif kind == "DATE":
                value = "DATE '2002-01-01'"
            elif kind == "BOOLEAN":
                value = "FALSE"
            else:
                raise ValueError(f"Unsupported required pinned column type: {table}.{name} {kind}")
            columns.append(_quote(name))
            values.append(value)
        statements.append(f"INSERT INTO {_quote(table)} ({', '.join(columns)}) VALUES ({', '.join(values)});")
    return "-- Inert full-schema coverage rows; these tables are absent from the exact query.\n" + "\n".join(statements) + "\n"


def _dsb_group(marital: str, count: int) -> list:
    if marital == "M":
        return ["F", "M", "Unknown", count, None, count, None, count,
                None, count, None, count, None, count]
    return ["F", "S", "4 yr Degree", count, 1200, count, "Good", count,
            2, count, 1, count, 0, count]


def _case_recipe(query_id: str, case: str) -> tuple[str, dict, list, list[str]]:
    if case == "empty":
        return "", {}, [], ["all_source_tables_empty"]
    if query_id == "rbot:dsb:query010_0":
        expected = [_dsb_group("M", 1), _dsb_group("S", 3)]
        if case == "null":
            return """-- C5 has legal joins but a NULL store numerator, so must not qualify.
INSERT INTO customer (c_customer_sk,c_customer_id,c_current_cdemo_sk,c_current_addr_sk,c_birth_month)
VALUES (5,'customer5',1,1,7);
INSERT INTO store_sales (ss_sold_date_sk,ss_item_sk,ss_customer_sk,ss_ticket_number,ss_sales_price,ss_list_price)
VALUES (1,1,5,501,NULL,100);
INSERT INTO web_sales (ws_sold_date_sk,ws_item_sk,ws_bill_customer_sk,ws_ship_customer_sk,ws_order_number,ws_sales_price,ws_list_price)
VALUES (1,1,5,9005,501,82,100);
""", {"customer": 1, "store_sales": 1, "web_sales": 1}, expected, [
                "null_output_group", "null_store_price_excluded", "coordinated_county_dates_customer_items_ratios"]
        if case == "duplicate":
            return """-- Distinct source primary keys, duplicate payloads read by EXISTS.
INSERT INTO store_sales (ss_sold_date_sk,ss_item_sk,ss_customer_sk,ss_ticket_number,ss_sales_price,ss_list_price)
VALUES (1,1,1,9999,77,100);
INSERT INTO web_sales (ws_sold_date_sk,ws_item_sk,ws_bill_customer_sk,ws_ship_customer_sk,ws_order_number,ws_sales_price,ws_list_price)
VALUES (1,1,1,9001,9999,82,100);
""", {"store_sales": 1, "web_sales": 1}, expected, [
                "duplicate_exists_payloads_do_not_multiply_customers", "web_catalog_or_overlap"]
        if case == "skew":
            return "UPDATE customer SET c_current_cdemo_sk=1 WHERE c_customer_sk=4;\n", {}, [
                _dsb_group("S", 4)], ["four_customers_in_one_demographic_group", "distinct_nonempty_output"]
        return """-- Orphan keys deliberately affect the queried joins, without FK violations.
UPDATE store_sales SET ss_sold_date_sk=999 WHERE ss_customer_sk=1;
UPDATE web_sales SET ws_item_sk=999 WHERE ws_bill_customer_sk=3;
UPDATE catalog_sales SET cs_sold_date_sk=999 WHERE cs_ship_customer_sk=3;
""", {}, [_dsb_group("M", 1), _dsb_group("S", 1)], [
            "unmatched_store_date", "unmatched_web_item", "unmatched_catalog_date"]
    base = [["Alpha", 1, 102, "2001-02-02", "900.00", "315.00"],
            ["Alpha", 1, 101, "2001-03-03", "900.00", "314.00"],
            ["Beta", 2, 105, "2001-01-01", "800.00", "313.01"]]
    if case == "null":
        return "-- All query columns are source NOT NULL; legal NULLs remain in unqueried dimensions.\n", {}, base, [
            "nullable_source_dimension_columns", "query_column_nulls_prohibited_by_source_ddl",
            "aggregate_totals_312_313_313_01_314_315", "no_individual_line_above_threshold"]
    if case == "duplicate":
        return """-- The source permits exact duplicate parents and lineitems.
INSERT INTO customer SELECT * FROM customer WHERE c_custkey=1;
INSERT INTO orders SELECT * FROM orders WHERE o_orderkey=101;
INSERT INTO lineitem SELECT * FROM lineitem WHERE l_orderkey=103 AND l_linenumber=1;
""", {"customer": 1, "orders": 1, "lineitem": 1}, [
            ["Beta", 2, 103, "2001-04-04", "1000.00", "362.00"],
            ["Alpha", 1, 102, "2001-02-02", "900.00", "630.00"],
            ["Alpha", 1, 101, "2001-03-03", "900.00", "1256.00"], base[2]], [
            "duplicate_line_crosses_strict_threshold", "duplicate_parent_join_multiplicity"]
    if case == "skew":
        return "UPDATE lineitem SET l_quantity=13 WHERE l_orderkey=101 AND l_linenumber=7;\n", {}, [base[0], base[2]], [
            "aggregate_equal_313_excluded", "aggregate_313_01_included", "distinct_nonempty_output"]
    return "UPDATE lineitem SET l_orderkey=888 WHERE l_orderkey=101;\n", {}, [base[0], base[2]], [
        "qualifying_lines_with_missing_order", "order_with_missing_customer", "unmatched_join_exclusion"]


def _render_rows(tables: dict, rows: dict) -> str:
    """Complete partial generated rows using exact DDL nullability and types."""
    statements = []
    for table, data in rows.items():
        definitions = {c.name.lower(): c for c in tables[table].this.expressions if isinstance(c, exp.ColumnDef)}
        primary = {c.name.lower() for key in tables[table].find_all(exp.PrimaryKey) for c in key.expressions}
        for row in data:
            if set(row) - definitions.keys():
                raise ValueError(f"Unknown recipe columns in {table}: {sorted(set(row) - definitions.keys())}")
            values = []
            for name, column in definitions.items():
                required = name in primary or column.find(exp.NotNullColumnConstraint) or column.find(exp.PrimaryKeyColumnConstraint)
                kind = column.args["kind"].this.value
                value = row.get(name)
                if name not in row and required:
                    value = (1 if kind in {"INT", "BIGINT", "SMALLINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL"}
                             else False if kind == "BOOLEAN" else '2000-01-01' if kind == "DATE" else 'X')
                if value is None and required:
                    raise ValueError(f"Illegal NULL for required source column {table}.{name}")
                if value is None:
                    literal = 'NULL'
                elif isinstance(value, bool):
                    literal = 'TRUE' if value else 'FALSE'
                elif kind in {"INT", "BIGINT", "SMALLINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL"}:
                    literal = str(value)
                    # Never interpret a generated numeric field as arbitrary SQL.
                    from decimal import Decimal
                    Decimal(literal)
                else:
                    literal = "'" + str(value).replace("'", "''") + "'"
                    if kind == "DATE":
                        literal = "DATE " + literal
                values.append(literal)
            statements.append(f"INSERT INTO {_quote(table)} ({', '.join(map(_quote, definitions))}) VALUES ({', '.join(values)});")
    return '\n'.join(statements) + '\n'


def _generated_query_plan(query_id: str) -> tuple[dict, dict[str, bytes]]:
    recipe = RECIPES[query_id]
    benchmark_path = ROOT / 'benchmarks/stage_a_30.jsonl'
    records = [json.loads(line) for line in benchmark_path.read_text().splitlines() if line.strip()]
    selected = [record for record in records if record['id'] == query_id]
    if len(selected) != 1:
        raise ValueError('Pinned query must occur exactly once in Stage A checkpoint')
    record = selected[0]
    original = record['sql'].encode()
    source = ROOT / 'results/stage_a_sf10/inputs/schemas' / (recipe['schema_sha256'] + '.sql')
    schema = source.read_bytes()
    if sha256(schema) != recipe['schema_sha256'] or record['schema_sha256'] != sha256(schema):
        raise ValueError('Pinned exact source schema hash mismatch')
    if sha256(original) != recipe['query_sha256'] or record['sha256'] != sha256(original):
        raise ValueError('Pinned exact checkpoint query hash mismatch')
    tables = _table_definitions(schema.decode())
    base_rows, _, _ = case_rows(query_id, 'null')
    query_tables = set(base_rows)
    complement = _inert_dimension_rows(tables, query_tables)
    files = {'schema.sql': schema, 'original.sql': original}
    cases = []
    for name in CASES:
        rows, expected, features = case_rows(query_id, name)
        data = ('-- Empty case: execute exact schema.sql only.\n' if name == 'empty' else
                '-- New deterministic v2 rows, completed using the authoritative source DDL.\n'
                + _render_rows(tables, rows) + '\n' + complement)
        counts = {table: (0 if name == 'empty' else len(rows.get(table, [])) if table in query_tables else 1)
                  for table in tables}
        path = f'cases/{name}.sql'
        files[path] = data.encode()
        cases.append({'name': name, 'data_file': path, 'data_sha256': sha256(files[path]),
                      'planned_rows_by_table': counts, 'planned_rows_inserted': sum(counts.values()),
                      'expected_original_rows': expected, 'features': features,
                      'planned_all_tables_populated': name == 'empty' or all(counts.values())})
    proof = query_id == PROOF_QUERY
    requirements = ['exact source catalog types and constraints', 'measured rows_by_table for every source table',
                    'every nonempty case populates every declared source table',
                    'exact original ordered rows match expected_original_rows',
                    ('structural proof of tautological_empty; finite empty results do not establish equivalence' if proof else
                     'at least two distinct nonempty original outputs with non-NULL values'),
                    'saved candidate differential checks using the strict verifier']
    nullable = []
    for table in sorted(query_tables):
        primary = {c.name.lower() for key in tables[table].find_all(exp.PrimaryKey) for c in key.expressions}
        nullable.extend(f'{table}.{c.name.lower()}' for c in tables[table].this.expressions
                        if isinstance(c, exp.ColumnDef) and c.name.lower() not in primary
                        and not c.find(exp.NotNullColumnConstraint) and not c.find(exp.PrimaryKeyColumnConstraint))
    metadata = {
        'version': VERSION, 'query_id': query_id, 'benchmark': recipe['benchmark'],
        'suggested_primary_case': 'duplicate', 'schema_file': 'schema.sql',
        'source_schema_sha256': sha256(schema), 'original_file': 'original.sql',
        'original_sha256': sha256(original), 'original_checkpoint_sha256': record['sha256'],
        'source_tables': list(tables), 'query_lane_tables': sorted(query_tables),
        'coverage_only_tables': sorted(set(tables) - query_tables), 'query_lane_nullable_columns': nullable,
        'historical_inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': sha256(path.read_bytes())}
                              for path in (benchmark_path, source)],
        'historical_validation': {'status': 'new_recipe_no_historical_validation'},
        'expected_output_encoding': {'char_output_trailing_spaces': 'strip for expectation display only',
                                     'decimal_values': 'PostgreSQL expression scale strings', 'date_values': 'ISO dates',
                                     'candidate_comparison': 'unchanged strict verifier; display formatting is not an equivalence relaxation'},
        'cases': cases, 'validation': {'status': 'plan_only', 'postgresql_executed': False,
                                     'benchmark_executed': False,
                                     'expected_outputs_are': 'new recipe expectations, not measured PostgreSQL results'},
        'witness_status': 'proof_required' if proof else 'materialization_required',
        'output_classification': 'tautological_empty' if proof else 'varied_nonempty_planned',
        'materialization_contract': {'required_case_names': list(CASES),
                                     'schema_search_path': 'owned schema, pg_catalog only',
                                     'required_before_ready': requirements, 'coverage_complete_must_be_measured': True,
                                     'do_not_pass_plan_as_mini_dbs': True, 'proof_required': proof},
    }
    if proof:
        metadata['structural_empty_reason'] = ('COUNT(empno) OVER w is never NULL for any input row; '
                                              'WHERE w_count IS NULL therefore rejects every row. '
                                              'The source additionally declares emp.empno PRIMARY KEY.')
    return metadata, files


def query_plan(query_id: str) -> tuple[dict, dict[str, bytes]]:
    """Return manifest metadata and relative files; read pinned inputs only."""
    if query_id not in RECIPES:
        raise ValueError(f"No pinned recipe for {query_id}")
    recipe = RECIPES[query_id]
    if recipe.get('generated'):
        return _generated_query_plan(query_id)
    witness = ROOT / "results/stage_a_followup" / recipe["witness"]
    source = ROOT / "results/stage_a_sf10/inputs/schemas" / (recipe["schema_sha256"] + ".sql")
    schema = source.read_bytes()
    provenance = json.loads((witness / "provenance.json").read_text())
    original = (witness / "original.sql").read_bytes()
    base = (witness / "data.sql").read_bytes()
    if sha256(schema) != recipe["schema_sha256"] or provenance["source_schema_sha256"] != sha256(schema):
        raise ValueError("Pinned exact source schema hash mismatch")
    # Historical files append a newline to the checkpoint SQL text.
    if provenance["query_id"] != query_id or sha256(original.rstrip()) != provenance["original_sha256"]:
        raise ValueError("Pinned original query/provenance mismatch")
    if sha256(base) != recipe["base_sha256"]:
        raise ValueError("Pinned historical witness data hash mismatch")
    tables = _table_definitions(schema.decode())
    query_tables = set(recipe["base_counts"])
    if not query_tables <= tables.keys():
        raise ValueError("Historical witness tables are not a subset of source tables")
    files = {"schema.sql": schema, "original.sql": original, "historical_base.sql": base}
    complement = _inert_dimension_rows(tables, query_tables)
    planned_cases = []
    for case in CASES:
        delta, additions, expected, features = _case_recipe(query_id, case)
        counts = {t: (0 if case == "empty" else recipe["base_counts"].get(t, 1) + additions.get(t, 0)) for t in tables}
        data = ("-- Empty case: execute exact schema.sql only.\n" if case == "empty" else
                "-- New v2 case: historical data input, full-schema coverage rows, then query-specific delta.\n"
                + base.decode() + "\n" + complement + "\n" + delta)
        path = f"cases/{case}.sql"
        files[path] = data.encode()
        planned_cases.append({"name": case, "data_file": path, "data_sha256": sha256(files[path]),
                              "planned_rows_by_table": counts, "planned_rows_inserted": sum(counts.values()),
                              "expected_original_rows": expected, "features": features,
                              "planned_all_tables_populated": case == "empty" or all(counts.values())})
    nullable_query_columns = []
    for table in sorted(query_tables):
        statement = tables[table]
        primary = {c.name.lower() for key in statement.find_all(exp.PrimaryKey) for c in key.expressions}
        for column in statement.this.expressions:
            if (isinstance(column, exp.ColumnDef) and column.name.lower() not in primary
                    and not column.find(exp.NotNullColumnConstraint) and not column.find(exp.PrimaryKeyColumnConstraint)):
                nullable_query_columns.append(f"{table}.{column.name.lower()}")
    metadata = {
        "version": VERSION, "query_id": query_id, "benchmark": recipe["benchmark"],
        "suggested_primary_case": "duplicate",
        "schema_file": "schema.sql", "source_schema_sha256": sha256(schema),
        "original_file": "original.sql", "original_sha256": sha256(original),
        "original_checkpoint_sha256": provenance["original_sha256"],
        "expected_output_encoding": {"char_output_trailing_spaces": "strip for expectation display only",
                                     "decimal_values": "fixed source scale strings", "date_values": "ISO dates",
                                     "candidate_comparison": "unchanged strict verifier; expectation formatting is not an equivalence relaxation"},
        "source_tables": list(tables), "query_lane_tables": sorted(query_tables),
        "coverage_only_tables": sorted(set(tables) - query_tables),
        "query_lane_nullable_columns": nullable_query_columns,
        "historical_inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes())}
                              for path in (source, witness / "data.sql", witness / "original.sql", witness / "provenance.json")],
        "historical_validation": {"path": str((witness / "validation.json").relative_to(ROOT)),
                                  "status": "historical_reference_only_not_rerun"},
        "cases": planned_cases,
        "validation": {"status": "plan_only", "postgresql_executed": False,
                       "benchmark_executed": False,
                       "expected_outputs_are": "new recipe expectations, not measured PostgreSQL results"},
        "materialization_contract": {
            "required_case_names": list(CASES), "schema_search_path": "owned schema, pg_catalog only",
            "required_before_ready": ["exact source catalog types and constraints", "measured rows_by_table for every source table",
                                      "every nonempty case populates every declared source table", "exact original ordered rows match expected_original_rows",
                                      "at least two distinct nonempty original outputs with non-NULL values",
                                      "saved candidate differential checks using the strict verifier"],
            "coverage_complete_must_be_measured": True,
            "do_not_pass_plan_as_mini_dbs": True,
        },
    }
    return metadata, files


def build_plan(output_dir: Path, query_ids: list[str] | None = None) -> dict:
    """Write into a fresh destination. Never overwrite historical or prior output."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output must be a new versioned directory: {output_dir}")
    ids = list(RECIPES) if query_ids is None else list(query_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Provide one or more distinct supported query IDs")
    prepared = [(query_id, *query_plan(query_id)) for query_id in ids]
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "status": "offline_plan_not_postgresql_validated", "queries": {}}
    for query_id, metadata, files in prepared:
        directory = RECIPES[query_id]["directory"]
        destination = output_dir / directory
        destination.mkdir()
        for relative, data in files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest["queries"][query_id] = {"directory": directory, "manifest": f"{directory}/manifest.json"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh, separately versioned output directory")
    parser.add_argument("--query", action="append", choices=RECIPES, help="defaults to all pinned query recipes")
    args = parser.parse_args()
    manifest = build_plan(args.output, args.query)
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"],
                      "queries": list(manifest["queries"])}, indent=2))


if __name__ == "__main__":
    main()
