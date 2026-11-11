import os
import copy
import json

import psycopg
from psycopg import sql
import pytest

from scripts.build_join_aware_fixtures import (
    CASES, Evidence, make_rows, parse_schema, probe_specs,
    replace_fresh_rows, validate_catalog, build_bundle,
)
from mcts_rewrite.verify import synthesize_mini_databases


SCHEMA = '''CREATE TABLE dim(id INT PRIMARY KEY, label TEXT);
CREATE TABLE fact(id INT PRIMARY KEY, dim_id INT REFERENCES dim(id), quantity NUMERIC,
price NUMERIC, note TEXT);'''
QUERY = '''SELECT sum(f.price) FROM fact f JOIN dim d ON f.dim_id=d.id
WHERE d.label='active' AND f.quantity <
(SELECT 0.2 * avg(other.quantity) FROM fact other WHERE other.dim_id=d.id)'''


def model():
    tables = parse_schema(SCHEMA)
    evidence = Evidence(tables)
    evidence.analyze({'id': 'test:correlated', 'sql': QUERY})
    return tables, evidence


def test_join_lineage_and_literal_provenance_follow_aliases():
    tables, evidence = model()
    groups = evidence.groups()
    assert groups[('fact', 'dim_id')] == groups[('dim', 'id')]
    assert ('fact', 'id') not in groups
    assert evidence.profiles['test:correlated']['choices'][('dim', 'label')] == 'active'
    assert evidence.literals[('dim', 'label')][0]['source_id'] == 'test:correlated'
    assert any(j.get('source_id') == 'test:correlated' for j in evidence.joins)
    assert not evidence.issues
    evidence.analyze({'id': 'test:cte', 'sql': '''WITH x AS (SELECT dim_id AS k FROM fact)
        SELECT d.id FROM dim d JOIN x ON x.k=d.id'''})
    assert any(j.get('source_id') == 'test:cte' for j in evidence.joins)


def test_unrelated_matching_names_do_not_create_join_evidence():
    tables = parse_schema('CREATE TABLE a(id INT); CREATE TABLE b(id INT)')
    evidence = Evidence(tables)
    evidence.analyze({'id': 'cross', 'sql': 'SELECT a.id,b.id FROM a CROSS JOIN b'})
    assert not evidence.groups()
    evidence.analyze({'id': 'ambiguous', 'sql': 'SELECT id FROM a,b WHERE id=id'})
    assert not evidence.groups()


def test_generated_rows_share_domains_vary_measures_and_preserve_primary_keys():
    tables, evidence = model()
    for case in CASES:
        rows, features = make_rows(tables, evidence, case, ['test:correlated'])
        if case == 'empty':
            assert all(not values for values in rows.values())
            continue
        assert {row['dim_id'] for row in rows['fact'] if row['dim_id'] is not None} <= {row['id'] for row in rows['dim']}
        assert len({row['id'] for row in rows['fact']}) == len(rows['fact'])
        assert {0, 1, 10, 100} <= {row['quantity'] for row in rows['fact']} or case in {'duplicate', 'skew'}
        if case == 'null': assert any(value is None for values in rows.values() for row in values for value in row.values())
        if case == 'duplicate': assert any(f['kind'] == 'duplicate_payload_with_distinct_key' for f in features)


def test_unmatched_case_uses_evidenced_join_without_breaking_declared_fk():
    tables = parse_schema('CREATE TABLE a(k INT, measure INT); CREATE TABLE b(k INT)')
    evidence = Evidence(tables)
    evidence.analyze({'id': 'join', 'sql': 'SELECT sum(a.measure) FROM a JOIN b ON a.k=b.k'})
    rows, features = make_rows(tables, evidence, 'unmatched', [])
    assert any(f['kind'] == 'unmatched_join_key' for f in features)
    assert {r['k'] for r in rows['a']} - {r['k'] for r in rows['b']}


def test_ratio_literal_profile_prevents_zero_division_and_activates_filter():
    tables = parse_schema('CREATE TABLE sales(k INT, discount NUMERIC, list_price NUMERIC, sales_price NUMERIC)')
    evidence = Evidence(tables)
    evidence.analyze({'id': 'ratio', 'sql': '''SELECT sum(discount) FROM sales WHERE list_price BETWEEN 80 AND 109
        AND sales_price/list_price BETWEEN 80*0.01 AND 100*0.01'''})
    rows, _ = make_rows(tables, evidence, 'duplicate', ['ratio'])
    witnesses = rows['sales'][13:]
    assert witnesses and all(80 <= r['list_price'] <= 109 for r in witnesses)
    assert all(0.8 <= r['sales_price']/r['list_price'] <= 1 for r in witnesses)


def test_failed_allocation_journals_owned_names_and_preserves_original_error(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'id': 'probe', 'benchmark': 'tpch', 'sql': 'SELECT p_partkey FROM part'}) + '\n')
    class Admin:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
    def interrupted(*args, **kwargs):
        kwargs['_created_schemas'].append('mcts_mini_owned_test')
        raise RuntimeError('allocation interrupted')
    def offline(*args, **kwargs): raise psycopg.OperationalError('offline')
    monkeypatch.setattr('scripts.build_join_aware_fixtures.Postgres', Admin)
    monkeypatch.setattr('scripts.build_join_aware_fixtures._synthesize_cases', interrupted)
    monkeypatch.setattr('scripts.build_join_aware_fixtures.psycopg.connect', offline)
    snapshots = []
    with pytest.raises(RuntimeError, match='allocation interrupted'):
        build_bundle(manifest, 'unused', 'unused', benchmarks=['tpch'],
                     checkpoint=lambda b: snapshots.append(copy.deepcopy(b)))
    assert snapshots[-1]['allocated_namespaces'] == ['mcts_mini_owned_test']
    assert snapshots[-1]['build_status'] == 'failed' and 'offline' in snapshots[-1]['cleanup_error']


def test_live_constraint_valid_fixtures_activate_correlated_aggregate():
    dsn = os.environ.get('MCTS_TEST_DSN')
    if not dsn: pytest.skip('set MCTS_TEST_DSN for live PostgreSQL integration')
    tables, evidence = model()
    specs = synthesize_mini_databases(dsn, SCHEMA, query_sqls=[QUERY])
    try:
        for spec in specs:
            assert validate_catalog(dsn, spec['search_path'], tables)['source_types_match']
            rows, _ = make_rows(tables, evidence, spec['name'], ['test:correlated'])
            counts, rejections = replace_fresh_rows(dsn, spec['search_path'], tables, rows)
            assert not rejections, rejections
            if spec['name'] != 'empty': assert all(counts.values())
        coverage = probe_specs(specs, [{'id': 'test:correlated', 'sql': QUERY}])['test:correlated']
        assert coverage['nonvacuous'], coverage
        assert set(coverage['active_cases']) == set(CASES) - {'empty'}
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            for spec in specs:
                connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(spec['search_path'])))
