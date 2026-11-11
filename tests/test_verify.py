import os
import uuid

import psycopg
from psycopg import sql
import pytest
import sqlglot

from mcts_rewrite.db import Postgres
from mcts_rewrite.verify import (
    EquivalenceGate, conservative_identity, parse_sqlsolver_verdict,
    _solver_input, _synthesize_cases,
    _approved_planner_hint, synthesize_mini_databases,
)


class EqualDB:
    def compare(self, original, candidate):
        return {'equivalent': True, 'status': 'ok', 'same_snapshot': True}


def test_identity_preserves_semantics_and_hints():
    assert conservative_identity('SELECT a FROM t; -- harmless') == conservative_identity('select A from T')
    for first, second in [
        ('SELECT "A" FROM t', 'SELECT "a" FROM t'),
        ("SELECT 'Hello'", "SELECT 'hello'"),
        ("SELECT E'\\n'", "SELECT '\\n'"),
        ('SELECT 1.00', 'SELECT 1.0'),
        ('SELECT /*+ SeqScan(t) */ a FROM t', 'SELECT a FROM t'),
        ('SELECT a FROM t ORDER BY a', 'SELECT a FROM t'),
        ('SELECT Ä FROM t', 'SELECT ä FROM t'),
    ]:
        assert conservative_identity(first) != conservative_identity(second)


@pytest.mark.parametrize('output', [None, '', 'NOT_EQ', 'EQ NEQ', 'answer EQ', 'EQ\nUNKNOWN', 'EQ;'])
def test_ambiguous_solver_outputs_never_proof(output):
    assert parse_sqlsolver_verdict(output) is None


def test_solver_single_verdict():
    assert parse_sqlsolver_verdict(' EQ\n') == 'EQ'
    assert parse_sqlsolver_verdict('NEQ') == 'NEQ'


def test_strict_rejects_finite_match_without_proof(monkeypatch):
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'UNKNOWN', 'status': 'ok'})
    gate = EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)')
    result = gate.verify('SELECT a FROM t WHERE a=1', 'SELECT a FROM t WHERE a=2')
    assert not result['accepted'] and not result['formal']
    assert gate.verify('SELECT a FROM t', 'select A from T;')['formal']


def test_solver_neq_overrides_differential_data(monkeypatch):
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'NEQ', 'status': 'ok'})
    gate = EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)', mode='differential', mini_dbs=[EqualDB()])
    result = gate.verify('SELECT a FROM t WHERE a=1', 'SELECT a FROM t WHERE a=2')
    assert not result['accepted'] and 'disproved' in result['reason']


def test_formal_requires_real_execution(monkeypatch):
    class WrongDB:
        def compare(self, *_):
            return {'equivalent': False, 'error': 'different duplicates'}
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'EQ', 'status': 'ok'})
    result = EquivalenceGate(WrongDB(), 'CREATE TABLE t(a INT)').verify('SELECT a FROM t', 'SELECT a FROM t')
    assert not result['accepted']


def test_numeric_relaxation_is_never_a_formal_proof(monkeypatch):
    class RelaxedDB(EqualDB):
        def compare(self, *_):
            return {'equivalent': True, 'status': 'ok', 'same_snapshot': True,
                    'numeric_relaxation_used': True}
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'EQ', 'status': 'ok'})
    original, candidate = 'SELECT a FROM t WHERE a=1 AND a=2', 'SELECT a FROM t WHERE a=2 AND a=1'
    result = EquivalenceGate(RelaxedDB(), 'CREATE TABLE t(a INT)').verify(original, candidate)
    assert not result['accepted'] and not result['formal']
    assert 'numeric comparison relaxation' in result['reason']
    class Miniature:
        def __init__(self, index): self.index = index
        def compare(self, *_):
            return {'equivalent': True, 'numeric_relaxation_used': True,
                    'original': {'row_count': self.index, 'nonnull_cells': self.index, 'sha256': str(self.index)}}
    result = EquivalenceGate(RelaxedDB(), 'CREATE TABLE t(a INT)', mode='hybrid',
                             mini_dbs=[Miniature(i) for i in range(5)]).verify(original, candidate)
    assert result['accepted'] and not result['formal'] and result['proof'] == 'differential'


def test_supported_solver_proof_and_differential_label(monkeypatch):
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'EQ', 'status': 'ok'})
    original = 'SELECT a FROM t WHERE a=1 AND a=2'
    candidate = 'SELECT a FROM t WHERE a=2 AND a=1'
    result = EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)').verify(original, candidate)
    assert result['accepted'] and result['formal'] and result['proof'] == 'sqlsolver'
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'UNKNOWN', 'status': 'ok'})
    class Miniature:
        def __init__(self, index):
            self.index = index
        def compare(self, *_):
            return {'equivalent': True, 'original': {'row_count': self.index,
                    'nonnull_cells': self.index, 'sha256': str(self.index)}}
    result = EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)', mode='differential', mini_dbs=[Miniature(i) for i in range(5)]).verify(original, candidate)
    assert result['accepted'] and not result['formal'] and result['proof'] == 'differential'


@pytest.mark.parametrize('hint', [
    '+ Set(enable_hashjoin off)', '+ HashJoin(a b) Leading((a b) c)',
    '+ SeqScan(t) Set(random_page_cost 4.0)', '+ IndexScan(t t_index)',
])
def test_planner_only_hint_syntax(hint):
    assert _approved_planner_hint(hint)


@pytest.mark.parametrize('hint', [
    '+ Set(search_path secret)', '+ Set(timezone UTC)', '+ Set(statement_timeout 0)',
    '+ Set(work_mem 100)', '+ Unknown(t)', '+ Set(enable_hashjoin off); DROP TABLE t',
    '+ Set(enable_hashjoin off) Set(search_path other)', '+ Leading((t u)',
    '+ SeqScan(t; DROP TABLE x)', '! arbitrary',
])
def test_semantic_or_ambiguous_hints_rejected(hint):
    assert not _approved_planner_hint(hint)


def test_hint_identity_requires_enabled_extension():
    original, candidate = 'SELECT a FROM t', '/*+ Set(enable_hashjoin off) */ SELECT a FROM t'
    assert not EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)').verify(original, candidate)['accepted']
    class HintedDB(EqualDB):
        def planner_hints_enabled(self):
            return {'enabled': True}
    result = EquivalenceGate(HintedDB(), 'CREATE TABLE t(a INT)').verify(original, candidate)
    assert result['accepted'] and result['formal'] and result['proof'] == 'ast'
    for mode in ('strict', 'hybrid'):
        bad = EquivalenceGate(HintedDB(), 'CREATE TABLE t(a INT)', mode=mode).verify(original, '/*+ Set(search_path other) */ SELECT a FROM t')
        assert not bad['accepted'] and 'semantic' in bad['reason']


@pytest.mark.parametrize('query', [
    'SELECT a+1 FROM t', 'SELECT a FROM t ORDER BY a', 'SELECT a FROM t LIMIT 1',
    'SELECT count(*) FROM t', 'SELECT "a" FROM t', 'SELECT a::bigint FROM t',
    'SELECT a FROM t LEFT JOIN u ON t.a=u.a', 'SELECT abs(a) FROM t',
])
def test_unsupported_solver_constructs_never_accepted(monkeypatch, query):
    def unexpected(*_):
        raise AssertionError('unsupported SQL reached solver')
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', unexpected)
    result = EquivalenceGate(EqualDB(), 'CREATE TABLE t(a INT)').verify('SELECT a FROM t', query)
    assert not result['accepted'] and not result['formal']


def test_solver_drops_untrusted_constraints():
    query = sqlglot.parse_one('SELECT a FROM t', read='postgres')
    schema, reason = _solver_input(query, query, 'CREATE TABLE t(a INT PRIMARY KEY, b TEXT NOT NULL, CHECK(a>0))')
    assert 'PRIMARY KEY' not in schema and 'NOT NULL' not in schema and 'CHECK' not in schema
    assert 'no schema constraints' in reason


def test_solver_projects_both_queries_and_resolves_aliases_and_ctes():
    original = sqlglot.parse_one('WITH x AS (SELECT a, b FROM t) SELECT a FROM x WHERE b=1')
    candidate = sqlglot.parse_one('SELECT q.a FROM t q WHERE q.b=1 AND q.c=q.c')
    schema, _ = _solver_input(original, candidate,
        'CREATE TABLE t(a INT, b TEXT, c INT, unused DATE); CREATE TABLE unused(d DATE)')
    assert schema == 'CREATE TABLE t (a INT, b TEXT, c INT);'


@pytest.mark.parametrize('original,candidate,schema', [
    ('SELECT a FROM t', 'SELECT d FROM t', 'CREATE TABLE t(a INT,d DATE)'),
    ('SELECT * FROM t', 'SELECT a FROM t', 'CREATE TABLE t(a INT,d DATE)'),
    ('SELECT a FROM t JOIN u ON t.b=u.b', 'SELECT t.a FROM t JOIN u ON t.b=u.b',
     'CREATE TABLE t(a INT,b INT); CREATE TABLE u(a INT,b INT)'),
    ('SELECT a FROM t', 'SELECT missing FROM t', 'CREATE TABLE t(a INT)'),
    ('SELECT a FROM t', 'SELECT a FROM missing', 'CREATE TABLE t(a INT)'),
    ('SELECT a FROM t', 'SELECT a FROM t WHERE a=a', 'CREATE TABLE t(a CHAR(3))'),
    ('SELECT a FROM t', 'SELECT a FROM t WHERE a=a', 'CREATE TABLE t(a TEXT COLLATE "C")'),
    ('SELECT t.a,u.b FROM t NATURAL JOIN u', 'SELECT t.a,u.b FROM t CROSS JOIN u',
     'CREATE TABLE t(a INT,d DATE); CREATE TABLE u(b INT,d DATE)'),
])
def test_solver_projection_fails_closed(original, candidate, schema):
    projected, _ = _solver_input(sqlglot.parse_one(original), sqlglot.parse_one(candidate), schema)
    assert projected is None


@pytest.fixture(scope='module')
def live():
    dsn = os.environ.get('MCTS_TEST_DSN')
    if not dsn:
        pytest.skip('set MCTS_TEST_DSN for live PostgreSQL integration')
    namespace = 'mcts_verifytest_' + uuid.uuid4().hex
    connection = psycopg.connect(dsn, autocommit=True)
    connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace)))
    connection.execute(sql.SQL('CREATE TABLE {}.t(a INT, b INT)').format(sql.Identifier(namespace)))
    connection.execute(sql.SQL('INSERT INTO {}.t VALUES (1,1)').format(sql.Identifier(namespace)))
    database = Postgres(dsn, search_path=namespace)
    yield database
    database.close()
    connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(namespace)))
    connection.close()


def test_live_adversarial_current_data_coincidence(live, monkeypatch):
    # The real database cannot distinguish a=b or DISTINCT here; miniature data must.
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'UNKNOWN', 'status': 'ok'})
    gate = EquivalenceGate(live, 'CREATE TABLE t(a INT, b INT)', mode='differential')
    result = gate.verify('SELECT a FROM t', 'SELECT DISTINCT a FROM t')
    assert result['execution']['equivalent']
    assert not result['accepted']
    assert any(not case['comparison']['equivalent'] for case in result['miniatures'])


def test_live_synthesized_constraints_and_coverage(live):
    schema = '''CREATE TABLE parent(id INT PRIMARY KEY, value INT);
    CREATE TABLE child(id INT PRIMARY KEY, parent_id INT REFERENCES parent(id), value INT);'''
    cases = _synthesize_cases(live, schema, 'SELECT value FROM child', 'SELECT value FROM child')
    assert [case['name'] for case in cases] == ['empty', 'null', 'duplicate', 'skew', 'unmatched']
    assert all(case['comparison']['equivalent'] for case in cases)
    assert all(case['rows_inserted'] > 0 for case in cases[1:])


def test_live_sqlsolver_proof(live):
    # This is a proof integration check, not a JVM startup latency assertion.
    # Benchmark solver budgets are independently fixed in each run config.
    gate = EquivalenceGate(live, 'CREATE TABLE t(a INT, b INT)', solver_timeout=60)
    result = gate.verify('SELECT a FROM t WHERE a=1 AND b=1', 'SELECT a FROM t WHERE b=1 AND a=1')
    assert result['accepted'], result
    assert result['formal'] and result['proof'] == 'sqlsolver'


def test_live_literal_aware_persistent_fixtures_and_hybrid(live):
    specs = synthesize_mini_databases(live.dsn, 'CREATE TABLE t(a INT, d DATE, name TEXT)',
        query_sqls=["SELECT a FROM t WHERE a>42 AND d>=DATE '1995-03-15' AND name='BRASS'"])
    try:
        assert {spec['name'] for spec in specs} == {'empty', 'null', 'duplicate', 'skew', 'unmatched'}
        assert all(spec['coverage_complete'] for spec in specs)
        spec = next(s for s in specs if s['name'] == 'duplicate')
        with Postgres(spec['dsn'], search_path=spec['search_path']) as mini:
            assert mini.compare("SELECT count(*) AS n FROM t WHERE name='BRASS'", "SELECT count(*) AS n FROM t WHERE name='BRASS'")['original']['row_count'] == 1
            assert mini.compare("SELECT d FROM t WHERE d=DATE '1995-03-15'", "SELECT d FROM t WHERE d=DATE '1995-03-15'")['original']['row_count'] > 0
        # Unsupported arithmetic is accepted only as finite evidence here.
        comparison = EquivalenceGate(live, 'CREATE TABLE t(a INT, d DATE, name TEXT)', mode='hybrid', mini_dbs=specs).verify('SELECT a+0 AS a FROM t', 'SELECT a AS a FROM t')
        assert comparison['accepted'] and not comparison['formal'], comparison
    finally:
        with psycopg.connect(live.dsn, autocommit=True) as setup:
            for spec in specs:
                setup.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(spec['search_path'])))


def test_live_vacuous_scalar_aggregate_does_not_pass_hybrid(live, monkeypatch):
    monkeypatch.setattr('mcts_rewrite.verify._run_sqlsolver', lambda *args: {'verdict': 'UNKNOWN', 'status': 'ok'})
    gate = EquivalenceGate(live, 'CREATE TABLE t(a INT, b INT)', mode='hybrid')
    result = gate.verify('SELECT count(*) AS n FROM t WHERE a>100000', 'SELECT count(*) AS n FROM t WHERE a>100001')
    # Query-literal-aware synthesis may find a counterexample, otherwise all
    # invariant zero outputs must be rejected as vacuous evidence.
    assert not result['accepted'], result


def test_live_synthesis_folds_only_unquoted_identifiers(live):
    schema = 'CREATE TABLE EMP (EMPNO INT PRIMARY KEY, DEPTNO INT, "MixedCase" TEXT)'
    cases = _synthesize_cases(live, schema, 'SELECT EMPNO, DEPTNO, "MixedCase" FROM EMP',
                              'SELECT EMPNO, DEPTNO, "MixedCase" FROM EMP',
                              query_sqls=['SELECT EMPNO FROM EMP WHERE DEPTNO=42'])
    assert len(cases) == 5
    assert all(case['comparison']['equivalent'] for case in cases)
    assert all(case['rows_by_table']['emp'] > 0 for case in cases[1:])
    assert all('deptno' in case['literal_pool_sizes'] for case in cases)


def test_live_synthesis_handles_dsb_time_column(live):
    cases = _synthesize_cases(live, 'CREATE TABLE dbgen_version(dv_create_time TIME)',
                             'SELECT dv_create_time FROM dbgen_version',
                             'SELECT dv_create_time FROM dbgen_version')
    assert len(cases) == 5 and all(case['comparison']['equivalent'] for case in cases)
    assert all(case['rows_by_table']['dbgen_version'] > 0 for case in cases[1:])
