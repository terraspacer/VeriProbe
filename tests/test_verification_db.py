from contextlib import contextmanager
import copy
import os
import uuid

import psycopg
from psycopg import sql
import pytest

from mcts_rewrite.db import Postgres
from mcts_rewrite.verification_db import (
    ScopedEquivalenceGate, VerificationDatabaseError, _tables_from_schema, _collation,
    validate_verification_database,
)


SCHEMA = '''CREATE TABLE parent(id INT PRIMARY KEY, label TEXT NOT NULL,
code VARCHAR(8) COLLATE "C", CHECK(id>0));
CREATE TABLE child(id INT PRIMARY KEY, parent_id INT REFERENCES parent(id),
value NUMERIC(12,2), UNIQUE(parent_id,value));'''


def test_schema_names_preserve_postgres_quoting_and_reject_ambiguous_context():
    assert _tables_from_schema('CREATE TABLE EMP(a INT); CREATE TABLE "Mixed"(a TEXT)') == ['Mixed', 'emp']
    for source in ('', 'CREATE VIEW t AS SELECT 1', 'CREATE TABLE x.t(a INT)',
                   'CREATE TABLE t(a INT); CREATE TABLE T(b INT)'):
        with pytest.raises(ValueError): _tables_from_schema(source)


def test_catalog_mismatch_is_not_accepted_and_exception_carries_evidence(monkeypatch):
    canonical = {'tables': {'t': {'columns': [{'name': 'a', 'type': 'int4'}]}}, 'settings': {}, 'database_locale': {}}
    def snapshot(db, names, data_version=None):
        content = copy.deepcopy(canonical)
        if db == 'test': content['tables']['t']['columns'][0]['type'] = 'int8'
        return {'canonical': content}
    monkeypatch.setattr('mcts_rewrite.verification_db._catalog_snapshot', snapshot)
    result = validate_verification_database('main', 'test', 'CREATE TABLE t(a INT)')
    assert not result['compatible'] and result['status'] == 'mismatch'
    assert result['mismatches'] == ['table:t']
    with pytest.raises(VerificationDatabaseError) as caught:
        ScopedEquivalenceGate('main', 'test', 'CREATE TABLE t(a INT)')
    assert not caught.value.evidence['compatible']


def test_nondeterministic_collation_fails_even_when_icu_is_unavailable():
    class Cursor:
        def execute(self, *_): pass
        def fetchone(self): return ('public', {'collisdeterministic': False}, None)
    with pytest.raises(ValueError, match='Nondeterministic'):
        _collation(Cursor(), 123)


@pytest.fixture(scope='module')
def live_dsn():
    dsn = os.environ.get('MCTS_TEST_DSN')
    if not dsn: pytest.skip('set MCTS_TEST_DSN for live PostgreSQL integration')
    return dsn


@contextmanager
def live_pair(dsn):
    names = ['mcts_scope_' + uuid.uuid4().hex for _ in range(2)]
    with psycopg.connect(dsn, autocommit=True) as connection:
        for namespace in names:
            connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace)))
            connection.execute('SELECT set_config(\'search_path\',%s,false)', (namespace,))
            connection.execute(SCHEMA)
            connection.execute("INSERT INTO parent VALUES (1,'one','a'),(2,'two','b'),(3,'three','c')")
            connection.execute('INSERT INTO child VALUES (1,1,10),(2,2,10),(3,3,NULL)')
        connection.execute('SELECT set_config(\'search_path\',%s,false)', (names[0],))
        connection.execute("INSERT INTO parent VALUES (4,'four','d')")
        connection.execute('INSERT INTO child VALUES (4,4,100)')
        databases = [Postgres(dsn, search_path=name) for name in names]
        try: yield connection, databases[0], databases[1]
        finally:
            for db in databases: db.close()
            for namespace in names:
                connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(namespace)))


def test_live_catalog_compatibility_ignores_data_size_and_fingerprints_both_databases(live_dsn):
    with live_pair(live_dsn) as (_, main, test):
        result = validate_verification_database(main, test, SCHEMA)
        assert result['compatible'], result
        assert result['performance']['catalog_sha256'] == result['verification']['catalog_sha256']
        assert result['performance']['environment_sha256'] != result['verification']['environment_sha256']
        assert result['performance']['resume_safe'] and result['verification']['resume_safe']
        assert result['performance']['resolved_relations']['parent']['schema'] == main.search_path
        assert result['verification']['resolved_relations']['parent']['schema'] == test.search_path


@pytest.mark.parametrize('mutation', [
    'ALTER TABLE child ALTER COLUMN value TYPE NUMERIC(13,2)',
    'ALTER TABLE parent ALTER COLUMN label DROP NOT NULL',
    'ALTER TABLE parent ALTER COLUMN code TYPE VARCHAR(8) COLLATE "POSIX"',
    'ALTER TABLE parent ADD CHECK(id<1000)',
    'ALTER TABLE child DROP CONSTRAINT child_parent_id_fkey',
    'ALTER TABLE child DROP CONSTRAINT child_parent_id_value_key',
    'ALTER TABLE child ENABLE ROW LEVEL SECURITY',
    'DROP TABLE child; CREATE VIEW child AS SELECT 1 AS id,1 AS parent_id,10::numeric(12,2) AS value',
])
def test_live_catalog_semantic_mismatches_fail_closed(live_dsn, mutation):
    with live_pair(live_dsn) as (setup, main, test):
        setup.execute('SELECT set_config(\'search_path\',%s,false)', (test.search_path,))
        setup.execute(mutation)
        result = validate_verification_database(main, test, SCHEMA)
        assert not result['compatible'], result


def test_live_nondeterministic_collation_is_rejected(live_dsn):
    with live_pair(live_dsn) as (setup, main, test):
        if not setup.execute("SELECT EXISTS(SELECT 1 FROM pg_collation WHERE collprovider='i')").fetchone()[0]:
            pytest.skip('ICU is unavailable')
        setup.execute('SELECT set_config(\'search_path\',%s,false)', (test.search_path,))
        setup.execute("CREATE COLLATION nondet(provider=icu,locale='und',deterministic=false)")
        setup.execute('ALTER TABLE parent ALTER COLUMN code TYPE VARCHAR(8) COLLATE nondet')
        result = validate_verification_database(main, test, SCHEMA)
        assert not result['compatible'] and 'Nondeterministic' in result['reason']


def test_live_custom_constraint_function_is_rejected(live_dsn):
    with live_pair(live_dsn) as (setup, main, test):
        setup.execute('SELECT set_config(\'search_path\',%s,false)', (test.search_path,))
        setup.execute("CREATE FUNCTION positive(INT) RETURNS BOOLEAN LANGUAGE SQL IMMUTABLE AS 'SELECT $1>0'")
        setup.execute('ALTER TABLE parent ADD CHECK(positive(id))')
        result = validate_verification_database(main, test, SCHEMA)
        assert not result['compatible'] and 'user code' in result['reason']


def test_live_custom_index_expression_is_rejected(live_dsn):
    with live_pair(live_dsn) as (setup, main, test):
        setup.execute('SELECT set_config(\'search_path\',%s,false)', (test.search_path,))
        setup.execute("CREATE FUNCTION custom_key(INT) RETURNS INT LANGUAGE SQL IMMUTABLE AS 'SELECT $1+1'")
        setup.execute('CREATE INDEX custom_expression ON parent(custom_key(id))')
        result = validate_verification_database(main, test, SCHEMA)
        assert not result['compatible'] and 'user code' in result['reason']


def test_live_matching_catalogs_require_the_actual_source_model_order_and_types(live_dsn):
    with live_pair(live_dsn) as (_, main, test):
        for source in (SCHEMA.replace('value NUMERIC(12,2)', 'value NUMERIC(5,2)'),
                       SCHEMA.replace('label TEXT NOT NULL,\ncode VARCHAR(8) COLLATE "C"',
                                      'code VARCHAR(8) COLLATE "C", label TEXT NOT NULL')):
            result = validate_verification_database(main, test, source)
            assert not result['compatible'] and 'Source DDL' in result['reason'], result


def test_live_semantic_session_setting_difference_is_rejected(live_dsn):
    target = 'UTC'
    class OtherTimezone(Postgres):
        def _configure(self, cursor):
            super()._configure(cursor)
            cursor.execute("SELECT set_config('TimeZone',%s,true)", (target,))
    with live_pair(live_dsn) as (_, main, test):
        current = main.connection.execute("SELECT current_setting('TimeZone')").fetchone()[0]
        target = 'Asia/Tokyo' if current == 'UTC' else 'UTC'
        with OtherTimezone(test.dsn, search_path=test.search_path) as changed:
            result = validate_verification_database(main, changed, SCHEMA)
        assert not result['compatible'] and 'settings' in result['mismatches']


def test_live_unlogged_test_rows_need_version_for_resume(live_dsn):
    with live_pair(live_dsn) as (setup, main, test):
        setup.execute(sql.SQL('ALTER TABLE {}.child SET UNLOGGED').format(sql.Identifier(test.search_path)))
        result = validate_verification_database(main, test, SCHEMA)
        assert result['compatible'] and not result['verification']['resume_safe'], result
        versioned = validate_verification_database(main, test, SCHEMA, verification_data_version='fixture-v1')
        assert versioned['compatible'] and versioned['verification']['resume_safe']
        assert versioned['verification']['environment_sha256'] != result['verification']['environment_sha256']


def test_live_scoped_gate_never_hashes_main_rows_and_keeps_duplicate_null_checks(live_dsn, monkeypatch):
    with live_pair(live_dsn) as (_, main, test):
        def forbidden(*_): raise AssertionError('performance result transport is forbidden')
        monkeypatch.setattr(main, 'compare', forbidden)
        gate = ScopedEquivalenceGate(main, test, SCHEMA,
            performance_provenance={'dataset': 'large', 'dsn': 'secret'},
            verification_provenance={'dataset': 'small', 'dsn': 'also secret'})
        evidence = gate.verify('SELECT value FROM child', 'SELECT value FROM child')
        assert evidence['accepted'] and evidence['formal'], evidence
        assert evidence['execution']['original']['row_count'] == 3
        assert evidence['execution']['verification_scope']['performance_database_rows_hashed'] is False
        assert not evidence['execution']['performance_database_plan_check']['analyze']
        assert evidence['evidence_scope']['scope'] == 'separate_verification_database'
        assert 'dsn' not in gate.metadata['performance_database']
        assert not gate.verify('SELECT value FROM child', 'SELECT DISTINCT value FROM child')['accepted']
        assert not gate.verify('SELECT value FROM child', 'SELECT value FROM child WHERE value IS NOT NULL')['accepted']
        assert not gate.verify('SELECT value FROM child', 'SELECT value::double precision AS value FROM child')['accepted']
        # Main execution/measurement still uses all four performance rows.
        assert main.measure('SELECT sum(value) FROM child', runs=1, warmups=0)['status'] == 'ok'


def test_live_scoped_gate_propagates_numeric_profile_and_tolerance_to_miniatures(live_dsn):
    with live_pair(live_dsn) as (_, main, test):
        with Postgres(test.dsn, search_path=test.search_path, comparison_mode='stream_hash',
                      comparison_profile='positional_numeric', float_rel_tol=0,
                      float_abs_tol=0.1, max_rows=100, max_result_bytes=65536) as numeric:
            specs = [{'name': name, 'dsn': test.dsn, 'search_path': test.search_path,
                      'coverage_complete': True} for name in ('empty', 'null', 'duplicate', 'skew', 'unmatched')]
            gate = ScopedEquivalenceGate(main, numeric, SCHEMA, mode='differential', mini_dbs=specs)
            result = gate.verify('SELECT value::double precision AS original FROM child',
                                 'SELECT (value+0.05)::double precision AS renamed FROM child')
            assert result['execution']['equivalent'], result
            assert not result['formal']
            # Repeated schemas are deliberately insufficient coverage, but each
            # cloned connection must apply the exact scoped numeric contract.
            assert len(result['miniatures']) == 5, result
            for case in result['miniatures']:
                comparison = case['comparison']
                assert comparison['equivalent'], comparison
                assert comparison['comparison_profile'] == 'positional_numeric'
                assert comparison['comparison_mode'] == 'stream_hash'
                assert comparison['float_abs_tol'] == 0.1 and comparison['float_rel_tol'] == 0


def test_live_scoped_gate_rejects_catalog_drift_and_relations_outside_context(live_dsn):
    with live_pair(live_dsn) as (setup, main, test):
        gate = ScopedEquivalenceGate(main, test, SCHEMA)
        for db in (main, test):
            setup.execute(sql.SQL('CREATE TABLE {}.outside(a INT)').format(sql.Identifier(db.search_path)))
        assert not gate.verify('SELECT value FROM child', 'SELECT a AS value FROM outside')['accepted']
        setup.execute(sql.SQL('ALTER TABLE {}.parent ALTER COLUMN label DROP NOT NULL').format(sql.Identifier(test.search_path)))
        result = gate.verify('SELECT value FROM child', 'SELECT value FROM child')
        assert not result['accepted'] and 'compatibility changed' in result['reason']


def test_live_catalog_guard_blocks_concurrent_ddl(live_dsn):
    from mcts_rewrite.verify import EquivalenceGate
    with live_pair(live_dsn) as (_, main, test):
        attempts = []
        class ConcurrentDDL(EquivalenceGate):
            def verify(self, original, candidate):
                with psycopg.connect(live_dsn, autocommit=True) as other:
                    other.execute("SET lock_timeout='25ms'")
                    with pytest.raises(psycopg.errors.LockNotAvailable):
                        other.execute(sql.SQL('ALTER TABLE {}.parent ALTER COLUMN label DROP NOT NULL').format(sql.Identifier(test.search_path)))
                    attempts.append('blocked')
                return super().verify(original, candidate)
        gate = ScopedEquivalenceGate(main, test, SCHEMA, gate_class=ConcurrentDDL)
        result = gate.verify('SELECT value FROM child', 'SELECT value FROM child')
        assert result['accepted'] and attempts == ['blocked'], result
