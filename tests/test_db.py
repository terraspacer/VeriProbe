import os
import uuid
from decimal import Decimal

import psycopg
from psycopg import sql
import pytest

from mcts_rewrite.db import (Postgres, _encode_row, parse_read_query, _StreamingRows,
                             _PositionalRows, _tolerant_rows_equal)


@pytest.mark.parametrize('query', [
    'SELECT 1; SELECT 2', 'DELETE FROM t',
    'WITH changed AS (DELETE FROM t RETURNING *) SELECT * FROM changed',
    'SELECT * INTO other FROM t', 'SELECT * FROM t FOR UPDATE',
    'SELECT random()', 'SELECT now()', 'SELECT pg_sleep(1)',
    'SELECT nextval(\'seq\')', 'SELECT public.abs(1)', 'SELECT unknown_function(1)',
])
def test_unsafe_sql_rejected(query):
    with pytest.raises(Exception):
        parse_read_query(query)


def test_decimal_encoding_never_rounds_through_float():
    assert _encode_row((Decimal('9007199254740992.01'),)) != _encode_row((Decimal('9007199254740992.02'),))
    assert _encode_row((1,)) != _encode_row((1.0,))
    assert _encode_row((None,)) != _encode_row(('None',))


def test_streaming_checksum_duplicate_null_and_order_sensitivity():
    rows = [(1, None), (2, 'x'), (1, None)]
    def checksum(values, ordered=False):
        digest = _StreamingRows(ordered)
        for row in values:
            digest.add(_encode_row(row))
        return digest.signature(), digest.hexdigest()
    assert checksum(rows) == checksum(list(reversed(rows)))
    assert checksum(rows) != checksum(rows[:-1])
    assert checksum(rows) != checksum([(None, 1), (2, 'x'), (None, 1)])
    assert checksum(rows, ordered=True) != checksum(rows[1:] + rows[:1], ordered=True)
    assert checksum(rows, ordered=True) == checksum(rows, ordered=True)


def test_comparison_mode_and_optional_limits():
    assert Postgres('unused', comparison_mode='stream_hash', max_rows=None,
                    max_result_bytes=None).max_rows is None
    with pytest.raises(ValueError):
        Postgres('unused', comparison_mode='unsafe')


def test_positional_numeric_encoding_keeps_exact_decimal_values():
    encode = lambda row: _encode_row(row, 'positional_numeric')
    assert encode((1, Decimal('1.000'), Decimal('1E+3'))) == encode((Decimal('1'), 1, 1000))
    assert encode((Decimal('9007199254740992.01'),)) != encode((Decimal('9007199254740992.02'),))
    assert encode((Decimal('123456789012345678901234567890.00'),)) == encode((123456789012345678901234567890,))
    assert encode((False,)) != encode((0,))
    assert encode((None,)) != encode(('None',))
    with pytest.raises(ValueError, match='comparison_profile'):
        Postgres('unused', comparison_profile='unknown')
    for value in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='tolerances'):
            Postgres('unused', float_abs_tol=value)


def _positional_rows(rows, ordered=False):
    result = _PositionalRows(None, ordered)
    size = 0
    for count, row in enumerate(rows, 1):
        encoded = _encode_row(row, 'positional_numeric')
        size += len(encoded)
        result.add(row, encoded, count, size)
    return result


def test_float_bag_matching_reassigns_exact_matches_and_preserves_duplicates():
    equal = lambda l, r: _tolerant_rows_equal(_positional_rows(l), _positional_rows(r), 1e-5, 1e-5)
    # Greedy cancellation of the shared zero would wrongly reject this bag.
    assert equal([(0.0,), (0.000009,)], [(0.0,), (-0.000009,)])
    assert equal([(0.0,), (0.0,), (0.000018,)], [(0.000009,), (0.000009,), (-0.000009,)])
    assert not equal([(0.0,), (0.0,), (1.0,)], [(0.0,), (1.0,), (1.0,)])
    assert not equal([(None,), (1.0,)], [(1.0,), (1.0,)])
    assert equal([(100000.0,)], [(Decimal('100000.9'),)])  # relative tolerance
    assert equal([(0.0,)], [(Decimal('0.000009'),)])  # absolute tolerance
    assert not equal([(0.0,)], [(Decimal('0.000011'),)])
    assert not equal([(Decimal('9007199254740992.01'),)], [(Decimal('9007199254740992.02'),)])
    assert not _tolerant_rows_equal(_positional_rows([(0.0,), (0.000009,)], True),
        _positional_rows([(0.0,), (-0.000009,)], True), 1e-5, 1e-5)


def test_float_matching_limit_is_explicit_error(monkeypatch):
    monkeypatch.setattr('mcts_rewrite.db._FLOAT_MATCH_EDGES', 1)
    rows = _positional_rows([(0.0,), (1.0,)])
    with pytest.raises(ValueError, match='candidate limit'):
        _tolerant_rows_equal(rows, rows, 1e-5, 1e-5)
    rows.rows = None
    with pytest.raises(ValueError, match='bounded materialized'):
        _tolerant_rows_equal(rows, rows, 1e-5, 1e-5)


@pytest.fixture(scope='module')
def live():
    dsn = os.environ.get('MCTS_TEST_DSN')
    if not dsn:
        pytest.skip('set MCTS_TEST_DSN for live PostgreSQL integration')
    namespace = 'mcts_dbtest_' + uuid.uuid4().hex
    setup = psycopg.connect(dsn, autocommit=True)
    with setup.cursor() as cursor:
        cursor.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('CREATE TABLE {}.t(a INT, b INT, n NUMERIC(30,2))').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('INSERT INTO {}.t VALUES (1,2,9007199254740992.01), (1,2,9007199254740992.01), (NULL,3,NULL), (2,NULL,0.02)').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('CREATE TABLE {}.snapshot_t(a INT)').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('INSERT INTO {}.snapshot_t VALUES(1)').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('CREATE TABLE {}.context_t(a INT, d DATE, c CHAR(3), n NUMERIC)').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('CREATE VIEW {}.context_v AS SELECT a FROM {}.context_t').format(sql.Identifier(namespace), sql.Identifier(namespace)))
        cursor.execute(sql.SQL('CREATE TABLE {}.context_rls(a INT)').format(sql.Identifier(namespace)))
        cursor.execute(sql.SQL('ALTER TABLE {}.context_rls ENABLE ROW LEVEL SECURITY').format(sql.Identifier(namespace)))
    database = Postgres(dsn, search_path=namespace)
    yield database
    database.close()
    with setup.cursor() as cursor:
        cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(namespace)))
    setup.close()


def test_live_duplicate_and_null_semantics(live):
    assert live.compare('SELECT a FROM t', 'SELECT a FROM t')['equivalent']
    assert not live.compare('SELECT a FROM t', 'SELECT DISTINCT a FROM t')['equivalent']
    assert not live.compare('SELECT a FROM t', 'SELECT a FROM t WHERE a IS NOT NULL')['equivalent']
    assert not live.compare('SELECT a FROM t', 'SELECT b AS a FROM t')['equivalent']


def test_live_streaming_comparison_labels_and_multiset_semantics(live):
    with Postgres(live.dsn, search_path=live.search_path, comparison_mode='stream_hash',
                  max_rows=None, max_result_bytes=None) as streaming:
        root = 'SELECT a,b FROM t'
        permuted = 'SELECT a,b FROM t WHERE a IS NULL UNION ALL SELECT a,b FROM t WHERE a IS NOT NULL'
        result = streaming.compare(root, permuted)
        assert result['equivalent'] and result['same_snapshot'], result
        assert result['comparison_kind'] == 'probabilistic_checksum' and result['probabilistic_checksum']
        assert result['original']['row_count'] == result['candidate']['row_count'] == 4
        assert result['original']['sha256'] == result['candidate']['sha256']
        assert result['original']['distinct_rows'] is None
        assert not streaming.compare('SELECT a FROM t', 'SELECT DISTINCT a FROM t')['equivalent']
        assert not streaming.compare('SELECT a FROM t ORDER BY a NULLS FIRST',
                                     'SELECT a FROM t ORDER BY a DESC NULLS LAST')['equivalent']
        assert not streaming.compare('SELECT a FROM t WHERE FALSE',
                                     'SELECT a::bigint AS a FROM t WHERE FALSE')['equivalent']
        many = 'WITH RECURSIVE x(n) AS (VALUES (1) UNION ALL SELECT n+1 FROM x WHERE n<10000) SELECT n FROM x'
        assert streaming.compare(many, many)['original']['row_count'] == 10000


def test_live_compare_deadline_is_shared_across_queries(live):
    class SlowCollection(Postgres):
        collections = 0
        def _collect(self, query, ordered):
            if not self.collections:
                self.collections += 1
                result = super()._collect(query, ordered)
                self._comparison_deadline = 0  # Simulate time exhausted by the first query.
                return result
            return super()._collect(query, ordered)
    with SlowCollection(live.dsn, search_path=live.search_path,
                        comparison_mode='stream_hash') as database:
        result = database.compare('SELECT a FROM t', 'SELECT a FROM t')
        assert not result['equivalent'] and result['status'] == 'timeout'
        assert database._comparison_deadline is None


def test_live_decimal_types_aliases_and_empty_results(live):
    assert live.compare('SELECT n FROM t', 'SELECT n FROM t')['equivalent']
    assert not live.compare('SELECT n FROM t', 'SELECT n::double precision AS n FROM t')['equivalent']
    assert not live.compare('SELECT a FROM t WHERE FALSE', 'SELECT a::bigint AS a FROM t WHERE FALSE')['equivalent']
    assert not live.compare('SELECT a FROM t', 'SELECT a AS other FROM t')['equivalent']


@pytest.mark.parametrize('mode', ['exact', 'stream_hash'])
def test_live_positional_numeric_profile(live, mode):
    with Postgres(live.dsn, search_path=live.search_path, comparison_mode=mode,
                  comparison_profile='positional_numeric') as db:
        result = db.compare('SELECT a FROM t', 'SELECT a::numeric AS renamed FROM t')
        assert result['equivalent'] and result['numeric_relaxation_used'], result
        assert result['comparison_profile'] == 'positional_numeric'
        assert db.compare('SELECT a FROM t', 'SELECT a AS renamed FROM t')['equivalent']
        assert db.compare('SELECT a FROM t WHERE FALSE', 'SELECT a::bigint AS renamed FROM t WHERE FALSE')['equivalent']
        assert not db.compare('SELECT a FROM t WHERE FALSE', 'SELECT a::text FROM t WHERE FALSE')['equivalent']
        assert not db.compare('SELECT a,b FROM t', 'SELECT b,a FROM t')['equivalent']
        assert not db.compare('SELECT a FROM t', 'SELECT DISTINCT a FROM t')['equivalent']
        assert not db.compare('SELECT a FROM t', 'SELECT a FROM t WHERE a IS NOT NULL')['equivalent']
        assert not db.compare('SELECT a FROM t ORDER BY a', 'SELECT a FROM t')['equivalent']
        close = db.compare('SELECT 100000.0::double precision AS original', 'SELECT 100000.9::numeric AS renamed')
        assert close['equivalent'] and close['numeric_relaxation_used'], close
        assert close['comparison_kind'] == 'tolerance_matching' and not close['probabilistic_checksum']
        assert close['original']['sha256'] != close['candidate']['sha256']
        assert not db.compare('SELECT 0.0::double precision', 'SELECT 0.000011::numeric')['equivalent']
        assert db.compare('SELECT x FROM (VALUES (0.0::double precision),(0.000009)) t(x)',
                          'SELECT y FROM (VALUES (0.0::double precision),(-0.000009)) t(y)')['equivalent']
        assert not db.compare('SELECT x FROM (VALUES (0.0::double precision),(0.0),(1.0)) t(x)',
                              'SELECT y FROM (VALUES (0.0::double precision),(1.0),(1.0)) t(y)')['equivalent']


def test_live_order_guarantees(live):
    assert not live.compare('SELECT a FROM t ORDER BY a NULLS FIRST', 'SELECT a FROM t ORDER BY a DESC NULLS LAST')['equivalent']
    assert not live.compare('SELECT a FROM t ORDER BY a', 'SELECT a FROM t')['equivalent']
    assert live.compare('SELECT a FROM t ORDER BY a', 'SELECT a FROM t ORDER BY a')['ordered']


def test_live_formal_context_audits_referenced_columns_and_base_relations(live):
    supported = live.formal_context_supported('SELECT x.a FROM context_t x', 'SELECT a FROM context_t',
                                              'CREATE TABLE context_t(a INT)')
    assert supported['supported'], supported
    assert supported['relations'][0]['columns'] == ['a']
    for column in ('d', 'c', 'n', '*'):
        assert not live.formal_context_supported(f'SELECT {column} FROM context_t',
                                                f'SELECT {column} FROM context_t')['supported']
    for relation in ('context_v', 'context_rls'):
        assert not live.formal_context_supported(f'SELECT a FROM {relation}',
                                                f'SELECT a FROM {relation}')['supported']
    mismatch = live.formal_context_supported('SELECT a FROM context_t', 'SELECT a FROM context_t',
                                            'CREATE TABLE context_t(a TEXT)')
    assert not mismatch['supported'] and 'types differ' in mismatch['reason']


def test_live_server_measurement(live):
    measurement = live.measure('SELECT count(*) FROM t', runs=3, warmups=1)
    assert measurement['status'] == 'ok'
    assert len(measurement['samples_ms']) == len(measurement['wall_samples_ms']) == 3
    assert measurement['median_ms'] >= 0
    assert 'Plan' in measurement['plan']
    assert 'Actual Total Time' not in measurement['plan']['Plan']
    assert 'TIMING OFF' in measurement['measurement_method']
    assert 'Actual Total Time' in live.explain('SELECT count(*) FROM t', analyze=True)['Plan']


def test_live_limits_and_errors_fail_closed(live):
    with Postgres(live.dsn, search_path=live.search_path, max_rows=2) as guarded:
        result = guarded.compare('SELECT a FROM t', 'SELECT a FROM t')
        assert not result['equivalent'] and 'limit' in result['error']
    assert not live.compare('SELECT 1/0', 'SELECT 1/0')['equivalent']
    assert live.compare('SELECT 1', 'SELECT 1')['equivalent']  # transaction recovered


def test_live_timeout_never_measurement(live):
    query = 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<1000000000) SELECT sum(n) FROM x'
    with Postgres(live.dsn, search_path=live.search_path, timeout_ms=5) as guarded:
        result = guarded.measure(query, runs=1, warmups=0)
        assert result['status'] == 'timeout'
        assert result['median_ms'] is None


def test_live_comparison_uses_one_snapshot(live):
    class ConcurrentMutation(Postgres):
        calls = 0
        def _collect(self, query, ordered):
            result = super()._collect(query, ordered)
            self.calls += 1
            if self.calls == 1:
                with psycopg.connect(self.dsn, autocommit=True) as other:
                    other.execute(sql.SQL('UPDATE {}.snapshot_t SET a=2').format(sql.Identifier(self.search_path)))
            return result
    with ConcurrentMutation(live.dsn, search_path=live.search_path) as database:
        comparison = database.compare('SELECT a FROM snapshot_t', 'SELECT a FROM snapshot_t')
        assert comparison['equivalent'] and comparison['same_snapshot']
    assert live.compare('SELECT a FROM snapshot_t', 'SELECT 2 AS a')['equivalent']
