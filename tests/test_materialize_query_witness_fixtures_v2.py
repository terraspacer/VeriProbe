"""Offline materializer contract tests. No PostgreSQL connection is permitted.

Fake database evidence exercises orchestration/failure behavior only. It does
not validate recipes on PostgreSQL or establish benchmark performance.
"""
from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path

import pytest

from scripts import materialize_query_witness_fixtures_v2 as materializer
from scripts.build_query_witness_fixtures_v2 import build_plan, sha256
from mcts_rewrite.study_v2 import configuration_for_query


QUERY = 'rbot:tpch:query18_0'


@pytest.fixture(autouse=True)
def forbid_live_connections(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline tests must never connect to PostgreSQL')
    monkeypatch.setattr(materializer.psycopg, 'connect', forbidden)


@pytest.fixture
def plan(tmp_path):
    directory = tmp_path / 'offline-plan'
    build_plan(directory, [QUERY])
    return directory


def strict_result(rows):
    fingerprint = {'sha256': sha256(json.dumps(rows).encode()), 'row_count': len(rows),
                   'nonnull_cells': sum(value is not None for row in rows for value in row),
                   'columns': [{'name': 'offline_fake_column', 'type_oid': 23}],
                   'comparison_kind': 'exact'}
    return {'status': 'ok', 'equivalent': True, 'same_snapshot': True,
            'comparison_mode': 'exact', 'comparison_profile': 'strict_metadata',
            'comparison_kind': 'exact', 'probabilistic_checksum': False,
            'numeric_relaxation_used': False, 'original': fingerprint, 'candidate': deepcopy(fingerprint)}


class FakeResult:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConnection:
    closed = False

    def __init__(self, database='mcts_fixtures'):
        self.database = database
        self.calls = []
        self.markers = {}

    def execute(self, statement, parameters=None):
        rendered = statement if isinstance(statement, str) else statement.as_string()
        self.calls.append((rendered, parameters))
        if rendered == 'SELECT current_database()':
            return FakeResult((self.database,))
        if rendered.startswith('SELECT obj_description'):
            namespace = parameters[0]
            return FakeResult((self.markers[namespace],) if namespace in self.markers else None)
        if rendered.startswith('DROP SCHEMA'):
            namespace = rendered.split('"')[1]
            del self.markers[namespace]
        return FakeResult(None)

    def transaction(self):
        return nullcontext()

    def close(self):
        self.closed = True


def fake_database(monkeypatch, plan, *, bad_case=None, bad_catalog=False, fail_create=None, wrong_database=False, query_id=QUERY):
    """Install fake SQL/catalog evidence without invoking any real DB method."""
    prepared = materializer.load_plan(plan)['queries'][query_id]
    by_case = {case['name']: case for case in prepared['metadata']['cases']}
    connection = FakeConnection('foreign_database' if wrong_database else 'mcts_fixtures')
    created, comparisons, drops = [], [], []
    monkeypatch.setattr(materializer.psycopg, 'connect', lambda *args, **kwargs: connection)
    monkeypatch.setattr(materializer, 'require_owned_lock', lambda fd: {'verified': True, 'method': 'offline fake only'})

    def create(conn, namespace, owner, prepared_query, case, indexes, timeout):
        connection.markers[namespace] = owner
        created.append(namespace)
        if fail_create == case['name']:
            raise RuntimeError('offline simulated case creation failure')
        return deepcopy(case['planned_rows_by_table'])

    class FakePostgres:
        def __init__(self, dsn, **kwargs):
            self.search_path = kwargs['search_path']
            self.case = self.search_path.split('"')[1].rsplit('_', 1)[1]
            if 'comparison_mode' in kwargs:
                assert kwargs['comparison_mode'] == 'exact'
                assert kwargs['comparison_profile'] == 'strict_metadata'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def compare(self, original, candidate):
            comparisons.append((self.case, original, candidate))
            result = strict_result(by_case[self.case]['expected_original_rows'])
            if self.case == bad_case:
                result['numeric_relaxation_used'] = True
            return result

    def drop(conn, names, owner):
        drops.extend(names)
        return materializer._drop_owned_original(conn, names, owner)

    monkeypatch.setattr(materializer, '_create_case', create)
    monkeypatch.setattr(materializer, 'Postgres', FakePostgres)
    monkeypatch.setattr(materializer, '_original_expectation', lambda db, original, expected: deepcopy(expected))
    monkeypatch.setattr(materializer, 'validate_verification_database', lambda *args, **kwargs:
                        {'compatible': not bad_catalog, 'catalog_sha256': 'offline_fake_catalog', 'reason': 'offline fake only'})
    monkeypatch.setattr(materializer, '_drop_owned_original', materializer._drop_owned, raising=False)
    monkeypatch.setattr(materializer, '_drop_owned', drop)
    return connection, created, comparisons, drops, prepared


def test_lock_requires_existing_exclusive_open_description_and_exact_file(tmp_path):
    lock_file = tmp_path / 'shared.lock'
    other_file = tmp_path / 'unrelated.lock'
    with lock_file.open('a+') as owned, other_file.open('a+') as other:
        with pytest.raises(ValueError, match='does not own'):
            materializer.require_owned_lock(owned.fileno(), lock_file)
        # This temporary-file lock is offline, never the real shared lock.
        fcntl.flock(owned, fcntl.LOCK_SH)
        with pytest.raises(ValueError, match='does not own'):
            materializer.require_owned_lock(owned.fileno(), lock_file)
        fcntl.flock(owned, fcntl.LOCK_EX)
        evidence = materializer.require_owned_lock(owned.fileno(), lock_file)
        assert evidence['verified'] is True
        assert evidence['owner_pid'] == os.getpid()
        with pytest.raises(ValueError, match='not the shared'):
            materializer.require_owned_lock(other.fileno(), lock_file)
    with pytest.raises(ValueError, match='inherited descriptor'):
        materializer.require_owned_lock(None, lock_file)


def test_pinned_plan_loads_but_retains_offline_status(plan):
    loaded = materializer.load_plan(plan)
    query = loaded['queries'][QUERY]
    assert len(query['metadata']['cases']) == 5
    assert len(query['candidates']) == 3
    assert sum(candidate['comparison_role'] == 'saved_candidate_differential' for candidate in query['candidates']) == 2
    assert query['metadata']['validation']['postgresql_executed'] is False
    assert sha256(query['original'].encode()) == query['metadata']['original_checkpoint_sha256']
    assert 'sql' in query['candidates'][0]


@pytest.mark.parametrize('target', ['schema.sql', 'original.sql', 'cases/duplicate.sql', 'manifest.json'])
def test_plan_drift_fails_before_any_connection_or_output(plan, tmp_path, target):
    path = plan / 'tpch_query18_0' / target
    if target == 'manifest.json':
        metadata = json.loads(path.read_text())
        metadata['cases'][1]['expected_original_rows'] = []
        path.write_text(json.dumps(metadata))
    else:
        path.write_text(path.read_text() + '\n-- copied-plan drift\n')
    output = tmp_path / 'bindings.json'
    with pytest.raises(ValueError, match='differs from the pinned'):
        materializer.materialize(plan, output, 'offline fake only', lock_fd=-1)
    assert not output.exists()
    assert not output.with_suffix('.failure.json').exists()


def test_unowned_shared_lock_fails_before_connection_or_output(plan, tmp_path, monkeypatch):
    def reject(fd):
        raise ValueError('not owned')
    monkeypatch.setattr(materializer, 'require_owned_lock', reject)
    output = tmp_path / 'bindings.json'
    with pytest.raises(ValueError, match='not owned'):
        materializer.materialize(plan, output, 'offline fake only', lock_fd=-1)
    assert not list(tmp_path.glob('bindings*'))


def test_success_with_fake_evidence_produces_study_contract_and_fresh_outputs(plan, tmp_path, monkeypatch):
    connection, created, comparisons, drops, prepared = fake_database(monkeypatch, plan)
    before = {str(path.relative_to(plan)): path.read_bytes() for path in plan.rglob('*') if path.is_file()}
    output = tmp_path / 'postgresql_validated_bindings.json'
    binding = materializer.materialize(plan, output, 'offline fake only', lock_fd=42)
    assert len(created) == 5 and not drops
    assert len(comparisons) == 20  # Five strict self-checks plus three archived SQLs per case.
    assert connection.closed
    assert len(set(created)) == 5
    assert all(name.startswith('qw2_') and len(name) < 64 for name in created)
    assert json.loads(output.read_text()) == binding
    assert output.stat().st_mode & 0o777 == 0o600
    evidence = json.loads(output.with_suffix('.evidence.json').read_text())
    assert evidence['status'] == 'postgresql_validated'
    assert evidence['benchmark_executed'] is False
    assert binding['queries'][QUERY]['validation']['distinct_nonempty_fingerprints'] == 3
    selected = binding['queries'][QUERY]['verification_database']
    assert '_duplicate"' in selected['search_path']
    assert selected['search_path'].endswith(', pg_catalog')
    schema = plan / 'tpch_query18_0/schema.sql'
    row = {'id': QUERY, 'benchmark': 'tpch', 'sql': prepared['original'], 'schema_path': str(schema)}
    config = {'databases': {'tpch10': {'dsn': 'performance_untouched'}}}
    configured = configuration_for_query(config, row, binding)
    assert configured['databases']['tpch10']['dsn'] == 'performance_untouched'
    configured_primary = configured['databases']['tpch10']['verification_database']
    assert all(configured_primary[key] == value for key, value in selected.items())
    assert before == {str(path.relative_to(plan)): path.read_bytes() for path in plan.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        materializer.materialize(plan, output, 'offline fake only', lock_fd=42)


@pytest.mark.parametrize('problem', ['comparison', 'catalog', 'creation', 'database'])
def test_failed_fake_validation_never_writes_binding_and_cleans_only_owned(plan, tmp_path, monkeypatch, problem):
    connection, created, comparisons, drops, prepared = fake_database(
        monkeypatch, plan, bad_case='duplicate' if problem == 'comparison' else None,
        bad_catalog=problem == 'catalog', fail_create='null' if problem == 'creation' else None,
        wrong_database=problem == 'database')
    connection.markers['do_not_drop_foreign'] = 'someone else'
    output = tmp_path / f'{problem}_bindings.json'
    with pytest.raises((ValueError, RuntimeError)):
        materializer.materialize(plan, output, 'offline fake only', lock_fd=42)
    assert not output.exists()
    failure = json.loads(output.with_suffix('.failure.json').read_text())
    assert failure['status'] == 'failed' and failure['ready_binding_written'] is False
    assert failure['benchmark_executed'] is False
    assert failure['cleanup_errors'] == []
    assert drops == created
    assert connection.markers == {'do_not_drop_foreign': 'someone else'}


def test_cleanup_refuses_foreign_ownership_marker():
    connection = FakeConnection()
    connection.markers = {'ours': 'this-run', 'theirs': 'different-run'}
    errors = materializer._drop_owned(connection, ['ours', 'absent', 'theirs'], 'this-run')
    assert len(errors) == 1 and errors[0]['schema'] == 'theirs'
    assert connection.markers == {'theirs': 'different-run'}


def test_create_executes_exact_source_ddl_and_records_indexes_separately(plan, monkeypatch):
    prepared = materializer.load_plan(plan)['queries'][QUERY]
    case = prepared['metadata']['cases'][1]
    counts = case['planned_rows_by_table']
    monkeypatch.setattr(materializer, 'measured_population', lambda conn, tables: counts)
    connection = FakeConnection()
    indexes = materializer.auxiliary_indexes('tpch')
    measured = materializer._create_case(connection, 'offline_fake_schema', 'offline_fake_owner', prepared, case, indexes, 1000)
    statements = [statement for statement, _ in connection.calls]
    assert prepared['files']['schema.sql'].decode() in statements
    assert prepared['files'][case['data_file']].decode() in statements
    index_statements = [statement for statement in statements if statement.startswith('CREATE INDEX')]
    assert len(index_statements) == len(indexes) == 4
    assert all('UNIQUE' not in statement for statement in index_statements)
    assert all(index['role'] == 'auxiliary_access_path_not_source_constraint' and index['unique'] is False for index in indexes)
    assert materializer.auxiliary_indexes('tpch', False) == []
    assert measured == counts


@pytest.mark.parametrize('change', [
    lambda result: result.update(equivalent=False),
    lambda result: result.update(status='timeout'),
    lambda result: result.update(same_snapshot=False),
    lambda result: result.update(comparison_mode='stream_hash'),
    lambda result: result.update(comparison_profile='positional_numeric'),
    lambda result: result.update(comparison_kind='tolerance_matching'),
    lambda result: result.update(probabilistic_checksum=True),
    lambda result: result.update(numeric_relaxation_used=True),
    lambda result: result.update(original=None),
])
def test_relaxed_incomplete_or_failed_candidate_evidence_rejected(change):
    comparison = strict_result([[1]])
    materializer.require_strict_comparison(comparison)
    change(comparison)
    with pytest.raises(ValueError, match='strict exact differential'):
        materializer.require_strict_comparison(comparison)


def test_population_requires_measured_every_table_exact_counts_and_not_booleans():
    case = {'name': 'null', 'planned_rows_by_table': {'a': 1, 'b': 1}, 'planned_rows_inserted': 2}
    materializer.validate_population(case, {'a': 1, 'b': 1}, ['a', 'b'])
    for counts in ({'a': 1}, {'a': True, 'b': 1}, {'a': 2, 'b': 1}, {'a': 1, 'b': 0}):
        with pytest.raises(ValueError):
            materializer.validate_population(case, counts, ['a', 'b'])
    empty = {**case, 'name': 'empty'}
    with pytest.raises(ValueError, match='Empty/nonempty'):
        materializer.validate_population(empty, {'a': 1, 'b': 1}, ['a', 'b'])


def test_nonvacuity_requires_distinct_nonempty_nonnull_original_outputs():
    def case(name, rows):
        return {'name': name, 'original_rows': rows, 'original_fingerprint': strict_result(rows)['original']}
    with pytest.raises(ValueError, match='two distinct nonempty'):
        materializer.validate_nonvacuity([case('empty', []), case('null', [[None]]), case('skew', [[None]])])
    with pytest.raises(ValueError):
        materializer.validate_nonvacuity([case('empty', []), case('null', [[1]]), case('skew', [[1]])])
    result = materializer.validate_nonvacuity([case('empty', []), case('null', [[1]]), case('skew', [[2]])])
    assert result['nonvacuous'] is True and result['distinct_nonempty_fingerprints'] == 2


def test_atomic_writer_never_replaces_prior_binding_and_cleans_temporary_files(tmp_path):
    output = tmp_path / 'binding.json'
    materializer.write_new_json(output, {'measured': False})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        materializer.write_new_json(output, {'measured': True})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob('.*.tmp'))


def test_expectation_formatter_is_only_display_encoding():
    rows = [('Alpha   ', date(2001, 2, 2), Decimal('313.01'), None)]
    assert materializer.expectation_rows(rows) == [['Alpha', '2001-02-02', '313.01', None]]
    assert rows[0][0] == 'Alpha   '


def test_all_thirteen_recipes_load_with_exact_archived_status_provenance(tmp_path):
    plan = tmp_path / 'all-recipes'
    build_plan(plan)
    loaded = materializer.load_plan(plan)
    assert len(loaded['queries']) == 13
    for query in loaded['queries'].values():
        archive = query['candidate_archive']
        assert archive['archive_sha256'] == materializer.ARCHIVE_SHA256
        assert len(archive['record_sha256']) == 64
        assert archive['record_line_number'] > 0
        assert archive['arms']
        if not query['candidates']:
            assert 'all historical generation arms were skipped or generation_error' in archive['not_applicable_reason']
            assert all(arm['status'] in {'skipped', 'generation_error'} for arm in archive['arms'].values())
        for candidate in query['candidates']:
            assert sha256(candidate['sql'].encode()) == candidate['sha256']
    assert materializer.auxiliary_indexes('calcite')


def test_archive_hash_drift_fails_before_any_live_access(plan, tmp_path, monkeypatch):
    copied = tmp_path / 'altered_archive.jsonl'
    copied.write_bytes(materializer.ARCHIVE.read_bytes() + b'\n')
    monkeypatch.setattr(materializer, 'ARCHIVE', copied)
    with pytest.raises(ValueError, match='archive hash drift'):
        materializer.load_plan(plan)


def test_absent_candidate_sql_requires_historical_skip_or_generation_error():
    original = 'SELECT 1'
    base = {'query_id': 'test-query', 'original_sql': original,
            'query_sha256': sha256(original.encode()), 'schema_sha256': 'source_schema',
            'arms': {'original': {'status': 'ok', 'sql': original}, 'mcts': {'status': 'skipped'}}}
    candidates, provenance = materializer.archived_candidates('test-query', original, 'source_schema', [(1, b'raw historical record', base)])
    assert candidates == [] and provenance['not_applicable_reason']
    base['arms']['mcts']['status'] = 'ok'
    with pytest.raises(ValueError, match='without a skipped/generation_error'):
        materializer.archived_candidates('test-query', original, 'source_schema', [(1, b'raw historical record', base)])


def test_structurally_empty_recipe_requires_bound_proof_and_populated_source_tables(tmp_path):
    query_id = 'rbot:calcite:testIsNullPushDown'
    plan = tmp_path / 'proof-plan'
    build_plan(plan, [query_id])
    prepared = materializer.load_plan(plan)['queries'][query_id]
    cases = [{'name': case['name'], 'original_rows': [], 'original_fingerprint': strict_result([])['original'],
              'rows_by_table': case['planned_rows_by_table']} for case in prepared['metadata']['cases']]
    validated = materializer.validate_output_coverage(cases, prepared)
    assert validated['nonvacuous'] is True
    assert validated['base_tables_nonempty'] is True
    assert validated['result_empty'] is True
    assert validated['structural_empty_proven'] is True
    assert validated['structural_empty_proof']['query_sha256'] == prepared['metadata']['original_checkpoint_sha256']
    assert validated['structural_empty_proof']['schema_sha256'] == prepared['metadata']['source_schema_sha256']
    assert validated['distinct_nonempty_fingerprints'] == 0
    broken = deepcopy(cases)
    broken[1]['rows_by_table']['emp'] = 0
    with pytest.raises(ValueError, match='fully populated'):
        materializer.validate_output_coverage(broken, prepared)
    altered = deepcopy(prepared)
    altered['original'] = 'SELECT empno FROM emp'
    with pytest.raises(ValueError, match='Independent structural empty proof'):
        materializer.validate_output_coverage(cases, altered)
    altered = deepcopy(prepared)
    altered['candidates'] = [{'sql': 'SELECT empno FROM emp', 'sha256': 'unproven'}]
    with pytest.raises(ValueError, match='Saved candidate lacks'):
        materializer.validate_output_coverage(cases, altered)


def test_symlink_cannot_place_materialization_output_inside_immutable_plan(plan, tmp_path):
    alias = tmp_path / 'plan-alias'
    alias.symlink_to(plan, target_is_directory=True)
    with pytest.raises(ValueError, match='separate from the immutable'):
        materializer.materialize(plan, alias / 'binding.json', 'offline fake only', lock_fd=-1)
    assert not (plan / 'binding.json').exists()


@pytest.mark.parametrize('query_id', ['rbot:dsb:query081_0', 'rbot:calcite:testIsNullPushDown'])
def test_fake_materialization_handles_skipped_candidates_and_structural_empty_primary(tmp_path, monkeypatch, query_id):
    plan = tmp_path / 'plan'
    build_plan(plan, [query_id])
    connection, created, comparisons, drops, prepared = fake_database(monkeypatch, plan, query_id=query_id)
    output = tmp_path / 'bindings.json'
    bindings = materializer.materialize(plan, output, 'offline fake only', lock_fd=42)
    validation = bindings['queries'][query_id]['validation']
    assert validation['nonvacuous'] is True
    assert validation['original_expectations_passed'] is True
    assert validation['saved_candidate_checks_passed'] is True
    assert len(created) == 5 and not drops
    if query_id.endswith('query081_0'):
        assert len(comparisons) == 5  # Original strict checks still run without saved candidates.
        assert validation['saved_candidate_sql_count'] == 0
        assert 'skipped' in validation['saved_candidate_checks_not_applicable_reason']
    else:
        assert validation['structural_empty_proven'] is True
        assert validation['result_empty'] is True and validation['base_tables_nonempty'] is True
        assert validation['finite_evidence_only'] is False
        assert len(comparisons) == 10  # Self checks plus one saved original-identity arm SQL.
