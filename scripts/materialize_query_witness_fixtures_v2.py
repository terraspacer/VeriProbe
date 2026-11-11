#!/usr/bin/env python3
"""Materialize a pinned offline query-witness-v2 plan in owned PostgreSQL schemas.

This command never starts services, creates databases, or runs performance
benchmarks. The caller must already own the shared experiment flock and pass
its inherited descriptor. A ready binding is written only after all five cases
pass exact-DDL catalog, measured population, expectation, nonvacuity, and
unchanged strict Postgres.compare checks against every saved candidate.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import stat
import sys
import uuid

import psycopg
from psycopg import sql
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_query_witness_fixtures_v2 import CASES, RECIPES, VERSION, query_plan, sha256
from mcts_rewrite.db import Postgres, parse_read_query
from mcts_rewrite.structural_empty_v2 import prove_tautological_empty
from mcts_rewrite.verification_db import validate_verification_database

SHARED_LOCK = Path('/workspace/VeriProbe/.runtime/heavy-experiment.lock')
DATABASE = 'mcts_fixtures'
ARCHIVE = ROOT / 'results/stage_a_sf10_local_resume/raw.jsonl'
ARCHIVE_SHA256 = 'a2a15a97059d6315ee129a6f8da0609e2bdaf9711519019ed645f6360810b353'

# These are ordinary nonunique access paths, never source constraints. The
# authoritative schema.sql is executed byte-for-byte and is never rewritten.
AUXILIARY_INDEXES = {
    'tpch': [('customer', ['c_custkey']), ('orders', ['o_orderkey']),
             ('orders', ['o_custkey']), ('lineitem', ['l_orderkey'])],
    'dsb': [('customer', ['c_current_cdemo_sk', 'c_current_addr_sk']),
            ('store_sales', ['ss_customer_sk', 'ss_sold_date_sk', 'ss_item_sk']),
            ('web_sales', ['ws_bill_customer_sk', 'ws_sold_date_sk', 'ws_item_sk']),
            ('catalog_sales', ['cs_ship_customer_sk', 'cs_sold_date_sk', 'cs_item_sk'])],
    'calcite': [('emp', ['deptno']), ('emp', ['mgr'])],
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Lossless JSON display; candidate comparison never uses this formatter."""
    if isinstance(value, (Decimal, date, datetime, time)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


def expectation_rows(rows):
    # The plan explicitly uses CHAR display values without PG padding. This
    # formatter applies only to recipe expectations; strict comparisons retain
    # all values, ordered multiplicities, types, and column metadata.
    return [[value.rstrip() if isinstance(value, str) else json_value(value)
             for value in row] for row in rows]


def write_new_json(path, value):
    """Atomic exclusive publication: prior files are never replaced."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            os.chmod(temporary, 0o600)
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_owned_lock(lock_fd, lock_path=SHARED_LOCK):
    """Inspect Linux fdinfo for this *open description's* existing flock.

    No flock call is made here: testing by acquiring a new lock would allow an
    unlocked invocation to start live work. fdinfo also handles a descriptor
    inherited from a supervising parent (the recorded PID may be the parent).
    """
    if type(lock_fd) is not int or lock_fd < 0:
        raise ValueError('An inherited descriptor for the owned shared experiment lock is required')
    expected = Path(lock_path).resolve(strict=True)
    details = os.fstat(lock_fd)
    wanted = expected.stat()
    if not stat.S_ISREG(details.st_mode) or (details.st_dev, details.st_ino) != (wanted.st_dev, wanted.st_ino):
        raise ValueError('Lock descriptor is not the shared experiment lock file')
    info = Path(f'/proc/self/fdinfo/{lock_fd}').read_text()
    key = f'{os.major(details.st_dev):02x}:{os.minor(details.st_dev):02x}:{details.st_ino}'
    for line in info.splitlines():
        fields = line.split()
        if (fields[:1] == ['lock:'] and len(fields) == 9
                and fields[2:5] == ['FLOCK', 'ADVISORY', 'WRITE']
                and fields[6] == key and fields[7:] == ['0', 'EOF']):
            return {'path': str(expected), 'fd': lock_fd, 'owner_pid': int(fields[5]),
                    'verified': True, 'method': 'inherited open-description flock in /proc/self/fdinfo'}
    raise ValueError('The inherited descriptor does not own an exclusive shared experiment flock')


def _child_path(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Plan paths must be relative files')
    root = Path(directory).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError('Plan path escapes its directory or is not a file')
    return path


def archived_candidates(query_id, original, schema_hash, archive):
    """Read actual saved arm SQL, never manufacture a missing candidate."""
    matches = [(number, line, record) for number, line, record in archive if record.get('query_id') == query_id]
    if len(matches) != 1:
        raise ValueError('Query must occur exactly once in the immutable historical archive')
    number, line, record = matches[0]
    query_hash = sha256(original.encode())
    if (record.get('original_sql') != original or record.get('query_sha256') != query_hash
            or record.get('schema_sha256') != schema_hash):
        raise ValueError('Historical archive SQL/schema binding differs from the exact plan')
    provenance = {'path': str(ARCHIVE.relative_to(ROOT)), 'archive_sha256': ARCHIVE_SHA256,
                  'record_line_number': number, 'record_sha256': sha256(line),
                  'record_hash_encoding': 'exact JSONL record bytes excluding line terminator',
                  'original_status': record.get('arms', {}).get('original', {}).get('status'), 'arms': {}}
    unique = {}
    for name, arm in record.get('arms', {}).items():
        if name == 'original':
            continue
        text = arm.get('sql')
        provenance['arms'][name] = {'status': arm.get('status'), 'sql_present': isinstance(text, str) and bool(text),
                                    'reason': arm.get('reason')}
        if not text:
            if arm.get('status') not in {'skipped', 'generation_error'}:
                raise ValueError(f'Historical arm lacks SQL without a skipped/generation_error status: {name}')
            continue
        if not isinstance(text, str):
            raise ValueError('Historical arm SQL must be a string')
        tree = parse_read_query(text)
        if any(table.db or table.catalog for table in tree.find_all(exp.Table)):
            raise ValueError('Saved candidates must use remappable unqualified tables')
        digest = sha256(text.encode())
        if digest not in unique:
            unique[digest] = {'sql': text, 'sha256': digest, 'source': 'historical_selected_arm',
                              'archive': {key: value for key, value in provenance.items() if key != 'arms'},
                              'historical_arms': [], 'comparison_role': 'identity_recheck' if text == original else 'saved_candidate_differential'}
        unique[digest]['historical_arms'].append({'name': name, 'status': arm.get('status')})
    if not provenance['arms']:
        raise ValueError('Historical archive has no generation-arm status evidence')
    provenance.update(distinct_saved_arm_sql_count=len(unique),
                      distinct_nonoriginal_arm_sql_count=sum(candidate['sql'] != original for candidate in unique.values()))
    if not unique:
        provenance['not_applicable_reason'] = 'No saved generated-arm SQL: all historical generation arms were skipped or generation_error'
    return list(unique.values()), provenance


def load_plan(plan_dir):
    """Validate all plan bytes and saved candidates before any DB connection.

    These v2 recipes are intentionally pinned. Reconstructing the offline plan
    rejects altered DDL, data, table counts, expectations, or validation labels,
    even when someone has updated a hash in a copied manifest.
    """
    plan_dir = Path(plan_dir).resolve(strict=True)
    manifest_path = _child_path(plan_dir, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('version') != VERSION
            or manifest.get('status') != 'offline_plan_not_postgresql_validated'
            or not isinstance(manifest.get('queries'), dict) or not manifest['queries']):
        raise ValueError('A nonempty offline query-witness-v2 plan is required')
    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError('Immutable Stage A historical candidate archive hash drift')
    archive = [(number, line, json.loads(line)) for number, line in enumerate(archive_bytes.splitlines(), 1) if line.strip()]
    prepared = {}
    for query_id, entry in manifest['queries'].items():
        if query_id not in RECIPES:
            raise ValueError(f'Unsupported pinned recipe: {query_id}')
        directory = RECIPES[query_id]['directory']
        if entry != {'directory': directory, 'manifest': f'{directory}/manifest.json'}:
            raise ValueError('Plan query manifest location differs from the pinned recipe')
        metadata_path = _child_path(plan_dir, entry['manifest'])
        metadata = json.loads(metadata_path.read_text())
        expected, files = query_plan(query_id)
        if metadata != expected:
            raise ValueError(f'Offline plan metadata differs from the pinned recipe: {query_id}')
        query_dir = metadata_path.parent
        for relative, payload in files.items():
            if _child_path(query_dir, relative).read_bytes() != payload:
                raise ValueError(f'Offline plan file differs from the pinned recipe: {query_id}/{relative}')
        original = files[metadata['original_file']].decode().rstrip()
        parse_read_query(original)
        if sha256(original.encode()) != metadata['original_checkpoint_sha256']:
            raise ValueError('Original SQL does not match the frozen checkpoint hash')
        candidates, archive_provenance = archived_candidates(query_id, original, metadata['source_schema_sha256'], archive)
        if not RECIPES[query_id].get('generated'):
            witness = ROOT / 'results/stage_a_followup' / RECIPES[query_id]['witness']
            provenance = json.loads((witness / 'provenance.json').read_text())
            if not provenance.get('candidates'):
                raise ValueError('Historical witness candidate checks must not be empty')
            for candidate in provenance['candidates']:
                path = _child_path(witness, candidate['file'])
                payload = path.read_bytes()
                text = payload.decode().rstrip()
                if sha256(text.encode()) != candidate['sha256']:
                    raise ValueError(f'Saved candidate hash drift: {path}')
                tree = parse_read_query(text)
                if any(table.db or table.catalog for table in tree.find_all(exp.Table)):
                    raise ValueError('Saved candidates must use remappable unqualified tables')
                if not any(saved['sql'] == text for saved in candidates):
                    candidates.append({'file': str(path.relative_to(ROOT)), 'sha256': candidate['sha256'],
                                       'file_sha256': sha256(payload), 'sql': text, 'source': 'historical_witness_trace',
                                       'comparison_role': 'saved_candidate_differential'})
        prepared[query_id] = {'metadata': metadata, 'files': files, 'original': original,
                              'candidates': candidates, 'candidate_archive': archive_provenance,
                              'manifest_path': str(metadata_path),
                              'manifest_sha256': sha256(metadata_path.read_bytes())}
    return {'directory': str(plan_dir), 'manifest_sha256': sha256(manifest_path.read_bytes()),
            'queries': prepared}


def measured_population(connection, tables):
    return {table: connection.execute(sql.SQL('SELECT count(*) FROM {}').format(
        sql.Identifier(table))).fetchone()[0] for table in tables}


def validate_population(case, measured, tables):
    if (set(measured) != set(tables)
            or any(type(value) is not int or value < 0 for value in measured.values())
            or measured != case['planned_rows_by_table']
            or sum(measured.values()) != case['planned_rows_inserted']):
        raise ValueError(f"Measured full-schema row counts differ from the plan: {case['name']}")
    if (case['name'] == 'empty' and any(measured.values())) or (case['name'] != 'empty' and not all(measured.values())):
        raise ValueError('Empty/nonempty source-table population requirement failed')


def require_strict_comparison(comparison):
    if (comparison.get('status') != 'ok' or comparison.get('equivalent') is not True
            or comparison.get('same_snapshot') is not True
            or comparison.get('comparison_mode') != 'exact'
            or comparison.get('comparison_profile') != 'strict_metadata'
            or comparison.get('comparison_kind') != 'exact'
            or comparison.get('probabilistic_checksum') is not False
            or comparison.get('numeric_relaxation_used') is not False
            or not comparison.get('original') or not comparison.get('candidate')):
        raise ValueError('Saved candidate failed the unchanged strict exact differential verifier')


def validate_nonvacuity(cases):
    active = [case for case in cases if case['name'] != 'empty' and case['original_rows']
              and case['original_fingerprint'].get('nonnull_cells', 0) > 0]
    signatures = {case['original_fingerprint']['sha256'] for case in active}
    if len(active) < 2 or len(signatures) < 2:
        raise ValueError('Original outputs require two distinct nonempty cases with non-NULL values')
    return {'nonvacuous': True, 'active_cases': [case['name'] for case in active],
            'distinct_nonempty_fingerprints': len(signatures)}


def validate_output_coverage(cases, prepared):
    """A proof-required empty result uses populated bases plus a bound proof.

    No finite sequence of empty outputs establishes this proof. The independent
    structural checker must prove the exact original SQL and every saved SQL.
    """
    metadata = prepared['metadata']
    if not metadata.get('materialization_contract', {}).get('proof_required'):
        return validate_nonvacuity(cases)
    if metadata.get('output_classification') != 'tautological_empty':
        raise ValueError('Proof-required recipe must declare the structural empty classification')
    proof = prove_tautological_empty(prepared['original'], prepared['files'][metadata['schema_file']].decode())
    if (proof.get('status') != 'proven' or proof.get('root_empty') is not True
            or proof.get('query_sha256') != metadata['original_checkpoint_sha256']
            or proof.get('schema_sha256') != metadata['source_schema_sha256']):
        raise ValueError('Independent structural empty proof does not bind the exact original SQL/schema')
    if any(case['original_rows'] or case['original_fingerprint'].get('row_count') != 0 for case in cases):
        raise ValueError('Structurally empty original produced nonempty rows')
    populated = [case for case in cases if case['name'] != 'empty']
    if len(populated) != 4 or any(not all(case['rows_by_table'].values()) for case in populated):
        raise ValueError('Structural empty proof requires fully populated nonempty base cases')
    candidates = []
    for candidate in prepared['candidates']:
        receipt = prove_tautological_empty(candidate['sql'], prepared['files'][metadata['schema_file']].decode())
        if receipt.get('status') != 'proven' or receipt.get('root_empty') is not True:
            raise ValueError('Saved candidate lacks the independent structural empty proof')
        candidates.append({'candidate_sha256': candidate['sha256'], 'proof': receipt})
    return {'nonvacuous': True, 'base_tables_nonempty': True, 'result_empty': True,
            'structural_empty_proven': True, 'structural_empty_proof': proof,
            'saved_candidate_structural_proofs': candidates,
            'validation_basis': 'populated_source_tables_and_independent_structural_empty_proof',
            'original_result_variation_required': False, 'active_cases': [case['name'] for case in populated],
            'distinct_nonempty_fingerprints': 0}


def auxiliary_indexes(benchmark, enabled=True):
    return [{'name': f'qw2_aux_{number}', 'table': table, 'columns': columns,
             'unique': False, 'method': 'btree', 'role': 'auxiliary_access_path_not_source_constraint'}
            for number, (table, columns) in enumerate(AUXILIARY_INDEXES[benchmark]) if enabled]


def _create_case(connection, namespace, owner, prepared, case, indexes, timeout_ms):
    with connection.transaction():
        connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace)))
        connection.execute(sql.SQL('COMMENT ON SCHEMA {} IS {}').format(sql.Identifier(namespace), sql.Literal(owner)))
        connection.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(namespace)))
        connection.execute("SELECT set_config('statement_timeout', %s, true), set_config('lock_timeout', '1000', true)",
                           (str(timeout_ms),))
        metadata = prepared['metadata']
        connection.execute(prepared['files'][metadata['schema_file']].decode())
        connection.execute(prepared['files'][case['data_file']].decode())
        measured = measured_population(connection, metadata['source_tables'])
        validate_population(case, measured, metadata['source_tables'])
        for index in indexes:
            connection.execute(sql.SQL('CREATE INDEX {} ON {} USING btree ({})').format(
                sql.Identifier(index['name']), sql.Identifier(index['table']),
                sql.SQL(', ').join(sql.Identifier(column) for column in index['columns'])))
        for table in metadata['source_tables']:
            connection.execute(sql.SQL('ANALYZE {}').format(sql.Identifier(table)))
    return measured


def _original_expectation(db, original, expected):
    with db.connection.transaction():
        with db.connection.cursor() as cursor:
            db._configure(cursor)
            db._audit_functions(cursor, parse_read_query(original))
            cursor.execute(original)
            rows = cursor.fetchmany(db.max_rows + 1)
            if len(rows) > db.max_rows:
                raise ValueError('Original expectation result exceeds the fixture row limit')
    displayed = expectation_rows(rows)
    if displayed != expected:
        raise ValueError('Measured ordered original rows differ from expected_original_rows')
    return displayed


def _drop_owned(connection, namespaces, owner):
    """Never drop a schema unless its persisted marker matches this run."""
    errors = []
    for namespace in reversed(namespaces):
        try:
            marker = connection.execute("SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname=%s",
                                        (namespace,)).fetchone()
            if marker is None:  # Rolled-back CREATE SCHEMA is already absent.
                continue
            if marker[0] != owner:
                raise ValueError(f'Refusing cleanup of schema with a foreign ownership marker: {namespace}')
            connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(namespace)))
        except Exception as error:
            errors.append({'schema': namespace, 'error_type': type(error).__name__, 'error': str(error)})
    return errors


def materialize(plan_dir, output_path, dsn, *, lock_fd, timeout_ms=10000,
                create_indexes=True, performance_databases=None):
    """Write a fresh binding JSON and sidecars; keep successful owned schemas.

    performance_databases optionally maps query IDs to explicit dsn/search_path
    specs. Its catalog-only audits never scan or time performance-database rows.
    """
    output_path = Path(output_path).resolve()
    evidence_path = output_path.with_name(output_path.stem + '.evidence.json')
    failure_path = output_path.with_name(output_path.stem + '.failure.json')
    if any(path.exists() for path in (output_path, evidence_path, failure_path)):
        raise FileExistsError(f'Binding and evidence outputs must be fresh: {output_path}')
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError('statement timeout must be positive')
    plan = load_plan(plan_dir)
    if Path(plan['directory']) == output_path or Path(plan['directory']) in output_path.parents:
        raise ValueError('Materialization output must be separate from the immutable offline plan')
    lock = require_owned_lock(lock_fd)
    performance_databases = performance_databases or {}
    if set(performance_databases) - set(plan['queries']):
        raise ValueError('Performance catalog specifications must name queries in this plan')
    run_id = uuid.uuid4().hex
    owner = f'query-witness-v2-materializer:{run_id}'
    owned = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {'version': VERSION, 'status': 'materializing', 'run_id': run_id, 'owner_marker': owner,
                'started_at_utc': stamp(), 'offline_plan': {key: value for key, value in plan.items() if key != 'queries'},
                'lock_precondition': lock, 'benchmark_executed': False, 'queries': {}}
    connection = None
    try:
        connection = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        if connection.execute('SELECT current_database()').fetchone()[0] != DATABASE:
            raise ValueError(f'Fixtures must be allocated only in the existing {DATABASE} database')
        bindings = {'version': VERSION, 'status': 'postgresql_validated', 'run_id': run_id,
                    'created_at_utc': stamp(), 'benchmark_executed': False, 'queries': {}}
        for number, (query_id, prepared) in enumerate(plan['queries'].items()):
            require_owned_lock(lock_fd)
            metadata = prepared['metadata']
            indexes = auxiliary_indexes(metadata['benchmark'], create_indexes)
            specs, case_evidence = [], []
            evidence['queries'][query_id] = {'plan_manifest': prepared['manifest_path'],
                                           'plan_manifest_sha256': prepared['manifest_sha256'],
                                           'candidate_archive': prepared['candidate_archive'], 'cases': case_evidence}
            for case in metadata['cases']:
                require_owned_lock(lock_fd)
                namespace = f'qw2_{run_id}_{number}_{case["name"]}'
                # Namespace is globally fresh. Record it before creation so even
                # a connection failure immediately after COMMIT is recoverable.
                owned.append(namespace)
                counts = _create_case(connection, namespace, owner, prepared, case, indexes, timeout_ms)
                search_path = f'"{namespace}", pg_catalog'
                spec = {'name': case['name'], 'dsn': dsn, 'search_path': search_path,
                        'schema': namespace, 'schema_sha256': metadata['source_schema_sha256'],
                        'rows_by_table': counts, 'rows_inserted': sum(counts.values()),
                        'constraint_rejections': 0, 'coverage_complete': True,
                        'data_sha256': case['data_sha256'], 'auxiliary_indexes': indexes,
                        'case_features': case['features'], 'data_version': run_id}
                reference = specs[0] if specs else spec
                result = {'name': case['name'], 'schema': namespace, 'rows_by_table': counts,
                          'rows_inserted': sum(counts.values()), 'candidate_checks': []}
                case_evidence.append(result)
                with Postgres(dsn, timeout_ms=timeout_ms, search_path=search_path,
                              max_rows=10000, comparison_mode='exact', comparison_profile='strict_metadata') as db:
                    with Postgres(dsn, timeout_ms=timeout_ms, search_path=reference['search_path']) as source_db:
                        catalog = validate_verification_database(source_db, db, prepared['files'][metadata['schema_file']].decode(),
                                                                 performance_data_version=run_id, verification_data_version=run_id)
                    result['catalog_validation'] = catalog
                    if catalog.get('compatible') is not True:
                        raise ValueError('Exact source-DDL fixture catalog validation failed: ' + catalog.get('reason', 'unknown'))
                    spec['catalog_validation'] = {'compatible': True, 'catalog_sha256': catalog['catalog_sha256'],
                                                  'source_ddl_executed_exactly': True}
                    result['original_rows'] = _original_expectation(db, prepared['original'], case['expected_original_rows'])
                    original_check = db.compare(prepared['original'], prepared['original'])
                    result['original_strict_self_check'] = original_check
                    require_strict_comparison(original_check)
                    result['original_fingerprint'] = original_check['original']
                    for candidate in prepared['candidates']:
                        comparison = db.compare(prepared['original'], candidate['sql'])
                        check = {key: value for key, value in candidate.items() if key != 'sql'}
                        check['comparison'] = comparison
                        result['candidate_checks'].append(check)
                        require_strict_comparison(comparison)
                        if 'original_fingerprint' in result and result['original_fingerprint'] != comparison['original']:
                            raise ValueError('Original fingerprint changed between saved candidate checks')
                        result['original_fingerprint'] = comparison['original']
                specs.append(spec)
            nonvacuity = validate_output_coverage(case_evidence, prepared)
            primary = next(spec for spec in specs if spec['name'] == metadata['suggested_primary_case'])
            primary_case = next(case for case in case_evidence if case['name'] == primary['name'])
            if primary['name'] == 'empty' or (not primary_case['original_rows'] and not nonvacuity.get('structural_empty_proven')):
                raise ValueError('Suggested primary must have a measured nonempty original output')
            validation = {'postgresql_executed': True, 'original_expectations_passed': True,
                          'saved_candidate_checks_passed': True, **nonvacuity,
                          'candidate_comparison_mode': 'exact', 'candidate_comparison_profile': 'strict_metadata',
                          'finite_evidence_only': not nonvacuity.get('structural_empty_proven', False),
                          'saved_candidate_sql_count': len(prepared['candidates']),
                          'saved_nonoriginal_sql_count': sum(candidate['sql'] != prepared['original'] for candidate in prepared['candidates']),
                          'candidate_archive': prepared['candidate_archive'], 'benchmark_executed': False}
            if not prepared['candidates']:
                validation['saved_candidate_checks_not_applicable_reason'] = prepared['candidate_archive']['not_applicable_reason']
            binding = {'query_sha256': metadata['original_checkpoint_sha256'],
                       'schema_sha256': metadata['source_schema_sha256'], 'source_tables': metadata['source_tables'],
                       'verification_database': {key: primary[key] for key in ('dsn', 'search_path', 'schema', 'data_version')},
                       'mini_dbs': specs, 'validation': validation}
            if query_id in performance_databases:
                main = performance_databases[query_id]
                with Postgres(main['dsn'], timeout_ms=timeout_ms, search_path=main.get('search_path', 'public')) as performance:
                    with Postgres(dsn, timeout_ms=timeout_ms, search_path=primary['search_path']) as verification:
                        audit = validate_verification_database(performance, verification, prepared['files'][metadata['schema_file']].decode(),
                                                                 verification_data_version=run_id)
                evidence['queries'][query_id]['performance_catalog_validation'] = audit
                if audit.get('compatible') is not True:
                    raise ValueError('Performance/fixture catalog compatibility failed: ' + audit.get('reason', 'unknown'))
            bindings['queries'][query_id] = binding
            evidence['queries'][query_id]['validation'] = validation
        require_owned_lock(lock_fd)
        evidence.update(status='postgresql_validated', completed_at_utc=stamp(), owned_schemas=owned)
        write_new_json(evidence_path, evidence)
        # Final success marker is written last. A failed run has no binding.
        write_new_json(output_path, bindings)
        return bindings
    except BaseException as error:
        cleanup = []
        if connection is not None:
            try:
                if connection.closed:
                    connection = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
                cleanup = _drop_owned(connection, owned, owner)
            except Exception as cleanup_error:
                cleanup = [{'error_type': type(cleanup_error).__name__, 'error': str(cleanup_error)}]
        evidence.update(status='failed', failed_at_utc=stamp(), error_type=type(error).__name__, error=str(error),
                        allocated_schemas=owned, cleanup_errors=cleanup,
                        ready_binding_written=False)
        write_new_json(failure_path, evidence)
        raise
    finally:
        if connection is not None:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path, help='immutable offline plan directory')
    parser.add_argument('--output', required=True, type=Path, help='fresh binding JSON path; evidence/failure sidecars use the same stem')
    parser.add_argument('--dsn', required=True, help='existing mcts_fixtures database')
    parser.add_argument('--lock-fd', required=True, type=int, help='inherited descriptor already owning the shared exclusive flock')
    parser.add_argument('--timeout-ms', type=int, default=10000)
    parser.add_argument('--no-auxiliary-indexes', action='store_true')
    parser.add_argument('--performance-databases', type=Path, help='optional JSON mapping query IDs to dsn/search_path for catalog-only compatibility audits')
    args = parser.parse_args()
    performance = json.loads(args.performance_databases.read_text()) if args.performance_databases else None
    bindings = materialize(args.plan, args.output, args.dsn, lock_fd=args.lock_fd,
                           timeout_ms=args.timeout_ms, create_indexes=not args.no_auxiliary_indexes,
                           performance_databases=performance)
    print(json.dumps({'status': bindings['status'], 'bindings': str(args.output.resolve()),
                      'queries': list(bindings['queries']), 'benchmark_executed': False}, indent=2))


if __name__ == '__main__':
    main()
