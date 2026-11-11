"""Optional, explicitly scoped verification on a separate PostgreSQL database.

Catalog compatibility is a prerequisite, not evidence of equal table contents.
Plans and timings still belong to the performance database. Hash comparisons
cover the configured test rows and miniature rows, never all performance rows.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, contextmanager

import psycopg
from psycopg import sql as psql
import sqlglot
from sqlglot import exp, ErrorLevel
from sqlglot.optimizer.scope import traverse_scope

from .db import Postgres, parse_read_query, _SAFE_FUNCTIONS
from .verify import EquivalenceGate, _pg_identifier_name


_SETTINGS = (
    'server_version_num', 'server_encoding', 'client_encoding', 'DateStyle',
    'IntervalStyle', 'TimeZone', 'timezone_abbreviations', 'standard_conforming_strings',
    'backslash_quote', 'escape_string_warning', 'extra_float_digits', 'bytea_output',
    'lc_numeric', 'lc_monetary', 'lc_time', 'default_text_search_config',
    'transform_null_equals', 'row_security', 'xmloption',
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class VerificationDatabaseError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def _tables_from_schema(schema_sql):
    tables = []
    for statement in sqlglot.parse(schema_sql, read='postgres', error_level=ErrorLevel.RAISE):
        if statement is None:
            continue
        if not isinstance(statement, exp.Create) or statement.kind != 'TABLE' or not isinstance(statement.this, exp.Schema):
            raise ValueError('Verification context requires source CREATE TABLE statements')
        table = statement.this.this
        if not isinstance(table, exp.Table) or table.db or table.catalog:
            raise ValueError('Verification context requires unqualified source table names')
        name = _pg_identifier_name(table)
        if name in tables:
            raise ValueError('Verification schema declares duplicate table names')
        tables.append(name)
    if not tables:
        raise ValueError('Verification context requires a nonempty source schema')
    return sorted(tables)


def _collation(cursor, oid):
    if not oid:
        return None
    cursor.execute('SELECT n.nspname,to_jsonb(c),pg_collation_actual_version(c.oid) '
                   'FROM pg_collation c JOIN pg_namespace n ON n.oid=c.collnamespace WHERE c.oid=%s', (oid,))
    found = cursor.fetchone()
    if not found:
        raise ValueError('Missing referenced collation')
    namespace, raw, actual = found
    if raw.get('collisdeterministic') is not True:
        raise ValueError('Nondeterministic collations are unsupported')
    stored = raw.get('collversion')
    if stored is not None and stored != actual:
        raise ValueError('Stored and actual collation versions differ')
    keys = ('collname', 'collprovider', 'collencoding', 'collisdeterministic',
            'collcollate', 'collctype', 'colliculocale', 'collicurules', 'colllocale', 'collversion')
    return {'namespace': namespace, **{key: raw.get(key) for key in keys}, 'actual_version': actual}


def _catalog_snapshot(db, table_names, data_version=None):
    """Read catalog semantics in one read-only snapshot; do not scan table rows."""
    with db.connection.transaction():
        with db.connection.cursor() as cursor:
            db._configure(cursor)
            cursor.execute('SELECT current_database(),current_user,version(),'
                           'pg_postmaster_start_time()::text,pg_current_wal_insert_lsn()::text')
            database, user, version, started, wal = cursor.fetchone()
            cursor.execute('SELECT name,setting FROM pg_settings WHERE lower(name)=ANY(%s) ORDER BY name',
                           ([name.lower() for name in _SETTINGS],))
            settings = {name.lower(): setting for name, setting in cursor.fetchall()}
            if set(settings) != {name.lower() for name in _SETTINGS} or settings.get('row_security') != 'on':
                raise ValueError('Missing semantic setting or row_security is disabled')
            cursor.execute('SELECT to_jsonb(d) FROM pg_database d WHERE datname=current_database()')
            raw_database = cursor.fetchone()[0]
            locale_keys = ('encoding', 'datcollate', 'datctype', 'datlocprovider', 'datlocale',
                           'daticulocale', 'daticurules', 'datcollversion')
            locale = {key: raw_database.get(key) for key in locale_keys}
            if int(settings['server_version_num']) >= 150000:
                cursor.execute('SELECT pg_database_collation_actual_version(oid) FROM pg_database WHERE datname=current_database()')
                locale['actual_version'] = cursor.fetchone()[0]
                if locale['datcollversion'] is not None and locale['actual_version'] != locale['datcollversion']:
                    raise ValueError('Database collation version is stale')
            elif locale['datcollate'] in {'C', 'POSIX'} and locale['datctype'] in {'C', 'POSIX'}:
                locale['actual_version'] = 'bytewise C/POSIX'
            else:
                raise ValueError('This server cannot audit the non-C database locale version')
            cursor.execute('SELECT abbrev,utc_offset::text,is_dst FROM pg_timezone_abbrevs ORDER BY abbrev')
            settings['timezone_abbreviation_definitions_sha256'] = _digest(cursor.fetchall())
            # Overloaded allowed functions or operators can execute code that
            # does not belong to the deterministic builtin query model.
            allowed = {name.lower() for name in _SAFE_FUNCTIONS} | {'date_trunc', 'strpos', 'bit_and', 'bit_or', 'bit_xor'}
            cursor.execute("SELECT n.nspname,p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "WHERE n.nspname=ANY(current_schemas(true)) AND n.nspname<>'pg_catalog' AND p.proname=ANY(%s) LIMIT 1",
                           (sorted(allowed),))
            if cursor.fetchone():
                raise ValueError('A visible user-defined deterministic-function overload is unsupported')
            cursor.execute("SELECT n.nspname,o.oprname FROM pg_operator o JOIN pg_namespace n ON n.oid=o.oprnamespace "
                           "WHERE n.nspname=ANY(current_schemas(true)) AND n.nspname<>'pg_catalog' LIMIT 1")
            if cursor.fetchone():
                raise ValueError('Visible user-defined operators are unsupported')
            cursor.execute("SELECT t.typname FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
                           "WHERE n.nspname=ANY(current_schemas(true)) AND n.nspname<>'pg_catalog' "
                           "AND EXISTS(SELECT 1 FROM pg_type b JOIN pg_namespace bn ON bn.oid=b.typnamespace "
                           "WHERE bn.nspname='pg_catalog' AND b.typname=t.typname) LIMIT 1")
            if cursor.fetchone():
                raise ValueError('A visible user-defined type shadows a builtin type')
            references = [psql.Identifier(name).as_string(db.connection) for name in table_names]
            cursor.execute("SELECT names.ordinality,c.oid,n.nspname,c.relname,c.relkind,c.relpersistence,"
                           "c.relrowsecurity,c.relforcerowsecurity,c.relispartition,am.amname "
                           "FROM unnest(%s::text[]) WITH ORDINALITY names(name,ordinality) "
                           "LEFT JOIN pg_class c ON c.oid=to_regclass(names.name) "
                           "LEFT JOIN pg_namespace n ON n.oid=c.relnamespace LEFT JOIN pg_am am ON am.oid=c.relam "
                           "ORDER BY names.ordinality", (references,))
            relations, resolved = {}, {}
            for number, oid, namespace, actual_name, kind, persistence, rls, force_rls, partition, method in cursor.fetchall():
                name = table_names[number - 1]
                if not oid or kind not in {'r', 'p'} or rls or force_rls or partition or method not in {'heap', None}:
                    raise ValueError(f'Unsupported/missing/nonbase/RLS relation: {name}')
                relations[oid] = name
                resolved[name] = {'schema': namespace, 'relation': actual_name, 'oid': oid,
                                  'persistence': persistence}
            oids = list(relations)
            cursor.execute('SELECT inhrelid,inhparent FROM pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s)', (oids, oids))
            if cursor.fetchone():
                raise ValueError('Inherited/partitioned relation contexts are unsupported')
            cursor.execute("SELECT i.indexrelid,n.nspname,am.amname,i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                           "JOIN pg_am am ON am.oid=c.relam JOIN pg_opclass o ON o.oid=ANY(i.indclass::oid[]) "
                           "JOIN pg_namespace n ON n.oid=o.opcnamespace WHERE i.indrelid=ANY(%s)", (oids,))
            indexes = set()
            for oid, namespace, method, valid in cursor.fetchall():
                indexes.add(oid)
                if namespace != 'pg_catalog' or method not in {'btree', 'hash', 'gist', 'gin', 'spgist', 'brin'} or not valid:
                    raise ValueError('Custom or invalid index semantics are unsupported')
            cursor.execute("SELECT a.attrelid,a.attnum,a.attname,n.nspname,t.typname,t.typtype,a.atttypmod,"
                           "a.attnotnull,a.attndims,a.attcollation,a.attidentity,a.attgenerated,"
                           "pg_get_expr(d.adbin,d.adrelid,false),d.oid,format_type(a.atttypid,a.atttypmod) FROM pg_attribute a "
                           "JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace "
                           "LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
                           "WHERE a.attrelid=ANY(%s) AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attrelid,a.attnum", (oids,))
            tables = {name: {'columns': [], 'constraints': []} for name in table_names}
            attribute_names, collation_cache, defaults = {}, {}, []
            for relid, number, name, namespace, typename, kind, typmod, notnull, dimensions, collid, identity, generated, default, default_oid, formatted_type in cursor.fetchall():
                if namespace != 'pg_catalog' or kind not in {'b', 'm', 'r'}:
                    raise ValueError(f'Unsupported custom/domain/pseudo column type: {relations[relid]}.{name}')
                attribute_names[(relid, number)] = name
                if collid not in collation_cache:
                    collation_cache[collid] = _collation(cursor, collid)
                tables[relations[relid]]['columns'].append({'name': name, 'type_schema': namespace, 'type': typename,
                    'typmod': typmod, 'formatted_type': formatted_type, 'nullable': not notnull, 'dimensions': dimensions,
                    'collation': collation_cache[collid], 'identity': identity, 'generated': generated, 'default': default})
                if default_oid: defaults.append(default_oid)
            cursor.execute('SELECT oid,conrelid,contype,convalidated,condeferrable,condeferred,connoinherit,'
                           'confrelid,conkey,confkey,confupdtype,confdeltype,confmatchtype,'
                           'pg_get_constraintdef(oid,false) FROM pg_constraint WHERE conrelid=ANY(%s) ORDER BY oid', (oids,))
            constraints = []
            for oid, relid, kind, validated, deferrable, deferred, noinherit, target, keys, target_keys, update, delete, match, definition in cursor.fetchall():
                if kind not in {'p', 'f', 'u', 'c', 'n'} or not validated:
                    raise ValueError('Unsupported or unvalidated constraint')
                if target and target not in relations:
                    raise ValueError('Foreign key references an unaudited relation')
                constraints.append(oid)
                tables[relations[relid]]['constraints'].append({'kind': kind, 'validated': validated,
                    'deferrable': deferrable, 'initially_deferred': deferred, 'no_inherit': noinherit,
                    'columns': [attribute_names[(relid, n)] for n in keys or []],
                    'target': relations.get(target),
                    'target_columns': [attribute_names[(target, n)] for n in target_keys or []],
                    'update_action': update, 'delete_action': delete, 'match': match, 'definition': definition})
            # Constraint/default/generated expressions must not hide user code.
            cursor.execute("SELECT d.refclassid::regclass::text,pn.nspname,p.provolatile,onsp.nspname "
                           "FROM pg_depend d LEFT JOIN pg_proc p ON d.refclassid='pg_proc'::regclass AND p.oid=d.refobjid "
                           "LEFT JOIN pg_namespace pn ON pn.oid=p.pronamespace "
                           "LEFT JOIN pg_operator o ON d.refclassid='pg_operator'::regclass AND o.oid=d.refobjid "
                           "LEFT JOIN pg_namespace onsp ON onsp.oid=o.oprnamespace "
                           "WHERE (d.classid='pg_constraint'::regclass AND d.objid=ANY(%s)) OR "
                           "(d.classid='pg_attrdef'::regclass AND d.objid=ANY(%s)) OR "
                           "(d.classid='pg_class'::regclass AND d.objid=ANY(%s))", (constraints, defaults, list(indexes)))
            for object_class, procedure_namespace, volatility, operator_namespace in cursor.fetchall():
                if (object_class == 'pg_proc' and (procedure_namespace != 'pg_catalog' or volatility != 'i') or
                        object_class == 'pg_operator' and operator_namespace != 'pg_catalog'):
                    raise ValueError('Constraint/default/generated expression uses unsupported nonimmutable or user code')
            for table in tables.values():
                table['constraints'].sort(key=lambda item: json.dumps(item, sort_keys=True))
            canonical = {'tables': tables, 'settings': settings, 'database_locale': locale}
            identity = {'database': database, 'user': user, 'server_version': version, 'server_started_at': started,
                        'wal_insert_lsn': wal, 'data_version': data_version, 'search_path': db.search_path}
            resume_safe = bool(data_version) or all(r['persistence'] == 'p' for r in resolved.values())
            return {'canonical': canonical, 'catalog_sha256': _digest(canonical), 'resolved_relations': resolved,
                    'identity': identity, 'resume_safe': resume_safe,
                    'environment_sha256': _digest({'identity': identity, 'catalog': canonical}),
                    'fingerprint_method': 'server start + cluster WAL insert LSN + catalog/settings + configured data_version'}


def validate_verification_database(performance_db, verification_db, schema_sql, *, performance_data_version=None,
                                   verification_data_version=None):
    result = {'compatible': False, 'status': 'error', 'schema_source_sha256': hashlib.sha256(schema_sql.encode()).hexdigest()}
    try:
        names = _tables_from_schema(schema_sql)
        performance = _catalog_snapshot(performance_db, names, performance_data_version)
        test = _catalog_snapshot(verification_db, names, verification_data_version)
        result.update(performance=performance, verification=test, tables=names)
        if performance['canonical'] != test['canonical']:
            mismatches = []
            for section in ('settings', 'database_locale'):
                if performance['canonical'][section] != test['canonical'][section]: mismatches.append(section)
            mismatches.extend(f'table:{name}' for name in names if
                performance['canonical']['tables'][name] != test['canonical']['tables'][name])
            result.update(status='mismatch', reason='Catalog or semantic settings differ', mismatches=mismatches)
        else:
            # SQLSolver receives source DDL. A matching pair of live catalogs
            # must also preserve its column order and type domains, especially
            # for SELECT * and bounded character/numeric types.
            for statement in sqlglot.parse(schema_sql, read='postgres', error_level=ErrorLevel.RAISE):
                if statement is None: continue
                name = _pg_identifier_name(statement.this.this)
                declared = [column for column in statement.this.expressions if isinstance(column, exp.ColumnDef)]
                actual = performance['canonical']['tables'][name]['columns']
                if [_pg_identifier_name(c) for c in declared] != [c['name'] for c in actual]:
                    raise ValueError(f'Source DDL column names/order differ from actual catalog: {name}')
                if any(column.args['kind'].sql(dialect='postgres') !=
                       exp.DataType.build(live['formatted_type'], dialect='postgres').sql(dialect='postgres')
                       for column, live in zip(declared, actual)):
                    raise ValueError(f'Source DDL types/typmods differ from actual catalog: {name}')
            result.update(compatible=True, status='ok', catalog_sha256=performance['catalog_sha256'],
                          reason='Matching audited base-table catalogs and deterministic session semantics')
    except Exception as error:
        result.update(reason=f'{type(error).__name__}: {error}', error_type=type(error).__name__,
                      sqlstate=getattr(error, 'sqlstate', None))
    return result


def _safe_provenance(value):
    if isinstance(value, dict):
        return {k: _safe_provenance(v) for k, v in value.items() if k.lower() not in
                {'dsn', 'password', 'read_dsn', 'admin_dsn', 'token', 'secret', 'api_key'}}
    if isinstance(value, list): return [_safe_provenance(v) for v in value]
    return value


class _ScopedDatabase:
    def __init__(self, owner): self.owner = owner
    def __getattr__(self, name): return getattr(self.owner.verification_db, name)
    def compare(self, original, candidate): return self.owner._compare(original, candidate)
    def planner_hints_enabled(self):
        main = self.owner.performance_db.planner_hints_enabled()
        test = self.owner.verification_db.planner_hints_enabled()
        return {'enabled': bool(main.get('enabled') and test.get('enabled')), 'performance': main, 'verification': test}
    def formal_context_supported(self, original, candidate, schema_sql=None):
        main = self.owner.performance_db.formal_context_supported(original, candidate, schema_sql=schema_sql)
        test = self.owner.verification_db.formal_context_supported(original, candidate, schema_sql=schema_sql)
        return {'supported': bool(main.get('supported') and test.get('supported')), 'performance': main, 'verification': test}


class ScopedEquivalenceGate:
    """Gate on a schema-equivalent test DB while the caller measures the main DB.

    The caller owns both connections. Catalog drift and unknown query relation
    scope fail closed. Main-DB EXPLAIN checks do not execute or hash its results.
    """
    def __init__(self, performance_db, verification_db, schema_sql, mode='strict', mini_dbs=None,
                 solver_timeout=10, performance_provenance=None, verification_provenance=None,
                 gate_class=EquivalenceGate):
        self.performance_db, self.verification_db, self.schema_sql = performance_db, verification_db, schema_sql
        self.performance_provenance = _safe_provenance(performance_provenance or {})
        self.verification_provenance = _safe_provenance(verification_provenance or {})
        self.validation = self._validate()
        if not self.validation['compatible']:
            raise VerificationDatabaseError('Separate verification database is incompatible: ' + self.validation['reason'], self.validation)
        self._plans = set()
        self.db = _ScopedDatabase(self)
        self.mode = mode
        self.gate = gate_class(self.db, schema_sql, mode=mode, mini_dbs=mini_dbs, solver_timeout=solver_timeout)
        self.metadata = {'enabled': True, 'scope': 'separate_verification_database',
            'performance_database': self.performance_provenance, 'verification_database': self.verification_provenance,
            'catalog_validation': self.validation, 'performance_database_rows_hashed': False,
            'result_evidence': 'Finite rows of the configured verification database and configured miniature cases; performance-database rows are not hashed',
            'timing_scope': 'Designated performance database; supplied separately to search and holdout evaluation'}

    def _validate(self):
        return validate_verification_database(self.performance_db, self.verification_db, self.schema_sql,
            performance_data_version=self.performance_provenance.get('data_version'),
            verification_data_version=self.verification_provenance.get('data_version'))

    def _query_scope(self, sql, current):
        query = parse_read_query(sql)
        if query.find(exp.Collate):
            raise ValueError('Explicit query collations require a separately audited context')
        for scope in traverse_scope(query):
            for _, source in scope.selected_sources.values():
                if not isinstance(source, exp.Table): continue
                name = _pg_identifier_name(source)
                if source.catalog or name not in current['tables']:
                    raise ValueError('Query references a relation outside the audited benchmark schema')
                qualifier = _pg_identifier_name(source.args['db']) if source.db else None
                if qualifier and any(qualifier != current[side]['resolved_relations'][name]['schema']
                                     for side in ('performance', 'verification')):
                    raise ValueError('Qualified query table cannot be remapped between verification schemas')
        return query

    @contextmanager
    def _catalog_locks(self):
        # Separate short-lived guard connections keep existing Postgres.compare
        # transactions unchanged. ACCESS SHARE permits ordinary reads/writes
        # while blocking table/column/constraint DDL throughout verification.
        with ExitStack() as stack:
            for side, db in (('performance', self.performance_db), ('verification', self.verification_db)):
                connection = stack.enter_context(psycopg.connect(db.dsn, autocommit=True, connect_timeout=5))
                stack.enter_context(connection.transaction())
                cursor = stack.enter_context(connection.cursor())
                db._configure(cursor)
                relations = self.validation[side]['resolved_relations']
                cursor.execute(psql.SQL('LOCK TABLE {} IN ACCESS SHARE MODE').format(psql.SQL(',').join(
                    psql.Identifier(relation['schema'], relation['relation']) for relation in relations.values())))
            yield

    def _compare(self, original, candidate):
        scope = {'kind': 'separate_verification_database', 'performance_database_rows_hashed': False,
                 'verification_database': self.validation['verification']['identity']['database'],
                 'catalog_sha256': self.validation['catalog_sha256']}
        try:
            current = self._validate()
            if not current.get('compatible') or current.get('catalog_sha256') != self.validation['catalog_sha256']:
                raise VerificationDatabaseError('Catalog/settings compatibility changed during verification', current)
            for side in ('performance', 'verification'):
                if current[side]['identity']['server_started_at'] != self.validation[side]['identity']['server_started_at']:
                    raise VerificationDatabaseError('Database server restarted during verification', current)
                if current[side]['resolved_relations'] != self.validation[side]['resolved_relations']:
                    raise VerificationDatabaseError('Audited relation identity/resolution changed during verification', current)
            for query in (original, candidate):
                self._query_scope(query, current)
                key = hashlib.sha256(query.encode()).hexdigest()
                if key not in self._plans:
                    self.performance_db.explain(query, analyze=False)
                    self._plans.add(key)
            result = self.verification_db.compare(original, candidate)
            return {**result, 'verification_scope': scope,
                    'performance_database_plan_check': {'status': 'ok', 'analyze': False,
                                                       'performance_rows_executed_or_hashed': False}}
        except Exception as error:
            return {'equivalent': False, 'status': 'error', 'same_snapshot': False,
                    'original': None, 'candidate': None, 'verification_scope': scope,
                    'error': f'{type(error).__name__}: {error}', 'error_type': type(error).__name__,
                    'catalog_validation': getattr(error, 'evidence', None), 'sqlstate': getattr(error, 'sqlstate', None)}

    def verify(self, original, candidate):
        try:
            with self._catalog_locks():
                result = self.gate.verify(original, candidate)
                if not isinstance(result, dict): raise TypeError('Equivalence gate returned no evidence dictionary')
        except Exception as error:
            result = {'accepted': False, 'formal': False, 'proof': 'none', 'execution': {},
                      'reason': f'Catalog guard failed: {type(error).__name__}: {error}',
                      'error_type': type(error).__name__, 'sqlstate': getattr(error, 'sqlstate', None)}
        return {**result, 'evidence_scope': {k: self.metadata[k] for k in
            ('scope', 'performance_database_rows_hashed', 'result_evidence', 'timing_scope')},
            'catalog_compatibility_sha256': self.validation['catalog_sha256']}
