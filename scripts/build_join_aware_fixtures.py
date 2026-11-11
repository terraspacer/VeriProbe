#!/usr/bin/env python3
"""Build fresh, join-aware finite fixtures without changing an active run.

The original source DDL remains authoritative. Query literals and resolved SQL
equality joins seed data; these examples never constitute an equivalence proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
import sys
import uuid

import psycopg
from psycopg import sql
import sqlglot
from sqlglot import exp, ErrorLevel
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.db import Postgres, parse_read_query
from mcts_rewrite.verify import _pg_identifier_name, _pooled_value, _value_for, _synthesize_cases

CASES = ('empty', 'null', 'duplicate', 'skew', 'unmatched')
PROBES = {'tpch': ('rbot:tpch:query17_0', 'rbot:tpch:query20_0'),
          'dsb': ('rbot:dsb:query032_0',)}
NUMERIC = {'INT', 'BIGINT', 'SMALLINT', 'TINYINT', 'DECIMAL', 'DOUBLE', 'FLOAT', 'REAL'}
SUPPORTED = NUMERIC | {'TEXT', 'VARCHAR', 'CHAR', 'BOOLEAN', 'DATE', 'TIME', 'TIMETZ', 'TIMESTAMP', 'TIMESTAMPTZ'}


def json_value(value):
    if isinstance(value, (Decimal, date, datetime, time)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    return value


def parse_schema(source):
    tables = {}
    statements = sqlglot.parse(source, read='postgres', error_level=ErrorLevel.RAISE)
    for statement in statements:
        if statement is None:
            continue
        if not isinstance(statement, exp.Create) or statement.kind != 'TABLE' or not isinstance(statement.this, exp.Schema):
            raise ValueError('Only source CREATE TABLE statements are supported')
        table = statement.this.this
        if table.db or table.catalog:
            raise ValueError('Qualified source tables cannot be remapped')
        name = _pg_identifier_name(table)
        columns = {_pg_identifier_name(c): c for c in statement.this.expressions if isinstance(c, exp.ColumnDef)}
        if name in tables or not columns:
            raise ValueError('Duplicate/empty source table')
        for column in columns.values():
            if column.args['kind'].this.value not in SUPPORTED:
                raise ValueError(f'Unsupported source type: {column.sql()}')
        primary, unique, foreign = set(), set(), []
        for node in statement.this.walk():
            if isinstance(node, exp.PrimaryKey):
                primary.update(_pg_identifier_name(c.this if isinstance(c, exp.Ordered) else c)
                               for c in node.expressions)
            elif isinstance(node, exp.ForeignKey):
                reference = node.args.get('reference')
                if not reference or not isinstance(reference.this, exp.Schema):
                    raise ValueError('Unsupported foreign-key shape')
                foreign.extend(zip((_pg_identifier_name(c) for c in node.expressions),
                                   itertools.repeat(_pg_identifier_name(reference.this.this)),
                                   (_pg_identifier_name(c) for c in reference.this.expressions)))
        for column_name, column in columns.items():
            if column.find(exp.PrimaryKeyColumnConstraint):
                primary.add(column_name)
            if column.find(exp.UniqueColumnConstraint):
                unique.add(column_name)
            reference = column.find(exp.Reference)
            if reference is not None:
                if not isinstance(reference.this, exp.Schema) or len(reference.this.expressions) != 1:
                    raise ValueError('Unsupported inline foreign-key shape')
                foreign.append((column_name, _pg_identifier_name(reference.this.this),
                                _pg_identifier_name(reference.this.expressions[0])))
        nullable = {name for name, column in columns.items() if name not in primary and
                    not column.find(exp.NotNullColumnConstraint)}
        tables[name] = {'columns': columns, 'primary': primary, 'unique': unique,
                        'foreign': foreign, 'nullable': nullable}
    if not tables:
        raise ValueError('Empty schema')
    return tables


def constant(node):
    if isinstance(node, (exp.Cast, exp.Paren)):
        return constant(node.this)
    if isinstance(node, exp.Literal):
        return node.this if node.is_string else Decimal(node.this)
    if isinstance(node, exp.Neg):
        value = constant(node.this)
        return -value if isinstance(value, Decimal) else None
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        left, right = constant(node.this), constant(node.expression)
        if isinstance(left, Decimal) and isinstance(right, Decimal):
            if isinstance(node, exp.Add): return left + right
            if isinstance(node, exp.Sub): return left - right
            if isinstance(node, exp.Mul): return left * right
            if right: return left / right
    return None


class Evidence:
    """Resolve aliases/CTE projections; never infer joins merely from names."""
    def __init__(self, tables):
        self.tables = tables
        self.parents = {}
        self.joins, self.literals, self.issues = [], {}, []
        self.aggregate_columns, self.denominators = set(), set()
        self.profiles = {}
        for table, spec in tables.items():
            for column, target, target_column in spec['foreign']:
                self.add_join((table, column), (target, target_column),
                              {'kind': 'source_foreign_key'})

    def root(self, ref):
        self.parents.setdefault(ref, ref)
        if self.parents[ref] != ref:
            self.parents[ref] = self.root(self.parents[ref])
        return self.parents[ref]

    def add_join(self, left, right, provenance):
        if left == right or left[0] not in self.tables or right[0] not in self.tables:
            return
        self.parents[self.root(right)] = self.root(left)
        self.joins.append({'left': '.'.join(left), 'right': '.'.join(right), **provenance})

    def add_literal(self, ref, value, provenance):
        try:
            typed = coerce(self.tables[ref[0]]['columns'][ref[1]], value)
        except (ValueError, TypeError, ArithmeticError):
            return
        pool = self.literals.setdefault(ref, [])
        if typed not in [item['value'] for item in pool]:
            pool.append({'value': typed, **provenance})

    def analyze(self, row):
        source_id = row['id']
        try:
            parsed = sqlglot.parse(row['sql'], read='postgres', error_level=ErrorLevel.RAISE)
            if len([p for p in parsed if p is not None]) != 1:
                raise ValueError('not one statement')
            tree = parsed[0]
            schema = {t: {c: spec['columns'][c].args['kind'].sql(dialect='postgres')
                          for c in spec['columns']} for t, spec in self.tables.items()}
            try:
                tree = qualify(tree.copy(), dialect='postgres', schema=schema,
                               infer_schema=False, expand_stars=False,
                               validate_qualify_columns=False, quote_identifiers=False,
                               identify=False, allow_partial_qualification=True)
            except Exception:
                pass  # The resolver below still rejects ambiguous references.
            scopes = list(traverse_scope(tree))
            owners = {}
            for scope in scopes:
                for column in scope.columns:
                    owners.setdefault(id(column), scope)

            def from_source(source, name, visiting):
                if isinstance(source, exp.Table):
                    table = _pg_identifier_name(source)
                    if table in self.tables and name in self.tables[table]['columns']:
                        return (table, name)
                    return None
                if isinstance(source, Scope):
                    projections = [p for p in source.expression.selects if p.alias_or_name == name]
                    if len(projections) != 1:
                        return None
                    projection = projections[0]
                    if isinstance(projection, exp.Alias): projection = projection.this
                    if isinstance(projection, exp.Column): return resolve(projection, source, visiting)
                return None

            def resolve(column, scope=None, visiting=None):
                if not isinstance(column, exp.Column): return None
                scope = owners.get(id(column), scope)
                visiting = set() if visiting is None else set(visiting)
                if scope is None or id(column) in visiting: return None
                visiting.add(id(column))
                name = _pg_identifier_name(column)
                qualifier = column.table
                while scope is not None:
                    sources = scope.selected_sources
                    if qualifier:
                        item = sources.get(qualifier)
                        if item is not None: return from_source(item[1], name, visiting)
                    else:
                        matches = [ref for _, source in sources.values()
                                   if (ref := from_source(source, name, visiting)) is not None]
                        if len(matches) == 1: return matches[0]
                        if matches: return None
                    scope = scope.parent
                return None

            provenance = {'kind': 'query_ast', 'source_id': source_id,
                          'query_sha256': hashlib.sha256(row['sql'].encode()).hexdigest()}
            choices, ratios = {}, []
            for node in tree.walk():
                detail = {**provenance, 'predicate': node.sql(dialect='postgres')}
                if isinstance(node, exp.EQ):
                    left, right = resolve(node.this), resolve(node.expression)
                    if left and right: self.add_join(left, right, detail)
                    elif isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Column):
                        self.issues.append({'source_id': source_id, 'predicate': detail['predicate'],
                                            'error': 'ambiguous or unresolved equality-column lineage'})
                if isinstance(node, exp.AggFunc):
                    self.aggregate_columns.update(ref for c in node.find_all(exp.Column) if (ref := resolve(c)))
                if isinstance(node, exp.Div):
                    if ref := resolve(node.expression): self.denominators.add(ref)
                column, value = None, None
                if isinstance(node, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)):
                    column, value = resolve(node.this), constant(node.expression)
                    if column is None and isinstance(node, exp.EQ):
                        column, value = resolve(node.expression), constant(node.this)
                    if isinstance(node, exp.Like) and isinstance(value, str):
                        value = value.replace('%', '').replace('_', 'x')
                    elif isinstance(node, (exp.GT, exp.LT)) and isinstance(value, Decimal):
                        value += 1 if isinstance(node, exp.GT) else -1
                elif isinstance(node, exp.In) and node.expressions:
                    column = resolve(node.this)
                    for expression in node.expressions:
                        literal = constant(expression)
                        if column and literal is not None: self.add_literal(column, literal, detail)
                    value = constant(node.expressions[0])
                elif isinstance(node, exp.Between):
                    column, value = resolve(node.this), constant(node.args['low'])
                    if isinstance(node.this, exp.Div):
                        numerator, denominator = resolve(node.this.this), resolve(node.this.expression)
                        low, high = constant(node.args['low']), constant(node.args['high'])
                        if numerator and denominator and isinstance(low, Decimal) and isinstance(high, Decimal):
                            ratios.append((numerator, denominator, (low + high) / 2, detail))
                if column and value is not None:
                    self.add_literal(column, value, detail)
                    try: choices.setdefault(column, coerce(self.tables[column[0]]['columns'][column[1]], value))
                    except (ValueError, TypeError, ArithmeticError): pass
            self.profiles[source_id] = {'id': source_id, 'choices': choices, 'ratios': ratios}
        except Exception as error:
            self.issues.append({'source_id': source_id, 'error': f'{type(error).__name__}: {error}'})

    def groups(self):
        members = {}
        for ref in self.parents:
            members.setdefault(self.root(ref), []).append(ref)
        return {ref: index for index, group in enumerate(sorted(members.values())) for ref in group}

    def serializable(self):
        return json_value({'joins': self.joins, 'join_groups': {'.'.join(ref): group for ref, group in self.groups().items()},
            'literal_pools': {'.'.join(ref): values for ref, values in sorted(self.literals.items())},
            'aggregate_columns': ['.'.join(ref) for ref in sorted(self.aggregate_columns)],
            'denominator_columns': ['.'.join(ref) for ref in sorted(self.denominators)],
            'parse_or_resolution_issues': self.issues})


def coerce(column, value):
    dtype = column.args['kind']
    typed = _pooled_value(dtype, value)
    if dtype.this.value in {'CHAR', 'VARCHAR'} and dtype.expressions:
        limit = int(dtype.expressions[0].this.this)
        if len(typed) > limit: raise ValueError('literal exceeds source text length')
    if dtype.this.value == 'INT' and not -(2**31) <= typed < 2**31:
        raise ValueError('literal exceeds source INT range')
    return typed


def join_value(column, number):
    kind = column.args['kind'].this.value
    if kind == 'BOOLEAN': return bool(number % 2)
    if kind == 'DATE': return date(2000, 1, 1) + timedelta(days=number)
    if kind in {'TIMESTAMP', 'TIMESTAMPTZ'}: return datetime(2000, 1, 1) + timedelta(days=number)
    if kind in {'TIME', 'TIMETZ'}: return time(number % 24, number % 60)
    return coerce(column, number)


def make_rows(tables, evidence, case, profile_ids):
    if case == 'empty': return {t: [] for t in tables}, []
    groups = evidence.groups()
    profiles = [{'id': 'global_literals', 'choices': {}, 'ratios': []}]
    profiles.extend(evidence.profiles[p] for p in profile_ids)
    result = {t: [] for t in tables}
    features = []
    aggregate_tables = {ref[0] for ref in evidence.aggregate_columns}
    for lane, profile in enumerate(profiles):
        for table, spec in tables.items():
            primary = spec['primary'] | spec['unique']
            free_primary = sorted(c for c in primary if (table, c) not in groups)
            primary_groups = sorted({groups[(table, c)] for c in primary if (table, c) in groups})
            if primary and not free_primary:
                coordinates = list(itertools.product(range(3), repeat=len(primary_groups)))
            else:
                coordinates = [()] * (3 if len(primary) == 1 and table not in aggregate_tables else 12)
            for index, coordinate in enumerate(coordinates):
                assigned = dict(zip(primary_groups, coordinate))
                numeric = (0, 0, 10, 100) if case == 'duplicate' else ((0, 1, 1, 100) if case == 'skew' else (0, 1, 10, 100))
                row = {}
                for column_name, column in spec['columns'].items():
                    ref = (table, column_name)
                    dtype = column.args['kind']
                    kind = dtype.this.value
                    if ref in groups:
                        domain_index = assigned.get(groups[ref], index // 4 % 3)
                        if case == 'skew' and not primary and index < 8: domain_index = 0
                        value = join_value(column, lane * 3 + domain_index + 1)
                    elif column_name in primary:
                        value = coerce(column, lane * 1000 + index + 1)
                    elif ref in profile['choices']:
                        value = profile['choices'][ref]
                    elif kind in NUMERIC:
                        pool = evidence.literals.get(ref, [])
                        value = pool[index % len(pool)]['value'] if lane == 0 and pool else coerce(column, numeric[index % 4])
                    else:
                        pool = evidence.literals.get(ref, [])
                        if pool:
                            value = pool[index % len(pool)]['value']
                        elif kind in {'CHAR', 'VARCHAR', 'TEXT'}:
                            value = f'v{CASES.index(case)}{index % 3}'
                            if dtype.expressions: value = value[:int(dtype.expressions[0].this.this)]
                        else:
                            value = _value_for(dtype, index % 3 + 1)
                    if ref in evidence.denominators and value == 0: value = coerce(column, 1)
                    row[column_name] = value
                for numerator, denominator, ratio, _ in profile['ratios']:
                    if numerator[0] == denominator[0] == table:
                        if not row[denominator[1]]: row[denominator[1]] = coerce(spec['columns'][denominator[1]], 1)
                        row[numerator[1]] = coerce(spec['columns'][numerator[1]], Decimal(row[denominator[1]]) * ratio)
                if case == 'null' and index == len(coordinates) - 1 and spec['nullable']:
                    row[sorted(spec['nullable'])[0]] = None
                result[table].append(row)
            if case == 'duplicate' and not primary:
                result[table].append(dict(result[table][-len(coordinates)]))
                features.append({'kind': 'exact_duplicate', 'table': table, 'profile': profile['id']})
            elif case == 'duplicate' and free_primary:
                duplicate = dict(result[table][-len(coordinates)])
                duplicate[free_primary[0]] = coerce(spec['columns'][free_primary[0]], lane * 1000 + 999)
                result[table].append(duplicate)
                features.append({'kind': 'duplicate_payload_with_distinct_key', 'table': table, 'profile': profile['id']})
        if case == 'unmatched':
            candidates = sorted(groups, key=lambda ref: (ref[0] not in aggregate_tables, len(tables[ref[0]]['primary']) == 1, ref))
            for table, column in candidates:
                if any(column == child for child, _, _ in tables[table]['foreign']): continue
                if not any(other[0] != table and group == groups[(table, column)] for other, group in groups.items()): continue
                spec = tables[table]
                unmatched = dict(result[table][-1])
                if spec['columns'][column].args['kind'].this.value not in NUMERIC: continue
                unmatched[column] = coerce(spec['columns'][column], 900000 + lane)
                if column not in spec['primary']:
                    free = [c for c in spec['primary'] if (table, c) not in groups]
                    if not free and spec['primary']: continue
                    if free: unmatched[free[0]] = coerce(spec['columns'][free[0]], lane * 1000 + 998)
                result[table].append(unmatched)
                features.append({'kind': 'unmatched_join_key', 'column': f'{table}.{column}',
                                 'value': 900000 + lane, 'profile': profile['id']})
                break
    return result, features


def replace_fresh_rows(admin_dsn, namespace, tables, rows):
    if not namespace.startswith('mcts_mini_'):
        raise ValueError('Refusing a namespace outside newly allocated miniature schemas')
    counts, rejections = Counter(), []
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        with connection.transaction():
            connection.execute(sql.SQL('TRUNCATE {}').format(sql.SQL(',').join(
                sql.Identifier(namespace, table) for table in tables)))
            pending = [(table, row) for table, values in rows.items() for row in values]
            while pending:
                remaining = []
                for table, row in pending:
                    query = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                        sql.Identifier(namespace, table), sql.SQL(',').join(map(sql.Identifier, row)),
                        sql.SQL(',').join(sql.Placeholder() for _ in row))
                    try:
                        with connection.transaction(): connection.execute(query, list(row.values()))
                        counts[table] += 1
                    except psycopg.errors.ForeignKeyViolation:
                        remaining.append((table, row))
                    except (psycopg.IntegrityError, psycopg.DataError) as error:
                        rejections.append({'table': table, 'sqlstate': error.sqlstate, 'error': str(error)[:300]})
                if len(remaining) == len(pending):
                    rejections.extend({'table': table, 'error': 'unresolved foreign key'} for table, _ in remaining)
                    break
                pending = remaining
    return {t: counts[t] for t in tables}, rejections


def validate_catalog(admin_dsn, namespace, tables):
    expected_types = {'INT': 'int4', 'BIGINT': 'int8', 'SMALLINT': 'int2', 'TINYINT': 'int2',
        'DECIMAL': 'numeric', 'DOUBLE': 'float8', 'FLOAT': 'float8', 'REAL': 'float4',
        'TEXT': 'text', 'VARCHAR': 'varchar', 'CHAR': 'bpchar', 'BOOLEAN': 'bool',
        'DATE': 'date', 'TIME': 'time', 'TIMETZ': 'timetz', 'TIMESTAMP': 'timestamp', 'TIMESTAMPTZ': 'timestamptz'}
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        attributes = connection.execute(
            'SELECT c.relname,a.attname,t.typname,a.atttypmod,a.attnotnull FROM pg_class c '
            'JOIN pg_namespace n ON n.oid=c.relnamespace '
            'JOIN pg_attribute a ON a.attrelid=c.oid JOIN pg_type t ON t.oid=a.atttypid '
            "WHERE n.nspname=%s AND c.relkind IN ('r','p') AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attnum",
            (namespace,)).fetchall()
        expected = {(table, name): expected_types[column.args['kind'].this.value]
                    for table, spec in tables.items() for name, column in spec['columns'].items()}
        actual = {(table, name): kind for table, name, kind, _, _ in attributes}
        if expected != actual: raise ValueError('Allocated catalog types differ from source DDL')
        primary = connection.execute(
            "SELECT c.relname,count(*) FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND k.contype='p' GROUP BY c.relname",
            (namespace,)).fetchall()
        if {t for t, _ in primary} != {t for t, spec in tables.items() if spec['primary']}:
            raise ValueError('Allocated primary keys differ from source DDL')
        return {'source_types_match': True, 'primary_keys_match': True,
                'column_count': len(attributes),
                'catalog_sha256': hashlib.sha256(json.dumps(attributes).encode()).hexdigest()}


def probe_specs(specs, probes, timeout_ms=10000):
    results = {}
    for row in probes:
        cases = []
        for spec in specs:
            with Postgres(spec['dsn'], search_path=spec['search_path'], timeout_ms=timeout_ms) as db:
                comparison = db.compare(row['sql'], row['sql'])
                item = {'name': spec['name'], 'comparison': comparison}
                if comparison.get('equivalent'):
                    query = parse_read_query(row['sql'])
                    with db.connection.transaction():
                        with db.connection.cursor() as cursor:
                            db._configure(cursor)
                            db._audit_functions(cursor, query)
                            cursor.execute(row['sql'])
                            item['sample_rows'] = json_value(cursor.fetchmany(5))
                cases.append(item)
        active = [c for c in cases if c['comparison'].get('original', {}).get('row_count', 0) and
                  c['comparison'].get('original', {}).get('nonnull_cells', 0)]
        signatures = {c['comparison']['original']['sha256'] for c in active}
        results[row['id']] = {'query_sha256': hashlib.sha256(row['sql'].encode()).hexdigest(),
            'active_cases': [c['name'] for c in active], 'nonempty_value_fingerprints': len(signatures),
            'nonvacuous': len(active) >= 2 and len(signatures) >= 2 and
                          all(c['comparison'].get('equivalent') for c in cases), 'cases': cases}
    return results


def build_bundle(manifest, admin_dsn, read_dsn, previous=None, benchmarks=None, checkpoint=None):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    bundle = {'format_version': 1, 'generator': 'join-aware-v1', 'build_id': uuid.uuid4().hex,
              'build_status': 'building',
              'generator_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'evidence': 'Finite constraint-valid examples and read-only output probes; never an equivalence proof',
              'mini_dbs': {}, 'schema_hashes': {}, 'inference': {}, 'coverage_probes': {},
              'previous_fixture_coverage': {}, 'requested_probes': {}, 'allocated_namespaces': []}
    def save():
        if checkpoint: checkpoint(bundle)
    class TrackedNamespaces(list):
        def append(self, namespace):
            super().append(namespace)
            bundle['allocated_namespaces'] = list(self)
            save()
    allocated = TrackedNamespaces()
    save()
    try:
        for benchmark in benchmarks or sorted({r['benchmark'] for r in rows}):
            selected = [r for r in rows if r['benchmark'] == benchmark]
            schema_path = ROOT / 'vendor/LLM4Rewrite' / benchmark / 'create_tables.sql'
            source = schema_path.read_text()
            tables = parse_schema(source)
            evidence = Evidence(tables)
            for row in selected: evidence.analyze(row)
            ids = [p for p in PROBES.get(benchmark, ()) if p in evidence.profiles]
            bundle['requested_probes'][benchmark] = {'requested': list(PROBES.get(benchmark, ())),
                'resolved': ids, 'missing': [p for p in PROBES.get(benchmark, ()) if p not in ids]}
            probes = [r for r in selected if r['id'] in ids]
            if previous and benchmark in previous.get('mini_dbs', {}):
                bundle['previous_fixture_coverage'][benchmark] = probe_specs(previous['mini_dbs'][benchmark], probes)
            with Postgres(admin_dsn) as admin:
                specs = _synthesize_cases(admin, source, 'SELECT 1 AS fixture_probe', 'SELECT 1 AS fixture_probe',
                    retain=True, read_dsn=read_dsn, query_sqls=[r['sql'] for r in selected],
                    _created_schemas=allocated)
            for spec in specs: spec.pop('comparison', None)
            old_names = {s['search_path'] for values in (previous or {}).get('mini_dbs', {}).values() for s in values}
            if old_names.intersection(allocated): raise ValueError('New schemas overlap existing fixtures')
            schema_hash = hashlib.sha256(source.encode()).hexdigest()
            for spec in specs:
                catalog = validate_catalog(admin_dsn, spec['search_path'], tables)
                generated, features = make_rows(tables, evidence, spec['name'], ids)
                counts, rejections = replace_fresh_rows(admin_dsn, spec['search_path'], tables, generated)
                spec.update(rows_by_table=counts, rows_inserted=sum(counts.values()),
                            constraint_rejections=len(rejections), rejection_details=rejections,
                            coverage_complete=not rejections and (spec['name'] == 'empty' or all(counts.values())),
                            schema_sha256=schema_hash, generator='join-aware-v1',
                            fixture_data_sha256=hashlib.sha256(json.dumps(json_value(generated), sort_keys=True).encode()).hexdigest(),
                            case_features=features, profiles=['global_literals', *ids], catalog_validation=catalog)
            bundle['mini_dbs'][benchmark] = specs
            bundle['schema_hashes'][benchmark] = schema_hash
            bundle['inference'][benchmark] = evidence.serializable()
            bundle['coverage_probes'][benchmark] = probe_specs(specs, probes)
            save()
            print(json.dumps({'benchmark': benchmark, 'cases': [{k: s[k] for k in ('name', 'rows_inserted', 'coverage_complete')} for s in specs],
                              'probes': {p: {'nonvacuous': v['nonvacuous'], 'active_cases': v['active_cases']}
                                         for p, v in bundle['coverage_probes'][benchmark].items()}}), flush=True)
        probe_results = [p for probes in bundle['coverage_probes'].values() for p in probes.values()]
        bundle['representative_coverage_complete'] = bool(probe_results) and all(p['nonvacuous'] for p in probe_results) and not any(
            p['missing'] for p in bundle['requested_probes'].values()) and all(
            spec['coverage_complete'] for specs in bundle['mini_dbs'].values() for spec in specs)
        bundle['build_status'] = 'complete'
        save()
        return bundle
    except Exception as error:
        bundle.update(build_status='failed', build_error=f'{type(error).__name__}: {error}')
        try:
            with psycopg.connect(admin_dsn, autocommit=True) as connection:
                for namespace in allocated:
                    connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(namespace)))
            bundle['cleanup_status'] = 'owned schemas dropped'
        except Exception as cleanup_error:
            bundle['cleanup_status'] = 'pending database recovery; only allocated_namespaces are owned'
            bundle['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/manifest.jsonl')
    parser.add_argument('--previous', type=Path, default=ROOT / 'configs/miniatures.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/miniatures-join-aware.json')
    parser.add_argument('--admin-dsn', default=f'host={ROOT}/.runtime/socket port=55432 dbname=mcts_fixtures user=steve')
    parser.add_argument('--read-dsn', default=f'host={ROOT}/.runtime/socket port=55432 dbname=mcts_fixtures user=mcts_read')
    parser.add_argument('--benchmarks', nargs='+', choices=['calcite', 'dsb', 'tpch'])
    args = parser.parse_args()
    if args.output.resolve() in {args.previous.resolve(), (ROOT / 'configs/miniatures.json').resolve()}:
        raise ValueError('Output must be a new bundle; active fixture files are protected')
    if args.output.exists(): raise FileExistsError('Choose a new output path to preserve prior fixture bundles')
    journal = args.output.with_suffix('.journal.json')
    if journal.exists(): raise FileExistsError(f'Existing build journal must be reviewed first: {journal}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(bundle):
        temporary = journal.with_suffix('.tmp')
        temporary.write_text(json.dumps(bundle, indent=2) + '\n')
        temporary.replace(journal)
    previous = json.loads(args.previous.read_text()) if args.previous.exists() else None
    bundle = build_bundle(args.manifest, args.admin_dsn, args.read_dsn, previous, args.benchmarks, checkpoint)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(bundle, indent=2) + '\n')
    temporary.replace(args.output)
    return 0 if bundle['representative_coverage_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
