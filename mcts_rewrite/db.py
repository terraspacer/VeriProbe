"""Read-only PostgreSQL execution and bounded or streaming result comparison.

No connection is shared across processes. Comparisons run both SELECTs in one
REPEATABLE READ, READ ONLY transaction; a failure is never an equality result.
"""
from __future__ import annotations

import base64
from collections import Counter, deque
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from statistics import median
from time import perf_counter
import uuid
from typing import Any

import psycopg
import sqlglot
from sqlglot import exp, ErrorLevel
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope


# Reject unknown functions as well as known volatile functions. Merely using a
# read-only transaction does not prevent advisory locks or external side effects.
_SAFE_FUNCTIONS = frozenset("""
ABS AVG COUNT SUM MIN MAX COALESCE NULLIF CAST TRY_CAST EXTRACT SUBSTRING
LOWER UPPER LENGTH CHAR_LENGTH CHARACTER_LENGTH TRIM LTRIM RTRIM REPLACE
CONCAT CONCAT_WS ROUND FLOOR CEIL CEILING SIGN POWER SQRT MOD GREATEST LEAST
POSITION STR_POSITION STRPOS SPLIT_PART LEFT RIGHT REPEAT REVERSE TRANSLATE
DATE_TRUNC TIMESTAMP_TRUNC DATE_PART TO_CHAR TO_DATE TO_TIMESTAMP
BOOL_AND BOOL_OR EVERY ARRAY_AGG STRING_AGG JSON_AGG JSONB_AGG
STDDEV STDDEV_POP STDDEV_SAMP VARIANCE VAR_POP VAR_SAMP GROUPING
BIT_AND BIT_OR BIT_XOR BITWISE_AND_AGG BITWISE_OR_AGG BITWISE_XOR_AGG
ROW_NUMBER RANK DENSE_RANK PERCENT_RANK CUME_DIST NTILE LAG LEAD FIRST_VALUE
LAST_VALUE NTH_VALUE ARRAY_SIZE ARRAY_LENGTH CARDINALITY
EXISTS AND OR NOT IF CASE STR_TO_DATE TIME_TO_STR
""".split())


def parse_read_query(sql: str) -> exp.Expression:
    """Parse exactly one read query and reject side-effectful/unknown constructs."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a nonempty string")
    expressions = sqlglot.parse(sql, read="postgres", error_level=ErrorLevel.RAISE)
    expressions = [expression for expression in expressions
                   if expression is not None and not isinstance(expression, exp.Semicolon)]
    if len(expressions) != 1:
        raise ValueError("exactly one SQL statement is required")
    query = expressions[0]
    if not isinstance(query, (exp.Select, exp.SetOperation, exp.Subquery, exp.Values)):
        raise ValueError("only read queries are allowed")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                 exp.Command, exp.Into, exp.Lock, exp.Merge, exp.Copy,
                 exp.Transaction, exp.Commit, exp.Rollback)
    for node in query.walk():
        if isinstance(node, forbidden):
            raise ValueError(f"unsafe query construct: {type(node).__name__}")
        if isinstance(node, exp.DataType) and node.this.value == 'USER-DEFINED':
            raise ValueError('user-defined casts/types are unsupported')
        if isinstance(node, exp.Func):
            name = node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name().upper()
            if name not in _SAFE_FUNCTIONS:
                raise ValueError(f"function is not in the read-only deterministic allowlist: {name}")
            # A schema-qualified builtin can resolve to arbitrary user code.
            if isinstance(node.parent, exp.Dot):
                raise ValueError("schema-qualified functions are unsupported")
    return query


def has_order(query: exp.Expression) -> bool:
    while isinstance(query, exp.Subquery):
        query = query.this
    return query.args.get("order") is not None


def _formal_base_tables(queries) -> dict[str, exp.Table]:
    """Resolve base sources separately from CTE names and derived tables."""
    tables = {}
    for query in queries:
        for scope in traverse_scope(query):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    if source.db or source.catalog or source.this.args.get('quoted'):
                        raise ValueError('formal context requires simple unquoted base tables')
                    tables[source.name.lower()] = source
    return tables


def _formal_column_references(queries, columns_by_table: dict[str, set[str]]) -> dict[str, set[str]]:
    """Find all base-column dependencies, rejecting missing/ambiguous names.

    Qualification is used only to inspect dependencies, never to rewrite the
    SQL submitted to PostgreSQL or the prover. A star keeps the whole relation
    schema, so removing an unsupported output type cannot manufacture a proof.
    """
    tables = _formal_base_tables(queries)
    if any(name not in columns_by_table for name in tables):
        raise ValueError('query refers to a table absent from the supplied schema')
    references = {name: set() for name in tables}
    schema = {name: {column: 'UNKNOWN' for column in columns}
              for name, columns in columns_by_table.items()}
    for query in queries:
        qualified = qualify(query.copy(), dialect='postgres', schema=schema,
                            infer_schema=False, expand_stars=True,
                            quote_identifiers=False, identify=False)
        seen = set()
        for scope in traverse_scope(qualified):
            for column in scope.columns:
                if id(column) in seen:
                    continue
                seen.add(id(column))
                owner = scope
                source = None
                while owner is not None:
                    item = owner.selected_sources.get(column.table)
                    if item is not None:
                        source = item[1]
                        break
                    owner = owner.parent
                if isinstance(source, exp.Table):
                    name = source.name.lower()
                    if column.name not in columns_by_table[name]:
                        raise ValueError('query refers to a column absent from the supplied schema')
                    references[name].add(column.name)
                elif not isinstance(source, Scope):
                    raise ValueError('query column source cannot be resolved conservatively')
        if query.find(exp.Star):
            for name in _formal_base_tables((query,)):
                references[name].update(columns_by_table[name])
    return references


def _encode_value(value: Any) -> Any:
    """Type-tag values; never compare a Decimal through a lossy float conversion."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, float):
        return ["float", value.hex() if math.isfinite(value) else str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", base64.b64encode(bytes(value)).decode("ascii")]
    if isinstance(value, (datetime, date, time)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, timedelta):
        return ["timedelta", value.days, value.seconds, value.microseconds]
    if isinstance(value, uuid.UUID):
        return ["uuid", str(value)]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_encode_value(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[_encode_value(key), _encode_value(item)] for key, item in value.items()]]
    raise ValueError(f"result contains unsupported Python type: {type(value).__name__}")


def _positional_value(value):
    """Canonicalize exact SQL numbers without rounding through float/context."""
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        number = Decimal(value)
        if not number.is_finite():
            return ['numeric', str(number)]
        sign, digits, exponent = number.as_tuple()
        digits = list(digits)
        while digits and digits[-1] == 0:
            digits.pop()
            exponent += 1
        return ['numeric', sign if digits else 0, ''.join(map(str, digits)), exponent if digits else 0]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_positional_value(item) for item in value]]
    if isinstance(value, dict):
        return ['dict', [[_positional_value(key), _positional_value(item)] for key, item in value.items()]]
    return _encode_value(value)


def _encode_row(row: tuple, profile: str = 'strict_metadata') -> bytes:
    encoder = _positional_value if profile == 'positional_numeric' else _encode_value
    return json.dumps([encoder(value) for value in row], ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


_NUMERIC_OIDS = frozenset({20, 21, 23, 700, 701, 1700})
_FLOAT_FALLBACK_ROWS = 100_000
_FLOAT_FALLBACK_BYTES = 64 * 1024 * 1024
_FLOAT_MATCH_EDGES = 1_000_000


def _has_float(value):
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_float(item) for item in value)
    if isinstance(value, dict):
        return any(_has_float(item) for pair in value.items() for item in pair)
    return False


def _numeric(value):
    return isinstance(value, (int, Decimal, float)) and not isinstance(value, bool)


def _positional_equal(left, right, rel_tol, abs_tol):
    if _numeric(left) and _numeric(right):
        if isinstance(left, float) or isinstance(right, float):
            # Fractions preserve large NUMERIC values and avoid float overflow.
            l, r = Decimal(left) if isinstance(left, int) else left, Decimal(right) if isinstance(right, int) else right
            if not math.isfinite(l) or not math.isfinite(r):
                return ((math.isnan(l) and math.isnan(r)) or l == r)
            l, r = Fraction(l), Fraction(r)
            return abs(l - r) <= max(abs_tol, rel_tol * max(abs(l), abs(r)))
        return left == right
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(_positional_equal(l, r, rel_tol, abs_tol) for l, r in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return len(left) == len(right) and all(
            _positional_equal(lk, rk, rel_tol, abs_tol) and _positional_equal(lv, rv, rel_tol, abs_tol)
            for (lk, lv), (rk, rv) in zip(left.items(), right.items()))
    return _encode_value(left) == _encode_value(right)


class _PositionalRows:
    """Retain a bounded fallback for tolerance; exact streams remain constant space."""
    def __init__(self, values, ordered):
        self.values, self.ordered = values, ordered
        self.rows = [] if ordered else {}
        self.has_float = False

    def add(self, row, encoded, count, size):
        self.has_float |= _has_float(row)
        if count > _FLOAT_FALLBACK_ROWS or size > _FLOAT_FALLBACK_BYTES:
            self.rows = None
        if self.rows is None:
            if self.has_float:
                raise ValueError('floating tolerance comparison fallback limit exceeded (100000 rows / 64 MiB)')
            return
        if self.ordered:
            self.rows.append(row)
        elif encoded in self.rows:
            self.rows[encoded][1] += 1
        else:
            self.rows[encoded] = [row, 1]


def _tolerant_rows_equal(left, right, rel_tol, abs_tol, deadline=None):
    """Find a full bag matching; closeness is nontransitive, so greedy is wrong."""
    if left.rows is None or right.rows is None:
        raise ValueError('floating tolerance comparison requires bounded materialized results')
    rel_tol, abs_tol = Fraction(str(rel_tol)), Fraction(str(abs_tol))
    def check_deadline():
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError('floating tolerance matching exceeded comparison deadline')
    if left.ordered:
        if len(left.rows) != len(right.rows):
            return False
        for l, r in zip(left.rows, right.rows):
            check_deadline()
            if not _positional_equal(l, r, rel_tol, abs_tol):
                return False
        return True
    lrows, rrows = list(left.rows.values()), list(right.rows.values())
    if sum(count for _, count in lrows) != sum(count for _, count in rrows):
        return False
    if len(lrows) * len(rrows) > _FLOAT_MATCH_EDGES:
        raise ValueError('floating tolerance matching candidate limit exceeded (1000000 row pairs)')
    # Capacitated bipartite flow preserves duplicates without expanding them.
    source, sink = len(lrows) + len(rrows), len(lrows) + len(rrows) + 1
    graph = [[] for _ in range(sink + 1)]
    def edge(a, b, capacity):
        graph[a].append([b, capacity, len(graph[b])])
        graph[b].append([a, 0, len(graph[a]) - 1])
    for i, (_, count) in enumerate(lrows):
        edge(source, i, count)
    for j, (_, count) in enumerate(rrows):
        edge(len(lrows) + j, sink, count)
    for i, (l, count) in enumerate(lrows):
        check_deadline()
        for j, (r, rcount) in enumerate(rrows):
            if _positional_equal(l, r, rel_tol, abs_tol):
                edge(i, len(lrows) + j, min(count, rcount))
    matched, needed = 0, sum(count for _, count in lrows)
    while matched < needed:
        check_deadline()
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for index, (target, capacity, _) in enumerate(graph[node]):
                if capacity and target not in parents:
                    parents[target] = node, index
                    queue.append(target)
        if sink not in parents:
            return False
        amount, node = needed - matched, sink
        while node != source:
            prior, index = parents[node]
            amount = min(amount, graph[prior][index][1])
            node = prior
        node = sink
        while node != source:
            prior, index = parents[node]
            target, _, reverse = graph[prior][index]
            graph[prior][index][1] -= amount
            graph[target][reverse][1] += amount
            node = prior
        matched += amount
    return True


def _description(description) -> list[dict]:
    return [{"name": column.name, "type_oid": column.type_code,
             "internal_size": column.internal_size, "precision": column.precision,
             "scale": column.scale} for column in description]


class _StreamingRows:
    """Constant-space, probabilistic row checksums; these are not exact proofs."""
    _modulus = (1 << 521) - 1

    def __init__(self, ordered: bool):
        self.ordered = ordered
        self.count = self.total = self.squares = 0
        self.digest = hashlib.sha256()

    def add(self, encoded: bytes):
        self.count += 1
        if self.ordered:
            self.digest.update(len(encoded).to_bytes(8, 'big'))
            self.digest.update(encoded)
        else:
            value = int.from_bytes(hashlib.sha256(encoded).digest(), 'big')
            self.total = (self.total + value) % self._modulus
            self.squares = (self.squares + value * value) % self._modulus

    def signature(self):
        if self.ordered:
            return ('ordered_sha256_v1', self.count, self.digest.hexdigest())
        return ('sha256_multiset_moments_mod_2^521-1_v1', self.count,
                format(self.total, 'x'), format(self.squares, 'x'))

    def hexdigest(self):
        return hashlib.sha256(json.dumps(self.signature(), separators=(',', ':')).encode('ascii')).hexdigest()


class Postgres:
    def __init__(self, dsn: str, timeout_ms: int = 30000, search_path: str = "public",
                 max_rows: int | None = 1_000_000, max_result_bytes: int | None = 256 * 1024 * 1024,
                 comparison_mode: str = 'exact', comparison_profile: str = 'strict_metadata',
                 float_rel_tol: float = 1e-5, float_abs_tol: float = 1e-5):
        if (timeout_ms <= 0 or (max_rows is not None and max_rows <= 0) or
                (max_result_bytes is not None and max_result_bytes <= 0)):
            raise ValueError("execution and result limits must be positive")
        if comparison_mode not in {'exact', 'stream_hash'}:
            raise ValueError('comparison_mode must be exact or stream_hash')
        if comparison_profile not in {'strict_metadata', 'positional_numeric'}:
            raise ValueError('comparison_profile must be strict_metadata or positional_numeric')
        if any(not math.isfinite(value) or value < 0 for value in (float_rel_tol, float_abs_tol)):
            raise ValueError('floating tolerances must be finite and nonnegative')
        self.dsn = dsn
        self.timeout_ms = int(timeout_ms)
        self.search_path = search_path
        self.max_rows = int(max_rows) if max_rows is not None else None
        self.max_result_bytes = int(max_result_bytes) if max_result_bytes is not None else None
        self.comparison_mode = comparison_mode
        self.comparison_profile = comparison_profile
        self.float_rel_tol, self.float_abs_tol = float(float_rel_tol), float(float_abs_tol)
        self._conn = None
        self._comparison_deadline = None

    @property
    def connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def _configure(self, cursor):
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cursor.execute("SELECT set_config('statement_timeout', %s, true), "
                       "set_config('search_path', %s, true), "
                       "set_config('idle_in_transaction_session_timeout', %s, true)",
                       (str(self.timeout_ms), self.search_path, str(max(self.timeout_ms * 2, 60000))))

    def _audit_functions(self, cursor, query):
        names = set()
        aliases = {'TIMESTAMP_TRUNC': 'date_trunc', 'BITWISE_AND_AGG': 'bit_and',
                   'BITWISE_OR_AGG': 'bit_or', 'BITWISE_XOR_AGG': 'bit_xor',
                   'STR_POSITION': 'strpos', 'STR_TO_DATE': 'to_date',
                   'TIME_TO_STR': 'to_char', 'ARRAY_SIZE': 'array_length'}
        for function in query.find_all(exp.Func):
            name = function.name.upper() if isinstance(function, exp.Anonymous) else function.sql_name().upper()
            names.add(aliases.get(name, name.lower()))
        if names:
            cursor.execute("SELECT n.nspname, p.proname FROM pg_proc p "
                           "JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "WHERE p.proname=ANY(%s) AND n.nspname=ANY(current_schemas(true)) "
                           "AND n.nspname <> 'pg_catalog' LIMIT 1", (list(names),))
            found = cursor.fetchone()
            if found:
                raise ValueError(f'user-defined function overload is unsupported: {found[0]}.{found[1]}')

    def explain(self, sql: str, analyze: bool = False, timing: bool = True) -> dict:
        query = parse_read_query(sql)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                self._configure(cursor)
                self._audit_functions(cursor, query)
                options = f"ANALYZE, BUFFERS, TIMING {'ON' if timing else 'OFF'}, FORMAT JSON" if analyze else "FORMAT JSON"
                cursor.execute(f"EXPLAIN ({options}) {sql}")
                document = cursor.fetchone()[0]
                if isinstance(document, str):
                    document = json.loads(document)
                if not isinstance(document, list) or len(document) != 1 or "Plan" not in document[0]:
                    raise ValueError("unexpected PostgreSQL JSON EXPLAIN response")
                return document[0]

    def measure(self, sql: str, runs: int = 3, warmups: int = 1) -> dict:
        if runs < 1 or warmups < 0:
            raise ValueError("runs must be positive and warmups nonnegative")
        result = {"status": "error", "median_ms": None, "samples_ms": [],
                  "wall_samples_ms": [], "estimated_cost": None, "plan": None,
                  "measurement_method": "EXPLAIN (ANALYZE, BUFFERS, TIMING OFF); server Execution Time"}
        try:
            for iteration in range(warmups + runs):
                start = perf_counter()
                document = self.explain(sql, analyze=True, timing=False)
                wall_ms = (perf_counter() - start) * 1000
                timing = document.get("Execution Time")
                if not isinstance(timing, (int, float)) or not math.isfinite(timing) or timing < 0:
                    raise ValueError("missing or invalid server Execution Time")
                result["plan"] = document
                result["estimated_cost"] = document["Plan"].get("Total Cost")
                if iteration >= warmups:
                    result["samples_ms"].append(float(timing))
                    result["wall_samples_ms"].append(wall_ms)
            result.update(status="ok", median_ms=median(result["samples_ms"]))
        except Exception as error:
            result.update(status="timeout" if isinstance(error, (psycopg.errors.QueryCanceled, TimeoutError)) else "error",
                          error=f"{type(error).__name__}: {error}", error_type=type(error).__name__,
                          sqlstate=getattr(error, "sqlstate", None))
        return result

    def planner_hints_enabled(self) -> dict:
        """Require both an installed extension and its registered, enabled GUC."""
        try:
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    self._configure(cursor)
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='pg_hint_plan'), "
                                   "current_setting('pg_hint_plan.enable_hint', true)")
                    installed, setting = cursor.fetchone()
                    return {"enabled": installed and setting == "on", "installed": installed,
                            "pg_hint_plan.enable_hint": setting}
        except Exception as error:
            return {"enabled": False, "error": str(error)}

    def formal_context_supported(self, original: str, candidate: str, schema_sql: str | None = None) -> dict:
        """Audit the live relation semantics assumed by the narrow solver model."""
        try:
            queries = [parse_read_query(source) for source in (original, candidate)]
            relations = _formal_base_tables(queries)
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    self._configure(cursor)
                    catalog = {}
                    for name, table in relations.items():
                        cursor.execute("SELECT c.oid,c.relkind,c.relrowsecurity FROM pg_class c "
                                       "WHERE c.oid=to_regclass(%s)", (table.this.sql(dialect='postgres'),))
                        relation = cursor.fetchone()
                        if not relation or relation[1] not in {'r', 'p'} or relation[2]:
                            return {'supported': False, 'reason': 'nonbase, missing, or row-security relation', 'relation': name}
                        cursor.execute("SELECT a.attname,a.atttypid,co.collisdeterministic "
                                       "FROM pg_attribute a LEFT JOIN pg_collation co ON co.oid=a.attcollation "
                                       "WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped", (relation[0],))
                        catalog[name] = {column[0]: column[1:] for column in cursor.fetchall()}
                    references = _formal_column_references(queries, {name: set(columns) for name, columns in catalog.items()})
                    declared = {}
                    if schema_sql is not None:
                        for statement in sqlglot.parse(schema_sql, read='postgres', error_level=ErrorLevel.RAISE):
                            if statement is not None:
                                declared[statement.this.this.name.lower()] = {
                                    column.name.lower(): column.args['kind'].this.value
                                    for column in statement.this.expressions if isinstance(column, exp.ColumnDef)}
                    type_names = {16: 'BOOLEAN', 20: 'BIGINT', 21: 'SMALLINT', 23: 'INT', 25: 'TEXT', 1043: 'VARCHAR'}
                    evidence = []
                    for name, referenced in references.items():
                        # bpchar/CHAR has blank-padding equality; numerical
                        # arithmetic, domains, floats and custom operators are
                        # outside the relational integer/text proof model.
                        if any(catalog[name][column][0] not in type_names or catalog[name][column][1] is False
                               for column in referenced):
                            return {'supported': False, 'reason': 'unsupported live type or nondeterministic collation', 'relation': name}
                        if schema_sql is not None and any(
                                declared.get(name, {}).get(column) != type_names[catalog[name][column][0]]
                                for column in referenced):
                            return {'supported': False, 'reason': 'supplied schema and live referenced column types differ', 'relation': name}
                        evidence.append({'relation': name, 'columns': sorted(referenced),
                                         'type_oids': [catalog[name][column][0] for column in sorted(referenced)]})
            return {'supported': True, 'relations': evidence}
        except Exception as error:
            return {'supported': False, 'reason': str(error)}

    def _collect(self, sql: str, ordered: bool):
        streaming = self.comparison_mode == 'stream_hash'
        values = _StreamingRows(ordered) if streaming else ([] if ordered else Counter())
        positional = _PositionalRows(values, ordered) if self.comparison_profile == 'positional_numeric' else None
        count = size = nonnull_cells = 0
        deadline = perf_counter() + self.timeout_ms / 1000
        if self._comparison_deadline is not None:
            deadline = min(deadline, self._comparison_deadline)
        remaining_ms = int((deadline - perf_counter()) * 1000)
        if remaining_ms <= 0:
            raise TimeoutError('comparison exceeded its total execution deadline')
        with self.connection.cursor() as control:
            control.execute("SELECT set_config('statement_timeout', %s, true)", (str(remaining_ms),))
        # Server-side cursor bounds transfer and memory before fetching all rows.
        with self.connection.cursor(name="mcts_" + uuid.uuid4().hex) as cursor:
            cursor.execute(sql)
            columns = _description(cursor.description)
            while True:
                remaining_ms = int((deadline - perf_counter()) * 1000)
                if remaining_ms <= 0:
                    raise TimeoutError('comparison query exceeded its total execution deadline')
                with self.connection.cursor() as control:
                    control.execute("SELECT set_config('statement_timeout', %s, true)", (str(remaining_ms),))
                rows = cursor.fetchmany(1024)
                if not rows:
                    break
                for row in rows:
                    encoded = _encode_row(row, self.comparison_profile)
                    count += 1
                    nonnull_cells += sum(value is not None for value in row)
                    size += len(encoded)
                    if ((self.max_rows is not None and count > self.max_rows) or
                            (self.max_result_bytes is not None and size > self.max_result_bytes)):
                        raise ValueError("comparison result limit exceeded")
                    if positional is not None:
                        positional.add(row, encoded, count, size)
                    if streaming:
                        values.add(encoded)
                    elif ordered:
                        values.append(encoded)
                    else:
                        values[encoded] += 1
        if perf_counter() >= deadline:
            raise TimeoutError('comparison exceeded its total execution deadline')
        if streaming:
            if positional is not None:
                positional.values = values.signature()
            return positional if positional is not None else values.signature(), {"row_count": count, "nonnull_cells": nonnull_cells,
                        "distinct_rows": None, "bytes": size, "sha256": values.hexdigest(),
                        "columns": columns, "comparison_kind": 'probabilistic_checksum',
                        "checksum_algorithm": values.signature()[0]}
        digest = hashlib.sha256()
        entries = ((value, 1) for value in values) if ordered else sorted(values.items())
        for value, multiplicity in entries:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
            digest.update(multiplicity.to_bytes(8, "big"))
        if perf_counter() >= deadline:
            raise TimeoutError('comparison exceeded its total execution deadline')
        return positional if positional is not None else values, {"row_count": count, "nonnull_cells": nonnull_cells,
                        "distinct_rows": len(set(values)) if ordered else len(values),
                        "bytes": size, "sha256": digest.hexdigest(), "columns": columns,
                        "comparison_kind": 'exact'}

    def compare(self, original: str, candidate: str) -> dict:
        result = {"equivalent": False, "status": "error", "ordered": False,
                  "original": None, "candidate": None, "same_snapshot": False,
                  "comparison_mode": self.comparison_mode,
                  "comparison_profile": self.comparison_profile,
                  "float_rel_tol": self.float_rel_tol, "float_abs_tol": self.float_abs_tol,
                  "numeric_relaxation_used": False,
                  "comparison_kind": 'probabilistic_checksum' if self.comparison_mode == 'stream_hash' else 'exact',
                  "probabilistic_checksum": self.comparison_mode == 'stream_hash'}
        self._comparison_deadline = perf_counter() + self.timeout_ms / 1000
        try:
            original_ast = parse_read_query(original)
            candidate_ast = parse_read_query(candidate)
            ordered = has_order(original_ast) or has_order(candidate_ast)
            result["ordered"] = ordered
            if has_order(original_ast) and not has_order(candidate_ast):
                raise ValueError("candidate removes the root query's ORDER BY guarantee")
            with self.connection.transaction():
                with self.connection.cursor() as cursor:
                    self._configure(cursor)
                    self._audit_functions(cursor, original_ast)
                    self._audit_functions(cursor, candidate_ast)
                left, left_info = self._collect(original, ordered)
                right, right_info = self._collect(candidate, ordered)
                if self.comparison_profile == 'positional_numeric':
                    lcols, rcols = left_info['columns'], right_info['columns']
                    compatible = len(lcols) == len(rcols) and all(
                        l['type_oid'] == r['type_oid'] or {l['type_oid'], r['type_oid']} <= _NUMERIC_OIDS
                        for l, r in zip(lcols, rcols))
                    exact = left.values == right.values
                    equivalent = compatible and exact
                    if compatible and not exact and (left.has_float or right.has_float):
                        equivalent = _tolerant_rows_equal(left, right, self.float_rel_tol, self.float_abs_tol,
                                                          self._comparison_deadline)
                        result.update(comparison_kind='tolerance_matching', probabilistic_checksum=False)
                    result['numeric_relaxation_used'] = bool(equivalent and (not exact or any(
                        l['type_oid'] != r['type_oid'] for l, r in zip(lcols, rcols))))
                else:
                    equivalent = left_info['columns'] == right_info['columns'] and left == right
                result.update(original=left_info, candidate=right_info, same_snapshot=True,
                              status="ok", equivalent=equivalent)
                if not result["equivalent"]:
                    result["error"] = "result values/checksums, duplicate multiplicities, ordering, or column metadata differ"
        except Exception as error:
            result.update(status="timeout" if isinstance(error, (psycopg.errors.QueryCanceled, TimeoutError)) else "error",
                          error=f"{type(error).__name__}: {error}", error_type=type(error).__name__,
                          sqlstate=getattr(error, "sqlstate", None))
        finally:
            self._comparison_deadline = None
        return result

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
