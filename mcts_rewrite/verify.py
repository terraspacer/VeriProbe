"""Fail-closed equivalence evidence against the original, unrevised root query.

Finite execution testing is not a proof. Strict mode requires conservative
identity or a supported SQLSolver proof AND a real-database comparison. Streaming
comparisons explicitly retain their probabilistic-checksum evidence label.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

import psycopg
from psycopg import sql as psql
import sqlglot
from sqlglot import exp, ErrorLevel

from .db import Postgres, parse_read_query, _formal_column_references


def _comparison_options(db):
    """Every fixture uses the gate's result contract, including numeric tolerance."""
    return {name: getattr(db, name) for name in (
        'comparison_mode', 'comparison_profile', 'float_rel_tol', 'float_abs_tol',
        'max_rows', 'max_result_bytes') if hasattr(db, name)}

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _pg_identifier_name(node) -> str:
    identifier = node if isinstance(node, exp.Identifier) else node.this
    if not isinstance(identifier, exp.Identifier):
        raise ValueError('expected a simple SQL identifier')
    name = identifier.this
    return name if identifier.args.get('quoted') else name.translate(_ASCII_LOWER)


def conservative_identity(sql: str) -> tuple:
    """Formatting key preserving literal spelling, quoted names, and hints.

    This intentionally does not transpile or optimize SQL. SQLGlot's AST alone
    is insufficient: some syntax distinctions disappear during parsing.
    """
    parse_read_query(sql)
    tokens = sqlglot.Dialect.get_or_raise("postgres").tokenize(sql)
    key, hints = [], []
    for token in tokens:
        raw = sql[token.start:token.end + 1]
        if token.token_type.name != "SEMICOLON":
            # Strings/quoted identifiers retain the complete original spelling.
            # Bare words and keywords are case-insensitive in PostgreSQL.
            protected = raw.startswith(("'", '"', "$")) or token.token_type.name in {
                "STRING", "IDENTIFIER", "BIT_STRING", "HEX_STRING", "BYTE_STRING",
                "RAW_STRING", "HEREDOC_STRING", "UNICODE_STRING"}
            key.append((token.token_type.name, raw if protected else raw.translate(_ASCII_LOWER)))
        for comment in token.comments or []:
            if comment.strip().startswith(("+", "!")):
                hints.append(comment.strip())
    return tuple(key), tuple(hints)


_BOOLEAN_PLANNER_GUCS = frozenset("""
enable_hashjoin enable_mergejoin enable_nestloop enable_seqscan enable_indexscan
enable_indexonlyscan enable_bitmapscan enable_tidscan enable_sort enable_incremental_sort
enable_material enable_memoize enable_hashagg enable_gathermerge enable_parallel_append
enable_parallel_hash enable_partitionwise_join enable_partitionwise_aggregate geqo
""".split())
_INTEGER_PLANNER_GUCS = frozenset("join_collapse_limit from_collapse_limit geqo_threshold geqo_effort geqo_pool_size geqo_generations".split())
_COST_PLANNER_GUCS = frozenset("seq_page_cost random_page_cost cpu_tuple_cost cpu_index_tuple_cost cpu_operator_cost parallel_setup_cost parallel_tuple_cost".split())
_JOIN_HINTS = frozenset("HashJoin MergeJoin NestLoop NoHashJoin NoMergeJoin NoNestLoop".split())
_SCAN_HINTS = frozenset("SeqScan NoSeqScan IndexScan NoIndexScan IndexOnlyScan NoIndexOnlyScan BitmapScan NoBitmapScan TidScan NoTidScan".split())


def _approved_planner_hint(comment: str) -> bool:
    if not comment.startswith("+"):
        return False
    body, offset, count = comment[1:], 0, 0
    while offset < len(body):
        if not body[offset:].strip():
            break
        match = re.match(r"\s*([A-Za-z]+)\s*\(", body[offset:])
        if not match:
            return False
        name = match.group(1)
        start = offset + match.end()
        end, depth = start, 1
        while end < len(body) and depth:
            depth += (body[end] == "(") - (body[end] == ")")
            end += 1
        if depth:
            return False
        arguments = body[start:end - 1]
        count += 1
        offset = end
        if name == "Set":
            parts = arguments.split()
            if len(parts) != 2:
                return False
            guc, value = parts
            if guc in _BOOLEAN_PLANNER_GUCS:
                if value.lower() not in {"on", "off", "true", "false", "0", "1"}:
                    return False
            elif guc in _INTEGER_PLANNER_GUCS:
                if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= 10000:
                    return False
            elif guc in _COST_PLANNER_GUCS:
                if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or len(value) > 12:
                    return False
            else:
                return False
        elif name in _JOIN_HINTS | _SCAN_HINTS | {"Leading"}:
            if not re.fullmatch(r"[A-Za-z0-9_$()\s]+", arguments):
                return False
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", arguments)
            residue = re.sub(r"[A-Za-z_][A-Za-z0-9_$]*|[()\s]", "", arguments)
            if residue or not names or (name in _JOIN_HINTS | {"Leading"} and len(names) < 2):
                return False
            if name != "Leading" and any(c in arguments for c in "()"):
                return False
            if name in {"SeqScan", "NoSeqScan", "TidScan", "NoTidScan"} and len(names) != 1:
                return False
        else:
            return False
    return count > 0


# SQLSolver's relational proof does not model PostgreSQL result ordering,
# floating-point arithmetic, exceptions, user functions, casts, or collations.
# Keep its accepted language deliberately small. Nonidentity queries outside
# this fragment remain available for explicitly finite differential exploration.
_SOLVER_NODES = frozenset("""
Select From Table Identifier Column Alias TableAlias Where Join EQ NEQ
And Or Not Paren Literal Null Boolean Is Exists In Subquery Star Distinct
Union Intersect Except With CTE
""".split())
_SOLVER_TYPES = frozenset("INT BIGINT SMALLINT TINYINT TEXT VARCHAR BOOLEAN".split())


def _solver_input(original_ast, candidate_ast, schema_sql: str) -> tuple[str | None, str]:
    for query in (original_ast, candidate_ast):
        for node in query.walk():
            if type(node).__name__ not in _SOLVER_NODES:
                return None, f"SQLSolver fragment excludes {type(node).__name__}"
            if isinstance(node, exp.With) and node.args.get('recursive'):
                return None, 'SQLSolver recursive queries are unsupported'
            if isinstance(node, exp.Identifier) and node.args.get("quoted"):
                return None, "SQLSolver quoted identifiers are unsupported"
            if isinstance(node, exp.Identifier) and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', node.this):
                return None, 'SQLSolver identifier is outside the ASCII fragment'
            if isinstance(node, exp.Table) and (node.db or node.catalog):
                return None, "SQLSolver qualified tables are unsupported"
            if isinstance(node, exp.Join) and (node.side or node.kind not in ("", "INNER", "CROSS") or
                                              node.args.get('method')):
                return None, "SQLSolver outer/semi/anti/natural joins are unsupported"
            if isinstance(node, exp.Literal) and not node.is_string and not re.fullmatch(r"[0-9]+", node.this):
                return None, "SQLSolver noninteger numeric literal is unsupported"
            if isinstance(node, exp.Select):
                for projection in node.expressions:
                    value = projection.this if isinstance(projection, exp.Alias) else projection
                    if not isinstance(value, (exp.Column, exp.Star)):
                        return None, "SQLSolver only supports direct column projections here"
        if any(comment.strip().startswith(("+", "!")) for node in query.walk() for comment in (node.comments or [])):
            return None, "SQLSolver optimizer hints are unsupported"
    try:
        statements = sqlglot.parse(schema_sql, read="postgres", error_level=ErrorLevel.RAISE)
    except Exception as error:
        return None, f"schema parse failed: {error}"
    declarations = {}
    for statement in statements:
        if statement is None:
            continue
        if not isinstance(statement, exp.Create) or statement.kind != "TABLE" or not isinstance(statement.this, exp.Schema):
            return None, "SQLSolver schema must consist of CREATE TABLE statements"
        table = statement.this.this
        if not isinstance(table, exp.Table) or table.db or table.catalog or any(x.args.get("quoted") for x in table.find_all(exp.Identifier)):
            return None, "SQLSolver schema requires simple unquoted table names"
        name = _pg_identifier_name(table)
        if name in declarations:
            return None, 'SQLSolver schema declares the same table more than once'
        columns = {}
        for column in statement.this.expressions:
            if not isinstance(column, exp.ColumnDef):
                continue
            column_name = _pg_identifier_name(column)
            if column_name in columns:
                return None, 'SQLSolver schema declares the same column more than once'
            columns[column_name] = column
        declarations[name] = (table, columns)
    if not declarations:
        return None, "SQLSolver requires a nonempty schema"
    try:
        references = _formal_column_references((original_ast, candidate_ast),
                        {name: set(columns) for name, (_, columns) in declarations.items()})
    except Exception as error:
        return None, f'SQLSolver column dependencies are unresolved or ambiguous: {error}'
    cleaned = []
    for name, referenced in references.items():
        table, available = declarations[name]
        # A relation used only for its multiplicity still needs a schema in
        # SQLSolver. Keeping every column is conservative for that rare case.
        columns = []
        for column_name, column in available.items():
            if referenced and column_name not in referenced:
                continue
            dtype = column.args.get('kind')
            if (dtype is None or dtype.this.value not in _SOLVER_TYPES or
                    column.this.args.get('quoted') or
                    not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', column.this.this) or
                    column.find(exp.CollateColumnConstraint)):
                return None, "SQLSolver referenced schema type/identifier/collation is outside the supported fragment"
            # Deliberately strip PK/FK/UNIQUE/NOT NULL/CHECK assumptions. The
            # supplied schema is context, not an independently audited catalog.
            columns.append(f"{column.this.sql(dialect='postgres')} {dtype.sql(dialect='postgres')}")
        if not columns:
            return None, "SQLSolver schema has no supported columns"
        cleaned.append(f"CREATE TABLE {table.sql(dialect='postgres')} ({', '.join(columns)});")
    if not cleaned:
        return None, "SQLSolver requires at least one referenced base table"
    return "\n".join(cleaned), "supported nullable relational fragment; no schema constraints assumed"


def parse_sqlsolver_verdict(output: str | None) -> str | None:
    """Accept one exact verdict only; logs, mixed verdicts and substrings fail."""
    if not isinstance(output, str):
        return None
    value = output.strip()
    return value if value in {"EQ", "NEQ", "UNKNOWN", "TIMEOUT", "UNKNOWN_TIMEOUT"} else None


def _run_sqlsolver(original: str, candidate: str, schema_sql: str, timeout: float) -> dict:
    directory = Path(__file__).resolve().parents[1] / "vendor/QUITE/src/Rewrite_Middleware/Hybrid_SQL_Corrector"
    jar = directory / "sqlsolver-v1.1.0.jar"
    if not jar.is_file() or shutil.which("java") is None:
        return {"verdict": None, "status": "unavailable", "reason": "SQLSolver jar or Java unavailable"}
    with tempfile.TemporaryDirectory(prefix="mcts_solver_") as temporary:
        temporary = Path(temporary)
        for name, content in (("original.sql", original), ("candidate.sql", candidate), ("schema.sql", schema_sql)):
            (temporary / name).write_text(content, encoding="utf-8")
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(directory) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        command = ["java", "-Xmx1g", "-jar", str(jar),
                   f"-sql1={temporary / 'original.sql'}", f"-sql2={temporary / 'candidate.sql'}",
                   f"-schema={temporary / 'schema.sql'}", f"-output={temporary / 'verdict.txt'}"]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired:
            return {"verdict": "TIMEOUT", "status": "timeout", "reason": "SQLSolver wall-clock limit"}
        except OSError as error:
            return {"verdict": None, "status": "error", "reason": str(error)}
        output_path = temporary / "verdict.txt"
        output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        verdict = parse_sqlsolver_verdict(output)
        diagnostics = {"returncode": process.returncode, "output": output[:4096],
                       "stdout": process.stdout[-4096:], "stderr": process.stderr[-4096:]}
        if process.returncode != 0 or re.search(r"exception|\berror\b", process.stderr, re.I):
            return {"verdict": None, "status": "error", **diagnostics}
        return {"verdict": verdict, "status": "ok" if verdict else "ambiguous", **diagnostics}


def _value_for(dtype, index: int):
    kind = dtype.this.value
    if kind in {"INT", "BIGINT", "SMALLINT", "TINYINT", "SERIAL", "BIGSERIAL"}:
        return index
    if kind in {"DECIMAL", "DOUBLE", "FLOAT", "REAL"}:
        return Decimal(index) / Decimal(10)
    if kind in {"TEXT", "VARCHAR", "CHAR"}:
        return str(index % 10)
    if kind == "BOOLEAN":
        return bool(index % 2)
    if kind == "DATE":
        return date(2000, 1, min(28, max(1, index)))
    if kind in {"TIMESTAMP", "TIMESTAMPTZ"}:
        return datetime(2000, 1, min(28, max(1, index)))
    if kind in {'TIME', 'TIMETZ'}:
        return time(index % 24, index % 60, index % 60)
    raise ValueError(f"miniature synthesis unsupported type: {kind}")


def _literal_pools(query_sqls) -> dict:
    """Seed column domains from actual comparison/IN/BETWEEN query literals."""
    pools = {}
    def add(column, value):
        if not isinstance(column, exp.Column):
            return
        if isinstance(value, exp.Cast):
            value = value.this
        sign = 1
        if isinstance(value, exp.Neg):
            value, sign = value.this, -1
        if not isinstance(value, exp.Literal):
            return
        if value.is_string:
            samples = [value.this]
            try:
                parsed = date.fromisoformat(value.this)
                samples += [(parsed + timedelta(days=offset)).isoformat() for offset in (-1, 1)]
            except ValueError:
                if '%' in value.this or '_' in value.this:
                    samples.append(value.this.replace('%', '').replace('_', 'x'))
        else:
            numeric = Decimal(value.this) * sign
            samples = [numeric, numeric - 1, numeric + 1]
        pool = pools.setdefault(_pg_identifier_name(column), [])
        for sample in samples:
            if sample not in pool:
                pool.append(sample)
    for source in query_sqls or []:
        try:
            query = parse_read_query(source)
        except Exception:
            continue
        for node in query.walk():
            if isinstance(node, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Like, exp.ILike)):
                add(node.this, node.expression)
                add(node.expression, node.this)
            elif isinstance(node, exp.In):
                for value in node.expressions:
                    add(node.this, value)
            elif isinstance(node, exp.Between):
                add(node.this, node.args.get('low'))
                add(node.this, node.args.get('high'))
    return pools


def _pooled_value(dtype, value):
    kind = dtype.this.value
    if kind in {"INT", "BIGINT", "SMALLINT", "TINYINT", "SERIAL", "BIGSERIAL"}:
        return int(Decimal(value))
    if kind in {"DECIMAL", "DOUBLE", "FLOAT", "REAL"}:
        return Decimal(value)
    if kind in {"TEXT", "VARCHAR", "CHAR"}:
        return str(value)
    if kind == "DATE":
        return date.fromisoformat(str(value))
    if kind in {"TIMESTAMP", "TIMESTAMPTZ"}:
        return datetime.fromisoformat(str(value))
    if kind in {'TIME', 'TIMETZ'}:
        return time.fromisoformat(str(value))
    if kind == "BOOLEAN":
        if str(value).lower() not in {"true", "false", "0", "1"}:
            raise ValueError("unsupported boolean literal")
        return str(value).lower() in {"true", "1"}
    raise ValueError(f"unsupported pool type: {kind}")


def _synthesize_cases(db: Postgres, schema_sql: str, original: str, candidate: str,
                      *, retain=False, read_dsn=None, query_sqls=None, _created_schemas=None) -> list[dict]:
    """Create constraint-enforced, isolated schemas; all are dropped on exit.

    Constraint violations are recorded, never ignored as successful coverage.
    FK references use the same key domains; insertion retries handle acyclic
    table dependencies. Arbitrary CHECK constraints may reduce usable coverage.
    """
    statements = [s for s in sqlglot.parse(schema_sql, read="postgres", error_level=ErrorLevel.RAISE) if s is not None]
    if not statements or any(not isinstance(s, exp.Create) or s.kind != "TABLE" or not isinstance(s.this, exp.Schema) for s in statements):
        raise ValueError("miniature synthesis requires CREATE TABLE DDL only")
    if any(t.db or t.catalog for s in statements for t in s.find_all(exp.Table)):
        raise ValueError("miniature synthesis does not remap explicitly qualified tables")
    if any(t.db or t.catalog for q in (original, candidate) for t in parse_read_query(q).find_all(exp.Table)):
        raise ValueError("miniature query uses qualified tables")
    cases = []
    pools = _literal_pools(query_sqls or [original, candidate])
    for case_name in ("empty", "null", "duplicate", "skew", "unmatched"):
        namespace = "mcts_mini_" + uuid.uuid4().hex
        setup = psycopg.connect(db.dsn, autocommit=True)
        retained = False
        try:
            with setup.cursor() as cursor:
                cursor.execute(psql.SQL("CREATE SCHEMA {} ").format(psql.Identifier(namespace)))
                if _created_schemas is not None:
                    _created_schemas.append(namespace)
                cursor.execute("SELECT set_config('search_path', %s, false)", (namespace,))
                pending_ddl = list(statements)
                while pending_ddl:
                    remaining = []
                    for statement in pending_ddl:
                        try:
                            cursor.execute(statement.sql(dialect="postgres"))
                        except psycopg.errors.UndefinedTable:
                            remaining.append(statement)
                    if len(remaining) == len(pending_ddl):
                        raise ValueError("miniature DDL has cyclic or unresolved references")
                    pending_ddl = remaining
                inserted, violations = 0, 0
                rows_by_table = {_pg_identifier_name(statement.this.this): 0 for statement in statements}
                pending_rows = []
                if case_name != "empty":
                    for table_index, statement in enumerate(statements):
                        table = statement.this.this
                        columns = [c for c in statement.this.expressions if isinstance(c, exp.ColumnDef)]
                        key_names = set()
                        for node in statement.this.walk():
                            if isinstance(node, (exp.PrimaryKey, exp.UniqueColumnConstraint)):
                                key_names.update(_pg_identifier_name(x) for x in node.find_all(exp.Identifier))
                        for column in columns:
                            if any(isinstance(n, (exp.PrimaryKeyColumnConstraint, exp.UniqueColumnConstraint)) for n in column.walk()):
                                key_names.add(_pg_identifier_name(column))
                        foreign_names = {_pg_identifier_name(identifier) for node in statement.this.find_all(exp.ForeignKey)
                                         for identifier in node.expressions}
                        foreign_names.update(_pg_identifier_name(column) for column in columns if column.find(exp.Reference))
                        pool_size = max([len(pools.get(_pg_identifier_name(c), [])) for c in columns] + [3])
                        row_count = max(6 if case_name == "skew" else 3, min(64, pool_size * 2))
                        if case_name == "unmatched":
                            row_count += len(statements) - table_index
                        for index in range(1, row_count + 1):
                            values = []
                            for column_index, column in enumerate(columns):
                                column_name = _pg_identifier_name(column)
                                unique = column_name in key_names
                                not_null = unique or any(isinstance(n, exp.NotNullColumnConstraint) for n in column.walk())
                                sample = (1 if column_name in foreign_names else
                                          index if unique else column_index + 1)
                                value = _value_for(column.args["kind"], sample)
                                pool = pools.get(column_name, [])
                                if pool and not unique and column_name not in foreign_names:
                                    pool_index = (index - 1) // 2 if case_name == 'duplicate' else index - 1
                                    if case_name == 'skew' and index % 5:
                                        pool_index = 0
                                    try:
                                        value = _pooled_value(column.args['kind'], pool[pool_index % len(pool)])
                                    except (ValueError, TypeError, ArithmeticError):
                                        pass
                                if case_name == "null" and not not_null and index % 2:
                                    value = None
                                values.append(value)
                            pending_rows.append((_pg_identifier_name(table), [_pg_identifier_name(c) for c in columns], values))
                while pending_rows:
                    remaining = []
                    for table, columns, values in pending_rows:
                        query = psql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                            psql.Identifier(table), psql.SQL(",").join(map(psql.Identifier, columns)),
                            psql.SQL(",").join(psql.Placeholder() for _ in values))
                        try:
                            cursor.execute(query, values)
                            inserted += 1
                            rows_by_table[table] += 1
                        except psycopg.errors.ForeignKeyViolation:
                            remaining.append((table, columns, values))
                        except (psycopg.IntegrityError, psycopg.DataError):
                            violations += 1
                    if len(remaining) == len(pending_rows):
                        violations += len(remaining)
                        break
                    pending_rows = remaining
            with Postgres(db.dsn, timeout_ms=db.timeout_ms, search_path=namespace,
                          **_comparison_options(db)) as miniature:
                comparison = miniature.compare(original, candidate)
            cases.append({"name": case_name, "rows_inserted": inserted,
                          "rows_by_table": rows_by_table, "constraint_rejections": violations,
                          "literal_pool_sizes": {key: len(value) for key, value in pools.items()},
                          "dsn": read_dsn or db.dsn, "search_path": namespace,
                          "comparison": comparison})
            retained = retain and comparison.get('equivalent', False)
            if not comparison.get("equivalent"):
                break
        finally:
            try:
                with setup.cursor() as cursor:
                    if not retained:
                        cursor.execute(psql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(psql.Identifier(namespace)))
            finally:
                setup.close()
    return cases


def synthesize_mini_databases(admin_dsn: str, schema_sql: str, read_dsn: str | None = None,
                              query_sqls=None) -> list[dict]:
    """Prepare five persistent, constraint-enforced schemas for a read-only role.

    Return JSON-serializable specs accepted by EquivalenceGate(mini_dbs=...).
    The caller owns eventual DROP SCHEMA cleanup. If any nonempty case cannot
    populate every declared table, retain evidence but mark coverage incomplete.
    Query literals seed domains so benchmark filters do not all see empty data.
    """
    created_schemas = []
    try:
        with Postgres(admin_dsn) as admin:
            specs = _synthesize_cases(admin, schema_sql, 'SELECT 1 AS fixture_probe',
                                      'SELECT 1 AS fixture_probe', retain=True,
                                      read_dsn=read_dsn, query_sqls=query_sqls,
                                      _created_schemas=created_schemas)
    except Exception:
        with psycopg.connect(admin_dsn, autocommit=True) as setup:
            for namespace in created_schemas:
                setup.execute(psql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(psql.Identifier(namespace)))
        raise
    for spec in specs:
        spec.pop('comparison', None)
        spec['coverage_complete'] = (spec['name'] == 'empty' or
                                     all(spec['rows_by_table'].values()))
    return specs


class EquivalenceGate:
    def __init__(self, db: Postgres, schema_sql: str, mode: str = "strict", mini_dbs=None, solver_timeout: float = 10):
        if mode not in {"strict", "differential", "exploratory", "hybrid"}:
            raise ValueError("equivalence mode must be strict, hybrid or differential")
        if solver_timeout <= 0:
            raise ValueError("solver_timeout must be positive")
        self.db, self.schema_sql, self.mode = db, schema_sql, mode
        self.mini_dbs, self.solver_timeout = mini_dbs, solver_timeout

    def verify(self, original: str, candidate: str) -> dict:
        result = {"accepted": False, "proof": "none", "formal": False,
                  "execution": {}, "reason": "no equivalence evidence"}
        try:
            original_ast, candidate_ast = parse_read_query(original), parse_read_query(candidate)
            original_key, candidate_key = conservative_identity(original), conservative_identity(candidate)
            identity = original_key == candidate_key
        except Exception as error:
            result["reason"] = f"unsupported or unsafe SQL: {error}"
            return result
        all_hints = original_key[1] + candidate_key[1]
        if all_hints:
            if not all(_approved_planner_hint(comment) for comment in all_hints):
                result['reason'] = 'unknown or semantic optimizer hint is not approved'
                return result
            capability = self.db.planner_hints_enabled() if hasattr(self.db, 'planner_hints_enabled') else {'enabled': False}
            result['planner_hints'] = capability
            if not capability.get('enabled'):
                result['reason'] = 'pg_hint_plan is not installed, loaded and enabled'
                return result
        execution = self.db.compare(original, candidate)
        result["execution"] = execution
        execution_error = str(execution.get("error", "")).lower()
        comparison_timeout = (execution.get("sqlstate") == "57014"
                              or "statement timeout" in execution_error
                              or "canceling statement" in execution_error)
        if not execution.get("equivalent") and not comparison_timeout:
            result["reason"] = "real-database comparison failed: " + execution.get("error", "unequal results")
            return result
        if comparison_timeout:
            # A slow Original is not evidence of inequivalence.  Continue to
            # the audited SQLSolver/miniature/witness lanes; the verification
            # database or the five-case fixtures can still establish equality
            # without requiring the full performance query to finish.
            result["comparison_timeout_fallback"] = True
            result["comparison_timeout_reason"] = execution.get("error", "statement timeout")
        relaxed = execution.get('numeric_relaxation_used', False)
        if identity and not relaxed:
            result.update(accepted=True, proof="identity" if original.strip() == candidate.strip() else "ast",
                          formal=True, reason="conservative identity and same-snapshot real-database comparison")
            return result
        if not relaxed and original_key[0] == candidate_key[0] and all(
                _approved_planner_hint(comment) for comment in original_key[1] + candidate_key[1]):
            if result.get('planner_hints', {}).get('enabled'):
                result.update(accepted=True, proof='ast', formal=True,
                              reason='audited planner-only hint identity and same-snapshot real-database comparison')
                return result
        try:
            solver_schema, support_reason = _solver_input(original_ast, candidate_ast, self.schema_sql)
        except Exception as error:
            solver_schema, support_reason = None, f"solver support check failed: {error}"
        solver = {"verdict": None, "status": "unsupported", "reason": support_reason}
        if solver_schema is not None:
            solver = _run_sqlsolver(original, candidate, solver_schema, self.solver_timeout)
            solver["fragment"] = support_reason
        result["solver"] = solver
        if solver.get("verdict") == "NEQ":
            result["reason"] = "SQLSolver disproved equivalence"
            return result
        formal = solver.get("verdict") == "EQ" and solver.get("status") == "ok" and not relaxed
        if relaxed:
            solver['reason'] = 'numeric comparison relaxation is finite acceptance evidence, not a formal equivalence proof'
        if formal and hasattr(self.db, 'formal_context_supported'):
            context = self.db.formal_context_supported(original, candidate, schema_sql=solver_schema)
            result['formal_context'] = context
            if not context.get('supported'):
                formal = False
                solver['reason'] = 'live schema context is outside the supported formal fragment'
        if self.mode == "strict" or (self.mode == "hybrid" and formal):
            if formal:
                result.update(accepted=True, proof="sqlsolver", formal=True,
                              reason="supported SQLSolver proof and same-snapshot real-database comparison")
            else:
                result["reason"] = "strict mode requires formal evidence; " + solver.get("reason", solver.get("status", "unknown"))
            return result
        # Exploration may use finite tests, but must never label them formal.
        try:
            if self.mini_dbs is None or self.mini_dbs == "auto":
                miniatures = _synthesize_cases(self.db, self.schema_sql, original, candidate)
            else:
                miniatures = []
                for index, mini_db in enumerate(self.mini_dbs):
                    if isinstance(mini_db, dict):
                        if not mini_db.get('coverage_complete'):
                            raise ValueError('prebuilt miniature lacks complete table coverage')
                        if any(t.db or t.catalog for query in (original_ast, candidate_ast) for t in query.find_all(exp.Table)):
                            raise ValueError('prebuilt miniature cannot remap schema-qualified queries')
                        with Postgres(mini_db['dsn'], timeout_ms=self.db.timeout_ms,
                                      search_path=mini_db['search_path'], **_comparison_options(self.db)) as fixture:
                            comparison = fixture.compare(original, candidate)
                        miniatures.append({**{k:v for k,v in mini_db.items() if k != 'dsn'}, 'comparison': comparison})
                    else:
                        comparison = mini_db.compare(original, candidate)
                        miniatures.append({"name": f"provided-{index}", "comparison": comparison})
            result["miniatures"] = miniatures
        except Exception as error:
            result["reason"] = f"miniature differential testing failed: {type(error).__name__}: {error}"
            return result
        if len(miniatures) < 5 or any(not m["comparison"].get("equivalent") for m in miniatures):
            result["reason"] = "miniature differential case failed or no cases were supplied"
            return result
        if any(m['comparison'].get('numeric_relaxation_used') for m in miniatures):
            formal = False
        if any(m.get("rows_inserted") == 0 for m in miniatures if m["name"] != "empty" and "rows_inserted" in m):
            result["reason"] = "miniature synthesis did not produce nonempty coverage"
            return result
        if any(m.get('rows_by_table') and not all(m['rows_by_table'].values())
               for m in miniatures if m['name'] != 'empty'):
            result['reason'] = 'miniature synthesis left a declared table empty'
            return result
        if not formal and all(not m['comparison'].get('original', {}).get('row_count', 0)
                              for m in miniatures):
            result['reason'] = 'all miniature query outputs were empty; finite evidence is vacuous'
            return result
        fingerprints = [m['comparison'].get('original', {}).get('sha256') for m in miniatures]
        if not formal and (all(not m['comparison'].get('original', {}).get('nonnull_cells', 0)
                               for m in miniatures) or len(set(fingerprints)) <= 1):
            result['reason'] = 'miniature outputs lack non-NULL value variation across cases; finite evidence is vacuous'
            return result
        result.update(accepted=True, proof="sqlsolver" if formal else "differential", formal=formal,
                      reason="formal proof plus finite checks" if formal else "finite real/miniature data tests only; not a proof")
        return result
