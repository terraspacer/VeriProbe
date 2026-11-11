"""Opt-in, deliberately small PostgreSQL empty-result proof for study v2.

This is not an inference from empty fixtures. COUNT used as a window function
is non-NULL even for an empty frame. An IS NULL filter of that exact value is
therefore false. The other supported empty root is a literal WHERE FALSE.
Only transparent, single-source SELECT/CTE/subquery chains are modeled.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import re

from psycopg import sql as psql
import sqlglot
from sqlglot import ErrorLevel, exp

from .db import parse_read_query
from .verification_db import ScopedEquivalenceGate
from .verify import EquivalenceGate, conservative_identity


VERSION = "structural-empty-v2"
CONTEXT_KEY = "structural_empty_v2_context"


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _identifier(node):
    if (not isinstance(node, exp.Identifier) or node.args.get("quoted")
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node.this)):
        raise ValueError("only simple unquoted identifiers are supported")
    return node.this.lower()


def _only_args(node, names):
    if any(value is not None and value is not False and value != []
           for key, value in node.args.items() if key not in names):
        raise ValueError(f"unsupported {type(node).__name__} option")


def _schema(schema_sql):
    tables = {}
    for statement in sqlglot.parse(schema_sql, read="postgres", error_level=ErrorLevel.RAISE):
        if statement is None:
            continue
        if (not isinstance(statement, exp.Create) or statement.kind != "TABLE"
                or not isinstance(statement.this, exp.Schema)):
            raise ValueError("source schema must contain only CREATE TABLE declarations")
        table = statement.this.this
        if not isinstance(table, exp.Table):
            raise ValueError("invalid source table")
        _only_args(table, {"this"})
        name = _identifier(table.this)
        columns = [_identifier(column.this) for column in statement.this.expressions
                   if isinstance(column, exp.ColumnDef)]
        if not columns or len(columns) != len(set(columns)) or name in tables:
            raise ValueError("source schema has missing or duplicate names")
        tables[name] = columns
    if not tables:
        raise ValueError("source schema must not be empty")
    return tables


@dataclass
class _Relation:
    # The only non-unknown fact is the exact non-NULL COUNT-window value.
    columns: dict[str, str]
    tables: set[str]
    steps: list[dict] = field(default_factory=list)
    empty: bool = False


def _alias_columns(relation, alias):
    if alias is None:
        return relation
    _only_args(alias, {"this", "columns"})
    _identifier(alias.this)
    renames = [_identifier(column) for column in alias.args.get("columns") or []]
    if len(renames) > len(relation.columns):
        raise ValueError("relation column alias list is too long")
    names = renames + list(relation.columns)[len(renames):]
    if len(names) != len(set(names)):
        raise ValueError("duplicate relation output names")
    return _Relation(dict(zip(names, relation.columns.values())), relation.tables,
                     list(relation.steps), relation.empty)


class _Proof:
    def __init__(self, schema):
        self.schema = schema

    def column(self, column, source, qualifier):
        if not isinstance(column, exp.Column):
            raise ValueError("only direct input columns are supported")
        _only_args(column, {"this", "table"})
        name = _identifier(column.this)
        if column.args.get("table") is not None and _identifier(column.args["table"]) != qualifier:
            raise ValueError("column qualifier does not bind to this source")
        if name not in source.columns:
            raise ValueError("column is absent from its bound source")
        return source.columns[name]

    def window_spec(self, window, source, qualifier, *, definition=False):
        _only_args(window, {"this", "partition_by", "order", "spec", "alias", "over"})
        if definition:
            _identifier(window.this)
            if window.args.get("alias") or window.args.get("over"):
                raise ValueError("inherited named windows are unsupported")
        for column in window.args.get("partition_by") or []:
            self.column(column, source, qualifier)
        order = window.args.get("order")
        if order is not None:
            _only_args(order, {"expressions"})
            for ordered in order.expressions:
                if not isinstance(ordered, exp.Ordered):
                    raise ValueError("unsupported window ordering")
                _only_args(ordered, {"this", "desc", "nulls_first"})
                self.column(ordered.this, source, qualifier)
        spec = window.args.get("spec")
        if spec is not None:
            _only_args(spec, {"kind", "start", "start_side", "end", "end_side"})
            if spec.args.get("kind") != "ROWS":
                raise ValueError("only simple ROWS frames are supported")
            for key in ("start", "end"):
                value = spec.args.get(key)
                if value is None or value in ("UNBOUNDED", "CURRENT ROW"):
                    continue
                if (not isinstance(value, exp.Literal) or value.is_string
                        or not re.fullmatch(r"[0-9]+", value.this)
                        or int(value.this) > 2147483647):
                    raise ValueError("unsupported window frame bound")
            if any(spec.args.get(key) not in (None, "PRECEDING", "FOLLOWING")
                   for key in ("start_side", "end_side")):
                raise ValueError("unsupported window frame direction")

    def expression(self, value, source, qualifier, windows):
        if isinstance(value, exp.Column):
            return self.column(value, source, qualifier)
        if isinstance(value, exp.Window):
            if not isinstance(value.this, exp.Count) or value.args.get("over") != "OVER":
                raise ValueError("only COUNT window functions are supported")
            count = value.this
            _only_args(count, {"this", "big_int"})
            if isinstance(count.this, exp.Star):
                _only_args(count.this, set())
            else:
                # The counted column may itself be nullable: COUNT still is not.
                self.column(count.this, source, qualifier)
            self.window_spec(value, source, qualifier)
            reference = value.args.get("alias")
            if reference is not None and _identifier(reference) not in windows:
                raise ValueError("window reference is not declared in this SELECT")
            return "count_window_nonnull"
        if isinstance(value, exp.Literal):
            if value.is_string or not re.fullmatch(r"[0-9]+", value.this) or int(value.this) > 2**63 - 1:
                raise ValueError("only bounded nonnegative integer literals are supported")
            return "unknown"
        if type(value) is exp.Cast:
            # The saved candidate uses 0::bigint for exact result metadata.
            _only_args(value, {"this", "to"})
            dtype = value.args.get("to")
            if (not isinstance(dtype, exp.DataType) or dtype.this != exp.DataType.Type.BIGINT
                    or dtype.expressions or not isinstance(value.this, exp.Literal)):
                raise ValueError("only an integer literal cast to BIGINT is supported")
            self.expression(value.this, source, qualifier, windows)
            return "unknown"
        raise ValueError(f"unsupported expression: {type(value).__name__}")

    def select(self, node, inherited=None, path="root"):
        if not isinstance(node, exp.Select):
            raise ValueError("only SELECT relations are supported")
        _only_args(node, {"expressions", "from", "where", "windows", "with"})
        ctes = dict(inherited or {})
        with_ = node.args.get("with")
        if with_ is not None:
            _only_args(with_, {"expressions"})
            local_names = set()
            for cte in with_.expressions:
                _only_args(cte, {"this", "alias", "materialized"})
                alias = cte.args.get("alias")
                name = _identifier(alias.this)
                if name in local_names or name in self.schema or name in ctes:
                    raise ValueError("duplicate or shadowed CTE names are unsupported")
                local_names.add(name)
                relation = self.select(cte.this, ctes, path + ".cte." + name)
                ctes[name] = _alias_columns(relation, alias)
        from_ = node.args.get("from")
        if from_ is None:
            raise ValueError("a single bound base-table source is required")
        _only_args(from_, {"this"})
        source_node = from_.this
        if isinstance(source_node, exp.Table):
            _only_args(source_node, {"this", "alias"})
            name = _identifier(source_node.this)
            if name in ctes:
                source = ctes[name]
            elif name in self.schema:
                source = _Relation(dict.fromkeys(self.schema[name], "unknown"), {name})
            else:
                raise ValueError("source table is absent from the supplied schema")
            source = _alias_columns(source, source_node.args.get("alias"))
            qualifier = _identifier(source_node.args["alias"].this) if source_node.args.get("alias") else name
        elif isinstance(source_node, exp.Subquery):
            _only_args(source_node, {"this", "alias"})
            alias = source_node.args.get("alias")
            if alias is None:
                raise ValueError("a derived source needs an explicit alias")
            source = _alias_columns(self.select(source_node.this, ctes, path + ".from"), alias)
            qualifier = _identifier(alias.this)
        else:
            raise ValueError("unsupported FROM source")
        windows = {}
        for window in node.args.get("windows") or []:
            name = _identifier(window.this)
            if name in windows:
                raise ValueError("duplicate named window")
            self.window_spec(window, source, qualifier, definition=True)
            windows[name] = window
        columns = {}
        steps = list(source.steps)
        for projection in node.expressions:
            if isinstance(projection, exp.Alias):
                _only_args(projection, {"this", "alias"})
                name, value = _identifier(projection.args["alias"]), projection.this
            elif isinstance(projection, exp.Column) and not isinstance(projection.this, exp.Star):
                name, value = _identifier(projection.this), projection
            elif isinstance(projection, exp.Window) and isinstance(projection.this, exp.Count):
                name, value = "count", projection
            else:
                raise ValueError("projections require direct columns or explicit aliases")
            if name in columns:
                raise ValueError("duplicate projection names are unsupported")
            columns[name] = self.expression(value, source, qualifier, windows)
            if isinstance(value, exp.Window):
                steps.append({"path": path, "rule": "count_window_is_nonnull",
                              "output_column": name,
                              "count_argument": value.this.this.sql(dialect="postgres")})
        if not columns:
            raise ValueError("empty projection")
        empty = source.empty
        where = node.args.get("where")
        if where is not None:
            _only_args(where, {"this"})
            predicate = where.this
            # Parentheses alone are transparent; compound predicates are not.
            while isinstance(predicate, exp.Paren):
                predicate = predicate.this
            if isinstance(predicate, exp.Boolean) and predicate.this is False:
                steps.append({"path": path, "rule": "literal_where_false"})
                empty = True
            elif (isinstance(predicate, exp.Is) and isinstance(predicate.expression, exp.Null)
                  and isinstance(predicate.this, exp.Column)):
                fact = self.column(predicate.this, source, qualifier)
                if fact != "count_window_nonnull":
                    raise ValueError("IS NULL input is not an exact COUNT-window value")
                steps.append({"path": path, "rule": "is_null_of_nonnull_count_window",
                              "input_column": _identifier(predicate.this.this)})
                empty = True
            else:
                raise ValueError("only literal FALSE or exact COUNT-window IS NULL predicates are supported")
        return _Relation(columns, set(source.tables), steps, empty)


def prove_tautological_empty(sql, schema_sql):
    """Return a bound proof receipt; unsupported SQL always fails closed."""
    if not isinstance(sql, str) or not isinstance(schema_sql, str):
        return {"version": VERSION, "status": "unsupported", "root_empty": False,
                "reason": "SQL and schema must be strings"}
    receipt = {"version": VERSION, "status": "unsupported", "root_empty": False,
               "query_sha256": _sha(sql), "schema_sha256": _sha(schema_sql)}
    try:
        schema = _schema(schema_sql)
        query = parse_read_query(sql)
        if any(comment.strip().startswith(("+", "!"))
               for node in query.walk() for comment in (node.comments or [])):
            raise ValueError("optimizer hints are outside this proof fragment")
        relation = _Proof(schema).select(query)
        if not relation.empty:
            raise ValueError("no supported contradiction reaches the root result")
        kind = ("count_window_is_null" if any(step["rule"] == "is_null_of_nonnull_count_window"
                                            for step in relation.steps) else "where_false")
        receipt.update(status="proven", root_empty=True, proof_kind=kind,
                       source_tables=sorted(relation.tables), steps=relation.steps)
    except Exception as error:
        receipt["reason"] = str(error)
    return receipt


class StructuralEmptyEquivalenceGateV2(EquivalenceGate):
    """Add the empty proof only with an exact, populated audited context."""
    def __init__(self, *args, structural_context=None, context_validator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.structural_context = deepcopy(structural_context)
        self.context_validator = context_validator

    def verify(self, original, candidate):
        proofs = {"original": prove_tautological_empty(original, self.schema_sql),
                  "candidate": prove_tautological_empty(candidate, self.schema_sql)}
        if any(proof["status"] != "proven" for proof in proofs.values()):
            return super().verify(original, candidate)
        # Preserve the existing identity and hint audit, including Original.
        if conservative_identity(original) == conservative_identity(candidate):
            return super().verify(original, candidate)
        result = {"accepted": False, "formal": False, "proof": "none", "execution": {},
                  "structural_empty": {"version": VERSION, "proofs": proofs},
                  "reason": "structural empty proof requires a bound populated verification context"}
        context = self.structural_context
        if (not isinstance(context, dict) or context.get("version") != VERSION
                or context.get("query_sha256") != _sha(original)
                or context.get("schema_sha256") != _sha(self.schema_sql)
                or context.get("postgresql_validated") is not True
                or self.context_validator is None):
            return result
        tables = set(_schema(self.schema_sql))
        counts = context.get("rows_by_table", {})
        if (set(context.get("source_tables", [])) != tables or set(counts) != tables
                or any(type(count) is not int or count <= 0 for count in counts.values())
                or any(not set(proof["source_tables"]) <= tables for proof in proofs.values())):
            return result
        try:
            execution = self.db.compare(original, candidate)
            result["execution"] = execution
            execution_error = str(execution.get("error", "")).lower()
            comparison_timeout = (execution.get("sqlstate") == "57014"
                                  or "statement timeout" in execution_error
                                  or "canceling statement" in execution_error)
            if comparison_timeout:
                # The structural proof is specifically intended for cases
                # where the full Original is too slow to execute.  A populated
                # audited witness context is sufficient to accept the proven
                # empty projection without fail-closing on that timeout.
                live = self.context_validator(original, candidate, context)
                result["structural_empty"]["verification_context"] = live
                if live.get("supported") is True:
                    result.update(accepted=True, formal=True, proof="structural_empty_v2",
                                  reason="independent structural empty proofs and populated audited verification tables; full comparison timed out")
                else:
                    result["reason"] = "structural proof verification context is not populated and bound"
                return result
            left_columns = execution.get("original", {}).get("columns")
            right_columns = execution.get("candidate", {}).get("columns")
            if (execution.get("equivalent") is not True or execution.get("same_snapshot") is not True
                    or execution.get("status") != "ok"
                    or execution.get("comparison_profile") != "positional_numeric"
                    or any(type(execution.get(side, {}).get("row_count")) is not int
                           or execution[side]["row_count"] != 0 for side in proofs)
                    or not isinstance(left_columns, list) or not isinstance(right_columns, list)
                    or not left_columns or len(left_columns) != len(right_columns)):
                result["reason"] = "structural proof requires equivalent same-snapshot empty positional_numeric results with equal nonzero arity"
                return result
            live = self.context_validator(original, candidate, context)
            result["structural_empty"]["verification_context"] = live
            if live.get("supported") is not True:
                result["reason"] = "structural proof verification context is not populated and bound"
                return result
            result.update(accepted=True, formal=True, proof="structural_empty_v2",
                          reason="independent structural empty proofs, positional_numeric execution compatibility, and populated audited verification tables")
        except Exception as error:
            result["reason"] = f"structural proof context failed: {type(error).__name__}: {error}"
        return result


class ScopedStructuralEmptyGateV2(ScopedEquivalenceGate):
    """Keep every legacy catalog/plan/lock guard around the v2-only proof."""
    def __init__(self, *args, **kwargs):
        provenance = kwargs.get("verification_provenance") or {}
        context = provenance.get(CONTEXT_KEY)
        if "gate_class" in kwargs:
            raise ValueError("v2 scoped gate owns its equivalence gate class")

        def factory(*gate_args, **gate_kwargs):
            return StructuralEmptyEquivalenceGateV2(*gate_args, **gate_kwargs,
                structural_context=context, context_validator=self._structural_context)

        super().__init__(*args, **kwargs, gate_class=factory)
        self.metadata["structural_empty_v2"] = {"enabled": context is not None, "version": VERSION}

    def _structural_context(self, original, candidate, context):
        if (context.get("schema_sha256") != self.validation.get("schema_source_sha256")
                or context.get("search_path") != self.verification_db.search_path
                or context.get("query_sha256") != _sha(original)):
            return {"supported": False, "reason": "query/schema/verification namespace binding mismatch"}
        for query in (original, candidate):
            self._query_scope(query, self.validation)
        nonempty = {}
        # Observe base-table existence, never query output variation. This read
        # does not touch performance rows and does not synthesize fixture data.
        with self.verification_db.connection.transaction():
            with self.verification_db.connection.cursor() as cursor:
                self.verification_db._configure(cursor)
                for name in sorted(context["source_tables"]):
                    relation = self.validation["verification"]["resolved_relations"][name]
                    cursor.execute(psql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(
                        psql.Identifier(relation["schema"], relation["relation"])))
                    nonempty[name] = cursor.fetchone()[0] is True
        return {"supported": bool(nonempty) and all(nonempty.values()),
                "method": "live_verification_base_table_exists", "nonempty_by_table": nonempty,
                "query_sha256": _sha(original), "schema_sha256": _sha(self.schema_sql),
                "catalog_sha256": self.validation["catalog_sha256"],
                "performance_database_rows_executed_or_hashed": False}
