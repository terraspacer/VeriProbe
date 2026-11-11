"""Conservative PostgreSQL output-contract proposals for Class A rewrites.

This adapter never establishes equivalence.  Its output must pass the existing
root result/proof/miniature gate before it can become a search node.  Native
standalone baseline SQL must not be passed through this adapter.
"""
from __future__ import annotations

import psycopg
from psycopg import sql
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from .db import Postgres, _description, has_order, parse_read_query


_CONTRACT_FIELDS = ("type_oid", "type_modifier", "internal_size", "precision", "scale")
_INTEGER_TYPES = {21: (2, "SMALLINT"), 23: (4, "INTEGER"), 20: (8, "BIGINT")}
_WIDENINGS = frozenset({(21, 23), (21, 20), (23, 20)})
_RELATION_ALIAS = "__mcts_contract_source"


class _UnsupportedContract(ValueError):
    pass


def _statement_body(source: str) -> str:
    """Remove only terminal statement delimiters, retaining raw SQL/comments."""
    tokens = sqlglot.tokenize(source, read="postgres")
    delimiters = []
    for token in reversed(tokens):
        if token.token_type != TokenType.SEMICOLON:
            break
        delimiters.append(token)
    for token in delimiters:
        source = source[:token.start] + source[token.end + 1:]
    return source


def _audit_context(db: Postgres, cursor, queries) -> None:
    """Exclude hidden executable expressions before PostgreSQL plans LIMIT 0.

    LIMIT 0 avoids scanning rows, but planning can fold immutable functions.
    Keep the eligible context narrow: ordinary builtin-typed base tables, no
    RLS/inheritance/expression indexes, and no visible operator overloads or
    custom builtin-to-builtin cast implementations.
    """
    tables = set()
    for query in queries:
        db._audit_functions(cursor, query)
        for scope in traverse_scope(query):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    relation = source.copy()
                    relation.set("alias", None)
                    tables.add(relation.sql(dialect="postgres"))
    cursor.execute(
        "SELECT n.nspname,o.oprname FROM pg_operator o "
        "JOIN pg_namespace n ON n.oid=o.oprnamespace "
        "WHERE n.nspname=ANY(current_schemas(true)) "
        "AND n.nspname<>'pg_catalog' LIMIT 1"
    )
    if cursor.fetchone():
        raise _UnsupportedContract("visible user-defined operators are unsupported")
    cursor.execute(
        "SELECT 1 FROM pg_cast c JOIN pg_proc p ON p.oid=c.castfunc "
        "JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_type s ON s.oid=c.castsource JOIN pg_type t ON t.oid=c.casttarget "
        "WHERE n.nspname<>'pg_catalog' "
        "AND s.typnamespace='pg_catalog'::regnamespace "
        "AND t.typnamespace='pg_catalog'::regnamespace LIMIT 1"
    )
    if cursor.fetchone():
        raise _UnsupportedContract("custom casts between builtin types are unsupported")
    if not tables:
        return
    cursor.execute(
        "SELECT c.oid,c.relkind,c.relrowsecurity,c.relforcerowsecurity "
        "FROM unnest(%s::text[]) names(name) "
        "LEFT JOIN pg_class c ON c.oid=to_regclass(names.name)",
        (sorted(tables),),
    )
    relations = cursor.fetchall()
    if any(not oid or kind != "r" or rls or force_rls
           for oid, kind, rls, force_rls in relations):
        raise _UnsupportedContract("only ordinary base tables without RLS are supported")
    relation_oids = [row[0] for row in relations]
    cursor.execute(
        "SELECT 1 FROM pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s) LIMIT 1",
        (relation_oids, relation_oids),
    )
    if cursor.fetchone():
        raise _UnsupportedContract("inherited or partitioned relations are unsupported")
    cursor.execute(
        "SELECT 1 FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid "
        "WHERE a.attrelid=ANY(%s) AND a.attnum>0 AND NOT a.attisdropped "
        "AND t.typnamespace<>'pg_catalog'::regnamespace LIMIT 1",
        (relation_oids,),
    )
    if cursor.fetchone():
        raise _UnsupportedContract("base-table user-defined column types are unsupported")
    cursor.execute(
        "SELECT 1 FROM pg_index WHERE indrelid=ANY(%s) "
        "AND (indexprs IS NOT NULL OR indpred IS NOT NULL) LIMIT 1",
        (relation_oids,),
    )
    if cursor.fetchone():
        raise _UnsupportedContract("expression or partial indexes are unsupported")


def _describe(cursor, body: str) -> list[dict]:
    cursor.execute(sql.SQL("SELECT * FROM (\n{}\n) AS {} LIMIT 0").format(
        sql.SQL(body), sql.Identifier(_RELATION_ALIAS)))
    if cursor.description is None or cursor.pgresult is None:
        raise _UnsupportedContract("query has no PostgreSQL result description")
    columns = _description(cursor.description)
    for index, column in enumerate(columns):
        column["type_modifier"] = cursor.pgresult.fmod(index)
    return columns


def _same_type(left: dict, right: dict) -> bool:
    return all(left[key] == right[key] for key in _CONTRACT_FIELDS)


def _cast_for(source: dict, target: dict) -> str | None:
    if _same_type(source, target):
        return None
    pair = (source["type_oid"], target["type_oid"])
    if pair not in _WIDENINGS:
        raise _UnsupportedContract("output type/typmod changes require a nonpermitted cast")
    for column in (source, target):
        size, _ = _INTEGER_TYPES[column["type_oid"]]
        if (column["type_modifier"] != -1 or column["internal_size"] != size
                or column["precision"] is not None or column["scale"] is not None):
            raise _UnsupportedContract("unexpected integer output metadata")
    return _INTEGER_TYPES[target["type_oid"]][1]


def _projection(body: str, original_columns: list[dict], casts: list[str | None]) -> str:
    aliases = [f"__mcts_contract_{index + 1}" for index in range(len(original_columns))]
    expressions = []
    for alias, column, cast in zip(aliases, original_columns, casts):
        reference = sql.Identifier(_RELATION_ALIAS, alias)
        if cast is not None:
            reference = sql.SQL("CAST({} AS {})").format(reference, sql.SQL(cast))
        expressions.append(sql.SQL("{} AS {}").format(reference, sql.Identifier(column["name"])))
    return sql.SQL("SELECT {}\nFROM (\n{}\n) AS {} ({})").format(
        sql.SQL(", ").join(expressions), sql.SQL(body), sql.Identifier(_RELATION_ALIAS),
        sql.SQL(", ").join(map(sql.Identifier, aliases)),
    ).as_string()


def restore_output_contract(db: Postgres, original_sql: str, candidate_sql: str) -> dict:
    """Propose positional relabeling and lossless builtin integer widening.

    ``status`` is ``unchanged``, ``restored``, ``rejected``, ``error``, or
    ``timeout``.  Only the first two return a non-None ``sql``.  All results
    retain the raw candidate and explicitly require normal root verification.
    A metadata match is never an equivalence or formal-proof verdict.
    """
    result = {
        "status": "error", "sql": None, "raw_sql": candidate_sql,
        "original_sql": original_sql, "original_columns": [], "candidate_columns": [],
        "restored_columns": [], "restoration": [], "formal": False,
        "requires_verification": True, "policy": "postgres_projection_contract_v1",
        "description_method": "audited SELECT subquery LIMIT 0 in one read-only transaction",
    }
    try:
        original = parse_read_query(original_sql)
        candidate = parse_read_query(candidate_sql)
        original_body = _statement_body(original_sql)
        candidate_body = _statement_body(candidate_sql)
        with db.connection.transaction():
            with db.connection.cursor() as cursor:
                db._configure(cursor)
                _audit_context(db, cursor, (original, candidate))
                original_columns = _describe(cursor, original_body)
                result["original_columns"] = original_columns
                candidate_columns = _describe(cursor, candidate_body)
                result["candidate_columns"] = candidate_columns
                if not original_columns or len(original_columns) != len(candidate_columns):
                    raise _UnsupportedContract("nonzero equal positional output counts are required")
                casts = [_cast_for(source, target)
                         for source, target in zip(candidate_columns, original_columns)]
                if original_columns == candidate_columns:
                    result.update(status="unchanged", sql=candidate_sql)
                    return result
                if has_order(original) or has_order(candidate):
                    raise _UnsupportedContract("restoration of root-ordered queries is unsupported")
                restored_sql = _projection(candidate_body, original_columns, casts)
                restored = parse_read_query(restored_sql)
                db._audit_functions(cursor, restored)
                restored_columns = _describe(cursor, restored_sql)
                result["restored_columns"] = restored_columns
                if restored_columns != original_columns:
                    raise _UnsupportedContract("restored PostgreSQL metadata does not match the original")
                result["restoration"] = [
                    {"position": index + 1, "source_name": source["name"],
                     "target_name": target["name"], "source_type_oid": source["type_oid"],
                     "target_type_oid": target["type_oid"], "cast": cast}
                    for index, (source, target, cast)
                    in enumerate(zip(candidate_columns, original_columns, casts))
                ]
                result.update(status="restored", sql=restored_sql)
    except Exception as error:
        result.update(
            status="rejected" if isinstance(error, (ValueError, sqlglot.errors.SqlglotError))
            else "timeout" if isinstance(error, (psycopg.errors.QueryCanceled, TimeoutError)) else "error",
            reason=f"{type(error).__name__}: {error}", error_type=type(error).__name__,
            sqlstate=getattr(error, "sqlstate", None),
        )
    return result
