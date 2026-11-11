"""Opt-in follow-up operators; the completed Stage A implementation is unchanged.

These are development proposals informed by Stage A, not new performance
measurements. Every emitted SQL still requires the existing equivalence gate.
"""
from __future__ import annotations

from copy import deepcopy

import sqlglot
from sqlglot import exp, ErrorLevel

from .adapters import exact_sha256
from .batched_operators import BatchedOperators
from .verify import _pg_identifier_name


ACTION_PATTERNS = [
    {"id": "not_null_count_star", "condition": "COUNT of a direct base-table column declared NOT NULL or PRIMARY KEY",
     "action": "Consider COUNT(*) with the same outer casts and filters. Check scope and null-extension before rewriting.",
     "safety": "Do not apply to nullable columns, DISTINCT counts, expressions, or nullable sides of outer joins."},
    {"id": "selective_dimension_materialization", "condition": "Multi-join star/snowflake query with selective dimension filtering, even when the dimension is referenced only once",
     "action": "Propose BOTH a compact filtered-dimension WITH ... AS MATERIALIZED CTE and an inline-join alternative in the same batch; project only needed keys and attributes.",
     "safety": "Preserve join multiplicity, correlation keys, and all nation/date predicates. Do not force materialization of a large fact relation."},
    {"id": "nested_aggregate_collapse", "condition": "An outer aggregate combines partitioned inner aggregates",
     "action": "Consider a single aggregate over the base rows, retaining the exact additive/min/max/COUNT and empty-input behavior.",
     "safety": "Retain SUM result width, NULL behavior, COALESCE placement, and DISTINCT boundaries. COUNT(*) needs audited non-nullability."},
]


STRATEGY_FAMILIES = ("selective_dimension_materialization", "inline_join")


def _select_tree(sql):
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres", error_level=ErrorLevel.RAISE) if s]
        return statements[0] if len(statements) == 1 and isinstance(statements[0], exp.Select) else None
    except (ValueError, TypeError, AttributeError, sqlglot.errors.ParseError):
        return None


def _has_literal_filter(select):
    # Shape evidence only: no claim about selectivity or dimension cardinality.
    conditions = [select.args.get("where")]
    conditions.extend(join.args.get("on") for join in select.args.get("joins", []))
    return any(condition is not None and condition.find(exp.Column) is not None
               and condition.find(exp.Literal) is not None for condition in conditions)


def _needs_join_strategy_pair(sql):
    tree = _select_tree(sql)
    if tree is None:
        return False
    selects = list(tree.find_all(exp.Select))
    return (any(len(select.args.get("joins", [])) >= 2 for select in selects)
            and any(_has_literal_filter(select) for select in selects))


def _join_strategy_family(sql):
    """Classify actual SQL structure, never a model's claimed action label."""
    tree = _select_tree(sql)
    if tree is None or not _needs_join_strategy_pair(sql):
        return None
    materialized = [cte for cte in tree.find_all(exp.CTE) if cte.args.get("materialized") is True]
    for cte in materialized:
        if not isinstance(cte.this, exp.Select) or not _has_literal_filter(cte.this):
            continue
        name = _pg_identifier_name(cte.args["alias"].this)
        if any(not table.db and not table.catalog and _pg_identifier_name(table) == name
               and table.find_ancestor(exp.CTE) is not cte for table in tree.find_all(exp.Table)):
            return STRATEGY_FAMILIES[0]
    return STRATEGY_FAMILIES[1] if not materialized else None


def canonicalize_not_null_counts(sql: str, schema_sql: str):
    """A deliberately narrow candidate normalization, never a gate verdict.

    Only one direct, unqualified base-table SELECT is handled. Rejecting joins,
    CTEs, subqueries and windows avoids confusing source NOT NULL constraints
    with null-extended/derived columns. PostgreSQL catalog compatibility remains
    the caller's responsibility before the resulting proposal is accepted.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres", error_level=ErrorLevel.RAISE) if s]
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return None
        tree = statements[0]
        if (tree.args.get("with") or tree.args.get("joins") or tree.args.get("into")
                or tree.find(exp.Subquery) or tree.find(exp.Window)
                or len(list(tree.find_all(exp.Select))) != 1):
            return None
        source = tree.args.get("from")
        table = source.this if source else None
        if not isinstance(table, exp.Table) or table.db or table.catalog or table.args.get("sample"):
            return None
        if table.args.get("pivots") or table.args.get("version"):
            return None
        table_name = _pg_identifier_name(table)
        alias = table.args.get("alias")
        qualifier = _pg_identifier_name(alias.this) if alias else table_name
        if alias and alias.args.get("columns"):
            return None
        schemas = [s for s in sqlglot.parse(schema_sql, read="postgres", error_level=ErrorLevel.RAISE) if s]
        definitions = [s for s in schemas if isinstance(s, exp.Create) and s.kind == "TABLE"
                       and isinstance(s.this, exp.Schema) and not s.this.this.db and not s.this.this.catalog
                       and _pg_identifier_name(s.this.this) == table_name]
        if len(definitions) != 1:
            return None
        definition = definitions[0].this
        nonnull = set()
        for column in definition.expressions:
            if isinstance(column, exp.ColumnDef) and (column.find(exp.NotNullColumnConstraint)
                                                      or column.find(exp.PrimaryKeyColumnConstraint)):
                nonnull.add(_pg_identifier_name(column))
            elif isinstance(column, exp.PrimaryKey):
                for key in column.expressions:
                    key = key.this if isinstance(key, exp.Ordered) else key
                    nonnull.add(_pg_identifier_name(key))
        changed = []
        for count in tree.find_all(exp.Count):
            column = count.this
            if not isinstance(column, exp.Column) or column.db or column.catalog:
                continue
            if column.table and _pg_identifier_name(column.args["table"]) != qualifier:
                continue
            if _pg_identifier_name(column) not in nonnull or count.expressions:
                continue
            changed.append(column.sql(dialect="postgres"))
            count.set("this", exp.Star())
        if not changed:
            return None
        return {"sql": tree.sql(dialect="postgres"), "rule": "direct_not_null_count_star_v2",
                "table": table_name, "columns": changed,
                "source_sql_sha256": exact_sha256(sql), "schema_sha256": exact_sha256(schema_sql),
                "requires_catalog_compatibility": True, "verified": False,
                "evidence_scope": "Source-DDL guarded proposal; independent root gate still required"}
    except (ValueError, TypeError, AttributeError, sqlglot.errors.ParseError):
        return None


class BatchedOperatorsV2(BatchedOperators):
    """Add reviewed generic advice and narrow normalization to MCTS only.

    Baseline arms retain the inherited implementation. The same batches and
    candidate limits are used, with no additional model invocations.
    """
    def _call(self, payload):
        if payload.get("operator") == "mcts":
            payload = {**payload, "reviewed_action_patterns": deepcopy(ACTION_PATTERNS),
                       "feedback_scope": "transductive_stage_a_followup",
                       "pattern_evidence": "Development hypotheses informed by completed Stage A; no latency labels supplied"}
            if payload.get("task") == "mcts_batched_generation":
                eligible = getattr(self, "_join_strategy_parents", set())
                payload["parents"] = [{**parent, **({"required_strategy_families": list(STRATEGY_FAMILIES)}
                    if str(parent["parent_id"]) in eligible else {})} for parent in payload["parents"]]
                if eligible:
                    payload["instruction"] += (
                        " For each parent with required_strategy_families, use two of its existing three proposal slots"
                        " for BOTH families: (1) a selective dimension or dimension-chain WITH ... AS MATERIALIZED"
                        " CTE, and (2) an inline-join alternative. Preserve all join multiplicities, correlation keys,"
                        " nation/date predicates, output semantics and selective filters; never materialize a large fact"
                        " relation. The third slot remains available for another macro or Calcite sequence."
                        " Existing EXPLAIN JSON and bounded canary ranking choose between these unverified hypotheses."
                    )
        return super()._call(payload)

    def _deduplicate(self, rows, excluded):
        if not getattr(self, "_join_strategy_parents", set()):
            return super()._deduplicate(rows, excluded)
        # Run before the inherited three-slot truncation. Its class-A/class-B
        # reservation then yields [one symbolic, materialized, inline] when both
        # operator classes exist, rather than letting extra symbolic or macro
        # proposals crowd out one family. The controller still ranks all three.
        seen, unique = set(), []
        for row in rows:
            key = str(row["parent_id"]), exact_sha256(row["sql"])
            if row["sql"] not in excluded and key not in seen:
                seen.add(key)
                unique.append(row)
        preferred = []
        for parent_id in sorted(self._join_strategy_parents):
            for family in STRATEGY_FAMILIES:
                row = next((row for row in unique if str(row["parent_id"]) == parent_id
                            and _join_strategy_family(row["sql"]) == family), None)
                if row is not None:
                    preferred.append(row)
        return preferred + [row for row in unique if all(row is not chosen for chosen in preferred)]

    def _generation_batch(self, parents, schema_sql, *, operator, evidence, depth,
                          limit_per_parent, include_calcite):
        self._join_strategy_parents = ({str(parent.get("id", i)) for i, parent in enumerate(parents)
            if _needs_join_strategy_pair(parent["sql"])} if operator == "mcts" and limit_per_parent >= 3 else set())
        strategy_parents = self._join_strategy_parents
        try:
            rows = super()._generation_batch(parents, schema_sql, operator=operator, evidence=evidence,
                                            depth=depth, limit_per_parent=limit_per_parent,
                                            include_calcite=include_calcite)
        finally:
            self._join_strategy_parents = set()
        if operator != "mcts":
            return rows
        for parent_id in sorted(strategy_parents):
            present = {_join_strategy_family(row["sql"]) for row in rows if str(row["parent_id"]) == parent_id}
            missing = [family for family in STRATEGY_FAMILIES if family not in present]
            self.last_diagnostics.append({"operator": "mcts", "parent_id": parent_id,
                "status": "join_strategy_pair_incomplete" if missing else "join_strategy_pair_preserved",
                "missing_strategy_families": missing, "additional_model_calls": 0,
                "verified": False, "scope": "proposal_coverage_only"})
        proposals = []
        for original in rows:
            receipt = canonicalize_not_null_counts(original["sql"], schema_sql)
            if receipt is None:
                family = _join_strategy_family(original["sql"]) if str(original["parent_id"]) in strategy_parents else None
                if family:
                    original["provenance"]["join_strategy_family"] = family
                proposals.append(original)
                continue
            row = deepcopy(original)
            row["sql"] = receipt["sql"]
            row["action_class"] = "B"
            row["action"] = "source_guarded_count_canonicalization"
            row["provenance"].update(verified=False, proposal_before_normalization=original["sql"],
                                     count_normalization=receipt,
                                     feedback_scope="transductive_stage_a_followup")
            proposals.append(row)
        # Parent attribution must survive byte-identical proposals in a beam.
        result, seen = [], set()
        parent_sql = {str(p.get("id", i)): p["sql"] for i, p in enumerate(parents)}
        for row in proposals:
            key = str(row["parent_id"]), exact_sha256(row["sql"])
            if key not in seen and row["sql"] != parent_sql.get(str(row["parent_id"])):
                seen.add(key)
                result.append(row)
        return result

    def metadata(self):
        return {**super().metadata(), "variant_version": 2,
                "feedback_scope": "transductive_stage_a_followup",
                "mcts_action_patterns": [p["id"] for p in ACTION_PATTERNS],
                "mcts_count_normalization": "source-DDL guarded; gate still required",
                "mcts_join_strategy_coverage": "request and preserve MATERIALIZED dimension and inline-join proposals within three slots; root gate and ranking unchanged",
                "additional_model_calls": 0, "standalone_baselines_modified": False}
