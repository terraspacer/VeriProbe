"""PostgreSQL plan features and operator-level bottleneck attribution.

Costs are planner units, not milliseconds.  Exclusive cost attribution is a
heuristic because PostgreSQL does not expose a per-operator exclusive estimate.
Nested-loop inner and correlated SubPlan rescans are accounted for explicitly.
Observed node loop counts take precedence for effective cost, including plans
collected with TIMING OFF.  Estimated loops remain separate.  In particular,
plain EXPLAIN cannot resolve Materialize reuse or Memoize cache misses.
Descendant repetition estimates are scenarios, not observed source scans.
"""
from __future__ import annotations

import math
import re
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else default
    except (TypeError, ValueError):
        return default


def plan_root(document: Any) -> dict:
    """Accept psycopg EXPLAIN documents, their outer list, or a bare node."""
    while isinstance(document, list):
        document = document[0] if document else {}
    if not isinstance(document, dict):
        return {}
    return document.get("Plan", document) if isinstance(document.get("Plan", document), dict) else {}


def _subplan_kind(parent: dict, child: dict) -> str | None:
    if child.get("Parent Relationship") == "InitPlan":
        return "initplan"
    name = str(child.get("Subplan Name", ""))
    if child.get("Parent Relationship") != "SubPlan" and not name.startswith("SubPlan"):
        return None
    # PostgreSQL prints hashed IN/NOT IN subplans in the parent's expression.
    # They build a hash table once and must not inherit a per-output-row rescan
    # estimate merely because their Parent Relationship is also "SubPlan".
    number = re.fullmatch(r"SubPlan\s*(\d+)", name, flags=re.IGNORECASE)
    reference = rf"\bhashed\s+SubPlan\s*{re.escape(number[1])}\b" if number else None
    if reference and any(re.search(reference, str(value), flags=re.IGNORECASE)
                         for key, value in parent.items() if key != "Plans"):
        return "hashed_subplan"
    return "correlated_subplan"


def _child_repetitions(parent: dict, child: dict, position: int) -> float:
    relationship = child.get("Parent Relationship", "")
    subplan_kind = _subplan_kind(parent, child)
    if subplan_kind in ("initplan", "hashed_subplan"):
        return 1.0
    if subplan_kind == "correlated_subplan":
        return max(1.0, _number(parent.get("Plan Rows"), 1))
    if parent.get("Node Type") == "Nested Loop" and (relationship == "Inner" or position == 1):
        outer = next((p for p in parent.get("Plans", []) if p.get("Parent Relationship") == "Outer"), None)
        if outer is None:
            outer = parent.get("Plans", [{}])[0]
        return max(1.0, _number(outer.get("Plan Rows"), 1))
    return 1.0


def operators(document: Any) -> list[dict]:
    """Return compatible planner fields plus explicitly sourced loop weights.

    ``estimated_loops`` retains the structural repetition heuristic.
    ``effective_loops`` prefers this node's valid Actual Loops, including zero;
    multiplying planner exclusive cost by it still produces planner units, not
    measured work or latency.  ``rescan_attribution_uncertain`` identifies an
    unobserved node whose estimate crosses a Materialize or Memoize cache.
    Ancestor paths describe that context even when this node has observed loops.
    """
    root = plan_root(document)
    if not root or not root.get("Node Type"):
        return []
    result: list[dict] = []

    def visit(node: dict, path: list[int], estimated_loops: float,
              subplan_kind: str | None = None,
              materialize_ancestors: tuple[tuple[int, ...], ...] = (),
              memoize_ancestors: tuple[tuple[int, ...], ...] = ()) -> None:
        children = [p for p in node.get("Plans", []) if isinstance(p, dict)]
        total = _number(node.get("Total Cost"))
        child_cost = sum(_number(child.get("Total Cost")) * _child_repetitions(node, child, i)
                         for i, child in enumerate(children))
        exclusive = max(0.0, total - child_cost)
        actual_loops = _number(node.get("Actual Loops"), math.nan)
        observed_loops = math.isfinite(actual_loops)
        effective_loops = actual_loops if observed_loops else estimated_loops
        # Actual Loops is already the total number of executions of this node,
        # not a per-parent multiplier.  Zero also matters for unexecuted nodes.
        uncertain_rescans = bool(materialize_ancestors or memoize_ancestors) and not observed_loops
        actual_total = _number(node.get("Actual Total Time")) * effective_loops
        actual_children = sum(_number(child.get("Actual Total Time")) * _number(child.get("Actual Loops"), 1)
                              for child in children)
        actual_exclusive = max(0.0, actual_total - actual_children)
        kind = subplan_kind if subplan_kind in ("hashed_subplan", "correlated_subplan") else {
            "Nested Loop": "nested_loop", "Seq Scan": "sequential_scan",
            "Parallel Seq Scan": "sequential_scan",
        }.get(node.get("Node Type"), "operator")
        row = {
            "node_type": node.get("Node Type"), "path": path,
            "kind": kind, "relation": node.get("Relation Name"),
            "alias": node.get("Alias"), "subplan_name": node.get("Subplan Name"),
            "total_cost": total, "exclusive_cost": exclusive,
            "estimated_loops": estimated_loops,
            "effective_loops": effective_loops,
            "loops_source": "observed" if observed_loops else "estimated",
            "rescan_attribution_uncertain": uncertain_rescans,
            "materialize_ancestor_paths": [list(ancestor) for ancestor in materialize_ancestors],
            "memoize_ancestor_paths": [list(ancestor) for ancestor in memoize_ancestors],
            "effective_exclusive_cost": exclusive * effective_loops,
            "plan_rows": _number(node.get("Plan Rows")),
            "plan_width": _number(node.get("Plan Width")),
            "actual_exclusive_ms": actual_exclusive if "Actual Total Time" in node else None,
            "actual_loops": actual_loops if observed_loops else None,
            "conditions": {key: node[key] for key in (
                "Filter", "Join Filter", "Hash Cond", "Merge Cond", "Index Cond", "Recheck Cond") if key in node},
        }
        result.append(row)
        for i, child in enumerate(children):
            repetitions = _child_repetitions(node, child, i)
            child_kind = _subplan_kind(node, child)
            loops = 1.0 if child_kind in ("initplan", "hashed_subplan") else estimated_loops * repetitions
            ancestors = materialize_ancestors
            memoized = memoize_ancestors
            if child_kind in ("initplan", "hashed_subplan"):
                ancestors = ()  # These estimates do not inherit parent rescans.
                memoized = ()
            elif node.get("Node Type") == "Materialize":
                # EXPLAIN JSON omits nestParams, chgParam and executor rewind
                # flags.  A blanket reset to one would undercount parameterized
                # children and correlated work.  Preserve the estimate and flag
                # the caching ambiguity until this child's loops are observed.
                # PostgreSQL 14: ExecInitNestLoop / ExecReScanMaterial.
                # https://github.com/postgres/postgres/blob/REL_14_STABLE/src/backend/executor/nodeMaterial.c
                # https://github.com/postgres/postgres/blob/REL_14_STABLE/src/backend/executor/nodeNestloop.c
                ancestors += (tuple(path),)
            elif node.get("Node Type") == "Memoize":
                # Hits can skip the parameterized child; plain EXPLAIN does not
                # tell us the realized misses, evictions, or bypass behavior.
                # Preserve estimated loops and prefer this child's Actual Loops.
                # PostgreSQL 14: src/backend/executor/nodeMemoize.c.
                # https://github.com/postgres/postgres/blob/REL_14_STABLE/src/backend/executor/nodeMemoize.c
                memoized += (tuple(path),)
            visit(child, path + [i], loops, child_kind, ancestors, memoized)

    visit(root, [], 1.0)
    return result


def bottleneck(plan: Any) -> dict:
    """Rank exclusive work across all nodes rather than the inclusive root cost."""
    rows = operators(plan)
    if not rows:
        return {"available": False, "reason": "missing PostgreSQL plan operators", "ranked": []}
    # Prefer actual exclusive time only when the document contains measured times.
    measured = any(row["actual_exclusive_ms"] is not None for row in rows)
    field = "actual_exclusive_ms" if measured else "effective_exclusive_cost"
    ranked = sorted(rows, key=lambda row: (row.get(field) or 0.0,
                    row["kind"] != "operator", len(row["path"])), reverse=True)
    selected = dict(ranked[0])
    uncertain = sum(row["rescan_attribution_uncertain"] for row in rows)
    selected.update({"available": True, "ranking_metric": field,
                     "attribution": "exclusive work heuristic; observed node loops when available, otherwise estimated repetitions",
                     "rescan_uncertain_operators": uncertain,
                     "rescan_uncertainty_note": (
                         "Materialize or Memoize descendants may replay cached rows without rerunning their sources. "
                         "EXPLAIN leaves cache reuse and Memoize misses unresolved; their unobserved loop estimates "
                         "are heuristic repetition scenarios, not observed source scans or guaranteed bounds."
                         if uncertain else None),
                     "estimated_cost_is_latency": False, "ranked": ranked})
    return selected


def plan_features(plan: Any) -> dict[str, float]:
    """Numeric features; effective work/counts prefer observed node loops.

    The extra estimated_* fields retain the estimated-only scenario for audit;
    rescan_uncertain_operators counts unresolved Materialize/Memoize attribution.
    """
    root, rows = plan_root(plan), operators(plan)
    return {
        "total_cost": _number(root.get("Total Cost")),
        "rows": _number(root.get("Plan Rows")),
        "width": _number(root.get("Plan Width")),
        "operators": float(len(rows)),
        "nested_loops": float(sum(row["kind"] == "nested_loop" for row in rows)),
        "seq_scans": float(sum(row["kind"] == "sequential_scan" for row in rows)),
        "subplans": float(sum(row["kind"] in ("correlated_subplan", "hashed_subplan") for row in rows)),
        "max_rescans": max((row["effective_loops"] for row in rows), default=0.0),
        "max_estimated_rescans": max((row["estimated_loops"] for row in rows), default=0.0),
        "observed_loop_operators": float(sum(row["loops_source"] == "observed" for row in rows)),
        "rescan_uncertain_operators": float(sum(row["rescan_attribution_uncertain"] for row in rows)),
        "exclusive_work": sum(row["effective_exclusive_cost"] for row in rows),
        "estimated_exclusive_work": sum(row["exclusive_cost"] * row["estimated_loops"] for row in rows),
    }
