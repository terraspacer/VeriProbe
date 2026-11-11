#!/usr/bin/env python3
"""Session-authored DSB Class B templates; never consumes baseline rewrites.

This file was authored by the live model after inspecting the source queries,
QUITE rewrite instructions/knowledge, Rbot rule summaries, and PostgreSQL plans.
It preserves original SQL fragments instead of transpiling source identifiers or
literals. EXPLAIN is syntax/plan evidence only, never equivalence evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psycopg
import sqlglot
from sqlglot import exp
from mcts_rewrite.db import Postgres, parse_read_query
from mcts_rewrite.plans import bottleneck, operators

MODEL = {"model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
         "generation": "session_subagent"}
SOURCES = [
    "vendor/QUITE/src/Query_Rewriter/agent_definition.py",
    "vendor/QUITE/src/Rewrite_Middleware/Structured_Knowledge_Base/storage/knowledge_base.json",
    "vendor/LLM4Rewrite/knowledge-base/rule_cluster_summaries_structured.jsonl",
    "vendor/LLM4Rewrite/dsb/create_tables.sql",
]


def tokens(sql):
    return sqlglot.Dialect.get_or_raise("postgres").tokenize(sql)


def pairs(sql):
    stack, found = [], {}
    for token in tokens(sql):
        if token.token_type.name == "L_PAREN":
            stack.append(token.start)
        elif token.token_type.name == "R_PAREN":
            found[stack.pop()] = token.end
    return found


def top_positions(sql):
    depth, result = 0, {}
    for token in tokens(sql):
        kind = token.token_type.name
        if kind == "L_PAREN":
            depth += 1
        elif kind == "R_PAREN":
            depth -= 1
        elif depth == 0:
            result.setdefault(kind, []).append((token.start, token.end + 1))
    return result


def ctes(sql):
    query = sqlglot.parse_one(sql, read="postgres")
    with_ = query.args.get("with")
    if with_ is None:
        return []
    parens, out = pairs(sql), []
    for node in with_.expressions:
        identifier = node.args["alias"].this
        end = identifier.meta["end"] + 1
        match = re.match(r"\s*AS\s*\(", sql[end:], re.I)
        if match is None:
            continue
        start = end + match.end() - 1
        out.append((node.alias, start, parens[start], node))
    return out


def add_cte(sql, name, body, materialized="MATERIALIZED"):
    definition = f"{name} AS {materialized} (\n{body}\n)"
    existing = ctes(sql)
    if existing:
        end = existing[-1][2] + 1
        return sql[:end] + ",\n" + definition + sql[end:]
    return "WITH " + definition + "\n" + sql


def add_outer_relation(sql, relation, predicate=None):
    p = top_positions(sql)
    where_start, where_end = p["WHERE"][0]
    addition = ", " + relation + "\n"
    sql = sql[:where_start] + addition + sql[where_start:]
    if predicate:
        p = top_positions(sql)
        _, where_end = p["WHERE"][0]
        sql = sql[:where_end] + " " + predicate + " AND " + sql[where_end:]
    return sql


def raw_key(sql):
    return tuple((t.token_type.name, sql[t.start:t.end + 1]
                  if sql[t.start:t.end + 1].startswith(("'", '"'))
                  else sql[t.start:t.end + 1].lower())
                 for t in tokens(sql) if t.token_type.name != "SEMICOLON")


def specs(sql, unique_constraints=()):
    """High-value, source-fragment-preserving structural hypotheses."""
    result = []
    # Grouped CTE thresholds: inner equality excludes NULL groups naturally.
    match = re.search(r"ctr1\.ctr_total_return\s*>\s*(\()", sql, re.I)
    if match:
        start = match.start(1)
        end = pairs(sql)[start]
        inner = sql[start + 1:end]
        key = re.search(r"ctr1\.(ctr_store_sk|ctr_state)\s*=\s*ctr2\.\1", inner, re.I)
        avg = re.match(r"\s*select\s+(.*?)\s+from\s+customer_total_return\s+ctr2\s+where\s+", inner, re.I | re.S)
        if key and avg:
            body = (f"SELECT {key[1]} AS macro_group_key, {avg[1]} AS macro_threshold\n"
                    f"FROM customer_total_return GROUP BY {key[1]}")
            changed = sql[:start] + "macro_return_thresholds.macro_threshold" + sql[end + 1:]
            changed = add_cte(changed, "macro_return_thresholds", body)
            changed = add_outer_relation(changed, "macro_return_thresholds",
                                        f"ctr1.{key[1]} = macro_return_thresholds.macro_group_key")
            result.append((changed, "decorrelate_return_average_by_group",
                           "Compute the correlated return AVG once per equality key; grouped inner join prevents row multiplication and rejects NULL keys as the original equality does."))

    # Discount thresholds: preserve every inner date/price restriction.
    match = re.search(r"(?:cs|ws)_ext_discount_amt\s*>\s*(\()", sql, re.I)
    if match:
        start, end = match.start(1), pairs(sql)[match.start(1)]
        inner = sql[start + 1:end]
        correlation = re.search(r"\bWHERE\s+((?:cs|ws)_item_sk)\s*=\s*i_item_sk\s+AND\s+", inner, re.I)
        aggregate = re.match(r"\s*SELECT\s+(.*?)\s+(FROM\s+.*)", inner, re.I | re.S)
        if correlation and aggregate:
            key = correlation[1]
            remainder = inner[aggregate.start(2):]
            remainder, count = re.subn(r"\bWHERE\s+" + key + r"\s*=\s*i_item_sk\s+AND\s+", "WHERE ", remainder, count=1, flags=re.I)
            if count == 1:
                body = (f"SELECT {key} AS macro_item_sk, {aggregate[1]} AS macro_threshold\n"
                        + remainder.rstrip() + f"\nGROUP BY {key}")
                changed = sql[:start] + "macro_discount_thresholds.macro_threshold" + sql[end + 1:]
                changed = add_cte(changed, "macro_discount_thresholds", body)
                changed = add_outer_relation(changed, "macro_discount_thresholds",
                                            "macro_discount_thresholds.macro_item_sk = i_item_sk")
                result.append((changed, "decorrelate_discount_average_by_item",
                               "Aggregate each item's restricted discount population once; the unique grouped key preserves outer duplicates and an inner equality join preserves missing/NULL threshold rejection."))

    # Reuse q65's literally identical store/item revenue aggregate.
    matches = []
    for start, end in pairs(sql).items():
        body = sql[start + 1:end]
        alias = re.match(r"\s*(sa|sc)\b", sql[end + 1:], re.I)
        if alias and re.match(r"\s*select\s+ss_store_sk\s*,\s*ss_item_sk\s*,", body, re.I):
            matches.append((start, end, body, alias[1]))
    if len(matches) == 2 and raw_key(matches[0][2]) == raw_key(matches[1][2]):
        changed = sql
        for start, end, _, _ in sorted(matches, reverse=True):
            changed = changed[:start] + "macro_store_item_revenue" + changed[end + 1:]
        changed = add_cte(changed, "macro_store_item_revenue", matches[0][2])
        result.append((changed, "share_store_item_revenue_aggregate",
                       "Materialize the identical store/item revenue GROUP BY once for both the per-store mean and detailed revenue inputs; no aggregation level or outer multiplicity changes."))

    # The two OR-ed EXISTS are eligibility tests, so duplicates and NULLs from
    # the projected key do not add any accepted outer row under this WHERE.
    matches = []
    parens = pairs(sql)
    for match in re.finditer(r"\bexists\s*(\()", sql, re.I):
        start = match.start(1)
        end = parens[start]
        inner = sql[start + 1:end]
        corr = re.search(r"c\.c_customer_sk\s*=\s*(ws_bill_customer_sk|cs_ship_customer_sk)\s+and\s+", inner, re.I)
        if corr:
            projected = re.sub(r"\bselect\s+\*", "select " + corr[1], inner, count=1, flags=re.I)
            projected = re.sub(r"c\.c_customer_sk\s*=\s*" + corr[1] + r"\s+and\s+", "", projected, count=1, flags=re.I)
            matches.append((match.start(), end + 1, projected))
    if len(matches) == 2 and re.fullmatch(r"\s*or\s*", sql[matches[0][1]:matches[1][0]], re.I):
        start, end = matches[0][0], matches[1][1]
        replacement = "c.c_customer_sk IN (" + matches[0][2] + "\nUNION\n" + matches[1][2] + ")"
        changed = sql[:start] + replacement + sql[end:]
        result.append((changed, "decorrelate_or_exists_to_union_eligibility",
                       "Replace the disjunctive correlated channel eligibility tests by a set of customer keys, enabling a single semijoin; WHERE membership retains accepted rows even with duplicate or NULL inner keys."))

    # DISTINCT is redundant under duplicate-eliminating INTERSECT/EXCEPT.
    parsed = sqlglot.parse_one(sql, read="postgres")
    distincts = list(re.finditer(r"\bselect\s+(distinct)\s+c_last_name\s*,\s*c_first_name\s*,\s*d_date\b", sql, re.I))
    setops = list(parsed.find_all(exp.SetOperation))
    if distincts and setops and all(not node.args.get("distinct") is False for node in setops):
        if all(isinstance(node, (exp.Intersect, exp.Except)) for node in setops):
            changed = sql
            for match in reversed(distincts):
                changed = changed[:match.start(1)] + changed[match.end(1):]
            result.append((changed, "remove_distinct_under_set_operations",
                           "INTERSECT/EXCEPT already implement set semantics including NULL equality; removing duplicate DISTINCT work from their direct operands leaves the final bag unchanged."))

    # Push outer year/group restrictions inside shared aggregate CTEs.
    cte_list = ctes(sql)
    year = re.search(r"\bss1\.d_year\s*=\s*(\d+)\b", sql, re.I)
    consumer_years = re.findall(r"\b(?:ss|ws)[123]\.d_year\s*=\s*(\d+)\b", sql, re.I)
    consumer_quarters = re.findall(r"\b(?:ss|ws)[123]\.d_qoy\s*=\s*([123])\b", sql, re.I)
    if (year and len(consumer_years) == 6 and set(consumer_years) == {year[1]}
            and sorted(consumer_quarters) == ["1", "1", "2", "2", "3", "3"]
            and {name.lower() for name, *_ in cte_list} == {"ss", "ws"}):
        changed = sql
        for _, start, end, _ in reversed(cte_list):
            body = changed[start + 1:end]
            group = re.search(r"\bgroup\s+by\b", body, re.I)
            if group:
                body = body[:group.start()] + f" AND d_year = {year[1]} AND d_qoy IN (1,2,3)\n" + body[group.start():]
                changed = changed[:start + 1] + body + changed[end:]
        result.append((changed, "push_shared_quarter_year_filters",
                       "All six consumers demand the same year and quarters 1/2/3; filter these GROUP BY keys in both shared sales CTEs before scanning and aggregation."))
    if cte_list and cte_list[0][0].lower() == "inv":
        first = re.search(r"inv1\.d_moy\s*=\s*([0-9]+(?:\s*\+\s*[0-9]+)?)", sql, re.I)
        second = re.search(r"inv2\.d_moy\s*=\s*([0-9]+(?:\s*\+\s*[0-9]+)?)", sql, re.I)
        if first and second:
            _, start, end, _ = cte_list[0]
            body = sql[start + 1:end]
            group = re.search(r"\bgroup\s+by\b", body, re.I)
            if group:
                body = body[:group.start()] + f" AND d_moy IN ({first[1]}, {second[1]})\n" + body[group.start():]
                result.append((sql[:start + 1] + body + sql[end:], "push_inventory_month_filters",
                               "Both self-join consumers require just two grouping months; restrict d_moy before computing the per-month mean and sample standard deviation."))
    for name, start, end, _ in cte_list:
        if name.lower() != "cross_sales":
            continue
        first = re.search(r"cs1\.syear\s*=\s*([0-9]+(?:\s*\+\s*[0-9]+)?)", sql, re.I)
        second = re.search(r"cs2\.syear\s*=\s*([0-9]+(?:\s*\+\s*[0-9]+)?)", sql, re.I)
        if first and second:
            body = sql[start + 1:end]
            group = re.search(r"\bgroup\s+by\b", body, re.I)
            if group:
                body = body[:group.start()] + f" AND d1.d_year IN ({first[1]}, {second[1]})\n" + body[group.start():]
                result.append((sql[:start + 1] + body + sql[end:], "push_cross_sales_year_filters",
                               "The self join consumes only two sale years; push that restriction on the d1.d_year GROUP BY key into the shared cross_sales aggregate."))
    if len(cte_list) == 1 and cte_list[0][0].lower() == "wss":
        ranges = list(re.finditer(r"\bd_month_seq\s+between\s+[0-9]+(?:\s*\+\s*[0-9]+)?\s+and\s+[0-9]+(?:\s*\+\s*[0-9]+)?", sql, re.I))
        if len(ranges) == 2:
            _, start, end, _ = cte_list[0]
            body = sql[start + 1:end]
            group = re.search(r"\bgroup\s+by\b", body, re.I)
            if group:
                predicate = (" AND d_week_seq IN (SELECT d_week_seq FROM date_dim WHERE ("
                             + ranges[0][0] + ") OR (" + ranges[1][0] + "))\n")
                body = body[:group.start()] + predicate + body[group.start():]
                result.append((sql[:start + 1] + body + sql[end:], "push_shared_week_eligibility",
                               "Both consumers accept weeks from only two month windows. Push their union of week keys into wss while retaining full-week aggregation, including dates outside a partial boundary month."))
    if any(name.lower() == "all_sales" for name, *_ in cte_list):
        curr = re.search(r"curr_yr\.d_year\s*=\s*([0-9]+)", sql, re.I)
        prev = re.search(r"prev_yr\.d_year\s*=\s*([0-9]+\s*-\s*[0-9]+)", sql, re.I)
        if curr and prev:
            changed, count = re.subn(r"\bWHERE\s+(i_category\s*=)",
                lambda m: f"WHERE d_year IN ({curr[1]}, {prev[1]}) AND " + m[1], sql, flags=re.I)
            if count == 3:
                result.append((changed, "push_all_sales_year_filters",
                               "Restrict all three UNION branches to the only two years consumed after grouping; d_year is part of every UNION projection and GROUP BY key."))

    # A catalog-enforced unique right side can never multiply a LEFT JOIN.
    # Inspect outside references before removing the unused nullable columns.
    for relation, columns, prefix in [("catalog_returns", ("cr_item_sk", "cr_order_number"), "cr_"),
                                      ("promotion", ("p_promo_sk",), "p_")]:
        if (relation, columns) not in unique_constraints:
            continue
        join = re.search(r"\bleft\s+(?:outer\s+)?join\s+" + relation + r"\s+on\s*(\()", sql, re.I)
        if not join:
            continue
        opening = join.start(1)
        closing = pairs(sql)[opening]
        on = sql[opening + 1:closing]
        if not all(re.search(r"\b" + col + r"\s*=|=\s*" + col + r"\b", on, re.I) for col in columns):
            continue
        changed = sql[:join.start()] + sql[closing + 1:]
        if any(t.text.lower().startswith(prefix) for t in tokens(changed)
               if t.token_type.name in {"VAR", "IDENTIFIER"}):
            continue
        result.append((changed, "remove_unused_unique_left_join_" + relation,
                       f"No projected or filtering expression uses {relation}; the local PostgreSQL catalog enforces unique {columns}, so its LEFT JOIN neither removes nor duplicates left rows."))

    # MATERIALIZED/NOT MATERIALIZED are deliberate physical alternatives.
    repeated = []
    for name, start, end, node in cte_list:
        refs = sum(table.name.lower() == name.lower() for table in parsed.find_all(exp.Table))
        if refs > 1:
            repeated.append((name, start))
    if repeated:
        changed = sql
        for name, start in reversed(repeated):
            changed = changed[:start] + "NOT MATERIALIZED " + changed[start:]
        result.append((changed, "inline_repeated_ctes_for_filter_pushdown",
                       "Allow each reference to push its selective grouping-key predicates into the deterministic CTE. This trades shared evaluation for reduced input; EXPLAIN/timing must decide."))
    elif cte_list:
        changed = sql
        for name, start, _, _ in reversed(cte_list):
            changed = changed[:start] + "MATERIALIZED " + changed[start:]
        result.append((changed, "materialize_cte_boundaries",
                       "Preserve deterministic CTE boundaries as a physical alternative, reducing the expanded join search space and allowing each intermediate result to be consumed separately."))
    return result


def summary(plan):
    bott = bottleneck(plan)
    return {"root_node": plan["Plan"].get("Node Type"),
            "total_cost": plan["Plan"].get("Total Cost"),
            "plan_rows": plan["Plan"].get("Plan Rows"),
            "bottleneck": {k: v for k, v in bott.items() if k != "ranked"},
            "ranked_operators": bott.get("ranked", [])[:8],
            "correlated_subplans": [row for row in operators(plan) if row["kind"] == "correlated_subplan"],
            "operator_counts": dict(Counter(row["node_type"] for row in operators(plan)))}


def shape(plan):
    def recurse(node):
        return (node.get("Node Type"), node.get("Join Type"), node.get("Relation Name"),
                node.get("Alias"), tuple(recurse(child) for child in node.get("Plans", [])))
    return recurse(plan["Plan"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="user=mcts_read dbname=dsb_dev host=/workspace/VeriProbe/.runtime/socket port=55432")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "benchmarks/manifest.jsonl").read_text().splitlines()
            if json.loads(line)["benchmark"] == "dsb"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sql"], []).append(row)
    db = Postgres(args.dsn, timeout_ms=args.timeout_ms)
    constraint_rows = db.connection.execute("""
        SELECT t.relname, c.conname, array_agg(a.attname ORDER BY u.ordinality)
        FROM pg_constraint c JOIN pg_class t ON t.oid=c.conrelid
        JOIN pg_namespace n ON n.oid=t.relnamespace
        CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY u(attnum, ordinality)
        JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=u.attnum
        WHERE n.nspname='public' AND c.contype IN ('p','u')
        GROUP BY t.relname,c.conname
    """).fetchall()
    unique_constraints = {(table, tuple(columns)) for table, _, columns in constraint_rows}
    notes = {"provenance": MODEL, "started": datetime.now(timezone.utc).isoformat(),
             "source_memberships": len(rows), "unique_originals": len(grouped),
             "knowledge_sources": [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for path in SOURCES],
             "baseline_rewrites_read": False, "analyze_executed": False,
             "equivalence_claimed": False, "queries": [],
             "audited_unique_constraints": [{"table": table, "name": name, "columns": columns}
                                             for table, name, columns in constraint_rows],
             "missed_opportunities": [
                 "Wide DSB joins may need selective dimension/fact materialization with catalog statistics; avoid dropping intentional multiplicities such as the store relation in q102.",
                 "q72 has an unused catalog_returns LEFT JOIN whose removal requires audited uniqueness of (cr_item_sk,cr_order_number); PostgreSQL may already eliminate it.",
                 "q75 return reason filters null-reject its LEFT JOINs, permitting inner joins; PostgreSQL may already simplify these.",
                 "q59's repeated weekly aggregate can be narrowed to the union of its two date-month windows, but preserving the duplicate date_dim joins requires care.",
                 "q13/q85 mutually exclusive OR branches may benefit from UNION ALL or explicit dimension eligibility CTEs; preserve branch overlap and NULL semantics.",
                 "Source order keys often have ties. Physical hints can change which tied LIMIT rows are returned; the root equivalence gate must retain ordered comparison and reject unsupported candidates.",
             ]}
    output = ROOT / "artifacts/model_dsb.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("w") as handle:
        for ordinal, (original, memberships) in enumerate(grouped.items(), 1):
            if args.max_queries and ordinal > args.max_queries:
                break
            sha = hashlib.sha256(original.encode()).hexdigest()
            record = {"original_sql": original, "original_sha256": sha,
                      "benchmark": "dsb", "candidates": [],
                      "provenance": {**MODEL, "evidence": [], "source_ids": [r["id"] for r in memberships]}}
            qnote = {"sha256": sha, "source_ids": record["provenance"]["source_ids"],
                     "original_parse": {}, "original_explain": {}, "attempts": []}
            original_plan = None
            try:
                parse_read_query(original)
                qnote["original_parse"] = {"status": "ok"}
            except Exception as error:
                qnote["original_parse"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            try:
                original_plan = db.explain(original)
                qnote["original_explain"] = {"status": "ok", **summary(original_plan)}
            except Exception as error:
                qnote["original_explain"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            try:
                hypotheses = specs(original, unique_constraints)
            except Exception as error:
                hypotheses = []
                qnote["generation_error"] = f"{type(error).__name__}: {error}"
            # Choose hint alternatives from operators actually present, never
            # invent relation aliases or change connection-wide settings.
            if original_plan is not None:
                counts_by_node = Counter(node["node_type"] for node in operators(original_plan))
                for node, setting in [("Nested Loop", "enable_nestloop"), ("Hash Join", "enable_hashjoin"), ("Merge Join", "enable_mergejoin")]:
                    if counts_by_node[node]:
                        hypotheses.append((f"/*+ Set({setting} off) */\n" + original,
                                           "pg_hint_plan_disable_" + setting.removeprefix("enable_"),
                                           f"The original plan contains {counts_by_node[node]} {node} operators; explore an alternative join strategy using a statement-local pg_hint_plan planner switch."))
            seen = {original}
            for candidate, action, rationale in hypotheses:
                if len(record["candidates"]) >= 3:
                    qnote["attempts"].append({"action": action, "status": "not_attempted", "reason": "three-candidate cap"})
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                attempt = {"action": action, "rationale": rationale,
                           "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest()}
                try:
                    parse_read_query(candidate)
                    attempt["parse"] = "ok"
                    candidate_plan = db.explain(candidate)
                    attempt["explain"] = {"status": "ok", **summary(candidate_plan)}
                    if action.startswith("remove_unused_unique_left_join_") and original_plan is not None and shape(candidate_plan) == shape(original_plan):
                        attempt.update(status="discarded", reason="original PostgreSQL plan already eliminates the unused unique LEFT JOIN")
                        qnote["attempts"].append(attempt)
                        continue
                    if original_plan is not None:
                        if action.startswith("pg_hint_plan_") and shape(candidate_plan) == shape(original_plan):
                            attempt.update(status="discarded", reason="hint produced no operator/alias/join-type plan change")
                            qnote["attempts"].append(attempt)
                            continue
                        ratio = candidate_plan["Plan"]["Total Cost"] / max(1, original_plan["Plan"]["Total Cost"])
                        if ratio > 4:
                            attempt.update(status="discarded", reason="planner alternative cost exceeds original by more than 4x", cost_ratio=ratio)
                            qnote["attempts"].append(attempt)
                            continue
                    evidence = [{"type": "source_knowledge", "paths": SOURCES[:3]},
                                {"type": "reasoning", "rationale": rationale},
                                {"type": "postgres_explain", "analyze": False,
                                 "original": qnote["original_explain"], "candidate": attempt["explain"],
                                 "equivalence_verified": False, "runtime_measured": False}]
                    record["candidates"].append({"sql": candidate, "action": action, "action_class": "B",
                                                 "provenance": {**MODEL, "evidence": evidence}})
                    attempt["status"] = "retained_unverified"
                    counts[action] += 1
                except Exception as error:
                    attempt.update(status="error", error=f"{type(error).__name__}: {error}")
                qnote["attempts"].append(attempt)
            notes["queries"].append(qnote)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            (ROOT / "artifacts/dsb_macro_notes.json").write_text(json.dumps(notes, indent=2))
            if ordinal % 10 == 0 or ordinal == len(grouped):
                print(f"DSB {ordinal}/{len(grouped)} originals, {sum(counts.values())} retained unverified candidates", flush=True)
    db.close()
    notes.update(completed=datetime.now(timezone.utc).isoformat(), candidate_counts=dict(counts),
                 candidates=sum(counts.values()), empty_originals=sum(not any(a.get("status") == "retained_unverified" for a in q["attempts"]) for q in notes["queries"]))
    (ROOT / "artifacts/dsb_macro_notes.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps({"output": str(output), "unique_originals": len(notes["queries"]),
                      "candidates": notes["candidates"], "empty_originals": notes["empty_originals"],
                      "candidate_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
