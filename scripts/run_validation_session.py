#!/usr/bin/env python3
"""Re-time original and rewritten statements in one session under equal cutoffs.

The campaign logs compare arms whose runs were cut off at different times
(20 s to 300 s) and, for the development cohort, were timed in different
campaigns.  This script closes both gaps without selecting anything new.  For
every (database, statement) pair for which some arm returned a rewrite it times

    the original, the rewrite that the selection layer returned, and the
    replayed R-Bot and QUITE rewrites

back to back, with the same cutoff for every statement and with the runs of the
statements interleaved, so that host load affects all of them alike.

Protocol per pair
    three rounds; in each round every statement runs once; the order rotates
    by one position per round; a statement that reaches the cutoff once is
    recorded as cut off and is not run again.

Nothing is verified here: every statement already passed the equivalence gate
in the campaign that produced it.  The output is append-only and resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcts_rewrite.db import Postgres  # noqa: E402

RAW = [
    ("dev", "results/icde_paper_campaign_v4/raw.jsonl"),
    ("tpch", "results/icde_full_suite_tpch10/raw.jsonl"),
    ("calcite", "results/icde_full_suite_calcite10/raw.jsonl"),
    ("zipf", "results/icde_calcite10zipf_stage/raw.jsonl"),
    ("sf50", "results/icde_sf50_cohort/raw.jsonl"),
]
ARMS = ("mcts", "rbot", "quite")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def censored(choice: dict) -> bool:
    return not choice.get("fallback") and choice.get("measurement", {}).get("status") != "ok"


def build_tasks() -> list[dict]:
    tasks, seen = [], set()
    dev_ids: set[str] = set()
    for source, path in RAW:
        for row in read_jsonl(ROOT / path):
            if source == "dev":
                dev_ids.add(row["query_id"])
            key = (row["dataset"], row["query_id"])
            if key in seen:
                continue
            seen.add(key)
            original = row["original_sql"]
            statements = [{"roles": ["original"], "sql": original}]
            for arm in ARMS:
                choice = row["arms"][arm]
                sql = choice.get("sql", "")
                if choice.get("fallback") or sql.strip() == original.strip():
                    continue
                for statement in statements:
                    if statement["sql"] == sql:
                        statement["roles"].append(arm)
                        break
                else:
                    statements.append({"roles": [arm], "sql": sql})
            if len(statements) == 1:
                continue
            replay_censored = any(censored(row["arms"][arm]) for arm in ("rbot", "quite"))
            ours = not row["arms"]["mcts"].get("fallback")
            large = row["dataset"].endswith("50")
            skewed = row["dataset"].endswith("zipf")
            if large:
                tier = 6
            elif skewed:
                tier = 4
            elif row["query_id"] in dev_ids:
                tier = 1
            elif replay_censored:
                tier = 2
            elif ours:
                tier = 3
            else:
                tier = 5
            tasks.append({"dataset": row["dataset"], "query_id": row["query_id"], "benchmark": row["benchmark"],
                          "tier": tier, "order": len(tasks), "statements": statements})
    tasks.sort(key=lambda task: (task["tier"], task["order"]))
    return tasks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/published-hybrid.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/icde_validation_session")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hours", type=float, default=7.0, help="stop before starting a pair after this budget")
    parser.add_argument("--max-tier", type=int, default=6)
    parser.add_argument("--tier-order", default="1,2,3,4,5,6",
                        help="comma-separated tiers in the order in which they are timed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="print the task plan and exit")
    args = parser.parse_args(argv)

    tier_order = [int(value) for value in args.tier_order.split(",") if value.strip()]
    rank = {tier: position for position, tier in enumerate(tier_order)}
    tasks = sorted((task for task in build_tasks() if task["tier"] <= args.max_tier and task["tier"] in rank),
                   key=lambda task: (rank[task["tier"]], task["order"]))
    if args.list:
        from collections import Counter
        print(json.dumps({"pairs": len(tasks),
                          "pairs_by_tier": dict(sorted(Counter(t["tier"] for t in tasks).items())),
                          "statements": sum(len(t["statements"]) for t in tasks)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw.jsonl"
    done = {(row["dataset"], row["query_id"]) for row in read_jsonl(raw_path)}
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    connections: dict[str, Postgres] = {}
    hints: dict[str, dict] = {}
    run = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "planned_pairs": len(tasks), "timeout_ms": args.timeout_ms, "rounds": args.rounds,
           "protocol": "interleaved rounds, rotating order, equal cutoff, one cutoff ends a statement"}

    def checkpoint(**fields) -> None:
        run.update(fields, completed_pairs=len(done),
                   updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (args.output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    checkpoint()
    executed = 0
    try:
        with raw_path.open("a") as stream:
            for task in tasks:
                key = (task["dataset"], task["query_id"])
                if key in done:
                    continue
                if (time.monotonic() - started) / 3600.0 >= args.hours:
                    checkpoint(status="stopped_at_time_budget")
                    return
                if args.limit is not None and executed >= args.limit:
                    checkpoint(status="stopped_at_limit")
                    return
                dataset = task["dataset"]
                if dataset not in connections:
                    database = config["databases"][dataset]
                    connections[dataset] = Postgres(database["dsn"], timeout_ms=args.timeout_ms,
                                                    search_path=database.get("search_path", "public"))
                    hints[dataset] = connections[dataset].planner_hints_enabled()
                db = connections[dataset]
                statements = [{"roles": s["roles"], "sql": s["sql"], "sql_sha256": sha(s["sql"]),
                               "samples_ms": [], "wall_ms": [], "status": "ok"} for s in task["statements"]]
                for statement in statements:
                    if "/*+" in statement["sql"] and not hints[dataset].get("enabled"):
                        statement.update(status="skipped", error="pg_hint_plan is not enabled on this database")
                record = {"dataset": dataset, "query_id": task["query_id"], "benchmark": task["benchmark"],
                          "tier": task["tier"], "timeout_ms": args.timeout_ms, "rounds": args.rounds,
                          "planner_hints": hints[dataset], "load_before": os.getloadavg(),
                          "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                for round_index in range(args.rounds):
                    shift = round_index % len(statements)
                    for statement in statements[shift:] + statements[:shift]:
                        if statement["status"] != "ok":
                            continue
                        measurement = db.measure(statement["sql"], runs=1, warmups=0)
                        if measurement.get("status") == "ok":
                            statement["samples_ms"].append(measurement["samples_ms"][0])
                            statement["wall_ms"].append(measurement["wall_samples_ms"][0])
                        elif measurement.get("status") == "timeout":
                            statement.update(status="cut_off", cut_off_round=round_index,
                                             error=str(measurement.get("error"))[:200])
                        else:
                            if measurement.get("error_type") in {"OperationalError", "InterfaceError"}:
                                raise RuntimeError("database connection failed: " + str(measurement.get("error")))
                            statement.update(status="error", error=str(measurement.get("error"))[:300])
                record.update(statements=statements, load_after=os.getloadavg(),
                              finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                done.add(key)
                executed += 1
                checkpoint(last_pair=list(key))
                summary = ", ".join(
                    "/".join(s["roles"]) + "=" + (f"{sorted(s['samples_ms'])[len(s['samples_ms']) // 2] / 1000:.1f}s"
                                                  if s["status"] == "ok" and s["samples_ms"] else s["status"])
                    for s in statements)
                print(json.dumps({"done": len(done), "of": len(tasks), "tier": task["tier"], "dataset": dataset,
                                  "query_id": task["query_id"], "result": summary,
                                  "elapsed_h": round((time.monotonic() - started) / 3600.0, 2)}), flush=True)
        checkpoint(status="completed")
    finally:
        for db in connections.values():
            db.close()


if __name__ == "__main__":
    main()
