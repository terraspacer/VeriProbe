#!/usr/bin/env python3
"""Wait for the published databases' insert-triggered maintenance to finish.

Only pg_class/pg_stat metadata is inspected. No user-table scans or manual
VACUUM commands are issued. An optional cost-limit override is journaled before
application and restored before the readiness gate can open.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
from pathlib import Path
import signal
import struct
import sys
import time

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

try:
    from .setup_datasets import ROOT, DEFAULT_DSN, TARGETS, atomic_json, emit
except ImportError:
    from setup_datasets import ROOT, DEFAULT_DSN, TARGETS, atomic_json, emit

DATABASES = tuple(target["dbname"] for target in TARGETS["published"].values())
COST_NAME = "autovacuum_vacuum_cost_limit"
SETTINGS = ("server_version_num", "autovacuum", COST_NAME,
            "autovacuum_vacuum_cost_delay", "vacuum_cost_limit",
            "autovacuum_max_workers", "maintenance_work_mem", "autovacuum_naptime",
            "autovacuum_vacuum_insert_threshold", "autovacuum_vacuum_insert_scale_factor")


class Interrupted(RuntimeError):
    pass


class Termination:
    def __init__(self):
        self.signum = None
        self.cleaning = False

    def handle(self, signum, _frame):
        self.signum = signum
        if not self.cleaning:
            raise Interrupted(f"Received signal {signum}")

    @contextmanager
    def cleanup(self):
        self.cleaning = True
        try:
            yield
        finally:
            self.cleaning = False


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def float4(value):
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def insert_trigger(table, settings):
    """PostgreSQL14 autovacuum.c:3218-3242, including float4 arithmetic."""
    options = dict(item.split("=", 1) for item in table.get("reloptions", []) if "=" in item)
    enabled = options.get("autovacuum_enabled", "true").lower() not in ("false", "off", "0")
    base = int(options.get("autovacuum_vacuum_insert_threshold", settings["autovacuum_vacuum_insert_threshold"]))
    scale = float4(options.get("autovacuum_vacuum_insert_scale_factor", settings["autovacuum_vacuum_insert_scale_factor"]))
    tuples = float4(max(float(table["estimated_rows"]), 0))
    threshold = float4(float4(base) + float4(scale * tuples))
    pending = enabled and base >= 0 and int(table["inserts_since_vacuum"]) > threshold
    return {"pending": pending, "enabled": enabled and base >= 0, "threshold": threshold}


def quiet(observation):
    if observation.get("stats_visibility_verified") is not True:
        raise RuntimeError("Statistics visibility is unknown; cannot infer quiet")
    if set(observation.get("published_databases", [])) != set(DATABASES):
        raise RuntimeError("Published database inventory is incomplete")
    return not observation["active_autovacuum"] and not observation["pending_insert_vacuum_tables"]


def host_pressure():
    result = {}
    for name in ("memory", "io", "cpu"):
        path = Path("/proc/pressure") / name
        result[name] = path.read_text().strip() if path.exists() else None
    values = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 3}
    result["MemAvailable_bytes"] = values.get("MemAvailable")
    result["SwapUsed_bytes"] = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    # Cumulative counters support rate calculations without a synthetic I/O test.
    result["diskstats"] = [line.split() for line in Path("/proc/diskstats").read_text().splitlines()
                           if line.split()[2] == "sda"]
    return result


class PostgresMonitor:
    def __init__(self, dsn):
        self.dsn = dsn

    @contextmanager
    def connect(self, database="postgres"):
        with psycopg.connect(make_conninfo(self.dsn, dbname=database, connect_timeout=5), autocommit=True) as conn:
            conn.execute("SET statement_timeout='5s'")
            conn.execute("SET lock_timeout='2s'")
            yield conn

    def settings(self, conn):
        values = dict(conn.execute("SELECT name,setting FROM pg_settings WHERE name=ANY(%s)", (list(SETTINGS),)))
        if not 140000 <= int(values["server_version_num"]) < 150000:
            raise RuntimeError("This monitor's insert-trigger rule is verified for PostgreSQL14 only")
        if values["autovacuum"] != "on":
            raise RuntimeError("Autovacuum must be on before maintenance can finish")
        return values

    def capture_policy(self):
        with self.connect() as conn:
            values = self.settings(conn)
            row = conn.execute("""SELECT setting FROM pg_file_settings
                WHERE sourcefile=current_setting('data_directory') || '/postgresql.auto.conf'
                AND name=%s AND error IS NULL ORDER BY seqno DESC LIMIT 1""", (COST_NAME,)).fetchone()
            return {"cost_limit": values[COST_NAME], "cost_delay": values["autovacuum_vacuum_cost_delay"],
                    "auto_conf_present": bool(row), "auto_conf_value": row[0] if row else None,
                    "settings": values}

    def reload_and_check(self, expected):
        with self.connect() as conn:
            if not conn.execute("SELECT pg_reload_conf()").fetchone()[0]:
                raise RuntimeError("PostgreSQL rejected configuration reload")
        deadline = time.monotonic() + 5
        while True:
            with self.connect() as conn:
                current = self.settings(conn)
            if all(current[name] == value for name, value in expected.items()):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Configuration reload not effective: expected {expected}")
            time.sleep(0.1)

    def apply_boost(self, limit):
        with self.connect() as conn:
            conn.execute(sql.SQL("ALTER SYSTEM SET {}={}").format(sql.Identifier(COST_NAME), sql.Literal(str(limit))))
        self.reload_and_check({COST_NAME: str(limit)})

    def restore_policy(self, original, may_mutate):
        current = self.capture_policy()
        same = all(current[key] == original[key] for key in ("cost_limit", "cost_delay", "auto_conf_present", "auto_conf_value"))
        if not same:
            if not may_mutate:
                raise RuntimeError("Cost policy changed externally during an observation-only run")
            with self.connect() as conn:
                if original["auto_conf_present"]:
                    conn.execute(sql.SQL("ALTER SYSTEM SET {}={}").format(sql.Identifier(COST_NAME), sql.Literal(original["auto_conf_value"])))
                else:
                    conn.execute(sql.SQL("ALTER SYSTEM RESET {}").format(sql.Identifier(COST_NAME)))
            self.reload_and_check({COST_NAME: original["cost_limit"],
                                   "autovacuum_vacuum_cost_delay": original["cost_delay"]})
        verified = self.capture_policy()
        if any(verified[key] != original[key] for key in ("cost_limit", "cost_delay", "auto_conf_present", "auto_conf_value")):
            raise RuntimeError("Original cost policy or override presence was not restored")
        return verified

    def observe(self):
        with self.connect() as conn:
            values = self.settings(conn)
            user, stats = conn.execute("SELECT current_user,pg_has_role(current_user,'pg_read_all_stats','USAGE')").fetchone()
            if not stats:
                raise RuntimeError("pg_read_all_stats is required; hidden backend_type must not imply quiet")
            active = [{"pid": pid, "database": db, "query": query, "wait_event_type": wait_type, "wait_event": wait}
                      for pid, db, query, wait_type, wait in conn.execute("""SELECT pid,datname,query,wait_event_type,wait_event
                        FROM pg_stat_activity WHERE backend_type='autovacuum worker' ORDER BY pid""")]
            progress = [{"pid": pid, "database": db, "phase": phase, "scanned_blocks": done, "total_blocks": total}
                        for pid, db, phase, done, total in conn.execute("""SELECT pid,datname,phase,heap_blks_scanned,heap_blks_total
                            FROM pg_stat_progress_vacuum ORDER BY pid""")]
        tables, pending = [], []
        for database in DATABASES:
            with self.connect(database) as conn:
                for table, size, inserts, tuples, options, last in conn.execute("""SELECT c.relname,pg_relation_size(c.oid),
                    s.n_ins_since_vacuum,c.reltuples,c.reloptions,s.last_autovacuum
                    FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    JOIN pg_stat_all_tables s ON s.relid=c.oid
                    WHERE n.nspname='public' AND c.relkind='r' AND c.relname NOT LIKE '_mcts_%%'
                    ORDER BY c.relname"""):
                    if inserts is None:
                        raise RuntimeError(f"Missing statistics for {database}.{table}")
                    record = {"database": database, "table": table, "heap_bytes": size,
                              "inserts_since_vacuum": inserts, "estimated_rows": tuples,
                              "reloptions": options or [], "last_autovacuum": last.isoformat() if last else None}
                    record.update(insert_trigger(record, values))
                    tables.append(record)
                    if record["pending"]:
                        pending.append(record)
        return {"observed": timestamp(), "published_databases": list(DATABASES),
                "stats_visibility_verified": True, "monitor_role": user,
                "settings": values, "active_autovacuum": active, "progress": progress,
                "pending_insert_vacuum_tables": pending, "table_count": len(tables),
                "pending_heap_bytes": sum(table["heap_bytes"] for table in pending),
                "source_rule": ".local/src/postgresql-14.13/src/backend/postmaster/autovacuum.c:3218-3242",
                "pressure": host_pressure()}


def run_monitor(backend, write, *, boost=None, poll_seconds=30, quiet_polls=2,
                max_seconds=7200, now=time.monotonic, sleep=time.sleep, termination=None):
    termination = termination or Termination()
    state = {"format_version": 2, "started": timestamp(), "observed": timestamp(),
             "status": "starting", "ready_for_timing": False, "settings_restored": False,
             "requested_boost_cost_limit": boost, "poll_seconds": poll_seconds,
             "quiet_polls_required": quiet_polls, "max_seconds": max_seconds,
             "poll_count": 0, "consecutive_quiet": 0, "history": []}
    write(state)
    original, success, error = None, False, None
    deadline = now() + max_seconds
    try:
        original = backend.capture_policy()
        state["original_policy"] = original
        state["settings_restored"] = boost is None
        state["status"] = "pending"
        # This journal is durable before a requested policy mutation occurs.
        write(state)
        if boost is not None:
            backend.apply_boost(boost)
        while True:
            if now() >= deadline:
                raise TimeoutError("Post-load maintenance exceeded its bounded runtime")
            observation = backend.observe()
            if now() >= deadline:
                raise TimeoutError("Post-load maintenance exceeded its bounded runtime")
            is_quiet = quiet(observation)
            state["poll_count"] += 1
            state["consecutive_quiet"] = state["consecutive_quiet"] + 1 if is_quiet else 0
            state["observed"] = observation["observed"]
            state["last_observation"] = observation
            state["active_autovacuum"] = observation["active_autovacuum"]
            state["pending_insert_vacuum_tables"] = observation["pending_insert_vacuum_tables"]
            state["history"] = (state["history"] + [observation])[-40:]
            write(state)
            emit(status="postload_maintenance_poll", active=len(observation["active_autovacuum"]),
                 pending=len(observation["pending_insert_vacuum_tables"]), quiet_polls=state["consecutive_quiet"])
            if state["consecutive_quiet"] >= quiet_polls:
                success = True
                break
            sleep(min(poll_seconds, max(0, deadline - now())))
    except BaseException as exc:
        error = exc
    finally:
        with termination.cleanup():
            try:
                if original is None:
                    raise RuntimeError("Original cost policy was not captured")
                state["restored_policy"] = backend.restore_policy(original, boost is not None)
                state["settings_restored"] = True
                if success:
                    final = backend.observe()
                    if not quiet(final):
                        raise RuntimeError("Maintenance reappeared after original policy restoration")
                    state["observed"] = final["observed"]
                    state["last_observation"] = final
                    state["active_autovacuum"] = []
                    state["pending_insert_vacuum_tables"] = []
            except BaseException as restore_error:
                state["settings_restored"] = False
                error = RuntimeError(f"Original policy restoration/verification failed: {restore_error}")
        if termination.signum is not None:
            error = Interrupted(f"Received signal {termination.signum}")
        if error is None and success and state["settings_restored"]:
            state.update(status="complete", ready_for_timing=True, completed=timestamp())
        else:
            state.update(status="interrupted" if isinstance(error, Interrupted) else "failed",
                         ready_for_timing=False, error=str(error))
        state["observed"] = timestamp()
        write(state)
    if error is not None:
        raise error
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-dsn", default=DEFAULT_DSN)
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/postload_maintenance.json")
    parser.add_argument("--boost-cost-limit", type=int, choices=(1000, 2000))
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--quiet-polls", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.quiet_polls < 2 or not 0 < args.max_seconds <= 7200:
        parser.error("poll-seconds >=30, quiet-polls >=2, and max-seconds in (0,7200] are required")
    report = args.report if args.report.is_absolute() else ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / ".runtime/postload_maintenance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    termination = Termination()
    signal.signal(signal.SIGTERM, termination.handle)
    signal.signal(signal.SIGINT, termination.handle)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_monitor(PostgresMonitor(args.admin_dsn), lambda value: atomic_json(report, value),
                        boost=args.boost_cost_limit, poll_seconds=args.poll_seconds,
                        quiet_polls=args.quiet_polls, max_seconds=args.max_seconds, termination=termination)
        except BaseException as exc:
            emit(status="postload_maintenance_failed", error=str(exc))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
