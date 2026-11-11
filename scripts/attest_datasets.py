#!/usr/bin/env python3
"""Attest loaded datasets and enable the required hint extension.

With --wait, process each target once its bulk loader has completed. This also
supports loaders started before the extension became available. It never starts
data generation, bulk loading, or latency measurements.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

import psycopg
from psycopg.conninfo import make_conninfo

from setup_datasets import DEFAULT_DSN, ROOT, TARGETS, GENERATORS, emit, atomic_json


def runtime_service_limits(conn):
    """Record the actual local instance limits, independently of SQL settings."""
    unit = "mcts-rewrite-postgres.service"
    if conn.info.host != str(ROOT / ".runtime/socket") or conn.info.port != 55432:
        return {"available": False, "reason": "Selected DSN is not the local runtime service"}
    properties = "MainPID,ActiveState,MemoryHigh,MemoryMax,MemorySwapMax,ManagedOOMPreference"
    try:
        output = subprocess.check_output(["systemctl", "--user", "show", unit,
            "--property=" + properties], text=True, stderr=subprocess.STDOUT)
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        postmaster = (ROOT / ".runtime/pgdata/postmaster.pid").read_text().splitlines()[0]
        if values.get("ActiveState") != "active" or values.get("MainPID") != postmaster:
            raise RuntimeError("The runtime unit does not identify the connected local postmaster")
        return {"available": True, "unit": unit,
            **{name + "_bytes": int(values[name]) if values[name].isdigit() else values[name]
               for name in ("MemoryHigh", "MemoryMax", "MemorySwapMax")},
            "ManagedOOMPreference": values["ManagedOOMPreference"],
            "observed": dt.datetime.now(dt.timezone.utc).isoformat()}
    except (OSError, subprocess.CalledProcessError, KeyError, RuntimeError) as exc:
        return {"available": False, "unit": unit, "reason": str(exc)}


def attest(target, admin_dsn):
    path = ROOT / ".runtime/datasets" / f"{target['dbname']}.json"
    report = json.loads(path.read_text())
    if not report.get("ready"):
        raise RuntimeError(f"Dataset not ready: {target['dbname']}")
    generation = report["generation"]
    if generation.get("size_parameter", generation.get("scale")) != target["scale"]:
        raise RuntimeError(f"Generator scale mismatch: {target['dbname']}")
    if generation.get("source_commit") != GENERATORS[target["benchmark"]][2]:
        raise RuntimeError(f"Generator revision mismatch: {target['dbname']}")
    if target["benchmark"] == "calcite" and generation.get("distribution") != target["distribution"]:
        raise RuntimeError(f"Generator distribution mismatch: {target['dbname']}")
    if target["benchmark"] == "calcite":
        if set(generation["tables"]) != set(report["tables"]):
            raise RuntimeError("Calcite loaded-table inventory differs from the generation manifest")
        for table, receipt in report["tables"].items():
            expected = generation["tables"][table]
            if (receipt["rows"], receipt["bytes"], receipt["source_sha256"]) != (expected["rows"], expected["bytes"], expected["sha256"]):
                raise RuntimeError(f"Calcite COPY receipt differs from the completed generation manifest: {table}")
    with psycopg.connect(make_conninfo(admin_dsn, dbname=target["dbname"]), autocommit=True) as conn:
        existing = conn.execute("SELECT record FROM _mcts_dataset_provenance").fetchone()[0]
        if existing["tables"] != report["tables"] or existing["scale"] != target["scale"]:
            raise RuntimeError(f"Inconsistent persisted provenance: {target['dbname']}")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_hint_plan")
        report["extensions"] = dict(conn.execute("SELECT extname,extversion FROM pg_extension"))
        report["postgres_settings"] = dict(conn.execute("SELECT name,setting FROM pg_settings WHERE name IN ('server_version','shared_buffers','work_mem','maintenance_work_mem','autovacuum','max_wal_size','checkpoint_timeout','checkpoint_completion_target','fsync','synchronous_commit','wal_level','max_parallel_workers_per_gather','jit','session_preload_libraries','shared_preload_libraries')"))
        report["postgres_runtime_service"] = runtime_service_limits(conn)
        report["attested"] = dt.datetime.now(dt.timezone.utc).isoformat()
        conn.execute("UPDATE _mcts_dataset_provenance SET record=%s::jsonb", (json.dumps(report),))
        atomic_json(path, report)
        emit(database=target["dbname"], status="attested", extension=report["extensions"]["pg_hint_plan"])
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["development", "published", "all"], default="all")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--restore-autovacuum", action="store_true",
                        help="Restore autovacuum after all requested ready targets are attested")
    parser.add_argument("--admin-dsn", default=DEFAULT_DSN)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/dataset_readiness.json")
    args = parser.parse_args()
    targets = [target for profile, data in TARGETS.items() if args.profile in ("all", profile) for target in data.values()]
    pending = list(targets)
    records = {}
    deadline = time.monotonic() + 24 * 3600
    while pending:
        for target in list(pending):
            path = ROOT / ".runtime/datasets" / f"{target['dbname']}.json"
            if path.exists() and json.loads(path.read_text()).get("ready"):
                try:
                    records[target["dbname"]] = attest(target, args.admin_dsn)
                except psycopg.OperationalError as exc:
                    if not args.wait:
                        raise
                    emit(status="attestation_waiting_for_postgresql", error=str(exc))
                    break
                pending.remove(target)
        output = {"format_version": 1, "complete": not pending and not args.restore_autovacuum,
                  "pending": [t["dbname"] for t in pending], "datasets": records,
                  "updated": dt.datetime.now(dt.timezone.utc).isoformat()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, output)
        if not pending and args.restore_autovacuum:
            with psycopg.connect(args.admin_dsn, autocommit=True) as conn:
                conn.execute("ALTER SYSTEM SET autovacuum='on'")
                conn.execute("SELECT pg_reload_conf()")
            restored = None
            for _ in range(20):
                with psycopg.connect(args.admin_dsn, autocommit=True) as conn:
                    restored = conn.execute("SHOW autovacuum").fetchone()[0]
                if restored == "on":
                    break
                time.sleep(0.1)
            if restored != "on":
                raise RuntimeError(f"Autovacuum reload not effective: {restored}")
            # Refresh early-ready datasets under the final shared server
            # profile rather than preserving their pre-recovery settings.
            records = {target["dbname"]: attest(target, args.admin_dsn) for target in targets}
            output["datasets"] = records
            output["complete"] = True
            output["autovacuum_restoration"] = {"requested": "on", "reload_requested": True,
                "observed": restored,
                "time": dt.datetime.now(dt.timezone.utc).isoformat()}
            output["updated"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(args.output, output)
            emit(status="autovacuum_restored", setting="on", reload=True)
        if not pending or not args.wait:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Dataset attestation incomplete: {output['pending']}")
        time.sleep(10)


if __name__ == "__main__":
    main()
