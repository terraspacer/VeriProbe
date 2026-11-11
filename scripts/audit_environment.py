#!/usr/bin/env python3
"""Audit exact query planning coverage and capture the local DB environment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.db import Postgres


def audit(config: Path, manifest: Path, output: Path):
    settings = json.loads(config.read_text())
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "environment_and_planning_audit_not_latency_benchmark",
        "python": sys.version,
        "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "databases": {}, "queries": [],
    }
    for variant, spec in settings["databases"].items():
        db = Postgres(spec["dsn"], timeout_ms=30000)
        record = {"benchmark": spec["benchmark"], "scale": spec["scale"], "status": "ok"}
        try:
            with db.connection.cursor() as cur:
                cur.execute("SELECT version(),current_database(),current_user")
                record["identity"] = list(cur.fetchone())
                cur.execute("SELECT name,setting,unit FROM pg_settings WHERE name = ANY(%s) ORDER BY name",
                            (["shared_buffers", "work_mem", "max_parallel_workers_per_gather", "jit", "track_io_timing", "session_preload_libraries", "server_version", "lc_collate", "lc_ctype", "default_transaction_read_only"],))
                record["settings"] = [dict(zip(("name", "setting", "unit"), row)) for row in cur.fetchall()]
                cur.execute("SELECT table_name,column_name,data_type,is_nullable FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position")
                record["schema"] = cur.fetchall()
                cur.execute("SELECT pg_database_size(current_database())")
                record["database_bytes"] = cur.fetchone()[0]
        except Exception as error:
            record.update(status="unavailable", error=str(error))
            report["databases"][variant] = record
            db.close()
            continue
        for query in rows:
            if query["benchmark"] != spec["benchmark"]:
                continue
            item = {"id": query["id"], "variant": variant, "sha256": query["sha256"]}
            try:
                plan = db.explain(query["sql"])
                item.update(status="ok", cost=plan["Plan"]["Total Cost"], node_type=plan["Plan"]["Node Type"])
            except Exception as error:
                item.update(status="planning_error", error=str(error))
            report["queries"].append(item)
        report["databases"][variant] = record
        db.close()
    report["status_counts"] = dict(Counter(row["status"] for row in report["queries"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"path": str(output), "status_counts": report["status_counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/development.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/environment_audit.json")
    args = parser.parse_args()
    audit(args.config, args.manifest, args.output)
