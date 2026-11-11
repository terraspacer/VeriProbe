#!/usr/bin/env python3
"""Build finite adversarial fixtures separately from the measured databases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.verify import synthesize_mini_databases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/development.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/development-hybrid.json")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "configs/miniatures.json")
    parser.add_argument("--admin-dsn", default=f"host={ROOT}/.runtime/socket port=55432 dbname=mcts_fixtures user=steve")
    parser.add_argument("--read-dsn", default=f"host={ROOT}/.runtime/socket port=55432 dbname=mcts_fixtures user=mcts_read")
    parser.add_argument("--reuse", action="store_true", help="Reuse saved schemas after validating their source hashes and existence")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    source_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    bundle = None
    if args.reuse and args.fixtures.exists():
        bundle = json.loads(args.fixtures.read_text())
        if bundle["manifest_sha256"] != source_hash:
            raise ValueError("Saved fixtures belong to a different manifest")
        with psycopg.connect(args.read_dsn) as conn:
            for specs in bundle["mini_dbs"].values():
                for spec in specs:
                    if not conn.execute("SELECT EXISTS(SELECT FROM pg_namespace WHERE nspname=%s)", (spec["search_path"],)).fetchone()[0]:
                        raise ValueError("Saved miniature schema is missing")
    if bundle is None:
        bundle = {"manifest_sha256": source_hash, "evidence": "Finite adversarial examples, never a universal proof", "mini_dbs": {}, "schema_hashes": {}}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        schema_path = ROOT / "vendor/LLM4Rewrite" / benchmark / "create_tables.sql"
        schema = schema_path.read_text()
        schema_hash = hashlib.sha256(schema.encode()).hexdigest()
        if benchmark in bundle["mini_dbs"]:
            if bundle["schema_hashes"][benchmark] != schema_hash:
                raise ValueError("Saved fixtures belong to a different schema")
            continue
        specs = synthesize_mini_databases(args.admin_dsn, schema, read_dsn=args.read_dsn,
                                          query_sqls=[row["sql"] for row in rows if row["benchmark"] == benchmark])
        bundle["mini_dbs"][benchmark] = specs
        bundle["schema_hashes"][benchmark] = schema_hash
        temporary = args.fixtures.with_suffix(".tmp")
        temporary.write_text(json.dumps(bundle, indent=2) + "\n")
        temporary.replace(args.fixtures)
        print(json.dumps({"benchmark": benchmark, "cases": [{"name": item["name"], "rows": item["rows_inserted"], "complete": item["coverage_complete"]} for item in specs]}), flush=True)
    config = json.loads(args.config.read_text())
    config["mini_dbs"] = bundle["mini_dbs"]
    config["fixture_manifest_sha256"] = source_hash
    config["verification_mode"] = "hybrid"
    config["model"] = {"file": "artifacts/model_candidates.jsonl"}
    config["pg_hint_plan"] = True
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
