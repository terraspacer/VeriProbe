#!/usr/bin/env python3
"""Import every released driver input in the three shared benchmark families.

Driver membership is the unit of observation: identical SQL is retained once per
upstream input, and parsing diagnostics never filter an entry from the manifest.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def diagnostics(sql: str) -> dict:
    try:
        import sqlglot
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
        return {"parse_status": "ok", "statement_count": len(statements),
                "statement_types": [type(s).__name__ for s in statements]}
    except Exception as exc:
        return {"parse_status": "error", "statement_count": None,
                "parse_error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/manifest.jsonl"))
    args = parser.parse_args()
    root = args.root.resolve()
    rbot, quite = root / "vendor/LLM4Rewrite", root / "vendor/QUITE"
    commits = {"rbot": revision(rbot), "quite": revision(quite)}
    rows = []

    def append(source: str, benchmark: str, key: str, sql: str, path: Path,
               schema: Path, index: int | None = None, raw: dict | None = None,
               raw_sql: str | None = None) -> None:
        row = {"id": f"{source}:{benchmark}:{key}", "source": source,
               "benchmark": benchmark, "sql": sql,
               "schema_path": str(schema.relative_to(root)),
               "schema_sha256": digest(schema.read_bytes()),
               "source_path": str(path.relative_to(root)),
               "sha256": digest(sql.encode()), "source_file_sha256": digest(path.read_bytes()),
               "source_commit": commits[source], "source_index": index,
               "source_query_id": key, **diagnostics(sql)}
        if raw_sql is not None:
            row["raw_source_sql"] = raw_sql
            row["raw_source_sql_sha256"] = digest(raw_sql.encode())
            row["input_processing"] = "R-Bot my_rewriter/test.py: strip -- comments, split ';', trim, append ';'"
        if raw is not None:
            row["source_metadata"] = {k: v for k, v in raw.items()
                                      if k not in ("input_sql", "query", "rewrites")}
        rows.append(row)

    for benchmark in ("tpch", "dsb"):
        paths = [p for p in (rbot / benchmark).glob("query*/*.sql")
                 if re.fullmatch(r"query\d+_[01]\.sql", p.name)]
        for path in sorted(paths, key=lambda p: tuple(map(int, re.findall(r"\d+", p.stem)))):
            raw_sql = path.read_text()
            content = re.sub(r"--.*\n", "", raw_sql)
            queries = [q.strip() + ";" for q in content.split(";") if q.strip()]
            for index, sql in enumerate(queries):
                key = path.stem if len(queries) == 1 else f"{path.stem}_{index}"
                append("rbot", benchmark, key, sql, path,
                       rbot / benchmark / "create_tables.sql", index, raw_sql=raw_sql)
    path = rbot / "calcite/calcite.jsonl"
    for index, line in enumerate(path.read_text().splitlines(), 1):
        raw = json.loads(line)
        key = sorted(x["name"] for x in raw["rewrites"])[0]
        append("rbot", "calcite", key, raw["input_sql"], path,
               rbot / "calcite/create_tables.sql", index, raw)
    for benchmark in ("tpch", "dsb", "calcite"):
        path = quite / f"dataset/queries/{benchmark}_queries.json"
        for index, raw in enumerate(json.loads(path.read_text()), 1):
            key = f"{raw.get('id', index)}:entry{index:03d}"
            append("quite", benchmark, key, raw["query"], path,
                   quite / f"dataset/schemas/{benchmark}_schemas.sql", index, raw)
    memberships = collections.defaultdict(list)
    for row in rows:
        memberships[row["sha256"]].append(row["id"])
    for row in rows:
        row["duplicate_memberships"] = memberships[row["sha256"]]
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    source_entries = {}
    for row in rows:
        key = (row["source_path"], row["source_index"] if row["source_path"].endswith((".json", ".jsonl")) else None)
        if key not in source_entries:
            source_entries[key] = {"source": row["source"], "benchmark": row["benchmark"],
                                   "source_path": row["source_path"], "source_index": key[1],
                                   "source_commit": row["source_commit"],
                                   "source_file_sha256": row["source_file_sha256"],
                                   "raw_sql": row.get("raw_source_sql", row["sql"]), "driver_inputs": []}
        source_entries[key]["driver_inputs"].append(row["id"])
    source_output = output.parent / "source_entries.jsonl"
    source_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_entries.values()))
    report = {"format_version": 1, "unit": "exact upstream executed driver input",
              "source_commits": commits, "manifest_sha256": digest(output.read_bytes()),
              "raw_source_entries": len(source_entries),
              "raw_source_entries_sha256": digest(source_output.read_bytes()),
              "split_source_entries": [r for r in source_entries.values() if len(r["driver_inputs"]) > 1],
              "entries": len(rows), "counts": dict(collections.Counter(
                  f"{r['source']}:{r['benchmark']}" for r in rows)),
              "parse_failures": [r["id"] for r in rows if r["parse_status"] != "ok"],
              "multi_statements": [r["id"] for r in rows if (r["statement_count"] or 0) > 1],
              "unique_exact_sql": len(memberships),
              "scope": ["tpch", "dsb", "calcite"],
              "other_upstream_families": {"quite:stackoverflow": {
                  "entries": 43, "path": "vendor/QUITE/dataset/queries/stackoverflow_queries.json",
                  "reason": "Outside requested shared TPC-H/DSB/Calcite scope; not silently merged."}}}
    (output.parent / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
