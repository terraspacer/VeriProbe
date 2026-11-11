#!/usr/bin/env python3
"""Run SlabCity's released Calcite generator with bounded row storage.

The only data changes are the explicit size parameter. Lists of generated rows
are spooled to CSV, range domains are lazy, and the nullable20 filter is applied
as each row arrives. These changes preserve random calls and CSV row order.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {"emp_table": "EMP", "emp_b_table": "EMP_B",
          "empnullables_table": "EMPNULLABLES",
          "empnullables_20_table": "EMPNULLABLES_20",
          "dept_table": "DEPT", "bonus_table": "BONUS"}


class StreamTable:
    def __init__(self, name: str, directory: Path, registry: dict):
        self.name, self.registry = name, registry
        self.path = directory / f"{name}.csv"
        self.handle = self.path.open("w")
        self.writer = csv.writer(self.handle)
        self.rows = 0
        self.columns = None
        self.started = time.monotonic()
        registry[name] = self

    def append(self, row: list) -> None:
        self.writer.writerow(row)
        self.rows += 1
        if self.rows == 1:
            self.columns = row
        if self.name == "EMPNULLABLES" and self.rows > 2:
            if row[1] != "" and row[6] != "" and row[1] == 20 and row[6] > 1000:
                self.registry["EMPNULLABLES_20"].append(row)
        if self.name == "EMP" and self.rows % 1000000 == 2:
            print(json.dumps({"table": self.name, "rows": self.rows - 2,
                              "elapsed_s": round(time.monotonic() - self.started, 1)}), flush=True)


class Rewrite(ast.NodeTransformer):
    def __init__(self, rows: int):
        self.rows = rows

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id == "size":
                node.value = ast.Constant(self.rows)
            elif target.id in TABLES:
                node.value = ast.Call(ast.Name("stream_table", ast.Load()),
                                      [ast.Constant(TABLES[target.id])], [])
            elif target.id in ("empno", "deptno"):
                # random.choice(s) and random.choices(s) use len/index identically.
                node.value = node.value.args[0]
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "row" and isinstance(node.iter, ast.Name) and node.iter.id == "empnullables_table":
            return ast.Pass()
        return self.generic_visit(node)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True, help="SlabCity size parameter, rows per main table")
    parser.add_argument("--distribution", choices=("uniform", "zipf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".local/generators/SlabCity")
    args = parser.parse_args()
    if args.rows < 1000:
        parser.error("rows must be at least 1000 so all source domain pools are populated")
    versions = {name: importlib.metadata.version(name) for name in ("Faker", "numpy")}
    expected = {"Faker": "22.2.0", "numpy": "1.26.3"}
    if versions != expected:
        raise RuntimeError(f"Generator requires upstream pinned dependencies {expected}, got {versions}")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source_root / f"benchmarks/data/calcite/calcite_{args.distribution}/calcite_sales.py"
    sys.path.insert(0, str(args.source_root / "benchmarks/data"))
    from database_generator.database_generator import DatabaseGenerator
    # Keep upstream constructor/helpers; rows are already written by spools.
    DatabaseGenerator.output = lambda self, name, table: table.handle.flush()
    tree = Rewrite(args.rows).visit(ast.parse(source.read_text()))
    ast.fix_missing_locations(tree)
    registry = {}
    namespace = {"__file__": str(source), "__name__": "__main__",
                 "stream_table": lambda name: StreamTable(name, args.output, registry)}
    old_argv = sys.argv
    sys.argv = [str(source), "explicit_size"]
    # Upstream zipf_transform retries negative int64 sums by calling
    # random.seed() without an argument, making 40M generation time-dependent.
    # Preserve its retry algorithm while making those retries reproducible.
    original_seed = random.seed
    retry_seeds = []
    def deterministic_seed(value=None, version=2):
        if value is None:
            value = 2333 + len(retry_seeds) + 1
            retry_seeds.append(value)
        return original_seed(value, version=version)
    random.seed = deterministic_seed
    try:
        exec(compile(tree, str(source), "exec"), namespace)
    finally:
        random.seed = original_seed
        sys.argv = old_argv
        for table in registry.values():
            table.handle.close()
    # DatabaseGenerator.output writes table objects, which our shim consumes.
    report = {"format_version": 1, "source_url": "https://github.com/eidos06/SlabCity",
              "source_commit": subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip(),
              "source_path": str(source.relative_to(args.source_root)), "source_sha256": sha256(source),
              "size_parameter": args.rows, "distribution": args.distribution, "seed": 2333,
              "dependencies": versions, "exact_author_snapshot": False,
              "zipf_overflow_retry_seeds": retry_seeds,
              "adaptations": ["explicit size parameter beyond upstream250K/1M/4M switches",
                              "row lists spooled to CSV without changing RNG calls",
                              "lazy range domains", "streaming nullable20 selection",
                              "system-time Zipf overflow retry seeds replaced with2334,2335,..."],
              "tables": {name.lower(): {"rows": table.rows - 2, "columns": table.columns,
                                        "file": table.path.name, "sha256": sha256(table.path),
                                        "bytes": table.path.stat().st_size}
                         for name, table in registry.items()}}
    temporary = args.output / "generation.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "generation.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
