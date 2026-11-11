#!/usr/bin/env python3
"""Run a genuinely fresh R-Bot or QUITE query with configured upstream deps."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcts_rewrite.adapters import UpstreamStandalone

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("system", choices=["rbot", "quite"])
parser.add_argument("--sql", type=Path, required=True)
parser.add_argument("--schema", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--database")
parser.add_argument("--timeout", type=float, default=600)
parser.add_argument("--upstream-python", help="Defaults to .venv-SYSTEM/bin/python when installed")
parser.add_argument("--pg-hint-plan", action="store_true", help="Enable only after verifying pg_hint_plan in the target DB")
args = parser.parse_args()
result = UpstreamStandalone(Path(__file__).resolve().parents[1], args.system,
                            timeout=args.timeout, python=args.upstream_python).rewrite(
    args.sql.read_text(), args.schema.read_text(), output_dir=args.output_dir,
    database=args.database, hint_capable=args.pg_hint_plan)
print(json.dumps(result, indent=2))
sys.exit(0 if result.get("status") == "ok" else 1)
