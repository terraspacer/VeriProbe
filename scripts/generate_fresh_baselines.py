#!/usr/bin/env python3
"""Generate serial, checkpointed actual upstream baselines for each dataset.

Supply upstream model endpoints through the process environment. A supervised
localhost session-model bridge can serve actual Codex subagent responses; its
model substitutions and request audits remain explicit in catalog provenance.
Missing configuration stops that system before any database or model calls.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.fresh_baselines import generate_catalog

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.jsonl")
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="Atomic catalog JSONL destination")
parser.add_argument("--system", action="append", choices=("rbot", "quite"), help="Repeat to select systems; defaults to both")
parser.add_argument("--dataset", action="append", help="Repeat to select exact configured dataset names")
parser.add_argument("--query-id", action="append", help="Exact manifest id; repeat for multiple inputs")
parser.add_argument("--limit", type=int, help="Limit distinct selected SQL/schema inputs before expanding systems/datasets")
parser.add_argument("--timeout", type=float, default=600, help="Timeout per actual upstream execution in seconds")
parser.add_argument("--threads", type=int, default=1, help="CPU library threads per serial child")
parser.add_argument("--retry-failed", action="store_true", help="Retry failed checkpoints in the same generation context")
args = parser.parse_args()
report = generate_catalog(ROOT, args.manifest, json.loads(args.config.read_text()), args.output,
                          systems=args.system or ("rbot", "quite"), datasets=args.dataset,
                          query_ids=args.query_id, limit=args.limit, timeout=args.timeout,
                          threads=args.threads, retry_failed=args.retry_failed)
print(json.dumps(report, indent=2))
raise SystemExit(2 if report["status"] == "blocked" else 1 if report["status"] == "error" else 0)
