#!/usr/bin/env python3
"""Run serial native and hybrid probes with actual supervised session workers.

The launcher supplies an ephemeral local credential in child environments only.
It does not create model workers: the supervising session supplies a fresh actual
GPT-6.1-sol xhigh agent for every request at the printed spool path. Shared helpers
also implement run_session_batch.py; neither entry point makes final performance
claims, including when selected datasets have publication scale.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ("full", "class_a_only", "class_b_only", "no_plan", "no_rag", "no_critic", "no_safety")


def parse_args(argv=None, *, batch=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=batch,
                        default=None if batch else ROOT / "configs/session-integration.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.jsonl")
    parser.add_argument("--output", type=Path, required=batch, default=None if batch else ROOT / "results/session_probe")
    parser.add_argument("--session-id", required=batch, default=None if batch else "native_probe_20261026")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 selects an available ephemeral port")
    parser.add_argument("--query-id", action="append", required=batch, help="Repeat exact manifest IDs")
    parser.add_argument("--dataset", action="append", required=batch, help="Repeat exact configured dataset names")
    parser.add_argument("--system", action="append", choices=("quite", "rbot"),
                        help="Native systems to generate serially; defaults to QUITE")
    parser.add_argument("--budgets", default=None if batch else "1", help="Comma-separated expansion budgets; batch defaults to config")
    parser.add_argument("--ablations", default=None if batch else "full", help="Comma-separated variants; batch defaults to config")
    parser.add_argument("--purpose", required=batch, help="Explicit functional/tuning scope; no final performance claim")
    parser.add_argument("--skip-native", action="store_true", help="Use seed catalog, or preserve unavailable baselines if absent")
    parser.add_argument("--skip-hybrid", action="store_true", help="Generate the native catalog only")
    parser.add_argument("--seed-catalog", type=Path, help="Preserve exact dataset-bound catalog rows; never relabel them new calls")
    parser.add_argument("--rbot-local-runtime", type=Path, help="Explicitly select a validated local embedding variant")
    parser.add_argument("--native-timeout", type=float, default=3600, help="Seconds per native query")
    args = parser.parse_args(argv)
    args.batch = batch
    args.query_id = list(dict.fromkeys(args.query_id or ["rbot:calcite:testAggregateRemove1"]))
    args.dataset = list(dict.fromkeys(args.dataset or ["calcite"]))
    args.system = list(dict.fromkeys(args.system or ["quite"]))
    if args.skip_native and args.skip_hybrid:
        parser.error("At least one of native or hybrid execution must be enabled")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", args.session_id):
        parser.error("Session ID must contain 1-96 letters, digits, underscores or hyphens")
    if not 0 <= args.port <= 65535 or not math.isfinite(args.native_timeout) or args.native_timeout <= 0:
        parser.error("Port must be 0-65535 and native timeout must be finite and positive")
    if args.purpose is not None and not args.purpose.strip():
        parser.error("Purpose must contain a nonempty description")
    return args


def _list(value, kind=str):
    return [kind(item.strip()) for item in value.split(",")] if isinstance(value, str) else list(value)


def prepare_configuration(args):
    """Validate selectors and filter both children before starting a bridge."""
    config = deepcopy(json.loads(args.config.read_text()))
    databases = config.get("databases", {})
    unknown = set(args.dataset) - set(databases)
    if unknown:
        raise ValueError(f"Unknown configured datasets: {sorted(unknown)}")
    config["databases"] = {name: databases[name] for name in args.dataset}
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    known = {row["id"]: row for row in rows}
    unknown = set(args.query_id) - set(known)
    if unknown:
        raise ValueError(f"Unknown query IDs: {sorted(unknown)}")
    selected = [known[query_id] for query_id in args.query_id]
    benchmarks = {row["benchmark"] for row in selected}
    selected_benchmarks = {database.get("benchmark", name) for name, database in config["databases"].items()}
    if benchmarks != selected_benchmarks:
        raise ValueError("Every selected query benchmark and dataset must have a matching selected counterpart")
    budgets = _list(args.budgets if args.budgets is not None else config.get("budgets", [1]), int)
    ablations = _list(args.ablations if args.ablations is not None else config.get("ablations", ["full"]))
    if not budgets or len(set(budgets)) != len(budgets) or any(type(n) is not int or not 1 <= n <= 16 for n in budgets):
        raise ValueError("Budgets must be distinct integers between 1 and 16")
    if not ablations or len(set(ablations)) != len(ablations) or set(ablations) - set(ABLATIONS):
        raise ValueError("Unknown or repeated ablation selection")
    config.update(budgets=budgets, ablations=ablations)
    purpose = args.purpose if args.purpose is not None else config.get("purpose", "functional session integration probe")
    config.update(purpose=purpose, final_performance_claim_allowed=False)
    if not isinstance(config.get("model"), dict):
        raise ValueError("Session configuration needs a model mapping")
    for database in config["databases"].values():
        database["pg_hint_plan"] = True
    if args.seed_catalog:
        for number, line in enumerate(args.seed_catalog.read_text().splitlines(), 1):
            if line.strip() and not isinstance(json.loads(line), dict):
                raise ValueError(f"Seed catalog line {number} is not an object")
    tasks = sum(1 for row in selected for name, db in config["databases"].items()
                if db.get("benchmark", name) == row["benchmark"])
    native_inputs = {(name, row["benchmark"], row["sha256"], row["schema_sha256"])
                     for row in selected for name, db in config["databases"].items()
                     if db.get("benchmark", name) == row["benchmark"]}
    plan = {"datasets": args.dataset, "query_ids": args.query_id, "budgets": budgets, "ablations": ablations,
            "native_systems": [] if args.skip_native else args.system,
            "planned_native_executions": 0 if args.skip_native else len(native_inputs) * len(args.system),
            "planned_hybrid_tasks": 0 if args.skip_hybrid else tasks * len(budgets) * len(ablations),
            "skip_native": args.skip_native, "skip_hybrid": args.skip_hybrid,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    return config, plan


def session_environment(base_url, spool, rbot_runtime=None, *, parent=None):
    """Create one environment-only secret for localhost clients and bridge."""
    environment = dict(os.environ if parent is None else parent)
    secret = secrets.token_urlsafe(48)
    environment.update(MCTS_SESSION_BRIDGE_SECRET=secret, MCTS_MODEL_API_KEY=secret,
                       MCTS_MODEL_TRANSPORT="codex_session_file_spool",
                       MCTS_SESSION_BRIDGE_SPOOL=str(spool), MCTS_PG_HINT_PLAN="1", OPENAI_TIMEOUT="600",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("MCTS_MODEL_FILE", None)
    environment.pop("MCTS_SESSION_BRIDGE_RUN_ID", None)
    environment.pop("RBOT_LOCAL_EMBEDDING_RUNTIME", None)
    bypass = ",".join(filter(None, [environment.get("NO_PROXY", environment.get("no_proxy", "")),
                                   "localhost,127.0.0.1,::1"]))
    environment["NO_PROXY"] = environment["no_proxy"] = bypass
    environment.update(OPENAI_API_KEY=secret, OPENAI_API_BASE=base_url + "/v1", OPENAI_BASE_URL=base_url + "/v1")
    if rbot_runtime:
        environment["RBOT_LOCAL_EMBEDDING_RUNTIME"] = str(rbot_runtime.resolve())
    for role in ("REASONING_MODEL", "DECISION_MODEL", "ASSISTANT_MODEL"):
        environment.update({role: "gpt-6.1-sol", role + "_URL": base_url + "/v1", role + "_API_KEY": secret})
    return environment


def build_commands(args, config_path, output, plan):
    common = ["--config", str(config_path), "--manifest", str(args.manifest.resolve())]
    queries = [part for query in plan["query_ids"] for part in ("--query-id", query)]
    datasets = [part for dataset in plan["datasets"] for part in ("--dataset", dataset)]
    commands = [("native_" + system, [sys.executable, str(ROOT / "scripts/generate_fresh_baselines.py"),
                 *common, "--output", str(output / "fresh_baselines.jsonl"), "--system", system,
                 *datasets, *queries, "--timeout", str(args.native_timeout)])
                for system in plan["native_systems"]]
    if not args.skip_hybrid:
        commands.append(("hybrid", [sys.executable, str(ROOT / "benchmark_runner.py"), "run", *common,
                         "--output", str(output / "hybrid"), "--mode", "hybrid",
                         "--budgets", ",".join(map(str, plan["budgets"])),
                         "--ablations", ",".join(plan["ablations"]), *queries]))
    return commands


def checkpoint(path, status):
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args):
    config, plan = prepare_configuration(args)
    output = args.output.resolve()
    spool = ROOT / ".runtime/session_model" / args.session_id
    if spool.exists():
        raise RuntimeError("Choose a new session ID; existing worker state is immutable")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
        port = reservation.getsockname()[1]
    output.mkdir(parents=True, exist_ok=False)
    base_url = f"http://127.0.0.1:{port}"
    environment = session_environment(base_url, spool, args.rbot_local_runtime)
    config["model"] = {"type": "http", "model": "gpt-6.1-sol",
                       "endpoint": base_url + "/runs/hybrid_probe/v1/chat/completions",
                       "api_key_env": "MCTS_SESSION_BRIDGE_SECRET", "timeout": 600}
    roles = {role: "gpt-6.1-sol" for role in ("rule_selection", "macro_rewrite", "cost_critic")}
    config["model_context"] = {
        "transport": "codex_session_file_spool", "actual_serving_model": "gpt-6.1-sol",
        "reasoning_effort": "xhigh", "requested_role_models": roles,
        "actual_role_models": dict(roles), "model_substitution": False,
        "published_model_configuration_reproduced": False, "sampling_controls_reproduced": False,
        "requested_sampling_options_not_enforced": ["temperature", "top_p", "seed", "max_tokens"], "usage_known": False,
        "role_mapping_scope": "configured provider; per-task traces determine actual successful calls",
        "embedding_source": {"provider": "local_bm25", "substituted": False,
                             "description": "hybrid lexical BM25 over pinned upstream rules/knowledge; no dense embedding model"},
        "requires_active_session_workers": True,
    }
    config.update(baseline_policy="fresh", baseline_catalog=str(output / "fresh_baselines.jsonl"))
    if args.seed_catalog:
        shutil.copy2(args.seed_catalog, output / "fresh_baselines.jsonl")
    else:
        (output / "fresh_baselines.jsonl").touch()
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    commands = build_commands(args, config_path, output, plan)
    status = {"purpose": config["purpose"], "spool": str(spool), "stages": {},
              "launch_mode": "batch" if args.batch else "probe", "selection": plan,
              "secret_persisted": False, "model_workers": "provided by active supervising session",
              "client_transport_timeout_seconds": 600, "native_systems": plan["native_systems"],
              "seed_catalog": str(args.seed_catalog.resolve()) if args.seed_catalog else None,
              "seed_catalog_sha256": hashlib.sha256(args.seed_catalog.read_bytes()).hexdigest() if args.seed_catalog else None,
              "launcher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "final_performance_claim_allowed": False}
    status_path = output / "probe_status.json"
    print(json.dumps({"spool": str(spool), "output": str(output), "selection": plan,
                      "requires_active_session_workers": True, "final_performance_claim_allowed": False}), flush=True)
    checkpoint(status_path, status)
    with (output / "bridge.log").open("w") as log:
        bridge = subprocess.Popen([sys.executable, str(ROOT / "scripts/session_model_bridge.py"), "serve",
                                   "--spool", str(spool.parent), "--session-id", args.session_id,
                                   "--port", str(port), "--timeout", "600"],
                                  cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            health_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if bridge.poll() is not None:
                    raise RuntimeError("Session bridge exited during startup; inspect bridge.log")
                try:
                    request = urllib.request.Request(base_url + "/health",
                        headers={"Authorization": "Bearer " + environment["MCTS_SESSION_BRIDGE_SECRET"]})
                    with health_opener.open(request, timeout=2) as response:
                        health = json.load(response)
                    if health.get("session_id") != args.session_id:
                        raise RuntimeError("Unexpected bridge session at configured port")
                    break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Bridge did not become ready")
                    time.sleep(0.25)
            for name, command in commands:
                status["stages"][name] = {"status": "running", "command": command}
                checkpoint(status_path, status)
                print(json.dumps({"stage": name, "status": "started"}), flush=True)
                with (output / (name + ".log")).open("w") as stage_log:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stage_log, stderr=subprocess.STDOUT)
                status["stages"][name].update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
                checkpoint(status_path, status)
            status.update(status="execution_finished", targets_established=False)
            checkpoint(status_path, status)
        except Exception as error:
            status.update(status="error", error=f"{type(error).__name__}: {error}")
            checkpoint(status_path, status)
            raise
        finally:
            if spool.exists():
                (spool / "STOP").touch()
            bridge.terminate()
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait()
    return 1 if any(stage.get("exit_code") for stage in status["stages"].values()) else 0


def main(argv=None, *, batch=False):
    return run(parse_args(argv, batch=batch))


if __name__ == "__main__":
    raise SystemExit(main())
