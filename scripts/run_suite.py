#!/usr/bin/env python3
"""Sequential, resumable main/scaling/ablation orchestration after data setup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.host_conditions import observe_host

STAGES = (
    ("main_m16", "16", "full"),
    ("scaling", "1,2,4,8", "full"),
    ("ablations", "16", "class_a_only,class_b_only,no_plan,no_rag,no_critic,no_safety"),
)


def emit(**fields):
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **fields}), flush=True)


def blockers(config):
    first = next(iter(config.get("databases", {}).values()), None)
    observation = observe_host(config, first)
    reasons = [*observation["reasons"], *observation["probe_errors"]]
    for key, spec in config["databases"].items():
        path = ROOT / spec["provenance"]["readiness_report"]
        if not path.exists() or not json.loads(path.read_text()).get("ready"):
            reasons.append("dataset not ready: " + key)
    return reasons


def wait_for_ready(config, output, wait, timeout):
    if timeout < 0 or not math.isfinite(timeout):
        raise ValueError("Suite wait timeout must be finite and nonnegative")
    started = time.monotonic()
    while True:
        reasons = blockers(config)
        event = {"time": datetime.now(timezone.utc).isoformat(), "stage": "preflight",
                 "elapsed_seconds": time.monotonic() - started, "blockers": reasons}
        with (output / "suite_conditions.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
        if not reasons:
            return
        if not wait or event["elapsed_seconds"] >= timeout:
            raise RuntimeError("Suite preflight blocked; pending stages preserved: " + "; ".join(reasons))
        emit(state="waiting", blockers=reasons)
        time.sleep(min(30, timeout - event["elapsed_seconds"]))


def diagnose_main(output, raw_path, environment):
    destination = output / "diagnosis_main_m16"
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "scripts/diagnose_misses.py"), "--raw", str(raw_path.resolve()),
               "--output", str(destination.resolve()), "--propose-patterns", str((destination / "draft_patterns.jsonl").resolve()),
               "--tuning-scope", "transductive"]
    with (destination / "process.log").open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False, stdout=stream, stderr=subprocess.STDOUT)
    return {"command": command, "exit_code": result.returncode,
            "status": "completed" if result.returncode == 0 else "failed", "output": str(destination.resolve()),
            "scope": "transductive feedback drafts for a future iteration; no promotion or library change",
            "primary_evaluation": "Preserved as the original frozen evaluation. Later reuse of consumed queries requires an adapted-result label and fresh untouched timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/published-hybrid.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/published")
    parser.add_argument("--stages", default=",".join(stage[0] for stage in STAGES))
    parser.add_argument("--wait", action="store_true", help="Wait for ready data and no competing setup/validation service")
    parser.add_argument("--wait-timeout", type=float, help="Bound initial readiness wait; defaults to 24 hours")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frozen_paths = [args.config.resolve(), ROOT / "prompts/action_patterns.jsonl"]
    model = config.get("model")
    model_file = model if isinstance(model, str) else model.get("file", model.get("path")) if isinstance(model, dict) else None
    if model_file:
        frozen_paths.append(ROOT / model_file)
    frozen = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None for path in frozen_paths}
    names = args.stages.split(",")
    if set(names) - {stage[0] for stage in STAGES}:
        raise ValueError("Unknown suite stage")
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, budgets, ablations in STAGES:
        if name not in names:
            continue
        destination = args.output / name
        command = [sys.executable, str(ROOT / "benchmark_runner.py"), "run", "--config", str(args.config.resolve()),
                   "--output", str(destination.resolve()), "--mode", config.get("verification_mode", "hybrid"),
                   "--budgets", budgets, "--ablations", ablations]
        if (destination / "run.json").exists():
            command.append("--resume")
        commands.append((name, command))
    if args.dry_run:
        emit(commands=commands, blockers=blockers(config))
        return
    wait_for_ready(config, args.output, args.wait,
                   args.wait_timeout if args.wait_timeout is not None else config.get("suite_wait_timeout_seconds", 86400))
    environment = dict(os.environ, MCTS_PG_HINT_PLAN="1")
    state = {"status": "running", "config": str(args.config.resolve()), "stages": {}, "frozen_inputs": frozen, "diagnostics": {}}
    state_path = args.output / "suite.json"
    for name, command in commands:
        current = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None for path in frozen_paths}
        if current != frozen:
            raise RuntimeError("Suite configuration, model file, or approved pattern library changed; pending stages were preserved")
        emit(stage=name, state="started", command=command)
        state["stages"][name] = {"status": "running", "command": command}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        state["stages"][name]["exit_code"] = result.returncode
        state["stages"][name]["status"] = "completed" if result.returncode == 0 else "failed"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}; preserved outputs for inspection")
        if name == "main_m16":
            state["diagnostics"][name] = diagnose_main(args.output, args.output / name / "raw.jsonl", environment)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
    state["status"] = "execution_completed"
    state["note"] = "Completion of commands is not achievement of the performance targets; inspect main_m16/acceptance.json."
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    emit(**state)


if __name__ == "__main__":
    main()
