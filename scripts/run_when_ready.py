#!/usr/bin/env python3
"""Wait for the suite readiness/quiet gates before starting an exact argv command.

The --output directory holds only readiness audit files and must be distinct
from any child --output directory. No bridge or benchmark fingerprints are
created until readiness succeeds and the config bytes are rechecked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_suite


class CommandInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New readiness audit directory, separate from batch output")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Exact command argv after --; never evaluated by a shell")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("Supply a command after --")
    if args.wait_timeout < 0 or not math.isfinite(args.wait_timeout):
        parser.error("Wait timeout must be finite and nonnegative")
    audit = args.output.resolve()
    # In particular, mkdir(parents=True) must not create the future batch
    # directory as an ancestor of the readiness audit directory.
    child_outputs = []
    for index, item in enumerate(args.command):
        if item == "--output" and index + 1 < len(args.command):
            child_outputs.append(args.command[index + 1])
        elif item.startswith("--output="):
            child_outputs.append(item.split("=", 1)[1])
    for value in child_outputs:
        selected = Path(value)
        selected = selected.resolve() if selected.is_absolute() else (ROOT / selected).resolve()
        if audit == selected or audit in selected.parents or selected in audit.parents:
            parser.error("Readiness audit and child output directories must be separate and non-overlapping")
    return args


def checkpoint(output, state, status, **fields):
    state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    temporary = output / "readiness.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output / "readiness.json")
    with (output / "readiness_events.jsonl").open("a") as stream:
        stream.write(json.dumps(state, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "audit": str(output), **fields}), flush=True)


def stop_process_group(process):
    """Let Python launchers run finally blocks, then bound termination escalation."""
    sent = []
    if process.poll() is not None:
        return process.returncode, sent
    # SIGINT lets run_session_batch execute its bridge STOP/finally cleanup.
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
            sent.append(int(signum))
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=5), sent
        except subprocess.TimeoutExpired:
            continue
    return None, sent


def run(args):
    config_path = args.config.resolve()
    frozen_bytes = config_path.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    config = json.loads(frozen_bytes)
    if not isinstance(config, dict):
        raise ValueError("Readiness config must be a JSON object")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"config": str(config_path), "config_sha256_before_wait": frozen_hash,
             "config_sha256_after_wait": None, "wait_timeout_seconds": args.wait_timeout,
             "command": list(args.command), "command_cwd": str(ROOT), "command_started": False,
             "exit_code": None, "wrapper_exit_code": None,
             "conditions_log": "suite_conditions.jsonl",
             "readiness_scope": "Initial suite dataset/maintenance/host gate; no claim of quiet throughout execution"}
    process = None
    phase = "waiting"
    pending_signal = None

    def interrupted(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
        # Avoid losing a just-spawned child before Popen's result is assigned,
        # and let bounded group cleanup finish if another signal arrives.
        if phase not in ("launching", "stopping"):
            raise CommandInterrupted(signum)

    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous_handlers:
        signal.signal(signum, interrupted)
    try:
        checkpoint(output, state, "waiting")
        # This uses run_suite.blockers and its existing at-most-30-second poll.
        run_suite.wait_for_ready(config, output, True, args.wait_timeout)
        current_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        state["config_sha256_after_wait"] = current_hash
        if current_hash != frozen_hash:
            checkpoint(output, state, "config_changed", wrapper_exit_code=2,
                       error="Config bytes changed during readiness wait; command was not started")
            return 2
        checkpoint(output, state, "ready")
        phase = "launching"
        process = subprocess.Popen(list(args.command), cwd=ROOT, shell=False, start_new_session=True)
        phase = "executing"
        if pending_signal is not None:
            raise CommandInterrupted(pending_signal)
        checkpoint(output, state, "executing", command_started=True, pid=process.pid)
        code = process.wait()
        wrapper_code = code if code >= 0 else 128 - code
        checkpoint(output, state, "completed" if code == 0 else "failed",
                   exit_code=code, wrapper_exit_code=wrapper_code)
        return wrapper_code
    except CommandInterrupted as error:
        phase = "stopping"
        code, sent = stop_process_group(process) if process is not None else (None, [])
        checkpoint(output, state, "interrupted", command_started=process is not None,
                   signal=int(error.signum), cleanup_signals=sent, exit_code=code,
                   cleanup_complete=process is None or code is not None,
                   wrapper_exit_code=128 + error.signum)
        return 128 + error.signum
    except Exception as error:
        phase = "stopping"
        code, sent = stop_process_group(process) if process is not None else (None, [])
        checkpoint(output, state, "wait_failed" if process is None else "execution_error",
                   error=f"{type(error).__name__}: {error}", exit_code=code,
                   cleanup_signals=sent, wrapper_exit_code=2)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
