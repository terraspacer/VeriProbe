import importlib.util
import json
from pathlib import Path
import signal
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_when_ready", ROOT / "scripts/run_when_ready.py")
ready = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ready)


@pytest.fixture
def args(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"databases": {}, "test_scope": "mocked readiness only"}))
    batch = tmp_path / "batch"
    return ready.parse_args(["--config", str(config), "--output", str(tmp_path / "readiness"),
                             "--wait-timeout", "60", "--", "python", "never-run.py",
                             "--literal", "a b; $(do-not-execute)", "--output", str(batch)])


class FinishedProcess:
    pid = 123456
    returncode = 7

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def test_actual_suite_gate_polls_then_launches_exact_argv_without_shell(args, monkeypatch):
    reasons = iter([["dataset not ready", "maintenance active"], []])
    calls, sleeps = [], []
    monkeypatch.setattr(ready.run_suite, "blockers", lambda config: next(reasons))
    monkeypatch.setattr(ready.run_suite.time, "sleep", sleeps.append)

    def start(command, **kwargs):
        assert json.loads((args.output / "readiness.json").read_text())["status"] == "ready"
        assert not Path(args.command[-1]).exists()
        calls.append((command, kwargs))
        return FinishedProcess()

    monkeypatch.setattr(ready.subprocess, "Popen", start)
    assert ready.run(args) == 7
    assert len(calls) == 1 and calls[0][0] == args.command
    assert calls[0][1] == {"cwd": ready.ROOT, "shell": False, "start_new_session": True}
    assert sleeps == [30]
    history = [json.loads(line) for line in (args.output / "readiness_events.jsonl").read_text().splitlines()]
    assert [r["status"] for r in history] == ["waiting", "ready", "executing", "failed"]
    state = history[-1]
    assert state["exit_code"] == state["wrapper_exit_code"] == 7
    assert state["config_sha256_before_wait"] == state["config_sha256_after_wait"]
    conditions = [json.loads(line) for line in (args.output / "suite_conditions.jsonl").read_text().splitlines()]
    assert conditions[0]["blockers"] and conditions[-1]["blockers"] == []
    assert not Path(args.command[-1]).exists()


def test_config_byte_change_even_whitespace_prevents_child(args, monkeypatch):
    def wait(config, output, enabled, timeout):
        args.config.write_bytes(args.config.read_bytes() + b"\n")

    monkeypatch.setattr(ready.run_suite, "wait_for_ready", wait)
    monkeypatch.setattr(ready.subprocess, "Popen", lambda *a, **k: pytest.fail("Child must not launch"))
    assert ready.run(args) == 2
    state = json.loads((args.output / "readiness.json").read_text())
    assert state["status"] == "config_changed" and not state["command_started"]
    assert state["config_sha256_before_wait"] != state["config_sha256_after_wait"]


def test_wait_timeout_preserves_blockers_and_never_creates_batch_dir(args, monkeypatch):
    args.wait_timeout = 0
    monkeypatch.setattr(ready.run_suite, "blockers", lambda config: ["maintenance active"])
    monkeypatch.setattr(ready.subprocess, "Popen", lambda *a, **k: pytest.fail("Child must not launch"))
    assert ready.run(args) == 2
    state = json.loads((args.output / "readiness.json").read_text())
    assert state["status"] == "wait_failed" and "maintenance active" in state["error"]
    assert not state["command_started"] and state["exit_code"] is None
    assert not Path(args.command[-1]).exists()


def test_interrupt_during_wait_leaves_no_child_and_restores_handlers(args, monkeypatch):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def wait(*a):
        raise ready.CommandInterrupted(signal.SIGTERM)

    monkeypatch.setattr(ready.run_suite, "wait_for_ready", wait)
    monkeypatch.setattr(ready.subprocess, "Popen", lambda *a, **k: pytest.fail("Child must not launch"))
    assert ready.run(args) == 143
    state = json.loads((args.output / "readiness.json").read_text())
    assert state["status"] == "interrupted" and not state["command_started"] and state["cleanup_complete"]
    assert previous == {sig: signal.getsignal(sig) for sig in previous}


def test_interrupt_after_spawn_is_gentle_then_bounded_escalation(args, monkeypatch):
    monkeypatch.setattr(ready.run_suite, "wait_for_ready", lambda *a: None)
    signals = []

    class Process:
        pid = 123456
        returncode = None
        waits = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.waits += 1
            if timeout is None:
                raise ready.CommandInterrupted(signal.SIGTERM)
            assert timeout == 5
            if self.waits < 4:
                raise subprocess.TimeoutExpired(args.command, timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(ready.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(ready.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert ready.run(args) == 143
    assert signals == [(123456, s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)]
    state = json.loads((args.output / "readiness.json").read_text())
    assert state["exit_code"] == -9 and state["cleanup_complete"]
    assert state["signal"] == signal.SIGTERM and state["wrapper_exit_code"] == 143


def test_child_signal_exit_keeps_raw_code_and_conventional_wrapper_status(args, monkeypatch):
    monkeypatch.setattr(ready.run_suite, "wait_for_ready", lambda *a: None)
    process = FinishedProcess()
    process.returncode = -signal.SIGTERM
    monkeypatch.setattr(ready.subprocess, "Popen", lambda *a, **k: process)
    assert ready.run(args) == 143
    state = json.loads((args.output / "readiness.json").read_text())
    assert state["exit_code"] == -15 and state["wrapper_exit_code"] == 143


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1"])
def test_invalid_wait_timeout_rejected(tmp_path, timeout):
    with pytest.raises(SystemExit):
        ready.parse_args(["--config", "config.json", "--output", str(tmp_path / "audit"),
                          "--wait-timeout", timeout, "--", "true"])


def test_missing_command_and_overlapping_output_rejected(tmp_path):
    with pytest.raises(SystemExit):
        ready.parse_args(["--config", "config.json", "--output", str(tmp_path / "audit")])
    for audit, batch in ((tmp_path / "same", tmp_path / "same"),
                         (tmp_path / "batch/audit", tmp_path / "batch"),
                         (tmp_path / "audit", tmp_path / "audit/batch")):
        with pytest.raises(SystemExit):
            ready.parse_args(["--config", "config.json", "--output", str(audit), "--",
                              "python", "batch.py", "--output=" + str(batch)])
    assert not (tmp_path / "batch").exists()
