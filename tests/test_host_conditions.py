import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from mcts_rewrite.host_conditions import COMPETING_UNITS, HostQuietTimeout, TaskHostMonitor, observe_host
from scripts import run_suite


def observation(quiet=True):
    return {"time": "2026-10-26T00:00:00+00:00", "quiet": quiet,
            "reasons": ["memory pressure"] if quiet is False else [],
            "probe_errors": ["probe unavailable"] if quiet is None else []}


class FakeClock:
    def __init__(self):
        self.value, self.sleeps = 0, []
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_development_tasks_record_boundaries_without_waiting(tmp_path):
    clock = FakeClock()
    monitor = TaskHostMonitor({"profile": "development", "wait_for_quiet_before_task": True},
        "hard-query", "run", tmp_path / "host.jsonl", observer=lambda: observation(False),
        clock=clock.now, sleeper=clock.sleep)
    with monitor:
        pass
    summary = monitor.summary()
    assert summary["status"] == "pressure_observed"
    assert summary["pre_task_wait"]["enabled"] is False
    assert not summary["periodic_monitoring_enabled"]
    assert clock.sleeps == []
    events = [json.loads(line) for line in monitor.log_path.read_text().splitlines()]
    assert [event["phase"] for event in events] == ["task_start", "task_end"]
    assert all(event["task_id"] == "hard-query" for event in events)


def test_published_wait_is_bounded_and_keeps_the_task_pending(tmp_path):
    clock = FakeClock()
    monitor = TaskHostMonitor({"profile": "published", "wait_for_quiet_before_task": True,
        "quiet_wait_timeout_seconds": 2, "quiet_poll_seconds": 1}, "hard-query", "run",
        tmp_path / "host.jsonl", observer=lambda: observation(False), clock=clock.now, sleeper=clock.sleep)
    with pytest.raises(HostQuietTimeout, match="task remains pending") as failure:
        with monitor:
            pytest.fail("The blocked task must not execute or become completed")
    assert failure.value.summary["pre_task_wait"]["status"] == "timeout_task_pending"
    assert failure.value.summary["task_observations"] == 0
    assert clock.sleeps == [1, 1]
    assert len(monitor.log_path.read_text().splitlines()) == 3


def test_wait_history_is_distinct_from_pressure_during_executed_task(tmp_path):
    clock = FakeClock()
    samples = iter([observation(False), observation(True), observation(True), observation(False)])
    monitor = TaskHostMonitor({"profile": "published", "wait_for_quiet_before_task": True,
        "host_pressure_sample_seconds": 0, "quiet_poll_seconds": 1}, "q", "run",
        tmp_path / "host.jsonl", observer=lambda: next(samples), clock=clock.now, sleeper=clock.sleep)
    with monitor:
        pass
    assert monitor.summary()["pre_task_wait"]["status"] == "quiet_observed"
    assert monitor.summary()["pre_task_wait"]["seconds"] == 1
    assert monitor.summary()["status"] == "pressure_observed"
    assert monitor.summary()["quiet_at_observed_samples"] is False


def test_periodic_monitor_preserves_midtask_pressure_even_if_boundaries_are_quiet(tmp_path):
    sampled = threading.Event()
    calls = []
    def observer():
        calls.append(1)
        if len(calls) == 2:
            sampled.set()
            return observation(False)
        return observation(True)
    monitor = TaskHostMonitor({"profile": "published", "host_pressure_sample_seconds": 1},
        "q", "run", tmp_path / "host.jsonl", observer=observer)
    with monitor:
        assert sampled.wait(3)
    assert monitor.summary()["status"] == "pressure_observed"
    assert monitor.summary()["during_task_observations"] >= 1
    assert "memory pressure" in monitor.summary()["observed_reasons"]


def test_missing_probe_data_cannot_support_quiet_claim(tmp_path):
    monitor = TaskHostMonitor({}, "q", "run", tmp_path / "host.jsonl", observer=lambda: observation(None))
    with monitor:
        pass
    assert monitor.summary()["status"] == "unknown"
    assert monitor.summary()["quiet_at_observed_samples"] is None


def test_observer_records_cpu_load_without_treating_own_work_as_external_interference(monkeypatch):
    def read_text(path, *args, **kwargs):
        assert str(path).startswith("/proc/pressure/")
        return "some avg10=95 avg60=0 avg300=0 total=100\nfull avg10=0 avg60=0 avg300=0 total=0\n"
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("mcts_rewrite.host_conditions.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="inactive\n" * len(COMPETING_UNITS)))
    sample = observe_host({})
    assert sample["pressure"]["cpu"]["some"]["avg10"] == 95
    assert sample["quiet"] is True
    constrained = observe_host({"maximum_idle_cpu_pressure_percent": 50})
    assert constrained["quiet"] is False


@pytest.fixture
def quiet_host_with_database_probe(monkeypatch):
    import psycopg

    original_read_text = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path).startswith("/proc/pressure/"):
            return "some avg10=0 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n"
        return original_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("mcts_rewrite.host_conditions.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="inactive\n" * len(COMPETING_UNITS)))
    calls = []
    counts = {"workers": 0, "unavailable": 0}

    class Connection:
        def __enter__(self):
            calls.append("entered")
            return self
        def __exit__(self, *_):
            calls.append("closed")
        def execute(self, sql):
            assert "pg_stat_activity" in sql
            assert "backend_type IS NULL" in sql
            assert sql.count("FILTER") == 2
            calls.append("queried")
            return SimpleNamespace(fetchone=lambda: (counts["workers"], counts["unavailable"]))

    def connect(dsn, **kwargs):
        assert dsn == "mock read-role connection"
        assert kwargs["connect_timeout"] == 2
        assert "statement_timeout=1000" in kwargs["options"]
        assert "default_transaction_read_only=on" in kwargs["options"]
        assert kwargs["application_name"] == "mcts_host_observer"
        return Connection()
    monkeypatch.setattr(psycopg, "connect", connect)
    return counts, calls


def test_masked_backend_types_cannot_be_reported_as_zero_workers_or_a_quiet_host(quiet_host_with_database_probe):
    counts, calls = quiet_host_with_database_probe
    counts["unavailable"] = 8
    sample = observe_host({}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is None
    assert sample["database_activity"] == {"autovacuum_workers": None,
        "observed_autovacuum_workers": 0, "backend_types_unavailable": 8, "visibility_complete": False}
    assert any(error.startswith("database activity visibility incomplete") for error in sample["probe_errors"])
    assert sample["reasons"] == []
    assert calls == ["entered", "queried", "closed"]


@pytest.mark.parametrize("workers", [0, 3])
def test_visible_backend_types_distinguish_quiet_and_active_workers(quiet_host_with_database_probe, workers):
    counts, calls = quiet_host_with_database_probe
    counts["workers"] = workers
    sample = observe_host({}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is (workers == 0)
    assert sample["database_activity"] == {"autovacuum_workers": workers,
        "observed_autovacuum_workers": workers, "backend_types_unavailable": 0, "visibility_complete": True}
    assert sample["probe_errors"] == []
    assert sample["reasons"] == ([f"autovacuum workers active: {workers}"] if workers else [])
    assert calls == ["entered", "queried", "closed"]


def test_visible_workers_still_report_pressure_when_other_backends_are_masked(quiet_host_with_database_probe):
    counts, _ = quiet_host_with_database_probe
    counts.update(workers=2, unavailable=8)
    sample = observe_host({}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is False
    assert sample["database_activity"]["autovacuum_workers"] is None
    assert sample["database_activity"]["observed_autovacuum_workers"] == 2
    assert "autovacuum workers active: 2" in sample["reasons"]
    assert sample["probe_errors"]


def test_masked_database_activity_keeps_published_task_pending(quiet_host_with_database_probe, tmp_path):
    counts, _ = quiet_host_with_database_probe
    counts["unavailable"] = 8
    clock = FakeClock()
    monitor = TaskHostMonitor({"profile": "published", "wait_for_quiet_before_task": True,
        "quiet_wait_timeout_seconds": 2, "quiet_poll_seconds": 1}, "q", "run",
        tmp_path / "host.jsonl", database={"dsn": "mock read-role connection"},
        clock=clock.now, sleeper=clock.sleep)
    with pytest.raises(HostQuietTimeout) as failure:
        with monitor:
            pytest.fail("Masked activity must not release the pending task")
    assert failure.value.summary["task_observations"] == 0
    assert failure.value.summary["pre_task_wait"]["status"] == "timeout_task_pending"
    assert failure.value.summary["quiet_at_observed_samples"] is None
    events = [json.loads(line) for line in monitor.log_path.read_text().splitlines()]
    assert all(event["phase"] == "pre_task_wait" and event["quiet"] is None for event in events)
    assert all(any(error.startswith("database activity visibility incomplete")
                   for error in event["probe_errors"]) for event in events)


def test_absent_maintenance_report_config_preserves_existing_quiet_probe(quiet_host_with_database_probe):
    sample = observe_host({}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is True
    assert sample["maintenance_readiness"] is None
    assert sample["probe_errors"] == []


@pytest.mark.parametrize("contents", [None, "{invalid json", "[]", "null", '"not an object"',
    '{"ready_for_timing":true,"settings_restored":true,"rows":NaN}'])
def test_missing_or_malformed_maintenance_report_fails_closed(quiet_host_with_database_probe, tmp_path, contents):
    path = tmp_path / "maintenance.json"
    if contents is not None:
        path.write_text(contents)
    sample = observe_host({"maintenance_readiness_report": str(path)}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is None
    assert sample["maintenance_readiness"]["status"] == "unknown"
    assert sample["maintenance_readiness"]["path"] == str(path)
    assert any(error.startswith("postload maintenance readiness report") for error in sample["probe_errors"])


@pytest.mark.parametrize("flags", [
    {}, {"ready_for_timing": True}, {"settings_restored": True},
    {"ready_for_timing": None, "settings_restored": True},
    {"ready_for_timing": 1, "settings_restored": True},
    {"ready_for_timing": "true", "settings_restored": True},
    {"ready_for_timing": True, "settings_restored": 1},
    {"ready_for_timing": True, "settings_restored": "true"},
])
def test_maintenance_report_requires_explicit_boolean_flags(quiet_host_with_database_probe, tmp_path, flags):
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps(flags))
    sample = observe_host({"maintenance_readiness_report": str(path)}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is None
    assert sample["maintenance_readiness"]["status"] == "unknown"
    assert "readiness flags must be booleans" in sample["probe_errors"][0]


@pytest.mark.parametrize("path", [None, False, 1, [], {}, ""])
def test_maintenance_report_path_types_fail_closed(quiet_host_with_database_probe, path):
    sample = observe_host({"maintenance_readiness_report": path}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is None
    assert sample["maintenance_readiness"]["status"] == "unknown"
    assert sample["maintenance_readiness"]["path"] is None
    assert "invalid path" in sample["probe_errors"][0]


@pytest.mark.parametrize("ready, restored", [(False, False), (False, True), (True, False)])
def test_maintenance_report_blocks_pending_work_or_unrestored_settings(quiet_host_with_database_probe, tmp_path, ready, restored):
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps({"ready_for_timing": ready, "settings_restored": restored}))
    sample = observe_host({"maintenance_readiness_report": str(path)}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is False
    assert sample["maintenance_readiness"]["status"] == "pending"
    assert sample["probe_errors"] == []
    assert "postload maintenance pending" in sample["reasons"]


def test_ready_maintenance_report_resolves_from_repo_and_records_only_status(quiet_host_with_database_probe, monkeypatch, tmp_path):
    monkeypatch.setattr("mcts_rewrite.host_conditions.ROOT", tmp_path)
    path = tmp_path / "benchmarks/maintenance.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"ready_for_timing": True, "settings_restored": True,
        "observed": "2026-10-27T00:00:00Z", "tables": [{"private": "must not be logged"}],
        "dsn": "must not be logged"}))
    sample = observe_host({"maintenance_readiness_report": "benchmarks/maintenance.json"},
                          {"dsn": "mock read-role connection"})
    assert sample["quiet"] is True
    assert sample["maintenance_readiness"] == {"status": "ready", "path": str(path),
        "ready_for_timing": True, "settings_restored": True, "report_observed_at": "2026-10-27T00:00:00+00:00"}
    assert "must not be logged" not in json.dumps(sample)


@pytest.mark.parametrize("report_contents", [
    {"ready_for_timing": True, "settings_restored": True},
    {"ready_for_timing": False, "settings_restored": False},
    {},
])
def test_maintenance_report_never_skips_live_worker_and_pressure_probes(quiet_host_with_database_probe, monkeypatch, tmp_path, report_contents):
    counts, calls = quiet_host_with_database_probe
    counts["workers"] = 3
    original_read_text = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/pressure/memory":
            return "some avg10=5 avg60=0 avg300=0 total=0\nfull avg10=5 avg60=0 avg300=0 total=0\n"
        return original_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps(report_contents))
    sample = observe_host({"maintenance_readiness_report": str(path)}, {"dsn": "mock read-role connection"})
    assert sample["quiet"] is False
    assert "autovacuum workers active: 3" in sample["reasons"]
    assert any(reason.startswith("memory full avg10 pressure exceeds") for reason in sample["reasons"])
    assert calls == ["entered", "queried", "closed"]


def test_zero_worker_interval_waits_for_maintenance_completion_receipt(quiet_host_with_database_probe, tmp_path):
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps({"ready_for_timing": False, "settings_restored": True}))
    clock = FakeClock()
    def sleep_and_complete(seconds):
        clock.sleep(seconds)
        path.write_text(json.dumps({"ready_for_timing": True, "settings_restored": True}))
    monitor = TaskHostMonitor({"profile": "published", "wait_for_quiet_before_task": True,
        "host_pressure_sample_seconds": 0, "quiet_wait_timeout_seconds": 2, "quiet_poll_seconds": 1,
        "maintenance_readiness_report": str(path)}, "q", "run", tmp_path / "host.jsonl",
        database={"dsn": "mock read-role connection"}, clock=clock.now, sleeper=sleep_and_complete)
    with monitor:
        assert clock.value == 1
    assert monitor.summary()["pre_task_wait"]["status"] == "quiet_observed"
    events = [json.loads(line) for line in monitor.log_path.read_text().splitlines()]
    assert events[0]["database_activity"]["autovacuum_workers"] == 0
    assert events[0]["maintenance_readiness"]["status"] == "pending"
    assert events[0]["quiet"] is False
    assert events[1]["maintenance_readiness"]["status"] == "ready"
    assert events[1]["quiet"] is True


def test_suite_wait_timeout_is_persisted_and_does_not_execute_stages(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(run_suite, "blockers", lambda config: ["competing task active"])
    monkeypatch.setattr(run_suite.time, "monotonic", clock.now)
    monkeypatch.setattr(run_suite.time, "sleep", clock.sleep)
    with pytest.raises(RuntimeError, match="pending stages preserved"):
        run_suite.wait_for_ready({}, tmp_path, True, 2)
    events = [json.loads(line) for line in (tmp_path / "suite_conditions.jsonl").read_text().splitlines()]
    assert len(events) == 2 and events[-1]["elapsed_seconds"] == 2


def test_post_main_diagnosis_creates_only_separate_transductive_drafts(monkeypatch, tmp_path):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_suite.subprocess, "run", run)
    result = run_suite.diagnose_main(tmp_path, tmp_path / "main_m16/raw.jsonl", {})
    command = commands[0]
    assert "--tuning-scope" in command and command[command.index("--tuning-scope") + 1] == "transductive"
    assert "--promote-validated" not in command and "--library" not in command
    assert str(tmp_path / "diagnosis_main_m16/draft_patterns.jsonl") in command
    assert result["exit_code"] == 0
