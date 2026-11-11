"""Sample benchmark host conditions without changing query results or cohorts.

Pressure includes the benchmark's own work. Samples can reveal interference or
resource contention; they cannot prove that the host was quiet between samples.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
COMPETING_UNITS = (
    "mcts-gen-dsb", "mcts-gen-tpch", "mcts-gen-calcite-uniform", "mcts-gen-calcite-zipf",
    "mcts-load-published", "mcts-attest-datasets", "mcts-validate-all",
)


def _setting(config, name, default, minimum=0, maximum=None):
    value = config.get(name, default)
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid host timing setting: {name}")
    return float(value)


def _maintenance_readiness(report_path):
    """Read only the readiness contract into the host observation."""
    status = {"path": None, "status": "unknown", "report_observed_at": None}
    if not isinstance(report_path, str) or not report_path.strip():
        return status, "postload maintenance readiness report unavailable: invalid path", None
    def invalid_constant(_):
        raise ValueError("non-finite JSON constant")
    try:
        path = Path(report_path)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        status["path"] = str(path)
        report = json.loads(path.read_text(), parse_constant=invalid_constant)
    except (OSError, ValueError, RuntimeError) as exc:
        return status, "postload maintenance readiness report unavailable: " + type(exc).__name__, None
    if not isinstance(report, dict):
        return status, "postload maintenance readiness report invalid: expected JSON object", None
    timestamp = report.get("observed")
    if isinstance(timestamp, str):
        try:
            status["report_observed_at"] = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass
    ready, restored = report.get("ready_for_timing"), report.get("settings_restored")
    if not isinstance(ready, bool) or not isinstance(restored, bool):
        return status, "postload maintenance readiness report invalid: readiness flags must be booleans", None
    status.update(ready_for_timing=ready, settings_restored=restored)
    if ready is True and restored is True:
        status["status"] = "ready"
        return status, None, None
    status["status"] = "pending"
    return status, None, "postload maintenance pending"


def observe_host(config, database=None):
    """Bounded probes; never include connection strings or process arguments."""
    pressure, errors, reasons = {}, [], []
    for resource in ("cpu", "memory", "io"):
        try:
            rows = {}
            for line in Path(f"/proc/pressure/{resource}").read_text().splitlines():
                kind, *fields = line.split()
                rows[kind] = {key: float(value) for key, value in (field.split("=", 1) for field in fields)}
            pressure[resource] = rows
        except (OSError, ValueError):
            errors.append(f"{resource} pressure probe unavailable")
    for resource, field, key, default in (
        ("memory", "full", "maximum_idle_memory_pressure_percent", 1.0),
        ("cpu", "some", "maximum_idle_cpu_pressure_percent", None),
        ("io", "full", "maximum_idle_io_pressure_percent", None),
    ):
        if default is None and key not in config:
            continue
        threshold = _setting(config, key, default, maximum=100)
        value = pressure.get(resource, {}).get(field, {}).get("avg10")
        if value is None:
            errors.append(f"{resource} quiet threshold cannot be checked")
        elif value > threshold:
            reasons.append(f"{resource} {field} avg10 pressure exceeds {threshold:g}%")
    units = list(dict.fromkeys([*COMPETING_UNITS, *config.get("host_competing_units", [])]))
    active = []
    try:
        probe = subprocess.run(["systemctl", "--user", "is-active", *units], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2, check=False)
        states = probe.stdout.splitlines()
        if len(states) != len(units):
            errors.append("competing-service probe unavailable")
        else:
            active = [unit for unit, state in zip(units, states) if state in {"active", "activating", "reloading"}]
            reasons.extend("competing task active: " + unit for unit in active)
    except (OSError, subprocess.SubprocessError):
        errors.append("competing-service probe unavailable")
    activity = None
    if database:
        try:
            import psycopg
            with psycopg.connect(database["dsn"], connect_timeout=2,
                                 options="-c statement_timeout=1000 -c default_transaction_read_only=on",
                                 application_name="mcts_host_observer") as connection:
                count, unavailable = connection.execute(
                    "SELECT count(*) FILTER (WHERE backend_type='autovacuum worker'), "
                    "count(*) FILTER (WHERE backend_type IS NULL) FROM pg_stat_activity").fetchone()
            # PostgreSQL can mask other roles' backend_type without raising an
            # error. A visible zero is insufficient when hidden rows may be
            # autovacuum workers. Do not present that lower count as a total.
            activity = {"autovacuum_workers": count if not unavailable else None,
                        "observed_autovacuum_workers": count,
                        "backend_types_unavailable": unavailable,
                        "visibility_complete": not bool(unavailable)}
            if unavailable:
                errors.append(f"database activity visibility incomplete: {unavailable} backend types unavailable")
            if count:
                reasons.append(f"autovacuum workers active: {count}")
        except Exception as exc:
            errors.append("database quiet-state probe unavailable: " + type(exc).__name__)
    maintenance = None
    if "maintenance_readiness_report" in config:
        maintenance, error, reason = _maintenance_readiness(config["maintenance_readiness_report"])
        if error:
            errors.append(error)
        if reason:
            reasons.append(reason)
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {"time": datetime.now(timezone.utc).isoformat(), "pressure": pressure,
            "load_average": load, "logical_cpus": os.cpu_count(), "active_competing_units": active,
            "database_activity": activity, "maintenance_readiness": maintenance,
            "reasons": reasons, "probe_errors": errors,
            "quiet": False if reasons else None if errors else True}


class HostQuietTimeout(RuntimeError):
    def __init__(self, summary):
        super().__init__("Bounded host quiet wait expired; task remains pending")
        self.summary = summary


class TaskHostMonitor:
    def __init__(self, config, task_id, run_hash, log_path, database=None, *, observer=None,
                 clock=time.monotonic, sleeper=time.sleep):
        self.config, self.task_id, self.run_hash = config, task_id, run_hash
        self.log_path, self.database = Path(log_path), database
        self.observer = observer or (lambda: observe_host(config, database))
        self.clock, self.sleep = clock, sleeper
        self.published = config.get("profile") == "published"
        self.wait_enabled = self.published and config.get("wait_for_quiet_before_task", False) is True
        self.interval = _setting(config, "host_pressure_sample_seconds", 10 if self.published else 0, maximum=60)
        if 0 < self.interval < 1:
            raise ValueError("host_pressure_sample_seconds must be zero or at least one second")
        self.wait_timeout = _setting(config, "quiet_wait_timeout_seconds", 600)
        self.poll = _setting(config, "quiet_poll_seconds", 15, minimum=1, maximum=30)
        self.samples, self.observer_errors = [], []
        self.wait_seconds, self.wait_status = 0.0, "disabled"
        self._stop, self._thread = threading.Event(), None
        self._origin = clock()

    def _sample(self, phase):
        try:
            sample = self.observer()
        except Exception as exc:
            sample = {"time": datetime.now(timezone.utc).isoformat(), "quiet": None,
                      "probe_errors": ["host observer failed: " + type(exc).__name__], "reasons": []}
        event = {"record_type": "host_observation", "run_hash": self.run_hash, "task_id": self.task_id,
                 "phase": phase, "elapsed_seconds": self.clock() - self._origin, **sample}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
        self.samples.append(event)
        return event

    def __enter__(self):
        if self.wait_enabled:
            self.wait_status = "waiting"
            started = self.clock()
            while True:
                sample = self._sample("pre_task_wait")
                self.wait_seconds = self.clock() - started
                if sample.get("quiet") is True:
                    self.wait_status = "quiet_observed"
                    break
                if self.wait_seconds >= self.wait_timeout:
                    self.wait_status = "timeout_task_pending"
                    raise HostQuietTimeout(self.summary())
                self.sleep(min(self.poll, self.wait_timeout - self.wait_seconds))
        self._sample("task_start")
        if self.interval:
            self._thread = threading.Thread(target=self._monitor, daemon=True, name="mcts-host-observer")
            self._thread.start()
        return self

    def _monitor(self):
        while not self._stop.wait(self.interval):
            try:
                self._sample("during_task")
            except Exception as exc:
                self.observer_errors.append("periodic host audit failed: " + type(exc).__name__)
                return

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                self.observer_errors.append("periodic host observer did not stop before task completion")
        self._sample("task_end")

    def summary(self):
        task_samples = [sample for sample in self.samples if sample["phase"] != "pre_task_wait"]
        pressure = any(sample.get("quiet") is False for sample in task_samples)
        unknown = bool(self.observer_errors) or any(sample.get("quiet") is None for sample in task_samples)
        status = "pressure_observed" if pressure else "unknown" if unknown or not task_samples else "no_pressure_observed"
        return {"status": status, "task_observations": len(task_samples),
                "during_task_observations": sum(sample["phase"] == "during_task" for sample in task_samples),
                "sample_interval_seconds": self.interval, "periodic_monitoring_enabled": bool(self.interval),
                "quiet_at_observed_samples": False if pressure else None if unknown or not task_samples else True,
                "observed_reasons": sorted({reason for sample in task_samples for reason in sample.get("reasons", [])}),
                "probe_errors": sorted(set(self.observer_errors) | {reason for sample in task_samples for reason in sample.get("probe_errors", [])}),
                "quiet_criteria": {"memory_full_avg10_max_percent": self.config.get("maximum_idle_memory_pressure_percent", 1.0),
                                   "cpu_some_avg10_max_percent": self.config.get("maximum_idle_cpu_pressure_percent"),
                                   "io_full_avg10_max_percent": self.config.get("maximum_idle_io_pressure_percent"),
                                   "known_competing_services_inactive": True,
                                   "autovacuum_workers_inactive": bool(self.database)},
                "pre_task_wait": {"enabled": self.wait_enabled, "status": self.wait_status,
                                  "seconds": self.wait_seconds, "timeout_seconds": self.wait_timeout},
                "log_path": str(self.log_path),
                "scope": "sampled whole-task conditions; includes benchmark load; not proof of quiet between observations"}
