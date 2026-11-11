#!/usr/bin/env python3
"""Run one fresh 30-query Stage A v2 study against already-running PostgreSQL.

This launcher never starts PostgreSQL. It requires a PostgreSQL-validated v2
witness binding, fresh output and session spool, and the existing external
heavy-experiment flock. A supervisor must service the v2 request spool. Use
--prepare-only for offline validation without a lock, service, SQL or model call.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import sys
import threading
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_stage_a as base
from scripts import report_stage_a_v2 as reporting
from scripts import session_model_bridge_v2 as transport
from mcts_rewrite import study_v2

ARMS, DATASETS = base.ARMS, base.DATASETS
WITNESS_QUERY_INDICES = (2, 4, 5, 12, 14, 15, 16, 17, 20, 23, 25, 26, 29)


class SessionGenerationError(RuntimeError):
    """A request-bound supervisor launch failure retained for diagnostics."""
    def __init__(self, failure):
        self.failure = failure
        super().__init__("Terminal session worker launch failure: " + str(failure.get("failure_code")))


class GenerationModel:
    """Retain the v2 HTTP error envelope while using the existing HTTPModel."""
    def __init__(self, model, failures, await_supervisor=None):
        self.model, self.failures = model, failures
        self.await_supervisor = await_supervisor

    def __call__(self, payload):
        # Defense in depth: BudgetedModel performs the same check before it
        # counts the call. This second check closes the small interval between
        # accounting and the actual HTTP dispatch.
        if self.await_supervisor is not None:
            self.await_supervisor()
        try:
            return self.model(payload)
        except urllib.error.HTTPError as exc:
            if exc.code == 422:
                try:
                    body = json.loads(exc.read(65_536))
                    error = body.get("error", {})
                    failure = error.get("generation_failure", {})
                    if (error.get("type") == "session_generation_error"
                            and error.get("code") == "worker_launch_failed"
                            and error.get("retryable") is False
                            and failure.get("terminal") is True
                            and failure.get("kind") == "worker_launch_failure"):
                        self.failures.append(failure)
                        raise SessionGenerationError(failure) from exc
                except (ValueError, TypeError, AttributeError):
                    pass
            raise


class SupervisorHeartbeatGate:
    """Block model accounting and dispatch until a fresh supervisor exists."""
    def __init__(self, output, context, *, max_age_seconds=45, poll_seconds=1):
        self.output = Path(output)
        self.context = context
        self.heartbeat = self.output / "supervisor_heartbeat.json"
        self.run_path = self.output / "run.json"
        self.max_age_seconds = max_age_seconds
        self.poll_seconds = poll_seconds

    def live(self):
        try:
            heartbeat = json.loads(self.heartbeat.read_text())
            observed = heartbeat["unix_time"]
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                return False
            age = time.time() - observed
            return 0 <= age <= self.max_age_seconds
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def await_supervisor(self):
        if self.live():
            return
        previous = self.context.get("status", "running")
        self.context.update(status="waiting_for_model_supervisor",
                            supervision_wait_started=base.utc_now(),
                            no_model_dispatch=True)
        base.atomic_json(self.run_path, self.context)
        print(json.dumps({"status": "waiting_for_model_supervisor",
                          "no_model_count": True, "no_model_dispatch": True}), flush=True)
        while not self.live():
            time.sleep(self.poll_seconds)
        self.context.update(status=previous, supervision_restored_at=base.utc_now(),
                            no_model_dispatch=False)
        base.atomic_json(self.run_path, self.context)


def supervised_budget_model(base_class, await_supervisor):
    """Build the per-run ledger guard without modifying the legacy class."""
    class SupervisedBudgetModel(base_class):
        def __call__(self, payload):
            # This check happens before BudgetedModel appends to its ledger.
            await_supervisor()
            return super().__call__(payload)
    return SupervisedBudgetModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--spool-root", type=Path, default=ROOT / ".runtime/session_model")
    parser.add_argument("--heavy-lock", type=Path, help="Existing shared external heavy-experiment lock file")
    parser.add_argument("--heavy-lock-fd", type=int, help="Inherited descriptor for that same external flock")
    parser.add_argument("--prepare-only", action="store_true", help="Offline readiness validation; create nothing")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", args.session_id):
        parser.error("Session ID must contain 1-96 letters, digits, underscores or hyphens")
    if args.heavy_lock_fd is not None and args.heavy_lock_fd < 0:
        parser.error("Inherited lock descriptor must be nonnegative")
    if not args.prepare_only and args.heavy_lock is None:
        parser.error("Execution requires --heavy-lock pointing to the existing external lock")
    return args


def prepare_inputs(args, *, execution=True):
    """All checks here are file-only; no database connection is attempted."""
    config, rows = base.prepare_inputs(args)
    if not reporting.finite_positive(config.get("timeout_ms")):
        raise ValueError("A finite positive explicit timeout_ms is required")
    if not reporting.finite_positive(config.get("model_timeout_seconds", 600)):
        raise ValueError("Model timeout must be finite and positive")
    # Bind the actual DSN database, rather than trusting display labels alone.
    from psycopg.conninfo import conninfo_to_dict
    for benchmark, dataset in DATASETS.items():
        database = config["databases"][dataset]
        if conninfo_to_dict(database["dsn"]).get("dbname") != dataset:
            raise ValueError("Each primary DSN must target its exact SF10/uniform database")
        if database.get("distribution", "uniform") != "uniform":
            raise ValueError("Only uniform primary datasets are permitted")
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("Every manifest query ID must be a nonempty string")
        if row.get("distribution", "uniform") != "uniform":
            raise ValueError("Only uniform manifest rows are permitted")
    if args.output.resolve().exists():
        raise FileExistsError("Choose a fresh output directory; v2 records are immutable")
    if (args.spool_root.resolve() / args.session_id).exists():
        raise FileExistsError("Choose a fresh session ID; v2 model spools are immutable")
    binding = None
    if args.fixture_bindings.exists():
        binding = json.loads(args.fixture_bindings.read_text())
    validated = isinstance(binding, dict) and binding.get("status") == "postgresql_validated"
    if execution and not validated:
        raise ValueError("Execution requires an existing postgresql_validated v2 fixture binding")
    if validated:
        required = {rows[index - 1]["id"] for index in WITNESS_QUERY_INDICES}
        if not isinstance(binding.get("queries"), dict) or set(binding["queries"]) != required:
            raise ValueError("The v2 witness binding must cover exactly the 13 designated manifest queries")
        for row in rows:
            study_v2.configuration_for_query(config, row, binding)
    return config, rows, binding, validated


@contextmanager
def heavy_lock(path, inherited_fd=None):
    """Acquire the existing shared flock; preserve a caller-owned inherited FD.

    flock operates on the open file description, so an inherited descriptor can
    safely share the lock already held by the supervising parent. A separate
    open would contend with that parent's lock. Neither route creates a lock.
    """
    path = Path(path).resolve(strict=True)
    own_fd = inherited_fd is None
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)) if own_fd else inherited_fd
    try:
        expected, actual = path.stat(), os.fstat(fd)
        if not stat.S_ISREG(actual.st_mode) or (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("Inherited lock descriptor must refer to the exact existing shared lock file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("External heavy-experiment lock is busy; no study was started") from exc
        yield {"path": str(path), "inherited_fd": not own_fd, "device": actual.st_dev, "inode": actual.st_ino,
               "acquired_at": base.utc_now(), "pid": os.getpid()}
    finally:
        # The parent owns an inherited lock's lifetime. Never unlock it here.
        if own_fd:
            os.close(fd)


@contextmanager
def session_bridge(session_id, *, spool_root, timeout=600):
    env_name = "MCTS_STAGE_A_V2_BRIDGE_SECRET"
    previous = os.environ.get(env_name)
    secret = secrets.token_urlsafe(48)
    bridge = transport.Bridge(Path(spool_root), secret, session_id=session_id, timeout=timeout)
    server = None
    thread = None
    os.environ[env_name] = secret
    try:
        server = transport.make_server(bridge, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="stage-a-v2-session-bridge")
        thread.start()
        yield bridge, f"http://127.0.0.1:{server.server_address[1]}", env_name
    finally:
        bridge.stop("Stage A v2 runner stopped")
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


@contextmanager
def termination_signals():
    """Translate termination into normal finally-block cleanup."""
    previous = {}
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def snapshot_inputs(args, output, config, rows, binding):
    inputs = output / "inputs"
    inputs.mkdir()
    (inputs / "manifest.jsonl").write_bytes(args.manifest.read_bytes())
    base.atomic_json(inputs / "config.public.json", base.public(config))
    base.atomic_json(inputs / "fixture_bindings.public.json", base.public(binding))
    schemas = inputs / "schemas"
    schemas.mkdir()
    for row in rows:
        path = Path(row["schema_path"])
        body = (path if path.is_absolute() else ROOT / path).read_bytes()
        (schemas / (base.digest(body) + ".sql")).write_bytes(body)
    source_paths = {Path(__file__), ROOT / "scripts/run_stage_a.py", ROOT / "scripts/report_stage_a.py",
                    ROOT / "scripts/report_stage_a_v2.py", ROOT / "scripts/session_model_bridge.py",
                    ROOT / "scripts/session_model_bridge_v2.py", *sorted((ROOT / "mcts_rewrite").glob("*.py"))}
    for optional in ("scripts/build_query_witness_fixtures_v2.py",
                     "scripts/query_fixture_recipes_v2.py",
                     "scripts/materialize_query_witness_fixtures_v2.py",
                     "configs/stage-a-v2-witness-performance-databases.json",
                     "upstream.lock.json", "requirements.lock"):
        path = ROOT / optional
        if path.exists():
            source_paths.add(path)
    hashes = {}
    for path in sorted(source_paths):
        relative = path.relative_to(ROOT)
        body = path.read_bytes()
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        hashes[str(relative)] = base.digest(body)
    return hashes


def mark_terminal_failures(record, failures):
    for name, observed in failures.items():
        if observed and name in record.get("arms", {}):
            arm = record["arms"][name]
            arm["terminal_generation_failures"] = list(observed)
            # A valid final measurement is the result. Launch/transport
            # failures remain diagnostics and must not invalidate it.
            if arm.get("measurement", {}).get("status") not in {"ok", "timeout"}:
                arm.update(status="generation_error", eligible_win=False)
    if any(failures.values()) and record.get("status") in {"ok", "original_timeout"} \
            and any(record.get("arms", {}).get(name, {}).get("measurement", {}).get("status")
                    not in {"ok", "timeout"} for name in failures if failures[name]):
        record["status"] = "partial_error"
    return record


def run(args):
    config, rows, binding, validated = prepare_inputs(args, execution=not args.prepare_only)
    if args.prepare_only:
        print(json.dumps({"status": "execution_ready" if validated else "not_execution_ready",
                          "expected_queries": len(rows), "bindings_postgresql_validated": validated,
                          "required_query_witness_indices": list(WITNESS_QUERY_INDICES),
                          "output": str(args.output.resolve()), "spool": str(args.spool_root.resolve() / args.session_id),
                          "services_started": False, "sql_executed": False, "model_invocations": 0}))
        return 0 if validated else 2
    output = args.output.resolve()
    records = []
    with termination_signals(), heavy_lock(args.heavy_lock, args.heavy_lock_fd) as lock_evidence:
        # Recheck inside the lock. mkdir(exist_ok=False) prevents concurrent reuse.
        if (args.spool_root.resolve() / args.session_id).exists():
            raise FileExistsError("Choose a fresh v2 session spool")
        output.mkdir(parents=True, exist_ok=False)
        (output / "queries").mkdir()
        (output / "raw.jsonl").touch(exist_ok=False)
        source_hashes = snapshot_inputs(args, output, config, rows, binding)
        context = {"format_version": "stage-a-v2-run-1", "started_at": base.utc_now(), "status": "starting",
            "expected_queries": 30, "completed_queries": 0, "session_id": args.session_id,
            "source_sha256": source_hashes, "manifest_sha256": base.digest(args.manifest.read_bytes()),
            "configuration_sha256": base.digest(args.config.read_bytes()),
            "fixture_binding_sha256": base.digest(args.fixture_bindings.read_bytes()),
            "heavy_experiment_lock": lock_evidence, "postgresql_started_by_runner": False,
            "secret_persisted": False, "arms": list(ARMS),
            "model_limits": {"total_per_input": 8, "per_arm": dict(base.ARM_LIMITS)},
            "configured_serving_model": transport.SERVING_MODEL, "configured_reasoning_effort": transport.REASONING_EFFORT,
            "actual_serving_model": None, "usage_known": False,
            "cohort": "All 30 inputs freshly generated under the same SF10/uniform I/O conditions; no historical timings reused",
            "study_scope": "Transductive Stage A v2 follow-up; fixed nonrandom ten queries per benchmark",
            "measurement": "One warmup and three final samples; identical final SQL explicitly shares samples within a query"}
        base.atomic_json(output / "run.json", context)
        supervisor_gate = SupervisorHeartbeatGate(output, context)
        original_budget_model = base.BudgetedModel
        base.BudgetedModel = supervised_budget_model(original_budget_model, supervisor_gate.await_supervisor)
        def reports():
            report = reporting.build_report(output / "raw.jsonl", output / "inputs/manifest.jsonl")
            reporting.write_report(output / "report", report)
        reports()
        try:
            components = study_v2.default_components()
            with session_bridge(args.session_id, spool_root=args.spool_root,
                                timeout=config.get("model_timeout_seconds", 600)) as (bridge, address, env_name):
                context.update(status="running", spool=str(bridge.directory), bridge_address=address)
                base.atomic_json(output / "run.json", context)
                print(json.dumps({"spool": str(bridge.directory), "output": str(output), "queries": 30,
                                  "protocol_version": 2, "supervised_session_workers_required": True}), flush=True)
                for index, row in enumerate(rows):
                    failures = {arm: [] for arm in base.ARM_LIMITS}
                    context.update(current_query_id=row["id"], current_query_index=index + 1)
                    base.atomic_json(output / "run.json", context)
                    query_path = output / "queries" / (f"{index + 1:02d}_" + base.digest(row["id"])[:16] + ".json")
                    def model_factory(arm, current):
                        run_id = "stage_a_v2_" + base.digest([args.session_id, current["id"], current["sql"], arm])[:24] + "_" + arm
                        model = components.HTTPModel(address + "/runs/" + run_id + "/v1/chat/completions",
                            transport.SERVING_MODEL, timeout=config.get("model_timeout_seconds", 600) + 10,
                            api_key_env=env_name)
                        return GenerationModel(model, failures[arm], supervisor_gate.await_supervisor)
                    from mcts_rewrite.host_conditions import TaskHostMonitor
                    monitor = TaskHostMonitor(config, row["id"], args.session_id,
                        output / "host_conditions.jsonl", config["databases"][DATASETS[row["benchmark"]]])
                    with monitor:
                        record = study_v2.run_query(row, config, model_factory, fixture_bindings=binding,
                            components=components, checkpoint=lambda value: base.atomic_json(query_path, mark_terminal_failures(value, failures)))
                    mark_terminal_failures(record, failures)
                    record.update(host_conditions=monitor.summary(), run_id=args.session_id)
                    base.atomic_json(query_path, record)
                    with (output / "raw.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(record)
                    reports()
                    context.update(completed_queries=len(records), last_query_status=record["status"])
                    base.atomic_json(output / "run.json", context)
                    print(json.dumps({"completed": len(records), "query_id": row["id"], "status": record["status"],
                                      "llm_invocations": record["model_call_audit"]["total_invocations"]}), flush=True)
                    if record["status"] == "infrastructure_error":
                        raise base.InfrastructureFailure("Database infrastructure failed; stop before the next input")
            context.update(status="completed", finished_at=base.utc_now(), completed_queries=len(records))
            base.atomic_json(output / "run.json", context)
            return 0
        except BaseException as exc:
            context.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                           finished_at=base.utc_now(), completed_queries=len(records), **base.error_record(exc))
            base.atomic_json(output / "run.json", context)
            reports()
            print(json.dumps({"status": context["status"], **base.error_record(exc)}), file=sys.stderr, flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            base.BudgetedModel = original_budget_model


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
