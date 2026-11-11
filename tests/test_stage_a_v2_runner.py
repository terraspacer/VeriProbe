from contextlib import contextmanager
from copy import deepcopy
import fcntl
import io
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from scripts import run_stage_a_v2 as runner


@pytest.fixture
def inputs(tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE t(a integer);")
    rows = [{"id": f"{benchmark}:{index}", "benchmark": benchmark, "sql": f"SELECT {index};",
             "schema_path": str(schema), "sha256": runner.base.digest(f"SELECT {index};".encode())}
            for benchmark in runner.DATASETS for index in range(10)]
    config = {"databases": {dataset: {"dbname": dataset, "benchmark": benchmark, "dsn": "dbname=" + dataset}
                             for benchmark, dataset in runner.DATASETS.items()},
              "verification_databases": {benchmark: {"dsn": "dbname=fixtures"} for benchmark in runner.DATASETS},
              "verification_mode": "hybrid", "timeout_ms": 300000, "model_timeout_seconds": 5}
    binding = {"version": "query-witness-v2", "status": "postgresql_validated", "queries": {}}
    for number in runner.WITNESS_QUERY_INDICES:
        row = rows[number - 1]
        cases = [{"name": name, "dsn": "dbname=fixtures", "search_path": name,
                  "schema_sha256": runner.base.digest(schema.read_bytes()), "coverage_complete": True,
                  "constraint_rejections": 0, "rows_by_table": {"t": 0 if name == "empty" else 3}}
                 for name in ("empty", "null", "duplicate", "skew", "unmatched")]
        binding["queries"][row["id"]] = {"query_sha256": row["sha256"], "schema_sha256": cases[0]["schema_sha256"],
            "source_tables": ["t"], "mini_dbs": cases,
            "verification_database": {"dsn": "dbname=fixtures", "search_path": "skew"},
            "validation": {"postgresql_executed": True, "original_expectations_passed": True,
                           "saved_candidate_checks_passed": True, "nonvacuous": True}}
    manifest, configuration, bindings = tmp_path / "manifest.jsonl", tmp_path / "config.json", tmp_path / "bindings.json"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    configuration.write_text(json.dumps(config))
    bindings.write_text(json.dumps(binding))
    lock = tmp_path / "external.lock"
    lock.touch()
    args = runner.parse_args(["--manifest", str(manifest), "--config", str(configuration),
        "--fixture-bindings", str(bindings), "--output", str(tmp_path / "output"), "--session-id", "test_v2",
        "--spool-root", str(tmp_path / "spools"), "--heavy-lock", str(lock)])
    return args, config, rows, binding


def test_prepare_only_is_file_only_without_lock_or_bindings(inputs, monkeypatch, capsys):
    args, _, _, _ = inputs
    args.fixture_bindings.unlink()
    args.prepare_only = True
    args.heavy_lock = None
    monkeypatch.setattr(runner.study_v2, "default_components", lambda: pytest.fail("Must not construct database components"))
    monkeypatch.setattr(runner, "session_bridge", lambda *a, **k: pytest.fail("Must not start bridge"))
    assert runner.run(args) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "not_execution_ready"
    assert status["sql_executed"] is False and status["model_invocations"] == 0
    assert not args.output.exists() and not args.spool_root.exists()


def test_validated_binding_is_checked_without_any_sql(inputs):
    args, _, rows, _ = inputs
    config, selected, binding, ready = runner.prepare_inputs(args)
    assert ready and len(selected) == 30 and config["measurement_runs"] == 3
    assert set(binding["queries"]) == {rows[index - 1]["id"] for index in runner.WITNESS_QUERY_INDICES}


@pytest.mark.parametrize("change,match", [
    (lambda args, c, rows, b: b.update(status="offline_plan_not_postgresql_validated"), "postgresql_validated"),
    (lambda args, c, rows, b: b["queries"].pop(rows[1]["id"]), "exactly the 13"),
    (lambda args, c, rows, b: b["queries"][rows[1]["id"]].update(query_sha256="different"), "does not match"),
    (lambda args, c, rows, b: b["queries"][rows[1]["id"]]["validation"].update(nonvacuous=False), "must pass"),
    (lambda args, c, rows, b: c["databases"]["tpch10"].update(dsn="postgresql:///tpch50"), "exact SF10"),
    (lambda args, c, rows, b: rows[0].update(distribution="zipf"), "uniform manifest"),
    (lambda args, c, rows, b: c.update(timeout_ms=True), "timeout_ms"),
])
def test_drifted_inputs_cannot_create_any_output(inputs, change, match):
    args, config, rows, binding = inputs
    change(args, config, rows, binding)
    args.config.write_text(json.dumps(config))
    args.fixture_bindings.write_text(json.dumps(binding))
    args.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        runner.run(args)
    assert not args.output.exists() and not args.spool_root.exists()


def test_refuses_existing_output_or_spool(inputs):
    args, _, _, _ = inputs
    args.output.mkdir()
    with pytest.raises(FileExistsError, match="output"):
        runner.run(args)
    args.output.rmdir()
    (args.spool_root / args.session_id).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="session ID"):
        runner.run(args)


def test_busy_or_missing_external_lock_prevents_execution(inputs, monkeypatch):
    args, _, _, _ = inputs
    monkeypatch.setattr(runner.study_v2, "default_components", lambda: pytest.fail("Must lock before execution"))
    with args.heavy_lock.open("r+") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="lock is busy"):
            runner.run(args)
    assert not args.output.exists() and not args.spool_root.exists()
    args.heavy_lock.unlink()
    with pytest.raises(FileNotFoundError):
        runner.run(args)


def test_inherited_fd_uses_same_flock_and_parent_keeps_ownership(tmp_path):
    path = tmp_path / "lock"
    path.touch()
    wrong = tmp_path / "wrong"
    wrong.touch()
    with path.open("r+") as parent, wrong.open("r+") as other:
        fcntl.flock(parent, fcntl.LOCK_EX)
        with runner.heavy_lock(path, parent.fileno()) as evidence:
            assert evidence["inherited_fd"] is True
        assert os.fstat(parent.fileno()).st_ino == path.stat().st_ino
        with path.open("r+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="exact existing"):
            with runner.heavy_lock(path, other.fileno()):
                pytest.fail("Wrong descriptor must be rejected")


@pytest.mark.parametrize("code", ["model_capacity", "worker_launch_error"])
def test_http_v2_terminal_failure_retains_structured_generation_evidence(code):
    failure = {"terminal": True, "kind": "worker_launch_failure", "failure_code": code}
    error = {"error": {"type": "session_generation_error", "code": "worker_launch_failed", "retryable": False,
                       "generation_failure": failure}}
    def model(payload):
        raise urllib.error.HTTPError("http://local", 422, "launch failed", {}, io.BytesIO(json.dumps(error).encode()))
    observed = []
    with pytest.raises(runner.SessionGenerationError, match=code):
        runner.GenerationModel(model, observed)({})
    record = {"status": "original_timeout", "arms": {"mcts": {"status": "timeout", "eligible_win": True,
              "measurement": {"status": "timeout"}}}}
    runner.mark_terminal_failures(record, {"mcts": observed})
    assert record["status"] == "original_timeout"
    assert record["arms"]["mcts"]["status"] == "timeout"
    assert record["arms"]["mcts"]["measurement"]["status"] == "timeout"
    assert record["arms"]["mcts"]["eligible_win"] is True


def test_unrelated_http_error_is_not_reclassified():
    def model(payload):
        raise urllib.error.HTTPError("http://local", 504, "late", {}, io.BytesIO(b"{}"))
    failures = []
    with pytest.raises(urllib.error.HTTPError):
        runner.GenerationModel(model, failures)({})
    assert failures == []


def test_supervisor_heartbeat_freshness_is_bounded(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    context = {"status": "running"}
    runner.base.atomic_json(output / "run.json", context)
    gate = runner.SupervisorHeartbeatGate(output, context)
    monkeypatch.setattr(runner.time, "time", lambda: 1000.0)
    for observed, expected in ((955.0, True), (954.999, False), (1000.0, True), (1000.001, False),
                               (True, False), ("1000", False)):
        runner.base.atomic_json(output / "supervisor_heartbeat.json", {"unix_time": observed})
        assert gate.live() is expected


def test_stale_supervisor_blocks_before_budget_count_and_http_dispatch(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    context = {"status": "running"}
    runner.base.atomic_json(output / "run.json", context)
    runner.base.atomic_json(output / "supervisor_heartbeat.json", {"unix_time": time.time() - 46})
    gate = runner.SupervisorHeartbeatGate(output, context, poll_seconds=.01)
    ledger, invoked, result = {}, [], []
    def model(payload):
        invoked.append(payload)
        return {"ok": True}
    guarded = runner.supervised_budget_model(runner.base.BudgetedModel, gate.await_supervisor)
    wrapper = guarded(model, arm="rbot", ledger=ledger, query_id="heartbeat-test")
    worker = threading.Thread(target=lambda: result.append(wrapper({"task": "test"})))
    worker.start()
    time.sleep(.05)
    assert invoked == [] and ledger.get("calls", []) == []
    waiting = json.loads((output / "run.json").read_text())
    assert waiting["status"] == "waiting_for_model_supervisor"
    assert waiting["no_model_dispatch"] is True
    runner.base.atomic_json(output / "supervisor_heartbeat.json", {"unix_time": time.time()})
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result == [{"ok": True}] and len(invoked) == 1 and ledger["total_invocations"] == 1


def test_generation_model_rechecks_heartbeat_immediately_before_dispatch():
    events = []
    model = runner.GenerationModel(lambda payload: events.append("http") or {"ok": True}, [],
                                   lambda: events.append("heartbeat"))
    assert model({"task": "test"}) == {"ok": True}
    assert events == ["heartbeat", "http"]


def test_v2_bridge_httpmodel_roundtrip_launch_failure_and_clean_stop(tmp_path, monkeypatch):
    from mcts_rewrite.adapters import HTTPModel
    transport = runner.transport
    monkeypatch.setenv("MCTS_STAGE_A_V2_BRIDGE_SECRET", "previous_value")
    with runner.session_bridge("fresh_v2", spool_root=tmp_path, timeout=5) as (bridge, address, env_name):
        secret = os.environ[env_name]
        errors = []
        def supervisor(fail):
            try:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    pending = transport.pending_requests(bridge.directory)
                    if pending:
                        request = transport.read_json(bridge.directory / "requests" / pending[0]["request_id"] / "request.json")
                        if fail:
                            envelope = transport.launch_failure_envelope(request, supervisor_id="unit_supervisor",
                                failure_code="model_capacity", failure_message="Synthetic capacity failure for transport test")
                            transport.submit_launch_failure(bridge.directory, envelope)
                        else:
                            transport.claim_request(bridge.directory, request["request_id"], "unit_worker")
                            transport.submit_response(bridge.directory, request["request_id"], "unit_worker", '{"candidates": []}')
                        return
                    time.sleep(.01)
                errors.append("No pending request")
            except Exception as exc:
                errors.append(repr(exc))
        for fail in (False, True):
            worker = threading.Thread(target=supervisor, args=(fail,))
            worker.start()
            observed = []
            model = runner.GenerationModel(HTTPModel(address + "/runs/query_mcts/v1/chat/completions",
                transport.SERVING_MODEL, timeout=6, api_key_env=env_name), observed)
            if fail:
                with pytest.raises(runner.SessionGenerationError):
                    model({"task": "transport_test"})
                assert observed[0]["failure_code"] == "model_capacity"
            else:
                value = model({"task": "transport_test"})
                assert value["provenance"]["session_transport"]["actual_serving_model"] == transport.SERVING_MODEL
            worker.join(timeout=5)
            assert errors == []
        persisted = "".join(path.read_text() for path in bridge.directory.rglob("*") if path.is_file())
        assert secret not in persisted
        assert transport.read_json(bridge.directory / "server.json")["protocol_version"] == 2
    assert os.environ[env_name] == "previous_value"
    assert (bridge.directory / "STOP").exists() and bridge.closed.is_set()
    assert not any(t.name == "stage-a-v2-session-bridge" and t.is_alive() for t in threading.enumerate())


class FakeDB:
    instances = []
    fail = None
    def __init__(self, dsn, **kwargs):
        self.dsn, self.closed = dsn, False
        self.timeout_ms = kwargs.get("timeout_ms")
        self.instances.append(self)
    def close(self):
        self.closed = True
    def measure(self, sql, runs, warmups):
        if self.fail:
            return dict(self.fail)
        return {"status": "ok", "samples_ms": [100] * 3, "plan": {"Plan": {"Total Cost": 20}}}
    def explain(self, sql, analyze=False):
        return {"Plan": {"Total Cost": 10}}


class FakeOps:
    def __init__(self, root, model):
        self.model, self.last_diagnostics, self.model_errors = model, [], []
    def begin_query(self, original, schema):
        pass
    def rbot(self, original, schema, **kwargs):
        self.model({"task": "rbot_batch"})
        return []
    def metadata(self):
        return {"synthetic": True}


class FakeSearch:
    def __init__(self, db, gate, ops, config, operator):
        self.ops, self.operator = ops, operator
    def run(self, original, schema, original_ms):
        for _ in range(3):
            self.ops.model({"task": self.operator + "_batch"})
        return {"sql": original, "fallback": True}


class FakeGate:
    def __init__(self, *args, **kwargs):
        self.metadata = {"synthetic": True}
    def verify(self, original, candidate):
        return {"accepted": True, "formal": True, "proof": "identity"}


@pytest.fixture
def mocked_execution(monkeypatch):
    import mcts_rewrite.host_conditions as host
    FakeDB.instances, FakeDB.fail = [], None
    components = SimpleNamespace(Postgres=FakeDB, ScopedGate=FakeGate, Operators=FakeOps,
        SearchConfig=lambda **kwargs: SimpleNamespace(**kwargs), Search=FakeSearch,
        HTTPModel=lambda *a, **k: lambda payload: {})
    monkeypatch.setattr(runner.study_v2, "default_components", lambda: components)
    # Full-launcher doubles exercise lifecycle, not the heartbeat wait itself;
    # dedicated tests above cover stale/fresh accounting and dispatch behavior.
    monkeypatch.setattr(runner.SupervisorHeartbeatGate, "live", lambda self: True)
    events = []
    @contextmanager
    def bridge(*args, **kwargs):
        events.append("start")
        try:
            yield SimpleNamespace(directory=Path("/synthetic/spool")), "http://127.0.0.1:1", "FAKE_KEY"
        finally:
            events.append("stop")
    monkeypatch.setattr(runner, "session_bridge", bridge)
    monkeypatch.setattr(host, "observe_host", lambda *a, **k: {"quiet": True, "probe_errors": [], "reasons": []})
    return events


def test_full_launcher_fresh_thirty_even_for_timeouts_with_snapshots(inputs, mocked_execution):
    args, _, _, _ = inputs
    FakeDB.fail = {"status": "timeout", "samples_ms": []}
    assert runner.run(args) == 0
    records = [json.loads(line) for line in (args.output / "raw.jsonl").read_text().splitlines()]
    assert len(records) == 30 and len({row["query_id"] for row in records}) == 30
    assert all(row["status"] == "original_timeout" for row in records)
    assert all(row["model_call_audit"]["total_invocations"] == 7 for row in records)
    assert all(row["arms"][arm]["fresh_generation"] is True for row in records for arm in runner.base.ARM_LIMITS)
    context = json.loads((args.output / "run.json").read_text())
    assert context["status"] == "completed" and context["completed_queries"] == 30
    assert context["postgresql_started_by_runner"] is False
    for path, digest in context["source_sha256"].items():
        assert runner.base.digest((args.output / "source_snapshot" / path).read_bytes()) == digest
    assert (args.output / "inputs/manifest.jsonl").read_bytes() == args.manifest.read_bytes()
    report = json.loads((args.output / "report/summary.json").read_text())
    assert report["strict_finite_only"]["paired_queries"] == 0
    assert report["timeout_capped_T_max"]["paired_queries"] == 30
    assert mocked_execution == ["start", "stop"]
    assert all(connection.closed for connection in FakeDB.instances)
    with runner.heavy_lock(args.heavy_lock):
        pass


def test_launcher_stops_at_first_infrastructure_failure(inputs, mocked_execution):
    args, _, _, _ = inputs
    FakeDB.fail = {"status": "error", "samples_ms": [], "error_type": "OperationalError", "sqlstate": "08006"}
    assert runner.run(args) == 1
    records = [json.loads(line) for line in (args.output / "raw.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["status"] == "infrastructure_error"
    assert json.loads((args.output / "run.json").read_text())["status"] == "failed"
    assert mocked_execution == ["start", "stop"]
    assert all(connection.closed for connection in FakeDB.instances)


def test_interrupt_updates_status_closes_bridge_and_releases_lock(inputs, mocked_execution, monkeypatch):
    args, _, _, _ = inputs
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("Synthetic interrupted test")
    monkeypatch.setattr(runner.study_v2, "run_query", interrupt)
    assert runner.run(args) == 130
    assert json.loads((args.output / "run.json").read_text())["status"] == "interrupted"
    assert mocked_execution == ["start", "stop"]
    assert (args.output / "report/REPORT.md").exists()
    with runner.heavy_lock(args.heavy_lock):
        pass


def test_hard_budget_remains_at_most_eight_including_failed_attempts():
    ledger, invoked = {}, []
    def model(payload):
        invoked.append(payload)
        raise runner.SessionGenerationError({"failure_code": "model_capacity"})
    for arm, maximum in runner.base.ARM_LIMITS.items():
        wrapper = runner.base.BudgetedModel(model, arm=arm, ledger=ledger, query_id="test")
        for _ in range(maximum):
            with pytest.raises(runner.SessionGenerationError):
                wrapper({"task": "synthetic"})
        with pytest.raises(runner.base.ModelBudgetExceeded):
            wrapper({"task": "ninth_blocked"})
    assert len(invoked) == ledger["total_invocations"] == 8
