import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_a_runner", ROOT / "scripts/run_stage_a.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def cohort(tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE t(a integer);")
    rows = [{"id": f"{benchmark}:{index}", "benchmark": benchmark, "sql": f"SELECT {index};",
             "schema_path": str(schema), "sha256": runner.digest(f"SELECT {index};".encode())}
            for benchmark in runner.DATASETS for index in range(10)]
    config = {"databases": {name: {"dbname": name, "benchmark": benchmark, "dsn": "dbname=" + name}
                             for benchmark, name in runner.DATASETS.items()},
              "verification_databases": {benchmark: {"dsn": "dbname=" + benchmark + "_dev"} for benchmark in runner.DATASETS},
              "verification_mode": "hybrid", "comparison_profile": "positional_numeric",
              "comparison_mode": "stream_hash", "float_abs_tol": 1e-5, "float_rel_tol": 1e-5}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = runner.parse_args(["--config", str(path), "--manifest", str(manifest), "--output", str(tmp_path / "output"),
                              "--session-id", "unit_test"])
    return args, config, rows


def test_fixed_cohort_validation_before_output_or_bridge(cohort):
    args, _, rows = cohort
    config, selected = runner.prepare_inputs(args)
    assert len(selected) == 30 and config["measurement_runs"] == 3
    assert config["float_abs_tol"] == config["float_rel_tol"] == 1e-5
    args.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="exactly 30"):
        runner.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["databases"].update(tpch50={"dsn": "dbname=tpch50"}), "exactly tpch10"),
    (lambda c: c["databases"]["tpch10"].update(dsn="dbname=tpch50"), "Excluded database"),
    (lambda c: c["databases"]["calcite10"].update(distribution="zipf"), "primary SF10"),
    (lambda c: c.update(float_abs_tol=0.001), "tolerances"),
    (lambda c: c.update(measurement_runs=1), "three final"),
    (lambda c: c.update(max_model_calls=9), "budget eight"),
    (lambda c: c.update(verification_databases={}), "development verification"),
])
def test_unapproved_datasets_and_protocol_rejected(cohort, mutation, match):
    args, config, _ = cohort
    mutation(config)
    args.config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        runner.prepare_inputs(args)
    assert not args.output.exists()


def test_budget_counts_failures_enforces_each_arm_and_shared_total():
    ledger, called = {}, []
    def model(payload):
        called.append(payload)
        if len(called) == 1:
            raise TimeoutError("worker late")
        return {"provenance": {"actual_model": "gpt-6.1-sol"}}
    wrappers = {arm: runner.BudgetedModel(model, arm=arm, ledger=ledger, query_id="q") for arm in runner.ARM_LIMITS}
    # Transient transport errors are retried inside the same logical call.
    wrappers["rbot"]({"task": "qa"})
    wrappers["rbot"]({"task": "rules"})
    with pytest.raises(runner.ModelBudgetExceeded):
        wrappers["rbot"]({"task": "retry"})
    for arm in ("quite", "mcts"):
        for _ in range(3):
            wrappers[arm]({"task": "batch"})
    with pytest.raises(runner.ModelBudgetExceeded):
        wrappers["mcts"]({"task": "extra"})
    # Eight logical calls plus one transient retry reached the transport.
    assert len(called) == ledger["total_invocations"] == 9
    assert ledger["invocations_by_arm"] == {"rbot": 3, "quite": 3, "mcts": 3}
    assert ledger["calls"][0]["status"] == "error"
    assert called[-1]["stage_a_context"]["input_call_index"] == 9
    assert all(not item["transport_invoked"] for item in ledger["blocked_calls"])


class FakeDB:
    instances = []
    fail = None
    def __init__(self, dsn, **kwargs):
        self.dsn, self.options = dsn, kwargs
        self.measurements, self.closed = [], False
        self.timeout_ms = kwargs.get("timeout_ms")
        self.instances.append(self)
    def close(self):
        self.closed = True
    def measure(self, sql, runs, warmups):
        self.measurements.append((sql, runs, warmups))
        if self.fail:
            return dict(self.fail)
        value = 100 if sql == "SELECT 0;" else 50
        return {"status": "ok", "samples_ms": [value - 1, value, value + 1], "median_ms": value,
                "plan": {"Plan": {"Total Cost": value}}}
    def explain(self, sql, analyze=False):
        assert analyze is False
        return {"Plan": {"Total Cost": 10 if "fast" in sql else 20}}


class FakeGate:
    def __init__(self, performance, verification, schema, **kwargs):
        self.metadata = {"scope": "separate_verification_database", "performance_database_rows_hashed": False}
    def verify(self, original, candidate):
        return {"accepted": "wrong" not in candidate, "formal": False, "proof": "differential"}


class FakeOps:
    instances = []
    sql = "SELECT fast;"
    def __init__(self, root, model):
        self.model = model
        self.last_diagnostics, self.model_errors = [], []
        self.instances.append(self)
    def begin_query(self, original, schema):
        self.original = original
    def rbot(self, original, schema, **kwargs):
        self.model({"task": "rbot_batch"})
        return [{"sql": self.sql}, {"sql": "SELECT wrong;"}]
    def metadata(self):
        return {"variant": "batching changed", "embedding": "lexical BM25 variant"}


class FakeSearch:
    def __init__(self, db, gate, ops, config, operator):
        self.ops, self.operator, self.config = ops, operator, config
        assert config.max_canaries == (2 if operator == "mcts" else 0)
    def run(self, original, schema, original_ms):
        assert original_ms in (100, 300000)
        for task in ("depth1", "depth2", "critic"):
            self.ops.model({"task": self.operator + "_" + task})
        return {"sql": FakeOps.sql, "fallback": False, "canaries": []}


@pytest.fixture
def fake_components():
    FakeDB.instances, FakeOps.instances = [], []
    FakeDB.fail = None
    FakeOps.sql = "SELECT fast;"
    return SimpleNamespace(Postgres=FakeDB, ScopedGate=FakeGate, Operators=FakeOps,
                           SearchConfig=lambda **kwargs: SimpleNamespace(**kwargs), Search=FakeSearch,
                           HTTPModel=lambda *args, **kwargs: lambda payload: {})


def test_independent_arms_final_samples_shared_only_for_exact_sql(cohort, fake_components):
    _, config, rows = cohort
    record = runner.run_query(rows[0], config, lambda arm, row: lambda payload: {"provenance": {"model": arm}},
                              components=fake_components)
    assert record["status"] == "ok"
    assert len(FakeOps.instances) == 3
    assert len({id(ops.model) for ops in FakeOps.instances}) == 3
    performance, verification = FakeDB.instances
    assert performance.measurements == [("SELECT 0;", 3, 1), ("SELECT fast;", 3, 1)]
    assert not verification.measurements
    assert record["arms"]["rbot"]["measurement"]["mean_ms"] == 50
    assert record["arms"]["mcts"]["measurement"]["shared_identical_sql_with"] == "rbot"
    assert record["arms"]["original"]["measurement"]["mean_ms"] == 100
    assert record["model_call_audit"]["total_invocations"] == 7
    assert performance.closed and verification.closed
    assert performance.options["comparison_profile"] == verification.options["comparison_profile"] == "positional_numeric"


def test_rejected_candidate_falls_back_and_never_wins(cohort, fake_components):
    _, config, rows = cohort
    FakeOps.sql = "SELECT wrong;"
    record = runner.run_query(rows[0], config, lambda arm, row: lambda payload: {}, components=fake_components)
    assert record["status"] == "ok"
    for arm in runner.ARM_LIMITS:
        assert record["arms"][arm]["fallback"] is True
        assert record["arms"][arm]["eligible_win"] is False
        assert record["arms"][arm]["measurement"]["shared_identical_sql_with"] == "original"
    assert FakeDB.instances[0].measurements == [("SELECT 0;", 3, 1)]


def test_original_timeout_preserves_censoring_but_still_generates_all_arms(cohort, fake_components):
    _, config, rows = cohort
    FakeDB.fail = {"status": "timeout", "samples_ms": [], "error_type": "QueryCanceled", "sqlstate": "57014"}
    record = runner.run_query(rows[0], config, lambda *args: lambda payload: {}, components=fake_components)
    assert record["status"] == "original_timeout"
    assert set(record["arms"]) == set(runner.ARMS)
    assert all(record["arms"][arm]["fresh_generation"] is True for arm in runner.ARM_LIMITS)
    assert record["model_call_audit"]["total_invocations"] == 7
    assert record["original_reference"]["kind"] == "censored_timeout_guard"
    assert record["original_reference"]["is_measured_latency"] is False
    assert record["arms"]["mcts"]["selection"]["original_reference"]["timeout_lower_bound_ms"] == 300000
    assert runner.arm_measurement(record, "original", "median_ms") is None


def test_database_connection_failure_classified_as_infrastructure(cohort, fake_components):
    _, config, rows = cohort
    FakeDB.fail = {"status": "error", "samples_ms": [], "error_type": "OperationalError", "sqlstate": "08006"}
    record = runner.run_query(rows[0], config, lambda *args: pytest.fail("No model should be requested"), components=fake_components)
    assert record["status"] == "infrastructure_error"


def measured_record(index, *, original=100, rbot=100, quite=100, mcts=80):
    row = {"query_id": str(index), "benchmark": list(runner.DATASETS)[index % 3], "dataset": "dataset",
           "status": "ok", "arms": {}, "model_call_audit": {"total_invocations": 8}}
    for arm, value in dict(original=original, rbot=rbot, quite=quite, mcts=mcts).items():
        row["arms"][arm] = {"status": "ok", "eligible_win": arm == "mcts", "fallback": False,
                            "measurement": {"status": "ok", "mean_ms": value, "median_ms": value}}
    return row


def test_gate_uses_fifteen_percent_latency_reduction_and_zero_regressions(tmp_path):
    records = [measured_record(index) for index in range(30)]
    report = runner.write_reports(tmp_path, records)
    assert report["stage_b_gate"]["passed"] is True
    assert report["mcts_vs_baselines"]["quite"]["mean_latency_reduction"] == pytest.approx(.2)
    records[0] = measured_record(0, original=79)
    report = runner.summarize(records)
    assert report["stage_b_gate"]["passed"] is False
    assert report["mcts_regression_count_vs_original"] == 1
    records[0] = measured_record(0)
    records[-1]["status"] = "partial_error"
    records[-1]["arms"]["mcts"].update(status="generation_error", eligible_win=False)
    report = runner.summarize(records)
    assert report["stage_b_gate"]["passed"] is False
    assert report["mcts_vs_baselines"]["quite"]["eligible_mcts_win_count"] == 29
    assert (tmp_path / "per_query.csv").exists()
    assert (tmp_path / "tpch_summary.json").exists()


def test_gate_does_not_confuse_fifteen_percent_speedup_with_latency_reduction():
    records = [measured_record(index, mcts=100 / 1.15) for index in range(30)]
    assert runner.summarize(records)["stage_b_gate"]["passed"] is False
    assert runner.percentile90([1, 2, 3, 4]) == pytest.approx(3.7)


def test_session_bridge_fresh_http_model_roundtrip_without_persisted_secret(tmp_path, monkeypatch):
    import threading
    import time
    from mcts_rewrite.adapters import HTTPModel
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from session_model_bridge import pending_requests, claim_request, submit_response
    with runner.session_bridge("fake_api_smoke", timeout=5) as (bridge, address, env_name):
        secret = runner.os.environ[env_name]
        errors = []
        def worker():
            try:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    pending = pending_requests(bridge.directory)
                    if pending:
                        request_id = pending[0]["request_id"]
                        claim_request(bridge.directory, request_id, "unit_test_worker")
                        submit_response(bridge.directory, request_id, "unit_test_worker", '{"candidates": []}')
                        return
                    time.sleep(.01)
                errors.append("no request")
            except Exception as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=worker)
        thread.start()
        response = HTTPModel(address + "/runs/query_arm/v1/chat/completions", "gpt-6.1-sol", timeout=6,
                             api_key_env=env_name)({"task": "fake_api_smoke", "output_schema": {"candidates": []}})
        thread.join(timeout=5)
        assert errors == []
        provenance = response["provenance"]["session_transport"]
        assert provenance["run_id"] == "query_arm"
        assert provenance["actual_serving_model"] == "gpt-6.1-sol"
        assert provenance["reasoning_effort"] == "xhigh"
        persisted = "".join(path.read_text() for path in bridge.directory.rglob("*") if path.is_file())
        assert secret not in persisted
    assert env_name not in runner.os.environ
    assert (bridge.directory / "STOP").exists()


def test_rbot_never_forces_rewrite_more_costly_than_original(fake_components):
    db = FakeDB("dbname=fake")
    gate = FakeGate(None, None, "")
    ops = FakeOps(ROOT, lambda payload: {})
    result = runner.choose_rbot(ops, db, gate, "SELECT original;", "schema", {"Plan": {"Total Cost": 1}}, "hybrid")
    assert result["sql"] == "SELECT original;"
    assert result["fallback"] is True
    assert result["trace"][0]["status"] == "root_competitor_preferred"


def test_full_launcher_completes_thirty_even_with_query_timeouts(cohort, fake_components, monkeypatch):
    from contextlib import contextmanager
    args, config, rows = cohort
    monkeypatch.setattr(runner, "default_components", lambda: fake_components)
    @contextmanager
    def bridge(*args, **kwargs):
        yield SimpleNamespace(directory=Path("/fake/spool")), "http://127.0.0.1:1", "FAKE_KEY"
    monkeypatch.setattr(runner, "session_bridge", bridge)
    # Real run_query executes all30 with independent fresh generation despite timeouts.
    FakeDB.fail = {"status": "timeout", "samples_ms": [], "error_type": "QueryCanceled", "sqlstate": "57014"}
    assert runner.run(args) == 0
    records = [json.loads(line) for line in (args.output / "raw.jsonl").read_text().splitlines()]
    assert len(records) == 30
    assert all(record["status"] == "original_timeout" for record in records)
    assert json.loads((args.output / "run.json").read_text())["status"] == "completed"
    assert json.loads((args.output / "stage_b_gate.json").read_text())["passed"] is False


def test_full_launcher_stops_at_first_infrastructure_connection_failure(cohort, fake_components, monkeypatch):
    from contextlib import contextmanager
    args, _, _ = cohort
    monkeypatch.setattr(runner, "default_components", lambda: fake_components)
    @contextmanager
    def bridge(*args, **kwargs):
        yield SimpleNamespace(directory=Path("/fake/spool")), "http://127.0.0.1:1", "FAKE_KEY"
    monkeypatch.setattr(runner, "session_bridge", bridge)
    FakeDB.fail = {"status": "error", "samples_ms": [], "error_type": "OperationalError", "sqlstate": "08006"}
    assert runner.run(args) == 1
    records = [json.loads(line) for line in (args.output / "raw.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "infrastructure_error"
    assert json.loads((args.output / "run.json").read_text())["status"] == "failed"
