import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcts_rewrite.reporting import _model_context_issue

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("session_launcher", ROOT / "scripts/run_session_probe.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def launch_files(tmp_path):
    config = {
        "databases": {
            "tpch10": {"benchmark": "tpch", "scale": 10, "dsn": "dbname=tpch10 user=mcts_read"},
            "tpch50": {"benchmark": "tpch", "scale": 50, "dsn": "dbname=tpch50 user=mcts_read"},
            "calcite": {"benchmark": "calcite", "dsn": "dbname=calcite user=mcts_read"},
        },
        "verification_databases": {"tpch": {"dsn": "dbname=tpch_dev user=mcts_read"}},
        "budgets": [1, 4], "ablations": ["full", "no_critic"],
        "model": {"type": "file", "path": "old_candidates.jsonl", "api_key": "test-inline-secret"},
        "model_context": {"actual_serving_model": "stale-configured-file-model"},
        "purpose": "source purpose",
    }
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(config))
    rows = [
        {"id": "rbot:tpch:q1", "benchmark": "tpch", "sha256": "sql1", "schema_sha256": "schema1"},
        {"id": "quite:tpch:q1", "benchmark": "tpch", "sha256": "sql1", "schema_sha256": "schema1"},
        {"id": "rbot:calcite:q2", "benchmark": "calcite", "sha256": "sql2", "schema_sha256": "schema2"},
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    base = ["--config", str(config_path), "--manifest", str(manifest_path), "--output", str(tmp_path / "out"),
            "--session-id", "batch_unit_test", "--purpose", "partial functional tuning; concurrent loaders",
            "--dataset", "tpch10", "--query-id", "rbot:tpch:q1"]
    return base, config


def test_batch_filters_datasets_preserves_correctness_db_and_source_config(launch_files):
    base, original = launch_files
    args = launcher.parse_args(base, batch=True)
    config, plan = launcher.prepare_configuration(args)
    assert list(config["databases"]) == ["tpch10"]
    assert config["databases"]["tpch10"]["scale"] == 10
    assert config["verification_databases"] == original["verification_databases"]
    assert json.loads(args.config.read_text()) == original
    assert plan["budgets"] == [1, 4] and plan["ablations"] == ["full", "no_critic"]
    assert plan["planned_native_executions"] == 1 and plan["planned_hybrid_tasks"] == 4
    assert config["final_performance_claim_allowed"] is False


def test_native_inputs_shared_across_memberships_but_never_dataset_scales(launch_files):
    base, _ = launch_files
    args = launcher.parse_args(base + ["--dataset", "tpch50", "--query-id", "quite:tpch:q1",
                                      "--system", "rbot", "--system", "quite"], batch=True)
    config, plan = launcher.prepare_configuration(args)
    assert plan["planned_native_executions"] == 4
    assert plan["planned_hybrid_tasks"] == 16
    commands = launcher.build_commands(args, args.output / "config.json", args.output, plan)
    assert [name for name, _ in commands] == ["native_rbot", "native_quite", "hybrid"]
    for _, command in commands[:2]:
        assert command.count("--dataset") == command.count("--query-id") == 2
        assert "tpch10" in command and "tpch50" in command
        assert "--budgets" not in command
    hybrid = commands[-1][1]
    assert hybrid.count("--query-id") == 2
    assert "--dataset" not in hybrid  # The shared config contains only selected datasets.
    assert hybrid[hybrid.index("--budgets") + 1] == "1,4"


@pytest.mark.parametrize("extra,match", [
    (["--dataset", "unknown"], "Unknown configured datasets"),
    (["--query-id", "unknown"], "Unknown query IDs"),
    (["--query-id", "rbot:calcite:q2"], "matching selected counterpart"),
    (["--dataset", "calcite"], "matching selected counterpart"),
    (["--budgets", "0"], "Budgets"),
    (["--budgets", "17"], "Budgets"),
    (["--budgets", "1,1"], "Budgets"),
    (["--ablations", "full,unknown"], "ablation"),
])
def test_invalid_selections_fail_before_bridge_or_output(launch_files, extra, match):
    base, _ = launch_files
    args = launcher.parse_args(base + extra, batch=True)
    with pytest.raises(ValueError, match=match):
        launcher.run(args)
    assert not args.output.exists()


def test_required_batch_scope_and_exclusive_skip_flags(launch_files):
    base, _ = launch_files
    with pytest.raises(SystemExit):
        launcher.parse_args(["--purpose", ""], batch=True)
    with pytest.raises(SystemExit):
        launcher.parse_args(base + ["--skip-native", "--skip-hybrid"], batch=True)
    with pytest.raises(SystemExit):
        launcher.parse_args(base + ["--purpose", " "], batch=True)
    with pytest.raises(SystemExit):
        launcher.parse_args(base + ["--session-id", "../escape"], batch=True)
    with pytest.raises(SystemExit):
        launcher.parse_args(base + ["--native-timeout", "nan"], batch=True)


@pytest.mark.parametrize("flag,stages", [("--skip-native", ["hybrid"]), ("--skip-hybrid", ["native_quite"])])
def test_skip_flags_select_only_requested_stage(launch_files, flag, stages):
    base, _ = launch_files
    args = launcher.parse_args(base + [flag, "--budgets", "2", "--ablations", "class_a_only"], batch=True)
    _, plan = launcher.prepare_configuration(args)
    assert [n for n, _ in launcher.build_commands(args, args.output / "config.json", args.output, plan)] == stages


def test_ephemeral_environment_replaces_inherited_endpoints_without_mutating_parent(tmp_path):
    parent = {"OPENAI_API_KEY": "inherited", "NO_PROXY": "intranet", "MCTS_MODEL_FILE": "saved.jsonl",
              "MCTS_SESSION_BRIDGE_RUN_ID": "old", "RBOT_LOCAL_EMBEDDING_RUNTIME": "/old/runtime"}
    env = launcher.session_environment("http://127.0.0.1:1234", tmp_path / "spool", parent=parent)
    secret = env["MCTS_SESSION_BRIDGE_SECRET"]
    assert len(secret) >= 32 and secret != "inherited"
    assert all(env[k] == secret for k in ["OPENAI_API_KEY", "MCTS_MODEL_API_KEY", "ASSISTANT_MODEL_API_KEY"])
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1"
    assert env["OPENAI_TIMEOUT"] == "600"
    assert "MCTS_MODEL_FILE" not in env and "MCTS_SESSION_BRIDGE_RUN_ID" not in env
    assert "RBOT_LOCAL_EMBEDDING_RUNTIME" not in env
    assert env["NO_PROXY"] == env["no_proxy"] == "intranet,localhost,127.0.0.1,::1"
    assert parent["OPENAI_API_KEY"] == "inherited" and parent["MCTS_MODEL_FILE"] == "saved.jsonl"


def test_serial_launcher_persists_no_secret_and_stops_bridge_on_child_failure(launch_files, tmp_path, monkeypatch):
    base, _ = launch_files
    args = launcher.parse_args(base + ["--skip-native"], batch=True)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    calls = []
    bridges = []

    class Bridge:
        def __init__(self, command, **kwargs):
            self.env = kwargs["env"]
            self.stopped = False
            Path(self.env["MCTS_SESSION_BRIDGE_SPOOL"]).mkdir(parents=True)
            bridges.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    def stage(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(launcher.subprocess, "Popen", Bridge)
    monkeypatch.setattr(launcher.subprocess, "run", stage)
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda handler:
                        SimpleNamespace(open=lambda request, timeout:
                        io.BytesIO(json.dumps({"session_id": args.session_id}).encode())))
    assert launcher.run(args) == 1
    assert len(calls) == 1 and bridges[0].stopped
    secret = bridges[0].env["MCTS_SESSION_BRIDGE_SECRET"]
    persisted = "".join(p.read_text() for p in args.output.rglob("*") if p.is_file())
    assert secret not in persisted and "test-inline-secret" not in persisted
    config = json.loads((args.output / "config.json").read_text())
    assert list(config["databases"]) == ["tpch10"]
    assert config["verification_databases"]["tpch"]["dsn"].startswith("dbname=tpch_dev")
    assert config["model_context"]["actual_serving_model"] == "gpt-6.1-sol"
    assert config["model_context"]["model_substitution"] is False
    assert _model_context_issue(config["model_context"]) is None
    assert config["model_context"]["requested_sampling_options_not_enforced"] == ["temperature", "top_p", "seed", "max_tokens"]
    assert set(config["model"]) == {"type", "model", "endpoint", "api_key_env", "timeout"}
    assert config["baseline_policy"] == "fresh"
    assert (args.output / "fresh_baselines.jsonl").read_text() == ""
    status = json.loads((args.output / "probe_status.json").read_text())
    assert status["stages"]["hybrid"]["status"] == "failed"
    assert status["selection"]["skip_native"] and not status["final_performance_claim_allowed"]
    assert (tmp_path / ".runtime/session_model" / args.session_id / "STOP").exists()


def test_single_probe_keeps_existing_defaults():
    args = launcher.parse_args([])
    assert args.dataset == ["calcite"] and args.query_id == ["rbot:calcite:testAggregateRemove1"]
    assert args.budgets == "1" and args.ablations == "full" and args.system == ["quite"]


def test_actual_published_hybrid_query_plan_without_execution(tmp_path):
    args = launcher.parse_args([
        "--config", str(ROOT / "configs/published-hybrid.json"),
        "--output", str(tmp_path / "never-created"), "--session-id", "plan_only_unit_test",
        "--purpose", "partial functional tuning with concurrent loaders",
        "--dataset", "tpch10", "--query-id", "rbot:tpch:query11_0",
        "--system", "rbot", "--system", "quite", "--budgets", "4", "--ablations", "full",
    ], batch=True)
    config, plan = launcher.prepare_configuration(args)
    assert list(config["databases"]) == ["tpch10"]
    assert config["databases"]["tpch10"]["scale"] == 10
    assert plan["planned_native_executions"] == 2 and plan["planned_hybrid_tasks"] == 1
    assert not args.output.exists()
