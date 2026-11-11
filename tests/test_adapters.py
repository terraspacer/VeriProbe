from pathlib import Path
import json
import os

from mcts_rewrite.adapters import FileModel, HybridGenerator, UpstreamStandalone, plan_sha256
from mcts_rewrite.baselines import BaselineRegistry, formatting_fingerprint


def test_file_model_requires_byte_identical_sql_and_correct_hash(tmp_path):
    path = tmp_path / "model.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"original_sql": "select 'X' from t;", "candidates": [{"sql": "select 'X' from t where true;", "action": "test"}]},
        {"original_sql": "select 2;", "original_sha256": "incorrect", "candidates": [{"sql": "select 3;"}]}]))
    model = FileModel(path)
    assert len(model({"sql": "select 'X' from t;", "task": "macro_rewrite"})["candidates"]) == 1
    for changed in ["SELECT 'X' from t;", "select 'x' from t;", "select  'X' from t;", "select /*+ Set(x y) */ 'X' from t;"]:
        assert not model({"sql": changed, "task": "macro_rewrite"})["candidates"]
    assert model.errors and model.errors[0]["line"] == 2
    assert not model({"sql": "select 2;", "task": "macro_rewrite"})["candidates"]


def test_file_critic_requires_explicit_task_and_model_config(tmp_path):
    path = tmp_path / "model.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"original_sql": "select 1;", "predicted_ms": 999},
        {"original_sql": "select 2;", "task": "cost_critic", "plan_sha256": plan_sha256({}), "predicted_ms": 12.5, "rationale": "Estimate only"}]))
    generator = HybridGenerator(tmp_path, model={"type": "file", "path": str(path)}, include_a=False)
    assert generator.model.supports_task("cost_critic")
    assert not generator.model.supports_task("cost_critic", "select 1;")
    assert generator.model.supports_task("cost_critic", "select 2;")
    assert not generator.model.supports_task("cost_critic", "select 2;", {"Plan": {"Total Cost": 1}})
    assert "predicted_ms" not in generator.model({"sql": "select 1;", "task": "cost_critic"})
    assert generator.model({"sql": "select 2;", "task": "cost_critic"})["predicted_ms"] == 12.5
    assert "predicted_ms" not in generator.model({"sql": "select 2;", "task": "cost_critic", "plan": {"Plan": {"Total Cost": 1}}})
    generator._memory["old_query"] = [{"action": "old"}]
    generator.begin_query("select 2;", "")
    assert generator._memory == {}


def test_formatting_fingerprint_preserves_semantic_tokens():
    original = "select a, 'two words' from \"MyTable\" where x >= 1;"
    assert formatting_fingerprint(original) == formatting_fingerprint(" select  a , 'two words' \nfrom \"MyTable\" where x >= 1 ")
    for changed in [original.replace("'two words'", "'two  words'"),
                    original.replace('"MyTable"', '"mytable"'),
                    original.replace("select", "SELECT"),
                    original.replace("1;", "2;"),
                    original.replace("select", "select /*+ Set(enable_hashjoin off) */"),
                    original.replace(">=", "> =")]:
        assert formatting_fingerprint(original) != formatting_fingerprint(changed)


def test_registry_rejects_saved_original_missing_from_shipped_inputs(tmp_path):
    dataset = tmp_path / "vendor/QUITE/dataset/queries"
    results = tmp_path / "vendor/QUITE/experiments_results/tpch"
    dataset.mkdir(parents=True)
    results.mkdir(parents=True)
    (dataset / "tpch_queries.json").write_text(json.dumps([{"id": "1", "query": "select 1;"}]))
    (results / "QUITE_tpch_2queries.json").write_text(json.dumps([
        {"id": "1", "original_query": "select 1;", "rewritten_query": "select 1 where true;", "original_execution_time": 999},
        {"id": "2", "original_query": "select 2;", "rewritten_query": "select 2 where true;"}]))
    registry = BaselineRegistry(tmp_path)
    row = registry.lookup("quite", " select  1 ", "tpch")
    assert row["sql"] == "select 1 where true;"
    assert "original_execution_time" not in row["provenance"]
    assert row["provenance"]["published_timings_used"] is False
    assert registry.lookup("quite", "select 2;", "tpch") is None
    assert registry.lookup("quite", "SELECT 1;", "tpch") is None
    assert registry.lookup("quite", "select 1;", "unsupported") is None
    assert registry.diagnostics[0]["status"] == "unsupported_original_source"


def test_macro_candidates_are_untrusted_and_hints_require_capability(tmp_path, monkeypatch):
    monkeypatch.delenv("MCTS_PG_HINT_PLAN", raising=False)
    captured = []
    def model(payload):
        captured.append(payload)
        return {"candidates": [
            {"sql": "select /*+ Set(enable_hashjoin off) */ 1;", "action": "pg_hint_plan"},
            {"sql": "select 1 where true;", "action": "redundancy", "provenance": {"verified": True, "standalone_quite": True}}]}
    generator = HybridGenerator(tmp_path, model=model, include_a=False)
    rows = generator.candidates("select 1;", "", {}, {}, {}, limit=4)
    assert len(rows) == 1
    assert rows[0]["action_class"] == "B"
    assert rows[0]["provenance"]["verified"] is False
    assert rows[0]["provenance"]["standalone_quite"] is False
    assert "pg_hint_plan" not in captured[0]["allowed_transformations"]
    monkeypatch.setenv("MCTS_PG_HINT_PLAN", "1")
    assert len(generator.candidates("select 1;", "", {}, {}, {}, limit=4)) == 2


def test_calcite_model_selection_and_labeled_fallback(tmp_path):
    generator = HybridGenerator(tmp_path, model=lambda payload: {"rule_sequences": [["FILTER_REDUCE_EXPRESSIONS"]]}, include_b=False)
    class Bridge:
        def match(self, sql, schema):
            return {"status": "ok", "matches": [{"name": "FILTER_REDUCE_EXPRESSIONS", "sql_after": "select 1;"}]}
        def rewrite(self, sql, schema, sequences):
            assert sequences == [["FILTER_REDUCE_EXPRESSIONS"]]
            return {"status": "ok", "rewrites": [{"sql": "select 1;", "rules": sequences[0]}]}
    generator.calcite = Bridge()
    row = generator.candidates("select 1 where true;", "", {}, {}, {}, limit=1)[0]
    assert row["provenance"]["selection"] == "model_rag"
    generator.model = None
    row = generator.candidates("select 1 where true;", "", {}, {}, {}, limit=1)[0]
    assert row["provenance"]["selection"] == "deterministic_matched_rule_fallback"
    assert row["provenance"]["standalone_rbot"] is False


def test_unconfigured_macro_has_no_heuristic_standalone_result(tmp_path, monkeypatch):
    monkeypatch.delenv("MCTS_MODEL_FILE", raising=False)
    monkeypatch.delenv("MCTS_MODEL_ENDPOINT", raising=False)
    generator = HybridGenerator(tmp_path, include_a=False)
    assert generator.candidates("select 1;", "", {}, {}, {}, limit=4) == []
    assert generator.last_diagnostics[0]["status"] == "unavailable"


def test_fresh_standalone_fails_explicitly_without_upstream(tmp_path):
    # Use the actual worker; missing pinned source cannot become an identity
    # baseline and a previous successful result must not survive failure.
    output = tmp_path / "out"
    output.mkdir()
    (output / "standalone_result.json").write_text(json.dumps({"status": "ok", "sql": "select 999;"}))
    result = UpstreamStandalone(tmp_path, "quite", timeout=10).rewrite("select 1;", "", output_dir=output)
    assert result["status"] == "error"
    assert result.get("sql") != "select 999;"


def test_quite_all_workers_failed_is_not_successful_identity(tmp_path, monkeypatch):
    import sys
    import types
    from mcts_rewrite import upstream_runtime, upstream_worker
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(upstream_runtime, "prepare_quite_runtime", lambda root, runtime: tmp_path)
    (tmp_path / "runtime_compatibility.json").write_text("[]")
    monkeypatch.setattr(upstream_worker, "_require_configuration", lambda names: None)
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda path: None))
    def failed_cli(path, run_name):
        print("## All the parallel reasoning workers failed. Use the original SQL.##")
    monkeypatch.setattr(upstream_worker.runpy, "run_path", failed_cli)
    result = upstream_worker._quite({"root": str(tmp_path), "output_dir": str(tmp_path),
                                     "sql": "select 1;", "schema_sql": "", "database": "test"})
    assert result["status"] == "error"
    assert result["upstream_identity_fallback"] is True
    assert "sql" not in result
