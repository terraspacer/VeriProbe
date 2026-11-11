from copy import deepcopy
import io
import json
from pathlib import Path
import sys

import pytest

from mcts_rewrite import fresh_baselines as fresh
from mcts_rewrite.adapters import exact_sha256


@pytest.fixture
def fixture_catalog(tmp_path, monkeypatch):
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "inputs").mkdir()
    schema = tmp_path / "inputs/schema.sql"
    schema.write_text("CREATE TABLE t (id integer);")
    rows = []
    for number, sql in enumerate(("select id from t;", "select id from t where id > 1;")):
        source = tmp_path / f"inputs/q{number}.sql"
        source.write_text(sql)
        rows.append({"id": f"rbot:tpch:q{number}", "source": "rbot", "benchmark": "tpch", "sql": sql,
                     "sha256": exact_sha256(sql), "source_path": str(source.relative_to(tmp_path)),
                     "source_file_sha256": fresh.file_sha256(source), "source_commit": "input-pinned",
                     "schema_path": "inputs/schema.sql", "schema_sha256": fresh.file_sha256(schema)})
    manifest = tmp_path / "benchmarks/manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(fresh, "_commit", lambda directory: "upstream-pinned")
    databases = {name: {"benchmark": "tpch", "dbname": name, "scale": scale,
                        "dsn": f"host=/tmp/fake_socket port=55432 user=readonly dbname={name}"}
                 for name, scale in (("tpch10", 10), ("tpch50", 50))}
    group = fresh.input_groups(tmp_path, manifest)[0]
    valid = {**group, "system": "rbot", "dataset": "tpch10", "status": "ok",
             "sql": "select id from t where true;", "sql_sha256": exact_sha256("select id from t where true;"),
             "dataset_configuration_sha256": fresh.dataset_configuration_sha256(databases["tpch10"]),
             "provenance": {"system": "rbot", "mode": "fresh_standalone", "source_commit": "upstream-pinned",
                            "source_repository": "LLM4Rewrite", "entry_point": "my_rewriter.test_utils.test"}}
    return tmp_path, manifest, rows, databases, valid


def test_registry_matches_raw_sql_schema_source_and_dataset_and_discards_proof_claims(fixture_catalog):
    root, _, inputs, databases, row = fixture_catalog
    row["formal"] = True
    row["verification"] = {"accepted": True, "formal": True, "proof": "sqlsolver"}
    row["provenance"].update({"formal": True, "verified": True, "claim_used_as_proof": True,
                               "equivalent": True, "proof": "sqlsolver"})
    catalog = root / "catalog.jsonl"
    fresh.write_catalog(catalog, [row])
    registry = fresh.FreshBaselineRegistry(root, catalog, databases=databases)
    result = registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch10", inputs[0]["schema_sha256"])
    assert result["sql"] == row["sql"]
    assert result["provenance"]["formal"] is False
    assert result["provenance"]["verified"] is False
    assert result["provenance"]["claim_used_as_proof"] is False
    assert "proof" not in result["provenance"] and "equivalent" not in result["provenance"]
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch50") is None
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"] + "\n", "tpch", "tpch10") is None
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch10", "0" * 64) is None
    assert registry.lookup_for_dataset("quite", inputs[0]["sql"], "tpch", "tpch10") is None
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "dsb", "tpch10") is None
    changed = deepcopy(databases)
    changed["tpch10"]["scale"] = 50
    assert not fresh.FreshBaselineRegistry(root, catalog, databases=changed).records


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(status="error", sql=row["original_sql"], sql_sha256=row["original_sha256"]),
    lambda row: row.update(status="timeout"),
    lambda row: row.update(upstream_identity_fallback=True),
    lambda row: row.update(original_sha256="f" * 64),
    lambda row: row.update(original_sql="SELECT id from t;", original_sha256=exact_sha256("SELECT id from t;")),
    lambda row: row.update(schema_sha256="f" * 64),
    lambda row: row.update(sql_sha256="f" * 64),
    lambda row: row["provenance"].update(mode="upstream_saved_sql_replay"),
    lambda row: row["provenance"].update(source_commit="different-upstream"),
    lambda row: row["input_sources"][0].update(source_commit="different-input-source"),
    lambda row: row["input_sources"][0].update(source_file_sha256="f" * 64),
    lambda row: row.update(input_sources=[]),
])
def test_registry_rejects_malicious_mismatches_and_failed_identity(fixture_catalog, mutation):
    root, _, inputs, databases, row = fixture_catalog
    mutation(row)
    catalog = root / "catalog.jsonl"
    fresh.write_catalog(catalog, [row])
    registry = fresh.FreshBaselineRegistry(root, catalog, databases=databases)
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch10") is None


def test_conflicting_successful_outputs_are_unavailable(fixture_catalog):
    root, _, inputs, databases, row = fixture_catalog
    other = deepcopy(row)
    other.update(sql="select id from t where 1=1;", sql_sha256=exact_sha256("select id from t where 1=1;"))
    catalog = root / "catalog.jsonl"
    fresh.write_catalog(catalog, [row, other])
    registry = fresh.FreshBaselineRegistry(root, catalog, databases=databases)
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch10") is None
    assert registry.diagnostics[-1]["status"] == "conflicting_fresh_outputs"


def test_bound_database_environment_overrides_both_upstream_and_libpq_without_mutation(fixture_catalog):
    _, _, _, databases, _ = fixture_catalog
    base = {"OPENAI_API_KEY": "credential-value-stays-in-child", "PGDATABASE": "wrong_dataset",
            "DB_NAME": "wrong_dataset", "DB_HOST": "wrong-host", "DB_USER": "wrong-user",
            "PGOPTIONS": "-c default_transaction_read_only=off", "PGSERVICE": "unrelated-service"}
    original = deepcopy(base)
    bound = fresh.bound_database_environment(databases["tpch50"], base, timeout_ms=1234, threads=2)
    assert base == original
    assert bound["DB_NAME"] == bound["PGDATABASE"] == "tpch50"
    assert bound["DB_USER"] == bound["PGUSER"] == "readonly"
    assert bound["DB_HOST"] == bound["PGHOST"] == "/tmp/fake_socket"
    assert bound["OPENAI_API_KEY"] == base["OPENAI_API_KEY"]
    assert "default_transaction_read_only=on" in bound["PGOPTIONS"]
    assert "statement_timeout=1234" in bound["PGOPTIONS"]
    assert "PGSERVICE" not in bound
    assert all(bound[name] == "2" for name in fresh.THREAD_VARIABLES)


def test_missing_credentials_fail_fast_without_database_or_upstream_calls(fixture_catalog, monkeypatch):
    root, manifest, _, databases, _ = fixture_catalog
    def forbidden(*args, **kwargs):
        raise AssertionError("Unconfigured generation must not make any database/model call")
    monkeypatch.setattr(fresh, "probe_read_only_database", forbidden)
    monkeypatch.setattr(fresh, "run_bound_standalone", forbidden)
    catalog = root / "catalog.jsonl"
    report = fresh.generate_catalog(root, manifest, {"databases": databases}, catalog, environment={})
    assert report["status"] == "blocked" and report["executed"] == 0
    assert len(report["blocked"]) == 2
    assert report["planned_executions"] == 8
    assert catalog.read_text() == ""


def test_generation_checkpoints_resume_and_never_reuses_other_dataset(fixture_catalog, monkeypatch):
    root, manifest, inputs, databases, valid = fixture_catalog
    calls = []
    monkeypatch.setattr(fresh, "probe_read_only_database", lambda env: {"database": env["PGDATABASE"], "read_only": True})
    def actual_wrapper_boundary(root, request, environment):
        assert request["database"] == environment["DB_NAME"] == environment["PGDATABASE"]
        assert "dsn" not in request and "password" not in request
        calls.append((request["database"], request["sql"]))
        output = "select id from t where true;"
        return {"status": "ok", "sql": output, "elapsed_s": 1, "provenance": valid["provenance"]}
    monkeypatch.setattr(fresh, "run_bound_standalone", actual_wrapper_boundary)
    catalog = root / "catalog.jsonl"
    environment = {"OPENAI_API_KEY": "private-test-value", "OPENAI_API_BASE": "https://user:password@example.invalid/v1?api_key=private-test-value"}
    config = {"databases": databases}
    report = fresh.generate_catalog(root, manifest, config, catalog, systems=["rbot"], limit=1, environment=environment)
    assert report["executed"] == 2 and [name for name, _ in calls] == ["tpch10", "tpch50"]
    assert "private-test-value" not in catalog.read_text() and "password" not in catalog.read_text()
    registry = fresh.FreshBaselineRegistry(root, catalog, databases=databases)
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch10")
    assert registry.lookup_for_dataset("rbot", inputs[0]["sql"], "tpch", "tpch50")
    environment["OPENAI_API_KEY"] = "another-private-test-value"
    report = fresh.generate_catalog(root, manifest, config, catalog, systems=["rbot"], limit=1, environment=environment)
    assert report["executed"] == 0 and report["resumed"] == 2
    databases["tpch10"]["scale"] = 11
    report = fresh.generate_catalog(root, manifest, config, catalog, systems=["rbot"], datasets=["tpch10"], limit=1, environment=environment)
    assert report["executed"] == 1 and calls[-1][0] == "tpch10"
    assert len(fresh._read_jsonl(catalog)) == 2
    assert not list(root.glob("catalog.jsonl.*.tmp"))


@pytest.mark.parametrize("upstream_result", [
    {"status": "error", "error": "APIConnectionError: contains private-test-value"},
    {"status": "error", "error": "all workers failed", "upstream_identity_fallback": True},
])
def test_model_service_failure_stops_after_one_call(fixture_catalog, monkeypatch, upstream_result):
    root, manifest, _, databases, _ = fixture_catalog
    monkeypatch.setattr(fresh, "probe_read_only_database", lambda env: {"read_only": True})
    calls = []
    def failing(*args):
        calls.append(1)
        return upstream_result
    monkeypatch.setattr(fresh, "run_bound_standalone", failing)
    catalog = root / "catalog.jsonl"
    report = fresh.generate_catalog(root, manifest, {"databases": databases}, catalog,
                                    systems=["rbot"], environment={"OPENAI_API_KEY": "private-test-value"})
    assert report["status"] == "blocked" and len(calls) == 1
    rows = fresh._read_jsonl(catalog)
    assert len(rows) == 1 and rows[0]["sql"] is None and rows[0]["status"] == "error"
    assert "private-test-value" not in catalog.read_text()


def test_gateway_worker_calls_actual_standalone_interface(fixture_catalog, monkeypatch):
    root, _, _, _, _ = fixture_catalog
    calls = []
    class ActualInterface:
        def __init__(self, root, system, timeout):
            calls.append((root, system, timeout))
        def rewrite(self, sql, schema_sql, **kwargs):
            calls.append((sql, schema_sql, kwargs))
            return {"status": "error", "error": "Missing upstream model configuration"}
    monkeypatch.setattr(fresh, "UpstreamStandalone", ActualInterface)
    request = {"root": str(root), "system": "rbot", "timeout": 10, "sql": "select 1;", "schema_sql": "",
               "output_dir": str(root / "run"), "database": "tpch50"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    fresh._worker()
    assert calls[0] == (root, "rbot", 10)
    assert calls[1][2]["database"] == "tpch50"
    assert json.loads((root / "run/gateway_result.json").read_text())["status"] == "error"


def test_source_memberships_deduplicate_calls_and_query_id_selection_is_exact(fixture_catalog):
    root, manifest, inputs, _, _ = fixture_catalog
    duplicate = {**inputs[0], "id": "quite:tpch:q0", "source": "quite"}
    manifest.write_text("".join(json.dumps(row) + "\n" for row in [inputs[0], duplicate, inputs[1]]))
    groups = fresh.input_groups(root, manifest)
    assert len(groups) == 2 and len(groups[0]["input_sources"]) == 2
    assert len(fresh.input_groups(root, manifest, query_ids=[duplicate["id"]])) == 1
    with pytest.raises(ValueError, match="Unknown query"):
        fresh.input_groups(root, manifest, query_ids=["missing-id"])


def test_fingerprints_exclude_connection_and_model_credentials(fixture_catalog):
    _, _, _, databases, _ = fixture_catalog
    first = deepcopy(databases["tpch10"])
    second = deepcopy(first)
    first["dsn"] += " password=first-secret-value"
    second["dsn"] += " password=second-secret-value"
    assert fresh.dataset_configuration_sha256(first) == fresh.dataset_configuration_sha256(second)
    first_model = {"OPENAI_API_KEY": "first-secret-value", "OPENAI_API_BASE": "https://user:first-secret-value@example.invalid/v1?api_key=first-secret-value"}
    second_model = {"OPENAI_API_KEY": "second-secret-value", "OPENAI_API_BASE": "https://user:second-secret-value@example.invalid/v1?api_key=second-secret-value"}
    assert fresh.model_configuration("rbot", first_model) == fresh.model_configuration("rbot", second_model)
    assert fresh.missing_model_configuration("quite", {"REASONING_MODEL_URL": "http://example.invalid:bad"})


def test_gateway_preserves_credentials_only_in_child_environment_and_redacts_diagnostics(fixture_catalog, monkeypatch):
    root, _, _, databases, _ = fixture_catalog
    environment = fresh.bound_database_environment(databases["tpch50"], {"OPENAI_API_KEY": "private-test-value"})
    request = {"output_dir": str(root / "gateway"), "timeout": 10, "database": "tpch50"}
    def failed_process(command, **kwargs):
        assert kwargs["env"]["OPENAI_API_KEY"] == "private-test-value"
        assert kwargs["env"]["DB_NAME"] == "tpch50"
        assert "private-test-value" not in json.dumps(kwargs["request"])
        return {"status": "error", "stdout": "", "stderr": "PermissionError: private-test-value", "elapsed_s": 0.1}
    monkeypatch.setattr(fresh, "_run_process", failed_process)
    result = fresh.run_bound_standalone(root, request, environment)
    assert result["status"] == "error" and fresh._configuration_failure(result)
    assert "private-test-value" not in (root / "gateway/gateway_stderr.log").read_text()


def test_fresh_registry_preserves_actual_session_role_context_without_claiming_published_settings(fixture_catalog):
    from mcts_rewrite.upstream_runtime import model_context
    root, _, inputs, databases, row = fixture_catalog
    context = model_context("rbot", {"MCTS_MODEL_TRANSPORT": "codex_session_file_spool"})
    context.update({"formal": True, "published_model_configuration_reproduced": True})
    row["provenance"].update({"model_context": context, "session_completion_count": 1,
        "session_completed_request_ids": ["req_" + "a" * 32], "session_workers": ["unit-test-harness"],
        "session_completion_audit_sha256": "b" * 64})
    catalog = root / "catalog.jsonl"
    fresh.write_catalog(catalog, [row])
    result = fresh.FreshBaselineRegistry(root, catalog, databases=databases).lookup_for_dataset(
        "rbot", inputs[0]["sql"], "tpch", "tpch10")
    preserved = result["provenance"]["model_context"]
    assert preserved["actual_serving_model"] == "gpt-6.1-sol" and preserved["reasoning_effort"] == "xhigh"
    assert preserved["requested_role_models"]["rewrite"] == "gpt-4o"
    assert preserved["embedding_source"]["model"] == "text-embedding-3-small"
    assert preserved["published_model_configuration_reproduced"] is False and "formal" not in preserved
    row["provenance"]["session_completion_count"] = 0
    fresh.write_catalog(catalog, [row])
    assert not fresh.FreshBaselineRegistry(root, catalog, databases=databases).records
