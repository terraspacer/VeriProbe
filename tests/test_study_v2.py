from copy import deepcopy

import pytest

from mcts_rewrite import study_v2 as v2


def setup(tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE t(a INT NOT NULL); CREATE TABLE u(b INT);")
    row = {"id": "rbot:tpch:query18_0", "benchmark": "tpch", "sql": "SELECT COUNT(a) FROM t", "schema_path": str(schema)}
    config = {"databases": {"tpch10": {"dsn": "performance", "benchmark": "tpch"}}}
    cases = [{"name": name, "dsn": "fixture", "search_path": "new_" + name,
              "coverage_complete": True, "schema_sha256": v2.stage.digest(schema.read_bytes()),
              "constraint_rejections": 0, "rows_by_table": {"t": 0 if name == "empty" else 3, "u": 0 if name == "empty" else 1}}
             for name in ["empty", "null", "duplicate", "skew", "unmatched"]]
    binding = {"query_sha256": v2.stage.digest(row["sql"].encode()), "schema_sha256": cases[0]["schema_sha256"],
               "source_tables": ["t", "u"], "mini_dbs": cases,
               "verification_database": {"dsn": "fixture", "search_path": "new_skew"},
               "validation": {"postgresql_executed": True, "original_expectations_passed": True,
                              "saved_candidate_checks_passed": True, "nonvacuous": True}}
    bundle = {"version": "query-witness-v2", "status": "postgresql_validated", "queries": {row["id"]: binding}}
    return config, row, bundle, binding


def test_route_only_query_correctness_to_nonempty_witness(tmp_path):
    config, row, bundle, binding = setup(tmp_path)
    before = deepcopy(config)
    selected = v2.configuration_for_query(config, row, bundle)
    assert config == before
    assert selected["databases"]["tpch10"]["dsn"] == "performance"
    verification = selected["databases"]["tpch10"]["verification_database"]
    assert {key: verification[key] for key in ("dsn", "search_path")} == binding["verification_database"]
    context = verification[v2.CONTEXT_KEY]
    assert context["query_sha256"] == binding["query_sha256"]
    assert context["schema_sha256"] == binding["schema_sha256"]
    assert context["rows_by_table"] == {"t": 3, "u": 1}
    assert len(selected["databases"]["tpch10"]["mini_dbs"]) == 5
    other = {**row, "id": "other"}
    assert v2.configuration_for_query(config, other, bundle) == config


@pytest.mark.parametrize("change", [
    lambda b, q: b.update(status="offline_plan_not_postgresql_validated"),
    lambda b, q: q.update(query_sha256="different"),
    lambda b, q: q.update(schema_sha256="different"),
    lambda b, q: q["validation"].update(postgresql_executed=False),
    lambda b, q: q["validation"].update(nonvacuous=False),
    lambda b, q: q["validation"].update(saved_candidate_checks_passed=False),
    lambda b, q: q.update(source_tables=["t"]),
    lambda b, q: q["mini_dbs"].pop(),
    lambda b, q: q["mini_dbs"][1].update(name="empty"),
    lambda b, q: q["mini_dbs"][1].update(coverage_complete=False),
    lambda b, q: q["mini_dbs"][1].update(constraint_rejections=1),
    lambda b, q: q["mini_dbs"][1]["rows_by_table"].update(u=0),
    lambda b, q: q["mini_dbs"][1]["rows_by_table"].update(t=True),
    lambda b, q: q["mini_dbs"][0]["rows_by_table"].update(t=1),
    lambda b, q: q["verification_database"].update(search_path="new_empty"),
])
def test_unmeasured_drifted_incomplete_or_empty_bindings_rejected(tmp_path, change):
    config, row, bundle, binding = setup(tmp_path)
    change(bundle, binding)
    with pytest.raises(ValueError):
        v2.configuration_for_query(config, row, bundle)


def test_wrapper_preserves_old_runner_guards_and_labels_every_checkpoint(tmp_path, monkeypatch):
    config, row, bundle, binding = setup(tmp_path)
    checkpoints = []
    components = object()
    model_factory = object()
    def run(r, c, factory, *, components, checkpoint):
        assert r is row and factory is model_factory
        assert c["databases"]["tpch10"]["verification_database"]["search_path"] == "new_skew"
        checkpoint({"status": "running"})
        return {"status": "ok"}
    monkeypatch.setattr(v2.stage, "run_query", run)
    result = v2.run_query(row, config, model_factory, fixture_bindings=bundle,
                          components=components, checkpoint=checkpoints.append)
    assert checkpoints[0]["study_variant"] == result["study_variant"]
    assert result["study_variant"]["feedback_scope"] == "transductive_stage_a_followup"


def test_default_components_only_replace_v2_operator_and_scoped_gate_classes():
    old = v2.stage.default_components()
    new = v2.default_components()
    assert new.Operators is v2.BatchedOperatorsV2
    assert new.ScopedGate is v2.ScopedStructuralEmptyGateV2
    for name in ["HTTPModel", "Postgres", "SearchConfig", "Search", "Gate"]:
        assert getattr(old, name) is getattr(new, name)
