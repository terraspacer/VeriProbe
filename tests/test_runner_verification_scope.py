import json

import pytest

import benchmark_runner as runner
from mcts_rewrite.reporting import summarize_group
from tests.test_benchmark_reporting import proof, record


def test_verification_database_selection_is_explicit_and_dataset_override_wins():
    config = {"verification_databases": {"tpch": {"dsn": "benchmark"}, "tpch50": {"dsn": "dataset"}}}
    select = runner.verification_database_for
    assert select(config, "tpch50", {"benchmark": "tpch"})["dsn"] == "dataset"
    assert select(config, "tpch10", {"benchmark": "tpch"})["dsn"] == "benchmark"
    assert select(config, "tpch50", {"benchmark": "tpch", "verification_database": {"dsn": "override"}})["dsn"] == "override"
    assert select(config, "tpch50", {"benchmark": "tpch", "verification_database": None}) is None
    assert select({}, "tpch50", {"benchmark": "tpch"}) is None
    with pytest.raises(ValueError, match="dsn"):
        select({"verification_databases": {"tpch": {"scale": 1}}}, "tpch50", {"benchmark": "tpch"})


@pytest.mark.parametrize("catalog_ok", [True, False])
def test_runner_routes_only_correctness_to_test_database_and_closes_both(tmp_path, monkeypatch, catalog_ok):
    from mcts_rewrite.search import SearchConfig
    from mcts_rewrite.verification_db import VerificationDatabaseError

    events = []

    class DB:
        def __init__(self, dsn, **kwargs):
            self.dsn = dsn
            events.append(("open", dsn))

        def measure(self, sql, runs=3, warmups=1):
            assert self.dsn == "performance"
            events.append(("measure", sql))
            return {"status": "ok", "median_ms": 1.0, "samples_ms": [1.0], "plan": {}}

        def close(self):
            events.append(("close", self.dsn))

    class Gate:
        def __init__(self, performance_db, verification_db, schema, **kwargs):
            assert performance_db.dsn == "performance" and verification_db.dsn == "test"
            assert kwargs["verification_provenance"] == {"scale": 0.01}
            if not catalog_ok:
                raise VerificationDatabaseError("mismatch", {"compatible": False, "mismatches": ["table:t"]})
            self.metadata = {"scope": "separate_verification_database", "performance_database_rows_hashed": False}

        def verify(self, original, candidate):
            events.append(("verify", candidate))
            return proof() if candidate == original else {"accepted": False, "formal": False, "proof": "none"}

    class Generator:
        def __init__(self, *args, **kwargs):
            pass

    class Controller:
        def __init__(self, db, gate, generator, config):
            assert db.dsn == "performance"

        def run(self, original, schema):
            return {"sql": "SELECT 999", "trace": [], "expanded": 1}

    class Registry:
        def lookup(self, *args):
            return None

    monkeypatch.setattr("mcts_rewrite.verification_db.ScopedEquivalenceGate", Gate)
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE t(a INT);")
    row = {"id": "q", "source": "rbot", "benchmark": "tpch", "sql": "SELECT 1",
           "schema_path": str(schema), "source_path": "q.sql", "sha256": "s", "source_commit": "c"}
    task = {"query": row, "dataset": "tpch50", "database": {"dsn": "performance", "benchmark": "tpch"},
            "budget": 1, "ablation": "full"}
    config = {"verification_databases": {"tpch": {"dsn": "test", "scale": 0.01}}}
    result = runner.run_task(task, config, "run", Registry(), (DB, Gate, Generator, SearchConfig, Controller))
    assert ("close", "performance") in events and ("close", "test") in events
    if catalog_ok:
        assert result["final_sql"] == "SELECT 1"
        assert result["verification_context"]["performance_database_rows_hashed"] is False
        assert all(sql != "SELECT 999" for kind, sql in events if kind == "measure")
    else:
        assert result["status"] == "error"
        assert not any(kind == "measure" for kind, _ in events)
        assert result["verification_context"]["catalog_validation"]["compatible"] is False
    assert '"dsn"' not in json.dumps(result)


def test_reporting_separates_test_rows_from_performance_rows():
    main, test = record("main"), record("test")
    test["verification"]["mcts"]["evidence_scope"] = {
        "scope": "separate_verification_database", "performance_database_rows_hashed": False}
    report = summarize_group([main, test], 2)
    assert report["final_test_database_compared_queries"] == 1
    assert report["final_performance_database_compared_queries"] == 1
    assert report["verified_equivalence_rate_all_queries"] == 1
