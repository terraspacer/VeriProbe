from mcts_rewrite.search import SearchController
from tests.test_search_engine import FakeDB, FakeGate, FakeGenerator, candidate, config, plan


def test_restored_symbolic_sql_still_requires_root_verification_before_measurement():
    log = []
    db, gate = FakeDB(log=log), FakeGate(reject={"SELECT 999"}, log=log)
    generated = FakeGenerator({"SELECT 0": [candidate("SELECT 1"), candidate("SELECT 2", "B")]})
    restored = []

    def restorer(database, root, proposed):
        restored.append((database, root, proposed))
        return {"status": "restored", "sql": "SELECT 999", "policy": "test-policy", "formal": False}

    result = SearchController(db, gate, generated, config(budget=1), contract_restorer=restorer).run("SELECT 0", "")
    assert [(root, proposed) for _, root, proposed in restored] == [("SELECT 0", "SELECT 1")]
    assert ("verify", "SELECT 999") in log
    assert ("measure", "SELECT 999") not in log
    event = next(event for event in result["trace"] if event.get("raw_symbolic_sql"))
    assert event["raw_symbolic_sql"] == "SELECT 1"
    assert event["status"] == "rejected"
    assert event["provenance"]["output_contract_policy"] == "test-policy"
    assert ("verify", "SELECT 2") in log


def test_unsupported_restoration_retains_raw_proposal_for_normal_gate():
    gate = FakeGate()
    generated = FakeGenerator({"SELECT 0": [candidate("SELECT 1")]})
    result = SearchController(FakeDB(), gate, generated, config(budget=1),
        contract_restorer=lambda *args: {"status": "rejected", "reason": "unsupported ordering"}).run("SELECT 0", "")
    assert ("SELECT 0", "SELECT 1") in gate.calls
    assert result["sql"] == "SELECT 1"


def test_next_expansion_uses_measured_loops_without_supplying_holdout_data():
    class DB(FakeDB):
        def measure(self, sql, runs=3, warmups=1):
            result = super().measure(sql, runs, warmups)
            result["plan"]["Plan"]["Actual Loops"] = 1
            return result

        def explain(self, sql, analyze=False):
            return plan(99999)

    generated = FakeGenerator({"SELECT 0": [candidate("SELECT 1", "B")]})
    result = SearchController(DB(), FakeGate(), generated, config(budget=2)).run("SELECT 0", "")
    assert len(generated.calls) == 2
    assert generated.calls[1]["plan"]["Plan"]["Actual Loops"] == 1
    assert generated.calls[1]["plan"]["Plan"]["Total Cost"] != 99999
    assert result["calibration"]["final_holdout_used"] is False
