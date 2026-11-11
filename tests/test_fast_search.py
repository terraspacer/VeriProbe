from __future__ import annotations

import math

import pytest

from mcts_rewrite.fast_search import FastSearchConfig, FastSearchController


def plan(cost):
    return {"Plan": {"Node Type": "Seq Scan", "Total Cost": cost,
                     "Plan Rows": 10, "Plan Width": 8}}


def proposal(number, parent=0, **extra):
    return {"sql": f"SELECT {number}", "parent_id": parent,
            "action_class": "B", "action": "test_rewrite", **extra}


class Database:
    def __init__(self, times=None, costs=None, log=None):
        self.timeout_ms = 30_000
        self.times = times or {}
        self.costs = costs or {}
        self.log = log if log is not None else []
        self.measurements = []
        self.explain_timeouts = []

    def explain(self, sql, analyze=False):
        assert analyze is False, "search must never EXPLAIN ANALYZE a node"
        self.log.append(("explain", sql))
        self.explain_timeouts.append(self.timeout_ms)
        return plan(self.costs.get(sql, 100))

    def measure(self, sql, runs=3, warmups=1):
        self.log.append(("canary", sql))
        self.measurements.append((sql, runs, warmups, self.timeout_ms))
        value = self.times.get(sql, 5)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return {"status": "timeout", "median_ms": None, "samples_ms": []}
        return {"status": "ok", "median_ms": value, "samples_ms": [value]}


class Gate:
    def __init__(self, reject=(), log=None):
        self.reject = set(reject)
        self.calls = []
        self.log = log if log is not None else []

    def verify(self, original, candidate):
        self.calls.append((original, candidate))
        self.log.append(("verify", candidate))
        return {"accepted": candidate not in self.reject, "formal": False,
                "proof": "differential", "reason": "finite miniature evidence"}


class Generator:
    def __init__(self, batches, predictions=None):
        self.batches = batches
        self.predictions = predictions or {}
        self.generation_calls = []
        self.critic_calls = []
        self.last_diagnostics = []
        self.begun = []

    def begin_query(self, sql, schema):
        self.begun.append((sql, schema))

    def generate_batch(self, parents, schema, **kwargs):
        self.generation_calls.append((parents, kwargs))
        return self.batches.get(kwargs["depth"], [])

    def macro_batch(self, parents, schema, **kwargs):
        assert kwargs.pop("operator") == "quite"
        return self.generate_batch(parents, schema, **kwargs)

    def critic_batch(self, candidates, **kwargs):
        self.critic_calls.append((candidates, kwargs))
        return {row["id"]: {"predicted_ms": self.predictions.get(row["sql"], 1),
                            "source": "llm_batched_cost_critic"} for row in candidates}


def config(**kwargs):
    return FastSearchConfig(use_rag=False, **kwargs)


def test_two_generation_batches_one_joint_critic_and_only_two_single_canaries():
    log = []
    first = [proposal(index) for index in range(1, 5)]
    second = [proposal(parent * 10 + index, parent) for parent in range(1, 4) for index in range(3)]
    generator = Generator({1: first, 2: second}, {"SELECT 32": 0.1, "SELECT 10": 0.2})
    db = Database(log=log)
    gate = Gate(log=log)
    result = FastSearchController(db, gate, generator, config()).run("SELECT 0", "", original_ms=10)

    assert len(generator.generation_calls) == 2
    assert len(generator.generation_calls[1][0]) == 3
    assert len(generator.critic_calls) == 1
    assert len(generator.critic_calls[0][0]) == 12
    assert result["generation_batches"] == 2 and result["critic_batches"] == 1
    assert result["expanded"] == 4
    assert [entry[:3] for entry in db.measurements] == [("SELECT 32", 1, 0), ("SELECT 10", 1, 0)]
    assert all(entry[3] == math.ceil(1.05 * 10) for entry in db.measurements)
    assert db.timeout_ms == 30_000
    assert all(original == "SELECT 0" for original, _ in gate.calls)
    assert not any(sql == "SELECT 0" for sql, *_ in db.measurements)
    for _, sql in gate.calls:
        assert log.index(("verify", sql)) < log.index(("explain", sql))
    assert result["verification"]["proof"] == "differential"
    assert result["verification"]["formal"] is False
    assert result["selection_validation"]["paired_checks"] == 0
    assert result["selection_validation"]["final_holdout_used"] is False


@pytest.mark.parametrize("bad", [None, 10.01, TimeoutError("too slow"), float("nan")])
def test_first_bad_canary_continues_to_next_candidate(bad):
    generator = Generator({1: [proposal(1), proposal(2)]})
    db = Database(times={"SELECT 1": bad})
    result = FastSearchController(db, Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert result["sql"] == "SELECT 2"
    assert result["fallback"] is False
    assert result["selection_validation"]["aborted"] is False
    assert len(db.measurements) == 2
    assert db.timeout_ms == 30_000


def test_second_bad_canary_retains_a_faster_first_candidate():
    generator = Generator({1: [proposal(1), proposal(2)]})
    db = Database(times={"SELECT 1": 2, "SELECT 2": 11})
    result = FastSearchController(db, Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert len(result["canaries"]) == 2
    assert result["sql"] == "SELECT 1" and result["measurement"]["median_ms"] == 2
    assert result["selection_validation"]["aborted"] is False


def test_all_failed_canaries_fall_back_only_after_testing_both_slots():
    generator = Generator({1: [proposal(1), proposal(2)]})
    db = Database(times={"SELECT 1": None, "SELECT 2": 10})
    result = FastSearchController(db, Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert [entry[0] for entry in db.measurements] == ["SELECT 1", "SELECT 2"]
    assert result["sql"] == "SELECT 0" and result["measurement"] is None
    assert result["selection_validation"]["aborted"] is False
    assert "did not improve" in result["selection_validation"]["reason"]


def test_unverified_duplicate_and_wrong_parent_nodes_never_reach_plan_or_critic():
    log = []
    generator = Generator({1: [proposal(1), proposal(2), proposal(2)],
                           2: [proposal(3, 1), proposal(4, 900), proposal(5, [])]})
    gate = Gate(reject={"SELECT 1", "SELECT 3"}, log=log)
    result = FastSearchController(Database(log=log), gate, generator, config()).run("SELECT 0", "", original_ms=10)
    assert {node["sql"] for node in result["nodes"]} == {"SELECT 0", "SELECT 2"}
    assert ("explain", "SELECT 1") not in log
    assert ("explain", "SELECT 3") not in log
    assert len(generator.critic_calls[0][0]) == 1
    assert all(original == "SELECT 0" for original, _ in gate.calls)


def test_first_beam_uses_plain_explain_cost_before_any_critic_request():
    generator = Generator({1: [proposal(1), proposal(2), proposal(3)]}, {"SELECT 1": 0.001})
    db = Database(costs={"SELECT 1": 50, "SELECT 2": 20, "SELECT 3": 30})
    result = FastSearchController(db, Gate(), generator, config(beam_width=2)).run("SELECT 0", "", original_ms=10)
    assert [row["sql"] for row in generator.generation_calls[1][0]] == ["SELECT 2", "SELECT 3"]
    assert result["canaries"][0]["sql"] == "SELECT 1"
    assert generator.critic_calls[0][1]["observations"][0]["ms"] == 10
    assert result["critic"]["target_candidate_observations"] == 0


def test_missing_critic_predictions_fall_back_to_original_calibrated_pg_cost():
    generator = Generator({1: [proposal(1), proposal(2)]})
    generator.critic_batch = lambda *args, **kwargs: {}
    db = Database(costs={"SELECT 0": 100, "SELECT 1": 90, "SELECT 2": 10})
    result = FastSearchController(db, Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert result["canaries"][0]["sql"] == "SELECT 2"
    assert result["nodes"][2]["critic"]["predicted_ms"] == 1


def test_mcts_canaries_are_stratified_between_symbolic_and_macro_actions():
    symbolic = {**proposal(1), "action_class": "A", "action": "CALCITE_RULE"}
    macro = {**proposal(2), "action_class": "B", "action": "LLM_MACRO"}
    generator = Generator({1: [symbolic, macro]}, {"SELECT 1": 1, "SELECT 2": 0.1})
    db = Database(times={"SELECT 1": 1, "SELECT 2": 1},
                 costs={"SELECT 0": 100, "SELECT 1": 20, "SELECT 2": 10})
    result = FastSearchController(db, Gate(), generator, config(max_depth=1)).run(
        "SELECT 0", "", original_ms=10)

    assert [receipt["sql"] for receipt in result["canaries"]] == ["SELECT 1", "SELECT 2"]
    assert result["sql"] == "SELECT 1"
    assert result["fallback"] is False


def test_mcts_keeps_much_faster_macro_winner():
    symbolic = {**proposal(1), "action_class": "A", "action": "CALCITE_RULE"}
    macro = {**proposal(2), "action_class": "B", "action": "LLM_MACRO"}
    generator = Generator({1: [symbolic, macro]}, {"SELECT 1": 1, "SELECT 2": 0.1})
    db = Database(times={"SELECT 1": 10, "SELECT 2": 1},
                 costs={"SELECT 0": 100, "SELECT 1": 20, "SELECT 2": 10})
    result = FastSearchController(db, Gate(), generator, config(max_depth=1)).run(
        "SELECT 0", "", original_ms=20)
    assert result["sql"] == "SELECT 2"


def test_mcts_allows_promising_macro_candidate_with_higher_planner_cost():
    symbolic = {**proposal(1), "action_class": "A", "action": "CALCITE_RULE"}
    macro = {**proposal(2), "action_class": "B", "action": "LLM_MACRO"}
    generator = Generator({1: [symbolic, macro]}, {"SELECT 1": 1, "SELECT 2": 0.1})
    db = Database(times={"SELECT 1": 1, "SELECT 2": 1},
                 costs={"SELECT 0": 100, "SELECT 1": 20, "SELECT 2": 200})
    result = FastSearchController(db, Gate(), generator, config(max_depth=1)).run(
        "SELECT 0", "", original_ms=20)
    assert [receipt["sql"] for receipt in result["canaries"]] == ["SELECT 1", "SELECT 2"]


def test_plan_timeout_is_bounded_and_restored():
    db = Database()
    controller = FastSearchController(db, Gate(), Generator({}), config())
    controller._plan("SELECT 0")
    assert db.explain_timeouts == [15_000]
    assert db.timeout_ms == 30_000


def test_quite_baseline_can_select_verified_ranked_candidate_without_target_execution():
    generator = Generator({1: [proposal(1)]})
    db = Database()
    result = FastSearchController(db, Gate(), generator, config(max_canaries=0), operator="quite").run("SELECT 0", "", original_ms=10)
    assert result["sql"] == "SELECT 1" and result["measurement"] is None
    assert not db.measurements
    assert result["selection_validation"]["scope"] == "explain_and_critic_without_canary"
    with pytest.raises(ValueError, match="requires a bounded canary"):
        FastSearchController(db, Gate(), generator, config(max_canaries=0))


@pytest.mark.parametrize("operator,max_canaries", [("mcts", 2), ("quite", 0)])
def test_slow_or_equal_predictions_are_pruned_and_original_wins_without_execution(operator, max_canaries):
    generator = Generator({1: [proposal(1), proposal(2)]}, {"SELECT 1": 10, "SELECT 2": 20})
    db = Database()
    result = FastSearchController(db, Gate(), generator, config(max_canaries=max_canaries),
                                  operator=operator).run("SELECT 0", "", original_ms=10)
    assert result["sql"] == "SELECT 0" and result["fallback"] is True
    assert not db.measurements and result["target_candidate_executions"] == 0
    assert len(generator.critic_calls) == 1
    assert len(result["ranking"]) == 2
    assert all(row["eligible_after_pruning"] is False for row in result["ranking"])
    assert result["promising_node_ids"] == [] and result["pruned_node_ids"] == [1, 2]
    assert result["root_reference"] == {"node_id": 0, "original_ms": 10}
    assert "prediction pruning" in result["selection_validation"]["reason"]
    assert next(event for event in result["trace"] if event["event"] == "prediction_pruning")["pruned"] == result["pruned_ranking"]


def test_pg_cost_fallback_also_prunes_predictions_and_only_promising_nodes_get_canaries():
    generator = Generator({1: [proposal(1), proposal(2), proposal(3)]})
    generator.critic_batch = lambda *args, **kwargs: {}
    db = Database(costs={"SELECT 0": 100, "SELECT 1": 50, "SELECT 2": 100, "SELECT 3": 150})
    result = FastSearchController(db, Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert [receipt["sql"] for receipt in result["canaries"]] == ["SELECT 1"]
    assert result["sql"] == "SELECT 1"
    assert result["promising_node_ids"] == [1] and result["pruned_node_ids"] == [2, 3]


def test_shared_canary_budget_is_respected_and_original_time_is_required():
    db, generator = Database(), Generator({1: [proposal(1), proposal(2)]})
    budget = {"remaining": 1}
    controller = FastSearchController(db, Gate(), generator, config(), canary_budget=budget)
    result = controller.run("SELECT 0", "", original_ms=10)
    assert result["target_candidate_executions"] == 1 and budget["remaining"] == 0
    second = controller.run("SELECT 0", "", original_ms=10)
    assert second["fallback"] is True and second["target_candidate_executions"] == 0
    for invalid in (None, 0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="original_ms"):
            controller.run("SELECT 0", "", original_ms=invalid)


def test_controller_and_real_batched_operator_api_share_three_model_requests(tmp_path):
    from mcts_rewrite.batched_operators import BatchedOperators

    class Calcite:
        def match(self, sql, schema_sql):
            return {"status": "ok", "matches": [{"name": "TEST_RULE"}]}

        def rewrite(self, sql, schema_sql, sequences):
            number = int(sql.split()[1]) + 10
            return {"status": "ok", "rewrites": [{"sql": f"SELECT {number}", "rules": seq}
                                                    for seq in sequences]}

    calls = []

    def model(payload):
        calls.append(payload)
        if payload["task"] == "mcts_batched_generation":
            return {"batches": [{"parent_id": parent["parent_id"],
                                  "rule_sequences": [["TEST_RULE"]],
                                  "candidates": [{"sql": f"SELECT {int(parent['sql'].split()[1]) + 1}"}]}
                                 for parent in payload["parents"]]}
        assert payload["task"] == "mcts_batched_cost_critic"
        return {"predictions": [{"id": row["id"], "predicted_ms": 1 if row["sql"] == "SELECT 20" else 2}
                                 for row in payload["candidates"]]}

    generator = BatchedOperators(tmp_path, model, calcite=Calcite())
    result = FastSearchController(Database(), Gate(), generator, config()).run("SELECT 0", "", original_ms=10)
    assert len(calls) == 3
    assert result["model_requests"] == result["model_call_cap"] == 3
    # Both action classes pass with equal canary latency; the new symbolic
    # tie-break keeps the lower-cost verified Calcite candidate.
    assert result["sql"] == "SELECT 1"
    assert {event["action_class"] for event in result["trace"] if event["event"] == "candidate"} == {"A", "B"}
    assert len(calls[1]["parents"]) == 2
    assert len(calls[2]["candidates"]) == 5


@pytest.mark.parametrize("options", [{"beam_width": 4}, {"max_depth": 3},
                                     {"max_canaries": 3}, {"candidate_limit_per_parent": 17}])
def test_config_cannot_expand_the_mandatory_fast_search_bounds(options):
    with pytest.raises(ValueError):
        FastSearchConfig(**options)
