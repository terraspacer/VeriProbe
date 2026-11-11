from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mcts_rewrite.critic import CostCritic
from mcts_rewrite.adapters import FileModel, plan_sha256
from mcts_rewrite.evidence import EvidenceStore, _sha256, _template_sha256
from mcts_rewrite.plans import bottleneck, operators, plan_features
from mcts_rewrite.search import SearchConfig, SearchController, _key


def plan(cost=100, rows=100, kind="Seq Scan"):
    return {"Plan": {"Node Type": kind, "Total Cost": cost, "Plan Rows": rows,
                     "Plan Width": 8, "Relation Name": "items"}}


class FakeDB:
    def __init__(self, times=None, validation=None, log=None):
        self.times = times or {"SELECT 0": 10, "SELECT 1": 5, "SELECT 2": 6}
        self.validation = validation or {}
        self.log = log if log is not None else []
        self.validation_indexes = {}
        self.calls = []

    def explain(self, sql, analyze=False):
        self.log.append(("explain", sql))
        return plan(self.times.get(sql, 10) * 10)

    def measure(self, sql, runs=3, warmups=1):
        self.log.append(("measure", sql))
        self.calls.append((sql, runs, warmups))
        value = self.times.get(sql, 10)
        if runs == 1:
            index = self.validation_indexes.get(sql, 0)
            values = self.validation.get(sql, [value])
            value = values[min(index, len(values) - 1)]
            self.validation_indexes[sql] = index + 1
        if value is None:
            return {"status": "timeout", "median_ms": None, "samples_ms": [], "plan": plan()}
        return {"status": "ok", "median_ms": value, "samples_ms": [value] * runs,
                "plan": plan(value * 10), "estimated_cost": value * 10}


class FakeGate:
    def __init__(self, reject=(), log=None, accept=None):
        self.reject = set(reject)
        self.calls = []
        self.log = log if log is not None else []
        self.accept = accept

    def verify(self, original, candidate):
        self.calls.append((original, candidate))
        self.log.append(("verify", candidate))
        if candidate == "SELECT 404":
            raise RuntimeError("solver unavailable")
        accepted = self.accept(original, candidate) if self.accept else candidate not in self.reject
        return {"accepted": accepted, "formal": accepted, "proof": "ast" if accepted else "none",
                "execution": {"equivalent": accepted}, "reason": "test proof" if accepted else "different root result"}


def candidate(sql, action_class="A"):
    return {"sql": sql, "action_class": action_class, "action": "test action", "provenance": {"test": True}}


class FakeGenerator:
    model = None

    def __init__(self, batches):
        self.batches = batches
        self.calls = []

    def candidates(self, sql, schema_sql, query_plan, detail, evidence, limit=4):
        self.calls.append({"sql": sql, "plan": query_plan, "detail": detail, "evidence": evidence})
        batch = self.batches.get(sql, [])
        if isinstance(batch, Exception):
            raise batch
        return batch


def config(**kwargs):
    return SearchConfig(use_rag=False, use_critic=False, **kwargs)


def test_budget_is_expansion_calls_and_propagates_descendant_rewards():
    batches = {f"SELECT {index}": [candidate(f"SELECT {index + 1}", "B" if index % 2 else "A")] for index in range(20)}
    times = {f"SELECT {index}": 100 / (index + 1) for index in range(21)}
    generator, gate = FakeGenerator(batches), FakeGate()
    result = SearchController(FakeDB(times), gate, generator, config(budget=16, use_safety=False)).run("SELECT 0", "")
    assert result["expanded"] == len(generator.calls) == 16
    assert len(result["nodes"]) == 17
    assert result["sql"] == "SELECT 16"
    assert result["nodes"][0]["visits"] == 16
    assert result["nodes"][1]["visits"] == 16
    assert result["nodes"][2]["visits"] == 15
    assert result["nodes"][0]["total_reward"] > result["nodes"][2]["total_reward"]
    assert all(original == "SELECT 0" for original, _ in gate.calls)
    with pytest.raises(ValueError, match="16"):
        SearchConfig(budget=17)


def test_budget_allows_four_candidates_per_each_of_sixteen_expansions():
    batches = {f"SELECT {index}": [candidate(f"SELECT {4 * index + offset}")
                                  for offset in range(1, 5)] for index in range(1000)}
    generator = FakeGenerator(batches)
    result = SearchController(FakeDB(), FakeGate(), generator,
        config(budget=16, use_safety=False)).run("SELECT 0", "")
    assert result["budget_unit"] == "node_expansions"
    assert result["expanded"] == len(generator.calls) == 16
    assert result["maximum_candidate_slots"] == 64
    assert len(result["nodes"]) == 65
    assert len([event for event in result["trace"] if event["event"] == "candidate"]) == 64


def test_all_action_classes_verified_against_root_before_insertion_and_timing():
    log = []
    gate = FakeGate(accept=lambda original, child: original == "SELECT 0" and child != "SELECT 2", log=log)
    generator = FakeGenerator({"SELECT 0": [candidate("SELECT 1", "A")],
                               "SELECT 1": [candidate("SELECT 2", "B")]})
    result = SearchController(FakeDB(log=log), gate, generator, config(budget=4, use_safety=False)).run("SELECT 0", "")
    assert ("SELECT 0", "SELECT 2") in gate.calls
    assert ("SELECT 1", "SELECT 2") not in gate.calls
    assert len(result["nodes"]) == 2
    assert ("measure", "SELECT 2") not in log
    assert log.index(("verify", "SELECT 1")) < log.index(("explain", "SELECT 1")) < log.index(("measure", "SELECT 1"))
    assert [event["status"] for event in result["trace"] if event["event"] == "candidate"] == ["accepted", "rejected"]


def test_trace_preserves_duplicates_rejections_errors_and_generator_error():
    generator = FakeGenerator({"SELECT 0": [candidate(" select   0 ;"), candidate("SELECT 9"),
                                             candidate("SELECT 404"), candidate("SELECT 1")],
                               "SELECT 1": RuntimeError("provider error")})
    result = SearchController(FakeDB(), FakeGate(reject=["SELECT 9"]), generator,
                              config(budget=3, use_safety=False)).run("SELECT 0", "")
    events = [event for event in result["trace"] if event["event"] == "candidate"]
    assert [event["status"] for event in events] == ["duplicate", "rejected", "error", "accepted"]
    assert all(event["attempted"] for event in events)
    assert "verification error" in events[2]["reason"]
    assert any(event["event"] == "generation_error" for event in result["trace"])
    assert result["expanded"] == 2
    assert _key("SELECT 'a  b'") != _key("SELECT 'a b'")
    assert _key('SELECT "X"') != _key('SELECT "x"')
    assert _key("SELECT /*+ IndexScan(t) */ * FROM t") != _key("SELECT * FROM t")


def test_default_safety_uses_fresh_alternating_pairs_and_no_holdout():
    db = FakeDB(validation={"SELECT 0": [10] * 7, "SELECT 1": [7] * 7})
    result = SearchController(db, FakeGate(), FakeGenerator({"SELECT 0": [candidate("SELECT 1")]}),
                              config(budget=1)).run("SELECT 0", "")
    assert result["sql"] == "SELECT 1" and not result["fallback"]
    validation = result["selection_validation"]
    assert validation["accepted"] and len(validation["pairs"]) == 7
    assert validation["one_sided_99pct_log_t_upper_ratio"] == pytest.approx(0.7)
    assert validation["sign_test_p_if_all_pairs_improve"] < .01
    assert len([call for call in db.calls if call[1:] == (3, 1)]) == 2
    assert len([call for call in db.calls if call[1:] == (1, 0)]) == 14
    orders = [pair["order"] for pair in validation["pairs"]]
    assert all(orders[index] == list(reversed(orders[index - 1])) for index in range(1, 7))
    assert result["calibration"]["final_holdout_used"] is False
    assert validation["guaranteed_regression_rate"] is None


def test_noisy_winner_falls_back_without_trying_another_validation_winner():
    db = FakeDB(validation={"SELECT 1": [4, 4, 12, 4, 4, 4, 4]})
    generator = FakeGenerator({"SELECT 0": [candidate("SELECT 1"), candidate("SELECT 2")]})
    result = SearchController(db, FakeGate(), generator, config(budget=1)).run("SELECT 0", "")
    assert result["sql"] == "SELECT 0" and result["fallback"]
    assert result["selection_validation"]["accepted"] is False
    assert db.validation_indexes.get("SELECT 2", 0) == 0


def test_safety_measurement_timeout_falls_back_and_preserves_failure():
    db = FakeDB(validation={"SELECT 1": [None]})
    result = SearchController(db, FakeGate(), FakeGenerator({"SELECT 0": [candidate("SELECT 1")]}),
                              config(budget=1)).run("SELECT 0", "")
    assert result["fallback"] and result["selection_validation"]["pairs"][0]["candidate"]["status"] == "timeout"


def test_root_remains_fallback_when_no_verified_or_measurable_rewrite():
    result = SearchController(FakeDB(), FakeGate(reject=["SELECT 1"]),
        FakeGenerator({"SELECT 0": [candidate("SELECT 1")]}), config(budget=16)).run("SELECT 0", "")
    assert result["fallback"] and result["sql"] == "SELECT 0"
    assert result["expanded"] == 1
    empty = SearchController(FakeDB(), FakeGate(), FakeGenerator({}), config(budget=0)).run("SELECT 0", "")
    assert empty["expanded"] == 0 and empty["nodes"][0]["id"] == 0


def test_plan_and_rag_ablation_remove_feedback_but_keep_gate():
    generator, gate = FakeGenerator({"SELECT 0": [candidate("SELECT 1", "B")]}), FakeGate()
    result = SearchController(FakeDB(), gate, generator,
        SearchConfig(budget=1, use_plan=False, use_rag=False, use_critic=True, use_safety=False),
        evidence={"secret": "must not be used"}).run("SELECT 0", "")
    assert generator.calls[0]["plan"] == generator.calls[0]["detail"] == generator.calls[0]["evidence"] == {}
    assert ("SELECT 0", "SELECT 1") in gate.calls
    assert result["critic"]["calibration_observations"] == 2
    accepted = next(event for event in result["trace"] if event.get("status") == "accepted")
    assert accepted["critic"]["features"]["total_cost"] == 0


def test_bottleneck_attributes_exclusive_work_and_rescans_below_root():
    document = {"Plan": {"Node Type": "Aggregate", "Total Cost": 1012, "Plan Rows": 1, "Plans": [
        {"Node Type": "Nested Loop", "Total Cost": 1010, "Plan Rows": 100, "Plans": [
            {"Node Type": "Seq Scan", "Total Cost": 10, "Plan Rows": 100, "Parent Relationship": "Outer"},
            {"Node Type": "Seq Scan", "Total Cost": 10, "Plan Rows": 1, "Parent Relationship": "Inner", "Relation Name": "repeated"}]}]}}
    detail = bottleneck(document)
    assert detail["node_type"] == "Seq Scan" and detail["relation"] == "repeated"
    assert detail["effective_exclusive_cost"] == 1000
    assert detail["estimated_loops"] == 100
    assert next(row for row in detail["ranked"] if row["node_type"] == "Aggregate")["exclusive_cost"] == 2
    assert next(row for row in detail["ranked"] if row["node_type"] == "Nested Loop")["exclusive_cost"] == 0
    assert detail["estimated_cost_is_latency"] is False


def test_correlated_subplan_bottleneck_and_actual_time_avoid_inclusive_root():
    document = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 1100, "Plan Rows": 100, "Plans": [
        {"Node Type": "Aggregate", "Total Cost": 10, "Plan Rows": 1, "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1"}]}}
    assert bottleneck(document)["kind"] == "correlated_subplan"
    assert plan_features(document)["subplans"] == 1
    measured = {"Plan": {"Node Type": "Aggregate", "Total Cost": 900, "Actual Total Time": 20, "Actual Loops": 1, "Plans": [
        {"Node Type": "Seq Scan", "Total Cost": 100, "Actual Total Time": 19, "Actual Loops": 1}]}}
    assert bottleneck(measured)["node_type"] == "Seq Scan"
    assert bottleneck(measured)["ranking_metric"] == "actual_exclusive_ms"
    assert bottleneck({})["available"] is False


def test_hashed_not_in_subplan_builds_once_while_scalar_subplan_rescans():
    document = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 1100, "Plan Rows": 100,
        "Filter": "((NOT (hashed SubPlan 1)) AND (value > (SubPlan 2)))", "Plans": [
            {"Node Type": "Seq Scan", "Total Cost": 10, "Plan Rows": 10,
             "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1"},
            {"Node Type": "Aggregate", "Total Cost": 10, "Plan Rows": 1,
             "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 2"}]}}
    rows = operators(document)
    hashed, scalar = rows[1:]
    assert hashed["kind"] == "hashed_subplan" and hashed["estimated_loops"] == 1
    assert hashed["effective_exclusive_cost"] == 10
    assert scalar["kind"] == "correlated_subplan" and scalar["estimated_loops"] == 100
    assert scalar["effective_exclusive_cost"] == 1000
    assert bottleneck(document)["subplan_name"] == "SubPlan 2"
    assert plan_features(document)["subplans"] == 2
    assert rows[0]["exclusive_cost"] == 90


def test_hashed_subplan_does_not_inherit_nested_loop_parent_rescans():
    document = {"Plan": {"Node Type": "Nested Loop", "Total Cost": 1000, "Plan Rows": 100,
        "Plans": [
            {"Node Type": "Seq Scan", "Total Cost": 1, "Plan Rows": 100,
             "Parent Relationship": "Outer"},
            {"Node Type": "Seq Scan", "Total Cost": 10, "Plan Rows": 1,
             "Parent Relationship": "Inner", "Filter": "(NOT (hashed SubPlan 1))", "Plans": [
                {"Node Type": "Seq Scan", "Total Cost": 2, "Plan Rows": 10,
                 "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1"}]}]}}
    rows = operators(document)
    assert rows[2]["estimated_loops"] == 100
    assert rows[3]["kind"] == "hashed_subplan" and rows[3]["estimated_loops"] == 1


def test_critic_calibrates_pg_features_and_never_accepts_holdout():
    critic = CostCritic()
    assert critic.score(plan())["predicted_ms"] is None
    critic.observe(plan(100), {"status": "ok", "median_ms": 10})
    assert critic.score(plan(200))["predicted_ms"] == pytest.approx(20)
    for cost, latency in [(200, 19), (300, 32), (400, 39)]:
        critic.observe(plan(cost, rows=cost / 10), {"status": "ok", "median_ms": latency})
    score = critic.score(plan(250))
    assert score["calibration_method"] == "ridge_log_residual_on_pg_features"
    assert math.isfinite(score["predicted_ms"]) and score["predicted_ms"] > 0
    with pytest.raises(ValueError, match="holdout"):
        critic.observe(plan(), {"status": "ok", "median_ms": 1}, phase="holdout")
    assert critic.metadata()["calibration_observations"] == 4
    assert critic.metadata()["holdout_observations"] == 0


def test_llm_critic_callback_calibrates_and_fallback_is_explicit():
    calls = []
    def model(payload):
        calls.append(payload)
        return {"predicted_ms": 100, "rationale": "operator cardinality"}
    critic = CostCritic(model)
    predicted = critic.score(plan())
    critic.observe(plan(), {"status": "ok", "median_ms": 20}, predicted)
    calibrated = critic.score(plan())
    assert calibrated["source"] == "llm" and calibrated["predicted_ms"] == pytest.approx(20)
    assert calibrated["llm_calibration_observations"] == 1
    assert calls[1]["task"] == "cost_critic" and calls[1]["pg_features"]["seq_scans"] == 1
    unavailable = CostCritic(lambda payload: {"candidates": []})
    fallback = unavailable.score(plan())
    assert fallback["source"] == "non_llm_pg_fallback" and "llm_error" in fallback
    assert unavailable.metadata()["fallback_predictions"] == 1
    assert unavailable.metadata()["enabled"] is True
    assert unavailable.metadata()["llm_configured"] is True
    assert unavailable.metadata()["llm_available"] is False
    assert critic.metadata()["llm_predictions"] == 2


def test_critic_distinguishes_no_critic_responses_from_disabled_critic():
    class ProposalOnlyModel:
        def supports_task(self, task):
            return task != "cost_critic"
        def __call__(self, payload):
            raise AssertionError("A proposal-only provider must not be called as critic")
    critic = CostCritic(ProposalOnlyModel())
    result = critic.score(plan())
    assert result["source"] == "non_llm_pg_fallback"
    assert "no cost_critic responses" in result["llm_unavailable_reason"]
    assert critic.metadata()["enabled"] is True
    assert critic.metadata()["model_requests"] == 0
    assert critic.metadata()["llm_available"] is False


def test_file_critic_uses_only_matching_sql_and_plan_with_explicit_coverage(tmp_path):
    path = tmp_path / "critic.jsonl"
    path.write_text(json.dumps({"task": "cost_critic", "original_sql": "SELECT 1",
        "plan_sha256": plan_sha256(plan()), "predicted_ms": 12,
        "provenance": {"provider": "reviewed_model_output"}}) + "\n")
    critic = CostCritic(FileModel(path))
    predicted = critic.score(plan(), "SELECT 1")
    assert predicted["source"] == "llm" and predicted["llm_raw_ms"] == 12
    mismatch = critic.score(plan(200), "SELECT 1")
    assert mismatch["source"] == "non_llm_pg_fallback"
    assert "this SQL and plan" in mismatch["llm_unavailable_reason"]
    assert critic.metadata()["model_requests"] == 1
    assert critic.metadata()["llm_predictions"] == 1


def test_evidence_uses_real_knowledge_sources_and_query_local_memory(tmp_path):
    files = {
      "LLM4Rewrite/explain_rule/calcite_rewrite_rules_structured.jsonl": json.dumps({"name": "FILTER_INTO_JOIN", "rewrite_rules_structured": [{"conditions": "join predicate", "transformations": "filter pushdown"}]}) + "\n",
      "LLM4Rewrite/knowledge-base/rule_cluster_summaries_structured.jsonl": json.dumps({"index": 1, "conditions": "sequential scan predicate index", "transformations": "filter pushdown"}) + "\n",
      "QUITE/src/Rewrite_Middleware/Structured_Knowledge_Base/storage/knowledge_base.json": json.dumps([
         {"id": "leaking", "group": "predicate", "question_query_example": "select * from items where x = 1", "answer_summary": "filter pushdown"},
         {"id": "legit", "group": "predicate", "question_query_example": "select * from other", "answer_summary": "predicate filter pushdown index"}]),
      "QUITE/experiments_results/tpch/QUITE_tpch.json": json.dumps({"fastest_sql": "secret evaluation answer"}),
    }
    for name, content in files.items():
        path = tmp_path / "vendor" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    store = EvidenceStore(tmp_path)
    store.begin_query("SELECT * FROM items WHERE x = 1;", "CREATE TABLE items(x INT)")
    store.record_attempt("SELECT * FROM items", "rejected", {"accepted": False, "reason": "different rows"})
    retrieved = store.retrieve("SELECT * FROM items WHERE x = 1", bottleneck={"kind": "sequential_scan"})
    assert {doc["source"] for doc in retrieved["documents"]} == {"rbot", "quite"}
    assert "leaking" not in {doc["id"] for doc in retrieved["documents"]}
    assert retrieved["retrieval"]["exact_input_examples_excluded"] == 1
    assert retrieved["retrieval"]["benchmark_results_loaded"] is False
    assert all(len(doc["provenance"]["sha256"]) == 64 for doc in retrieved["documents"])
    assert "secret evaluation answer" not in json.dumps(retrieved)
    assert len(retrieved["memory"]["attempts"]) == 1
    store.begin_query("SELECT * FROM new_query", "")
    assert store.memory["attempts"] == []


def approved_pattern(sql, schema, scope="template_disjoint", identity="reviewed"):
    query_hash, template_hash, schema_hash = _sha256(sql), _template_sha256(sql), _sha256(schema)
    return {"id": identity, "status": "validated", "action_class": "B",
        "conditions": ["sequential scan predicate filter"],
        "transformations": ["consider safe filter pushdown"],
        "safety_checks": ["prove equivalence against original root"],
        "trigger": {"kind": "sequential_scan"},
        "validation": {"status": "validated", "timing_scope": "search_training_only",
            "requires_root_equivalence_gate": True, "receipts": [{
                "receipt_sha256": "a" * 64, "phase": "search", "query_sha256": query_hash,
                "template_sha256": template_hash, "schema_sha256": schema_hash,
                "root_median_ms": 999, "candidate_median_ms": 1,
                "evaluation": "SECRET_HELDOUT_METRIC"}]},
        "leakage": {"scope": scope, "exclude_query_sha256": [query_hash],
            "exclude_template_sha256": [template_hash], "allow_schema_sha256": [schema_hash],
            "allow_query_sha256": [query_hash] if scope == "transductive_query_only" else []},
        "provenance": {"corpus_role": "validated_action_pattern",
                       "final_holdout": "SECRET_HELDOUT_METRIC"}}


def write_patterns(root, entries):
    path = root / "prompts/action_patterns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_validated_patterns_exclude_source_templates_schemas_and_observation_values(tmp_path):
    schema = "CREATE TABLE items(x INT)"
    entry = approved_pattern("SELECT * FROM items WHERE x = 1", schema)
    draft = json.loads(json.dumps(entry))
    draft["status"] = "draft"
    heldout = json.loads(json.dumps(entry))
    heldout["validation"]["receipts"][0]["phase"] = "holdout"
    write_patterns(tmp_path, [entry, draft, heldout])
    store = EvidenceStore(tmp_path)
    assert len(store.documents) == 1
    store.begin_query("SELECT * FROM items WHERE x = 2", schema)
    excluded = store.retrieve("SELECT * FROM items WHERE x = 2")
    assert excluded["documents"] == []
    assert excluded["retrieval"]["action_pattern_scope_excluded"] == 1
    store.begin_query("SELECT * FROM items WHERE x > 2", schema)
    retrieved = store.retrieve("SELECT * FROM items WHERE x > 2", bottleneck={"kind": "sequential_scan"})
    assert retrieved["documents"][0]["id"] == "reviewed"
    assert retrieved["retrieval"]["action_patterns_rejected"] == 2
    assert retrieved["retrieval"]["feedback_scopes"] == ["template_disjoint"]
    assert retrieved["retrieval"]["action_pattern_metrics_loaded"] is False
    assert "SECRET_HELDOUT_METRIC" not in json.dumps(retrieved)
    assert "root_median_ms" not in json.dumps(retrieved)
    assert "candidate_median_ms" not in json.dumps(retrieved)
    store.begin_query("SELECT * FROM items WHERE x > 2", "CREATE TABLE items(x BIGINT)")
    assert store.retrieve("SELECT * FROM items WHERE x > 2")["documents"] == []


def test_transductive_patterns_allow_only_the_recorded_query_and_schema(tmp_path):
    sql, schema = "SELECT * FROM items WHERE x = 1", "CREATE TABLE items(x INT)"
    write_patterns(tmp_path, [approved_pattern(sql, schema, "transductive_query_only")])
    store = EvidenceStore(tmp_path)
    store.begin_query(sql, schema)
    retrieved = store.retrieve(sql, bottleneck={"kind": "sequential_scan"})
    assert len(retrieved["documents"]) == 1
    assert retrieved["retrieval"]["transductive_patterns_used"] is True
    store.begin_query("SELECT * FROM items WHERE x = 2", schema)
    assert store.retrieve(sql)["documents"] == []
    store.begin_query(sql, "CREATE TABLE items(x BIGINT)")
    assert store.retrieve(sql)["documents"] == []


def test_source_balancing_keeps_validated_patterns_retrievable(tmp_path):
    schema = "CREATE TABLE items(x INT)"
    write_patterns(tmp_path, [approved_pattern("SELECT * FROM items WHERE x = 1", schema)])
    sources = {
        "LLM4Rewrite/explain_rule/calcite_rewrite_rules_structured.jsonl": "".join(
            json.dumps({"name": f"predicate{index}", "conditions": "sequential scan predicate filter",
                        "transformations": "consider safe filter pushdown"}) + "\n" for index in range(6)),
        "QUITE/src/Rewrite_Middleware/Structured_Knowledge_Base/storage/knowledge_base.json": json.dumps([
            {"id": f"predicate{index}", "group": "sequential scan predicate filter",
             "answer_summary": "consider safe filter pushdown"} for index in range(6)]),
    }
    for name, content in sources.items():
        path = tmp_path / "vendor" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    store = EvidenceStore(tmp_path)
    store.begin_query("SELECT * FROM items WHERE x > 2", schema)
    retrieved = store.retrieve("SELECT * FROM items WHERE x > 2", bottleneck={"kind": "sequential_scan"})
    assert len(retrieved["documents"]) == 6
    assert {doc["source"] for doc in retrieved["documents"]} == {"rbot", "quite", "action_patterns"}


def test_reused_controller_resets_model_calibration_and_generator_memory():
    class ResettableGenerator(FakeGenerator):
        def begin_query(self, original_sql, schema_sql):
            self.memory = {"root": original_sql}
    generator = ResettableGenerator({"SELECT 0": [candidate("SELECT 1")]})
    controller = SearchController(FakeDB(), FakeGate(), generator,
        SearchConfig(budget=1, use_rag=False, use_critic=True, use_safety=False))
    first = controller.run("SELECT 0", "")
    second = controller.run("SELECT 0", "")
    assert first["critic"]["calibration_observations"] == second["critic"]["calibration_observations"] == 2
    assert generator.memory == {"root": "SELECT 0"}


def test_uct_prefers_reward_then_explores_other_root_branch():
    batches = {"SELECT 0": [candidate("SELECT 1"), candidate("SELECT 2")]}
    batches.update({f"SELECT {index}": [candidate(f"SELECT {index + 2}")] for index in range(1, 20)})
    times = {"SELECT 0": 10} | {f"SELECT {index}": 5 if index % 2 else 8 for index in range(1, 22)}
    generator = FakeGenerator(batches)
    result = SearchController(FakeDB(times), FakeGate(), generator,
        config(budget=8, use_safety=False)).run("SELECT 0", "")
    expansions = [call["sql"] for call in generator.calls]
    assert expansions[:2] == ["SELECT 0", "SELECT 1"]
    assert "SELECT 2" in expansions
    assert result["expanded"] == 8


def test_search_preserves_finite_evidence_label_from_configured_gate():
    class DifferentialGate(FakeGate):
        def verify(self, original, proposed):
            result = super().verify(original, proposed)
            if original != proposed:
                result.update(proof="differential", formal=False, reason="finite miniature evidence only")
            return result
    result = SearchController(FakeDB(), DifferentialGate(),
        FakeGenerator({"SELECT 0": [candidate("SELECT 1", "B")]}),
        config(budget=1, use_safety=False)).run("SELECT 0", "")
    assert result["sql"] == "SELECT 1"
    assert result["verification"]["proof"] == "differential"
    assert result["verification"]["formal"] is False
