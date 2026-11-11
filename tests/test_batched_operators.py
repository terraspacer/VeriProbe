import json

import pytest

from mcts_rewrite.batched_operators import BatchedOperators, RBotLexicalRetriever


class Calcite:
    def __init__(self):
        self.applied = []

    def match(self, sql, schema):
        return {"status": "ok", "matches": [{"name": "FILTER_REDUCE_EXPRESSIONS", "type": "normal"}]}

    def rewrite(self, sql, schema, sequences):
        self.applied.append((sql, sequences))
        return {"status": "ok", "provenance": {"engine": "rbot_calcite_java"},
                "rewrites": [{"sql": sql.replace(" WHERE TRUE", ""), "rules": sequence} for sequence in sequences]}


class Retriever:
    rule_descriptions = {"FILTER_REDUCE_EXPRESSIONS": {"conditions": "constant filter"}}

    def retrieve(self, sql, **kwargs):
        return {"documents": [{"id": "1-2", "text": "Actual upstream Q&A fixture"}],
                "retrieval": {"method": "rbot_real_qa_bm25_variant"}}


def test_rbot_joint_relevance_and_arrangement_are_two_calls(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"case_relevance": [{"id": "1-2", "relevant": True}],
                    "selected_rules": ["FILTER_REDUCE_EXPRESSIONS"], "rewrite_advice": "Drop true filter"}
        return {"rule_sequences": [["FILTER_REDUCE_EXPRESSIONS"], ["INVENTED_RULE"]]}
    calcite = Calcite()
    ops = BatchedOperators(tmp_path, model=model, calcite=calcite, retriever=Retriever())
    rows = ops.rbot("SELECT 1 WHERE TRUE", "", {}, {}, {})
    assert len(calls) == ops.model_requests == 2
    assert [payload["task"] for payload in calls] == ["rbot_batched_relevance", "rbot_batched_arrange"]
    assert calls[1]["batched_relevance"]["rewrite_advice"] == "Drop true filter"
    assert calcite.applied == [("SELECT 1 WHERE TRUE", [["FILTER_REDUCE_EXPRESSIONS"]])]
    assert rows[0]["sql"] == "SELECT 1"
    assert rows[0]["provenance"]["engine"] == "rbot_calcite_java"
    assert rows[0]["provenance"]["verified"] is False
    assert rows[0]["provenance"]["standalone_rbot"] is False


def test_beam_generation_batches_all_parents_and_keeps_parent_identity(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"batches": [
            {"parent_id": 0, "rule_sequences": [["FILTER_REDUCE_EXPRESSIONS"]],
             "candidates": [{"sql": "SELECT 1 + 0", "action": "macro"}]},
            {"parent_id": 1, "candidates": [{"sql": "SELECT 2 + 0", "action": "macro"}]},
            {"parent_id": 2, "candidates": [{"sql": "SELECT 3 + 0", "action": "macro"}]},
            {"parent_id": "unknown", "candidates": [{"sql": "SELECT 999"}]}]}
    calcite = Calcite()
    ops = BatchedOperators(tmp_path, model=model, calcite=calcite)
    parents = [{"id": index, "sql": f"SELECT {index + 1} WHERE TRUE"} for index in range(3)]
    rows = ops.generate_batch(parents, "", depth=2)
    assert len(calls) == 1
    assert len(calls[0]["parents"]) == 3
    assert calcite.applied == [(parents[0]["sql"], [["FILTER_REDUCE_EXPRESSIONS"]])]
    assert {row["parent_id"] for row in rows} == {0, 1, 2}
    assert {row["action_class"] for row in rows} == {"A", "B"}
    assert all(row["parent_sql"] == parents[row["parent_id"]]["sql"] for row in rows)
    assert all(row["provenance"]["verified"] is False for row in rows)
    assert all(row["sql"] != "SELECT 999" for row in rows)


def test_quite_generation_does_not_invoke_calcite_or_accept_hints(tmp_path):
    class NoCalcite:
        def match(self, *args):
            raise AssertionError("QUITE must not use the symbolic arm")
    calls = []
    def model(payload):
        calls.append(payload)
        return {"batches": [{"parent_id": "root", "candidates": [
            {"sql": "SELECT /*+ Set(enable_hashjoin off) */ 1"},
            {"sql": "SELECT 1", "provenance": {"verified": True}},
            {"sql": "SELECT 1 WHERE TRUE"},
            {"sql": ""}]}]}
    ops = BatchedOperators(tmp_path, model=model, calcite=NoCalcite())
    rows = ops.macro_batch([{"id": "root", "sql": "SELECT 1 WHERE TRUE"}], "", operator="quite")
    assert len(calls) == 1
    assert [row["sql"] for row in rows] == ["SELECT 1"]
    assert rows[0]["provenance"]["verified"] is False
    assert "pg_hint_plan" not in calls[0]["allowed_transformations"]


def test_critic_batch_validates_ids_and_finite_positive_predictions(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"predictions": [{"id": "a", "predicted_ms": 1.25},
            {"id": "b", "predicted_ms": -1}, {"id": "c", "predicted_ms": "NaN"},
            {"id": "d", "predicted_ms": True}, {"id": "unknown", "predicted_ms": 1}]}
    ops = BatchedOperators(tmp_path, model=model)
    predictions = ops.critic_batch([{"id": name, "sql": "SELECT 1", "plan": {"Plan": {}}}
                                   for name in "abcd"], observations=[{"measured_ms": 10}])
    assert len(calls) == 1
    assert set(predictions) == {"a"}
    assert predictions["a"]["predicted_ms"] == 1.25
    assert predictions["a"]["used_for"] == "exploration_order_only"
    assert calls[0]["observations"] == [{"measured_ms": 10}]


def test_failures_are_counted_and_not_retried(tmp_path):
    calls = []
    def fail(payload):
        calls.append(payload)
        raise TimeoutError("deadline")
    ops = BatchedOperators(tmp_path, model=fail, calcite=Calcite(), retriever=Retriever())
    assert ops.rbot("SELECT 1 WHERE TRUE", "") == []
    assert len(calls) == ops.model_requests == 2
    assert len(ops.model_errors) == 2
    assert ops.metadata()["automatic_retries"] == 0


def test_query_local_memory_is_separate_between_arms_and_queries(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"batches": [{"parent_id": 1, "rewrite_advice": "source advice",
                             "candidates": [{"sql": "SELECT 1"}]}]}
    ops = BatchedOperators(tmp_path, model=model)
    parent = [{"id": 1, "sql": "SELECT 1 WHERE TRUE"}]
    ops.macro_batch(parent, "", operator="quite")
    ops.macro_batch(parent, "", operator="mcts")
    assert calls[-1]["parents"][0]["memory"] == []
    ops.macro_batch(parent, "", operator="quite", depth=2)
    assert calls[-1]["parents"][0]["memory"][0]["optimization_advice"] == "source advice"
    ops.begin_query("SELECT 2", "")
    ops.macro_batch(parent, "", operator="quite")
    assert calls[-1]["parents"][0]["memory"] == []


def test_rbot_lexical_evidence_uses_original_qa_and_excludes_exact_input(tmp_path):
    directory = tmp_path / ".cache/mcts_rewrite/rbot_runtime/rag"
    directory.mkdir(parents=True)
    rows = [{"id": "1", "answer_id": "2", "question_title": "Remove redundant joins",
             "answer_body": "An inner join filter can remove redundant work", "question_body_sqls": ["SELECT 1"]},
            {"id": "3", "answer_id": "4", "question_title": "Remove redundant filter",
             "answer_body": "A true filter can be removed", "question_body_sqls": ["SELECT 2"]}]
    (directory / "stackoverflow-rewrite-query-optimization.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (directory / "stackoverflow-rewrite-rules-query-optimization.jsonl").write_text(json.dumps(
        {"context": {"id": "3", "answer_id": "4"}, "rules": ["FILTER_REDUCE_EXPRESSIONS"]}))
    retriever = RBotLexicalRetriever(tmp_path)
    result = retriever.retrieve("SELECT 1", bottleneck={"kind": "filter"})
    assert result["retrieval"]["exact_input_examples_excluded"] == 1
    assert result["retrieval"]["native_embedding_fusion_reproduced"] is False
    assert result["retrieval"]["benchmark_results_loaded"] is False
    assert result["documents"][0]["id"] == "3-4"
    assert result["documents"][0]["rules"] == ["FILTER_REDUCE_EXPRESSIONS"]
    assert all(len(source["sha256"]) == 64 for source in result["retrieval"]["sources"])


def test_duplicate_ids_are_rejected_before_model_call(tmp_path):
    def model(payload):
        raise AssertionError("Must reject ambiguous parent IDs before making a model call")
    ops = BatchedOperators(tmp_path, model=model)
    with pytest.raises(ValueError, match="distinct"):
        ops.macro_batch([{"id": 1, "sql": "SELECT 1"}, {"id": "1", "sql": "SELECT 2"}], "")
