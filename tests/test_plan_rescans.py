"""Loop attribution distinguishes observations from uncertain cache scenarios."""
from copy import deepcopy

import pytest

from mcts_rewrite.plans import bottleneck, operators, plan_features


def materialized_plan():
    return {"Plan": {"Node Type": "Nested Loop", "Total Cost": 12000, "Plan Rows": 1000,
        "Plans": [
            {"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Relation Name": "outer",
             "Total Cost": 1, "Plan Rows": 1000},
            {"Node Type": "Materialize", "Parent Relationship": "Inner", "Total Cost": 11,
             "Plan Rows": 10, "Plans": [
                {"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Relation Name": "cached",
                 "Total Cost": 10, "Plan Rows": 10}]}]}}


def memoized_plan():
    plan = materialized_plan()
    cache = plan["Plan"]["Plans"][1]
    cache["Node Type"] = "Memoize"
    cache["Cache Key"] = "outer.key"
    cache["Plans"][0].update({"Node Type": "Index Scan", "Index Cond": "(cached.key = outer.key)"})
    return plan


def test_timing_off_observed_loops_override_estimates_without_parent_multiplication():
    plan = materialized_plan()
    root = plan["Plan"]
    root["Actual Loops"] = 1
    root["Plans"][0]["Actual Loops"] = 1
    materialize = root["Plans"][1]
    materialize["Actual Loops"] = 8
    materialize["Plans"][0]["Actual Loops"] = 1
    rows = operators(plan)
    cache, source = rows[2:]
    assert cache["estimated_loops"] == source["estimated_loops"] == 1000
    assert cache["effective_loops"] == 8
    assert source["effective_loops"] == source["actual_loops"] == 1
    assert source["loops_source"] == "observed"
    assert source["effective_exclusive_cost"] == 10
    assert not source["rescan_attribution_uncertain"]
    assert all(row["actual_exclusive_ms"] is None for row in rows)
    features = plan_features(plan)
    assert features["max_rescans"] == 8
    assert features["max_estimated_rescans"] == 1000
    assert features["observed_loop_operators"] == 4
    assert features["rescan_uncertain_operators"] == 0
    assert features["exclusive_work"] < features["estimated_exclusive_work"]
    assert bottleneck(plan)["ranking_metric"] == "effective_exclusive_cost"


def test_unexecuted_node_has_zero_effective_work_even_with_large_estimates():
    plan = materialized_plan()
    source = plan["Plan"]["Plans"][1]["Plans"][0]
    source["Actual Loops"] = 0
    row = operators(plan)[-1]
    assert row["estimated_loops"] == 1000
    assert row["actual_loops"] == row["effective_loops"] == row["effective_exclusive_cost"] == 0
    assert row["loops_source"] == "observed"
    assert not row["rescan_attribution_uncertain"]


@pytest.mark.parametrize("invalid", [None, "invalid", -1, float("nan"), float("inf")])
def test_invalid_observations_fall_back_to_labeled_estimates(invalid):
    plan = materialized_plan()
    plan["Plan"]["Plans"][1]["Plans"][0]["Actual Loops"] = invalid
    source = operators(plan)[-1]
    assert source["actual_loops"] is None
    assert source["effective_loops"] == source["estimated_loops"] == 1000
    assert source["loops_source"] == "estimated"
    assert source["rescan_attribution_uncertain"]


def test_unobserved_materialize_source_is_explicitly_uncertain_not_claimed_single_scan():
    plan = materialized_plan()
    rows = operators(plan)
    assert not rows[2]["rescan_attribution_uncertain"]
    source = rows[-1]
    assert source["estimated_loops"] == source["effective_loops"] == 1000
    assert source["materialize_ancestor_paths"] == [[1]]
    assert source["rescan_attribution_uncertain"]
    detail = bottleneck(plan)
    assert detail["rescan_uncertain_operators"] == 1
    assert "not observed source scans or guaranteed bounds" in detail["rescan_uncertainty_note"]
    assert plan_features(plan)["rescan_uncertain_operators"] == 1


@pytest.mark.parametrize("cache_type", ["Materialize", "Memoize"])
def test_parameterized_and_correlated_work_is_not_reset_below_cache(cache_type):
    plan = materialized_plan()
    plan["Plan"]["Plans"][1]["Node Type"] = cache_type
    source = plan["Plan"]["Plans"][1]["Plans"][0]
    source.update({"Node Type": "Index Scan", "Index Cond": "(cached.key = outer.key)",
        "Plans": [{"Node Type": "Aggregate", "Parent Relationship": "SubPlan",
            "Subplan Name": "SubPlan 1", "Total Cost": 1, "Plan Rows": 1, "Plans": [
                {"Node Type": "Seq Scan", "Total Cost": 0.5,
                 "Filter": "(correlated.key = cached.key)", "Plan Rows": 1}]}]})
    rows = operators(plan)
    source_row, subplan, correlated_scan = rows[-3:]
    assert source_row["estimated_loops"] == 1000
    assert subplan["kind"] == "correlated_subplan"
    assert subplan["estimated_loops"] == correlated_scan["estimated_loops"] == 10000
    assert all(row["rescan_attribution_uncertain"] for row in rows[-3:])
    subplan_node = source["Plans"][0]
    subplan_node["Actual Loops"] = 35
    subplan_node["Plans"][0]["Actual Loops"] = 35
    observed = operators(plan)
    assert observed[-2]["effective_loops"] == observed[-1]["effective_loops"] == 35
    assert observed[-1]["estimated_loops"] == 10000
    assert not observed[-2]["rescan_attribution_uncertain"]
    assert not observed[-1]["rescan_attribution_uncertain"]


@pytest.mark.parametrize("cache_type", ["Materialize", "Memoize"])
def test_observed_parent_does_not_imply_child_cache_misses(cache_type):
    plan = materialized_plan()
    plan["Plan"]["Plans"][1]["Node Type"] = cache_type
    plan["Plan"]["Plans"][1]["Actual Loops"] = 8
    rows = operators(plan)
    assert rows[2]["effective_loops"] == 8
    assert rows[3]["estimated_loops"] == 1000
    assert rows[3]["rescan_attribution_uncertain"]


@pytest.mark.parametrize("relationship, name, expression", [
    ("InitPlan", "InitPlan 1 (returns $0)", "(cached.key > $0)"),
    ("SubPlan", "SubPlan 1", "(hashed SubPlan 1)"),
])
@pytest.mark.parametrize("cache_type", ["Materialize", "Memoize"])
def test_independent_subplan_estimate_does_not_inherit_cache_ambiguity(relationship, name, expression, cache_type):
    plan = materialized_plan()
    plan["Plan"]["Plans"][1]["Node Type"] = cache_type
    source = plan["Plan"]["Plans"][1]["Plans"][0]
    source["Filter"] = expression
    source["Plans"] = [{"Node Type": "Seq Scan", "Parent Relationship": relationship,
        "Subplan Name": name, "Total Cost": 1, "Plan Rows": 1}]
    subplan = operators(plan)[-1]
    assert subplan["estimated_loops"] == 1
    assert not subplan["rescan_attribution_uncertain"]
    assert subplan["materialize_ancestor_paths"] == []
    assert subplan["memoize_ancestor_paths"] == []


def test_observed_subplan_loops_take_precedence_even_when_estimated_independent():
    plan = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 100, "Plan Rows": 50,
        "Filter": "(hashed SubPlan 1)", "Plans": [
            {"Node Type": "Seq Scan", "Total Cost": 5, "Plan Rows": 1, "Actual Loops": 3,
             "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1"}]}}
    row = operators(plan)[-1]
    assert row["estimated_loops"] == 1
    assert row["effective_loops"] == 3
    assert row["effective_exclusive_cost"] == 15


def test_measured_time_remains_ranking_metric_and_operator_input_is_unchanged():
    plan = materialized_plan()
    plan["Plan"].update({"Actual Total Time": 20, "Actual Loops": 1})
    plan["Plan"]["Plans"][0].update({"Actual Total Time": 1, "Actual Loops": 1})
    materialize = plan["Plan"]["Plans"][1]
    materialize.update({"Actual Total Time": 2, "Actual Loops": 8})
    materialize["Plans"][0].update({"Actual Total Time": 12, "Actual Loops": 1})
    before = deepcopy(plan)
    detail = bottleneck(plan)
    assert plan == before
    assert detail["ranking_metric"] == "actual_exclusive_ms"
    assert detail["relation"] == "cached"
    assert detail["actual_exclusive_ms"] == 12
    assert next(row for row in detail["ranked"] if row["node_type"] == "Materialize")["actual_exclusive_ms"] == 4


def test_unobserved_memoize_child_is_uncertain_without_assuming_hits_or_resetting_loops():
    plan = memoized_plan()
    rows = operators(plan)
    cache, source = rows[-2:]
    assert not cache["rescan_attribution_uncertain"]
    assert source["estimated_loops"] == source["effective_loops"] == 1000
    assert source["effective_exclusive_cost"] == 10000
    assert source["rescan_attribution_uncertain"]
    assert source["materialize_ancestor_paths"] == []
    assert source["memoize_ancestor_paths"] == [[1]]
    assert plan_features(plan)["rescan_uncertain_operators"] == 1
    assert "Memoize misses unresolved" in bottleneck(plan)["rescan_uncertainty_note"]


@pytest.mark.parametrize("source_loops", [0, 3])
def test_memoize_observed_source_loops_override_parent_and_estimates(source_loops):
    plan = memoized_plan()
    cache = plan["Plan"]["Plans"][1]
    cache.update({"Actual Loops": 100 if source_loops else 0,
                  "Cache Hits": 97 if source_loops else 0, "Cache Misses": source_loops})
    cache["Plans"][0]["Actual Loops"] = source_loops
    source = operators(plan)[-1]
    assert source["estimated_loops"] == 1000
    assert source["effective_loops"] == source["actual_loops"] == source_loops
    assert source["effective_exclusive_cost"] == 10 * source_loops
    assert source["loops_source"] == "observed"
    assert source["memoize_ancestor_paths"] == [[1]]
    assert not source["rescan_attribution_uncertain"]


def test_memoize_counters_do_not_replace_missing_child_loop_observations():
    plan = memoized_plan()
    cache = plan["Plan"]["Plans"][1]
    cache.update({"Actual Loops": 100, "Cache Hits": 97, "Cache Misses": 3})
    cache["Plans"][0]["Actual Loops"] = float("nan")
    source = operators(plan)[-1]
    assert source["actual_loops"] is None
    assert source["effective_loops"] == 1000
    assert source["rescan_attribution_uncertain"]


@pytest.mark.parametrize("outer_cache, inner_cache", [
    ("Materialize", "Memoize"), ("Memoize", "Materialize"),
])
def test_nested_materialize_and_memoize_keep_distinct_cache_ancestry(outer_cache, inner_cache):
    plan = memoized_plan()
    cache = plan["Plan"]["Plans"][1]
    cache["Node Type"] = outer_cache
    source = cache["Plans"][0]
    cache["Plans"] = [{"Node Type": inner_cache, "Total Cost": 10.5,
                       "Plan Rows": 10, "Plans": [source], "Actual Loops": 8}]
    row = operators(plan)[-1]
    assert row["estimated_loops"] == row["effective_loops"] == 1000
    assert row["rescan_attribution_uncertain"]
    expected_materialize = [1] if outer_cache == "Materialize" else [1, 0]
    expected_memoize = [1] if outer_cache == "Memoize" else [1, 0]
    assert row["materialize_ancestor_paths"] == [expected_materialize]
    assert row["memoize_ancestor_paths"] == [expected_memoize]
    source["Actual Loops"] = 2
    observed = operators(plan)[-1]
    assert observed["effective_loops"] == 2
    assert observed["estimated_loops"] == 1000
    assert not observed["rescan_attribution_uncertain"]
