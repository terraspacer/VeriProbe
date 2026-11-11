#!/usr/bin/env python3
"""Generate session-authored Calcite macro candidates from manifest inputs only.

This file deliberately never opens an upstream rewrite or saved baseline. The
manual transformations below were authored from the manifest's original SQL,
the original DDL, and PostgreSQL 14 development EXPLAIN plans. EXPLAIN and column
metadata validation do not constitute an equivalence proof or a speedup claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import psycopg
import sqlglot


MODEL = {
    "model": "gpt-6.1-sol",
    "reasoning_effort": "xhigh",
    "generation": "session_subagent",
}

# The indices are Calcite-only manifest positions, checked against the complete
# expected source-id inventory before any file is emitted.
EXPECTED_IDS = [
    "testEmptyProject", "testEmptyFilterProjectUnion",
    "testNotPushFilterPastWindowWhenPredicateNotOnPartitionKey",
    "testDecorrelateTwoExists", "testEmptyMinus", "testIsNullPushDown2",
    "testDistinctWithGrouping", "testWhereNotInCorrelated",
    "testSelectAnyCorrelated", "testWhereInCorrelated",
    "testJoinConditionPushdown6", "testWhereOrSubQuery", "testSome",
    "testPushAggregateFunctionsThroughJoin", "testReduceConstantsCalc",
    "testReduceConstantsNegatedInverted", "testDecorrelateTwoIn",
    "testAggregateRemove1", "testAggregateRemove2", "testAggregateRemove3",
    "testWhereAnyCorrelatedInSelect", "testReduceConstantsIsNull",
    "testReduceConstantsNegated", "testIsNullPushDown",
    "testMaxReuseDistinctAttrWithMixedOptionality",
    "testSumAndDistinctSumWithExpandSumType", "testAggregateMerge1",
    "testReduceValuesToEmpty", "testAggregateMergeSum0",
    "testPushAggregateThroughOuterJoin14",
    "testPushFilterPastAggWithGroupingSets2",
    "testBitOrReuseDistinctAttrWithMixedOptionality",
    "testExistsWithAtLeastOneRowSubQuery", "testAnyInProjectNonNullable",
    "testAggregateJoinRemove4", "testAggregateJoinRemove5",
    "testSelectNotInCorrelated", "testDistinctCountWithExpandSumType",
    "testPullConstantIntoFilter", "testEmptySortLimitZero",
    "testRemoveDistinctOnAgg", "testEmptyIntersect",
    "testExpandProjectInNullable", "testMinReuseDistinctAttrWithMixedOptionality",
]


def candidates_by_index():
    result = {}

    def add(indices, sql, action, rationale, constraints=()):
        for index in ([indices] if isinstance(indices, int) else indices):
            result.setdefault(index, []).append({
                "sql": " ".join(sql.split()), "action": action,
                "rationale": rationale, "constraints": list(constraints),
            })

    add(0, 'SELECT NULL::integer AS "?column?" WHERE FALSE',
        "constant_empty_values_project",
        "Both finite VALUES rows fail x+y>50; the integer projection has an empty bag.")
    add(1, 'SELECT * FROM (VALUES (30, 3)) AS t(x, y)',
        "evaluate_values_union_filter",
        "Only (30,3) satisfies x+y>30; UNION ALL multiplicity is retained.")
    add(2, '''SELECT d.name, d.deptno,
        (SELECT count(*) FROM dept all_dept WHERE all_dept.name=d.name) AS count
        FROM dept d WHERE d.deptno=0''',
        "replace_filtered_partition_window_with_correlated_count",
        "Filter the indexed deptno first but count every row of its name partition; the window predicate is never pushed into the count input.",
        ["dept.name NOT NULL"])
    add(3, '''SELECT * FROM emp WHERE NOT EXISTS
        (SELECT 1 FROM emp ee WHERE ee.job=emp.job AND ee.sal=34)''',
        "remove_self_witness_exists",
        "The first EXISTS has the outer emp row itself as a witness.",
        ["emp.deptno NOT NULL"])
    add(4, 'SELECT NULL::integer AS x, NULL::integer AS y WHERE FALSE',
        "constant_empty_except", "The leftmost VALUES/filter relation is empty; subtracting sets preserves emptiness.")
    add([5, 23], 'SELECT empno, deptno, 0::bigint AS w_count FROM emp WHERE FALSE',
        "eliminate_is_null_count_window",
        "COUNT returns a non-NULL bigint even for empty window frames, so w_count IS NULL is never TRUE.")
    add([6, 91], 'SELECT sal, SUM(comm), MIN(comm), sal::bigint AS sum FROM emp GROUP BY sal',
        "remove_distinct_sum_of_group_key",
        "Within a sal group SUM(DISTINCT sal) is sal cast to bigint, including a nullable sal group.")
    add(7, '''SELECT sal FROM emp WHERE NOT EXISTS
        (SELECT 1 FROM dept WHERE emp.job=dept.name AND emp.empno=dept.deptno)''',
        "decorrelate_nonnull_not_in_to_anti_exists",
        "Both NOT IN operands are declared non-NULL, so anti-existence preserves SQL three-valued logic.",
        ["emp.empno PRIMARY KEY", "dept.deptno PRIMARY KEY"])
    add(8, '''SELECT COALESCE(e.empno > d.min_deptno, FALSE) AS "?column?"
        FROM emp e LEFT JOIN
        (SELECT name, MIN(deptno) AS min_deptno FROM dept GROUP BY name) d
        ON e.job=d.name''',
        "decorrelate_projected_any_via_grouped_min",
        "A > ANY non-NULL set is a comparison to its minimum; COALESCE restores FALSE for the empty set.",
        ["emp.empno PRIMARY KEY", "dept.deptno PRIMARY KEY"])
    add(9, '''SELECT sal FROM emp WHERE EXISTS
        (SELECT 1 FROM dept WHERE emp.job=dept.name AND emp.empno=dept.deptno)''',
        "decorrelate_where_in_to_exists",
        "In WHERE, IN is TRUE exactly when an equal correlated row exists; FALSE and UNKNOWN are both rejected.")
    add(10, '''SELECT * FROM (SELECT * FROM emp WHERE deptno=empno) e
        RIGHT JOIN dept d ON e.deptno=d.deptno''',
        "push_right_join_match_predicate_to_nonpreserved_input",
        "The original equality conjunction implies e.deptno=e.empno; filtering only the nonpreserved input keeps unmatched dept rows.")
    add(11, '''SELECT * FROM emp WHERE sal=4 OR NOT EXISTS
        (SELECT 1 FROM dept WHERE emp.empno=dept.deptno)''',
        "replace_nonnull_not_in_under_or",
        "The NOT IN operands are non-NULL, so its Boolean value equals NOT EXISTS even beneath OR.",
        ["emp.empno PRIMARY KEY", "dept.deptno PRIMARY KEY"])
    add([12, 92], 'SELECT * FROM emp e1 WHERE e1.empno > (SELECT MIN(deptno) FROM dept)',
        "replace_where_some_with_scalar_min",
        "The predicate is TRUE exactly when empno exceeds the minimum non-NULL deptno; the empty-set UNKNOWN is rejected by WHERE.")
    add(13, '''SELECT e.job, e.min_sal, e.min_deptno,
        e.sum_sal*d.n+1 AS sum_sal_plus, e.max_sal,
        e.sum_sal*d.n AS sum_sal_2,
        e.count_sal*d.n AS count_sal, e.count_mgr*d.n AS count_mgr
        FROM (SELECT job, MIN(sal) AS min_sal, MIN(deptno) AS min_deptno,
        SUM(sal) AS sum_sal, MAX(sal) AS max_sal,
        COUNT(sal) AS count_sal, COUNT(mgr) AS count_mgr FROM emp GROUP BY job) e
        JOIN (SELECT name, COUNT(*) AS n FROM dept GROUP BY name) d ON e.job=d.name''',
        "preaggregate_join_with_multiplicity_weights",
        "Aggregate each job/name before joining; multiply additive aggregates by the exact dept-name multiplicity and leave extrema unchanged.")
    add([14, 48], "SELECT 'TABLE'::text AS u, 't'::text AS s",
        "evaluate_constant_string_set", "Evaluate the three constant strings and their deterministic projections; only TABLE survives the filter.")
    add([15, 21, 22], 'SELECT empno FROM emp WHERE FALSE',
        "reduce_contradictory_where_to_false",
        "The conjunction is never TRUE, including for NULL operands; WHERE may therefore be replaced by FALSE.")
    add(16, '''SELECT sal FROM emp WHERE empno IN
        (SELECT deptno FROM dept WHERE emp.job=dept.name)''',
        "remove_second_self_witness_in",
        "The second correlated IN contains the outer emp row itself with the same non-NULL empno and ename.",
        ["emp.empno PRIMARY KEY", "emp.ename NOT NULL"])
    add([17, 97], '''SELECT empno, sal::bigint AS sum, sal AS min, sal AS max,
        sal AS bit_and, sal AS bit_or, 1::bigint AS count, 0::integer AS grouping FROM emp''',
        "remove_singleton_primary_key_aggregate",
        "Every (empno,deptno) group is a singleton; preserve all aggregate output types and the GROUPING bit.",
        ["emp.empno PRIMARY KEY", "emp.sal NOT NULL"])
    add(18, 'SELECT empno, deptno FROM emp',
        "remove_distinct_containing_primary_key", "empno uniqueness already makes each projected row distinct.",
        ["emp.empno PRIMARY KEY"])
    add([19, 98], '''SELECT empno,
        CASE WHEN mgr IS NULL THEN 0::bigint ELSE 1::bigint END AS count FROM emp''',
        "remove_singleton_count_aggregate",
        "The primary-key group has one row, so COUNT(mgr) is zero or one according to mgr nullability.",
        ["emp.empno PRIMARY KEY"])
    add([20, 99], '''SELECT * FROM emp WHERE EXISTS
        (SELECT 1 FROM dept WHERE emp.job=dept.name AND emp.empno>dept.deptno)''',
        "decorrelate_where_any_to_exists",
        "In WHERE a > ANY is TRUE exactly when a correlated row satisfies >, including empty and NULL-containing sets.")
    add(24, '''SELECT SUM(deptno), COUNT(deptno), MAX(deptno)
        FROM (SELECT DISTINCT deptno FROM emp) d''',
        "share_distinct_input_with_idempotent_max",
        "SUM/COUNT DISTINCT and duplicate-insensitive MAX can consume the same distinct nullable input.")
    add(25, '''SELECT SUM(s)::bigint AS sum, SUM(comm) AS sum
        FROM (SELECT comm, SUM(comm) AS s FROM emp GROUP BY comm) d''',
        "share_distinct_sum_via_value_preaggregation",
        "One group per comm carries its ordinary sum; the outer cast preserves PostgreSQL SUM(integer)'s bigint type.")
    add([26, 100], '''SELECT deptno AS c, MIN(sal), MAX(sal) AS z,
        COUNT(mgr)::numeric AS sum, COUNT(hiredate)::numeric AS n,
        SUM(sal)::numeric AS sal FROM emp GROUP BY deptno''',
        "merge_nested_distributive_aggregates",
        "MIN/MIN, MAX/MAX, SUM/COUNT and SUM/SUM merge across ename groups; casts retain numeric results of SUM(bigint).")
    add(27, 'SELECT NULL::integer AS x, NULL::integer AS b, NULL::integer AS a WHERE FALSE',
        "evaluate_empty_values_filter", "Both finite VALUES rows fail a-b<0.")
    add([28, 101], 'SELECT COUNT(comm)::numeric AS coalesce FROM emp',
        "merge_sum_count_with_empty_input_zero",
        "SUM of per-group COUNT(comm), COALESCE zero, equals the global count; numeric preserves SUM(bigint)'s result type.")
    add([29, 93], 'SELECT DISTINCT mgr, mgr FROM emp',
        "remove_duplicate_insensitive_full_self_join",
        "Each non-NULL mgr matches itself; all unmatched NULL sides collapse into one (NULL,NULL) group. The empty input remains empty.")
    add([30, 94], '''SELECT name AS dname, deptno AS ddeptno, COUNT(*) AS c
        FROM dept WHERE name='Charlie' GROUP BY GROUPING SETS ((name,deptno),(name))''',
        "push_common_grouping_key_filter_below_grouping_sets",
        "name occurs in every grouping set, so its filter may be applied before aggregation without removing generated grouping rows.")
    add(31, '''SELECT SUM(deptno), COUNT(deptno), BIT_OR(deptno)
        FROM (SELECT DISTINCT deptno FROM emp) d''',
        "share_distinct_input_with_idempotent_bit_or",
        "Duplicate elimination preserves BIT_OR while sharing the input needed by SUM/COUNT DISTINCT.")
    add(32, 'SELECT * FROM dept AS d',
        "eliminate_exists_global_aggregate",
        "The correlated COUNT(*) without GROUP BY always emits exactly one row, even when no emp matches.")
    add(33, '''SELECT name, COALESCE(deptno > (SELECT MIN(deptno) FROM emp), FALSE)
        AS "?column?" FROM dept''',
        "replace_projected_any_with_nonnull_min",
        "The non-NULL minimum summarizes ANY and COALESCE gives FALSE for the empty subquery.",
        ["dept.deptno PRIMARY KEY", "emp.deptno NOT NULL"])
    add([34, 95], 'SELECT DISTINCT d.deptno FROM dept d',
        "remove_outer_join_under_distinct_preserved_key",
        "A right join preserves every dept; DISTINCT removes any multiplicity introduced by emp matches.")
    add([34, 95], 'SELECT deptno FROM dept',
        "remove_outer_join_and_unique_distinct",
        "The duplicate-insensitive outer join can be removed and deptno is already a unique key.",
        ["dept.deptno PRIMARY KEY"])
    add(35, 'SELECT deptno, COUNT(DISTINCT name) FROM dept GROUP BY deptno',
        "remove_outer_join_under_distinct_aggregate",
        "Right-join repetitions do not change COUNT(DISTINCT dept.name), and every dept survives.")
    add(35, 'SELECT deptno, 1::bigint AS count FROM dept',
        "remove_outer_join_and_singleton_distinct_count",
        "The unique deptno groups contain one non-NULL name after removing duplicate-insensitive emp matches.",
        ["dept.deptno PRIMARY KEY", "dept.name NOT NULL"])
    add(36, '''SELECT sal, NOT EXISTS
        (SELECT 1 FROM dept WHERE emp.job=dept.name AND emp.empno=dept.deptno)
        AS "?column?" FROM emp''',
        "replace_projected_nonnull_not_in",
        "Non-NULL outer empno and inner deptno make NOT IN two-valued, including the empty set.",
        ["emp.empno PRIMARY KEY", "dept.deptno PRIMARY KEY"])
    add(37, '''SELECT COALESCE(SUM(n),0)::bigint AS count, COUNT(comm) AS count
        FROM (SELECT comm, COUNT(comm) AS n FROM emp GROUP BY comm) d''',
        "share_distinct_and_ordinary_count_input",
        "Per-comm counts preserve all multiplicities; COUNT(comm) of those groups is the distinct non-NULL count. COALESCE preserves zero on empty input.")
    add(38, 'SELECT * FROM emp WHERE deptno=10 AND 15>empno',
        "substitute_filter_equality_into_arithmetic",
        "deptno=10 makes deptno+5 exactly 15; a NULL deptno still cannot survive the conjunction.")
    add(40, '''SELECT empno, sal::bigint AS sum, sal AS min, sal AS min, sal AS max,
        sal AS bit_and, sal AS bit_or, 1::bigint AS count FROM emp''',
        "remove_singleton_distinct_aggregates",
        "Primary-key singleton groups allow direct projections while preserving the duplicate MIN column names and aggregate types.",
        ["emp.empno PRIMARY KEY", "emp.sal NOT NULL"])
    add(41, 'SELECT NULL::integer AS x, NULL::integer AS y WHERE FALSE',
        "evaluate_empty_intersect", "One INTERSECT input is the constant empty relation, making the full intersection empty.")
    add([42, 96], '''SELECT empno, deptno IN (SELECT deptno FROM emp WHERE empno<20) AS d FROM emp''',
        "fold_true_case_and_inline_pure_cte",
        "CASE WHEN TRUE is its deptno branch; inlining the deterministic relation avoids a multiply referenced CTE spool while preserving IN's NULL semantics.")
    add(43, '''SELECT SUM(deptno), COUNT(deptno), MIN(deptno)
        FROM (SELECT DISTINCT deptno FROM emp) d''',
        "share_distinct_input_with_idempotent_min",
        "SUM/COUNT DISTINCT and duplicate-insensitive MIN can consume the same distinct nullable input.")
    add(44, '''SELECT 4 AS expr0, 5 AS f1, 5 AS max WHERE EXISTS (SELECT 1 FROM emp)''',
        "replace_constant_group_aggregate_with_exists",
        "The constant group emits (4,5,5) once exactly when EMP is nonempty; EXISTS avoids scanning the full group.")
    add([45, 47], '''SELECT name FROM dept WHERE EXISTS
        (SELECT 1 FROM dept d2 WHERE dept.deptno=2*d2.deptno)''',
        "remove_grouping_and_projections_under_exists",
        "EXISTS only needs a matching doubled deptno, so grouping that expression and outer projection wrappers are unnecessary.")
    add([49, 62], '''SELECT DISTINCT e.ename FROM emp e WHERE EXISTS
        (SELECT 1 FROM emp a WHERE a.ename='A' AND a.job=e.job)''',
        "replace_duplicate_insensitive_self_left_join_with_semijoin",
        "Each selected A row witnesses a same-job match to itself because job is non-NULL; the left join has no unmatched row and final grouping is DISTINCT.",
        ["emp.job NOT NULL"])
    add(50, '''SELECT e.empno,e.ename,e.job,e.mgr,e.hiredate,e.sal,e.comm,e.deptno,e.slacker,
        d.deptno AS deptno0,d.name FROM emp e JOIN dept d ON e.deptno=d.deptno
        ORDER BY e.sal FETCH NEXT 10 ROWS ONLY''',
        "remove_unconsumed_inner_sort_and_projection_wrappers",
        "The inner sort has no row limit and no semantic consumer; the outer SAL order and limit are retained. SAL ties remain unspecified in both queries.")
    add(51, '''SELECT * FROM emp t1 JOIN dept dept0 ON t1.deptno=dept0.deptno
        WHERE t1.deptno=t1.empno''',
        "flatten_filtered_join_input", "The inner-only predicate can be expressed directly beside the inner join without changing NULL or duplicate behavior.")
    add(53, '''SELECT SUM(100) OVER (PARTITION BY deptno) AS col1,
        SUM(100) OVER (ORDER BY deptno RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS col2,
        SUM(sal) OVER (PARTITION BY deptno) AS col3 FROM emp WHERE sal=5000''',
        "remove_filter_constant_window_keys",
        "SAL is fixed to 5000; its partition key is redundant and RANGE order by SAL includes each complete deptno partition.")
    full_join_template = '''SELECT job FROM emp WHERE ename='A' GROUP BY job
        UNION SELECT NULL::varchar AS job WHERE EXISTS
        (SELECT 1 FROM dept d WHERE NOT EXISTS
        (SELECT 1 FROM emp e WHERE e.ename='A' AND e.job=d.name {extra}))'''
    add([54, 72], full_join_template.format(extra='AND e.deptno+e.empno=d.deptno+5'),
        "replace_grouped_full_join_with_union_and_anti_exists",
        "All left job groups survive a full join; unmatched right groups contribute only one NULL job. Grouping inputs are immaterial to these existence tests.")
    add(55, '''SELECT DISTINCT d.name FROM emp e LEFT JOIN dept d ON e.job=d.name WHERE e.ename='A' ''',
        "remove_input_grouping_under_distinct_left_join",
        "Final DISTINCT ignores multiplicities on both join inputs while retaining NULL for unmatched left rows.")
    add(56, '''SELECT DISTINCT d.name FROM emp e LEFT JOIN dept d
        ON e.job=d.name AND e.deptno+e.empno=d.deptno+5 WHERE e.ename='A' ''',
        "remove_input_expression_grouping_under_distinct_left_join",
        "Final DISTINCT absorbs duplicate input rows and keeps the unmatched NULL output; computed match expressions are preserved.")
    add([57, 84], '''SELECT DISTINCT e.job,d.name FROM emp e LEFT JOIN dept d
        ON e.job=d.name AND e.deptno+e.empno=d.deptno+5 WHERE e.ename='A' ''',
        "flatten_duplicate_insensitive_expression_outer_join",
        "Projection wrappers and input grouping can be removed before a final DISTINCT of the retained outer-join columns.")
    add(58, '''SELECT LEAST(COUNT(*) FILTER (WHERE sal<500),1::bigint) AS count,
        LEAST(COUNT(*) FILTER (WHERE sal>1000),1::bigint) AS count FROM emp''',
        "collapse_boolean_group_presence_counts",
        "For sal<500 the grouped booleans are uniquely (FALSE,TRUE), and for sal>1000 they are (TRUE,FALSE); each requested count is a zero/one presence test.")
    add(59, '''SELECT deptno,name FROM dept WHERE EXISTS
        (SELECT 1 FROM emp WHERE emp.deptno IS NOT NULL)
        ORDER BY deptno OFFSET 2 ROWS FETCH NEXT 10 ROWS ONLY''',
        "resolve_inner_scope_and_remove_redundant_sort",
        "The unqualified DEPTNO inside the t4 subquery binds to t4.DEPTNO, so its equality is an inner self-comparison, not correlation; the outer sort supplies the requested order.")
    add(60, '''SELECT e.sal+b.comm,
        SUM(b.sal+b.sal+100) OVER
        (PARTITION BY b.job RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        FROM emp e JOIN bonus b ON e.ename=b.ename WHERE e.deptno=10''',
        "fold_projected_boolean_join_filter",
        "Inline the deterministic arithmetic/projections and place the inner-join Boolean filter directly on emp; retain the window over joined-row multiplicities.")
    add([63, 79], "SELECT job FROM emp WHERE ename='A' GROUP BY job",
        "remove_left_join_under_left_only_grouping",
        "A left join preserves every left row, and grouping only the left job discards every repetition from right matches.")
    add([64, 80], 'SELECT sal FROM empnullables WHERE deptno=20 AND sal>1000',
        "collapse_repeated_filtered_salary_self_joins",
        "SAL>1000 implies SAL non-NULL and SAL=4 is FALSE. The grouped inner relations reduce to one identity witness per salary, so each original filtered row survives exactly once.")
    add(65, '''SELECT SUM(CAST(t3.f1*t3.sal AS integer)) FROM
        (SELECT sal,COUNT(*) AS f1 FROM emp GROUP BY sal) t3 WHERE t3.sal IS NOT NULL''',
        "remove_self_join_to_distinct_group_key",
        "The second grouped SAL input supplies exactly one match to each non-NULL SAL group. Preserve the original integer product cast and its overflow behavior.")
    add(67, '''SELECT d.deptno,d.name FROM dept d WHERE d.deptno<=10 AND EXISTS
        (SELECT 1 FROM emp e WHERE e.deptno=d.deptno)''',
        "replace_join_to_distinct_key_with_semijoin",
        "A join to one row per deptno is exactly an existence filter on each dept row, preserving dept duplicates.")
    add(68, '''SELECT e.sal+b.comm FROM emp e JOIN bonus b ON e.ename=b.ename WHERE e.deptno=10''',
        "fold_projected_boolean_join_filter", "Inline the Boolean join projection as an emp filter; the inner join preserves all match multiplicities.")
    add(69, full_join_template.format(extra=''),
        "replace_grouped_full_join_with_union_and_anti_exists",
        "All filtered left jobs survive, and unmatched right names add only a single NULL group; existence tests avoid forming the full joined bag.")
    add(70, '''SELECT SUM(comm) AS min,
        COUNT(DISTINCT sal) FILTER (WHERE sal>1000) AS count FROM emp''',
        "collapse_grouping_sets_filtered_aggregate",
        "g_3 selects the sole grand-total SUM(comm); the other filter counts distinct non-NULL salary groups above 1000, including zero on empty input.")
    add(71, '''SELECT ename,sal,deptno FROM emp WHERE sal>5000 GROUP BY ROLLUP(ename,sal,deptno)''',
        "remove_distinct_input_under_aggregate_free_rollup",
        "The outer ROLLUP emits groups without consuming multiplicities, so the inner distinct grouping is redundant; HAVING on SAL becomes a prefilter.")
    add(73, 'SELECT mgr,SUM(sal)::numeric AS sum FROM emp WHERE deptno=100 GROUP BY mgr',
        "merge_fixed_department_nested_sum",
        "Filter the grouping key to its fixed value and merge the two SUM levels, casting to numeric to retain SUM(bigint)'s output type.")
    add(74, "SELECT ename FROM emp WHERE ename='A' GROUP BY ename",
        "remove_left_join_under_left_only_grouping",
        "The left join cannot remove the filtered emp rows, and the final ename grouping ignores all right multiplicities.")
    add(77, '''SELECT name,COUNT(*) AS sum,CAST(SUM(deptno) AS integer) AS sum FROM dept GROUP BY name''',
        "collapse_grouping_sets_pivot_of_singleton_groups",
        "The first stage already has one row per name. Each later grouping set carries exactly its original count or sum into the filtered pivot.")
    add(78, '''SELECT SUM(sal) AS sum_sal,
        COUNT(DISTINCT deptno) FILTER (WHERE job='CLERK') AS count_distinct_clerk,
        SUM(sal) FILTER (WHERE deptno=10) AS sum_sal_d10,
        SUM(CASE WHEN deptno=20 THEN sal ELSE 0 END) AS sum_sal_d20,
        CASE WHEN COUNT(*)=0 THEN NULL ELSE COUNT(*) FILTER (WHERE deptno=30) END AS count_d30,
        COUNT(*) FILTER (WHERE deptno=40) AS count_d40,
        NULLIF(COUNT(*) FILTER (WHERE deptno=45),0) AS count_d45,
        NULLIF(COUNT(*) FILTER (WHERE deptno=50),0) AS count_d50,
        NULL::numeric AS sum_null_d60,
        NULLIF(COUNT(*) FILTER (WHERE deptno IS DISTINCT FROM 70),0) AS sum_null_d70,
        COUNT(*) FILTER (WHERE deptno=20) AS count_d20 FROM emp''',
        "replace_case_aggregates_with_filters_preserving_empty_nulls",
        "Use aggregate FILTER and count presence while explicitly preserving SUM's NULL on empty/no qualifying rows and ELSE semantics for nullable predicates.")
    add(81, '''SELECT t7.deptno,t12.deptno FROM emp t7 JOIN emp t12
        ON t7.deptno=t12.deptno WHERE t7.deptno<4 AND t12.deptno<4''',
        "prune_contradictory_union_all_branch",
        "The first UNION ALL branch would require deptno>7 and deptno<4. Retain the entire second branch and duplicate-sensitive self join.")
    add(82, '''SELECT s.n::numeric AS coalesce,v.t2 FROM
        (SELECT COUNT(*) FILTER (WHERE deptno=0) AS n,COUNT(*) AS total FROM emp) s
        CROSS JOIN (VALUES(1),(2)) v(t2) WHERE s.total>0''',
        "share_identical_union_aggregate_scan",
        "Compute the identical deptno=0 count once and replicate it to the two tags; total>0 preserves the missing constant groups on empty emp.")
    add(83, '''SELECT d.name FROM dept d WHERE EXISTS
        (SELECT 1 FROM emp e WHERE d.deptno=e.deptno AND d.deptno-10=e.sal+1)''',
        "replace_join_to_grouped_expression_with_exists",
        "For each dept row, the join fixes both inner group keys, so at most one grouped row matches; EXISTS preserves that multiplicity without building all groups.")
    add(85, '''SELECT 1::integer AS deptno,e.ename FROM empnullables e
        CROSS JOIN (VALUES(1),(2)) v(n) WHERE e.deptno=1''',
        "share_repeated_union_all_filtered_scan",
        "The two UNION ALL inputs are identical; replicate each qualifying row exactly twice through a two-row VALUES relation.")
    add(86, '''SELECT name,deptno AS ddeptno,COUNT(*) AS c FROM dept WHERE name='Charlie'
        GROUP BY GROUPING SETS((name,deptno),name)''',
        "flatten_projection_below_grouping_sets", "Inline the pure renaming projection while preserving both grouping sets and the existing name prefilter.")
    add(87, '''SELECT name AS name,deptno AS ddeptno,COUNT(*) AS c FROM dept WHERE name='Charlie'
        GROUP BY GROUPING SETS((name,deptno),name)''',
        "push_common_grouping_key_having_to_where",
        "name occurs in every grouping set, so HAVING name='Charlie' can filter input rows before either aggregate.")
    add(88, '''SELECT d.name AS job FROM dept d WHERE EXISTS
        (SELECT 1 FROM emp e WHERE e.ename='A' AND e.job=d.name) GROUP BY d.name
        UNION SELECT NULL::varchar AS job WHERE EXISTS
        (SELECT 1 FROM dept d WHERE NOT EXISTS
        (SELECT 1 FROM emp e WHERE e.ename='A' AND e.job=d.name))''',
        "replace_grouped_right_join_with_semi_and_anti_exists",
        "Matching right names equal the projected left jobs; any unmatched preserved name contributes one NULL group. UNION retains that exact set.")
    add(89, '''SELECT sal,SUM(comm) AS expr1,AVG(DISTINCT comm) AS expr2,
        sal::bigint AS expr3 FROM emp GROUP BY sal''',
        "fuse_null_safe_joined_aggregates",
        "All three inputs use exactly the same salary group domain; AVG over distinct (sal,comm) is AVG(DISTINCT comm), and the singleton distinct-sal SUM is a bigint cast.")
    return result


SKIPS = {
    39: "LIMIT 0 is already a constant empty plan; no useful distinct optimization candidate is asserted.",
    46: "The partition window consumes joined-row multiplicities; no constraint proves a safe prejoin window aggregation. PostgreSQL already pushes the emp filter.",
    52: "Both top-N stages and SAL ties require care; no new candidate is asserted without a measured tie-safe benefit.",
    61: "The query already preaggregates both join inputs with explicit integer casts; reversing that reduction can expand the joined bag or alter overflow behavior.",
    66: "Weighted grouped sums use explicit integer casts and special empty-input CASE behavior; no safely simplifying beneficial macro is asserted.",
    75: "Both join inputs are already grouped counts; expanding to a raw join can be much more costly and is not proposed as an optimization.",
    76: "The existing preaggregation avoids join expansion and preserves explicit integer-cast behavior; no better safe candidate is asserted.",
    90: "The top-N left join is already thin. Pushing the limit requires tie handling and a uniqueness assumption; no candidate is asserted here.",
}


def metadata(cursor, query):
    cursor.execute('SELECT * FROM (' + query.rstrip().rstrip(';') + ') AS macro_metadata LIMIT 0')
    return [{"name": c.name, "type_oid": c.type_code, "internal_size": c.internal_size,
             "precision": c.precision, "scale": c.scale} for c in cursor.description]


def query_inspect(connection, query):
    result = {"sql_sha256": hashlib.sha256(query.encode()).hexdigest()}
    try:
        sqlglot.parse_one(query, read='postgres')
        result['parse_status'] = 'ok'
    except Exception as error:
        result.update(parse_status='error', parse_error=f'{type(error).__name__}: {error}')
        return result
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute('SET TRANSACTION READ ONLY')
                cursor.execute("SET LOCAL statement_timeout='5s'")
                cursor.execute('EXPLAIN (FORMAT JSON) ' + query)
                document = cursor.fetchone()[0][0]
                result.update(explain_status='ok', explain=document,
                              estimated_cost=document['Plan']['Total Cost'])
                result['columns'] = metadata(cursor, query)
    except Exception as error:
        result.update(explain_status='error', explain_error=f'{type(error).__name__}: {error}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dsn', default='user=mcts_read dbname=calcite_dev host=/workspace/VeriProbe/.runtime/socket port=55432')
    args = parser.parse_args()
    root = args.root
    rows = [json.loads(line) for line in (root/'benchmarks/manifest.jsonl').read_text().splitlines()
            if line.strip()]
    rows = [row for row in rows if row['benchmark']=='calcite']
    expected = [f'rbot:calcite:{name}' for name in EXPECTED_IDS]
    expected += [f'quite:calcite:{i}:entry{i:03d}' for i in range(1,59)]
    assert [row['id'] for row in rows] == expected, 'Calcite manifest inventory/order changed'
    assert len(rows)==102 and len({row['sql'] for row in rows})==102
    authored = candidates_by_index()
    assert set(authored)|set(SKIPS)==set(range(102))
    assert not(set(authored)&set(SKIPS))
    notes = {
        'format_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': {**MODEL, 'evidence': [
            {'kind':'input_boundary','manifest':'benchmarks/manifest.jsonl',
             'upstream_rewrites_inspected':False},
            {'kind':'schema_inputs','paths':sorted({r['schema_path'] for r in rows})},
        ]},
        'validation_scope':'SQLGlot parse, read-only PostgreSQL EXPLAIN, and exact column-metadata comparison only. No EXPLAIN ANALYZE, latency measurement, formal proof, or differential execution was performed by this generator.',
        'formal_gate_required':True,
        'plan_hints': {'generated':False,
            'reason':'No timing-backed physical-plan preference is claimed. Structural rewrites remain independently gateable; hint applicability is recorded per query.'},
        'queries': [],
    }
    output=[]
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version(),current_user,current_database(),current_setting('pg_hint_plan.enable_hint',true)")
            environment=cursor.fetchone()
            notes['database']={'version':environment[0], 'role':environment[1],
                'database':environment[2],
                'pg_hint_plan.enable_hint':environment[3], 'profile':'development',
                'publication_measurement':False}
            cursor.execute("SELECT conname,pg_get_constraintdef(oid) FROM pg_constraint WHERE connamespace='public'::regnamespace ORDER BY conname")
            notes['catalog_constraints']=[{'name':n,'definition':d} for n,d in cursor.fetchall()]
        for index,row in enumerate(rows):
            original=row['sql']
            sha=hashlib.sha256(original.encode()).hexdigest()
            assert sha==row['sha256'], row['id']
            original_info=query_inspect(connection,original)
            qnote={'index':index, 'id':row['id'], 'original_sha256':sha,
                'original':original_info, 'candidates':[],
                'hint_applicability':'PostgreSQL pg_hint_plan is loaded; no hint is asserted without a measured plan benefit.'}
            out_candidates=[]
            for candidate in authored.get(index,[]):
                sql=candidate['sql']
                info=query_inspect(connection,sql)
                same_columns=info.get('columns')==original_info.get('columns')
                valid=(info.get('explain_status')=='ok' and original_info.get('explain_status')=='ok' and same_columns)
                evidence=[{'kind':'semantic_transformation','rationale':candidate['rationale'],
                           'schema_assumptions':candidate['constraints']},
                    {'kind':'postgres_explain', 'profile':'development',
                     'root_estimated_cost':original_info.get('estimated_cost'),
                     'candidate_estimated_cost':info.get('estimated_cost'),
                     'column_metadata_equal':same_columns,
                     'formal_proof':False, 'differential_test':False,
                     'notes_path':'artifacts/calcite_macro_notes.json'}]
                qnote['candidates'].append({'action':candidate['action'],
                    'sql':sql, 'rationale':candidate['rationale'],
                    'schema_assumptions':candidate['constraints'],
                    'inspection':info, 'column_metadata_equal':same_columns,
                    'emitted':valid})
                if valid:
                    assert sql.strip().rstrip(';')!=original.strip().rstrip(';')
                    out_candidates.append({'sql':sql,'action':candidate['action'],
                        'action_class':'B','provenance':{**MODEL,'evidence':evidence}})
            if not out_candidates:
                qnote['empty_candidate_reason']=SKIPS.get(index,'No authored candidate passed PostgreSQL EXPLAIN and exact column metadata validation; see candidate inspections.')
            assert len(out_candidates)<=4
            output.append({'original_sql':original,'original_sha256':sha,
                'benchmark':'calcite','candidates':out_candidates,
                'provenance':{**MODEL,'evidence':[
                    {'kind':'manifest_original','id':row['id'],
                     'source':row['source'],'schema_path':row['schema_path']},
                    {'kind':'input_only_generation','upstream_rewrites_inspected':False},
                ]}})
            notes['queries'].append(qnote)
            print(json.dumps({'id':row['id'],'root_explain':original_info.get('explain_status'),
                'authored':len(authored.get(index,[])),'emitted':len(out_candidates)}),flush=True)
    notes['summary']={
        'manifest_rows':len(rows),'distinct_originals':len(output),
        'source_memberships':dict(Counter(row['source'] for row in rows)),
        'candidate_count':sum(len(row['candidates']) for row in output),
        'originals_with_candidates':sum(bool(row['candidates']) for row in output),
        'originals_without_candidates':sum(not row['candidates'] for row in output),
        'original_explain_failures':sum(q['original'].get('explain_status')!='ok' for q in notes['queries']),
        'candidate_inspection_failures':sum(not c['emitted'] for q in notes['queries'] for c in q['candidates']),
        'candidate_actions':dict(Counter(c['action'] for row in output for c in row['candidates'])),
    }
    artifact_dir=root/'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir/'model_calcite.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in output))
    (artifact_dir/'calcite_macro_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(notes['summary'],indent=2))


if __name__=='__main__':
    main()
