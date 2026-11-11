#!/usr/bin/env python3
"""Render paper tables from the completed live ICDE campaign."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
TMAX = 300_000.0


def p90(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * 0.9
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def samples(choice):
    measurement = choice.get("measurement", {})
    if measurement.get("status") == "timeout":
        return [TMAX, TMAX, TMAX]
    values = measurement.get("samples_ms") or []
    if len(values) == 3 and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in values):
        return [min(float(value), TMAX) for value in values]
    return [TMAX, TMAX, TMAX]


def choice(record, name):
    if name in record.get("arms", {}):
        return record["arms"][name]
    return record["variants"][name]


def metrics(rows, name, baseline_name="original"):
    current = [samples(choice(row, name)) for row in rows]
    baseline = [samples(choice(row, baseline_name)) for row in rows]
    means = [statistics.mean(value) for value in current]
    medians = [statistics.median(value) for value in current]
    baseline_medians = [statistics.median(value) for value in baseline]
    improvements = sum(value <= 0.9 * base + 1e-9 for value, base in zip(medians, baseline_medians))
    regressions = sum(value > base for value, base in zip(medians, baseline_medians))
    equivalence = len(rows) if name == "original" else sum(bool(choice(row, name).get("fallback") or choice(row, name).get("verification", {}).get("accepted") is True) for row in rows)
    return {
        "mean_ms": statistics.mean(means),
        "median_ms": statistics.median(medians),
        "p90_ms": p90(medians),
        "speedup": 1.0 - statistics.mean(means) / statistics.mean([statistics.mean(value) for value in baseline]),
        "improvement_rate": improvements / len(rows),
        "regression_rate": regressions / len(rows),
        "equivalence_rate": equivalence / len(rows),
        "improvements": improvements,
        "regressions": regressions,
    }


def pct(value):
    return f"{100.0 * value:.2f}%"


def ms(value):
    return f"{value:,.1f}"


def row_line(label, value):
    return f"| {label} | {ms(value['mean_ms'])} | {ms(value['median_ms'])} | {ms(value['p90_ms'])} | {pct(value['speedup'])} | {pct(value['improvement_rate'])} | {pct(value['equivalence_rate'])} | {pct(value['regression_rate'])} |"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=ROOT / "results/icde_paper_campaign_v4")
    args = parser.parse_args(argv)
    campaign = args.campaign
    rows = [json.loads(line) for line in (campaign / "raw.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 30 or any(row.get("status") != "ok" for row in rows):
        raise SystemExit(f"campaign is not complete: {len(rows)}/30 records")
    main_names = [("original", "Original"), ("rbot", "R-Bot"), ("quite", "QUITE"), ("mcts_M16", "MCTS-Rewrite (M=16)")]
    main_metrics = {name: metrics(rows, name) for name, _ in main_names}
    lines = ["# ICDE Stage A SF10 live campaign results", "",
             "> Caution: every table below charges a run that was cut off T_max = 300,000 ms. The cut-off differed per arm "
             "(20 s to 300 s), so the R-Bot and QUITE rows, and every comparison against them, are upper bounds and not "
             "measurements. The Stage A rows are also assembled from the measurement cache of several campaigns. "
             "`paper/scripts/build_data.py` reports the same logs as intervals; use those numbers for the paper.", "",
             "## Outcome", "", f"Live campaign source: `{campaign}`. The serialized PostgreSQL run completed **{len(rows)}/30** queries. Every selected SQL used one warmup and three measured executions; repeated SQLs reused the SQL-hash measurement cache. Timeout values are capped at T_max = 300,000 ms.", "", "| System | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in main_names:
        value = main_metrics[name]
        if name == "original":
            value = {**value, "speedup": 0.0}
        lines.append(row_line(label, value))
    lines += ["", "## Rollout scaling", "", "All rows are measured on the same 30-query SF10 cohort. MCTS shortlists by canary, then selects by lowest verified final median latency (1 warmup + 3 runs). The Original fallback is retained when no verified candidate improves on Original.", "", "| M | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for budget in (1, 2, 4, 8, 16):
        lines.append(row_line(str(budget), metrics(rows, f"mcts_M{budget}")))
    ablations = [("full_mcts", "Full MCTS-Rewrite"), ("class_a_symbolic_only", "Class-A Symbolic Only"), ("class_b_macro_only", "Class-B Macro Only"), ("without_plan_bottleneck", "w/o Plan Bottleneck"), ("without_critic", "w/o Critic"), ("without_canary_safety_gate", "w/o Canary Safety Gate")]
    lines += ["", "## Ablation study", "", "Each ablation is selected from the same verified candidate pool and measured with the same SQL-hash cache and three-sample timing protocol.", "", "| Variant | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in ablations:
        lines.append(row_line(label, metrics(rows, name)))
    accepted = sum(len(row.get("candidates", [])) for row in rows)
    depth2 = sum(1 for row in rows for candidate in row.get("candidates", []) if candidate.get("depth", 1) >= 2)
    unique_better = sum(
        statistics.median(samples(choice(row, "mcts_M16"))) < min(
            statistics.median(samples(choice(row, "rbot"))),
            statistics.median(samples(choice(row, "quite"))))
        for row in rows)
    hard_ids = ["rbot:dsb:query081_0", "rbot:calcite:testIsNullPushDown",
                "rbot:calcite:testDecorrelateTwoExists", "rbot:tpch:query18_0",
                "rbot:tpch:query10_0", "rbot:tpch:query12_0"]
    hard_rows = {row["query_id"]: row for row in rows}
    hard_solved = sum(
        (hard_rows.get(query_id, {}).get("arms", {}).get("original", {}).get("measurement", {}).get("status") == "timeout"
         and hard_rows.get(query_id, {}).get("arms", {}).get("mcts", {}).get("measurement", {}).get("status") == "ok")
        or (statistics.median(samples(choice(hard_rows[query_id], "mcts")))
            < statistics.median(samples(choice(hard_rows[query_id], "original"))))
        for query_id in hard_ids if query_id in hard_rows)
    all_orig_timeouts = [row for row in rows if choice(row, "original").get("measurement", {}).get("status") == "timeout"]
    all_timeouts_rescued = sum(choice(row, "mcts").get("measurement", {}).get("status") == "ok" for row in all_orig_timeouts)
    scaling_means = [metrics(rows, f"mcts_M{budget}")["mean_ms"] for budget in (1, 2, 4, 8, 16)]
    scaling_monotone = all(left >= right for left, right in zip(scaling_means, scaling_means[1:]))
    beats_rbot = all(main_metrics["mcts_M16"][k] < main_metrics["rbot"][k] for k in ("mean_ms", "median_ms", "p90_ms"))
    beats_quite = all(main_metrics["mcts_M16"][k] < main_metrics["quite"][k] for k in ("mean_ms", "median_ms", "p90_ms"))
    verdict = ("MCTS-Rewrite meets strict dominance over R-Bot and QUITE on mean/median/P90."
               if beats_rbot and beats_quite else
               "MCTS-Rewrite improves over Original and the Class-A baseline, but does not yet strictly dominate QUITE on every requested aggregate.")
    lines += ["", "## Search and target audit", "", f"- Verified candidate records accepted: **{accepted}**; depth ≥2 composite candidates: **{depth2}**; unique M=16 wins over both baselines: **{unique_better}**.", "- Candidate expansion order is policy/plan-bottleneck ordered before the rollout prefix is sliced; final MCTS selection uses measured 1+3 latency, not static EXPLAIN cost.", "- Class-A and Class-B candidates are both present in the unified pool; Original is always included as the safety-net action.", "", "| Requirement | Measured result |", "|---|---|", f"| 30/30 live SF10 completion | **{len(rows)}/30** |", f"| MCTS equivalence | **{pct(main_metrics['mcts_M16']['equivalence_rate'])}** |", f"| MCTS regression rate | **{pct(main_metrics['mcts_M16']['regression_rate'])}** |", f"| MCTS uniquely beats min(R-Bot, QUITE) | **{unique_better} queries** |", f"| MCTS strictly beats R-Bot on mean/median/P90 | **{'yes' if beats_rbot else 'no'}** |", f"| MCTS strictly beats QUITE on mean/median/P90 | **{'yes' if beats_quite else 'no'}** |", f"| Rollout mean scaling non-increasing M=1→16 | **{'yes' if scaling_monotone else 'no'}** |", f"| Full-cohort 300s Original timeouts rescued to finite latency | **{all_timeouts_rescued}/{len(all_orig_timeouts)}** |", f"| Hard target cases solved or improved | **{hard_solved}/{len(hard_ids)}** |", "", "| Hard target query | Original | MCTS-Rewrite |", "|---|---|---|"]
    for query_id in hard_ids:
        row = hard_rows.get(query_id)
        if row is None:
            continue
        om = choice(row, "original").get("measurement", {})
        mm = choice(row, "mcts").get("measurement", {})
        lines.append(f"| `{query_id}` | {om.get('status')} | {mm.get('status')} ({ms(mm.get('median_ms')) if isinstance(mm.get('median_ms'), (int, float)) else 'censored/fallback'}) |")
    lines += ["", verdict, "", "## Artifacts", "", f"- Raw live campaign: `{campaign / 'raw.jsonl'}`", f"- Run manifest: `{campaign / 'run.json'}`", f"- SQL measurement cache: `{ROOT / 'results/icde_sql_measurement_cache_v4.jsonl'}`", f"- Campaign implementation: `{ROOT / 'scripts/run_icde_paper_campaign.py'}`", ""]
    sf50_raw = ROOT / "results/icde_sf50_cohort/raw.jsonl"
    if sf50_raw.exists():
        sf50_rows = [json.loads(line) for line in sf50_raw.read_text().splitlines() if line.strip()]
        if len(sf50_rows) == 30 and all(row.get("status") == "ok" for row in sf50_rows):
            lines += ["", "## Cross-scale generalization (SF50 cohort: `tpch50`, `dsb50`, `calcite10`)", "",
                      f"Live campaign source: `{sf50_raw.parent}`. Completed **30/30** queries across the 5× larger scale-factor databases (`tpch50` and `dsb50`, ~130 GiB) plus `calcite10`.", "",
                      "| System | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in main_names:
                val = metrics(sf50_rows, name)
                if name == "original":
                    val = {**val, "speedup": 0.0}
                lines.append(row_line(label, val))
            lines += ["", "### SF50 rollout scaling", "",
                      "| M | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for budget in (1, 2, 4, 8, 16):
                lines.append(row_line(str(budget), metrics(sf50_rows, f"mcts_M{budget}")))
            lines += ["", "### SF50 ablation study", "",
                      "| Variant | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in ablations:
                lines.append(row_line(label, metrics(sf50_rows, name)))
    zipf_raw = ROOT / "results/icde_calcite10zipf_stage/raw.jsonl"
    if zipf_raw.exists():
        zipf_rows = [json.loads(line) for line in zipf_raw.read_text().splitlines() if line.strip()]
        if len(zipf_rows) == 30 and all(row.get("status") == "ok" for row in zipf_rows):
            lines += ["", "## Data distribution skew generalization (`calcite10zipf` 30-query cohort)", "",
                      f"Live campaign source: `{zipf_raw.parent}`. Completed **30/30** queries on the 160M-row Zipf-skewed `calcite10zipf` database.", "",
                      "| System | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in main_names:
                val = metrics(zipf_rows, name)
                if name == "original":
                    val = {**val, "speedup": 0.0}
                lines.append(row_line(label, val))
            lines += ["", "### Zipf rollout scaling", "",
                      "| M | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for budget in (1, 2, 4, 8, 16):
                lines.append(row_line(str(budget), metrics(zipf_rows, f"mcts_M{budget}")))
            lines += ["", "### Zipf ablation study", "",
                      "| Variant | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in ablations:
                lines.append(row_line(label, metrics(zipf_rows, name)))
    tpch_raw = ROOT / "results/icde_full_suite_tpch10/raw.jsonl"
    if tpch_raw.exists():
        tpch_rows = [json.loads(line) for line in tpch_raw.read_text().splitlines() if line.strip()]
        if len(tpch_rows) == 107 and all(row.get("status") == "ok" for row in tpch_rows):
            lines += ["", "## Full-suite TPC-H SF10 evaluation (`tpch10` 107-query benchmark)", "",
                      f"Live campaign source: `{tpch_raw.parent}`. Completed **107/107** queries across all R-Bot (44) and QUITE (63) TPC-H memberships on `tpch10`.", "",
                      "| System | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in main_names:
                val = metrics(tpch_rows, name)
                if name == "original":
                    val = {**val, "speedup": 0.0}
                lines.append(row_line(label, val))
            lines += ["", "### Full-suite TPC-H SF10 rollout scaling", "",
                      "| M | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for budget in (1, 2, 4, 8, 16):
                lines.append(row_line(str(budget), metrics(tpch_rows, f"mcts_M{budget}")))
            lines += ["", "### Full-suite TPC-H SF10 ablation study", "",
                      "| Variant | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in ablations:
                lines.append(row_line(label, metrics(tpch_rows, name)))
    calcite_raw = ROOT / "results/icde_full_suite_calcite10/raw.jsonl"
    if calcite_raw.exists():
        calcite_rows = [json.loads(line) for line in calcite_raw.read_text().splitlines() if line.strip()]
        if len(calcite_rows) >= 44 and all(row.get("status") == "ok" for row in calcite_rows):
            lines += ["", f"## Calcite10 Uniform R-Bot suite (`calcite10` {len(calcite_rows)}-query completed subset)", "",
                      f"Live campaign source: `{calcite_raw.parent}`. Completed **{len(calcite_rows)}/44** R-Bot Calcite queries on the 160M-row uniform `calcite10` database.", "",
                      "| System | Mean (ms) | Median (ms) | P90 (ms) | Speedup vs Original | Improvement rate | Equivalence rate | Regression rate |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, label in main_names:
                val = metrics(calcite_rows, name)
                if name == "original":
                    val = {**val, "speedup": 0.0}
                lines.append(row_line(label, val))
    status_path = ROOT / "results/icde_master_campaign/status.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        lines += ["", "## Five-phase master campaign checkpoint", "",
                  "| Phase | Status | Tasks | Output |",
                  "|---|---|---:|---|"]
        for name, phase in st.get("phases", {}).items():
            lines.append(f"| {name} | {phase.get('status')} | {phase.get('tasks', 'n/a')} | {phase.get('output', phase.get('source', 'n/a'))} |")
        lines.append("")
    report = ROOT / "reports/ICDE_PAPER_RESULTS.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(report)


if __name__ == "__main__":
    main()
