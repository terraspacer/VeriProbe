#!/usr/bin/env python3
"""Derive every number, table and plot series used by the paper from raw logs.

Nothing in the manuscript is typed by hand: this script reads the campaign
``raw.jsonl`` files and writes

* ``paper/generated/numbers.tex``   LaTeX macros for numbers quoted in prose,
* ``paper/generated/tab_*.tex``     table bodies,
* ``paper/generated/*.dat``         pgfplots series,
* ``paper/generated/audit.json``    the per-query evidence behind all of them.

Censoring is handled explicitly.  A (system, query) execution that was cut off
after ``c`` seconds has an unknown latency in ``[c, T_max]``.  Earlier reports
replaced every such value by ``T_max``; that is only an upper bound, and it is
not comparable across arms because the cut-off differed between arms (20 s to
300 s).  We therefore carry both ends of the interval:

* ``L`` (lower convention): censored latency := its own cut-off,
* ``U`` (upper convention): censored latency := ``T_max``.

A claim is called *robust* only if it holds under the convention that is least
favourable to our system.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated"
TMAX = 300.0          # seconds, statement timeout of the evaluation
THETA = 0.10          # relative band for "faster"/"slower"
UNKNOWN_CUTOFF = 20.0  # cap of the earliest campaign code; used when no cutoff is logged
PROBE_CAP = 45.0       # probe cap of the selection layer

RAW = OrderedDict([
    ("dev", "results/icde_paper_campaign_v4/raw.jsonl"),
    ("tpch", "results/icde_full_suite_tpch10/raw.jsonl"),
    ("calcite", "results/icde_full_suite_calcite10/raw.jsonl"),
    ("zipf", "results/icde_calcite10zipf_stage/raw.jsonl"),
    ("sf50", "results/icde_sf50_cohort/raw.jsonl"),
])
BUDGETS = (1, 2, 4, 8, 16)
VARIANTS = OrderedDict([
    ("full_mcts", "Full (class A + B, priority order, probe, fallback)"),
    ("class_a_symbolic_only", "Class A candidates only"),
    ("class_b_macro_only", "Class B candidates only"),
    ("without_plan_bottleneck", "No plan-steering candidates"),
    ("without_critic", "Generation order instead of priority order"),
    ("without_canary_safety_gate", "Planner cost only (no probe, no fallback)"),
])


def load(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Per-execution latency with explicit censoring
# --------------------------------------------------------------------------
def execution(measurement: dict, original_s: float | None, original_cutoff: float | None = None) -> dict:
    """Return {'lo','hi','exact','cutoff','live'} in seconds for one measurement.

    ``original_s`` is the latency of the original statement when it finished.
    ``original_cutoff`` is set only when the measurement is that of an original.

    For a run that was cut off, ``lo`` must not exceed the cutoff that was really
    applied, otherwise the lower end of the interval is wrong.  The logs do not
    always record that cutoff, so every rule below takes the smallest cutoff
    that the campaign code can have applied:

    * probe short-circuit (scope ``canary_timeout_shortcircuit``): the arm did
      not time the statement; it reused a probe that was cut off.  The probe is
      either one of the selection layer, with cutoff min(original, 45 s), or
      one of the replay arm, with a cutoff between 60 s and 120 s.  The log
      stores a constant 120 s in both cases.  We take min(original, 45 s).
    * final timing: cutoff min(T_max, max(60 s, 1.05 * original)).  The logged
      ``live_seconds`` covers the warmup and all runs, so it can exceed the
      cutoff; we take the smaller of the two.
    * no cutoff logged at all (entries cached from earlier campaigns): 20 s,
      the cap of the earliest campaign code.
    """
    status = measurement.get("status")
    live = bool(measurement.get("live_execution")) and not measurement.get("cache_hit")
    samples = [float(v) for v in (measurement.get("samples_ms") or [])
               if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    if status == "ok" and len(samples) == 3:
        value = min(statistics.median(samples) / 1000.0, TMAX)
        return {"lo": value, "hi": value, "exact": True, "cutoff": None, "live": live}
    scope = measurement.get("scope")
    live_seconds = measurement.get("live_seconds")
    logged = (float(live_seconds) if isinstance(live_seconds, (int, float)) and math.isfinite(live_seconds)
              else None)
    wall = sorted(float(v) / 1000.0 for v in (measurement.get("wall_samples_ms") or [])
                  if isinstance(v, (int, float)) and math.isfinite(v) and v > 0)
    reference = original_s if original_s is not None else TMAX
    final_cutoff = min(TMAX, max(60.0, 1.05 * reference))
    if original_cutoff is not None:
        cutoff = original_cutoff if logged is None else min(original_cutoff, logged)
    elif scope == "canary_timeout_shortcircuit":
        cutoff = min(PROBE_CAP, reference)
    elif logged is not None:
        cutoff = min(logged, final_cutoff)
    else:
        cutoff = min(UNKNOWN_CUTOFF, final_cutoff)
    # Two finished runs plus one censored run still pin down the median exactly.
    if len(wall) >= 2 and wall[1] <= cutoff:
        return {"lo": wall[1], "hi": wall[1], "exact": True, "cutoff": None, "live": live,
                "note": "median of two finished runs and one censored run"}
    return {"lo": cutoff, "hi": TMAX, "exact": False, "cutoff": cutoff, "live": live,
            "probe": scope == "canary_timeout_shortcircuit"}


def original_execution(row: dict, cutoff: float = TMAX) -> dict:
    """The original runs with the cutoff T_max, or 120 s at SF 50 when its SF 10 run is cut off."""
    return execution(dict(row["arms"]["original"]["measurement"]), None, original_cutoff=cutoff)


def arm_execution(row: dict, choice: dict, original: dict) -> dict:
    if choice.get("fallback") or choice.get("sql", "").strip() == row["original_sql"].strip():
        return {**original, "fallback": True}
    original_s = original["lo"] if original["exact"] else None
    return {**execution(choice["measurement"], original_s), "fallback": False}


# --------------------------------------------------------------------------
# Paired outcome of system x against reference y (both are executions)
# --------------------------------------------------------------------------
def versus(x: dict, y: dict, same_sql: bool) -> str:
    if same_sql:
        return "same"
    if x["exact"] and y["exact"]:
        if x["lo"] <= (1 - THETA) * y["lo"]:
            return "faster"
        if x["lo"] >= (1 + THETA) * y["lo"]:
            return "slower"
        return "within"
    if x["exact"] and not y["exact"]:
        return "faster" if x["lo"] <= (1 - THETA) * y["lo"] else "undetermined"
    if not x["exact"] and y["exact"]:
        return "slower" if x["lo"] >= (1 + THETA) * y["lo"] else "undetermined"
    return "undetermined"


def p90(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(executions: list[dict]) -> dict:
    lo = [e["lo"] for e in executions]
    hi = [e["hi"] for e in executions]
    return {"n": len(executions), "mean_L": statistics.mean(lo), "mean_U": statistics.mean(hi),
            "median_L": statistics.median(lo), "median_U": statistics.median(hi),
            "p90_L": p90(lo), "p90_U": p90(hi), "censored": sum(not e["exact"] for e in executions)}


def source_category(row: dict, choice: dict) -> str:
    if choice.get("fallback"):
        return "fallback"
    sql = choice["sql"]
    candidate = choice.get("candidate") or {}
    if "/*+" in sql:
        body = sql.split("*/", 1)[1].strip().rstrip(";").strip()
        return "hint" if body == row["original_sql"].strip().rstrip(";").strip() else "rewrite_hint"
    origin = candidate.get("origin")
    if origin in {"rbot_standalone", "calcite_symbolic"}:
        return "rule"
    if origin == "quite_standalone":
        return "quite"
    return "macro"


def evidence_category(verification: dict) -> str:
    proof = verification.get("proof")
    if proof in {"identity", "ast"}:
        return "syntactic"
    if proof == "structural_empty_v2":
        return "structural"
    if proof == "sqlsolver":
        return "solver"
    return "differential"


# --------------------------------------------------------------------------
# Did the selection layer itself probe a replayed rewrite?
# --------------------------------------------------------------------------
def _plan_signature(document):
    """Same signature as the campaign runner uses to share probes between equal plans."""
    if not isinstance(document, dict):
        return None
    root = document.get("Plan") if "Plan" in document else document
    if not isinstance(root, dict):
        return None
    return (root.get("Node Type"), root.get("Relation Name"), root.get("Join Type"),
            round(float(root.get("Total Cost") or 0.0), 1), round(float(root.get("Plan Rows") or 0.0), 0),
            tuple(_plan_signature(child) for child in (root.get("Plans") or [])))


def layer_probe_state(row: dict, choice: dict, cost_below_original: bool) -> str | None:
    """What the selection layer did with the rewrite that a replay arm returns.

    The log stores one probe record per candidate but not who ran it: the layer
    (cutoff of at most 45 s) or the replay arm's own first run (60 s to 120 s).
    ``inherited``   the log says the probe is copied from a candidate with the same plan;
    ``finished``    the logged first run finishes (the layer's probe or the arm's own run);
    ``not_probed``  two candidates certainly precede the rewrite in the layer's priority
                    order and use both probes, so the cut-off run is the arm's own;
    ``probed``      fewer than two candidates can precede it, so the layer probes it;
    ``uncertain``   the logged data do not decide (the planner cost of an original that
                    is cut off is not logged, so one priority flag can be unknown).
    """
    candidate = choice.get("candidate") or {}
    probe = candidate.get("canary") or {}
    if choice.get("fallback") or not candidate:
        return None
    source, status = probe.get("source"), probe.get("status")
    if source in {"shared_physical_plan_canary", "identical_physical_plan_to_original"}:
        return "inherited"
    if source == "verified_measurement_cache":
        return "cached"
    if source != "live_canary":
        return "none"
    if status == "ok":
        return "finished"

    def bottleneck(c: dict) -> int:
        text = str(c.get("action", "")).lower() + json.dumps(c.get("plan", {}), sort_keys=True).lower()
        return 0 if any(token in text for token in ("nested", "subplan", "material", "decorrel", "join")) else 1

    def tail(c: dict) -> tuple:
        return (bottleneck(c), c.get("cost", 1e300), c.get("generation_index", 10 ** 9))

    target = next((c for c in row.get("candidates", []) if c["sql"] == candidate.get("sql", choice["sql"])), None)
    if target is None:
        return "uncertain"
    target_signature = _plan_signature(target.get("plan"))
    possible, certain = set(), set()
    for c in row.get("candidates", []):
        if c is target:
            continue
        signature = _plan_signature(c.get("plan"))
        if signature == target_signature:
            return "uncertain"          # either of the two can be the one that runs
        depth, target_depth = int(c.get("depth", 1)), int(target.get("depth", 1))
        if depth < target_depth:
            possible.add(signature); certain.add(signature)
        elif depth == target_depth and tail(c) < tail(target):
            possible.add(signature)
            # its priority flag is known only when its cost is below that of a rewrite
            # whose cost is itself below the cost of the original
            if cost_below_original and c.get("cost", 1e300) < target.get("cost", 1e300):
                certain.add(signature)
    if len(possible) < 2:
        return "probed"
    if len(certain) >= 2:
        return "not_probed"
    return "uncertain"


# --------------------------------------------------------------------------
# Build the per-query table for one result file
# --------------------------------------------------------------------------
def build(rows: list[dict], short_cutoff_ids: frozenset = frozenset()) -> list[dict]:
    table = []
    for row in rows:
        original = original_execution(row, 120.0 if row["query_id"] in short_cutoff_ids else TMAX)
        record = {"id": row["query_id"], "benchmark": row["benchmark"], "dataset": row["dataset"],
                  "original": original, "pool": row.get("candidate_pool_size", 0),
                  "accepted": len(row.get("candidates", [])), "arms": {}, "variants": {}}
        for arm in ("rbot", "quite", "mcts"):
            choice = row["arms"][arm]
            ex = arm_execution(row, choice, original)
            ex["sql_sha"] = hash(choice["sql"].strip())
            ex["vs_original"] = versus(ex, original, ex["fallback"])
            if arm != "mcts":
                # the R-Bot arm returns a rewrite only when its planner cost is below that of the original
                ex["layer_probe"] = layer_probe_state(row, choice, cost_below_original=(arm == "rbot"))
            record["arms"][arm] = ex
        for name, choice in row["variants"].items():
            ex = arm_execution(row, choice, original)
            ex["vs_original"] = versus(ex, original, ex["fallback"])
            record["variants"][name] = ex
        ours = record["arms"]["mcts"]
        for arm in ("rbot", "quite"):
            other = record["arms"][arm]
            record["arms"][arm]["ours_vs"] = versus(ours, other, ours["sql_sha"] == other["sql_sha"])
            # Hypothetical: the replayed rewrite guarded by an oracle fallback.
            if other["fallback"]:
                guarded = dict(original)
            elif other["exact"] and original["exact"]:
                value = min(other["lo"], original["lo"])
                guarded = {"lo": value, "hi": value, "exact": True}
            else:
                guarded = {"lo": min(other["lo"], original["lo"]), "hi": min(other["hi"], original["hi"]),
                           "exact": False}
            record["arms"][arm]["guarded"] = guarded
        choice = row["arms"]["mcts"]
        record["source"] = source_category(row, choice)
        record["evidence"] = None if choice.get("fallback") else evidence_category(choice.get("verification", {}))
        record["depth"] = None if choice.get("fallback") else (choice.get("candidate") or {}).get("depth")
        record["max_depth"] = max((int(c.get("depth") or 1) for c in row.get("candidates", [])), default=0)
        record["candidate_evidence"] = Counter(evidence_category(c.get("verification", {}))
                                               for c in row.get("candidates", []))
        table.append(record)
    return table


def summarise(table: list[dict]) -> dict:
    result = {"n": len(table), "original": aggregate([r["original"] for r in table])}
    result["original"]["live"] = sum(r["original"]["live"] for r in table)
    for arm in ("rbot", "quite", "mcts"):
        executions = [r["arms"][arm] for r in table]
        entry = aggregate(executions)
        entry["vs_original"] = dict(Counter(e["vs_original"] for e in executions))
        entry["rewrites"] = sum(not e["fallback"] for e in executions)
        entry["live"] = sum(e["live"] for e in executions if not e["fallback"])
        if arm != "mcts":
            entry["ours_vs"] = dict(Counter(e["ours_vs"] for e in executions))
            entry["guarded"] = aggregate([e["guarded"] for e in executions])
        result[arm] = entry
    result["variants"] = {}
    for name in table[0]["variants"]:
        executions = [r["variants"][name] for r in table]
        entry = aggregate(executions)
        entry["vs_original"] = dict(Counter(e["vs_original"] for e in executions))
        result["variants"][name] = entry
    result["sources"] = dict(Counter(r["source"] for r in table))
    result["evidence"] = dict(Counter(r["evidence"] for r in table if r["evidence"]))
    result["candidate_evidence"] = dict(sum((r["candidate_evidence"] for r in table), Counter()))
    result["pool"] = sum(r["pool"] for r in table)
    result["accepted"] = sum(r["accepted"] for r in table)
    result["depth2"] = sum(1 for r in table if (r["depth"] or 0) >= 2)
    # Latency saved by source (upper convention for the original, exact for ours).
    saved = Counter()
    for r in table:
        if r["source"] != "fallback" and r["arms"]["mcts"]["exact"]:
            saved[r["source"]] += r["original"]["hi"] - r["arms"]["mcts"]["lo"]
    result["saved_by_source"] = dict(saved)
    return result


# --------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------
MACROS: "OrderedDict[str, str]" = OrderedDict()


def macro(name: str, value) -> None:
    assert name.isalpha(), name
    assert name not in MACROS, name
    MACROS[name] = str(value)


def sec(value: float) -> str:
    return f"{value:.1f}"


def interval(low: float, high: float) -> str:
    return sec(low) if abs(high - low) < 0.05 else f"[{sec(low)}, {sec(high)}]"


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def reduction(summary: dict, arm: str = "mcts") -> tuple[float, float]:
    """Mean-latency reduction against Original under both conventions."""
    original = summary["original"]
    entry = summary[arm] if arm in summary else summary["variants"][arm]
    return (1 - entry["mean_L"] / original["mean_L"], 1 - entry["mean_U"] / original["mean_U"])


def count(entry: dict, key: str) -> int:
    return int(entry.get(key, 0))


# --------------------------------------------------------------------------
# Validation session: one session, equal cutoff, interleaved runs
# --------------------------------------------------------------------------
VALIDATION = "results/icde_validation_session/raw.jsonl"


def validation(cohorts: dict, dev_ids: set) -> str:
    """Summarise the equal-cutoff session and return the LaTeX table body.

    Every statement of a pair runs with the cutoff T_max, so a statement that
    is cut off has the capped latency T_max exactly; its true latency is still
    only bounded from below, which the outcome classes respect.
    """
    path = ROOT / VALIDATION
    records = load(VALIDATION) if path.exists() else []
    by_key = {(r["dataset"], r["query_id"]): r for r in records}

    failures = Counter()

    def execution_of(statement: dict) -> dict | None:
        """Median of three rounds; a round that is cut off counts as >= T_max."""
        samples = sorted(min(value / 1000.0, TMAX) for value in statement["samples_ms"])
        if statement["status"] == "ok" and len(samples) >= 3:
            value = statistics.median(samples)
            return {"lo": value, "hi": value, "exact": True}
        if statement["status"] == "cut_off":
            if len(samples) >= 2:      # two finished rounds and one cut off: the median is known
                return {"lo": samples[-1], "hi": samples[-1], "exact": True}
            if len(samples) == 1:      # one finished round, then cut off: the median is bounded
                return {"lo": samples[0], "hi": TMAX, "exact": False}
            return {"lo": TMAX, "hi": TMAX, "exact": False}
        failures[statement["status"]] += 1
        return None

    def observed(statement: dict) -> list[float]:
        """Every round that ran: finished rounds, plus T_max for the round that was cut off."""
        values = [min(value / 1000.0, TMAX) for value in statement["samples_ms"]]
        if statement["status"] == "cut_off":
            values.append(TMAX)
        return values

    def separated(fast: dict, slow: dict) -> bool:
        """All three rounds of `fast` finish and each is shorter than every round of `slow`."""
        return (fast["status"] == "ok" and len(fast["samples_ms"]) >= 3 and bool(observed(slow))
                and max(observed(fast)) < min(observed(slow)))

    groups = OrderedDict([
        ("dev", ("ValDev", r"\textsc{Dev}")),
        ("tpchHeld", ("ValTpchHeld", r"\textsc{TPC-H} held-out")),
        ("calciteHeld", ("ValCalciteHeld", r"\textsc{Calcite} held-out")),
        ("zipf", ("ValZipf", r"\textsc{Zipf}")),
        ("sfFifty", ("ValSfFifty", r"\textsc{SF\,50}")),
    ])
    arm_tags = {"mcts": "Ours", "rbot": "Rbot", "quite": "Quite"}
    arm_labels = {"rbot": "R-Bot replay", "quite": "QUITE replay", "mcts": r"\sys{}"}
    outcomes = (("faster", "Faster"), ("within", "Within"), ("slower", "Slower"), ("undetermined", "Undet"))
    body, pooled = [], {}

    def summarise(pairs: list[dict], tag: str, planned: int) -> dict:
        result = {"done": len(pairs)}
        macro(f"{tag}Planned", planned)
        macro(f"{tag}Done", len(pairs))
        macro(f"{tag}OrigCutOff", sum(1 for p in pairs if not p["original"]["exact"]))

        def mean_interval(executions: list[dict]) -> str:
            if not executions:
                return "n/a"
            return interval(statistics.mean(e["lo"] for e in executions), statistics.mean(e["hi"] for e in executions))

        result["original_mean"] = mean_interval([p["original"] for p in pairs])
        macro(f"{tag}OrigMean", result["original_mean"])
        # the same pairs in the campaign logs, upper end of the interval
        macro(f"{tag}CampaignOrigMean", sec(statistics.mean(p["campaign_original"]["hi"] for p in pairs)) if pairs else "n/a")
        macro(f"{tag}CampaignOursMean", sec(statistics.mean(p["campaign_ours"]["hi"] for p in pairs)) if pairs else "n/a")
        macro(f"{tag}CampaignOursRed", pct(1 - statistics.mean(p["campaign_ours"]["hi"] for p in pairs)
                                           / statistics.mean(p["campaign_original"]["hi"] for p in pairs))
              if pairs else "n/a")
        # originals that finish in the session and in the logs: session latency over logged latency
        ratios = sorted(p["original"]["lo"] / p["campaign_original"]["lo"] for p in pairs
                        if p["original"]["exact"] and p["campaign_original"]["exact"])
        macro(f"{tag}DriftN", len(ratios))
        macro(f"{tag}DriftMin", f"{ratios[0]:.2f}" if ratios else "n/a")
        macro(f"{tag}DriftMax", f"{ratios[-1]:.2f}" if ratios else "n/a")
        macro(f"{tag}DriftMedian", f"{statistics.median(ratios):.2f}" if ratios else "n/a")
        for arm, arm_tag in arm_tags.items():
            executions = [p[arm] or p["original"] for p in pairs]
            versus_original = Counter(versus(p[arm], p["original"], False) for p in pairs if p[arm])
            result[arm] = {"mean": mean_interval(executions),
                           "rewrites": sum(1 for p in pairs if p[arm]), "vs": versus_original}
            macro(f"{tag}{arm_tag}Mean", result[arm]["mean"])
            macro(f"{tag}{arm_tag}Rewrites", result[arm]["rewrites"])
            for outcome, label in outcomes:
                macro(f"{tag}{arm_tag}{label}", versus_original.get(outcome, 0))
            macro(f"{tag}{arm_tag}FasterSep", sum(
                1 for p in pairs if p[arm] and versus(p[arm], p["original"], False) == "faster"
                and separated(p["stmt_" + arm], p["stmt_original"])))
            macro(f"{tag}{arm_tag}SlowerSep", sum(
                1 for p in pairs if p[arm] and versus(p[arm], p["original"], False) == "slower"
                and separated(p["stmt_original"], p["stmt_" + arm])))
            if pairs:
                original_hi = statistics.mean(p["original"]["hi"] for p in pairs)
                macro(f"{tag}{arm_tag}Red", pct(1 - statistics.mean(e["hi"] for e in executions) / original_hi))
            if arm != "mcts":
                against = Counter()
                for p in pairs:
                    ours = p["mcts"] or p["original"]
                    other = p[arm] or p["original"]
                    same = (p["mcts_sha"] or "original") == (p[arm + "_sha"] or "original")
                    against[versus(ours, other, same)] += 1
                result[arm]["ours_vs"] = against
                for outcome, label in outcomes + (("same", "Same"),):
                    macro(f"{tag}{arm_tag}Ours{label}", against.get(outcome, 0))
        return result

    all_pairs = {}
    for key, (tag, label) in groups.items():
        table = cohorts[key]
        planned = [r for r in table if any(not r["arms"][a]["fallback"] for a in arm_tags)]
        pairs = []
        for r in planned:
            record = by_key.get((r["dataset"], r["id"]))
            if record is None:
                continue
            roles = {}
            for statement in record["statements"]:
                for role in statement["roles"]:
                    roles[role] = statement
            original = execution_of(roles["original"])
            if original is None:
                continue
            pair = {"id": r["id"], "dataset": r["dataset"], "original": original,
                    "campaign_ours": r["arms"]["mcts"], "campaign_original": r["original"],
                    "campaign_mcts": r["arms"]["mcts"], "campaign_rbot": r["arms"]["rbot"],
                    "campaign_quite": r["arms"]["quite"], "campaign_source": r["source"]}
            pair["stmt_original"] = roles["original"]
            for arm in arm_tags:
                statement = roles.get(arm)
                pair[arm] = execution_of(statement) if statement else None
                pair[arm + "_sha"] = statement["sql_sha256"] if statement and pair[arm] else None
                pair["stmt_" + arm] = statement if statement and pair[arm] else None
            pairs.append(pair)
        all_pairs[key] = pairs
        summary = summarise(pairs, tag, len(planned))
        if not pairs:
            continue
        cells = [label, f"{len(pairs)} of {len(planned)}", summary["original_mean"]]
        for arm in ("rbot", "quite", "mcts"):
            entry = summary[arm]
            cells += [entry["mean"], str(entry["rewrites"])] + [str(entry["vs"].get(outcome, 0)) for outcome, _ in outcomes]
        body.append(" & ".join(cells) + r" \\")
    held = all_pairs.get("tpchHeld", []) + all_pairs.get("calciteHeld", [])
    planned_held = sum(1 for key in ("tpchHeld", "calciteHeld") for r in cohorts[key]
                       if any(not r["arms"][a]["fallback"] for a in arm_tags))
    summarise(held, "ValHeldAll", planned_held)
    # What each kind of selection finds on the held-out pairs that the other misses.
    def faster(p: dict, arm: str) -> bool:
        return bool(p[arm]) and versus(p[arm], p["original"], False) == "faster"

    missed = [p for p in held if not faster(p, "mcts") and (faster(p, "rbot") or faster(p, "quite"))]
    macro("ValHeldAllMissed", len(missed))
    macro("ValHeldAllMissedKeptOriginal", sum(1 for p in missed if not p["mcts"]))
    macro("ValHeldAllMissedOrigCutOff", sum(1 for p in missed if not p["original"]["exact"]))
    macro("ValHeldAllMissedRbot", sum(1 for p in missed if faster(p, "rbot")))
    macro("ValHeldAllMissedQuiteOnly", sum(1 for p in missed if faster(p, "quite") and not faster(p, "rbot")))
    # the solid subset: the layer keeps an original that is cut off while a replayed rewrite finishes
    rescues = [p for p in missed if not p["mcts"] and not p["original"]["exact"]]
    missed_latency = sorted(min(p[arm]["lo"] for arm in ("rbot", "quite") if faster(p, arm)) for p in rescues)
    macro("ValHeldAllMissedMin", sec(missed_latency[0]) if missed_latency else "n/a")
    macro("ValHeldAllMissedMax", sec(missed_latency[-1]) if missed_latency else "n/a")
    macro("ValHeldAllMissedUnderCap", sum(1 for value in missed_latency if value < PROBE_CAP))
    macro("ValHeldAllMissedOverCap", sum(1 for value in missed_latency if value >= PROBE_CAP))
    # what the layer did with the rewrite that finishes (fastest faster replay statement of the pair)
    states = Counter()
    for p in rescues:
        arm = min((a for a in ("rbot", "quite") if faster(p, a)), key=lambda a: p[a]["lo"])
        state = p["campaign_" + arm].get("layer_probe")
        # In a rescue pair the layer keeps an original that is cut off.  Had the layer seen a probe
        # of this rewrite finish, it would have timed and returned it; a logged first run that
        # finishes is therefore the replay arm's own run, and the layer did not probe the rewrite.
        states["not_probed" if state == "finished" else state] += 1
    macro("ValHeldAllRescueProbed", states.get("probed", 0))
    macro("ValHeldAllRescueInherited", states.get("inherited", 0))
    macro("ValHeldAllRescueNotProbed", states.get("not_probed", 0))
    macro("ValHeldAllRescueUnclear", sum(v for k, v in states.items() if k not in ("probed", "inherited", "not_probed")))
    only_ours = [p for p in held if faster(p, "mcts") and not faster(p, "rbot") and not faster(p, "quite")]
    macro("ValHeldAllOursOnly", len(only_ours))
    macro("ValHeldAllOursOnlyNoRbotRewrite", sum(1 for p in only_ours if not p["rbot"]))
    macro("ValHeldAllOursOnlyRule", sum(1 for p in only_ours if p["campaign_source"] == "rule"))
    macro("ValHeldAllOursOnlyMacro", sum(1 for p in only_ours if p["campaign_source"] == "macro"))
    macro("ValHeldAllOursOnlyOther", sum(1 for p in only_ours if p["campaign_source"] not in ("rule", "macro")))
    # the fastest timed statement of every pair: a reference for a selection without a probe cap
    if held:
        best_lo = statistics.mean(min(e["lo"] for e in (p["original"], p["mcts"], p["rbot"], p["quite"]) if e)
                                  for p in held)
        best_hi = statistics.mean(min(e["hi"] for e in (p["original"], p["mcts"], p["rbot"], p["quite"]) if e)
                                  for p in held)
        macro("ValHeldAllBestMean", interval(best_lo, best_hi))

    everything = [p for pairs in all_pairs.values() for p in pairs]
    planned_all = sum(
        1 for key in groups for r in cohorts[key] if any(not r["arms"][a]["fallback"] for a in arm_tags))
    summarise(everything, "ValAll", planned_all)
    # Wording of the coverage, so that the text reads correctly while the session runs and after it.
    if len(everything) == planned_all:
        macro("ValCoverageLong", f"all {planned_all} pairs")
        macro("ValCoverageShort", f"all {planned_all} pairs")
    else:
        held_note = f", among them all {len(held)} held-out pairs" if len(held) == planned_held else ""
        macro("ValCoverageLong", f"{len(everything)} of the {planned_all} pairs{held_note}")
        macro("ValCoverageShort", f"the {len(everything)} of {planned_all} pairs that it covers")

    # How the two sessions agree on the rewrites that the layer returned.
    confirmed = Counter()
    drift = [p["original"]["lo"] / p["campaign_original"]["lo"] for p in everything
             if p["original"]["exact"] and p["campaign_original"]["exact"]]
    for p in everything:
        if not p["mcts"]:
            continue
        campaign = versus(p["campaign_ours"], p["campaign_original"], False)
        session = versus(p["mcts"], p["original"], False)
        confirmed[(campaign, session)] += 1
    # replayed rewrites whose comparison with the original is undetermined in the logs
    resolved = Counter()
    for p in everything:
        for arm in ("rbot", "quite"):
            if p[arm] and p["campaign_" + arm].get("vs_original") == "undetermined":
                resolved[versus(p[arm], p["original"], False)] += 1
    macro("ValReplayUndetTimed", sum(resolved.values()))
    macro("ValReplayUndetNowFaster", resolved.get("faster", 0))
    macro("ValReplayUndetNowWithin", resolved.get("within", 0))
    macro("ValReplayUndetNowSlower", resolved.get("slower", 0))
    macro("ValReplayUndetStill", resolved.get("undetermined", 0))
    macro("ValOrigCampaignFinished", sum(1 for p in everything if p["campaign_original"]["exact"]))
    macro("ValOrigNowCutOff", sum(1 for p in everything
                                  if p["campaign_original"]["exact"] and not p["original"]["exact"]))
    macro("ValOursTimed", sum(confirmed.values()))
    macro("ValOursConfirmed", confirmed.get(("faster", "faster"), 0))
    macro("ValOursNotConfirmedWithin", confirmed.get(("faster", "within"), 0))
    macro("ValOursNotConfirmedSlower", confirmed.get(("faster", "slower"), 0))
    macro("ValOursNotConfirmedUndet", confirmed.get(("faster", "undetermined"), 0))
    if drift:
        drift.sort()
        macro("ValDriftMedian", f"{statistics.median(drift):.2f}")
        macro("ValDriftMin", f"{drift[0]:.2f}")
        macro("ValDriftMax", f"{drift[-1]:.2f}")
        macro("ValDriftN", len(drift))
    # session against campaign, one point per timed statement (cut-off runs at T_max)
    floor = 1e-3
    originals, rewrites, seen = ["campaign session"], ["campaign session"], set()
    for p in everything:
        originals.append(f"{max(p['campaign_original']['hi'], floor):.4f} {max(p['original']['hi'], floor):.4f}")
        for arm in arm_tags:
            if not p[arm] or p["campaign_" + arm]["fallback"]:
                continue
            key = (p["dataset"], p["id"], p[arm + "_sha"])
            if key in seen:
                continue
            seen.add(key)
            rewrites.append(f"{max(p['campaign_' + arm]['hi'], floor):.4f} {max(p[arm]['hi'], floor):.4f}")
    (OUT / "validation_scatter_original.dat").write_text("\n".join(originals) + "\n")
    (OUT / "validation_scatter_rewrite.dat").write_text("\n".join(rewrites) + "\n")
    macro("ValStatementsTimed", len(originals) + len(rewrites) - 2)
    rewrite_drift = sorted(p[arm]["lo"] / p["campaign_" + arm]["lo"] for p in everything for arm in ("mcts",)
                           if p[arm] and p[arm]["exact"] and not p["campaign_" + arm]["fallback"]
                           and p["campaign_" + arm]["exact"] and p["campaign_" + arm]["lo"] >= 1.0)
    if rewrite_drift:
        macro("ValRewriteDriftMedian", f"{statistics.median(rewrite_drift):.2f}")
        macro("ValRewriteDriftMin", f"{rewrite_drift[0]:.2f}")
        macro("ValRewriteDriftMax", f"{rewrite_drift[-1]:.2f}")
        macro("ValRewriteDriftN", len(rewrite_drift))
    macro("ValFailures", sum(failures.values()))
    # spread of the three rounds of one statement: largest over smallest sample
    spreads = sorted(max(s["samples_ms"]) / min(s["samples_ms"]) for r in records for s in r["statements"]
                     if s["status"] == "ok" and len(s["samples_ms"]) >= 3 and min(s["samples_ms"]) >= 1000.0)
    if spreads:
        macro("ValSpreadMedian", f"{statistics.median(spreads):.2f}")
        macro("ValSpreadPninety", f"{p90(spreads):.2f}")
        macro("ValSpreadN", len(spreads))
    # originals that the campaign logs record as cut off but that finish in the session
    macro("ValOrigNowFinished", sum(1 for p in everything
                                    if p["original"]["exact"] and not p["campaign_original"]["exact"]))
    macro("ValOrigCampaignCutOff", sum(1 for p in everything if not p["campaign_original"]["exact"]))
    loads = [r["load_before"][0] for r in records if "load_before" in r]
    if loads:
        macro("ValLoadMedian", f"{statistics.median(loads):.1f}")
    macro("ValRecords", len(records))
    (OUT / "validation.json").write_text(json.dumps(
        {key: [{k: v for k, v in p.items() if not k.startswith(("campaign", "stmt_"))} for p in pairs]
         for key, pairs in all_pairs.items()}, indent=1, sort_keys=True))
    return "\n".join(body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {name: load(path) for name, path in RAW.items()}
    # At SF 50 the campaign code times an original with a cutoff of 120 s when the
    # cached SF 10 measurement of the same statement is not a finished timing.
    sf10_unfinished = frozenset(row["query_id"] for row in raw["dev"]
                                if row["arms"]["original"]["measurement"].get("status") != "ok")
    tables = {name: build(rows, sf10_unfinished if name == "sf50" else frozenset()) for name, rows in raw.items()}
    dev_ids = {r["id"] for r in tables["dev"]}

    cohorts = OrderedDict()
    cohorts["dev"] = tables["dev"]
    cohorts["tpchHeld"] = [r for r in tables["tpch"] if r["id"] not in dev_ids]
    cohorts["calciteHeld"] = [r for r in tables["calcite"] if r["id"] not in dev_ids]
    cohorts["zipf"] = tables["zipf"]
    zipf_ids = {r["id"] for r in tables["zipf"]}
    cohorts["uniformSame"] = [r for r in tables["calcite"] if r["id"] in zipf_ids]
    cohorts["sfFifty"] = [r for r in tables["sf50"] if r["benchmark"] in {"tpch", "dsb"}]
    sf50_ids = {r["id"] for r in cohorts["sfFifty"]}
    cohorts["sfTenSame"] = [r for r in tables["dev"] if r["id"] in sf50_ids]
    cohorts["heldAll"] = cohorts["tpchHeld"] + cohorts["calciteHeld"]
    cohorts["tpchAll"] = tables["tpch"]
    cohorts["calciteAll"] = tables["calcite"]
    summaries = {name: summarise(table) for name, table in cohorts.items()}

    # ------------------------------------------------------------------ macros
    macro("Tmax", f"{TMAX:.0f}")
    macro("ThetaPct", f"{100 * THETA:.0f}")
    macro("UnknownCutoff", f"{UNKNOWN_CUTOFF:.0f}")
    names = {"dev": "Dev", "tpchHeld": "TpchHeld", "calciteHeld": "CalciteHeld", "zipf": "Zipf",
             "uniformSame": "UniformSame", "sfFifty": "SfFifty", "sfTenSame": "SfTenSame",
             "heldAll": "HeldAll", "tpchAll": "TpchAll", "calciteAll": "CalciteAll"}
    arm_names = {"rbot": "Rbot", "quite": "Quite", "mcts": "Ours"}
    for key, tag in names.items():
        s = summaries[key]
        macro(f"{tag}N", s["n"])
        macro(f"{tag}OrigMean", interval(s["original"]["mean_L"], s["original"]["mean_U"]))
        macro(f"{tag}OrigMeanU", sec(s["original"]["mean_U"]))
        macro(f"{tag}OrigCensored", s["original"]["censored"])
        macro(f"{tag}OrigLive", s["original"]["live"])
        for arm, arm_tag in arm_names.items():
            e = s[arm]
            macro(f"{tag}{arm_tag}Mean", interval(e["mean_L"], e["mean_U"]))
            macro(f"{tag}{arm_tag}MeanL", sec(e["mean_L"]))
            macro(f"{tag}{arm_tag}MeanU", sec(e["mean_U"]))
            macro(f"{tag}{arm_tag}Rewrites", e["rewrites"])
            macro(f"{tag}{arm_tag}Censored", e["censored"] - sum(
                1 for r in cohorts[key] if r["arms"][arm]["fallback"] and not r["arms"][arm]["exact"]))
            for outcome, label in (("faster", "Faster"), ("slower", "Slower"), ("within", "Within"),
                                   ("undetermined", "Undet"), ("same", "Same")):
                macro(f"{tag}{arm_tag}{label}", count(e["vs_original"], outcome))
            low, high = reduction(s, arm)
            macro(f"{tag}{arm_tag}RedL", pct(low))
            macro(f"{tag}{arm_tag}RedU", pct(high))
            if arm != "mcts":
                for outcome, label in (("faster", "OursFaster"), ("slower", "OursSlower"),
                                       ("within", "OursWithin"), ("undetermined", "OursUndet"),
                                       ("same", "OursSame")):
                    macro(f"{tag}{arm_tag}{label}", count(e["ours_vs"], outcome))
                macro(f"{tag}{arm_tag}GuardedMean", interval(e["guarded"]["mean_L"], e["guarded"]["mean_U"]))
        macro(f"{tag}Pool", s["pool"])
        macro(f"{tag}Accepted", s["accepted"])
        macro(f"{tag}RejectedPct", pct(1 - s["accepted"] / s["pool"]) if s["pool"] else "0.0")
        for category, label in (("macro", "SrcMacro"), ("rule", "SrcRule"), ("quite", "SrcQuite"),
                                ("hint", "SrcHint"), ("rewrite_hint", "SrcRewriteHint"),
                                ("fallback", "SrcFallback")):
            macro(f"{tag}{label}", count(s["sources"], category))
        for category, label in (("syntactic", "EvSyntactic"), ("structural", "EvStructural"),
                                ("solver", "EvSolver"), ("differential", "EvDifferential")):
            macro(f"{tag}{label}", count(s["evidence"], category))
            macro(f"{tag}Cand{label}", count(s["candidate_evidence"], category))
        macro(f"{tag}DepthTwo", s["depth2"])

    for key in ("dev", "tpchHeld", "calciteHeld", "heldAll", "zipf", "uniformSame", "sfFifty", "sfTenSame"):
        s = summaries[key]
        macro(f"{names[key]}OursFasterPct", pct(count(s["mcts"]["vs_original"], "faster") / s["n"]))
        macro(f"{names[key]}OursLive", s["mcts"]["live"])
    for key in ("dev", "tpchHeld", "calciteHeld", "heldAll", "zipf", "sfFifty"):
        macro(f"{names[key]}Rescued", sum(1 for r in cohorts[key] if not r["original"]["exact"]
                                           and not r["arms"]["mcts"]["fallback"] and r["arms"]["mcts"]["exact"]))
    # derived quantities quoted in prose
    quite_low, quite_high = reduction(summaries["tpchHeld"], "quite")
    macro("TpchHeldQuiteIncL", pct(-quite_low))
    macro("TpchHeldQuiteIncU", pct(-quite_high))
    macro("DevGenOrderLoss", sec(summaries["dev"]["variants"]["without_critic"]["mean_U"]
                                 - summaries["dev"]["variants"]["full_mcts"]["mean_U"]))
    macro("HeldAllClassAGap", sec(summaries["heldAll"]["variants"]["class_a_symbolic_only"]["mean_U"]
                                  - summaries["heldAll"]["variants"]["full_mcts"]["mean_U"]))
    macro("PenaltyFactor", f"{TMAX / UNKNOWN_CUTOFF:.0f}")
    macro("HeldAllAcceptedPerStatement", f"{summaries['heldAll']['accepted'] / summaries['heldAll']['n']:.2f}")
    held = summaries["heldAll"]
    macro("HeldAllBudgetOneImproved", count(held["variants"]["mcts_M1"]["vs_original"], "faster"))
    macro("HeldAllBudgetSixteenImproved", count(held["variants"]["mcts_M16"]["vs_original"], "faster"))
    macro("HeldAllBudgetSaturation", next(
        b for b in BUDGETS if abs(held["variants"][f"mcts_M{b}"]["mean_U"]
                                  - held["variants"]["mcts_M16"]["mean_U"]) < 1e-9))
    macro("HeldAllVarFullMeanU", sec(held["variants"]["full_mcts"]["mean_U"]))
    # the one held-out query where a replayed rewrite beats our first pass
    missed = [r for r in cohorts["heldAll"] if r["arms"]["rbot"].get("ours_vs") == "slower"]
    macro("MissedCount", len(missed))
    if missed:
        macro("MissedReplay", sec(missed[0]["arms"]["rbot"]["lo"]))

    # replayed rewrites whose logged timeout is a reused probe (cutoff not logged)
    for key in ("dev", "tpchHeld", "calciteHeld", "heldAll", "zipf", "sfFifty"):
        for arm, arm_tag in (("rbot", "Rbot"), ("quite", "Quite")):
            macro(f"{names[key]}{arm_tag}ProbeCut", sum(
                1 for r in cohorts[key] if not r["arms"][arm]["fallback"] and r["arms"][arm].get("probe")))
    macro("AllReplayProbeCut", sum(1 for key in ("dev", "tpchHeld", "calciteHeld", "zipf", "sfFifty")
                                   for r in cohorts[key] for arm in ("rbot", "quite")
                                   if not r["arms"][arm]["fallback"] and r["arms"][arm].get("probe")))
    macro("AllReplayCutOff", sum(1 for key in ("dev", "tpchHeld", "calciteHeld", "zipf", "sfFifty")
                                 for r in cohorts[key] for arm in ("rbot", "quite")
                                 if not r["arms"][arm]["fallback"] and not r["arms"][arm]["exact"]))
    macro("ProbeCap", f"{PROBE_CAP:.0f}")

    # provenance of the development cohort
    dev = cohorts["dev"]
    macro("DevCutTwenty", sum(1 for r in dev for a in ("rbot", "quite")
                               if not r["arms"][a]["fallback"] and not r["arms"][a]["exact"]
                               and r["arms"][a]["cutoff"] is not None and r["arms"][a]["cutoff"] <= 21))
    macro("DevBaselineCensored", sum(1 for r in dev for a in ("rbot", "quite")
                                      if not r["arms"][a]["fallback"] and not r["arms"][a]["exact"]))
    seeded = {"rbot:tpch:query1_0", "rbot:tpch:query3_0", "rbot:tpch:query9_0", "rbot:tpch:query10_0",
              "rbot:tpch:query12_0", "rbot:tpch:query18_0", "rbot:tpch:query21_0", "rbot:dsb:query065_0",
              "rbot:dsb:query069_0", "rbot:dsb:query081_0", "rbot:calcite:testDecorrelateTwoExists",
              "rbot:calcite:testIsNullPushDown"}
    macro("SeededQueries", len(seeded))
    macro("SeededInHeld", sum(1 for r in cohorts["heldAll"] if r["id"] in seeded))
    macro("DevSeededImproved", sum(1 for r in dev if r["id"] in seeded
                                    and r["arms"]["mcts"]["vs_original"] == "faster"))
    macro("DevUnseededN", sum(1 for r in dev if r["id"] not in seeded))
    macro("DevUnseededImproved", sum(1 for r in dev if r["id"] not in seeded
                                      and r["arms"]["mcts"]["vs_original"] == "faster"))

    # budget curves
    lines = ["M dev tpchHeld calciteHeld zipf sfFifty devImp tpchHeldImp calciteHeldImp zipfImp sfFiftyImp"]
    keys = ("dev", "tpchHeld", "calciteHeld", "zipf", "sfFifty")
    for budget in BUDGETS:
        ratio, improved = [], []
        for key in keys:
            s = summaries[key]
            entry = s["variants"][f"mcts_M{budget}"]
            ratio.append(entry["mean_U"] / s["original"]["mean_U"])
            improved.append(100.0 * count(entry["vs_original"], "faster") / s["n"])
        lines.append(" ".join([str(budget)] + [f"{v:.4f}" for v in ratio] + [f"{v:.2f}" for v in improved]))
    (OUT / "budget.dat").write_text("\n".join(lines) + "\n")
    for key in keys:
        s = summaries[key]
        tag = names[key]
        first = s["variants"]["mcts_M1"]
        last = s["variants"]["mcts_M16"]
        macro(f"{tag}BudgetOneImproved", count(first["vs_original"], "faster"))
        macro(f"{tag}BudgetSixteenImproved", count(last["vs_original"], "faster"))
        macro(f"{tag}BudgetOneRedU", pct(1 - first["mean_U"] / s["original"]["mean_U"]))
        macro(f"{tag}BudgetSixteenRedU", pct(1 - last["mean_U"] / s["original"]["mean_U"]))
        saturation = next(b for b in BUDGETS
                          if abs(s["variants"][f"mcts_M{b}"]["mean_U"] - last["mean_U"]) < 1e-9)
        macro(f"{tag}BudgetSaturation", saturation)

    # policy variants (development cohort and held-out statements)
    for name, tag in (("class_a_symbolic_only", "ClassA"), ("class_b_macro_only", "ClassB"),
                      ("without_plan_bottleneck", "NoSteer"), ("without_critic", "GenOrder"),
                      ("without_canary_safety_gate", "CostOnly"), ("full_mcts", "Full")):
        for key in ("dev", "heldAll"):
            entry = summaries[key]["variants"][name]
            prefix = f"{names[key]}Var{tag}"
            macro(f"{prefix}Mean", interval(entry["mean_L"], entry["mean_U"]))
            macro(f"{prefix}Faster", count(entry["vs_original"], "faster"))
            macro(f"{prefix}Slower", count(entry["vs_original"], "slower"))
            macro(f"{prefix}Undet", count(entry["vs_original"], "undetermined"))

    # main results table
    body = []
    labels = {"dev": r"\textsc{Dev}, 30 statements, timings from the cache",
              "tpchHeld": r"\textsc{TPC-H} held-out, 97 statements, live timings",
              "calciteHeld": r"\textsc{Calcite} held-out, 34 statements, live timings"}
    for key in ("dev", "tpchHeld", "calciteHeld"):
        s = summaries[key]
        body.append(r"\multicolumn{7}{@{}l}{\textit{" + labels[key] + r"}} \\")
        o = s["original"]
        body.append(" & ".join(["Original", interval(o["mean_L"], o["mean_U"]),
                                interval(o["median_L"], o["median_U"]), "--", "--", "--", "--"]) + r" \\")
        for arm, label in (("rbot", "R-Bot replay"), ("quite", "QUITE replay"), ("mcts", r"\sys{}")):
            e = s[arm]
            v = e["vs_original"]
            body.append(" & ".join([label, interval(e["mean_L"], e["mean_U"]),
                                    interval(e["median_L"], e["median_U"]), str(e["rewrites"]),
                                    str(count(v, "faster")), str(count(v, "slower")),
                                    str(count(v, "undetermined"))]) + r" \\")
        if key != "calciteHeld":
            body.append(r"\addlinespace")
    # A table body cannot be \input directly in front of \multicolumn, so it
    # is wrapped in a macro that the preamble loads.
    (OUT / "tables.tex").write_text("% Generated by paper/scripts/build_data.py. Do not edit.\n"
                                    "\\newcommand{\\TabMainBody}{%\n" + "\n".join(body) + "\n}\n")

    # paired outcomes (ours against each replay arm): stacked bar data
    lines = ["label faster within same slower undetermined"]
    for key, label in (("dev", "Dev"), ("tpchHeld", "TPC-H held-out"), ("calciteHeld", "Calcite held-out"),
                       ("zipf", "Zipf"), ("sfFifty", "SF 50")):
        for arm, arm_label in (("rbot", "R-Bot"), ("quite", "QUITE")):
            v = summaries[key][arm]["ours_vs"]
            total = summaries[key]["n"]
            lines.append(" ".join([f"{{{label}, {arm_label}}}"] + [f"{100.0 * count(v, k) / total:.2f}" for k in
                                  ("faster", "within", "same", "slower", "undetermined")]))
    (OUT / "paired.dat").write_text("\n".join(lines) + "\n")

    # per-query latency, one small file per mark type so the plot needs no filters
    floor = 1e-3  # seconds; instant rewrites are drawn at the axis floor
    for key in ("dev", "heldAll"):
        table = sorted(cohorts[key], key=lambda r: (r["original"]["hi"], r["original"]["lo"], r["id"]))
        finished, censored, rewritten = ["rank value"], ["rank value"], ["rank value"]
        for index, r in enumerate(table, 1):
            target = finished if r["original"]["exact"] else censored
            target.append(f"{index} {max(r['original']['hi'], floor):.4f}")
            ours = r["arms"]["mcts"]
            if not ours["fallback"]:
                rewritten.append(f"{index} {max(ours['hi'], floor):.4f}")
        (OUT / f"perquery_{key}_finished.dat").write_text("\n".join(finished) + "\n")
        (OUT / f"perquery_{key}_censored.dat").write_text("\n".join(censored) + "\n")
        (OUT / f"perquery_{key}_ours.dat").write_text("\n".join(rewritten) + "\n")

    # scale and skew: same queries on two databases (bar = upper convention,
    # downward whisker reaches the lower convention)
    for name, pairs in (("scale", (("SF10", "sfTenSame"), ("SF50", "sfFifty"))),
                        ("skew", (("Uniform", "uniformSame"), ("Zipf", "zipf")))):
        lines = ["group orig origGap ours oursGap"]
        for label, key in pairs:
            s = summaries[key]
            lines.append(f"{label} {s['original']['mean_U']:.3f} "
                         f"{s['original']['mean_U'] - s['original']['mean_L']:.3f} "
                         f"{s['mcts']['mean_U']:.3f} {s['mcts']['mean_U'] - s['mcts']['mean_L']:.3f}")
        (OUT / f"{name}.dat").write_text("\n".join(lines) + "\n")
    # how many of the same queries improve on both databases
    for a, b, tag in (("sfTenSame", "sfFifty", "Scale"), ("uniformSame", "zipf", "Skew")):
        first = {r["id"]: r for r in cohorts[a]}
        second = {r["id"]: r for r in cohorts[b]}
        both = sum(1 for q in first if first[q]["arms"]["mcts"]["vs_original"] == "faster"
                   and second[q]["arms"]["mcts"]["vs_original"] == "faster")
        only_first = sum(1 for q in first if first[q]["arms"]["mcts"]["vs_original"] == "faster"
                         and second[q]["arms"]["mcts"]["vs_original"] != "faster")
        only_second = sum(1 for q in first if first[q]["arms"]["mcts"]["vs_original"] != "faster"
                          and second[q]["arms"]["mcts"]["vs_original"] == "faster")
        macro(f"{tag}BothImproved", both)
        macro(f"{tag}OnlyFirstImproved", only_first)
        macro(f"{tag}OnlySecondImproved", only_second)
        macro(f"{tag}OrigCensoredFirst", summaries[a]["original"]["censored"])
        macro(f"{tag}OrigCensoredSecond", summaries[b]["original"]["censored"])

    # source attribution (improved queries only), stacked bars
    lines = ["label macro rule quite hint rewritehint"]
    for key, label in (("dev", "Dev"), ("tpchHeld", "{TPC-H held-out}"), ("calciteHeld", "{Calcite held-out}"),
                       ("zipf", "Zipf"), ("sfFifty", "{SF 50}")):
        counter = Counter(r["source"] for r in cohorts[key] if r["arms"]["mcts"]["vs_original"] == "faster")
        lines.append(" ".join([label] + [str(counter.get(k, 0)) for k in
                                         ("macro", "rule", "quite", "hint", "rewrite_hint")]))
    (OUT / "sources.dat").write_text("\n".join(lines) + "\n")
    pooled = Counter(r["source"] for key in ("dev", "heldAll") for r in cohorts[key]
                     if r["arms"]["mcts"]["vs_original"] == "faster")
    for category, label in (("macro", "Macro"), ("rule", "Rule"), ("quite", "Quite"), ("hint", "Hint"),
                            ("rewrite_hint", "RewriteHint")):
        macro(f"PooledSrc{label}", pooled.get(category, 0))
    macro("PooledImproved", sum(pooled.values()))
    macro("PooledReplayShare", pct((pooled.get("rule", 0) + pooled.get("quite", 0)) / max(1, sum(pooled.values()))))

    # evidence behind selected rewrites, pooled over all distinct (database, query) pairs
    distinct = {}
    for key in ("dev", "tpchHeld", "calciteHeld", "zipf", "sfFifty"):
        for r in cohorts[key]:
            distinct[(r["dataset"], r["id"])] = r
    evidence = Counter(r["evidence"] for r in distinct.values() if r["evidence"])
    macro("AllPairs", len(distinct))
    macro("AllSelected", sum(evidence.values()))
    for category, label in (("syntactic", "Syntactic"), ("structural", "Structural"),
                            ("solver", "Solver"), ("differential", "Differential")):
        macro(f"AllEv{label}", evidence.get(category, 0))
    macro("AllEvFormalPct", pct((evidence.get("syntactic", 0) + evidence.get("structural", 0)
                                 + evidence.get("solver", 0)) / max(1, sum(evidence.values()))))
    macro("AllPool", sum(r["pool"] for r in distinct.values()))
    macro("AllAccepted", sum(r["accepted"] for r in distinct.values()))
    macro("AllRejectedPct", pct(1 - sum(r["accepted"] for r in distinct.values())
                                / max(1, sum(r["pool"] for r in distinct.values()))))
    improved_all = [r for r in distinct.values() if r["arms"]["mcts"]["vs_original"] == "faster"]
    macro("AllImproved", len(improved_all))

    # amortisation: worst-case probe budget against per-execution saving
    long_targets = {"rbot:tpch:query18_0", "rbot:dsb:query081_0", "rbot:calcite:testDecorrelateTwoExists",
                    "rbot:tpch:query10_0", "rbot:tpch:query12_0"}
    breakeven = []
    for r in improved_all:
        ours = r["arms"]["mcts"]["lo"]
        original = r["original"]["lo"]          # conservative: censored originals count at their cut-off
        cap = 120.0 if r["id"] in long_targets else 45.0
        budget = 2 * min(original, cap) + 4 * ours
        saving = original - ours
        if saving > 0:
            breakeven.append(budget / saving)
    breakeven.sort()
    macro("BreakevenMedian", f"{statistics.median(breakeven):.1f}")
    macro("BreakevenPninety", f"{p90(breakeven):.1f}")
    macro("BreakevenMax", f"{max(breakeven):.1f}")
    macro("BreakevenWithinFive", sum(1 for v in breakeven if v <= 5))
    macro("BreakevenN", len(breakeven))
    (OUT / "breakeven.dat").write_text("rank runs\n" + "\n".join(
        f"{i} {v:.3f}" for i, v in enumerate(breakeven, 1)) + "\n")

    # running example
    example = next(r for r in raw["dev"] if r["query_id"] == "rbot:dsb:query081_0")
    ours = example["arms"]["mcts"]["measurement"]
    macro("ExampleOurs", sec(statistics.median(ours["samples_ms"]) / 1000.0))
    macro("ExamplePool", example["candidate_pool_size"])
    macro("ExampleAccepted", len(example["candidates"]))
    macro("ExampleMacroCost", f"{next(c['cost'] for c in example['candidates'] if c['action'] == 'decorrelate_return_average_by_group'):,.0f}".replace(",", "\\,"))
    macro("ExampleRuleCost", f"{next(c['cost'] for c in example['candidates'] if c['source'] == 'BaselineRegistry'):,.0f}".replace(",", "\\,"))

    # session variance evidence (same SQL, two sessions)
    r2 = {r["query_id"]: r for r in load("results/stage_a_sf10_v2_clean_r2/raw.jsonl")}
    dev_by_id = {r["id"]: r for r in dev}
    q12_old = r2["rbot:tpch:query12_0"]["arms"]["original"]["measurement"].get("wall_samples_ms", [])
    if q12_old:
        macro("DriftOld", sec(q12_old[0] / 1000.0))
        macro("DriftNew", sec(dev_by_id["rbot:tpch:query12_0"]["original"]["lo"]))

    # candidate funnel per cohort: proposed, admitted by the gate, returned
    plot_cohorts = (("dev", "Dev"), ("tpchHeld", "{TPC-H held-out}"), ("calciteHeld", "{Calcite held-out}"),
                    ("zipf", "Zipf"), ("sfFifty", "{SF 50}"))
    lines = ["label proposed admitted returned"]
    for key, label in plot_cohorts:
        s_ = summaries[key]
        lines.append(f"{label} {s_['pool']} {s_['accepted']} {s_['mcts']['rewrites']}")
    (OUT / "funnel.dat").write_text("\n".join(lines) + "\n")

    # outcome of every arm against the original, all cohorts (share of statements)
    lines = ["label faster within same slower undetermined"]
    for key, label in plot_cohorts:
        for arm, arm_label in (("rbot", "R-Bot"), ("quite", "QUITE"), ("mcts", "ours")):
            v = summaries[key][arm]["vs_original"]
            total = summaries[key]["n"]
            lines.append(" ".join([f"{{{label.strip('{}')}, {arm_label}}}"] + [
                f"{100.0 * count(v, k) / total:.2f}" for k in ("faster", "within", "same", "slower", "undetermined")]))
    (OUT / "outcomes.dat").write_text("\n".join(lines) + "\n")

    # pool statistics over all distinct pairs
    macro("MaxPool", max(r["pool"] for r in distinct.values()))
    macro("EmptyPools", sum(1 for r in distinct.values() if r["pool"] == 0))
    macro("TpchHeldEmptyPools", sum(1 for r in cohorts["tpchHeld"] if r["pool"] == 0))
    macro("MaxDepth", max(r["max_depth"] for r in distinct.values()))

    validation_tables = validation(cohorts, dev_ids)
    with (OUT / "tables.tex").open("a") as stream:
        stream.write("\\newcommand{\\TabValBody}{%\n" + validation_tables + "\n}\n")

    (OUT / "numbers.tex").write_text("% Generated by paper/scripts/build_data.py. Do not edit.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in MACROS.items()))

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items() if k != "sql_sha"}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, Counter):
            return dict(obj)
        return obj
    (OUT / "audit.json").write_text(json.dumps({"summaries": clean(summaries),
                                                "queries": {k: clean(v) for k, v in cohorts.items()}},
                                               indent=1, sort_keys=True))
    print(f"wrote {len(MACROS)} macros")
    for key in ("dev", "tpchHeld", "calciteHeld", "heldAll", "zipf", "uniformSame", "sfFifty", "sfTenSame"):
        s = summaries[key]
        print(f"\n== {key} n={s['n']}  original mean [{s['original']['mean_L']:.1f}, {s['original']['mean_U']:.1f}] "
              f"censored={s['original']['censored']} live={s['original']['live']}")
        for arm in ("rbot", "quite", "mcts"):
            e = s[arm]
            low, high = reduction(s, arm)
            extra = "" if arm == "mcts" else f" | ours_vs={e['ours_vs']} | guarded=[{e['guarded']['mean_L']:.1f}, {e['guarded']['mean_U']:.1f}]"
            print(f"   {arm:5s} mean [{e['mean_L']:.1f}, {e['mean_U']:.1f}] red L/U {100*low:.1f}/{100*high:.1f}% "
                  f"rewrites={e['rewrites']} live={e['live']} vs_orig={e['vs_original']}{extra}")
        print("   sources:", s["sources"], "| evidence:", s["evidence"], "| pool/accepted:", s["pool"], s["accepted"])
        for name in VARIANTS:
            e = s["variants"][name]
            print(f"   var {name:28s} mean [{e['mean_L']:.1f}, {e['mean_U']:.1f}] {e['vs_original']}")
        print("   budget improved:", [count(s['variants'][f'mcts_M{b}']['vs_original'], 'faster') for b in BUDGETS],
              "mean_U:", [round(s['variants'][f'mcts_M{b}']['mean_U'], 1) for b in BUDGETS])
    print("\nbreak-even runs:", [round(v, 1) for v in breakeven])

    claim_checks()


def claim_checks() -> None:
    """Re-test every comparative sentence of the paper that is not itself a macro.

    The numbers in the text update automatically, but a sentence such as "A is
    robustly faster than B" is prose.  Each entry below names the sections that
    contain such a sentence and the condition under which it is true.
    """
    import re

    def bounds(name: str) -> tuple[float, float]:
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", MACROS[name])]
        if not numbers:
            raise KeyError(f"{name} has no value yet")
        return numbers[0], numbers[-1]

    def lo(name: str) -> float:
        return bounds(name)[0]

    def hi(name: str) -> float:
        return bounds(name)[1]

    def within(inner: str, outer: str) -> bool:
        return lo(outer) <= lo(inner) and hi(inner) <= hi(outer)

    checks = [
        ("Abstract, I, VIII-G, XI", "session, held-out: the layer is faster than QUITE replay",
         lambda: hi("ValHeldAllOursMean") < lo("ValHeldAllQuiteMean")),
        ("Abstract, I, VIII-G, XI", "session, held-out: R-Bot replay is faster than the layer",
         lambda: hi("ValHeldAllRbotMean") < lo("ValHeldAllOursMean")),
        ("VIII-G", "session, held-out: the layer is faster than Original",
         lambda: hi("ValHeldAllOursMean") < lo("ValHeldAllOrigMean")),
        ("VIII-B, VIII-G", "session, held-out: more than one cut-off original is kept although a replay finishes",
         lambda: lo("ValHeldAllMissedOrigCutOff") > 1),
        ("I, VIII-B", "logs, held-out: the layer is robustly faster than QUITE replay",
         lambda: hi("HeldAllOursMean") < lo("HeldAllQuiteMean")),
        ("VIII-B", "logs, held-out: the R-Bot interval contains the mean of the layer",
         lambda: within("HeldAllOursMean", "HeldAllRbotMean")),
        ("VIII-B", "logs, TPC-H held-out: R-Bot replay is robustly better than the layer",
         lambda: hi("TpchHeldRbotMean") < lo("TpchHeldOursMean")),
        ("I, VIII-C", "logs, Zipf: the layer is robustly faster than both replays",
         lambda: hi("ZipfOursMean") < min(lo("ZipfRbotMean"), lo("ZipfQuiteMean"))),
        ("VIII-C", "logs: the reduction on Zipf data is within 10 points of the reduction on uniform data",
         lambda: abs(lo("ZipfOursRedU") - lo("UniformSameOursRedU")) <= 10.0),
        ("VIII-A", "logs, Dev: the layer is robustly faster than QUITE replay",
         lambda: hi("DevOursMean") < lo("DevQuiteMean")),
        ("VIII-A", "logs, Dev: only the lower end of the R-Bot interval is close to the layer",
         lambda: abs(lo("DevRbotMean") - lo("DevOursMean")) <= 0.1 * lo("DevOursMean")
         and hi("DevRbotMean") > 1.5 * hi("DevOursMean")),
        ("VIII-D", "logs: a budget of one yields more than half of the final improvements",
         lambda: 2 * lo("DevBudgetOneImproved") > lo("DevBudgetSixteenImproved")
         and 2 * lo("HeldAllBudgetOneImproved") > lo("HeldAllBudgetSixteenImproved")),
        ("VIII-E", "logs, Dev: planner cost only is robustly worse than the full policy",
         lambda: lo("DevVarCostOnlyMean") > hi("DevVarFullMean")),
        ("VIII-E", "logs, held-out: the cost-only interval contains the means of Original and of the full policy",
         lambda: within("HeldAllOrigMean", "HeldAllVarCostOnlyMean")
         and within("HeldAllVarFullMean", "HeldAllVarCostOnlyMean")),
        ("VIII-E", "logs: class B carries Dev, class A carries the held-out statements",
         lambda: hi("DevVarClassBMean") < lo("DevVarClassAMean")
         and hi("HeldAllVarClassAMean") < lo("HeldAllVarClassBMean")),
    ]
    if lo("ValZipfDone") > 0:
        checks.append(("I, VIII-C next to Table IV",
                       "session, Zipf (covered pairs): the layer is faster than both replays",
                       lambda: hi("ValZipfOursMean") < min(lo("ValZipfRbotMean"), lo("ValZipfQuiteMean"))))
    if lo("ValDevDone") > 0:
        checks.append(("VIII-A next to Table IV",
                       "session, Dev (covered pairs): the layer is faster than QUITE replay",
                       lambda: hi("ValDevOursMean") < lo("ValDevQuiteMean")))
    lines = []
    for where, text, test in checks:
        try:
            verdict = "ok    " if test() else "REVIEW"
        except Exception as error:
            verdict, text = "REVIEW", f"{text} (cannot evaluate: {error})"
        lines.append(f"claim check {verdict} [{where}] {text}")
    (OUT / "claim_checks.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
