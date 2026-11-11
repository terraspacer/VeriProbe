#!/usr/bin/env python3
"""Execute the fixed 30-query SF10 comparison with fresh batched model calls.

The supervising Codex session must service the printed file spool. Credentials
stay in memory and the process environment. Original and each distinct final
SQL receive one warmup and three measured executions. Search observations never
replace these final samples. No SF50 or Zipf dataset is accepted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("original", "rbot", "quite", "mcts")
ARM_LIMITS = {"rbot": 2, "quite": 3, "mcts": 3}
DATASETS = {"tpch": "tpch10", "dsb": "dsb10", "calcite": "calcite10"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    body = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def public(value):
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items()
                if key.lower() not in {"dsn", "read_dsn", "admin_dsn", "password", "api_key", "token", "secret"}}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value


def error_record(exc):
    return {"error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__,
            "sqlstate": getattr(exc, "sqlstate", None)}


def infrastructure_failure(value):
    if isinstance(value, dict):
        if value.get("error_type") in {"OperationalError", "InterfaceError", "AdminShutdown", "CrashShutdown", "CannotConnectNow", "ConnectionFailure", "InfrastructureFailure", "ConnectionRefusedError"}:
            return True
        state = value.get("sqlstate") or ""
        if isinstance(state, str) and (state.startswith("08") or state in {"57P01", "57P02", "57P03"}):
            return True
        # Some existing adapter error paths retain only the typed text.
        if any(marker in str(value.get("error", "")) for marker in
               ("OperationalError:", "InterfaceError:", "AdminShutdown:", "CrashShutdown:", "ConnectionFailure:", "Connection refused")):
            return True
        return any(infrastructure_failure(item) for item in value.values())
    return isinstance(value, list) and any(infrastructure_failure(item) for item in value)


class InfrastructureFailure(RuntimeError):
    pass


class ModelBudgetExceeded(RuntimeError):
    pass


def retryable_model_error(exc):
    """Recognize transient transport failures without retrying semantic errors."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return (isinstance(exc, (TimeoutError, ConnectionError))
            or any(marker in text for marker in (
                "http error 504", "gateway timeout", "timed out", "timeout",
                "temporarily unavailable", "service unavailable", "http error 502",
                "http error 503", "connection reset", "connection aborted")))


class BudgetedModel:
    """Count actual attempted calls before dispatch, including failed requests.

    All three wrappers for one input share the same ledger. Transient transport
    failures are retried within the same logical call and remain diagnostic;
    the final verified fallback is still a valid measured arm.
    """
    def __init__(self, model, *, arm, ledger, query_id, on_update=None, retries=2):
        if arm not in ARM_LIMITS:
            raise ValueError("Unknown budgeted arm")
        self.model, self.arm, self.ledger = model, arm, ledger
        # ``None`` means the standard two retry policy.  Older configs wrote a
        # JSON null here, which accidentally disabled retries through int(None).
        self.query_id, self.on_update = query_id, on_update
        self.retries = 2 if retries is None else max(0, int(retries))
        ledger.setdefault("limit", 8)
        ledger.setdefault("arm_limits", dict(ARM_LIMITS))
        ledger.setdefault("calls", [])
        ledger.setdefault("blocked_calls", [])

    def __call__(self, payload):
        calls = self.ledger["calls"]
        logical_calls = [call for call in calls if call.get("retry") is not True]
        arm_count = sum(call["arm"] == self.arm for call in logical_calls)
        if len(logical_calls) >= 8 or arm_count >= ARM_LIMITS[self.arm]:
            blocked = {"arm": self.arm, "task": payload.get("task"), "at": utc_now(),
                       "reason": "shared or per-arm hard budget exhausted", "transport_invoked": False}
            self.ledger["blocked_calls"].append(blocked)
            self._update()
            raise ModelBudgetExceeded(blocked["reason"])
        bound = {**payload, "stage_a_context": {"query_id": self.query_id, "arm": self.arm,
                 "input_call_index": len(calls) + 1, "arm_call_index": arm_count + 1,
                 "shared_call_limit": 8, "arm_call_limit": ARM_LIMITS[self.arm]}}
        for attempt in range(self.retries + 1):
            call = {"input_call_index": len(calls) + 1, "arm_call_index": arm_count + 1,
                    "arm": self.arm, "task": payload.get("task"), "started_at": utc_now(),
                    "payload_sha256": digest(bound), "status": "running", "retry": attempt > 0,
                    "retry_attempt": attempt, "transport_invoked": True}
            calls.append(call)
            self._update()
            started = time.monotonic()
            try:
                response = self.model(bound)
                if not isinstance(response, dict):
                    raise TypeError("Model returned no JSON object")
                call.update(status="ok", provenance=public(response.get("provenance", {})))
                return response
            except Exception as exc:
                call.update(status="error", **error_record(exc))
                if attempt >= self.retries or not retryable_model_error(exc):
                    raise
                call["retryable"] = True
                call["retry_scheduled"] = True
                self._update()
                time.sleep(min(2 ** attempt, 8))
            finally:
                call.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
                self._update()

    def _update(self):
        self.ledger["total_invocations"] = len(self.ledger["calls"])
        self.ledger["invocations_by_arm"] = {arm: sum(call["arm"] == arm for call in self.ledger["calls"])
                                               for arm in ARM_LIMITS}
        logical = [call for call in self.ledger["calls"] if call.get("retry") is not True]
        self.ledger["logical_invocations"] = len(logical)
        self.ledger["logical_invocations_by_arm"] = {
            arm: sum(call["arm"] == arm for call in logical) for arm in ARM_LIMITS}
        if self.on_update:
            self.on_update()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", args.session_id):
        parser.error("Session ID must contain 1-96 letters, digits, underscores or hyphens")
    return args


def prepare_inputs(args):
    config = deepcopy(json.loads(args.config.read_text()))
    databases = config.get("databases", {})
    if set(databases) != set(DATASETS.values()):
        raise ValueError("Stage A accepts exactly tpch10, dsb10 and calcite10; no SF50 or Zipf")
    for benchmark, name in DATASETS.items():
        db = databases[name]
        if db.get("benchmark", benchmark) != benchmark or not isinstance(db.get("dsn"), str):
            raise ValueError("Dataset benchmark and DSN must be explicitly bound")
        if db.get("dbname", name) != name or db.get("distribution", "uniform") != "uniform":
            raise ValueError("Only the three primary SF10/uniform dataset bindings are accepted")
        if benchmark != "calcite" and db.get("scale", 10) != 10:
            raise ValueError("TPC-H and DSB must use scale factor 10")
        # Prevent a mislabeled configuration from targeting an excluded database.
        if re.search(r"(?:dbname\s*=\s*['\"]?)(?:tpch50|dsb50|calcite10zipf)(?:['\"\s]|$)", db["dsn"], re.I):
            raise ValueError("Excluded database in primary DSN")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 30 or Counter(row.get("benchmark") for row in rows) != Counter({name: 10 for name in DATASETS}):
        raise ValueError("Stage A requires exactly 30 queries, ten per benchmark")
    if len({row["id"] for row in rows}) != 30:
        raise ValueError("Stage A manifest IDs must be unique")
    for row in rows:
        if not isinstance(row.get("sql"), str) or not row["sql"].strip() or not row.get("schema_path"):
            raise ValueError("Every manifest input needs SQL and schema_path")
        actual = digest(row["sql"].encode())
        if row.get("sha256") and row["sha256"] != actual:
            raise ValueError(f"Manifest SQL hash mismatch: {row['id']}")
        path = Path(row["schema_path"])
        path = path if path.is_absolute() else ROOT / path
        schema = path.read_text()
        if row.get("schema_sha256") and digest(schema.encode()) != row["schema_sha256"]:
            raise ValueError(f"Manifest schema hash mismatch: {row['id']}")
    if config.get("measurement_runs", 3) != 3 or config.get("warmups", 1) != 1:
        raise ValueError("Stage A requires exactly one warmup and three final runs")
    if config.get("comparison_profile", "positional_numeric") != "positional_numeric":
        raise ValueError("Stage A requires positional_numeric comparison")
    if config.get("float_abs_tol", 1e-5) != 1e-5 or config.get("float_rel_tol", 1e-5) != 1e-5:
        raise ValueError("Stage A floating tolerances must both equal 1e-5")
    if config.get("max_model_calls", 8) != 8 or config.get("beam_width", 3) != 3 or config.get("max_depth", 2) != 2:
        raise ValueError("Stage A requires model budget eight, beam width three and depth two")
    for name, database in databases.items():
        if verification_database(config, name, database) is None:
            raise ValueError("Stage A requires a separate development verification database per benchmark")
    config.update(comparison_profile="positional_numeric", float_abs_tol=1e-5, float_rel_tol=1e-5,
                  measurement_runs=3, warmups=1)
    return config, rows


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def measure(db, sql):
    try:
        result = dict(db.measure(sql, runs=3, warmups=1))
        samples = result.get("samples_ms", [])
        if result.get("status") == "ok" and (len(samples) != 3 or not all(positive(item) for item in samples)):
            result.update(status="error", error="Expected three finite positive final execution samples")
        if result.get("status") == "ok":
            result.update(mean_ms=statistics.mean(samples), median_ms=statistics.median(samples))
        else:
            result.update(mean_ms=None, median_ms=None)
    except Exception as exc:
        result = {"status": "error", "samples_ms": [], "mean_ms": None, "median_ms": None, **error_record(exc)}
    result.update(runs=3, warmups=1, scope="final_evaluation")
    return result


def verify(gate, original, candidate):
    try:
        result = gate.verify(original, candidate)
        if not isinstance(result, dict):
            raise TypeError("Verifier returned no evidence object")
        return result
    except Exception as exc:
        return {"accepted": False, "formal": False, "proof": "none", **error_record(exc)}


def accepted(result, mode):
    return result.get("accepted") is True and (mode != "strict" or result.get("formal") is True)


def default_components():
    from mcts_rewrite.adapters import HTTPModel
    from mcts_rewrite.batched_operators import BatchedOperators
    from mcts_rewrite.db import Postgres
    from mcts_rewrite.fast_search import FastSearchConfig, FastSearchController
    from mcts_rewrite.verify import EquivalenceGate
    from mcts_rewrite.verification_db import ScopedEquivalenceGate
    from types import SimpleNamespace
    return SimpleNamespace(HTTPModel=HTTPModel, Operators=BatchedOperators, Postgres=Postgres,
                           SearchConfig=FastSearchConfig, Search=FastSearchController,
                           Gate=EquivalenceGate, ScopedGate=ScopedEquivalenceGate)


def verification_database(config, dataset, database):
    if "verification_database" in database:
        return database["verification_database"]
    configured = config.get("verification_databases", {})
    return configured.get(dataset, configured.get(database.get("benchmark", dataset)))


def make_gate(config, row, db, components):
    suffix = os.environ.get("ICDE_DATABASE_SUFFIX", "10").strip().lower()
    if suffix == "zipf" and row["benchmark"] == "calcite":
        dataset = "calcite10zipf"
    elif suffix in {"10", "50"} and row["benchmark"] in {"tpch", "dsb"}:
        dataset = row["benchmark"] + suffix
    else:
        dataset = DATASETS[row["benchmark"]]
    database = config["databases"][dataset]
    path = Path(row["schema_path"])
    schema = (path if path.is_absolute() else ROOT / path).read_text()
    miniature = database.get("mini_dbs")
    if miniature is None:
        configured = config.get("mini_dbs", {})
        miniature = configured.get(dataset, configured.get(row["benchmark"])) if isinstance(configured, dict) else configured
    options = dict(mode=config.get("verification_mode", "hybrid"), mini_dbs=miniature,
                   solver_timeout=config.get("solver_timeout", 1))
    selected = verification_database(config, dataset, database)
    if selected is None:
        raise ValueError("Stage A requires a separate development verification database")
    verify_db = components.Postgres(selected["dsn"], timeout_ms=config.get("verification_timeout_ms", 30000),
                       search_path=selected.get("search_path", "public"), **comparison_options(config))
    try:
        gate = components.ScopedGate(db, verify_db, schema, **options,
                    performance_provenance=public(database), verification_provenance=public(selected))
        return gate, verify_db, schema, miniature
    except BaseException:
        verify_db.close()
        raise


def comparison_options(config):
    return {key: config[key] for key in ("comparison_mode", "comparison_profile", "float_abs_tol", "float_rel_tol", "max_rows", "max_result_bytes") if key in config}


def choose_rbot(ops, db, gate, original, schema, plan, mode):
    proposals = ops.rbot(original, schema, plan=plan, limit=3)
    trace, available = [], []
    root_cost = plan.get("Plan", {}).get("Total Cost")
    root_cost = root_cost if positive(root_cost) else math.inf
    for proposal in proposals:
        sql = proposal.get("sql")
        if not isinstance(sql, str) or not sql.strip() or sql == original:
            continue
        evidence = verify(gate, original, sql)
        entry = {"candidate": proposal, "verification": evidence}
        if infrastructure_failure(evidence):
            raise InfrastructureFailure("Database connection failed during R-Bot root verification")
        if accepted(evidence, mode):
            try:
                estimated = db.explain(sql, analyze=False)
                cost = estimated["Plan"]["Total Cost"]
                if not positive(cost):
                    raise ValueError("Invalid PostgreSQL estimated cost")
                entry.update(status="eligible", estimated_cost=cost, plan=estimated)
                if cost < root_cost:
                    available.append((cost, len(trace), proposal, evidence))
                else:
                    entry.update(status="root_competitor_preferred", original_estimated_cost=root_cost)
            except Exception as exc:
                entry.update(status="plan_error", **error_record(exc))
        else:
            entry["status"] = "rejected_verification"
        trace.append(entry)
        if infrastructure_failure(entry):
            raise InfrastructureFailure("Database connection failed during R-Bot EXPLAIN")
    if not available:
        return {"sql": original, "fallback": True, "trace": trace,
                "selection_method": "minimum_postgres_explain_cost_after_root_verification",
                "original_estimated_cost": root_cost if math.isfinite(root_cost) else None,
                "reason": "No verified R-Bot candidate with lower EXPLAIN cost than Original"}
    _, _, proposal, evidence = min(available, key=lambda item: (item[0], item[1]))
    return {"sql": proposal["sql"], "verification": evidence, "fallback": False, "trace": trace,
            "selected_candidate": proposal,
            "selection_method": "minimum_postgres_explain_cost_after_root_verification"}


def run_query(row, config, model_factory, *, components=None, checkpoint=None):
    components = components or default_components()
    started = time.monotonic()
    dataset = DATASETS[row["benchmark"]]
    original = row["sql"]
    record = {"record_type": "stage_a_query_result", "query_id": row["id"], "benchmark": row["benchmark"],
              "dataset": dataset, "original_sql": original, "query_sha256": digest(original.encode()),
              "manifest_provenance": public({key: value for key, value in row.items() if key != "sql"}),
              "dataset_provenance": public(config["databases"][dataset]), "started_at": utc_now(),
              "status": "running", "arms": {},
              "model_call_audit": {"limit": 8, "arm_limits": dict(ARM_LIMITS), "calls": [], "blocked_calls": [],
                                   "total_invocations": 0, "invocations_by_arm": {arm: 0 for arm in ARM_LIMITS}}}
    db = verification_db = None
    mode = config.get("verification_mode", "hybrid")

    def save():
        if checkpoint:
            checkpoint(record)

    def check_infrastructure():
        if infrastructure_failure(record):
            raise InfrastructureFailure("Database infrastructure failed; stop before further query execution")

    try:
        database = config["databases"][dataset]
        db = components.Postgres(database["dsn"], timeout_ms=config.get("timeout_ms", 300000),
                  search_path=database.get("search_path", "public"), **comparison_options(config))
        gate, verification_db, schema, miniatures = make_gate(config, row, db, components)
        record.update(schema_sha256=digest(schema.encode()), verification_context=public(gate.metadata),
                      miniature_configuration=public(miniatures), verification_mode=mode)
        original_verification = verify(gate, original, original)
        record["arms"]["original"] = {"sql": original, "verification": original_verification, "status": "running"}
        check_infrastructure()
        if not accepted(original_verification, mode):
            record["arms"]["original"]["status"] = "rejected_verification"
            raise ValueError("Original query did not pass identity verification")
        original_measurement = measure(db, original)
        record["arms"]["original"].update(measurement=original_measurement, status=original_measurement["status"])
        save()
        check_infrastructure()
        original_timed_out = original_measurement["status"] == "timeout"
        if original_measurement["status"] != "ok" and not original_timed_out:
            record["status"] = "original_" + original_measurement["status"]
            return record
        reference_ms = original_measurement["median_ms"] if not original_timed_out else float(config.get("timeout_ms", 300000))
        record["original_reference"] = {"ms": reference_ms,
            "kind": "censored_timeout_guard" if original_timed_out else "measured_final_median",
            "is_measured_latency": not original_timed_out,
            "timeout_lower_bound_ms": reference_ms if original_timed_out else None,
            "scope": "Ranking and bounded canary guard only; censored references cannot support latency ratios" if original_timed_out else "Original final evaluation"}
        original_plan = original_measurement.get("plan") or db.explain(original, analyze=False)
        # Identical final SQL share exactly the same final measurement, explicitly.
        final_measurements = {original: ("original", original_measurement)}
        for arm in ARM_LIMITS:
            arm_started = time.monotonic()
            selected = {"sql": original, "fallback": True}
            arm_record = {"status": "running", "label": "batched_standalone_" + arm if arm != "mcts" else "batched_hybrid_mcts",
                          "fresh_generation": True, "cross_arm_candidate_reuse": False}
            record["arms"][arm] = arm_record
            save()
            try:
                budget_model = BudgetedModel(model_factory(arm, row), arm=arm, ledger=record["model_call_audit"],
                                             query_id=row["id"], on_update=save,
                                             retries=config.get("model_retries", 2))
                ops = components.Operators(ROOT, model=budget_model)
                ops.begin_query(original, schema)
                if arm == "rbot":
                    selected = choose_rbot(ops, db, gate, original, schema, original_plan, mode)
                else:
                    search_config = components.SearchConfig(beam_width=3, max_depth=2, candidate_limit_per_parent=3,
                                                            max_canaries=2 if arm == "mcts" else 0,
                                                            use_rag=config.get("use_rag", True))
                    selected = components.Search(db, gate, ops, config=search_config, operator=arm).run(
                                    original, schema, original_ms=reference_ms)
                    selected["original_reference"] = dict(record["original_reference"])
                arm_record.update(selection=selected, operator_metadata=ops.metadata(),
                                  generator_diagnostics=list(ops.last_diagnostics),
                                  generation_errors=list(ops.model_errors))
            except InfrastructureFailure:
                raise
            except Exception as exc:
                arm_record.update(generation_error=error_record(exc))
            candidate = selected.get("sql", original)
            if not isinstance(candidate, str) or not candidate.strip():
                candidate = original
            # Search's cached evidence cannot silently substitute for final verification.
            evidence = original_verification if candidate == original else verify(gate, original, candidate)
            if not accepted(evidence, mode):
                arm_record["rejected_selection"] = {"sql": candidate, "verification": evidence}
                candidate, evidence = original, original_verification
            check_infrastructure()
            if candidate in final_measurements:
                source, measurement = final_measurements[candidate]
                measurement = {**measurement, "shared_identical_sql_with": source,
                               "shared_identical_sql_sha256": digest(candidate.encode())}
            else:
                measurement = measure(db, candidate)
                final_measurements[candidate] = (arm, measurement)
            had_errors = bool(arm_record.get("generation_error") or arm_record.get("generation_errors"))
            # Model transport errors remain diagnostic metadata once a final
            # measurement exists. Failed or rejected rewrites have already
            # fallen back to Original, whose timing is the safety-net result.
            measured_ok = measurement.get("status") in {"ok", "timeout"}
            arm_record.update(sql=candidate, sql_sha256=digest(candidate.encode()), verification=evidence,
                              fallback=candidate == original, measurement=measurement,
                              status=measurement["status"] if measurement["status"] != "ok" else "ok",
                              # A valid measured rewrite remains an eligible
                              # result even when a sub-call timed out. The
                              # transport error is preserved in diagnostics.
                              eligible_win=measurement["status"] == "ok" and candidate != original,
                              measurement_valid=measured_ok,
                              wall_seconds=time.monotonic() - arm_started)
            save()
            check_infrastructure()
        # Query status follows final measurement validity. Model transport
        # diagnostics never turn a completed four-arm comparison into a
        # partial record; a genuine SQL timeout remains a censored result.
        valid_arms = all(record["arms"].get(arm, {}).get("measurement", {}).get("status") in {"ok", "timeout"}
                         for arm in ARMS)
        record["status"] = "original_timeout" if original_timed_out else "ok" if valid_arms else "partial_error"
    except InfrastructureFailure as exc:
        record.update(status="infrastructure_error", **error_record(exc))
    except Exception as exc:
        record.update(status="error", **error_record(exc), traceback=traceback.format_exc())
        if getattr(exc, "evidence", None):
            record["verification_database_error_evidence"] = public(exc.evidence)
        if infrastructure_failure(record):
            record["status"] = "infrastructure_error"
    finally:
        for connection in (verification_db, db):
            if connection is not None:
                try:
                    connection.close()
                except Exception as exc:
                    record.setdefault("close_errors", []).append(error_record(exc))
        for arm in ARMS:
            record["arms"].setdefault(arm, {"status": "skipped", "reason": "Original or query setup failed",
                                          "eligible_win": False, "measurement": {"status": "skipped", "mean_ms": None, "median_ms": None}})
        if infrastructure_failure(record):
            record["status"] = "infrastructure_error"
        record.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        save()
    return record


def percentile90(values):
    """Linear interpolation at (n-1)*0.9; matches NumPy's default percentile."""
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * 0.9
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def arm_measurement(record, arm, field):
    measurement = record.get("arms", {}).get(arm, {}).get("measurement", {})
    value = measurement.get(field)
    return value if measurement.get("status") == "ok" and positive(value) else None


def summarize(records, *, expected=30):
    complete = [row for row in records if all(arm_measurement(row, arm, "mean_ms") is not None for arm in ARMS)]
    systems = {}
    for arm in ARMS:
        means = [value for row in records if (value := arm_measurement(row, arm, "mean_ms")) is not None]
        medians = [value for row in records if (value := arm_measurement(row, arm, "median_ms")) is not None]
        systems[arm] = {"measured_queries": len(means), "arithmetic_mean_ms": statistics.mean(means) if means else None,
                        "p90_query_median_ms": percentile90(medians), "p90_query_mean_ms": percentile90(means),
                        "status_counts": dict(Counter(row.get("arms", {}).get(arm, {}).get("status", "missing") for row in records)),
                        "fallbacks": sum(row.get("arms", {}).get(arm, {}).get("fallback") is True for row in records)}
    comparisons = {}
    regressions = []
    for row in complete:
        if arm_measurement(row, "mcts", "median_ms") > arm_measurement(row, "original", "median_ms"):
            regressions.append(row["query_id"])
    for baseline in ("rbot", "quite"):
        paired = [row for row in complete]
        mcts_means = [arm_measurement(row, "mcts", "mean_ms") for row in paired]
        base_means = [arm_measurement(row, baseline, "mean_ms") for row in paired]
        mcts_medians = [arm_measurement(row, "mcts", "median_ms") for row in paired]
        base_medians = [arm_measurement(row, baseline, "median_ms") for row in paired]
        mcts_mean = statistics.mean(mcts_means) if paired else None
        base_mean = statistics.mean(base_means) if paired else None
        mcts_p90, base_p90 = percentile90(mcts_medians), percentile90(base_medians)
        mean_reduction = 1 - mcts_mean / base_mean if paired else None
        p90_reduction = 1 - mcts_p90 / base_p90 if paired else None
        wins = [row["query_id"] for row in paired if row["arms"]["mcts"].get("eligible_win") is True and
                row["arms"][baseline].get("status") == "ok" and
                arm_measurement(row, "mcts", "median_ms") < arm_measurement(row, baseline, "median_ms")]
        comparisons[baseline] = {"paired_queries": len(paired), "mcts_mean_ms": mcts_mean, "baseline_mean_ms": base_mean,
                                 "mean_latency_reduction": mean_reduction,
                                 "mcts_p90_query_median_ms": mcts_p90, "baseline_p90_query_median_ms": base_p90,
                                 "p90_latency_reduction": p90_reduction, "eligible_mcts_win_count": len(wins),
                                 "eligible_mcts_win_ids": wins,
                                 "target_met": mean_reduction is not None and mean_reduction >= 0.15 and p90_reduction >= 0.15}
    errors = [row["query_id"] for row in records if row.get("status") != "ok"]
    reasons = []
    if len(records) != expected or len(complete) != expected:
        reasons.append(f"Need {expected} complete measured records; observed {len(records)} records and {len(complete)} measured records")
    if errors:
        reasons.append(f"{len(errors)} query records contain errors or incomplete evaluation")
    if regressions:
        reasons.append(f"{len(regressions)} MCTS final medians regress versus Original")
    for baseline, comparison in comparisons.items():
        if not comparison["target_met"]:
            reasons.append(f"Mean and p90 latency reductions must each reach 15% versus {baseline}")
    return {"queries": len(records), "expected_queries": expected, "complete_measured_queries": len(complete),
            "systems": systems, "mcts_vs_baselines": comparisons,
            "mcts_regression_count_vs_original": len(regressions), "mcts_regression_ids_vs_original": regressions,
            "error_query_ids": errors,
            "total_llm_invocations": sum(row.get("model_call_audit", {}).get("total_invocations", 0) for row in records),
            "maximum_llm_invocations_per_input": max((row.get("model_call_audit", {}).get("total_invocations", 0) for row in records), default=0),
            "stage_b_gate": {"passed": not reasons, "reasons": reasons,
                             "required_mean_latency_reduction": 0.15, "required_p90_latency_reduction": 0.15,
                             "required_regressions_vs_original": 0},
            "metric_definitions": {"mean": "Arithmetic mean of per-query means of the three final samples",
                "p90": "Linearly interpolated 90th percentile of per-query medians of the three final samples",
                "regression": "MCTS final per-query median strictly greater than Original final per-query median",
                "win": "Strictly lower MCTS median with valid final measurements, error-free generation and verified changed SQL",
                "cohort": "Selected fixed cohort; not a random sample and not a full-suite estimate"}}


def write_reports(output, records):
    output = Path(output)
    summary = summarize(records)
    summary["per_benchmark"] = {benchmark: summarize([row for row in records if row["benchmark"] == benchmark], expected=10)
                                 for benchmark in DATASETS}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "stage_b_gate.json", summary["stage_b_gate"])
    fields = ["query_id", "benchmark", "dataset", "status", "llm_invocations"]
    for arm in ARMS:
        fields.extend([arm + "_status", arm + "_mean_ms", arm + "_median_ms", arm + "_fallback"])
    fields.extend(["mcts_regresses_vs_original", "mcts_vs_rbot_latency_reduction", "mcts_vs_quite_latency_reduction"])
    with (output / "per_query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            result = {key: record.get(key) for key in ("query_id", "benchmark", "dataset", "status")}
            result["llm_invocations"] = record.get("model_call_audit", {}).get("total_invocations", 0)
            for arm in ARMS:
                detail = record.get("arms", {}).get(arm, {})
                result.update({arm + "_status": detail.get("status"), arm + "_fallback": detail.get("fallback", False),
                               arm + "_mean_ms": arm_measurement(record, arm, "mean_ms"),
                               arm + "_median_ms": arm_measurement(record, arm, "median_ms")})
            original, mcts = result["original_median_ms"], result["mcts_median_ms"]
            result["mcts_regresses_vs_original"] = mcts > original if mcts is not None and original is not None else None
            for baseline in ("rbot", "quite"):
                base = result[baseline + "_median_ms"]
                result["mcts_vs_" + baseline + "_latency_reduction"] = 1 - mcts / base if mcts is not None and base is not None else None
            writer.writerow(result)
    for benchmark, report in summary["per_benchmark"].items():
        atomic_json(output / (benchmark + "_summary.json"), report)
    return summary


@contextmanager
def session_bridge(session_id, *, timeout=600):
    # Import as a sibling, including when tests load this file by absolute path.
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from session_model_bridge import Bridge, make_server, atomic_json as bridge_json
    secret = secrets.token_urlsafe(48)
    env_name = "MCTS_STAGE_A_BRIDGE_SECRET"
    previous = os.environ.get(env_name)
    bridge = Bridge(ROOT / ".runtime/session_model", secret, session_id=session_id, timeout=timeout)
    server = make_server(bridge, port=0)
    address = f"http://127.0.0.1:{server.server_address[1]}"
    os.environ[env_name] = secret
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="stage-a-session-bridge")
    thread.start()
    try:
        yield bridge, address, env_name
    finally:
        bridge.closed.set()
        bridge_json(bridge.directory / "STOP", {"stopped_at": utc_now(), "reason": "Stage A runner stopped"})
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def run(args):
    config, rows = prepare_inputs(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; Stage A records are immutable")
    if (ROOT / ".runtime/session_model" / args.session_id).exists():
        raise FileExistsError("Choose a fresh session ID; model request spools are immutable")
    output.mkdir(parents=True)
    # Reuse the consolidated dual-action pool when present. The cache contains
    # proposals only; every candidate still passes the normal equivalence,
    # EXPLAIN, critic and canary gates before selection.
    candidate_cache = config.get("candidate_cache") or os.environ.get("MCTS_CANDIDATE_CACHE")
    if candidate_cache:
        cache_path = Path(candidate_cache)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        if cache_path.is_file():
            os.environ["MCTS_CANDIDATE_CACHE"] = str(cache_path.resolve())
            context_cache = str(cache_path.resolve())
        else:
            context_cache = None
    else:
        default_cache = ROOT / "artifacts/model_candidates_stage_a_consolidated.jsonl"
        if default_cache.is_file():
            os.environ["MCTS_CANDIDATE_CACHE"] = str(default_cache.resolve())
            context_cache = str(default_cache.resolve())
        else:
            context_cache = None
    (output / "queries").mkdir()
    (output / "inputs").mkdir()
    (output / "inputs/manifest.jsonl").write_bytes(args.manifest.read_bytes())
    atomic_json(output / "inputs/config.public.json", public(config))
    source_paths = [Path(__file__), *sorted((ROOT / "mcts_rewrite").glob("*.py")),
                    ROOT / "scripts/session_model_bridge.py", ROOT / "upstream.lock.json", ROOT / "requirements.lock"]
    source_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in source_paths}
    for path in source_paths:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    schema_inputs = output / "inputs/schemas"
    schema_inputs.mkdir()
    for row in rows:
        path = Path(row["schema_path"])
        path = path if path.is_absolute() else ROOT / path
        body = path.read_bytes()
        (schema_inputs / (digest(body) + ".sql")).write_bytes(body)
    context = {"started_at": utc_now(), "status": "starting", "expected_queries": 30,
               "completed_queries": 0, "session_id": args.session_id,
               "source_sha256": source_hashes, "manifest_sha256": digest(args.manifest.read_bytes()),
               "configuration_sha256": digest(config), "secret_persisted": False,
               "cohort": "Stage A fixed nonrandom cohort: 10 queries per primary benchmark",
               "arms": list(ARMS), "model_limits": {"total_per_input": 8, "per_arm": dict(ARM_LIMITS)},
               "actual_serving_model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
               "published_model_configuration_reproduced": False,
               "sampling_controls_reproduced": False, "usage_known": False,
               "candidate_generation": "Fresh independent invocation per arm and query; batched algorithm variants",
               "measurement": "One warmup and three final runs; identical SQL explicitly shares final samples",
               "scope": "Primary tpch10, dsb10, calcite10 only; no SF50/Zipf work"}
    context["candidate_cache"] = context_cache
    atomic_json(output / "run.json", context)
    records = []
    write_reports(output, records)
    try:
        components = default_components()
        with session_bridge(args.session_id, timeout=config.get("model_timeout_seconds", 600)) as (bridge, address, env_name):
            context.update(status="running", spool=str(bridge.directory), bridge_address=address)
            atomic_json(output / "run.json", context)
            print(json.dumps({"spool": str(bridge.directory), "output": str(output), "queries": 30,
                              "supervised_session_workers_required": True}), flush=True)

            def model_factory(arm, row):
                run_id = "stage_a_" + digest([row["id"], row["sql"], arm])[:24] + "_" + arm
                return components.HTTPModel(address + "/runs/" + run_id + "/v1/chat/completions", "gpt-6.1-sol",
                            timeout=config.get("model_timeout_seconds", 600) + 10, api_key_env=env_name)

            for index, row in enumerate(rows):
                context.update(current_query_id=row["id"], current_query_index=index + 1)
                atomic_json(output / "run.json", context)
                query_path = output / "queries" / (f"{index + 1:02d}_" + digest(row["id"])[:16] + ".json")
                from mcts_rewrite.host_conditions import TaskHostMonitor
                monitor = TaskHostMonitor(config, row["id"], args.session_id,
                                          output / "host_conditions.jsonl", config["databases"][DATASETS[row["benchmark"]]])
                with monitor:
                    record = run_query(row, config, model_factory, components=components,
                                       checkpoint=lambda value: atomic_json(query_path, value))
                record["host_conditions"] = monitor.summary()
                record["run_id"] = args.session_id
                atomic_json(query_path, record)
                with (output / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(record)
                write_reports(output, records)
                context.update(completed_queries=len(records), last_query_status=record["status"])
                atomic_json(output / "run.json", context)
                print(json.dumps({"completed": len(records), "query_id": row["id"], "status": record["status"],
                                  "llm_invocations": record["model_call_audit"]["total_invocations"]}), flush=True)
                if record["status"] == "infrastructure_error":
                    raise InfrastructureFailure("Database connection infrastructure failed; remaining inputs were not executed")
        context.update(status="completed", finished_at=utc_now(), completed_queries=len(records))
        atomic_json(output / "run.json", context)
        return 0
    except BaseException as exc:
        context.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", finished_at=utc_now(),
                       completed_queries=len(records), **error_record(exc))
        atomic_json(output / "run.json", context)
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(json.dumps({"status": "failed", **error_record(exc)}), file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
