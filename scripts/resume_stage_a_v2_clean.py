#!/usr/bin/env python3
"""Resume the interrupted clean Stage A v2 ledger without replaying completed work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage_a_v2 as launcher
from mcts_rewrite import study_v2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--spool-root", type=Path, default=ROOT / ".runtime/session_model")
    parser.add_argument("--heavy-lock", type=Path, required=True)
    parser.add_argument("--heavy-lock-fd", type=int)
    return parser.parse_args()


def load_boundary(args):
    output = args.output.resolve(strict=True)
    config, rows = launcher.base.prepare_inputs(args)
    binding = json.loads(args.fixture_bindings.read_text())
    if binding.get("status") != "postgresql_validated":
        raise ValueError("Continuation requires the PostgreSQL-validated v2 fixture binding")
    required = {rows[index - 1]["id"] for index in launcher.WITNESS_QUERY_INDICES}
    if not isinstance(binding.get("queries"), dict) or set(binding["queries"]) != required:
        raise ValueError("Unexpected v2 witness binding coverage")
    for row in rows:
        study_v2.configuration_for_query(config, row, binding)

    context = json.loads((output / "run.json").read_text())
    records = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines() if line.strip()]
    if not 0 < len(records) < len(rows):
        raise ValueError("Continuation requires a nonempty incomplete clean ledger")
    if [record["query_id"] for record in records] != [row["id"] for row in rows[:len(records)]]:
        raise ValueError("Completed clean ledger is not a manifest prefix")
    if context.get("completed_queries") != len(records):
        raise ValueError("run.json and raw.jsonl completion boundaries differ")
    if context.get("manifest_sha256") != launcher.base.digest(args.manifest.read_bytes()):
        raise ValueError("Manifest changed after the interrupted run")
    if context.get("configuration_sha256") != launcher.base.digest(args.config.read_bytes()):
        raise ValueError("Configuration changed after the interrupted run")
    if context.get("fixture_binding_sha256") != launcher.base.digest(args.fixture_bindings.read_bytes()):
        raise ValueError("Fixture binding changed after the interrupted run")
    spool = args.spool_root.resolve() / args.session_id
    if spool.exists():
        raise FileExistsError("Continuation requires a fresh v2 session spool")

    next_index = len(records)
    partial_path = output / "queries" / (f"{next_index + 1:02d}_" + launcher.base.digest(rows[next_index]["id"])[:16] + ".json")
    partial = json.loads(partial_path.read_text())
    if partial.get("status") != "running" or partial.get("arms", {}).get("original", {}).get("status") != "ok":
        raise ValueError("Expected a running partial query with a completed Original arm")
    calls = partial.get("model_call_audit", {}).get("calls", [])
    if len(calls) != 1 or calls[0].get("arm") != "rbot" or calls[0].get("status") != "running":
        raise ValueError("Expected exactly one interrupted R-Bot dispatch at the continuation boundary")

    recovery = output / "recovery"
    recovery.mkdir(exist_ok=True)
    snapshot = recovery / ("interrupted_" + partial_path.name)
    if not snapshot.exists():
        shutil.copy2(partial_path, snapshot)
    return output, config, rows, binding, context, records, next_index, partial, snapshot


def run(args):
    output, config, rows, binding, context, records, next_index, partial, snapshot = load_boundary(args)
    recovered_measurement = deepcopy(partial["arms"]["original"]["measurement"])
    recovered_call = deepcopy(partial["model_call_audit"]["calls"][0])
    recovered_call.update(
        status="error",
        finished_at=launcher.base.utc_now(),
        error_type="ControllerInterrupted",
        error="Claimed request had no response when the controller was externally terminated",
        recovery={
            "kind": "consumed_dispatch_preserved_without_replay",
            "source_checkpoint": str(snapshot),
            "new_model_calls": 0,
            "transport_invoked": True,
        },
    )

    with launcher.termination_signals(), launcher.heavy_lock(args.heavy_lock, args.heavy_lock_fd) as lock_evidence:
        history = list(context.get("continuations", []))
        history.append({
            "resumed_at": launcher.base.utc_now(),
            "session_id": args.session_id,
            "preserved_completed_queries": len(records),
            "resumed_query_index": next_index + 1,
            "reused_original_measurement": True,
            "preserved_consumed_dispatches_for_resumed_query": 1,
            "source_checkpoint": str(snapshot),
        })
        context.update(
            status="starting_continuation",
            session_id=args.session_id,
            completed_queries=len(records),
            current_query_id=rows[next_index]["id"],
            current_query_index=next_index + 1,
            heavy_experiment_lock=lock_evidence,
            continuations=history,
            no_model_dispatch=False,
        )
        launcher.base.atomic_json(output / "run.json", context)
        gate = launcher.SupervisorHeartbeatGate(output, context)
        original_budget_model = launcher.base.BudgetedModel
        guarded = launcher.supervised_budget_model(original_budget_model, gate.await_supervisor)

        class RecoveredBudgetModel(guarded):
            def __init__(self, model, *, arm, ledger, query_id, on_update=None):
                if query_id == rows[next_index]["id"] and not ledger.get("calls"):
                    ledger.setdefault("calls", []).append(deepcopy(recovered_call))
                super().__init__(model, arm=arm, ledger=ledger, query_id=query_id, on_update=on_update)
                self._recovered_interruption_pending = (
                    query_id == rows[next_index]["id"] and arm == "rbot"
                )

            def __call__(self, payload):
                if self._recovered_interruption_pending:
                    self._recovered_interruption_pending = False
                    self._update()
                    raise RuntimeError(
                        "Interrupted R-Bot relevance dispatch is preserved; the same task is not retried"
                    )
                return super().__call__(payload)

        launcher.base.BudgetedModel = RecoveredBudgetModel
        original_measure = launcher.base.measure
        reused = False

        def continuation_measure(db, sql):
            nonlocal reused
            if not reused and sql == rows[next_index]["sql"]:
                reused = True
                measurement = deepcopy(recovered_measurement)
                measurement["continuation_reused"] = True
                measurement["continuation_source_checkpoint"] = str(snapshot)
                return measurement
            return original_measure(db, sql)

        def reports():
            report = launcher.reporting.build_report(output / "raw.jsonl", output / "inputs/manifest.jsonl")
            launcher.reporting.write_report(output / "report", report)

        try:
            components = study_v2.default_components()
            with launcher.session_bridge(args.session_id, spool_root=args.spool_root,
                                         timeout=config.get("model_timeout_seconds", 600)) as (bridge, address, env_name):
                context.update(status="running", spool=str(bridge.directory), bridge_address=address,
                               continuation_original_measurement_reused=True)
                launcher.base.atomic_json(output / "run.json", context)
                print(json.dumps({"status": "running_continuation", "spool": str(bridge.directory),
                                  "next_query_index": next_index + 1, "preserved_completed": len(records)}), flush=True)
                for index, row in enumerate(rows[next_index:], start=next_index):
                    failures = {arm: [] for arm in launcher.base.ARM_LIMITS}
                    context.update(current_query_id=row["id"], current_query_index=index + 1)
                    launcher.base.atomic_json(output / "run.json", context)
                    query_path = output / "queries" / (f"{index + 1:02d}_" + launcher.base.digest(row["id"])[:16] + ".json")

                    def model_factory(arm, current):
                        run_id = "stage_a_v2_" + launcher.base.digest(
                            [args.session_id, current["id"], current["sql"], arm])[:24] + "_" + arm
                        model = components.HTTPModel(address + "/runs/" + run_id + "/v1/chat/completions",
                            launcher.transport.SERVING_MODEL,
                            timeout=config.get("model_timeout_seconds", 600) + 10,
                            api_key_env=env_name)
                        return launcher.GenerationModel(model, failures[arm], gate.await_supervisor)

                    from mcts_rewrite.host_conditions import TaskHostMonitor
                    monitor = TaskHostMonitor(config, row["id"], args.session_id,
                        output / "host_conditions.jsonl", config["databases"][launcher.DATASETS[row["benchmark"]]])
                    if index == next_index:
                        launcher.base.measure = continuation_measure
                    try:
                        with monitor:
                            record = study_v2.run_query(row, config, model_factory, fixture_bindings=binding,
                                components=components,
                                checkpoint=lambda value: launcher.base.atomic_json(
                                    query_path, launcher.mark_terminal_failures(value, failures)))
                    finally:
                        launcher.base.measure = original_measure
                    launcher.mark_terminal_failures(record, failures)
                    record.update(host_conditions=monitor.summary(), run_id=args.session_id)
                    if index == next_index:
                        record["continuation_recovery"] = {
                            "original_measurement_reused": True,
                            "preserved_consumed_dispatches": 1,
                            "source_checkpoint": str(snapshot),
                            "fresh_dispatch_budget_after_recovery": 7,
                            "rbot_retried": False,
                            "cumulative_dispatch_cap": 8,
                        }
                    launcher.base.atomic_json(query_path, record)
                    with (output / "raw.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(record)
                    reports()
                    context.update(completed_queries=len(records), last_query_status=record["status"])
                    launcher.base.atomic_json(output / "run.json", context)
                    print(json.dumps({"completed": len(records), "query_id": row["id"],
                                      "status": record["status"],
                                      "llm_invocations": record["model_call_audit"]["total_invocations"]}), flush=True)
                    if record["status"] == "infrastructure_error":
                        raise launcher.base.InfrastructureFailure("Database infrastructure failed; stop before next input")
            context.update(status="completed", finished_at=launcher.base.utc_now(), completed_queries=len(records))
            launcher.base.atomic_json(output / "run.json", context)
            return 0
        except BaseException as exc:
            context.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                           finished_at=launcher.base.utc_now(), completed_queries=len(records),
                           **launcher.base.error_record(exc))
            launcher.base.atomic_json(output / "run.json", context)
            reports()
            print(json.dumps({"status": context["status"], **launcher.base.error_record(exc)}),
                  file=sys.stderr, flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            launcher.base.BudgetedModel = original_budget_model
            launcher.base.measure = original_measure


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
