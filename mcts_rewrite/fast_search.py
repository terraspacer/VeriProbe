"""Shallow, batched SQL search with a bounded execution canary phase.

Search uses PostgreSQL EXPLAIN without ANALYZE and verification against the
original query.  Only the final two ranked candidates may execute on the target
database, once each.  Final benchmark timing belongs to the runner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Any

from .evidence import EvidenceStore
from .plans import bottleneck, operators, plan_features, plan_root
from .verify import conservative_identity


@dataclass
class FastSearchConfig:
    beam_width: int = 3
    max_depth: int = 2
    candidate_limit_per_parent: int = 3
    max_canaries: int = 2
    use_rag: bool = True

    def __post_init__(self) -> None:
        for name, maximum in (("beam_width", 3), ("max_depth", 2),
                              ("candidate_limit_per_parent", 16), ("max_canaries", 2)):
            value = getattr(self, name)
            minimum = 0 if name == "max_canaries" else 1
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


@dataclass
class _Node:
    id: int
    sql: str
    parent_id: int | None
    depth: int
    plan: dict
    verification: dict
    action: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _key(sql: str) -> str:
    try:
        body = repr(conservative_identity(sql))
    except Exception:
        body = sql.strip()
    return hashlib.sha256(body.encode()).hexdigest()


def _error(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__,
            "sqlstate": getattr(exc, "sqlstate", None)}


def _planner_cost(node: _Node) -> float:
    cost = _positive(plan_features(node.plan).get("total_cost"))
    return cost if cost is not None else math.inf


class FastSearchController:
    """Two generation batches, one joint critic batch, then at most two canaries.

    ``generator`` supplies ``generate_batch`` for MCTS, ``macro_batch`` for
    QUITE, and ``critic_batch`` for joint ranking.  A query-wide model budget is
    owned by the caller and shared by these methods and the other rewrite arms.
    ``gate`` should use the configured miniature verification databases, never
    the target benchmark database.
    """

    def __init__(self, db: Any, gate: Any, generator: Any,
                 config: FastSearchConfig | None = None, evidence: Any = None,
                 *, operator: str = "mcts", canary_budget: dict | None = None):
        if operator not in {"mcts", "quite"}:
            raise ValueError("operator must be mcts or quite")
        self.db, self.gate, self.generator = db, gate, generator
        self.config = config or FastSearchConfig()
        if operator == "mcts" and self.config.max_canaries == 0:
            raise ValueError("MCTS requires a bounded canary before selecting a rewrite")
        self.evidence, self.operator = evidence, operator
        self.canary_budget = canary_budget

    def _plan(self, sql: str) -> dict:
        # Planning can itself become the runaway operation on large benchmark
        # relations.  Keep the search controller bounded while preserving the
        # runner's timeout for final timing and canaries.
        sentinel = object()
        previous = getattr(self.db, "timeout_ms", sentinel)
        try:
            if previous is not sentinel:
                self.db.timeout_ms = min(int(previous), 15_000)
            result = self.db.explain(sql, analyze=False)
            if not isinstance(result, dict) or not result:
                raise ValueError("database returned no EXPLAIN plan")
            return result
        finally:
            if previous is sentinel:
                try:
                    delattr(self.db, "timeout_ms")
                except AttributeError:
                    pass
            else:
                self.db.timeout_ms = previous

    @staticmethod
    def _action_class(node: _Node) -> str | None:
        value = node.action.get("action_class") if isinstance(node.action, dict) else None
        return value if value in {"A", "B"} else None

    def _canary_candidates(self, candidates: list[_Node], promising: list[_Node],
                           root: _Node, maximum: int) -> list[_Node]:
        """Reserve MCTS canary slots for both verified action classes.

        The LLM critic is allowed to order candidates within a class, but its
        uncalibrated latency guesses cannot remove a lower-cost verified
        symbolic candidate from the canary set.
        """
        if maximum <= 0:
            return []
        root_cost = _planner_cost(root)
        promising_ids = {node.id for node in promising}
        class_candidates = {
            "A": [node for node in candidates if _planner_cost(node) < root_cost],
            "B": [node for node in candidates
                  if _planner_cost(node) < root_cost or node.id in promising_ids],
        }
        eligible = class_candidates["A"] + class_candidates["B"]
        if not eligible:
            # If the root plan was unavailable, retain the prior critic gate.
            return promising[:maximum]
        by_class = {
            action_class: sorted(
                [node for node in class_candidates[action_class]
                 if self._action_class(node) == action_class],
                key=lambda node: (_planner_cost(node),
                                  _positive(node.prediction.get("predicted_ms")) or math.inf,
                                  node.id),
            )
            for action_class in ("A", "B")
        }
        selected: list[_Node] = []
        if self.operator == "mcts" and maximum >= 2:
            available = [action_class for action_class in ("A", "B") if by_class[action_class]]
            if len(available) >= 2:
                selected = [by_class["A"][0], by_class["B"][0]]
            # With only one action class, retain the critic/beam order.  The
            # stratification guard is needed only when both classes compete;
            # changing the single-class ranking would discard the calibrated
            # critic ordering used by the standalone arms.
            elif available:
                selected = [node for node in promising
                            if self._action_class(node) == available[0]][:maximum]
        if not selected:
            selected = promising[:maximum]
        return selected[:maximum]

    @staticmethod
    def _single_table_seq_scan(node: _Node) -> bool:
        rows = operators(node.plan)
        return (len(rows) == 1 and rows[0].get("node_type") == "Seq Scan"
                and plan_features(node.plan).get("nested_loops", 0) == 0
                and plan_features(node.plan).get("subplans", 0) == 0)

    def _prefer_symbolic_winner(self, winner: _Node, winner_measurement: dict | None,
                                canaries: list[dict], nodes: list[_Node]) -> tuple[_Node, dict | None]:
        """Prefer a verified symbolic winner when both action classes pass."""
        if self.operator != "mcts" or winner is None or winner is nodes[0]:
            return winner, winner_measurement
        passed = {receipt["node_id"]: receipt for receipt in canaries
                  if receipt.get("accepted") is True}
        symbolic = [node for node in nodes if self._action_class(node) == "A" and node.id in passed]
        if not symbolic:
            return winner, winner_measurement
        best_symbolic = min(symbolic, key=lambda node: (_planner_cost(node), node.id))
        winner_class = self._action_class(winner)
        if winner_class == "B":
            symbolic_receipt = passed[best_symbolic.id]
            symbolic_ms = _positive(symbolic_receipt["measurement"].get("median_ms"))
            winner_ms = _positive(winner_measurement.get("median_ms")) if winner_measurement else None
            symbolic_features = plan_features(best_symbolic.plan)
            winner_features = plan_features(winner.plan)
            within_ten_percent = (symbolic_ms is not None and winner_ms is not None
                                  and symbolic_ms <= winner_ms * 1.10)
            simple_scan_tie_break = (
                self._single_table_seq_scan(best_symbolic)
                and self._single_table_seq_scan(winner)
                and _planner_cost(best_symbolic) < _planner_cost(winner)
                and symbolic_features.get("width", math.inf) < winner_features.get("width", math.inf))
            if within_ten_percent or simple_scan_tie_break:
                return best_symbolic, symbolic_receipt["measurement"]
        return winner, winner_measurement

    def _verify(self, original_sql: str, candidate_sql: str) -> dict:
        try:
            result = self.gate.verify(original_sql, candidate_sql)
            if not isinstance(result, dict):
                raise TypeError("equivalence verifier returned no result dictionary")
            return result
        except Exception as exc:
            return {"accepted": False, "formal": False, "proof": "none", "execution": {},
                    "reason": f"verification error: {type(exc).__name__}: {exc}", **_error(exc)}

    def _retrieve(self, node: _Node) -> dict:
        if not self.config.use_rag or self._evidence is None:
            return {}
        if isinstance(self._evidence, dict):
            return self._evidence
        try:
            result = self._evidence.retrieve(node.sql, plan=node.plan,
                                            bottleneck=bottleneck(node.plan), limit=6)
            return result if isinstance(result, dict) else {"documents": result}
        except Exception as exc:
            self.trace.append({"event": "retrieval_error", "node_id": node.id,
                               **_error(exc)})
            return {}

    def _record(self, sql: str, status: str, verification: dict | None = None) -> None:
        if self.config.use_rag and callable(getattr(self._evidence, "record_attempt", None)):
            try:
                self._evidence.record_attempt(sql, status, verification=verification)
            except Exception as exc:
                self.trace.append({"event": "evidence_record_error", "error": str(exc)})

    def _generate(self, parents: list[_Node], schema_sql: str, depth: int) -> list[dict]:
        contexts = [{"id": node.id, "sql": node.sql, "plan": node.plan,
                     "depth": node.depth, "bottleneck": bottleneck(node.plan),
                     "evidence": self._retrieve(node)} for node in parents]
        evidence = {"parents": [{"id": row["id"], "evidence": row["evidence"]} for row in contexts]}
        if self.operator == "mcts":
            result = self.generator.generate_batch(
                contexts, schema_sql, depth=depth,
                limit_per_parent=self.config.candidate_limit_per_parent, evidence=evidence)
        else:
            result = self.generator.macro_batch(
                contexts, schema_sql, operator="quite", depth=depth,
                limit_per_parent=self.config.candidate_limit_per_parent, evidence=evidence)
        if not isinstance(result, list):
            raise TypeError("generator returned a non-list candidate batch")
        return result

    def _critic(self, candidates: list[_Node], root: _Node, original_ms: float) -> None:
        root_cost = _planner_cost(root)
        # An absent/model-budget-exhausted critic still has an auditable
        # PostgreSQL estimate.  No target candidate timing enters this ranking.
        for node in candidates:
            cost = _planner_cost(node)
            estimate = original_ms * cost / root_cost if math.isfinite(cost) and math.isfinite(root_cost) else None
            node.prediction = {"predicted_ms": estimate, "source": "postgres_cost_ratio",
                               "used_for": "candidate_order_only", "target_candidate_observations": 0}
        try:
            rows = [{"id": node.id, "sql": node.sql, "plan": node.plan,
                     "depth": node.depth, "parent_id": node.parent_id,
                     "verification": node.verification} for node in candidates]
            predictions = self.generator.critic_batch(
                rows, original_sql=root.sql,
                observations=[{"sql": root.sql, "plan": root.plan,
                               "measured_ms": original_ms, "ms": original_ms}],
                operator=self.operator)
            if not isinstance(predictions, dict):
                raise TypeError("critic returned no candidate prediction dictionary")
            for node in candidates:
                prediction = predictions.get(node.id, predictions.get(str(node.id)))
                if isinstance(prediction, dict) and _positive(prediction.get("predicted_ms")) is not None:
                    node.prediction = {**node.prediction, **prediction,
                                       "predicted_ms": _positive(prediction["predicted_ms"]),
                                       "used_for": "candidate_order_only",
                                       "target_candidate_observations": 0}
        except Exception as exc:
            self.trace.append({"event": "critic_error", **_error(exc)})

    def _canary(self, node: _Node, timeout_ms: int) -> dict:
        sentinel = object()
        previous = getattr(self.db, "timeout_ms", sentinel)
        try:
            self.db.timeout_ms = timeout_ms
            measurement = self.db.measure(node.sql, runs=1, warmups=0)
            if not isinstance(measurement, dict):
                raise TypeError("database measurement is not a dictionary")
            return dict(measurement)
        except Exception as exc:
            return {"status": "timeout" if isinstance(exc, TimeoutError) or getattr(exc, "sqlstate", None) == "57014" else "error",
                    "median_ms": None, "samples_ms": [],
                    **_error(exc)}
        finally:
            if previous is sentinel:
                delattr(self.db, "timeout_ms")
            else:
                self.db.timeout_ms = previous

    def run(self, original_sql: str, schema_sql: str, original_ms: float | None = None) -> dict:
        started = time.monotonic()
        original_ms = _positive(original_ms)
        if original_ms is None:
            raise ValueError("original_ms must be a finite positive time measured by the runner")
        self.trace: list[dict] = []
        initial_model_requests = getattr(self.generator, "model_requests", None)
        self._evidence = self.evidence
        if self.config.use_rag and self._evidence is None:
            self._evidence = EvidenceStore(Path(getattr(self.generator, "root", Path(__file__).resolve().parents[1])))
        if callable(getattr(self._evidence, "begin_query", None)):
            self._evidence.begin_query(original_sql, schema_sql)
        if callable(getattr(self.generator, "begin_query", None)):
            self.generator.begin_query(original_sql, schema_sql)
        # The original query is the reference, whose final-timing result was
        # supplied by the runner.  Do not execute it again inside search.
        identity = {"accepted": True, "formal": True, "proof": "identity",
                    "execution": {}, "reason": "original unrevised reference query"}
        root = _Node(0, original_sql, None, 0, {}, identity)
        nodes, candidates, beam = [root], [], [root]
        generation_batches = critic_batches = expanded = 0
        reason = "no verified candidate available"
        try:
            root.plan = self._plan(original_sql)
        except Exception as exc:
            beam = []
            reason = f"original EXPLAIN failed: {type(exc).__name__}: {exc}"
            self.trace.append({"event": "root_plan_error", **_error(exc)})
        self.trace.append({"event": "root", "node_id": 0, "sql": original_sql,
                           "original_ms": original_ms, "plan": root.plan,
                           "verification": identity, "timing_source": "runner_original_measurement"})
        seen = {_key(original_sql)}
        for depth in range(1, self.config.max_depth + 1):
            if not beam:
                break
            parents = {node.id: node for node in beam}
            expanded += len(beam)
            generation_batches += 1
            try:
                proposed = self._generate(beam, schema_sql, depth)
                self.trace.append({"event": "generation_batch", "depth": depth,
                                   "parent_ids": list(parents), "proposed": len(proposed),
                                   "generator_diagnostics": list(getattr(self.generator, "last_diagnostics", []))})
            except Exception as exc:
                self.trace.append({"event": "generation_error", "depth": depth,
                                   **_error(exc)})
                break
            children, per_parent = [], {node_id: 0 for node_id in parents}
            for proposal in proposed:
                if not isinstance(proposal, dict) or not isinstance(proposal.get("sql"), str) or not proposal["sql"].strip():
                    self.trace.append({"event": "candidate", "depth": depth, "status": "malformed"})
                    continue
                candidate_sql = proposal["sql"]
                parent_id = proposal.get("parent_id")
                # A missing parent is unambiguous only in the first batch.
                if parent_id is None and len(parents) == 1:
                    parent_id = next(iter(parents))
                if isinstance(parent_id, str) and parent_id.isdigit():
                    parent_id = int(parent_id)
                event = {"event": "candidate", "depth": depth, "parent_id": parent_id,
                         "sql": candidate_sql, "action_class": proposal.get("action_class"),
                         "action": proposal.get("action"), "provenance": proposal.get("provenance", {})}
                self.trace.append(event)
                if isinstance(parent_id, bool) or not isinstance(parent_id, int) or parent_id not in parents:
                    event.update(status="rejected", reason="candidate has no current beam parent")
                    continue
                if per_parent[parent_id] >= self.config.candidate_limit_per_parent:
                    event.update(status="truncated", reason="candidate limit per parent reached")
                    continue
                fingerprint = _key(candidate_sql)
                if fingerprint in seen:
                    event.update(status="duplicate", reason="SQL already attempted in this query")
                    self._record(candidate_sql, "duplicate")
                    continue
                seen.add(fingerprint)
                if proposal.get("action_class") not in {"A", "B"}:
                    event.update(status="rejected", reason="unknown action class")
                    self._record(candidate_sql, "rejected")
                    continue
                verification = self._verify(original_sql, candidate_sql)
                event["verification"] = verification
                if verification.get("accepted") is not True:
                    event.update(status="rejected", reason=verification.get("reason", "root equivalence was not established"))
                    self._record(candidate_sql, "rejected", verification)
                    continue
                try:
                    candidate_plan = self._plan(candidate_sql)
                except Exception as exc:
                    event.update(status="plan_error", reason=f"{type(exc).__name__}: {exc}", **_error(exc))
                    self._record(candidate_sql, "plan_error", verification)
                    continue
                # Count only candidates that survived equivalence and plan
                # validation. Invalid early proposals must not consume the
                # per-parent slots needed by later cached or symbolic SQL.
                per_parent[parent_id] += 1
                child = _Node(len(nodes), candidate_sql, parent_id, depth, candidate_plan,
                              verification, proposal)
                # Root verification precedes every insertion, beam selection,
                # critic request, and target canary execution.
                nodes.append(child)
                candidates.append(child)
                children.append(child)
                event.update(status="accepted", node_id=child.id, plan=candidate_plan)
                self._record(candidate_sql, "accepted", verification)
            beam = sorted(children, key=lambda node: (_planner_cost(node), node.id))[:self.config.beam_width]
            self.trace.append({"event": "beam", "depth": depth, "node_ids": [node.id for node in beam],
                               "ranking": "postgres_explain_total_cost", "target_executions": 0})

        if candidates:
            critic_batches = 1
            self._critic(candidates, root, original_ms)
            self.trace.append({"event": "critic_batch", "node_ids": [node.id for node in candidates],
                               "generator_diagnostics": list(getattr(self.generator, "last_diagnostics", [])),
                               "target_candidate_observations": 0})
        ranked = sorted(candidates, key=lambda node: (_positive(node.prediction.get("predicted_ms")) or math.inf,
                                                       _planner_cost(node), node.id))
        promising = [node for node in ranked
                     if (prediction := _positive(node.prediction.get("predicted_ms"))) is not None
                     and prediction < original_ms]
        promising_ids = {node.id for node in promising}
        pruned = [{"node_id": node.id, "predicted_ms": node.prediction.get("predicted_ms"),
                   "reason": "prediction does not improve on the original"}
                  for node in ranked if node.id not in promising_ids]
        self.trace.append({"event": "prediction_pruning", "original_ms": original_ms,
                           "required_predicted_ms": f"< {original_ms}",
                           "promising_node_ids": [node.id for node in promising],
                           "pruned": pruned, "target_executions": 0})
        winner, winner_measurement, canaries = root, None, []
        timeout_ms = max(1, math.ceil(1.05 * original_ms))
        maximum = self.config.max_canaries
        if self.canary_budget is not None:
            maximum = min(maximum, max(0, int(self.canary_budget.get("remaining", 0))))
        abort = False
        canary_nodes = self._canary_candidates(candidates, promising, root, maximum)
        for node in canary_nodes:
            if self.canary_budget is not None:
                self.canary_budget["remaining"] -= 1
            measurement = self._canary(node, timeout_ms)
            measured_ms = _positive(measurement.get("median_ms")) if measurement.get("status") == "ok" else None
            if measured_ms is not None:
                measurement["median_ms"] = measured_ms
            passed = measured_ms is not None and measured_ms < original_ms
            receipt = {"node_id": node.id, "sql": node.sql, "measurement": measurement,
                       "statement_timeout_ms": timeout_ms, "original_ms": original_ms,
                       "accepted": passed, "runs": 1, "warmups": 0}
            canaries.append(receipt)
            self.trace.append({"event": "canary", **receipt})
            if not passed:
                continue
            if measured_ms < original_ms and (winner_measurement is None or measured_ms < winner_measurement["median_ms"]):
                winner, winner_measurement = node, measurement
                reason = "verified candidate improved on the original in its bounded canary"
        winner, winner_measurement = self._prefer_symbolic_winner(
            winner, winner_measurement, canaries, nodes)
        baseline_without_canary = self.operator == "quite" and self.config.max_canaries == 0
        if baseline_without_canary and promising:
            winner = promising[0]
            reason = "verified QUITE candidate selected by EXPLAIN and joint critic ranking"
        if winner is root and not abort and canaries:
            reason = "candidate canaries did not improve on the original"
        elif winner is root and candidates and not promising:
            reason = "no candidate prediction improves on the original; prediction pruning selected the root"
        elif winner is root and promising and maximum == 0:
            reason = "query-wide canary budget exhausted"
        selection = {"accepted": winner is not root, "reason": reason,
                     "scope": "explain_and_critic_without_canary" if baseline_without_canary else "single_canary_selection_not_final_evaluation",
                     "canaries": len(canaries), "aborted": abort,
                     "paired_checks": 0, "independent_holdout_used": False,
                     "final_holdout_used": False}
        self.trace.append({"event": "selection", "node_id": winner.id,
                           "fallback": winner is root, "validation": selection,
                           "final_holdout_used": False})
        final_model_requests = getattr(self.generator, "model_requests", None)
        model_requests = (final_model_requests - initial_model_requests
                          if isinstance(initial_model_requests, int) and isinstance(final_model_requests, int)
                          else None)
        return {"sql": winner.sql, "verification": winner.verification,
                "measurement": winner_measurement, "trace": self.trace,
                "fallback": winner is root, "expanded": expanded,
                "generation_batches": generation_batches, "critic_batches": critic_batches,
                "model_call_cap": self.config.max_depth + 1, "model_requests": model_requests,
                "canaries": canaries, "target_candidate_executions": len(canaries),
                "canary_statement_timeout_ms": timeout_ms,
                "selection_validation": selection, "original_ms": original_ms,
                "operator": self.operator, "search_mode": "fast_batched_beam",
                "budget": self.config.max_depth, "budget_unit": "batched_generation_depths",
                "candidate_limit_per_expansion": self.config.candidate_limit_per_parent,
                "maximum_candidate_slots": self.config.candidate_limit_per_parent *
                    (1 + self.config.beam_width * (self.config.max_depth - 1)),
                "critic": {"enabled": True, "mode": "one_joint_batch_after_generation",
                           "batches": critic_batches, "target_candidate_observations": 0},
                "calibration": {"scope": "runner_original_only", "final_holdout_used": False},
                "config": asdict(self.config), "search_seconds": time.monotonic() - started,
                "ranking": [{"node_id": node.id, "predicted_ms": node.prediction.get("predicted_ms"),
                             "source": node.prediction.get("source"),
                             "eligible_after_pruning": node.id in promising_ids} for node in ranked],
                "root_reference": {"node_id": 0, "original_ms": original_ms},
                "promising_node_ids": [node.id for node in promising],
                "pruned_node_ids": [row["node_id"] for row in pruned],
                "pruned_ranking": pruned,
                "nodes": [{"id": node.id, "parent_id": node.parent_id, "depth": node.depth,
                           "sql": node.sql, "verification": node.verification,
                           "critic": node.prediction} for node in nodes]}
