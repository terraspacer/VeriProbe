"""Budgeted, root-verified tree search with independent selection validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median, stdev
import time
from typing import Any

from .critic import CostCritic
from .evidence import EvidenceStore
from .plans import bottleneck


@dataclass
class SearchConfig:
    budget: int = 16
    use_plan: bool = True
    use_rag: bool = True
    use_critic: bool = True
    use_safety: bool = True
    seed: int = 0
    candidate_limit: int = 4
    safety_pairs: int = 7
    safety_margin: float = 0.01
    exploration: float = math.sqrt(2.0)
    restore_symbolic_output_contract: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.budget, int) or not 0 <= self.budget <= 16:
            raise ValueError("budget must be an integer between 0 and 16 expansions")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if self.safety_pairs < 7:
            raise ValueError("at least seven independent safety pairs are required")
        if not 0 <= self.safety_margin < 1:
            raise ValueError("safety_margin must lie in [0, 1)")


def _ms(measurement: dict) -> float | None:
    try:
        value = float(measurement.get("median_ms"))
        return value if measurement.get("status") == "ok" and math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _key(sql: str) -> str:
    """Ignore formatting, while retaining literal values, identifiers and hints."""
    try:
        import sqlglot
        tokens = sqlglot.Dialect.get_or_raise("postgres").tokenize(sql)
        while tokens and tokens[-1].token_type.name == "SEMICOLON":
            tokens.pop()
        preserved = {"STRING", "NUMBER", "IDENTIFIER", "BYTE_STRING", "RAW_STRING", "NATIONAL_STRING", "HEX_STRING", "BIT_STRING"}
        parts = [(token.token_type.name, token.text if token.token_type.name in preserved else token.text.lower(), token.comments) for token in tokens]
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()
    except Exception:
        return hashlib.sha256(sql.strip().encode()).hexdigest()


@dataclass
class _Node:
    id: int
    sql: str
    parent: "_Node | None"
    measurement: dict
    verification: dict
    plan: dict
    prediction: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    children: list["_Node"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    expanded: bool = False
    prior: float = 0.0

    def expandable(self) -> bool:
        return not self.expanded or any(child.expandable() for child in self.children)

    def propagate(self, reward: float) -> None:
        current: _Node | None = self
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent


class SearchController:
    def __init__(self, db: Any, gate: Any, generator: Any, config: SearchConfig | None = None,
                 evidence: Any = None, contract_restorer: Any = None):
        self.db, self.gate, self.generator = db, gate, generator
        self.config = config or SearchConfig()
        self.evidence = evidence
        if contract_restorer is None:
            from .db import Postgres
            if isinstance(db, Postgres):
                from .output_contract import restore_output_contract
                contract_restorer = restore_output_contract
        self.contract_restorer = contract_restorer

    def _measure(self, sql: str, runs: int = 3, warmups: int = 1) -> dict:
        try:
            result = self.db.measure(sql, runs=runs, warmups=warmups)
            if not isinstance(result, dict):
                raise TypeError("database measurement is not a dictionary")
            result = dict(result)
            if result.get("status") == "ok" and _ms(result) is None:
                result.update(status="error", median_ms=None, error="invalid or nonpositive measurement")
            return result
        except Exception as exc:
            return {"status": "error", "median_ms": None, "samples_ms": [],
                    "error": f"{type(exc).__name__}: {exc}"}

    def _verify(self, original: str, candidate: str) -> dict:
        try:
            result = self.gate.verify(original, candidate)
            if not isinstance(result, dict):
                raise TypeError("equivalence verifier returned no result dictionary")
            return result
        except Exception as exc:
            return {"accepted": False, "formal": False, "proof": "none", "execution": {},
                    "reason": f"verification error: {type(exc).__name__}: {exc}",
                    "error": f"{type(exc).__name__}: {exc}"}

    def _plan(self, sql: str, measurement: dict | None = None) -> dict:
        if not self.config.use_plan:
            return {}
        if isinstance((measurement or {}).get("plan"), dict) and measurement["plan"]:
            return measurement["plan"]
        try:
            plan = self.db.explain(sql, analyze=False)
            return plan if isinstance(plan, dict) else {}
        except Exception as exc:
            self.trace.append({"event": "plan_error", "sql": sql, "error": f"{type(exc).__name__}: {exc}"})
            return {}

    def _retrieve(self, sql: str, plan: dict, detail: dict) -> dict:
        if not self.config.use_rag or self._evidence is None:
            return {}
        try:
            if isinstance(self._evidence, dict):
                # An explicit evidence dictionary is treated as a knowledge input;
                # query-local observations remain separate from it.
                return self._evidence
            result = self._evidence.retrieve(sql, plan=plan, bottleneck=detail, limit=6)
            return result if isinstance(result, dict) else {"documents": result}
        except Exception as exc:
            self.trace.append({"event": "retrieval_error", "error": f"{type(exc).__name__}: {exc}"})
            return {"documents": [], "retrieval": {"error": str(exc)}}

    def _record(self, candidate: dict, status: str, verification: dict | None = None,
                detail: dict | None = None) -> None:
        if self.config.use_rag and hasattr(self._evidence, "record_attempt"):
            self._evidence.record_attempt(candidate.get("sql", ""), status, verification=verification, bottleneck=detail)

    def _select(self, root: _Node) -> _Node:
        current = root
        while current.expanded:
            children = [child for child in current.children if child.expandable()]
            if not children:
                return current
            scores = [(child.total_reward / max(1, child.visits)
                       + self.config.exploration * math.sqrt(math.log(current.visits + 2) / (child.visits + 1))
                       + 0.1 * child.prior + self.rng.random() * 1e-12, child) for child in children]
            current = max(scores, key=lambda item: item[0])[1]
        return current

    def _reward(self, measurement: dict, root_ms: float | None) -> float:
        value = _ms(measurement)
        return max(-2.0, min(2.0, math.log(root_ms / value))) if root_ms and value else -2.0

    def _validate(self, root: _Node, winner: _Node) -> dict:
        """A fresh, single-winner validation set; it is not final evaluation."""
        pairs = []
        orientation = self.rng.randrange(2)
        for index in range(self.config.safety_pairs):
            order = ["root", "candidate"] if (index + orientation) % 2 == 0 else ["candidate", "root"]
            measurements = {}
            for role in order:
                measurements[role] = self._measure(root.sql if role == "root" else winner.sql, runs=1, warmups=0)
            root_ms, candidate_ms = _ms(measurements["root"]), _ms(measurements["candidate"])
            pairs.append({"index": index, "order": order, "root": measurements["root"],
                          "candidate": measurements["candidate"],
                          "ratio": candidate_ms / root_ms if root_ms and candidate_ms else None})
            if root_ms is None or candidate_ms is None:
                return {"accepted": False, "reason": "independent paired measurement failed", "pairs": pairs,
                        "scope": "selection_validation_not_final_holdout"}
        ratios = [pair["ratio"] for pair in pairs]
        logs = [math.log(ratio) for ratio in ratios]
        # 3.143 is the one-sided 99% Student-t critical value at six degrees of
        # freedom, conservative for all permitted larger sample counts.
        upper_log = mean(logs) + 3.143 * stdev(logs) / math.sqrt(len(logs))
        bound = math.exp(min(upper_log, 700))
        margin = 1.0 - self.config.safety_margin
        unanimous = all(ratio < margin for ratio in ratios)
        accepted = unanimous and bound < margin
        return {"accepted": accepted, "reason": "paired evidence supports the selected improvement" if accepted else "insufficient independent evidence of improvement",
                "pairs": pairs, "paired_median_ratio": median(ratios),
                "one_sided_99pct_log_t_upper_ratio": bound,
                "all_pairs_improve_by_margin": unanimous, "required_ratio": margin,
                "sign_test_p_if_all_pairs_improve": 2.0 ** (-len(ratios)),
                "assumptions": "independent representative timing pairs; log-ratio t bound assumes approximately normal errors",
                "guaranteed_regression_rate": None, "scope": "selection_validation_not_final_holdout"}

    def run(self, original_sql: str, schema_sql: str) -> dict:
        started = time.monotonic()
        self.trace: list[dict] = []
        self.rng = random.Random(self.config.seed)
        self._evidence = self.evidence
        if self.config.use_rag and self._evidence is None:
            self._evidence = EvidenceStore(Path(getattr(self.generator, "root", Path(__file__).resolve().parents[1])))
        if self.config.use_rag and hasattr(self._evidence, "begin_query"):
            self._evidence.begin_query(original_sql, schema_sql)
        if callable(getattr(self.generator, "begin_query", None)):
            self.generator.begin_query(original_sql, schema_sql)
        model = getattr(self.generator, "model", None)
        critic = CostCritic(model=model if callable(model) else None) if self.config.use_critic else None
        root_verification = self._verify(original_sql, original_sql)
        root_measurement = self._measure(original_sql)
        root_plan = self._plan(original_sql, root_measurement)
        root = _Node(0, original_sql, None, root_measurement, root_verification, root_plan)
        nodes = [root]
        if critic:
            critic.observe(root_plan, root_measurement)
        self.trace.append({"event": "root", "node_id": 0, "sql": original_sql,
                           "measurement": root_measurement, "verification": root_verification})
        seen = {_key(original_sql)}
        root_ms = _ms(root_measurement)
        expanded = 0
        while expanded < self.config.budget and root.expandable():
            node = self._select(root)
            if node.expanded:
                break
            node.expanded = True
            expanded += 1
            detail = bottleneck(node.plan) if self.config.use_plan else {}
            knowledge = self._retrieve(node.sql, node.plan, detail)
            expansion = {"event": "expansion", "expansion": expanded, "node_id": node.id,
                         "parent_id": node.parent.id if node.parent else None,
                         "bottleneck": detail, "evidence": knowledge}
            self.trace.append(expansion)
            try:
                candidates = self.generator.candidates(node.sql, schema_sql, node.plan, detail, knowledge,
                                                       limit=self.config.candidate_limit)
                if not isinstance(candidates, list):
                    raise TypeError("generator returned a non-list candidate batch")
                expansion["generator_diagnostics"] = list(getattr(self.generator, "last_diagnostics", []))
            except Exception as exc:
                self.trace.append({"event": "generation_error", "node_id": node.id,
                                   "error": f"{type(exc).__name__}: {exc}"})
                node.propagate(-2.0)
                continue
            # Enforce the configured branch factor even if a provider ignores it;
            # retain an audit event for every truncated provider output.
            accepted_rewards = []
            for position, candidate in enumerate(candidates):
                event = {"event": "candidate", "attempted": True, "parent_id": node.id,
                         "expansion": expanded, "verification_against": "original_root"}
                if not isinstance(candidate, dict) or not isinstance(candidate.get("sql"), str) or not candidate["sql"].strip():
                    event.update(status="error", reason="malformed candidate", candidate=repr(candidate)[:2000])
                    self.trace.append(event)
                    continue
                event.update(sql=candidate["sql"], action_class=candidate.get("action_class"),
                             action=candidate.get("action"), provenance=candidate.get("provenance", {}))
                if position >= self.config.candidate_limit:
                    event.update(status="budget_truncated", reason="provider exceeded candidate_limit")
                    self.trace.append(event)
                    self._record(candidate, "budget_truncated")
                    continue
                if (candidate.get("action_class") == "A" and self.config.restore_symbolic_output_contract
                        and callable(self.contract_restorer)):
                    try:
                        restoration = self.contract_restorer(self.db, original_sql, candidate["sql"])
                    except Exception as exc:
                        restoration = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                    event["output_contract"] = restoration
                    if (isinstance(restoration, dict) and restoration.get("status") == "restored"
                            and isinstance(restoration.get("sql"), str) and restoration["sql"].strip()):
                        event["raw_symbolic_sql"] = candidate["sql"]
                        candidate = {**candidate, "sql": restoration["sql"], "provenance": {
                            **candidate.get("provenance", {}), "output_contract_policy": restoration.get("policy")}}
                        event.update(sql=candidate["sql"], provenance=candidate["provenance"])
                key = _key(candidate["sql"])
                if key in seen:
                    event.update(status="duplicate", reason="SQL already attempted in this search")
                    self.trace.append(event)
                    self._record(candidate, "duplicate")
                    continue
                seen.add(key)
                if candidate.get("action_class") not in ("A", "B"):
                    event.update(status="rejected", reason="unknown action class")
                    self.trace.append(event)
                    self._record(candidate, "rejected")
                    continue
                verification = self._verify(original_sql, candidate["sql"])
                event["verification"] = verification
                if verification.get("accepted") is not True:
                    event.update(status="error" if verification.get("error") else "rejected", reason=verification.get("reason", "equivalence was not established"))
                    self.trace.append(event)
                    self._record(candidate, event["status"], verification)
                    continue
                # Root equivalence was established before any tree insertion or timing.
                candidate_plan = self._plan(candidate["sql"])
                prediction = critic.score(candidate_plan, candidate["sql"], bottleneck(candidate_plan) if candidate_plan else {}, knowledge) if critic else {}
                measurement = self._measure(candidate["sql"])
                event.update(measurement=measurement, critic=prediction)
                if _ms(measurement) is None:
                    event.update(status="error", reason="candidate timing failed after equivalence verification")
                    self.trace.append(event)
                    self._record(candidate, "error", verification)
                    continue
                if not candidate_plan:
                    candidate_plan = self._plan(candidate["sql"], measurement)
                if critic:
                    critic.observe(candidate_plan, measurement, prediction)
                # Keep calibration paired with the plan used for prediction.
                # Subsequent expansions can use observed loops from TIMING OFF
                # measurements instead of multiplying uncertain cached rescans.
                measured_plan = measurement.get("plan")
                if self.config.use_plan and isinstance(measured_plan, dict) and measured_plan:
                    candidate_plan = measured_plan
                    event["expansion_plan_source"] = "search_measurement"
                predicted_ms = prediction.get("predicted_ms")
                prior = max(-2.0, min(2.0, math.log(root_ms / predicted_ms))) if root_ms and predicted_ms else 0.0
                child = _Node(len(nodes), candidate["sql"], node, measurement, verification,
                              candidate_plan, prediction, candidate, prior=prior)
                node.children.append(child)
                nodes.append(child)
                reward = self._reward(measurement, root_ms)
                child.propagate(reward)
                accepted_rewards.append(reward)
                event.update(status="accepted", node_id=child.id, reward=reward)
                self.trace.append(event)
                self._record(candidate, "accepted", verification, bottleneck(candidate_plan) if candidate_plan else {})
            if not accepted_rewards:
                node.propagate(self._reward(node.measurement, root_ms))

        valid = [node for node in nodes if _ms(node.measurement) is not None]
        # A failed root timing cannot support a safe relative-improvement decision.
        winner = min(valid, key=lambda node: (_ms(node.measurement), node.id)) if valid and root_ms else root
        validation: dict = {"accepted": winner is root, "reason": "root timing unavailable; relative improvement cannot be established" if root_ms is None else "root is the measured best candidate",
                            "scope": "selection_validation_not_final_holdout"}
        if winner is not root:
            if self.config.use_safety:
                validation = self._validate(root, winner)
                if not validation["accepted"]:
                    winner = root
            else:
                validation = {"accepted": None, "reason": "paired safety validation disabled by ablation",
                              "scope": "selection_validation_not_final_holdout"}
        self.trace.append({"event": "selection", "node_id": winner.id, "fallback": winner is root,
                           "validation": validation, "final_holdout_used": False})
        return {"sql": winner.sql, "measurement": winner.measurement,
                "verification": winner.verification, "trace": self.trace,
                "budget": self.config.budget, "expanded": expanded, "fallback": winner is root,
                "budget_unit": "node_expansions", "candidate_limit_per_expansion": self.config.candidate_limit,
                "maximum_candidate_slots": self.config.budget * self.config.candidate_limit,
                "selection_validation": validation,
                "critic": critic.metadata() if critic else {"enabled": False},
                "calibration": {"scope": "current_query_search_only", "final_holdout_used": False},
                "config": asdict(self.config), "search_seconds": time.monotonic() - started,
                "nodes": [{"id": node.id, "parent_id": node.parent.id if node.parent else None,
                           "visits": node.visits, "total_reward": node.total_reward,
                           "expanded": node.expanded, "children": [child.id for child in node.children]}
                          for node in nodes]}
