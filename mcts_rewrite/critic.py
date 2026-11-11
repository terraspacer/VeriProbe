"""Optional LLM cost critic with query-local PostgreSQL calibration.

A critic orders exploration only. It is never an equivalence verifier or a
substitute for timed execution. No holdout measurements enter its calibration.
"""
from __future__ import annotations

import json
import inspect
import math
import re
from statistics import median
from typing import Any, Callable

import numpy as np

from .plans import plan_features

_FEATURES = ("total_cost", "rows", "width", "operators", "nested_loops", "seq_scans", "subplans", "max_rescans", "exclusive_work")


def _positive(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


class CostCritic:
    def __init__(self, model: Callable[[dict], Any] | None = None):
        self.model = model
        self.observations: list[dict] = []
        self.model_requests = 0
        self.llm_predictions = 0
        self.model_errors: list[str] = []
        self.model_unavailable: list[str] = []
        self.fallback_predictions = 0

    @staticmethod
    def _vector(features: dict) -> np.ndarray:
        return np.array([math.log1p(features.get(key, 0.0)) for key in _FEATURES], dtype=float)

    def _pg_prediction(self, features: dict) -> tuple[float | None, str]:
        if not self.observations:
            return None, "uncalibrated_planner_units"
        costs = [max(row["features"]["total_cost"], 1.0) for row in self.observations]
        residuals = np.array([math.log(row["ms"] / cost) for row, cost in zip(self.observations, costs)])
        target_cost = max(features["total_cost"], 1.0)
        intercept = float(median(residuals))
        if len(self.observations) < 4:
            return target_cost * math.exp(intercept), "median_ms_per_planner_cost"
        matrix = np.vstack([self._vector(row["features"]) for row in self.observations])
        center, scale = matrix.mean(axis=0), np.maximum(matrix.std(axis=0), 1.0)
        design = (matrix - center) / scale
        # Shrink heavily: a short search offers only a handful of noisy observations.
        ridge = design.T @ design + np.eye(len(_FEATURES)) * 10.0
        weights = np.linalg.solve(ridge, design.T @ (residuals - intercept))
        adjustment = float((self._vector(features) - center) / scale @ weights)
        predicted = target_cost * math.exp(max(-20, min(20, intercept + adjustment)))
        return predicted, "ridge_log_residual_on_pg_features"

    def score(self, plan: dict, sql: str = "", bottleneck: dict | None = None,
              evidence: dict | None = None) -> dict:
        features = plan_features(plan)
        pg_ms, method = self._pg_prediction(features)
        result = {"predicted_ms": pg_ms, "source": "non_llm_pg_fallback",
                  "calibration_method": method, "calibration_observations": len(self.observations),
                  "features": features, "used_for": "exploration_order_only",
                  "uncertainty": "not a latency guarantee", "llm_raw_ms": None}
        available = self.model is not None
        supports_task = getattr(self.model, "supports_task", None)
        reason = None
        if available and callable(supports_task):
            try:
                if not supports_task("cost_critic"):
                    reason = "configured provider has no cost_critic responses"
                else:
                    try:
                        inspect.signature(supports_task).bind("cost_critic", sql, plan)
                    except (TypeError, ValueError):
                        pass  # Providers may declare task capability without request coverage.
                    else:
                        if not supports_task("cost_critic", sql, plan):
                            reason = "configured provider has no cost_critic response for this SQL and plan"
            except Exception as exc:
                reason = f"cost critic capability check failed: {type(exc).__name__}: {exc}"
        if reason:
            available = False
            if reason not in self.model_unavailable:
                self.model_unavailable.append(reason)
            result["llm_unavailable_reason"] = reason
        elif self.model is None:
            result["llm_unavailable_reason"] = "no model provider configured"
        if available:
            payload = {"task": "cost_critic", "sql": sql, "plan": plan,
                       "bottleneck": bottleneck or {}, "evidence": evidence or {},
                       "pg_features": features, "calibrated_pg_predicted_ms": pg_ms,
                       "observations": [{"features": row["features"], "measured_ms": row["ms"]}
                                        for row in self.observations[-16:]],
                       "output_schema": {"predicted_ms": "positive number", "rationale": "brief explanation"},
                       "instruction": "Predict latency from PostgreSQL features and the supplied search observations. Do not assert equivalence."}
            self.model_requests += 1
            try:
                response = self.model(payload)
                if isinstance(response, str):
                    response = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip()))
                raw = _positive(response.get("predicted_ms")) if isinstance(response, dict) else None
                if raw is None:
                    raise ValueError("cost critic response has no finite positive predicted_ms")
                residuals = [math.log(row["ms"] / row["llm_raw_ms"]) for row in self.observations if row.get("llm_raw_ms")]
                factor = math.exp(median(residuals)) if residuals else 1.0
                predicted = _positive(raw * factor)
                if predicted is None:
                    raise ValueError("calibrated cost prediction is not finite")
                result.update({"predicted_ms": predicted, "source": "llm",
                               "llm_raw_ms": raw, "llm_calibration_factor": factor,
                               "llm_calibration_observations": len(residuals),
                               "rationale": str(response.get("rationale", ""))[:2000],
                               "model_provenance": response.get("provenance", {})})
                self.llm_predictions += 1
                return result
            except Exception as exc:
                self.model_errors.append(f"{type(exc).__name__}: {exc}")
                result["llm_error"] = self.model_errors[-1]
        self.fallback_predictions += 1
        return result

    def observe(self, plan: dict, measurement: dict, prediction: dict | None = None,
                phase: str = "search") -> None:
        if phase != "search":
            raise ValueError("CostCritic only accepts search observations, never validation or holdout timing")
        milliseconds = _positive(measurement.get("median_ms")) if measurement.get("status") == "ok" else None
        if milliseconds is None:
            return
        self.observations.append({"features": plan_features(plan), "ms": milliseconds,
                                  "llm_raw_ms": _positive((prediction or {}).get("llm_raw_ms"))})

    def metadata(self) -> dict:
        return {"enabled": True, "llm_configured": self.model is not None,
                "llm_available": self.llm_predictions > 0, "llm_predictions": self.llm_predictions,
                "llm_unavailable_reasons": list(self.model_unavailable) or
                    (["no model provider configured"] if self.model is None else []),
                "model_requests": self.model_requests,
                "model_errors": list(self.model_errors), "fallback_predictions": self.fallback_predictions,
                "calibration_observations": len(self.observations),
                "calibration_scope": "current_query_search_only", "holdout_observations": 0,
                "features": list(_FEATURES)}
