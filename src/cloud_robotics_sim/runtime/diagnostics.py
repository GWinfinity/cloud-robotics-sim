"""Diagnostic engine for identifying problems in simulation metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .metrics import MetricSummary

logger = logging.getLogger(__name__)


@dataclass
class Diagnosis:
    """A diagnosed problem in the simulation."""

    category: str
    severity: str  # "critical", "warning", "info"
    message: str
    metric: str
    value: float
    threshold: float
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
        }


class DiagnosticEngine:
    """Analyze MetricSummary and produce a list of Diagnosis objects."""

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds or self.default_thresholds()

    @staticmethod
    def default_thresholds() -> dict[str, float]:
        return {
            "min_success_rate": 0.8,
            "max_latency_ms": 100.0,
            "max_physics_violations": 5.0,
            "min_stability_score": 0.85,
            "min_mean_reward": -50.0,
            "min_episodes": 10,
        }

    def diagnose(self, summary: MetricSummary) -> list[Diagnosis]:
        """Diagnose problems from a metric summary."""
        diagnoses: list[Diagnosis] = []

        if summary.episodes < self.thresholds.get("min_episodes", 10):
            diagnoses.append(
                Diagnosis(
                    category="insufficient_data",
                    severity="warning",
                    message=f"Too few episodes ({summary.episodes}) for reliable diagnosis",
                    metric="episodes",
                    value=float(summary.episodes),
                    threshold=float(self.thresholds["min_episodes"]),
                    recommendation="Run more episodes before making decisions",
                )
            )
            return diagnoses

        if summary.success_rate < self.thresholds["min_success_rate"]:
            diagnoses.append(
                Diagnosis(
                    category="success_rate_low",
                    severity="critical" if summary.success_rate < 0.5 else "warning",
                    message=f"Success rate {summary.success_rate:.1%} below threshold {self.thresholds['min_success_rate']:.1%}",
                    metric="success_rate",
                    value=summary.success_rate,
                    threshold=self.thresholds["min_success_rate"],
                    recommendation="Improve reward shaping, tune success threshold, or increase task diversity",
                )
            )

        if summary.mean_latency_ms > self.thresholds["max_latency_ms"]:
            diagnoses.append(
                Diagnosis(
                    category="sim_slow",
                    severity="warning",
                    message=f"Mean latency {summary.mean_latency_ms:.1f}ms exceeds threshold {self.thresholds['max_latency_ms']:.1f}ms",
                    metric="mean_latency_ms",
                    value=summary.mean_latency_ms,
                    threshold=self.thresholds["max_latency_ms"],
                    recommendation="Reduce substeps, lower camera resolution, or enable vectorization",
                )
            )

        if summary.mean_physics_violations > self.thresholds["max_physics_violations"]:
            diagnoses.append(
                Diagnosis(
                    category="physics_unstable",
                    severity="critical",
                    message=f"Physics violations {summary.mean_physics_violations:.1f} exceed threshold {self.thresholds['max_physics_violations']:.1f}",
                    metric="mean_physics_violations",
                    value=summary.mean_physics_violations,
                    threshold=self.thresholds["max_physics_violations"],
                    recommendation="Increase substeps, tune joint stiffness/damping, or adjust collision shapes",
                )
            )

        if summary.mean_stability_score < self.thresholds["min_stability_score"]:
            diagnoses.append(
                Diagnosis(
                    category="physics_unstable",
                    severity="warning",
                    message=f"Stability score {summary.mean_stability_score:.2f} below threshold {self.thresholds['min_stability_score']:.2f}",
                    metric="mean_stability_score",
                    value=summary.mean_stability_score,
                    threshold=self.thresholds["min_stability_score"],
                    recommendation="Review controller gains and contact parameters",
                )
            )

        if summary.mean_reward < self.thresholds["min_mean_reward"]:
            diagnoses.append(
                Diagnosis(
                    category="reward_shaping_bad",
                    severity="warning",
                    message=f"Mean reward {summary.mean_reward:.2f} below threshold {self.thresholds['min_mean_reward']:.2f}",
                    metric="mean_reward",
                    value=summary.mean_reward,
                    threshold=self.thresholds["min_mean_reward"],
                    recommendation="Adjust reward weights, penalties, and shaping terms",
                )
            )

        if not diagnoses:
            diagnoses.append(
                Diagnosis(
                    category="healthy",
                    severity="info",
                    message="All metrics within thresholds",
                    metric="overall",
                    value=0.0,
                    threshold=0.0,
                    recommendation="No action needed; consider harder generalization tests",
                )
            )

        return diagnoses

    def worst_problem(self, summary: MetricSummary) -> Diagnosis | None:
        """Return the most severe diagnosed problem."""
        diagnoses = self.diagnose(summary)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        if not diagnoses:
            return None
        return min(diagnoses, key=lambda d: severity_order.get(d.severity, 99))
