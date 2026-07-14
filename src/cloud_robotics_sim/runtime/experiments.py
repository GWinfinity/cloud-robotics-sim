"""Experiment validation for improvement proposals using A/B testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .metrics import MetricCollector, MetricSummary
from .proposals import ImprovementProposal

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single proposal."""

    proposal: ImprovementProposal
    baseline_summary: MetricSummary
    proposal_summary: MetricSummary
    improvement: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "baseline_summary": self.baseline_summary.to_dict(),
            "proposal_summary": self.proposal_summary.to_dict(),
            "improvement": self.improvement,
            "passed": self.passed,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class ExperimentValidator:
    """Run A/B experiments to validate improvement proposals."""

    def __init__(
        self,
        make_env_fn: Callable[[dict[str, Any]], Any],
        n_episodes: int = 50,
        min_success_rate_delta: float = 0.05,
        min_reward_delta: float = 5.0,
        max_latency_regression: float = 10.0,
    ) -> None:
        self.make_env_fn = make_env_fn
        self.n_episodes = n_episodes
        self.min_success_rate_delta = min_success_rate_delta
        self.min_reward_delta = min_reward_delta
        self.max_latency_regression = max_latency_regression

    def validate(
        self,
        baseline_config: dict[str, Any],
        proposal: ImprovementProposal,
    ) -> ValidationResult:
        """Run baseline vs proposal and return a validation result."""
        logger.info(f"Validating proposal: {proposal.name}")

        baseline_env = self.make_env_fn(baseline_config)
        proposal_env = self.make_env_fn(
            self._apply_delta(baseline_config, proposal.config_delta)
        )

        collector_a = MetricCollector()
        collector_b = MetricCollector()

        baseline_summary = collector_a.collect_n_episodes(
            baseline_env, n=self.n_episodes, max_steps=1000
        )
        proposal_summary = collector_b.collect_n_episodes(
            proposal_env, n=self.n_episodes, max_steps=1000
        )

        improvement = self._compute_improvement(baseline_summary, proposal_summary)
        passed, confidence, reason = self._judge(
            baseline_summary, proposal_summary, improvement
        )

        return ValidationResult(
            proposal=proposal,
            baseline_summary=baseline_summary,
            proposal_summary=proposal_summary,
            improvement=improvement,
            passed=passed,
            confidence=confidence,
            reason=reason,
        )

    def validate_all(
        self,
        baseline_config: dict[str, Any],
        proposals: list[ImprovementProposal],
    ) -> list[ValidationResult]:
        """Validate multiple proposals and return sorted results."""
        results = []
        for proposal in proposals:
            result = self.validate(baseline_config, proposal)
            results.append(result)
        # Best first: highest confidence among passing results, then failing ones
        results.sort(key=lambda r: (r.passed, r.confidence), reverse=True)
        return results

    @staticmethod
    def _apply_delta(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
        """Recursively apply delta to base config."""
        import copy

        result = copy.deepcopy(base)
        for key, value in delta.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                result[key] = ExperimentValidator._apply_delta(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _compute_improvement(
        baseline: MetricSummary, proposal: MetricSummary
    ) -> dict[str, float]:
        return {
            "success_rate_delta": proposal.success_rate - baseline.success_rate,
            "reward_delta": proposal.mean_reward - baseline.mean_reward,
            "latency_delta_ms": proposal.mean_latency_ms - baseline.mean_latency_ms,
            "stability_delta": proposal.mean_stability_score
            - baseline.mean_stability_score,
            "physics_violations_delta": proposal.mean_physics_violations
            - baseline.mean_physics_violations,
        }

    def _judge(
        self,
        baseline: MetricSummary,
        proposal: MetricSummary,
        improvement: dict[str, float],
    ) -> tuple[bool, float, str]:
        """Decide whether a proposal is an improvement."""
        # Must not make things worse on latency
        if improvement["latency_delta_ms"] > self.max_latency_regression:
            return False, 0.0, "Latency regression too large"

        # Must improve success rate or reward meaningfully
        success_ok = improvement["success_rate_delta"] >= self.min_success_rate_delta
        reward_ok = improvement["reward_delta"] >= self.min_reward_delta
        stability_ok = improvement["stability_delta"] >= 0.0

        if not (success_ok or reward_ok):
            return False, 0.0, "No significant improvement in success rate or reward"

        if not stability_ok:
            return False, 0.0, "Stability decreased"

        # Confidence is a simple heuristic based on sample size and improvement size
        confidence = min(
            1.0,
            (
                improvement["success_rate_delta"] * 2
                + max(0.0, improvement["reward_delta"] / 20.0)
                + proposal.episodes / 100.0
            )
            / 3.0,
        )

        reason = (
            f"Success rate {baseline.success_rate:.1%} → {proposal.success_rate:.1%} "
            f"({improvement['success_rate_delta']:+.1%}), reward {baseline.mean_reward:.1f} → "
            f"{proposal.mean_reward:.1f} ({improvement['reward_delta']:+.1f})"
        )
        return True, confidence, reason
