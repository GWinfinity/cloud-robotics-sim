"""A/B Testing Framework for Plugin Migration.

Supports comparing legacy vs plugin implementations with automatic
metric collection and migration recommendations.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Metrics collected from a single test run."""

    latency_ms: float = 0.0
    memory_mb: float = 0.0

    success: bool = True
    error_message: str = ""

    custom_metrics: dict[str, float] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)


@dataclass
class ABTestResult:
    """Aggregated A/B test results for two variants."""

    variant_a: str
    variant_b: str

    metrics_a: list[TestMetrics] = field(default_factory=list)
    metrics_b: list[TestMetrics] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Generate a summary dict of the A/B test results."""

        def _avg(metrics_list: list[TestMetrics], key: str):
            values = [
                getattr(m, key) for m in metrics_list if getattr(m, key) is not None
            ]
            return np.mean(values) if values else 0.0

        def _success_rate(metrics_list: list[TestMetrics]):
            if not metrics_list:
                return 0.0
            return sum(1 for m in metrics_list if m.success) / len(metrics_list)

        return {
            "variant_a": {
                "name": self.variant_a,
                "samples": len(self.metrics_a),
                "success_rate": _success_rate(self.metrics_a),
                "avg_latency_ms": _avg(self.metrics_a, "latency_ms"),
            },
            "variant_b": {
                "name": self.variant_b,
                "samples": len(self.metrics_b),
                "success_rate": _success_rate(self.metrics_b),
                "avg_latency_ms": _avg(self.metrics_b, "latency_ms"),
            },
            "improvement": {
                "success_rate_delta": _success_rate(self.metrics_b)
                - _success_rate(self.metrics_a),
                "latency_delta_percent": (
                    (
                        _avg(self.metrics_a, "latency_ms")
                        - _avg(self.metrics_b, "latency_ms")
                    )
                    / (_avg(self.metrics_a, "latency_ms") + 1e-6)
                    * 100
                ),
            },
        }


class ABTestRunner:
    """A/B test runner for comparing legacy vs plugin implementations.

    Automatically collects metrics, generates reports, and provides
    migration recommendations.

    Usage:
        >>> runner = ABTestRunner(
        ...     variant_a_name="legacy",
        ...     variant_a_fn=old_controller,
        ...     variant_b_name="plugin",
        ...     variant_b_fn=new_plugin_controller
        ... )
        >>>
        >>> for _ in range(100):
        ...     obs = env.reset()
        ...     runner.run_both(lambda fn: fn(obs))
        >>>
        >>> report = runner.generate_report()
        >>> print(report)
    """

    def __init__(
        self,
        variant_a_name: str,
        variant_a_fn: Callable,
        variant_b_name: str,
        variant_b_fn: Callable,
        output_dir: Optional[str] = None,
        warmup_steps: int = 10,
    ):
        """Args:
        variant_a_name: Name for variant A (legacy).
        variant_a_fn: Callable for variant A.
        variant_b_name: Name for variant B (plugin).
        variant_b_fn: Callable for variant B.
        output_dir: Report output directory.
        warmup_steps: Warmup steps excluded from statistics.
        """
        self.variant_a_name = variant_a_name
        self.variant_a_fn = variant_a_fn
        self.variant_b_name = variant_b_name
        self.variant_b_fn = variant_b_fn

        self.output_dir = Path(output_dir) if output_dir else Path("ab_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.warmup_steps = warmup_steps
        self.step_count = 0
        self._step_count_a = 0
        self._step_count_b = 0

        self.results = ABTestResult(variant_a=variant_a_name, variant_b=variant_b_name)

        self._a_warmed_up = False
        self._b_warmed_up = False

    def run_single(
        self,
        variant: str,
        test_fn: Callable,
        collect_custom_metrics: Optional[Callable] = None,
    ) -> TestMetrics:
        """Run a single test for one variant.

        Args:
            variant: 'a' or 'b'.
            test_fn: Test function that receives the variant callable.
            collect_custom_metrics: Optional function to collect custom metrics.

        Returns:
            Collected test metrics.
        """
        fn = self.variant_a_fn if variant == "a" else self.variant_b_fn

        metrics = TestMetrics()

        try:
            start_time = time.perf_counter()

            result = test_fn(fn)

            metrics.latency_ms = (time.perf_counter() - start_time) * 1000

            if collect_custom_metrics:
                metrics.custom_metrics = collect_custom_metrics(result)

            metrics.success = True

        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            if hasattr(e, "__traceback__"):
                metrics.error_message += f"\n{traceback.format_exc()}"

        # Record result (skip during warmup phase)
        self.step_count += 1
        if variant == "a":
            self._step_count_a += 1
            if self._step_count_a > self.warmup_steps:
                self.results.metrics_a.append(metrics)
        else:
            self._step_count_b += 1
            if self._step_count_b > self.warmup_steps:
                self.results.metrics_b.append(metrics)

        return metrics

    def run_both(
        self,
        test_fn: Callable,
        collect_custom_metrics: Optional[Callable] = None,
        random_order: bool = True,
    ) -> dict[str, TestMetrics]:
        """Run both A and B variants.

        Args:
            test_fn: Test function.
            collect_custom_metrics: Custom metrics collection function.
            random_order: Randomize execution order to avoid time bias.

        Returns:
            {'a': metrics_a, 'b': metrics_b}
        """
        import random

        order = ["a", "b"] if not random_order or random.random() > 0.5 else ["b", "a"]

        results = {}
        for variant in order:
            metrics = self.run_single(variant, test_fn, collect_custom_metrics)
            results[variant] = metrics

        return results

    def generate_report(self, detailed: bool = False) -> str:
        """Generate a human-readable test report."""
        summary = self.results.summary()

        lines = [
            "=" * 60,
            "A/B Test Report",
            "=" * 60,
            "",
            f"Total Steps: {self.step_count} (warmup: {self.warmup_steps})",
            "",
            "Variant A (Legacy):",
            f"  Name: {summary['variant_a']['name']}",
            f"  Samples: {summary['variant_a']['samples']}",
            f"  Success Rate: {summary['variant_a']['success_rate']:.2%}",
            f"  Avg Latency: {summary['variant_a']['avg_latency_ms']:.2f} ms",
            "",
            "Variant B (Plugin):",
            f"  Name: {summary['variant_b']['name']}",
            f"  Samples: {summary['variant_b']['samples']}",
            f"  Success Rate: {summary['variant_b']['success_rate']:.2%}",
            f"  Avg Latency: {summary['variant_b']['avg_latency_ms']:.2f} ms",
            "",
            "Improvement (B vs A):",
            f"  Success Rate Delta: {summary['improvement']['success_rate_delta']:+.2%}",
            f"  Latency Delta: {summary['improvement']['latency_delta_percent']:+.1f}%",
            "",
        ]

        if detailed:
            lines.extend(
                [
                    "Failure Analysis:",
                    "-" * 40,
                ]
            )

            failed_a = [m for m in self.results.metrics_a if not m.success]
            failed_b = [m for m in self.results.metrics_b if not m.success]

            if failed_a:
                lines.append(f"\nVariant A Failures ({len(failed_a)}):")
                for i, m in enumerate(failed_a[:5], 1):
                    lines.append(f"  {i}. {m.error_message[:100]}...")

            if failed_b:
                lines.append(f"\nVariant B Failures ({len(failed_b)}):")
                for i, m in enumerate(failed_b[:5], 1):
                    lines.append(f"  {i}. {m.error_message[:100]}...")

        lines.append("=" * 60)

        return "\n".join(lines)

    def save_report(self, filename: Optional[str] = None):
        """Save report to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ab_test_report_{timestamp}.txt"

        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            f.write(self.generate_report(detailed=True))

        json_path = report_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.results.summary(), f, indent=2)

        logger.info("Report saved to: %s", report_path)
        return report_path

    def recommend_migration(self) -> dict[str, Any]:
        """Provide a migration recommendation based on test results.

        Returns:
            {
                'recommend': bool,
                'confidence': float,
                'reason': str,
                'cautions': list[str]
            }
        """
        summary = self.results.summary()

        a_success = summary["variant_a"]["success_rate"]
        b_success = summary["variant_b"]["success_rate"]
        latency_improvement = summary["improvement"]["latency_delta_percent"]

        recommendation: dict[str, Any] = {
            "recommend": False,
            "confidence": 0.0,
            "reason": "",
            "cautions": [],
        }

        if b_success < a_success - 0.05:
            recommendation["reason"] = (
                f"Plugin success rate ({b_success:.1%}) is significantly lower than legacy ({a_success:.1%})"
            )
            recommendation["cautions"].append(
                "Investigate failure cases before migration"
            )
            return recommendation

        if summary["variant_b"]["samples"] < 100:
            recommendation["cautions"].append(
                "Sample size is small, consider more tests"
            )

        recommendation["recommend"] = True
        recommendation["confidence"] = min(b_success / (a_success + 1e-6), 1.0)

        if b_success >= a_success:
            recommendation["reason"] = (
                f"Plugin matches or exceeds legacy performance ({b_success:.1%} vs {a_success:.1%})"
            )
        else:
            recommendation["reason"] = (
                f"Plugin performance is acceptable ({b_success:.1%} vs {a_success:.1%})"
            )

        if latency_improvement < -20:
            recommendation["cautions"].append(
                f"Significant latency increase ({latency_improvement:+.1f}%), monitor performance"
            )

        return recommendation


class GradualMigration:
    """Gradual migration controller for phased traffic shifting.

    Supports incrementally shifting traffic from legacy to plugin
    implementation based on success rate thresholds.

    Usage:
        >>> migration = GradualMigration(
        ...     legacy_fn=old_controller,
        ...     plugin_fn=new_controller,
        ...     initial_plugin_ratio=0.0
        ... )
        >>>
        >>> for episode in range(1000):
        ...     fn = migration.select_implementation()
        ...     result = fn(obs)
        ...     migration.update_metrics(success=True)
        >>>
        >>> migration.increase_plugin_ratio(0.1)
    """

    def __init__(
        self,
        legacy_fn: Callable,
        plugin_fn: Callable,
        initial_plugin_ratio: float = 0.0,
        min_samples_before_increase: int = 100,
        success_threshold: float = 0.95,
    ):
        """Args:
        legacy_fn: Legacy implementation.
        plugin_fn: Plugin implementation.
        initial_plugin_ratio: Initial plugin traffic ratio (0-1).
        min_samples_before_increase: Minimum samples before increasing ratio.
        success_threshold: Success rate threshold for ratio increases.
        """
        self.legacy_fn = legacy_fn
        self.plugin_fn = plugin_fn

        self.plugin_ratio = initial_plugin_ratio
        self.min_samples = min_samples_before_increase
        self.success_threshold = success_threshold

        self.plugin_samples = 0
        self.plugin_successes = 0
        self.legacy_samples = 0
        self.legacy_successes = 0

        self.history: list[dict[str, Any]] = []

    def select_implementation(self) -> Callable:
        """Select which implementation to use based on current ratio."""
        import random

        if random.random() < self.plugin_ratio:
            return self.plugin_fn
        else:
            return self.legacy_fn

    def update_metrics(self, is_plugin: bool, success: bool):
        """Update metrics after an episode."""
        if is_plugin:
            self.plugin_samples += 1
            if success:
                self.plugin_successes += 1
        else:
            self.legacy_samples += 1
            if success:
                self.legacy_successes += 1

    def can_increase_ratio(self, increase_amount: float = 0.1) -> bool:
        """Check if plugin ratio can be increased."""
        if self.plugin_ratio >= 1.0:
            return False

        if self.plugin_samples < self.min_samples:
            return False

        plugin_success_rate = self.plugin_successes / (self.plugin_samples + 1e-6)

        if plugin_success_rate < self.success_threshold:
            return False

        return True

    def increase_plugin_ratio(self, amount: float = 0.1):
        """Increase plugin traffic ratio."""
        if not self.can_increase_ratio(amount):
            logger.warning("Cannot increase plugin ratio yet.")
            logger.warning("  Current ratio: %.1f%%", self.plugin_ratio * 100)
            logger.warning("  Plugin samples: %d", self.plugin_samples)
            logger.warning(
                "  Plugin success rate: %.1f%%",
                self.plugin_successes / (self.plugin_samples + 1e-6) * 100,
            )
            return False

        old_ratio = self.plugin_ratio
        self.plugin_ratio = min(1.0, self.plugin_ratio + amount)

        # Reset statistics
        self.plugin_samples = 0
        self.plugin_successes = 0

        logger.info(
            "Plugin ratio increased: %.1f%% -> %.1f%%",
            old_ratio * 100,
            self.plugin_ratio * 100,
        )
        return True

    def get_status(self) -> dict[str, Any]:
        """Get current migration status."""
        return {
            "plugin_ratio": self.plugin_ratio,
            "legacy_ratio": 1.0 - self.plugin_ratio,
            "plugin_stats": {
                "samples": self.plugin_samples,
                "successes": self.plugin_successes,
                "success_rate": self.plugin_successes / (self.plugin_samples + 1e-6),
            },
            "legacy_stats": {
                "samples": self.legacy_samples,
                "successes": self.legacy_successes,
                "success_rate": self.legacy_successes / (self.legacy_samples + 1e-6),
            },
        }
