"""Tests for the A/B testing framework."""

from cloud_robotics_sim.core.ab_test_framework import (
    ABTestResult,
    ABTestRunner,
    GradualMigration,
    TestMetrics,
)


class TestTestMetrics:
    """Tests for TestMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = TestMetrics()

        assert metrics.latency_ms == 0.0
        assert metrics.memory_mb == 0.0
        assert metrics.success is True


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_summary(self):
        """Test summary generation."""
        metrics_a = [TestMetrics(latency_ms=10.0, success=True)]
        metrics_b = [TestMetrics(latency_ms=12.0, success=True)]

        result = ABTestResult(
            variant_a="legacy",
            variant_b="plugin",
            metrics_a=metrics_a,
            metrics_b=metrics_b,
        )

        summary = result.summary()
        assert summary["variant_a"]["samples"] == 1
        assert summary["variant_b"]["samples"] == 1
        assert summary["variant_a"]["success_rate"] == 1.0


class TestABTestRunner:
    """Tests for ABTestRunner."""

    def test_run_single(self):
        """Test running a single variant."""
        runner = ABTestRunner(
            variant_a_name="legacy",
            variant_a_fn=lambda x: x * 2,
            variant_b_name="plugin",
            variant_b_fn=lambda x: x * 3,
        )

        metrics = runner.run_single(
            variant="a",
            test_fn=lambda fn: fn(5) == 10,
        )

        assert isinstance(metrics, TestMetrics)
        assert metrics.success is True

    def test_generate_report(self):
        """Test report generation."""
        runner = ABTestRunner(
            variant_a_name="legacy",
            variant_a_fn=lambda x: x,
            variant_b_name="plugin",
            variant_b_fn=lambda x: x,
        )

        runner.run_both(lambda fn: fn(1) == 1)
        report = runner.generate_report()

        assert "A/B Test Report" in report
        assert "legacy" in report
        assert "plugin" in report


class TestGradualMigration:
    """Tests for GradualMigration."""

    def test_select_implementation(self):
        """Test implementation selection."""
        migration = GradualMigration(
            legacy_fn=lambda: "legacy",
            plugin_fn=lambda: "plugin",
            initial_plugin_ratio=0.0,
        )

        # With ratio 0, should always select legacy
        assert migration.select_implementation()() == "legacy"

    def test_update_metrics(self):
        """Test metric updates."""
        migration = GradualMigration(
            legacy_fn=lambda: "legacy",
            plugin_fn=lambda: "plugin",
        )

        migration.update_metrics(is_plugin=True, success=True)
        assert migration.plugin_samples == 1
        assert migration.plugin_successes == 1

    def test_can_increase_ratio(self):
        """Test ratio increase check."""
        migration = GradualMigration(
            legacy_fn=lambda: "legacy",
            plugin_fn=lambda: "plugin",
            initial_plugin_ratio=1.0,
        )

        assert migration.can_increase_ratio() is False
