"""Tests for runtime diagnostics engine."""

import pytest

from cloud_robotics_sim.runtime.diagnostics import Diagnosis, DiagnosticEngine
from cloud_robotics_sim.runtime.metrics import MetricSummary


class TestDiagnosis:
    """Tests for Diagnosis dataclass."""

    def test_to_dict(self):
        diagnosis = Diagnosis(
            category="success_rate_low",
            severity="critical",
            message="bad",
            metric="success_rate",
            value=0.1,
            threshold=0.8,
            recommendation="fix it",
        )
        d = diagnosis.to_dict()
        assert d == {
            "category": "success_rate_low",
            "severity": "critical",
            "message": "bad",
            "metric": "success_rate",
            "value": pytest.approx(0.1),
            "threshold": pytest.approx(0.8),
            "recommendation": "fix it",
        }


class TestDiagnosticEngineThresholds:
    """Tests for DiagnosticEngine threshold handling."""

    def test_default_thresholds(self):
        thresholds = DiagnosticEngine.default_thresholds()
        assert thresholds["min_success_rate"] == pytest.approx(0.8)
        assert thresholds["max_latency_ms"] == pytest.approx(100.0)
        assert thresholds["max_physics_violations"] == pytest.approx(5.0)
        assert thresholds["min_stability_score"] == pytest.approx(0.85)
        assert thresholds["min_mean_reward"] == pytest.approx(-50.0)
        assert thresholds["min_episodes"] == pytest.approx(10.0)

    def test_custom_thresholds(self):
        engine = DiagnosticEngine(
            thresholds={"min_episodes": 5, "min_success_rate": 0.9}
        )
        assert engine.thresholds["min_episodes"] == pytest.approx(5)
        assert engine.thresholds["min_success_rate"] == pytest.approx(0.9)


class TestDiagnosticEngineDiagnose:
    """Tests for DiagnosticEngine.diagnose branches."""

    def _make_summary(self, **kwargs) -> MetricSummary:
        defaults = {
            "episodes": 20,
            "success_rate": 0.9,
            "mean_reward": 0.0,
            "mean_latency_ms": 50.0,
            "mean_length": 10.0,
            "mean_physics_violations": 0.0,
            "mean_stability_score": 0.95,
        }
        defaults.update(kwargs)
        return MetricSummary(**defaults)

    def test_insufficient_data(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(episodes=3)
        diagnoses = engine.diagnose(summary)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == "insufficient_data"
        assert diagnoses[0].severity == "warning"

    def test_healthy(self):
        engine = DiagnosticEngine()
        summary = self._make_summary()
        diagnoses = engine.diagnose(summary)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == "healthy"
        assert diagnoses[0].severity == "info"

    def test_success_rate_low_warning(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(success_rate=0.6)
        diagnoses = engine.diagnose(summary)
        assert any(
            d.category == "success_rate_low" and d.severity == "warning"
            for d in diagnoses
        )

    def test_success_rate_low_critical(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(success_rate=0.3)
        diagnoses = engine.diagnose(summary)
        assert any(
            d.category == "success_rate_low" and d.severity == "critical"
            for d in diagnoses
        )

    def test_sim_slow(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(mean_latency_ms=150.0)
        diagnoses = engine.diagnose(summary)
        assert any(d.category == "sim_slow" for d in diagnoses)

    def test_physics_unstable_critical(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(mean_physics_violations=10.0)
        diagnoses = engine.diagnose(summary)
        assert any(
            d.category == "physics_unstable" and d.severity == "critical"
            for d in diagnoses
        )

    def test_physics_unstable_warning(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(mean_stability_score=0.7)
        diagnoses = engine.diagnose(summary)
        assert any(
            d.category == "physics_unstable" and d.severity == "warning"
            for d in diagnoses
        )

    def test_reward_shaping_bad(self):
        engine = DiagnosticEngine()
        summary = self._make_summary(mean_reward=-100.0)
        diagnoses = engine.diagnose(summary)
        assert any(d.category == "reward_shaping_bad" for d in diagnoses)


class TestDiagnosticEngineWorstProblem:
    """Tests for DiagnosticEngine.worst_problem ordering."""

    def test_worst_problem_critical_over_warning(self):
        engine = DiagnosticEngine()
        summary = MetricSummary(
            episodes=20,
            success_rate=0.3,
            mean_latency_ms=150.0,
            mean_reward=0.0,
            mean_physics_violations=10.0,
            mean_stability_score=0.95,
        )
        worst = engine.worst_problem(summary)
        assert worst is not None
        assert worst.severity == "critical"

    def test_worst_problem_warning_over_info(self):
        engine = DiagnosticEngine()
        summary = MetricSummary(
            episodes=20,
            success_rate=0.6,
            mean_latency_ms=150.0,
            mean_reward=0.0,
            mean_physics_violations=0.0,
            mean_stability_score=0.95,
        )
        worst = engine.worst_problem(summary)
        assert worst is not None
        assert worst.severity == "warning"

    def test_worst_problem_healthy(self):
        engine = DiagnosticEngine()
        summary = MetricSummary(
            episodes=20, success_rate=0.9, mean_stability_score=0.95
        )
        worst = engine.worst_problem(summary)
        assert worst is not None
        assert worst.category == "healthy"

    def test_worst_problem_insufficient_data_returns_early(self):
        engine = DiagnosticEngine()
        summary = MetricSummary(episodes=3)
        worst = engine.worst_problem(summary)
        assert worst is not None
        assert worst.category == "insufficient_data"
