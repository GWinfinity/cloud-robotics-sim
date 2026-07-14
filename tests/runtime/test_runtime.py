"""Tests for runtime modules."""

import tempfile
from pathlib import Path

import pytest

from cloud_robotics_sim.runtime.diagnostics import Diagnosis, DiagnosticEngine
from cloud_robotics_sim.runtime.experiments import ValidationResult
from cloud_robotics_sim.runtime.knowledge_base import KnowledgeBase
from cloud_robotics_sim.runtime.main import LoopConfig
from cloud_robotics_sim.runtime.metrics import (
    EpisodeMetrics,
    MetricCollector,
    MetricSummary,
)
from cloud_robotics_sim.runtime.proposals import ImprovementProposal, ProposalGenerator


class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    def test_record_and_summary(self):
        """Test recording results and generating summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(path=Path(tmpdir) / "kb.json")
            proposal = ImprovementProposal(
                name="test",
                description="",
                target_diagnosis="success_rate_low",
                config_delta={},
            )
            result = ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=True,
                confidence=0.9,
                reason="ok",
            )
            kb.record(result)

            summary = kb.summary()
            assert summary["total_attempts"] == 1
            assert summary["passed"] == 1


class TestDiagnosticEngine:
    """Tests for DiagnosticEngine."""

    def _make_summary(self, success_rate: float, episodes: int = 10) -> MetricSummary:
        episodes_list = [
            EpisodeMetrics(success=i < int(success_rate * episodes))
            for i in range(episodes)
        ]
        return MetricSummary.from_episodes(episodes_list)

    def test_diagnose_healthy(self):
        """Test diagnosing a healthy summary."""
        engine = DiagnosticEngine()
        summary = self._make_summary(1.0)

        diagnoses = engine.diagnose(summary)
        assert any(d.category == "healthy" for d in diagnoses)

    def test_worst_problem(self):
        """Test finding the worst problem."""
        engine = DiagnosticEngine()
        summary = self._make_summary(0.0)

        worst = engine.worst_problem(summary)
        assert worst is not None
        assert worst.category == "success_rate_low"


class TestProposalGenerator:
    """Tests for ProposalGenerator."""

    def test_generate(self):
        """Test generating proposals."""
        generator = ProposalGenerator()
        diagnosis = Diagnosis(
            category="success_rate_low",
            severity="critical",
            message="Low success rate",
            metric="success_rate",
            value=0.0,
            threshold=0.8,
            recommendation="Adjust parameters",
        )

        proposals = generator.generate(diagnosis, {})
        assert len(proposals) > 0
        assert all(isinstance(p, ImprovementProposal) for p in proposals)


class TestMetricCollector:
    """Tests for MetricCollector."""

    def test_collect_episodes_dummy_env(self):
        """Test collecting metrics from a dummy environment."""
        collector = MetricCollector()

        class DummyEnv:
            def reset(self):
                return None, {}

            def step(self, _action):
                return None, 1.0, False, False, {}

            @property
            def action_space(self):
                return None

        summary = collector.collect_n_episodes(DummyEnv(), n=2, max_steps=3)
        assert summary.episodes == 2


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LoopConfig(
            baseline_config_path="configs/test.yaml",
            output_dir="./outputs/test",
        )

        config_dict = config.to_dict()
        assert config_dict["baseline_config_path"] == "configs/test.yaml"
        assert config_dict["output_dir"] == "./outputs/test"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        proposal = ImprovementProposal(
            name="test",
            description="",
            target_diagnosis="success_rate_low",
            config_delta={},
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=True,
            confidence=0.8,
            reason="passed",
        )

        assert result.passed is True
        assert result.confidence == pytest.approx(0.8)
