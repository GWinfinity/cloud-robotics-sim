"""Tests for the improvement loop main module."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from cloud_robotics_sim.runtime.diagnostics import Diagnosis
from cloud_robotics_sim.runtime.experiments import ValidationResult
from cloud_robotics_sim.runtime.main import ImprovementLoop, LoopConfig
from cloud_robotics_sim.runtime.metrics import MetricCollector, MetricSummary
from cloud_robotics_sim.runtime.proposals import ImprovementProposal


class TestLoopConfig:
    """Tests for LoopConfig."""

    def test_to_dict_roundtrip(self):
        config = LoopConfig(
            baseline_config_path="configs/test.yaml",
            output_dir="outputs/test",
            n_baseline_episodes=5,
            thresholds={"success_rate": 0.8},
        )
        data = config.to_dict()
        assert data["baseline_config_path"] == "configs/test.yaml"
        assert data["thresholds"]["success_rate"] == pytest.approx(0.8)


class TestImprovementLoopHelpers:
    """Tests for ImprovementLoop helper methods."""

    def _make_loop(self, tmp_path, make_env_fn=None):
        config_path = tmp_path / "baseline.yaml"
        config_path.write_text(
            yaml.safe_dump({"environment": {"scene": {"type": "empty_room"}}})
        )
        loop_config = LoopConfig(
            baseline_config_path=str(config_path),
            output_dir=str(tmp_path / "out"),
            max_iterations=1,
        )
        return ImprovementLoop(
            make_env_fn=make_env_fn or (lambda cfg: None), loop_config=loop_config
        )

    def test_load_config(self, tmp_path):
        loop = self._make_loop(tmp_path)
        assert loop.baseline_config == {
            "environment": {"scene": {"type": "empty_room"}}
        }

    def test_load_config_error_non_dict(self, tmp_path):
        config_path = tmp_path / "baseline.yaml"
        config_path.write_text("- a\n- b\n")
        loop_config = LoopConfig(
            baseline_config_path=str(config_path),
            output_dir=str(tmp_path / "out"),
            max_iterations=1,
        )
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            ImprovementLoop(make_env_fn=lambda cfg: None, loop_config=loop_config)

    def test_save_config(self, tmp_path):
        loop = self._make_loop(tmp_path)
        out_path = tmp_path / "saved.yaml"
        loop._save_config({"a": {"b": 1}}, out_path)
        loaded = yaml.safe_load(out_path.read_text())
        assert loaded == {"a": {"b": 1}}

    def test_select_best_passing_first(self, tmp_path):
        loop = self._make_loop(tmp_path)
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="", config_delta={}
        )
        results = [
            ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=False,
            ),
            ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=True,
                confidence=0.9,
            ),
        ]
        best = loop._select_best(results)
        assert best is not None
        assert best.passed is True
        assert best.confidence == pytest.approx(0.9)

    def test_select_best_none_passing(self, tmp_path):
        loop = self._make_loop(tmp_path)
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="", config_delta={}
        )
        results = [
            ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=False,
                confidence=0.1,
            )
        ]
        best = loop._select_best(results)
        assert best is not None
        assert best.passed is False

    def test_select_best_empty(self, tmp_path):
        loop = self._make_loop(tmp_path)
        assert loop._select_best([]) is None

    def test_save_artifacts(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop._save_artifacts(
            iteration=1,
            config={"environment": {"scene": {"type": "empty_room"}}},
            record={
                "iteration": 1,
                "summary": MetricSummary().to_dict(),
                "diagnoses": [],
                "best_proposal": None,
                "best_result": None,
            },
        )
        iter_dir = tmp_path / "out" / "iter_1"
        assert (iter_dir / "config.yaml").exists()
        assert (iter_dir / "record.json").exists()
        assert (iter_dir / "report.md").exists()

    def test_generate_final_report(self, tmp_path):
        loop = self._make_loop(tmp_path)
        report = loop._generate_final_report({"scene": "empty_room"})
        assert report["final_config"] == {"scene": "empty_room"}
        assert report["history"] == []
        assert "knowledge_base_summary" in report


class TestImprovementLoopRun:
    """Tests for ImprovementLoop.run early exits."""

    def _make_loop(self, tmp_path, **loop_kwargs):
        config_path = tmp_path / "baseline.yaml"
        config_path.write_text(yaml.safe_dump({"scene": "empty_room"}))
        loop_config = LoopConfig(
            baseline_config_path=str(config_path),
            output_dir=str(tmp_path / "out"),
            n_baseline_episodes=2,
            n_validation_episodes=1,
            max_iterations=1,
            **loop_kwargs,
        )
        return ImprovementLoop(make_env_fn=lambda cfg: None, loop_config=loop_config)

    def test_run_early_exit_no_problem(self, tmp_path):
        thresholds = {
            "min_episodes": 2,
            "min_success_rate": 0.8,
            "max_latency_ms": 100.0,
            "max_physics_violations": 5.0,
            "min_stability_score": 0.85,
            "min_mean_reward": -50.0,
        }
        loop = self._make_loop(tmp_path, thresholds=thresholds)
        healthy_summary = MetricSummary(
            episodes=2,
            success_rate=1.0,
            mean_reward=10.0,
            mean_latency_ms=20.0,
            mean_stability_score=0.95,
        )
        with (
            patch.object(loop, "make_env_fn") as mock_make_env,
            patch.object(
                MetricCollector, "collect_n_episodes", return_value=healthy_summary
            ),
        ):
            mock_make_env.return_value = MagicMock()
            report = loop.run()
        assert report["total_iterations"] == 0
        assert "knowledge_base_summary" in report

    def test_run_early_exit_no_proposals(self, tmp_path):
        thresholds = {
            "min_episodes": 2,
            "min_success_rate": 0.8,
            "max_latency_ms": 100.0,
            "max_physics_violations": 5.0,
            "min_stability_score": 0.85,
            "min_mean_reward": -50.0,
        }
        loop = self._make_loop(tmp_path, thresholds=thresholds)
        unhealthy_summary = MetricSummary(
            episodes=2,
            success_rate=0.3,
            mean_reward=-100.0,
            mean_latency_ms=20.0,
            mean_stability_score=0.95,
        )
        with (
            patch.object(loop, "make_env_fn") as mock_make_env,
            patch.object(
                MetricCollector, "collect_n_episodes", return_value=unhealthy_summary
            ),
            patch.object(loop.proposal_generator, "generate", return_value=[]),
        ):
            mock_make_env.return_value = MagicMock()
            report = loop.run()
        assert report["total_iterations"] == 0


class TestImprovementLoopMarkdown:
    """Tests for ImprovementLoop._iteration_markdown."""

    def _make_loop(self, tmp_path):
        config_path = tmp_path / "baseline.yaml"
        config_path.write_text(yaml.safe_dump({"scene": "empty_room"}))
        loop_config = LoopConfig(
            baseline_config_path=str(config_path),
            output_dir=str(tmp_path / "out"),
            max_iterations=1,
        )
        return ImprovementLoop(make_env_fn=lambda cfg: None, loop_config=loop_config)

    def test_iteration_markdown_without_best_proposal(self, tmp_path):
        loop = self._make_loop(tmp_path)
        record = {
            "iteration": 1,
            "summary": MetricSummary(
                episodes=10,
                success_rate=0.5,
                mean_reward=0.0,
                mean_latency_ms=20.0,
                mean_physics_violations=0.0,
                mean_stability_score=0.9,
            ).to_dict(),
            "diagnoses": [
                Diagnosis(
                    category="success_rate_low",
                    severity="warning",
                    message="low",
                    metric="success_rate",
                    value=0.5,
                    threshold=0.8,
                    recommendation="tune",
                ).to_dict()
            ],
            "best_proposal": None,
            "best_result": None,
        }
        md = loop._iteration_markdown(record)
        assert "# Iteration 1" in md
        assert "No proposal adopted." in md
        assert "success_rate_low" in md

    def test_iteration_markdown_with_best_proposal(self, tmp_path):
        loop = self._make_loop(tmp_path)
        proposal = ImprovementProposal(
            name="p", description="desc", target_diagnosis="x", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=True,
            confidence=0.9,
            reason="ok",
        )
        record = {
            "iteration": 1,
            "summary": MetricSummary(episodes=10, success_rate=0.9).to_dict(),
            "diagnoses": [],
            "best_proposal": proposal.to_dict(),
            "best_result": result.to_dict(),
        }
        md = loop._iteration_markdown(record)
        assert "p" in md
        assert "desc" in md
        assert "0.90" in md
        assert "ok" in md
