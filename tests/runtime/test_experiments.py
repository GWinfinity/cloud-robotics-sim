"""Tests for runtime experiments validation."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.runtime.experiments import ExperimentValidator, ValidationResult
from cloud_robotics_sim.runtime.metrics import MetricSummary
from cloud_robotics_sim.runtime.proposals import ImprovementProposal


class DummyEnv:
    """Minimal environment returning deterministic transition data."""

    def __init__(self, action_space=None):
        self.action_space = action_space
        self._step = 0

    def reset(self):
        self._step = 0
        return None, {}

    def step(self, _action):
        self._step += 1
        return None, 1.0, self._step >= 2, False, {}


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_to_dict(self):
        proposal = ImprovementProposal(
            name="test", description="", target_diagnosis="x", config_delta={"a": 1}
        )
        baseline = MetricSummary(episodes=5, success_rate=0.5, mean_reward=10.0)
        proposal_summary = MetricSummary(episodes=5, success_rate=0.7, mean_reward=20.0)
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=baseline,
            proposal_summary=proposal_summary,
            improvement={"reward_delta": 10.0},
            passed=True,
            confidence=0.8,
            reason="ok",
        )
        d = result.to_dict()
        assert d["proposal"]["name"] == "test"
        assert d["baseline_summary"]["episodes"] == 5
        assert d["proposal_summary"]["episodes"] == 5
        assert d["improvement"]["reward_delta"] == pytest.approx(10.0)
        assert d["passed"] is True
        assert d["confidence"] == pytest.approx(0.8)
        assert d["reason"] == "ok"


class TestExperimentValidatorApplyDelta:
    """Tests for ExperimentValidator._apply_delta."""

    def test_flat_delta(self):
        base = {"a": 1, "b": 2}
        delta = {"b": 3, "c": 4}
        result = ExperimentValidator._apply_delta(base, delta)
        assert result == {"a": 1, "b": 3, "c": 4}
        assert base == {"a": 1, "b": 2}

    def test_nested_delta(self):
        base = {"environment": {"simulation": {"dt": 0.01, "substeps": 10}}}
        delta = {"environment": {"simulation": {"dt": 0.005}}}
        result = ExperimentValidator._apply_delta(base, delta)
        assert result == {"environment": {"simulation": {"dt": 0.005, "substeps": 10}}}

    def test_replace_nested_with_scalar(self):
        base = {"environment": {"simulation": {"dt": 0.01}}}
        delta = {"environment": "replaced"}
        result = ExperimentValidator._apply_delta(base, delta)
        assert result == {"environment": "replaced"}


class TestExperimentValidatorComputeImprovement:
    """Tests for ExperimentValidator._compute_improvement."""

    def test_compute_improvement(self):
        baseline = MetricSummary(
            success_rate=0.5,
            mean_reward=10.0,
            mean_latency_ms=20.0,
            mean_stability_score=0.8,
            mean_physics_violations=3.0,
        )
        proposal = MetricSummary(
            success_rate=0.7,
            mean_reward=25.0,
            mean_latency_ms=22.0,
            mean_stability_score=0.85,
            mean_physics_violations=1.0,
        )
        improvement = ExperimentValidator._compute_improvement(baseline, proposal)
        assert improvement["success_rate_delta"] == pytest.approx(0.2)
        assert improvement["reward_delta"] == pytest.approx(15.0)
        assert improvement["latency_delta_ms"] == pytest.approx(2.0)
        assert improvement["stability_delta"] == pytest.approx(0.05)
        assert improvement["physics_violations_delta"] == pytest.approx(-2.0)


class TestExperimentValidatorJudge:
    """Tests for ExperimentValidator._judge."""

    def _make_summary(self, **kwargs):
        defaults = {
            "episodes": 10,
            "success_rate": 0.5,
            "mean_reward": 10.0,
            "mean_latency_ms": 20.0,
            "mean_length": 10.0,
            "mean_physics_violations": 0.0,
            "mean_stability_score": 0.9,
        }
        defaults.update(kwargs)
        return MetricSummary(**defaults)

    def test_pass_success_rate(self):
        validator = ExperimentValidator(make_env_fn=None)
        baseline = self._make_summary(success_rate=0.5)
        proposal = self._make_summary(success_rate=0.6, episodes=50)
        improvement = ExperimentValidator._compute_improvement(baseline, proposal)
        passed, confidence, reason = validator._judge(baseline, proposal, improvement)
        assert passed is True
        assert confidence > 0.0
        assert "Success rate" in reason

    def test_pass_reward(self):
        validator = ExperimentValidator(make_env_fn=None)
        baseline = self._make_summary(mean_reward=10.0)
        proposal = self._make_summary(mean_reward=20.0, episodes=50)
        improvement = ExperimentValidator._compute_improvement(baseline, proposal)
        passed, confidence, reason = validator._judge(baseline, proposal, improvement)
        assert passed is True
        assert confidence > 0.0
        assert "reward" in reason

    @pytest.mark.parametrize(
        "baseline_kwargs, proposal_kwargs, validator_kwargs, expected_fragment",
        [
            pytest.param(
                {"mean_latency_ms": 10.0},
                {"mean_latency_ms": 20.0},
                {"max_latency_regression": 5.0},
                "Latency regression",
                id="latency_regression",
            ),
            pytest.param(
                {"success_rate": 0.5, "mean_reward": 10.0},
                {"success_rate": 0.51, "mean_reward": 11.0},
                {},
                "No significant improvement",
                id="no_improvement",
            ),
            pytest.param(
                {"mean_stability_score": 0.9},
                {"success_rate": 0.9, "mean_reward": 20.0, "mean_stability_score": 0.8},
                {},
                "Stability decreased",
                id="stability_decreased",
            ),
        ],
    )
    def test_fail_cases(
        self, baseline_kwargs, proposal_kwargs, validator_kwargs, expected_fragment
    ):
        validator = ExperimentValidator(make_env_fn=None, **validator_kwargs)
        baseline = self._make_summary(**baseline_kwargs)
        proposal = self._make_summary(**proposal_kwargs)
        improvement = ExperimentValidator._compute_improvement(baseline, proposal)
        passed, confidence, reason = validator._judge(baseline, proposal, improvement)
        assert passed is False
        assert confidence == pytest.approx(0.0)
        assert expected_fragment in reason


class TestExperimentValidatorValidate:
    """Tests for ExperimentValidator.validate and validate_all."""

    def _make_factory(self):
        def make_env_fn(_config):
            return DummyEnv()

        return make_env_fn

    def test_validate(self):
        validator = ExperimentValidator(make_env_fn=self._make_factory(), n_episodes=2)
        baseline_config = {"environment": {"task": {"success_threshold": 0.05}}}
        proposal = ImprovementProposal(
            name="better",
            description="",
            target_diagnosis="success_rate_low",
            config_delta={"environment": {"task": {"success_threshold": 0.1}}},
        )
        result = validator.validate(baseline_config, proposal)
        assert isinstance(result, ValidationResult)
        assert result.proposal.name == "better"
        assert result.baseline_summary.episodes == 2
        assert result.proposal_summary.episodes == 2

    def test_validate_all_sorting(self):
        validator = ExperimentValidator(make_env_fn=self._make_factory(), n_episodes=1)
        baseline_config = {}

        class DeterministicEnv:
            def __init__(self, reward_bonus=0.0):
                self.action_space = None
                self.reward_bonus = reward_bonus
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return None, {}

            def step(self, _action):
                self.step_count += 1
                return None, 1.0 + self.reward_bonus, True, False, {}

        def make_env_fn(config):
            return DeterministicEnv(reward_bonus=config.get("bonus", 0.0))

        validator.make_env_fn = make_env_fn
        proposals = [
            ImprovementProposal(
                name="low",
                description="",
                target_diagnosis="success_rate_low",
                config_delta={"bonus": 0.0},
            ),
            ImprovementProposal(
                name="high",
                description="",
                target_diagnosis="success_rate_low",
                config_delta={"bonus": 100.0},
            ),
        ]
        results = validator.validate_all(baseline_config, proposals)
        assert len(results) == 2
        assert results[0].proposal.name == "high"
        assert results[0].passed is True
