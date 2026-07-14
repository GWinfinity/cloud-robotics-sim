"""Tests for runtime improvement proposal generation."""

import pytest

from cloud_robotics_sim.runtime.diagnostics import Diagnosis
from cloud_robotics_sim.runtime.proposals import (
    ImprovementProposal,
    ProposalGenerator,
    deep_update,
)


class TestImprovementProposal:
    """Tests for ImprovementProposal dataclass."""

    def test_to_dict(self):
        proposal = ImprovementProposal(
            name="p",
            description="desc",
            target_diagnosis="x",
            config_delta={"a": 1},
            expected_improvement={"reward": 2.0},
            priority=3,
        )
        d = proposal.to_dict()
        assert d["name"] == "p"
        assert d["description"] == "desc"
        assert d["target_diagnosis"] == "x"
        assert d["config_delta"] == {"a": 1}
        assert d["expected_improvement"] == {"reward": pytest.approx(2.0)}
        assert d["priority"] == 3


class TestDeepUpdate:
    """Tests for deep_update helper."""

    def test_nested_update(self):
        base = {"environment": {"simulation": {"dt": 0.01, "substeps": 10}}}
        update = {"environment": {"simulation": {"dt": 0.005}}}
        result = deep_update(base, update)
        assert result == {"environment": {"simulation": {"dt": 0.005, "substeps": 10}}}
        assert base["environment"]["simulation"]["dt"] == pytest.approx(0.01)

    def test_add_new_key(self):
        base = {"a": 1}
        update = {"b": 2}
        result = deep_update(base, update)
        assert result == {"a": 1, "b": 2}

    def test_replace_dict_with_scalar(self):
        base = {"environment": {"simulation": {"dt": 0.01}}}
        update = {"environment": "new"}
        result = deep_update(base, update)
        assert result == {"environment": "new"}


class TestProposalGeneratorGenerate:
    """Tests for ProposalGenerator.generate across categories."""

    def _make_diagnosis(self, category):
        return Diagnosis(
            category=category,
            severity="warning",
            message="msg",
            metric="m",
            value=0.0,
            threshold=1.0,
            recommendation="r",
        )

    def test_generate_success_rate_low(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("success_rate_low"), {})
        assert len(proposals) == 3
        assert proposals[0].name == "looser_success_threshold"

    def test_generate_physics_unstable(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("physics_unstable"), {})
        assert len(proposals) == 3
        assert proposals[0].name == "more_substeps"

    def test_generate_sim_slow(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("sim_slow"), {})
        assert len(proposals) == 2
        assert proposals[0].name == "fewer_substeps"

    def test_generate_reward_shaping_bad(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("reward_shaping_bad"), {})
        assert len(proposals) == 2
        assert proposals[0].name == "reduce_step_penalty"

    def test_generate_healthy(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("healthy"), {})
        assert len(proposals) == 1
        assert proposals[0].name == "randomize_targets"

    def test_generate_fallback(self):
        generator = ProposalGenerator()
        proposals = generator.generate(self._make_diagnosis("unknown"), {})
        assert len(proposals) == 1
        assert proposals[0].name == "tune_default"


class TestProposalGeneratorApply:
    """Tests for ProposalGenerator.apply."""

    def test_apply(self):
        generator = ProposalGenerator()
        baseline = {"environment": {"simulation": {"dt": 0.01}}}
        proposal = ImprovementProposal(
            name="p",
            description="",
            target_diagnosis="x",
            config_delta={"environment": {"simulation": {"dt": 0.005}}},
        )
        result = generator.apply(baseline, proposal)
        assert result["environment"]["simulation"]["dt"] == pytest.approx(0.005)
        assert baseline["environment"]["simulation"]["dt"] == pytest.approx(0.01)
