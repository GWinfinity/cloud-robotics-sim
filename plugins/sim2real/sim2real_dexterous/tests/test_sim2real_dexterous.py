"""Tests for Sim2Real Dexterous Plugin."""

import numpy as np
import pytest
import torch

from plugins.sim2real.sim2real_dexterous.core.algorithms.real2sim_tuning import Real2SimTuner
from plugins.sim2real.sim2real_dexterous.core.models.policy_distillation import PolicyDistillation
from plugins.sim2real.sim2real_dexterous.core.models.object_representation import HybridObjectRepresentation
from plugins.sim2real.sim2real_dexterous.core.models.reward_function import GeneralizedRewardFunction


class TestReal2SimTuner:
    """Real-to-Sim tuner tests."""

    def test_init(self):
        """Tuner initializes with dummy args."""
        tuner = Real2SimTuner(env=None, initial_params={})
        assert tuner is not None


class TestPolicyDistillation:
    """Policy distillation tests."""

    def test_init(self):
        """Distillation initializes with dummy args."""
        student = torch.nn.Linear(48, 19)
        pd = PolicyDistillation(expert_policies=[], student_policy=student)
        assert pd is not None


class TestHybridObjectRepresentation:
    """Object representation tests."""

    def test_init(self):
        """Representation initializes with config."""
        config = {
            'visual': {'enabled': False},
            'point_cloud': {'enabled': False},
            'proprio': {'enabled': True, 'dim': 19},
            'geometric': {'enabled': False},
            'fusion_method': 'concat'
        }
        rep = HybridObjectRepresentation(config=config)
        assert rep is not None


class TestGeneralizedRewardFunction:
    """Reward function tests."""

    def test_init(self):
        """Reward function initializes."""
        rf = GeneralizedRewardFunction(config={})
        assert rf is not None

    def test_compute_reward(self):
        """Reward computes."""
        rf = GeneralizedRewardFunction(config={})
        state = {
            'left_palm_contact_force': 1.0,
            'left_finger_contact_forces': [0.5]*5,
            'right_palm_contact_force': 0.0,
            'right_finger_contact_forces': [0.0]*5,
        }
        reward = rf.compute_reward(state, np.zeros(10), task_type='grasp_and_reach')
        assert isinstance(reward, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
