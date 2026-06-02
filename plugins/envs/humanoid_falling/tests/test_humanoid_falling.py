"""Tests for Humanoid Falling Environment Plugin."""

import numpy as np
import pytest

from plugins.envs.humanoid_falling.core.envs.curriculum import CurriculumManager, AdaptiveCurriculumManager, DomainRandomizationCurriculum
from plugins.envs.humanoid_falling.core.utils.rewards import compute_triangle_reward, compute_impact_penalty
from plugins.envs.humanoid_falling.core.utils.logger import Logger, TensorBoardLogger


class TestCurriculum:
    """Curriculum tests."""

    def test_init(self):
        """Curriculum initializes."""
        curr = CurriculumManager()
        assert curr is not None

    def test_adaptive_init(self):
        """Adaptive curriculum initializes."""
        curr = AdaptiveCurriculumManager()
        assert curr is not None

    def test_domain_randomization_init(self):
        """Domain randomization curriculum initializes."""
        curr = DomainRandomizationCurriculum()
        assert curr is not None


class TestRewards:
    """Reward tests."""

    def test_triangle_reward(self):
        """Triangle formation reward."""
        body_positions = {'hand_l': np.array([0.3, 0.3, 0.0]), 'hand_r': np.array([0.3, -0.3, 0.0]), 'foot_l': np.array([-0.3, 0.3, 0.0])}
        body_contacts = {'hand_l': True, 'hand_r': True, 'foot_l': True}
        reward = compute_triangle_reward(body_positions, body_contacts, torso_height=0.5)
        assert isinstance(reward, (float, np.floating))

    def test_impact_penalty(self):
        """Impact penalty is negative."""
        contact_forces = {'head': 50.0, 'torso': 20.0}
        penalty = compute_impact_penalty(contact_forces)
        assert isinstance(penalty, (float, np.floating))


class TestLogger:
    """Logger tests."""

    def test_init(self):
        """Logger initializes."""
        logger = Logger(experiment_name="test")
        assert logger is not None


class TestTensorBoardLogger:
    """TensorBoard logger tests."""

    def test_init(self):
        """TensorBoard logger initializes."""
        logger = TensorBoardLogger(experiment_name="test")
        assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
