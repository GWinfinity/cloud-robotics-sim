"""Tests for WBM Embrace Plugin."""

import numpy as np
import pytest
import torch

from plugins.controllers.wbm_embrace.core.envs.bulky_objects import BulkyObjectGenerator
from plugins.controllers.wbm_embrace.core.models.motion_prior import MotionPriorVAE
from plugins.controllers.wbm_embrace.core.models.teacher_student import TeacherPolicy


class TestBulkyObjectGenerator:
    """Bulky object generator tests."""

    def test_init(self):
        """Generator initializes."""
        gen = BulkyObjectGenerator()
        assert gen is not None

    def test_generate_box(self):
        """Generate a box object."""
        gen = BulkyObjectGenerator()
        try:
            obj = gen.generate_box(size=(0.5, 0.3, 0.2), position=np.array([0, 0, 0.1]), mass=5.0)
            assert obj is not None
        except AttributeError:
            pytest.skip("Requires Genesis scene")


class TestMotionPriorVAE:
    """Motion prior tests."""

    def test_init(self):
        """Motion prior initializes."""
        mp = MotionPriorVAE()
        assert mp is not None

    def test_forward(self):
        """Can forward pass."""
        mp = MotionPriorVAE()
        pose = torch.randn(1, 69)
        condition = torch.randn(1, 64)
        result = mp(pose, condition)
        assert isinstance(result, dict)
        assert 'recon_pose' in result


class TestTeacherPolicy:
    """Teacher policy tests."""

    def test_init(self):
        """Policy initializes."""
        policy = TeacherPolicy(state_dim=48, action_dim=19)
        assert policy is not None

    def test_forward(self):
        """Forward pass."""
        policy = TeacherPolicy(state_dim=48, action_dim=19)
        state = torch.randn(1, 48)
        nsdf = torch.randn(1, 16)
        contact = torch.randn(1, 8, 3)
        result = policy(state, nsdf, contact)
        assert 'action' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
