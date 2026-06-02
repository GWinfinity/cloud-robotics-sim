"""Tests for Table Tennis Environment Plugin."""

import numpy as np
import pytest
import torch

from plugins.envs.table_tennis.core.models.predictor import DualPredictor, LearnedPredictor, PhysicsPredictor
from plugins.envs.table_tennis.core.models.policy import UnifiedPolicy


class TestUnifiedPolicy:
    """Unified policy tests."""

    def test_init(self):
        """Policy initializes."""
        policy = UnifiedPolicy(obs_dim=48, action_dim=19)
        assert policy is not None

    def test_forward(self):
        """Forward pass returns dict."""
        policy = UnifiedPolicy(obs_dim=48, action_dim=19)
        obs = torch.randn(1, 48)
        result = policy(obs)
        assert isinstance(result, dict)
        assert "action" in result
        assert result["action"].shape == (1, 19)
        assert not torch.isnan(result["action"]).any()


class TestDualPredictor:
    """Dual predictor tests."""

    def test_init(self):
        """Predictor initializes."""
        lp = LearnedPredictor()
        pp = PhysicsPredictor(ball_config={})
        dp = DualPredictor(learned_predictor=lp, physics_predictor=pp)
        assert dp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
