"""Tests for BFM-Zero Predictor Plugin."""

import numpy as np
import pytest
import torch

from plugins.predictors.bfm_zero.core.models.fb_model import ForwardModel, BackwardModel
from plugins.predictors.bfm_zero.core.models.policy import LatentConditionedPolicy, PromptEncoder, MultiTaskPolicy


class TestForwardModel:
    """Forward model tests."""

    def test_init(self):
        """Model initializes."""
        model = ForwardModel(state_dim=48, action_dim=19, latent_dim=32)
        assert model is not None

    def test_forward(self):
        """Forward pass."""
        model = ForwardModel(state_dim=48, action_dim=19, latent_dim=32)
        state = torch.randn(1, 48)
        action = torch.randn(1, 19)
        latent = model(state, action)
        assert latent.shape == (1, 32)
        assert not torch.isnan(latent).any()


class TestBackwardModel:
    """Backward model tests."""

    def test_init(self):
        """Model initializes."""
        model = BackwardModel(state_dim=48, latent_dim=32)
        assert model is not None

    def test_forward(self):
        """Forward pass with reward."""
        model = BackwardModel(state_dim=48, latent_dim=32)
        goal = torch.randn(1, 48)
        reward = torch.randn(1, 1)
        latent = model(goal, reward)
        assert latent.shape == (1, 32)
        assert not torch.isnan(latent).any()


class TestLatentConditionedPolicy:
    """Latent conditioned policy tests."""

    def test_init(self):
        """Policy initializes."""
        policy = LatentConditionedPolicy(state_dim=48, action_dim=19, latent_dim=32)
        assert policy is not None

    def test_forward(self):
        """Forward pass returns dict."""
        policy = LatentConditionedPolicy(state_dim=48, action_dim=19, latent_dim=32)
        obs = torch.randn(1, 48)
        latent = torch.randn(1, 32)
        result = policy(obs, latent)
        assert isinstance(result, dict)
        assert "action" in result
        assert result["action"].shape == (1, 19)


class TestPromptEncoder:
    """Prompt encoder tests."""

    def test_init(self):
        """Encoder initializes."""
        encoder = PromptEncoder(state_dim=48, latent_dim=32)
        assert encoder is not None


class TestMultiTaskPolicy:
    """Multi-task policy tests."""

    def test_init(self):
        """Policy initializes."""
        policy = MultiTaskPolicy(state_dim=48, action_dim=19, num_tasks=3)
        assert policy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
