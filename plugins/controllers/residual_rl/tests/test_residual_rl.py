"""Tests for Residual RL Plugin."""

import numpy as np
import pytest
import torch

from plugins.controllers.residual_rl.core.residual_network import ResidualNetwork, CombinedPolicy
from plugins.controllers.residual_rl.core.bc_policy import BCPolicy
from plugins.controllers.residual_rl.core.vision_encoder import ResNet18Encoder, CustomVisionEncoder


class TestResidualNetwork:
    """Residual network tests."""

    def test_init(self):
        """Network initializes."""
        net = ResidualNetwork(obs_dim=48, action_dim=19)
        assert net is not None

    def test_forward(self):
        """Forward pass produces action delta."""
        net = ResidualNetwork(obs_dim=48, action_dim=19)
        obs = torch.randn(1, 48)
        base_action = torch.randn(1, 19)
        result = net(obs, base_action)
        assert isinstance(result, dict)
        assert 'final_action' in result
        assert result['final_action'].shape == (1, 19)
        assert not torch.isnan(result['final_action']).any()


class TestCombinedPolicy:
    """Combined policy tests."""

    def test_init(self):
        """Policy initializes."""
        bc = BCPolicy(obs_dim=48, action_dim=19)
        residual = ResidualNetwork(obs_dim=48, action_dim=19)
        policy = CombinedPolicy(bc, residual)
        assert policy is not None

    def test_forward(self):
        """Combined forward pass."""
        bc = BCPolicy(obs_dim=48, action_dim=19)
        residual = ResidualNetwork(obs_dim=48, action_dim=19)
        policy = CombinedPolicy(bc, residual)
        obs = torch.randn(1, 48)
        result = policy(obs)
        assert isinstance(result, dict)
        assert 'bc_action' in result
        assert result['bc_action'].shape == (1, 19)
        assert not torch.isnan(result['bc_action']).any()


class TestBCPolicy:
    """BC policy tests."""

    def test_init(self):
        """BC policy initializes."""
        policy = BCPolicy(obs_dim=48, action_dim=19)
        assert policy is not None

    def test_forward(self):
        """Forward pass."""
        policy = BCPolicy(obs_dim=48, action_dim=19)
        obs = torch.randn(1, 48)
        result = policy(obs)
        assert isinstance(result, dict)
        assert 'action' in result
        assert result['action'].shape == (1, 19)


class TestResNet18Encoder:
    """Vision encoder tests."""

    def test_init(self):
        """Encoder initializes."""
        enc = ResNet18Encoder(output_dim=256, pretrained=False)
        assert enc is not None

    def test_forward(self):
        """Forward pass on dummy image."""
        enc = ResNet18Encoder(output_dim=256, pretrained=False)
        img = torch.randn(1, 3, 224, 224)
        feat = enc(img)
        assert feat.shape == (1, 256)
        assert not torch.isnan(feat).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
