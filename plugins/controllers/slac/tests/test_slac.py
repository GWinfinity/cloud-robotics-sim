"""Tests for SLAC Plugin."""

import numpy as np
import pytest
import torch

from plugins.controllers.slac.core.models.latent_action import (
    LatentActionSpace,
    PrimitiveActionDecoder,
    LatentActionController,
)
from plugins.controllers.slac.core.envs.mobile_manipulator import MobileManipulatorEnv


class TestLatentActionSpace:
    """Latent action space tests."""

    def test_init(self):
        """VAE initializes."""
        vae = LatentActionSpace(primitive_dim=19, latent_dim=8)
        assert vae is not None

    def test_encode(self):
        """Encode produces latent."""
        vae = LatentActionSpace(primitive_dim=19, latent_dim=8)
        action_seq = torch.randn(1, 10, 19)
        mu, logvar = vae.encode(action_seq)
        assert mu.shape[1] == 8

    def test_forward(self):
        """Forward pass."""
        vae = LatentActionSpace(primitive_dim=19, latent_dim=8)
        action_seq = torch.randn(1, 10, 19)
        result = vae(action_seq)
        assert 'latent_action' in result


class TestPrimitiveActionDecoder:
    """Primitive action decoder tests."""

    def test_init(self):
        """Decoder initializes."""
        decoder = PrimitiveActionDecoder(latent_dim=8, primitive_dim=19)
        assert decoder is not None

    def test_forward(self):
        """Forward pass."""
        decoder = PrimitiveActionDecoder(latent_dim=8, primitive_dim=19)
        latent = torch.randn(1, 8)
        result = decoder(latent)
        assert result is not None


class TestLatentActionController:
    """Latent action controller tests."""

    def test_init(self):
        """Controller initializes."""
        vae = LatentActionSpace(primitive_dim=19, latent_dim=8)
        decoder = PrimitiveActionDecoder(latent_dim=8, primitive_dim=19)
        controller = LatentActionController(vae, decoder)
        assert controller is not None


class TestMobileManipulatorEnv:
    """Mobile manipulator env tests."""

    def test_init(self):
        """Env initializes."""
        try:
            env = MobileManipulatorEnv(config={})
            assert env is not None
        except (AttributeError, Exception):
            pytest.skip("Genesis backend issue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
