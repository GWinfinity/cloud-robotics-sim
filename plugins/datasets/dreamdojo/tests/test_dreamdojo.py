"""Tests for DreamDojo Dataset Plugin."""

import numpy as np
import pytest

from plugins.datasets.dreamdojo.core import (
    GenesisSimulator,
    GenesisSimulatorConfig,
    GenesisRobotType,
    GenesisDataset,
    GenesisRLDataset,
    create_genesis_simulator,
    create_genesis_dataset,
)


class TestGenesisSimulatorConfig:
    """Simulator config tests."""

    def test_init(self):
        """Config initializes."""
        cfg = GenesisSimulatorConfig()
        assert cfg is not None


class TestGenesisRobotType:
    """Robot type enum tests."""

    def test_values(self):
        """Robot types defined."""
        assert GenesisRobotType.HUMANOID is not None
        assert GenesisRobotType.FRANKA is not None


class TestGenesisDataset:
    """Dataset tests."""

    def test_init(self):
        """Dataset initializes."""
        ds = GenesisDataset()
        assert ds is not None

    def test_len(self):
        """Dataset has length."""
        ds = GenesisDataset()
        assert len(ds) >= 0


class TestGenesisRLDataset:
    """RL dataset tests."""

    def test_init(self):
        """RL dataset initializes."""
        ds = GenesisRLDataset(simulator="dummy")
        assert ds is not None


class TestFactoryFunctions:
    """Factory function tests."""

    def test_create_simulator(self):
        """Simulator factory."""
        try:
            sim = create_genesis_simulator()
            assert sim is not None
        except ImportError:
            pytest.skip("Genesis not installed")

    def test_create_dataset(self):
        """Dataset factory."""
        ds = create_genesis_dataset()
        assert ds is not None
