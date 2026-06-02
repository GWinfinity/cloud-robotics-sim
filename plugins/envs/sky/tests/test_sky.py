"""Tests for Sky (Genesis Engine) Plugin."""

import numpy as np
import pytest


class TestGenesisImport:
    """Genesis import tests."""

    def test_import_scene(self):
        """Scene class importable."""
        try:
            from plugins.envs.sky.core.genesis import Scene
            assert Scene is not None
        except ImportError:
            pytest.skip("Genesis not installed")

    def test_import_entity(self):
        """Entity class importable."""
        try:
            from plugins.envs.sky.core.genesis import Entity
            assert Entity is not None
        except ImportError:
            pytest.skip("Genesis not installed")

    def test_import_solvers(self):
        """Solvers importable."""
        try:
            from plugins.envs.sky.core.genesis import RigidSolver, MPMSolver
            assert RigidSolver is not None
            assert MPMSolver is not None
        except ImportError:
            pytest.skip("Genesis not installed")


class TestPluginMetadata:
    """Plugin metadata tests."""

    def test_version(self):
        """Version is set."""
        from plugins.envs.sky import __version__
        assert __version__ == "0.3.11"

    def test_source(self):
        """Source is set."""
        from plugins.envs.sky import __source__
        assert __source__ == "genesis-sky"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
