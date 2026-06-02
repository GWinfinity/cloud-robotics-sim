"""Tests for Scene Language Predictor Plugin."""

import numpy as np
import pytest


class TestPluginMetadata:
    """Plugin metadata tests."""

    def test_version(self):
        """Version is set."""
        from plugins.predictors.scene_language import __version__
        assert __version__ == "0.1.0"

    def test_source(self):
        """Source is set."""
        from plugins.predictors.scene_language import __source__
        assert __source__ == "genesis-scene-language"

    def test_paper(self):
        """Paper reference is set."""
        from plugins.predictors.scene_language import __paper__
        assert __paper__ == "CVPR 2025"


class TestSceneGenerator:
    """Scene generator tests."""

    def test_init(self):
        """Generator initializes."""
        try:
            from plugins.predictors.scene_language.core.engine.scene_generator import SceneGenerator
            gen = SceneGenerator()
            assert gen is not None
        except ImportError:
            pytest.skip("Optional dependencies not installed")

    def test_generate_from_text(self):
        """Generate scene from text."""
        try:
            from plugins.predictors.scene_language.core.engine.scene_generator import SceneGenerator
            gen = SceneGenerator()
            scene = gen.generate("a living room with a sofa")
            assert scene is not None
        except ImportError:
            pytest.skip("Optional dependencies not installed")


class TestProgramExecutor:
    """Program executor tests."""

    def test_init(self):
        """Executor initializes."""
        try:
            from plugins.predictors.scene_language.core.engine.program_executor import ProgramExecutor
            exec = ProgramExecutor()
            assert exec is not None
        except ImportError:
            pytest.skip("Optional dependencies not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
