"""Tests for environment composer."""

from cloud_robotics_sim import (
    ComposerConfig,
    EnvironmentComposer,
    EnvironmentVariantGenerator,
)


class TestComposerConfig:
    """Tests for ComposerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ComposerConfig()

        assert config.dt == 0.01
        assert config.substeps == 10
        assert config.headless is False
        assert config.resolution == (640, 480)
        assert config.num_envs == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ComposerConfig(
            dt=0.02,
            headless=True,
            resolution=(1280, 720),
        )

        assert config.dt == 0.02
        assert config.headless is True
        assert config.resolution == (1280, 720)


class TestEnvironmentComposer:
    """Tests for EnvironmentComposer."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        composer = EnvironmentComposer()
        assert composer.config is not None
        assert composer.config.dt == 0.01

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ComposerConfig(dt=0.02)
        composer = EnvironmentComposer(config)
        assert composer.config.dt == 0.02


class TestEnvironmentVariantGenerator:
    """Tests for EnvironmentVariantGenerator."""

    def test_generate_variants(self):
        """Test variant generation."""
        composer = EnvironmentComposer()
        generator = EnvironmentVariantGenerator(composer)

        variants = generator.generate_variants(
            scene_names=["scene1", "scene2"],
            robot_names=["robot1"],
            task_names=["task1", "task2"],
        )

        assert len(variants) == 4  # 2 x 1 x 2

        expected_names = [
            "scene1_robot1_task1",
            "scene1_robot1_task2",
            "scene2_robot1_task1",
            "scene2_robot1_task2",
        ]

        for variant in variants:
            assert variant["name"] in expected_names

    def test_generate_variants_with_filter(self):
        """Test variant generation with filter."""
        composer = EnvironmentComposer()
        generator = EnvironmentVariantGenerator(composer)

        def filter_fn(scene, robot, task):
            return scene == "scene1"

        variants = generator.generate_variants(
            scene_names=["scene1", "scene2"],
            robot_names=["robot1"],
            task_names=["task1"],
            filter_fn=filter_fn,
        )

        assert len(variants) == 1
        assert variants[0]["scene"] == "scene1"


class TestComposedEnvironment:
    """Tests for ComposedEnvironment."""

    def test_action_space_property(self):
        """Test action space property."""
        # This would require a full environment setup
        # Placeholder for actual implementation
        pass
