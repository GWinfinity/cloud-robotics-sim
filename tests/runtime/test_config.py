"""Tests for runtime configuration."""

from cloud_robotics_sim.runtime.main import LoopConfig


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_default_values(self):
        """Test default loop config values."""
        config = LoopConfig(baseline_config_path="configs/test.yaml")

        assert config.baseline_config_path == "configs/test.yaml"
        assert config.output_dir == "./outputs/improvement_loop"
        assert config.n_baseline_episodes == 30
        assert config.n_validation_episodes == 50
        assert config.max_iterations == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LoopConfig(
            baseline_config_path="configs/test.yaml",
            output_dir="./outputs/test",
            n_baseline_episodes=10,
        )

        config_dict = config.to_dict()

        assert config_dict["baseline_config_path"] == "configs/test.yaml"
        assert config_dict["output_dir"] == "./outputs/test"
        assert config_dict["n_baseline_episodes"] == 10
