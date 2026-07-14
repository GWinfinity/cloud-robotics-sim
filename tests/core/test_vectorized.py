"""Tests for vectorized environments."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cloud_robotics_sim import (
    GenesisVectorizedEnv,
    VecEnvConfig,
    VectorizedEnvironment,
)


class TestVecEnvConfig:
    """Tests for VecEnvConfig dataclass."""

    def test_default_values(self):
        """Test default vectorized environment configuration."""
        config = VecEnvConfig()

        assert config.num_envs == 128
        assert config.num_scenes_per_env == 1
        assert config.max_parallel == 32
        assert config.use_cuda is True

    def test_custom_values(self):
        """Test custom vectorized environment configuration."""
        config = VecEnvConfig(
            num_envs=16,
            max_parallel=4,
            use_cuda=False,
        )

        assert config.num_envs == 16
        assert config.max_parallel == 4
        assert config.use_cuda is False


class TestVectorizedEnvironment:
    """Tests for VectorizedEnvironment base class."""

    def test_init(self):
        """Test base initialization."""
        config = VecEnvConfig(num_envs=64)
        vec_env = VectorizedEnvironment(config)

        assert vec_env.num_envs == 64
        assert vec_env.config == config

    def test_reset_not_implemented(self):
        """Test reset raises NotImplementedError."""
        vec_env = VectorizedEnvironment(VecEnvConfig())

        with pytest.raises(NotImplementedError):
            vec_env.reset()

    def test_step_not_implemented(self):
        """Test step raises NotImplementedError."""
        vec_env = VectorizedEnvironment(VecEnvConfig())

        with pytest.raises(NotImplementedError):
            vec_env.step(np.zeros((1, 1)))

    def test_close_noop(self):
        """Test base close is a no-op."""
        vec_env = VectorizedEnvironment(VecEnvConfig())
        vec_env.close()


class TestGenesisVectorizedEnv:
    """Tests for GenesisVectorizedEnv."""

    def test_init(self):
        """Test initialization stores factory functions."""
        config = VecEnvConfig(num_envs=8)
        scene_fn = MagicMock()
        robot_fn = MagicMock()
        task_fn = MagicMock()

        env = GenesisVectorizedEnv(
            config=config,
            scene_fn=scene_fn,
            robot_fn=robot_fn,
            task_fn=task_fn,
        )

        assert env.num_envs == 8
        assert env.scene_fn is scene_fn
        assert env.robot_fn is robot_fn
        assert env.task_fn is task_fn
        assert env._initialized is False

    def test_initialize(self):
        """Test initialize calls Genesis initialization."""
        config = VecEnvConfig(num_envs=4, use_cuda=False)
        env = GenesisVectorizedEnv(config)

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ) as mock_init:
            env.initialize()

        mock_init.assert_called_once_with(use_cuda=False)
        assert env._initialized is True

    def test_initialize_idempotent(self):
        """Test initialize is idempotent."""
        env = GenesisVectorizedEnv(VecEnvConfig())

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ) as mock_init:
            env.initialize()
            env.initialize()

        assert mock_init.call_count == 2

    def test_reset(self):
        """Test reset returns placeholder observations."""
        env = GenesisVectorizedEnv(VecEnvConfig(num_envs=4))
        env._initialized = True

        obs, infos = env.reset()

        assert obs.shape == (4, 23)
        assert len(infos) == 4
        for i, info in enumerate(infos):
            assert info["seed"] == i

    def test_reset_with_seeds(self):
        """Test reset uses provided seeds."""
        env = GenesisVectorizedEnv(VecEnvConfig(num_envs=2))
        env._initialized = True

        obs, infos = env.reset(seeds=[100, 200])

        assert infos[0]["seed"] == 100
        assert infos[1]["seed"] == 200

    def test_reset_initializes(self):
        """Test reset initializes if not already initialized."""
        env = GenesisVectorizedEnv(VecEnvConfig(num_envs=2))

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ):
            obs, infos = env.reset()

        assert env._initialized is True
        assert obs.shape == (2, 23)
        assert len(infos) == 2

    def test_step(self):
        """Test step returns placeholder batched results."""
        env = GenesisVectorizedEnv(VecEnvConfig(num_envs=4))
        env._initialized = True

        actions = np.zeros((4, 8))
        obs, rewards, terminated, truncated, infos = env.step(actions)

        assert obs.shape == (4, 23)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert len(infos) == 4

    def test_step_initializes(self):
        """Test step initializes if not already initialized."""
        env = GenesisVectorizedEnv(VecEnvConfig(num_envs=3))

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ):
            actions = np.zeros((3, 8))
            obs, rewards, terminated, truncated, infos = env.step(actions)

        assert env._initialized is True
        assert obs.shape == (3, 23)

    def test_close(self, caplog):
        """Test close logs cleanup message."""
        import logging

        env = GenesisVectorizedEnv(VecEnvConfig())
        with caplog.at_level(logging.INFO):
            env.close()

        assert "Vectorized environment closed" in caplog.text


class TestTypeAliases:
    """Tests for vectorized environment type aliases."""

    def test_aliases(self):
        """Test backward-compatibility aliases."""
        from cloud_robotics_sim.core.vectorized import GenesisVecEnv, VectorizedEnv

        assert VectorizedEnv is VectorizedEnvironment
        assert GenesisVecEnv is GenesisVectorizedEnv
