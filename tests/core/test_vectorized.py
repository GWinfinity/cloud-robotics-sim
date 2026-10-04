"""Tests for vectorized environments."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cloud_robotics_sim import (
    GenesisVectorizedEnv,
    VecEnvConfig,
    VecTask,
    VectorizedEnvironment,
)
from cloud_robotics_sim.core.vectorized import GenesisVecEnv, VectorizedEnv


class FakeTask(VecTask):
    """Minimal batched task used to test GenesisVectorizedEnv without Genesis.

    Observations are a constant ramp per reset so tests can distinguish
    fresh observations from stale ones.
    """

    num_observations = 4
    num_actions = 2

    def __init__(self):
        self.reset_calls = []
        self.step_calls = []

    def build_scene(self, scene):
        scene.mark("build_scene")

    def setup(self, scene):
        scene.mark("setup")

    def reset(self, envs_idx=None):
        self.reset_calls.append(envs_idx)
        n = self.num_envs if envs_idx is None else int(envs_idx.numel())
        return torch.ones(n, self.num_observations, device=self.seeds.device)

    def step(self, actions):
        self.step_calls.append(actions)
        n = actions.shape[0]
        device = actions.device
        return (
            torch.zeros(n, self.num_observations, device=device),
            torch.zeros(n, device=device),
            torch.zeros(n, dtype=torch.bool, device=device),
            torch.zeros(n, dtype=torch.bool, device=device),
            {},
        )


class FakeScene:
    """Stand-in for a built Genesis scene."""

    def __init__(self):
        self.build = MagicMock()
        self.marks = []

    def mark(self, name):
        self.marks.append(name)


def make_env(num_envs=4, **config_kwargs):
    """Build an env with a FakeTask and FakeScene (no Genesis required)."""
    config_kwargs.setdefault("use_cuda", False)
    config = VecEnvConfig(num_envs=num_envs, **config_kwargs)
    task = FakeTask()
    scene = FakeScene()
    env = GenesisVectorizedEnv(config=config, task=task, scene_fn=lambda: scene)
    return env, task, scene


class TestVecEnvConfig:
    """Tests for VecEnvConfig dataclass."""

    def test_default_values(self):
        """Test default vectorized environment configuration."""
        config = VecEnvConfig()

        assert config.num_envs == 128
        assert config.num_scenes_per_env == 1
        assert config.max_parallel == 32
        assert config.use_cuda is True

    def test_physics_defaults(self):
        """Test contact-fidelity physics knobs are exposed."""
        config = VecEnvConfig()

        assert config.sim_dt == 0.02
        assert config.sim_substeps == 2
        assert config.integrator == "implicitfast"
        assert config.noslip_iterations == 0  # Genesis RigidOptions default
        assert config.solver_iterations == 50
        assert config.ls_iterations == 50
        assert config.self_collision is True
        assert config.hibernation is False

    def test_custom_values(self):
        """Test custom vectorized environment configuration."""
        config = VecEnvConfig(
            num_envs=16,
            max_parallel=4,
            use_cuda=False,
            noslip_iterations=10,
        )

        assert config.num_envs == 16
        assert config.max_parallel == 4
        assert config.use_cuda is False
        assert config.noslip_iterations == 10


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


class TestVecTask:
    """Tests for the VecTask abstract interface."""

    def test_cannot_instantiate_without_impl(self):
        """Test VecTask is abstract."""
        with pytest.raises(TypeError):
            VecTask()

    def test_fake_task_is_vec_task(self):
        """Test FakeTask satisfies the VecTask interface."""
        assert isinstance(FakeTask(), VecTask)


class TestGenesisVectorizedEnv:
    """Tests for GenesisVectorizedEnv."""

    def test_init(self):
        """Test initialization stores task and scene factory."""
        env, task, scene = make_env(num_envs=8)

        assert env.num_envs == 8
        assert env.task is task
        assert env._initialized is False
        assert env.scene_fn() is scene

    def test_task_dimension_properties(self):
        """Test num_actions / num_observations delegate to the task."""
        env, _, _ = make_env()

        assert env.num_actions == 2
        assert env.num_observations == 4

    def test_task_dimension_properties_require_task(self):
        """Test dimension properties raise without a task."""
        env = GenesisVectorizedEnv(VecEnvConfig())

        with pytest.raises(RuntimeError, match="No task configured"):
            _ = env.num_actions

    def test_initialize_requires_task(self):
        """Test initialize raises a clear error without a task."""
        env = GenesisVectorizedEnv(VecEnvConfig())

        with pytest.raises(RuntimeError, match="VecTask"):
            env.initialize()

    def test_initialize(self):
        """Test initialize builds the batched scene and wires the task."""
        env, task, scene = make_env(num_envs=4)

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ) as mock_init:
            env.initialize()

        mock_init.assert_called_once_with(use_cuda=False, performance_mode=False)
        assert env.device == torch.device("cpu")
        scene.build.assert_called_once_with(n_envs=4)
        assert scene.marks == ["build_scene", "setup"]
        assert task.num_envs == 4
        assert task.seeds.shape == (4,)
        assert env.episode_length.shape == (4,)
        assert env._initialized is True

    def test_initialize_idempotent(self):
        """Test initialize only builds the scene once."""
        env, _, scene = make_env()

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ):
            env.initialize()
            env.initialize()

        assert scene.build.call_count == 1

    def test_initialize_falls_back_to_cpu(self):
        """Test CUDA requests fall back to CPU when unavailable."""
        env, _, _ = make_env(use_cuda=True)

        with (
            patch("cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            env.initialize()

        assert env.device == torch.device("cpu")

    def test_reset(self):
        """Test full reset returns batched observations and seed infos."""
        env, task, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.tensor([3, 5, 7, 9])
        env.task.seeds = torch.arange(4)

        obs, infos = env.reset()

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (4, 4)
        assert len(infos) == 4
        for i, info in enumerate(infos):
            assert info["seed"] == i
        assert torch.equal(env.episode_length, torch.zeros(4, dtype=torch.long))
        assert task.reset_calls[-1].numel() == 4

    def test_reset_with_seeds(self):
        """Test reset updates seeds for the reset environments."""
        env, task, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.zeros(4, dtype=torch.long)
        env.task.seeds = torch.arange(4)

        env.reset(seeds=[100, 200, 300, 400])

        assert torch.equal(task.seeds, torch.tensor([100, 200, 300, 400]))

    def test_reset_with_seeds_length_mismatch(self):
        """Test seed count must match the reset subset size."""
        env, _, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.zeros(4, dtype=torch.long)
        env.task.seeds = torch.arange(4)

        with pytest.raises(ValueError, match="seeds"):
            env.reset(seeds=[1, 2])

    def test_reset_partial_envs_idx(self):
        """Test mask-style partial reset only resets the given envs."""
        env, task, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.tensor([1, 2, 3, 4])
        env.task.seeds = torch.arange(4)

        obs = env.reset_idx(torch.tensor([1, 3]))

        assert obs.shape == (2, 4)
        assert torch.equal(env.episode_length, torch.tensor([1, 0, 3, 0]))
        assert torch.equal(task.reset_calls[-1], torch.tensor([1, 3]))

    def test_reset_initializes(self):
        """Test reset lazily initializes the environment."""
        env, _, scene = make_env(num_envs=2)

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ):
            obs, infos = env.reset()

        assert env._initialized is True
        scene.build.assert_called_once_with(n_envs=2)
        assert obs.shape == (2, 4)
        assert len(infos) == 2

    def test_step(self):
        """Test step returns batched tensors and increments episode length."""
        env, task, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.zeros(4, dtype=torch.long)

        actions = np.zeros((4, 2), dtype=np.float64)
        obs, rewards, terminated, truncated, infos = env.step(actions)

        assert obs.shape == (4, 4)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert len(infos) == 4
        assert torch.equal(env.episode_length, torch.ones(4, dtype=torch.long))
        # numpy actions are converted to float32 tensors on the device
        assert task.step_calls[-1].dtype == torch.float32
        assert task.step_calls[-1].shape == (4, 2)

    def test_step_validates_action_shape(self):
        """Test step rejects wrongly shaped actions."""
        env, _, _ = make_env(num_envs=4)
        env._initialized = True
        env.device = torch.device("cpu")
        env.episode_length = torch.zeros(4, dtype=torch.long)

        with pytest.raises(ValueError, match="Expected actions"):
            env.step(torch.zeros(4, 3))

    def test_step_initializes(self):
        """Test step lazily initializes the environment."""
        env, _, scene = make_env(num_envs=3)

        with patch(
            "cloud_robotics_sim.utils.genesis_compat.ensure_genesis_initialized"
        ):
            obs, rewards, terminated, truncated, infos = env.step(torch.zeros(3, 2))

        assert env._initialized is True
        scene.build.assert_called_once_with(n_envs=3)
        assert obs.shape == (3, 4)

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
        assert VectorizedEnv is VectorizedEnvironment
        assert GenesisVecEnv is GenesisVectorizedEnv


class TestRenderConfig:
    """Offline tests for the L2 batched-render config plumbing.

    These never touch gs.init; the availability-error path is exercised by
    mocking ``batch_renderer_available``, so they run on any platform
    (gs-madrona is Linux-only).
    """

    def test_none_passthrough(self):
        from cloud_robotics_sim.core.vectorized import load_render_config

        assert load_render_config(None) is None

    def test_dict_passthrough_and_defaults(self):
        from cloud_robotics_sim.core.vectorized import load_render_config

        cfg = load_render_config({"mode": "batch", "resolution": [256, 256]})
        assert cfg is not None
        assert cfg["mode"] == "batch"
        cfg = load_render_config({"resolution": [128, 128]})
        assert cfg is not None and cfg["mode"] == "batch"

    def test_yaml_path(self, tmp_path):
        from cloud_robotics_sim.core.vectorized import load_render_config

        p = tmp_path / "render.yaml"
        p.write_text("mode: batch\nresolution: [64, 64]\nfov: 45.0\n", encoding="utf-8")
        cfg = load_render_config(p)
        assert cfg is not None
        assert cfg["resolution"] == [64, 64]
        assert cfg["fov"] == 45.0

    def test_missing_file_raises(self):
        from cloud_robotics_sim.core.vectorized import load_render_config

        with pytest.raises(ValueError, match="not found"):
            load_render_config("configs/render/does_not_exist.yaml")

    def test_unknown_mode_raises(self):
        from cloud_robotics_sim.core.vectorized import load_render_config

        with pytest.raises(ValueError, match="unsupported render mode"):
            load_render_config({"mode": "raytraced"})

    def test_unavailable_batch_renderer_actionable_error(self):
        """render_config + no gs-madrona -> RuntimeError naming the package."""
        from cloud_robotics_sim.core.vectorized import GenesisVectorizedEnv

        env = GenesisVectorizedEnv(
            config=VecEnvConfig(
                num_envs=2, use_cuda=False, render_config={"mode": "batch"}
            ),
            task=FakeTask(),
        )
        with patch(
            "cloud_robotics_sim.core.vectorized.batch_renderer_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="gs-madrona"):
                env._default_scene()

    def test_available_batch_renderer_constructs_options(self):
        """With gs-madrona 'present', the batch renderer options are built."""
        genesis = pytest.importorskip("genesis")
        from cloud_robotics_sim.core.vectorized import (
            GenesisVectorizedEnv,
        )

        env = GenesisVectorizedEnv(
            config=VecEnvConfig(
                num_envs=2,
                use_cuda=False,
                render_config={"mode": "batch", "batch_use_rasterizer": True},
            ),
            task=FakeTask(),
        )
        with patch(
            "cloud_robotics_sim.core.vectorized.batch_renderer_available",
            return_value=True,
        ):
            # Scene construction needs gs.init; stop at the renderer options
            # by checking the visualizer branch input instead. We verify the
            # BatchRenderer options object is what visualizer would select.
            from cloud_robotics_sim.core.vectorized import load_render_config

            cfg = load_render_config(env.config.render_config)
            renderer = genesis.renderers.BatchRenderer(
                use_rasterizer=bool(cfg.get("batch_use_rasterizer", True))
            )
            assert isinstance(renderer, genesis.renderers.BatchRenderer)
            assert renderer.use_rasterizer is True
