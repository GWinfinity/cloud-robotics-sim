"""Tests for environment composer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim import (
    ComposedEnvironment,
    ComposerConfig,
    EnvironmentComposer,
    EnvironmentVariantGenerator,
)
from cloud_robotics_sim.core.composer import GenesisGymEnv


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

    def test_compose(self, monkeypatch):
        """Test composing environment with mocked Genesis."""
        composer = EnvironmentComposer(ComposerConfig(headless=True))

        scene = MagicMock()
        scene.config = SimpleNamespace(name="test_scene")
        scene.get_spawn_positions.return_value = [(1.0, 2.0, 0.1)]

        robot = MagicMock()
        robot.config = SimpleNamespace(name="test_robot")
        robot.cameras = {}

        task = MagicMock()
        task.config = SimpleNamespace(name="test_task")

        gs_scene = MagicMock()

        gs = SimpleNamespace(
            Scene=MagicMock(return_value=gs_scene),
            options=SimpleNamespace(
                ViewerOptions=MagicMock(),
                SimOptions=MagicMock(),
            ),
            morphs=SimpleNamespace(),
        )

        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.gs",
            gs,
        )
        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.ensure_genesis_initialized",
            MagicMock(),
        )

        env = composer.compose(scene, robot, task)

        assert isinstance(env, ComposedEnvironment)
        assert env.scene is scene
        assert env.robot is robot
        assert env.task is task
        assert env.gs_scene is gs_scene
        scene.build.assert_called_once_with(gs_scene)
        robot.spawn.assert_called_once_with(gs_scene, position=(1.0, 2.0, 0.1))
        gs_scene.build.assert_called_once()

    def test_compose_with_spawn_position(self, monkeypatch):
        """Test composing environment with explicit spawn position."""
        composer = EnvironmentComposer(ComposerConfig(headless=True))

        scene = MagicMock()
        scene.config = SimpleNamespace(name="test_scene")

        robot = MagicMock()
        robot.config = SimpleNamespace(name="test_robot")
        robot.cameras = {}

        task = MagicMock()
        task.config = SimpleNamespace(name="test_task")

        gs_scene = MagicMock()
        gs = SimpleNamespace(
            Scene=MagicMock(return_value=gs_scene),
            options=SimpleNamespace(
                ViewerOptions=MagicMock(),
                SimOptions=MagicMock(),
            ),
        )

        monkeypatch.setattr("cloud_robotics_sim.core.composer.gs", gs)
        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.ensure_genesis_initialized",
            MagicMock(),
        )

        composer.compose(scene, robot, task, spawn_position=(5.0, 5.0, 0.1))

        robot.spawn.assert_called_once_with(gs_scene, position=(5.0, 5.0, 0.1))

    def test_compose_from_registry(self, monkeypatch):
        """Test compose_from_registry with a fake AssetRegistry."""
        from cloud_robotics_sim import AssetRegistry

        composer = EnvironmentComposer(ComposerConfig(headless=True))

        scene = MagicMock()
        scene.config = SimpleNamespace(name="registry_scene")
        scene.get_spawn_positions.return_value = []

        robot = MagicMock()
        robot.config = SimpleNamespace(name="registry_robot")
        robot.cameras = {}

        task = MagicMock()
        task.config = SimpleNamespace(name="registry_task")

        registry = AssetRegistry()
        registry.scenes.register("scene_name")(lambda **kwargs: scene)
        registry.robots.register("robot_name")(lambda **kwargs: robot)
        registry.tasks.register("task_name")(lambda **kwargs: task)

        gs_scene = MagicMock()
        gs = SimpleNamespace(
            Scene=MagicMock(return_value=gs_scene),
            options=SimpleNamespace(
                ViewerOptions=MagicMock(),
                SimOptions=MagicMock(),
            ),
        )

        monkeypatch.setattr("cloud_robotics_sim.core.composer.gs", gs)
        monkeypatch.setattr(
            "cloud_robotics_sim.core.composer.ensure_genesis_initialized",
            MagicMock(),
        )

        env = composer.compose_from_registry(
            "scene_name",
            "robot_name",
            "task_name",
            scene_kwargs={"size": (4.0, 4.0, 2.5)},
            robot_kwargs={"urdf_path": "/fake/robot.urdf"},
            task_kwargs={"max_episode_steps": 100},
            registry=registry,
        )

        assert isinstance(env, ComposedEnvironment)
        assert env.scene is scene
        assert env.robot is robot
        assert env.task is task

    def test_compose_from_registry_bad_registry(self):
        """Test compose_from_registry rejects non-AssetRegistry objects."""
        composer = EnvironmentComposer()
        bad_registry = MagicMock()
        with pytest.raises(TypeError):
            composer.compose_from_registry(
                "scene",
                "robot",
                "task",
                registry=bad_registry,
            )

    def test_select_spawn_position(self):
        """Test spawn position selection."""
        composer = EnvironmentComposer()

        scene_with_points = MagicMock()
        scene_with_points.get_spawn_positions.return_value = [
            (1.0, 0.0, 0.1),
            (0.0, 1.0, 0.1),
        ]
        pos = composer._select_spawn_position(scene_with_points)
        assert pos in [(1.0, 0.0, 0.1), (0.0, 1.0, 0.1)]

        scene_empty = MagicMock()
        scene_empty.get_spawn_positions.return_value = []
        assert composer._select_spawn_position(scene_empty) == (0.0, 0.0, 0.1)


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

    def test_create_variant(self, monkeypatch):
        """Test creating environment from variant config."""
        composer = MagicMock(spec=EnvironmentComposer)
        generator = EnvironmentVariantGenerator(composer)

        variant = {"scene": "s", "robot": "r", "task": "t"}
        generator.create_variant(variant)

        composer.compose_from_registry.assert_called_once_with(
            scene_name="s",
            robot_name="r",
            task_name="t",
            registry=None,
        )


class TestComposedEnvironment:
    """Tests for ComposedEnvironment."""

    def _make_env(self):
        scene = MagicMock()
        scene.config = SimpleNamespace(name="test_scene")
        scene.reset = MagicMock()

        robot = MagicMock()
        robot.config = SimpleNamespace(name="test_robot")
        robot.obs_dim = 23
        robot.action_dim = 8
        robot.action_space = {
            "low": -1.0,
            "high": 1.0,
            "shape": (8,),
            "dtype": "float32",
        }
        robot.cameras = {}
        robot.get_observation.return_value = {"joint_position": np.zeros(7)}

        task = MagicMock()
        task.config = SimpleNamespace(name="test_task")
        task.reset.return_value = {"task_info": "reset"}
        task.step.return_value = (1.0, False, False, {"task": "info"})

        gs_scene = MagicMock()
        gs_scene.step = MagicMock()

        return ComposedEnvironment(scene, robot, task, gs_scene)

    def test_reset(self):
        """Test environment reset."""
        env = self._make_env()

        obs, info = env.reset(seed=42)

        assert env.step_count == 0
        assert env.episode_reward == 0.0
        env.scene.reset.assert_called_once()
        env.robot.reset.assert_called_once()
        env.task.reset.assert_called_once_with(env.scene, env.robot, 42)
        assert list(obs.keys()) == ["joint_position"]
        np.testing.assert_array_equal(obs["joint_position"], np.zeros(7))
        assert info["seed"] == 42
        assert info["task_info"] == "reset"
        assert env.gs_scene.step.call_count == 10

    def test_reset_stabilization_failure(self):
        """Test reset when stabilization fails."""
        env = self._make_env()
        env.gs_scene.step.side_effect = RuntimeError("NaN")

        with pytest.raises(RuntimeError):
            env.reset()

    def test_step(self):
        """Test environment step."""
        env = self._make_env()
        action = np.zeros(8)

        obs, reward, terminated, truncated, info = env.step(action)

        env.robot.apply_action.assert_called_once_with(action)
        env.gs_scene.step.assert_called()
        assert env.step_count == 1
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert info["step"] == 1
        assert info["episode_reward"] == 1.0

    def test_render_no_camera(self):
        """Test render returns None without head camera."""
        env = self._make_env()
        assert env.render() is None
        assert env.render(mode="human") is None

    def test_render_with_camera(self):
        """Test render returns frame from head camera."""
        env = self._make_env()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env.robot.cameras["head_cam"] = MagicMock()
        env.robot.cameras["head_cam"].render.return_value = [frame]

        result = env.render()
        np.testing.assert_array_equal(result, frame)

    def test_close(self):
        """Test close is a no-op."""
        env = self._make_env()
        env.close()

    def test_observation_space(self):
        """Test observation space property."""
        env = self._make_env()
        space = env.observation_space
        assert space["proprioception_dim"] == 23
        assert space["has_camera"] is False

    def test_action_space(self):
        """Test action space property."""
        env = self._make_env()
        assert env.action_space == env.robot.action_space

    def test_get_info(self):
        """Test get_info method."""
        env = self._make_env()
        env.step_count = 5
        env.episode_reward = 3.0

        info = env.get_info()
        assert info["scene"] == "test_scene"
        assert info["robot"] == "test_robot"
        assert info["task"] == "test_task"
        assert info["step_count"] == 5
        assert info["episode_reward"] == 3.0

    def test_callbacks(self):
        """Test on_reset and on_step callbacks."""
        env = self._make_env()

        reset_callback = MagicMock()
        step_callback = MagicMock()
        env.on_reset = reset_callback
        env.on_step = step_callback

        env.reset(seed=0)
        reset_callback.assert_called_once()

        env.step(np.zeros(8))
        step_callback.assert_called_once()


class TestGenesisGymEnv:
    """Tests for GenesisGymEnv wrapper."""

    def _make_composed_env(self):
        robot = MagicMock()
        robot.action_dim = 8
        robot.obs_dim = 23
        robot.cameras = {}
        robot.get_observation.return_value = {"joint_position": np.zeros(7)}

        scene = MagicMock()
        scene.config = SimpleNamespace(name="test_scene")

        task = MagicMock()
        task.config = SimpleNamespace(name="test_task")
        task.reset.return_value = {}
        task.step.return_value = (0.0, False, False, {})

        gs_scene = MagicMock()
        return ComposedEnvironment(scene, robot, task, gs_scene)

    @pytest.mark.skipif(GenesisGymEnv is None, reason="gymnasium not available")
    def test_gym_env_spaces(self):
        """Test GenesisGymEnv space creation."""
        composed = self._make_composed_env()
        env = GenesisGymEnv(composed)

        assert env.action_space.shape == (8,)
        assert env.observation_space["proprioception"].shape == (23,)

    @pytest.mark.skipif(GenesisGymEnv is None, reason="gymnasium not available")
    def test_gym_env_reset(self):
        """Test GenesisGymEnv reset delegates to composed env."""
        composed = self._make_composed_env()
        env = GenesisGymEnv(composed)

        obs, info = env.reset(seed=7)
        assert info["seed"] == 7

    @pytest.mark.skipif(GenesisGymEnv is None, reason="gymnasium not available")
    def test_gym_env_step(self):
        """Test GenesisGymEnv step delegates to composed env."""
        composed = self._make_composed_env()
        env = GenesisGymEnv(composed)

        action = np.zeros(8)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0

    @pytest.mark.skipif(GenesisGymEnv is None, reason="gymnasium not available")
    def test_gym_env_render_and_close(self):
        """Test GenesisGymEnv render and close delegate."""
        composed = self._make_composed_env()
        env = GenesisGymEnv(composed)

        assert env.render() is None
        env.close()
