"""Tests for task definitions."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim import (
    NavigationTask,
    PickPlaceTask,
    TaskConfig,
)
from cloud_robotics_sim.core.task import ReachTask


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_default_values(self):
        """Test default task configuration."""
        config = TaskConfig()

        assert config.name == "unnamed_task"
        assert config.max_episode_steps == 500
        assert config.success_reward == 1.0
        assert config.timeout_penalty == -0.1
        assert config.step_penalty == -0.001

    def test_custom_values(self):
        """Test custom task configuration."""
        config = TaskConfig(
            name="pick_place",
            max_episode_steps=200,
            success_reward=10.0,
        )

        assert config.name == "pick_place"
        assert config.max_episode_steps == 200
        assert config.success_reward == 10.0


class TestPickPlaceTask:
    """Tests for PickPlaceTask."""

    def test_reset_finds_object(self):
        """Test reset stores initial object position."""
        task = PickPlaceTask(
            TaskConfig(),
            object_name="cube",
            target_position=(0.5, 0.0, 0.05),
        )

        entity = MagicMock()
        entity.get_pos.return_value = np.array([0.1, 0.2, 0.3])

        scene = MagicMock()
        scene.entities = {"cube": entity}

        robot = MagicMock()
        info = task.reset(scene, robot, seed=0)

        assert task.step_count == 0
        assert task.succeeded is False
        assert task._is_grasped is False
        np.testing.assert_array_equal(
            task._object_initial_pos, np.array([0.1, 0.2, 0.3])
        )
        assert info["object_name"] == "cube"
        assert info["target_position"] == [0.5, 0.0, 0.05]

    def test_reset_object_not_found(self):
        """Test reset when object is not in scene."""
        task = PickPlaceTask(object_name="missing")
        scene = MagicMock()
        scene.entities = {}

        info = task.reset(scene, MagicMock(), seed=0)
        assert task._object_initial_pos is None
        assert info["object_name"] == "missing"

    def test_step_success(self):
        """Test step when object reaches target."""
        task = PickPlaceTask(
            TaskConfig(max_episode_steps=10, success_reward=5.0),
            object_name="cube",
            target_position=(0.5, 0.0, 0.05),
            success_threshold=0.1,
        )

        entity = MagicMock()
        entity.get_pos.return_value = np.array([0.5, 0.0, 0.05])

        scene = MagicMock()
        scene.entities = {"cube": entity}

        reward, terminated, truncated, info = task.step(scene, MagicMock(), np.zeros(8))

        assert reward > 4.0  # success reward plus shaping
        assert terminated is True
        assert truncated is False
        assert task.succeeded is True
        assert info["success"] is True
        assert info["dist_to_target"] == 0.0

    def test_step_object_not_found(self):
        """Test step when object is missing."""
        task = PickPlaceTask(object_name="cube")
        scene = MagicMock()
        scene.entities = {}

        reward, terminated, truncated, info = task.step(scene, MagicMock(), np.zeros(8))

        assert reward == task.config.step_penalty
        assert terminated is False
        assert truncated is False
        assert info["dist_to_target"] is None

    def test_step_timeout(self):
        """Test step timeout handling."""
        config = TaskConfig(max_episode_steps=2, timeout_penalty=-1.0)
        task = PickPlaceTask(config, object_name="cube")

        entity = MagicMock()
        entity.get_pos.return_value = np.array([10.0, 10.0, 10.0])
        scene = MagicMock()
        scene.entities = {"cube": entity}

        # First step
        task.step(scene, MagicMock(), np.zeros(8))
        # Second step (timeout)
        reward, terminated, truncated, info = task.step(scene, MagicMock(), np.zeros(8))

        assert truncated is True
        assert terminated is False
        expected = (
            config.step_penalty
            + config.timeout_penalty
            + 0.1
            * np.exp(
                -np.linalg.norm(np.array([10.0, 10.0, 10.0]) - task.target_position)
            )
        )
        assert reward == pytest.approx(expected)
        assert info["success"] is False


class TestNavigationTask:
    """Tests for NavigationTask."""

    def test_reset_computes_distance(self):
        """Test reset computes initial distance to target."""
        task = NavigationTask(
            TaskConfig(),
            target_position=(3.0, 4.0, 0.0),
        )

        robot = MagicMock()
        robot.entity = MagicMock()
        robot.entity.get_pos.return_value = np.array([0.0, 0.0, 0.0])

        info = task.reset(MagicMock(), robot, seed=0)

        assert task._prev_distance == 5.0
        assert info["initial_distance"] == 5.0

    def test_step_progress_reward(self):
        """Test progress reward in navigation."""
        task = NavigationTask(
            TaskConfig(),
            target_position=(1.0, 0.0, 0.0),
        )
        task._prev_distance = 2.0

        robot = MagicMock()
        robot.entity = MagicMock()
        robot.entity.get_pos.return_value = np.array([0.5, 0.0, 0.0])

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(2))

        assert reward == task.config.step_penalty + 1.5  # progress = 2.0 - 0.5
        assert task._prev_distance == 0.5

    def test_step_success(self):
        """Test navigation success."""
        task = NavigationTask(
            TaskConfig(success_reward=2.0),
            target_position=(0.5, 0.0, 0.0),
            success_threshold=0.6,
        )

        robot = MagicMock()
        robot.entity = MagicMock()
        robot.entity.get_pos.return_value = np.array([0.5, 0.0, 0.0])

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(2))

        assert reward > 1.9
        assert terminated is True
        assert task.succeeded is True

    def test_step_no_robot_entity(self):
        """Test step without robot entity."""
        task = NavigationTask()
        robot = MagicMock()
        robot.entity = None

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(2))

        assert reward == task.config.step_penalty
        assert terminated is False

    def test_step_timeout(self):
        """Test navigation timeout."""
        config = TaskConfig(max_episode_steps=1, timeout_penalty=-0.5)
        task = NavigationTask(config)

        robot = MagicMock()
        robot.entity = MagicMock()
        robot.entity.get_pos.return_value = np.array([10.0, 10.0, 0.0])

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(2))

        assert truncated is True
        assert reward == config.step_penalty + config.timeout_penalty
        assert info["success"] is False


class TestReachTask:
    """Tests for ReachTask."""

    def test_reset_with_seed(self):
        """Test reset samples random target from seed."""
        task = ReachTask(
            TaskConfig(),
            target_position=(0.0, 0.0, 0.0),
        )

        info = task.reset(MagicMock(), MagicMock(), seed=42)

        assert info["target"] == task.target_position.tolist()
        # target should be within sampled range
        assert 0.3 <= task.target_position[0] <= 0.7
        assert -0.3 <= task.target_position[1] <= 0.3
        assert 0.2 <= task.target_position[2] <= 0.6

    def test_reset_without_seed(self):
        """Test reset preserves default target without seed."""
        task = ReachTask(target_position=(0.5, 0.0, 0.5))

        info = task.reset(MagicMock(), MagicMock(), seed=None)

        np.testing.assert_array_equal(task.target_position, np.array([0.5, 0.0, 0.5]))
        assert info["target"] == [0.5, 0.0, 0.5]

    def test_step_success(self):
        """Test reach task success."""
        task = ReachTask(
            TaskConfig(success_reward=3.0),
            target_position=(0.5, 0.0, 0.5),
            success_threshold=0.1,
        )

        robot = MagicMock()
        robot.entity = MagicMock()
        # End-effector position = base_pos + [0.5, 0.0, 0.5]
        robot.entity.get_pos.return_value = np.array([0.0, 0.0, 0.0])

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(8))

        assert reward > 2.9
        assert terminated is True
        assert task.succeeded is True
        assert info["success"] is True

    def test_step_no_entity(self):
        """Test reach task without robot entity."""
        task = ReachTask()
        robot = MagicMock()
        robot.entity = None

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(8))

        assert reward == task.config.step_penalty
        assert terminated is False
        assert info["distance"] is None

    def test_step_timeout(self):
        """Test reach task timeout."""
        config = TaskConfig(max_episode_steps=1, timeout_penalty=-0.5)
        task = ReachTask(config)

        robot = MagicMock()
        robot.entity = MagicMock()
        robot.entity.get_pos.return_value = np.array([100.0, 100.0, 100.0])

        reward, terminated, truncated, info = task.step(MagicMock(), robot, np.zeros(8))

        assert truncated is True
        assert reward == config.step_penalty + config.timeout_penalty
        assert info["success"] is False
