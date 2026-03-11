"""Task definitions for robot learning.

Tasks define the goal, reward function, and termination conditions
for robotic learning scenarios.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cloud_robotics_sim.core.embodiment import RobotEmbodiment
from cloud_robotics_sim.core.scene import Scene

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a task.

    Attributes:
        name: Task identifier.
        max_episode_steps: Maximum steps per episode.
        success_reward: Reward for task completion.
        timeout_penalty: Penalty for episode timeout.
        step_penalty: Small penalty per step to encourage efficiency.
    """
    name: str = "unnamed_task"
    max_episode_steps: int = 500
    success_reward: float = 1.0
    timeout_penalty: float = -0.1
    step_penalty: float = -0.001


class Task(ABC):
    """Abstract base class for robotic tasks.

    Tasks define the learning objective through the reward function
    and termination conditions.

    Attributes:
        config: Task configuration.
        step_count: Current step in episode.
        succeeded: Whether task was completed successfully.
    """

    def __init__(self, config: TaskConfig | None = None) -> None:
        self.config = config or TaskConfig()
        self.step_count: int = 0
        self.succeeded: bool = False

    @abstractmethod
    def reset(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        seed: int,
    ) -> dict:
        """Reset the task for a new episode.

        Args:
            scene: The simulation scene.
            robot: The robot embodiment.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with task-specific information.
        """
        pass

    @abstractmethod
    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]:
        """Execute one task step.

        Args:
            scene: The simulation scene.
            robot: The robot embodiment.
            action: The action taken.

        Returns:
            Tuple of (reward, terminated, truncated, info).
        """
        pass


class PickPlaceTask(Task):
    """Pick and place task.

    The robot must pick up an object and place it at a target location.

    Attributes:
        object_name: Name of the object to manipulate.
        target_position: Target placement position.
        grasp_height: Height at which to grasp the object.
        success_threshold: Distance threshold for success.
    """

    def __init__(
        self,
        config: TaskConfig | None = None,
        object_name: str = "target_object",
        target_position: tuple[float, float, float] = (0.5, 0.0, 0.05),
        success_threshold: float = 0.05,
    ) -> None:
        super().__init__(config)
        self.object_name = object_name
        self.target_position = np.array(target_position)
        self.success_threshold = success_threshold

        self._object_initial_pos: np.ndarray | None = None
        self._is_grasped: bool = False

    def reset(self, scene: Scene, robot: RobotEmbodiment, seed: int) -> dict:
        """Reset object positions."""
        self.step_count = 0
        self.succeeded = False
        self._is_grasped = False

        # Store initial object position
        if self.object_name in scene.entities:
            entity = scene.entities[self.object_name]
            if hasattr(entity, 'get_pos'):
                self._object_initial_pos = entity.get_pos()

        return {
            'object_name': self.object_name,
            'target_position': self.target_position.tolist(),
        }

    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]:
        """Compute reward and check termination."""
        self.step_count += 1

        reward = self.config.step_penalty
        terminated = False
        truncated = False

        # Get object position
        object_pos = None
        if self.object_name in scene.entities:
            entity = scene.entities[self.object_name]
            if hasattr(entity, 'get_pos'):
                object_pos = entity.get_pos()

        if object_pos is not None:
            # Distance to target
            dist_to_target = np.linalg.norm(object_pos - self.target_position)

            # Check success
            if dist_to_target < self.success_threshold:
                reward += self.config.success_reward
                terminated = True
                self.succeeded = True

            # Shaping reward: closer is better
            reward += 0.1 * np.exp(-dist_to_target)

        # Check timeout
        if self.step_count >= self.config.max_episode_steps:
            truncated = True
            if not self.succeeded:
                reward += self.config.timeout_penalty

        info = {
            'step': self.step_count,
            'success': self.succeeded,
            'dist_to_target': dist_to_target if object_pos is not None else None,
        }

        return reward, terminated, truncated, info


class NavigationTask(Task):
    """Navigation task.

    The robot must navigate to a target position.

    Attributes:
        target_position: Goal position.
        success_threshold: Distance threshold for success.
        collision_penalty: Penalty for collisions.
    """

    def __init__(
        self,
        config: TaskConfig | None = None,
        target_position: tuple[float, float, float] = (3.0, 3.0, 0.0),
        success_threshold: float = 0.3,
        collision_penalty: float = -1.0,
    ) -> None:
        super().__init__(config)
        self.target_position = np.array(target_position)
        self.success_threshold = success_threshold
        self.collision_penalty = collision_penalty

        self._prev_distance: float | None = None

    def reset(self, scene: Scene, robot: RobotEmbodiment, seed: int) -> dict:
        """Reset navigation state."""
        self.step_count = 0
        self.succeeded = False
        self._prev_distance = None

        # Compute initial distance
        if robot.entity and hasattr(robot.entity, 'get_pos'):
            robot_pos = robot.entity.get_pos()
            self._prev_distance = np.linalg.norm(
                robot_pos - self.target_position
            )

        return {
            'target_position': self.target_position.tolist(),
            'initial_distance': self._prev_distance,
        }

    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]:
        """Compute navigation reward."""
        self.step_count += 1

        reward = self.config.step_penalty
        terminated = False
        truncated = False

        if robot.entity and hasattr(robot.entity, 'get_pos'):
            robot_pos = robot.entity.get_pos()
            distance = np.linalg.norm(robot_pos - self.target_position)

            # Progress reward
            if self._prev_distance is not None:
                progress = self._prev_distance - distance
                reward += progress

            self._prev_distance = distance

            # Success check
            if distance < self.success_threshold:
                reward += self.config.success_reward
                terminated = True
                self.succeeded = True

        # Timeout
        if self.step_count >= self.config.max_episode_steps:
            truncated = True
            if not self.succeeded:
                reward += self.config.timeout_penalty

        info = {
            'step': self.step_count,
            'success': self.succeeded,
            'distance_to_goal': self._prev_distance,
        }

        return reward, terminated, truncated, info


class ReachTask(Task):
    """End-effector reaching task.

    The robot must move its end-effector to a target position.
    """

    def __init__(
        self,
        config: TaskConfig | None = None,
        target_position: tuple[float, float, float] = (0.5, 0.0, 0.5),
        success_threshold: float = 0.05,
    ) -> None:
        super().__init__(config)
        self.target_position = np.array(target_position)
        self.success_threshold = success_threshold

    def reset(self, scene: Scene, robot: RobotEmbodiment, seed: int) -> dict:
        """Reset reach task."""
        self.step_count = 0
        self.succeeded = False

        # Sample random target if needed
        if seed > 0:
            np.random.seed(seed)
            self.target_position = np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(0.2, 0.6),
            ])

        return {'target': self.target_position.tolist()}

    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]:
        """Compute reach reward."""
        self.step_count += 1

        reward = self.config.step_penalty
        terminated = False
        truncated = False

        # Get end-effector position (approximated from robot state)
        # In practice, you'd use forward kinematics
        ee_pos = self._get_end_effector_position(robot)

        if ee_pos is not None:
            distance = np.linalg.norm(ee_pos - self.target_position)
            reward += 0.5 * np.exp(-5.0 * distance)

            if distance < self.success_threshold:
                reward += self.config.success_reward
                terminated = True
                self.succeeded = True

        if self.step_count >= self.config.max_episode_steps:
            truncated = True
            if not self.succeeded:
                reward += self.config.timeout_penalty

        return reward, terminated, truncated, {
            'success': self.succeeded,
            'distance': distance if ee_pos is not None else None,
        }

    def _get_end_effector_position(
        self,
        robot: RobotEmbodiment,
    ) -> np.ndarray | None:
        """Approximate end-effector position."""
        if robot.entity and hasattr(robot.entity, 'get_pos'):
            # Simplified: use robot base position
            return robot.entity.get_pos() + np.array([0.5, 0.0, 0.5])
        return None
