"""Environment composition engine.

This module provides dynamic composition of Scene + Robot + Task components,
enabling flexible environment construction without code duplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import genesis as gs
import numpy as np

from cloud_robotics_sim.core.embodiment import RobotEmbodiment
from cloud_robotics_sim.core.scene import Scene
from cloud_robotics_sim.core.task import Task

logger = logging.getLogger(__name__)


@dataclass
class ComposerConfig:
    """Configuration for environment composition.

    Attributes:
        dt: Simulation timestep in seconds.
        substeps: Physics substeps per simulation step.
        headless: Whether to run in headless mode (no GUI).
        resolution: Camera resolution as (width, height).
        num_envs: Number of parallel environments (currently single env only).
        domain_randomization: Configuration for domain randomization.
    """
    dt: float = 0.01
    substeps: int = 10
    headless: bool = False
    resolution: tuple[int, int] = (640, 480)
    num_envs: int = 1
    domain_randomization: dict = field(default_factory=dict)


class ComposedEnvironment:
    """A fully composed robotic simulation environment.

    This class wraps Scene, Robot, and Task components into a unified
    Gymnasium-compatible interface for reinforcement learning.

    Attributes:
        scene: The scene component defining the environment layout.
        robot: The robot embodiment with sensors and actuators.
        task: The task specification with reward function.
        gs_scene: The underlying Genesis scene object.
        step_count: Current step count in the episode.
        episode_reward: Accumulated reward for the current episode.
    """

    def __init__(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        task: Task,
        gs_scene: Any,
    ) -> None:
        self.scene = scene
        self.robot = robot
        self.task = task
        self.gs_scene = gs_scene

        self.step_count: int = 0
        self.episode_reward: float = 0.0

        # Callbacks for extensibility
        self.on_reset: Callable | None = None
        self.on_step: Callable | None = None

    def reset(
        self,
        seed: int = 0,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            A tuple of (observation, info).
        """
        self.step_count = 0
        self.episode_reward = 0.0

        np.random.seed(seed)
        self.scene.reset()

        # Reset robot position
        spawn_pos = self._select_spawn_position()
        self.robot.reset()
        if hasattr(self.robot.entity, 'set_pos'):
            self.robot.entity.set_pos(spawn_pos)

        # Reset task state
        task_info = self.task.reset(self.scene, self.robot, seed)

        # Stabilize simulation
        for _ in range(10):
            self.gs_scene.step()

        obs = self._get_observation()
        info = {'seed': seed, **task_info}

        if self.on_reset:
            self.on_reset(seed, info)

        logger.info(f"Environment reset with seed={seed}")
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one simulation step.

        Args:
            action: The action to apply to the robot.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        self.robot.apply_action(action)
        self.gs_scene.step()
        self.step_count += 1

        reward, terminated, truncated, task_info = self.task.step(
            self.scene, self.robot, action
        )

        self.episode_reward += reward
        obs = self._get_observation()

        info = {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            **task_info,
        }

        if self.on_step:
            self.on_step(self.step_count, obs, reward, terminated, info)

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> dict:
        """Collect observations from all sensors."""
        obs = self.robot.get_observation()
        return obs

    def _select_spawn_position(self) -> tuple[float, float, float]:
        """Select a random spawn position for the robot."""
        spawn_points = self.scene.get_spawn_positions()
        if spawn_points:
            idx = np.random.randint(len(spawn_points))
            return spawn_points[idx]
        return (0.0, 0.0, 0.1)

    def render(self, mode: str = 'rgb_array') -> np.ndarray | None:
        """Render the environment.

        Args:
            mode: Rendering mode ('rgb_array' or 'human').

        Returns:
            Rendered frame as numpy array, or None if unavailable.
        """
        if mode == 'rgb_array' and 'head_cam' in self.robot.cameras:
            return self.robot.cameras['head_cam'].render(rgb=True)[0]
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    @property
    def observation_space(self) -> dict:
        """Get observation space specification."""
        return {
            'proprioception_dim': self.robot.obs_dim,
            'has_camera': len(self.robot.cameras) > 0,
        }

    @property
    def action_space(self) -> dict:
        """Get action space specification."""
        return self.robot.action_space

    def get_info(self) -> dict:
        """Get comprehensive environment information."""
        return {
            'scene': self.scene.config.name,
            'robot': self.robot.config.name,
            'task': self.task.config.name,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
        }


class EnvironmentComposer:
    """Composes robotic environments from modular components.

    This is the main entry point for creating simulation environments.
    It combines Scene, Robot, and Task components into a runnable
    ComposedEnvironment.

    Example:
        >>> composer = EnvironmentComposer(ComposerConfig(headless=True))
        >>> env = composer.compose(living_room_scene, franka_robot, pick_task)
        >>> obs, info = env.reset()
    """

    def __init__(self, config: ComposerConfig | None = None) -> None:
        self.config = config or ComposerConfig()
        self._scene_cache: dict[str, Scene] = {}
        self._robot_cache: dict[str, RobotEmbodiment] = {}

    def compose(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        task: Task,
        spawn_position: tuple[float, float, float] | None = None,
    ) -> ComposedEnvironment:
        """Compose a complete environment from components.

        Args:
            scene: The scene component defining the environment.
            robot: The robot embodiment.
            task: The task specification.
            spawn_position: Optional override for robot spawn position.

        Returns:
            A fully configured ComposedEnvironment.
        """
        logger.info("=" * 60)
        logger.info("Composing Environment")
        logger.info(f"  Scene: {scene.config.name}")
        logger.info(f"  Robot: {robot.config.name}")
        logger.info(f"  Task:  {task.config.name}")
        logger.info("=" * 60)

        # Initialize Genesis physics engine
        try:
            gs.init(backend=gs.backends.CUDA)
        except RuntimeError:
            logger.debug("Genesis already initialized")

        # Create viewer if not headless
        viewer_options = None
        if not self.config.headless:
            viewer_options = gs.options.ViewerOptions(
                camera_pos=scene.config.default_camera_pos,
                camera_lookat=scene.config.default_camera_lookat,
                res=self.config.resolution,
                max_FPS=60,
            )

        # Create Genesis scene
        gs_scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(
                dt=self.config.dt,
                substeps=self.config.substeps,
            ),
            show_viewer=not self.config.headless,
        )

        # Build scene and spawn robot
        scene.build(gs_scene)
        spawn_pos = spawn_position or self._select_spawn_position(scene)
        robot.spawn(gs_scene, position=spawn_pos)
        gs_scene.build()

        env = ComposedEnvironment(
            scene=scene,
            robot=robot,
            task=task,
            gs_scene=gs_scene,
        )

        logger.info("Environment composition complete")
        return env

    def compose_from_registry(
        self,
        scene_name: str,
        robot_name: str,
        task_name: str,
        scene_kwargs: dict | None = None,
        robot_kwargs: dict | None = None,
        task_kwargs: dict | None = None,
        registry: Any = None,
    ) -> ComposedEnvironment:
        """Compose environment using registered components.

        Args:
            scene_name: Registered name of the scene.
            robot_name: Registered name of the robot.
            task_name: Registered name of the task.
            scene_kwargs: Optional kwargs for scene creation.
            robot_kwargs: Optional kwargs for robot creation.
            task_kwargs: Optional kwargs for task creation.
            registry: Optional custom registry (uses default if None).

        Returns:
            A composed environment.

        Example:
            >>> composer = EnvironmentComposer()
            >>> env = composer.compose_from_registry(
            ...     "living_room", "franka_panda", "pick_place"
            ... )
        """
        from cloud_robotics_sim.core.registry import default_registry

        reg = registry or default_registry
        scene = reg.create_scene(scene_name, **(scene_kwargs or {}))
        robot = reg.create_robot(robot_name, **(robot_kwargs or {}))
        task = reg.create_task(task_name, **(task_kwargs or {}))

        return self.compose(scene, robot, task)

    def _select_spawn_position(self, scene: Scene) -> tuple[float, float, float]:
        """Select a default spawn position from the scene."""
        spawn_points = scene.get_spawn_positions()
        if spawn_points:
            idx = np.random.randint(len(spawn_points))
            return spawn_points[idx]
        return (0.0, 0.0, 0.1)


class EnvironmentVariantGenerator:
    """Generates multiple environment configurations for evaluation.

    Useful for creating diverse training/evaluation scenarios by
    combining different scenes, robots, and tasks.

    Example:
        >>> generator = EnvironmentVariantGenerator(composer)
        >>> variants = generator.generate_variants(
        ...     scene_names=["kitchen", "office"],
        ...     robot_names=["franka", "ur5"],
        ...     task_names=["pick_place", "push"]
        ... )
    """

    def __init__(self, composer: EnvironmentComposer) -> None:
        self.composer = composer

    def generate_variants(
        self,
        scene_names: list[str],
        robot_names: list[str],
        task_names: list[str],
        filter_fn: Callable | None = None,
    ) -> list[dict]:
        """Generate all possible environment variants.

        Args:
            scene_names: List of scene identifiers.
            robot_names: List of robot identifiers.
            task_names: List of task identifiers.
            filter_fn: Optional filter function (scene, robot, task) -> bool.

        Returns:
            List of variant configuration dictionaries.
        """
        variants = []

        for scene in scene_names:
            for robot in robot_names:
                for task in task_names:
                    if filter_fn and not filter_fn(scene, robot, task):
                        continue

                    variant = {
                        'name': f"{scene}_{robot}_{task}",
                        'scene': scene,
                        'robot': robot,
                        'task': task,
                    }
                    variants.append(variant)

        logger.info(f"Generated {len(variants)} environment variants")
        return variants

    def create_variant(
        self,
        variant_config: dict,
        registry: Any = None,
    ) -> ComposedEnvironment:
        """Create an environment from a variant configuration."""
        return self.composer.compose_from_registry(
            scene_name=variant_config['scene'],
            robot_name=variant_config['robot'],
            task_name=variant_config['task'],
            registry=registry,
        )


# Gymnasium compatibility wrapper
try:
    import gymnasium as gym

    class GenesisGymEnv(gym.Env):
        """Gymnasium-compatible wrapper for Genesis environments."""

        metadata = {'render_modes': ['rgb_array', 'human']}

        def __init__(self, composed_env: ComposedEnvironment) -> None:
            super().__init__()
            self.env = composed_env
            self.action_space = self._create_action_space()
            self.observation_space = self._create_obs_space()

        def _create_action_space(self):
            """Define the action space."""
            from gymnasium import spaces
            dim = self.env.robot.action_dim
            return spaces.Box(
                low=-1.0, high=1.0,
                shape=(dim,), dtype=np.float32
            )

        def _create_obs_space(self):
            """Define the observation space."""
            from gymnasium import spaces
            return spaces.Dict({
                'proprioception': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.env.robot.obs_dim,),
                    dtype=np.float32,
                )
            })

        def reset(self, seed: int | None = None, options: dict | None = None):
            return self.env.reset(seed=seed or 0, options=options)

        def step(self, action):
            return self.env.step(action)

        def render(self):
            return self.env.render()

        def close(self):
            self.env.close()

except ImportError:
    GenesisGymEnv = None
    logger.debug("Gymnasium not available, skipping GenesisGymEnv")
