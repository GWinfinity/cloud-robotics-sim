"""Robot embodiment definitions.

This module provides robot configurations and implementations for
various robotic platforms including Franka Panda and UR5.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import genesis as gs
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """Configuration for robot sensors.

    Attributes:
        camera_names: List of camera identifiers.
        camera_positions: Relative camera positions.
        camera_resolutions: Camera resolutions (width, height).
        use_proprioception: Whether to include joint state observations.
        use_imu: Whether to include IMU data.
    """

    camera_names: list[str] = field(default_factory=lambda: ["head_cam"])
    camera_positions: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {"head_cam": (0.1, 0.0, 0.05)}
    )
    camera_resolutions: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"head_cam": (640, 480)}
    )
    use_proprioception: bool = True
    use_imu: bool = False


@dataclass
class EmbodimentConfig:
    """Configuration for robot embodiment.

    Attributes:
        name: Robot identifier.
        urdf_path: Path to URDF file (optional for procedural robots).
        base_position: Initial base position.
        base_orientation: Initial base orientation (quaternion).
        joint_stiffness: PD controller stiffness for each joint.
        joint_damping: PD controller damping for each joint.
        action_scale: Scaling factor for actions.
        sensor_config: Sensor configuration.
    """

    name: str = "unnamed_robot"
    urdf_path: str | None = None
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_stiffness: float = 100.0
    joint_damping: float = 10.0
    action_scale: float = 1.0
    sensor_config: SensorConfig = field(default_factory=SensorConfig)


class RobotEmbodiment(ABC):
    """Abstract base class for robot embodiments.

    A RobotEmbodiment encapsulates a robot's physical representation,
    sensors, and control interface within the simulation.

    Attributes:
        config: Embodiment configuration.
        entity: The Genesis entity (set after spawn).
        scene: The Genesis scene (set after spawn).
        cameras: Dictionary of camera sensors.
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
    """

    def __init__(self, config: EmbodimentConfig | None = None) -> None:
        self.config = config or EmbodimentConfig()
        self.entity: Any = None
        self.scene: Any = None
        self.cameras: dict[str, Any] = {}

        self._obs_dim: int = 0
        self._action_dim: int = 0

    @property
    def obs_dim(self) -> int:
        """Observation space dimension."""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """Action space dimension."""
        return self._action_dim

    @property
    def action_space(self) -> dict:
        """Action space specification."""
        return {
            "low": -1.0,
            "high": 1.0,
            "shape": (self._action_dim,),
            "dtype": "float32",
        }

    @abstractmethod
    def spawn(self, scene: gs.Scene, position: tuple | None = None) -> RobotEmbodiment:
        """Spawn the robot in the scene.

        Args:
            scene: The Genesis scene.
            position: Optional override for spawn position.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot to initial state."""
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Apply an action to the robot.

        Args:
            action: Normalized action vector [-1, 1].
        """
        pass

    @abstractmethod
    def get_observation(self) -> dict:
        """Collect observations from all sensors.

        Returns:
            Dictionary containing sensor observations.
        """
        pass

    def _initialize_cameras(self) -> None:
        """Set up camera sensors."""
        for cam_name in self.config.sensor_config.camera_names:
            # Camera setup implementation depends on Genesis API
            self.cameras[cam_name] = None  # Placeholder


class FrankaPanda(RobotEmbodiment):
    """Franka Emika Panda robot.

    A 7-DOF collaborative robot arm with a parallel-jaw gripper.

    Example:
        >>> robot = FrankaPanda(EmbodimentConfig(
        ...     name="franka_01",
        ...     base_position=(0.0, 0.0, 0.0),
        ... ))
        >>> robot.spawn(scene)
    """

    def __init__(self, config: EmbodimentConfig | None = None) -> None:
        super().__init__(config)
        self._obs_dim = 23  # 7 joints + 7 velocities + 2 gripper + 7 target
        self._action_dim = 8  # 7 joints + 1 gripper

    def spawn(
        self,
        scene: gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn Franka Panda in the scene."""
        self.scene = scene
        pos = position or self.config.base_position

        try:
            # Use Genesis built-in Franka if available
            self.entity = scene.add_entity(
                morph=gs.morphs.MJCF(file="franka_emika_panda/panda.xml"),
                pos=pos,
            )
        except Exception as e:
            logger.warning(f"Failed to load MJCF Franka: {e}")
            # Fallback to procedural creation
            self._create_procedural_franka(pos)

        self._initialize_cameras()
        logger.info(f"Franka Panda spawned at {pos}")
        return self

    def _create_procedural_franka(self, position: tuple[float, float, float]) -> None:
        """Create a simplified procedural Franka."""
        # Simplified base representation
        self.entity = self.scene.add_entity(
            morph=gs.morphs.Box(size=(0.2, 0.2, 0.1)),
            pos=position,
        )

    def reset(self) -> None:
        """Reset joint positions and velocities."""
        if self.entity and hasattr(self.entity, "set_qpos"):
            # Reset to home configuration
            home_qpos = np.zeros(self._action_dim - 1)  # Exclude gripper
            self.entity.set_qpos(home_qpos)

    def apply_action(self, action: np.ndarray) -> None:
        """Apply joint position targets.

        Args:
            action: 8-dimensional vector [7 joints, gripper].
        """
        if self.entity and hasattr(self.entity, "control_dofs_position"):
            scaled_action = action * self.config.action_scale
            self.entity.control_dofs_position(scaled_action[:-1])
            # Gripper control would go here

    def get_observation(self) -> dict:
        """Get current robot state."""
        obs = {
            "joint_position": np.zeros(7),
            "joint_velocity": np.zeros(7),
            "gripper_width": np.array([0.04]),
            "target_joint_position": np.zeros(7),
        }

        if self.entity:
            if hasattr(self.entity, "get_qpos"):
                obs["joint_position"] = self.entity.get_qpos()[:7]
            if hasattr(self.entity, "get_qvel"):
                obs["joint_velocity"] = self.entity.get_qvel()[:7]

        return obs


class UniversalRobotUR5(RobotEmbodiment):
    """Universal Robots UR5 industrial arm.

    A 6-DOF industrial robot arm suitable for manufacturing tasks.

    Example:
        >>> robot = UniversalRobotUR5(EmbodimentConfig(
        ...     name="ur5_01",
        ...     base_position=(1.0, 0.0, 0.0),
        ... ))
    """

    def __init__(self, config: EmbodimentConfig | None = None) -> None:
        super().__init__(config)
        self._obs_dim = 18  # 6 joints + 6 velocities + 6 target
        self._action_dim = 6

    def spawn(
        self,
        scene: gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn UR5 in the scene."""
        self.scene = scene
        pos = position or self.config.base_position

        try:
            self.entity = scene.add_entity(
                morph=gs.morphs.URDF(file="ur5/ur5.urdf"),
                pos=pos,
            )
        except Exception as e:
            logger.warning(f"Failed to load URDF UR5: {e}")
            self._create_procedural_ur5(pos)

        self._initialize_cameras()
        logger.info(f"UR5 spawned at {pos}")
        return self

    def _create_procedural_ur5(self, position: tuple[float, float, float]) -> None:
        """Create simplified UR5 representation."""
        self.entity = self.scene.add_entity(
            morph=gs.morphs.Box(size=(0.18, 0.18, 0.12)),
            pos=position,
        )

    def reset(self) -> None:
        """Reset to home position."""
        if self.entity and hasattr(self.entity, "set_qpos"):
            self.entity.set_qpos(np.zeros(6))

    def apply_action(self, action: np.ndarray) -> None:
        """Apply joint position targets."""
        if self.entity and hasattr(self.entity, "control_dofs_position"):
            scaled_action = action * self.config.action_scale
            self.entity.control_dofs_position(scaled_action)

    def get_observation(self) -> dict:
        """Get current robot state."""
        obs = {
            "joint_position": np.zeros(6),
            "joint_velocity": np.zeros(6),
            "target_joint_position": np.zeros(6),
        }

        if self.entity:
            if hasattr(self.entity, "get_qpos"):
                obs["joint_position"] = self.entity.get_qpos()[:6]
            if hasattr(self.entity, "get_qvel"):
                obs["joint_velocity"] = self.entity.get_qvel()[:6]

        return obs


class MobileManipulator(RobotEmbodiment):
    """Mobile base with manipulator arm.

    A differential-drive mobile base with a mounted robotic arm,
    suitable for mobile manipulation tasks.

    Attributes:
        base_type: Type of mobile base ('diff_drive', 'omni', 'ackermann').
        arm_type: Type of manipulator ('panda', 'ur5', 'custom').
    """

    def __init__(
        self,
        config: EmbodimentConfig | None = None,
        base_type: str = "diff_drive",
        arm_type: str = "panda",
    ) -> None:
        super().__init__(config)
        self.base_type = base_type
        self.arm_type = arm_type
        self._obs_dim = 30  # Placeholder
        self._action_dim = 10  # 2 base + 7 arm + 1 gripper

    def spawn(
        self,
        scene: gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn mobile manipulator."""
        self.scene = scene
        pos = position or self.config.base_position

        # Create mobile base
        self.entity = scene.add_entity(
            morph=gs.morphs.Box(size=(0.6, 0.4, 0.2)),
            pos=pos,
        )

        logger.info(f"Mobile manipulator spawned at {pos}")
        return self

    def reset(self) -> None:
        """Reset base and arm."""
        pass

    def apply_action(self, action: np.ndarray) -> None:
        """Apply base and arm actions.

        Args:
            action: [linear_vel, angular_vel, arm_joints..., gripper]
        """
        pass

    def get_observation(self) -> dict:
        """Get combined base and arm observations."""
        return {
            "base_position": np.zeros(3),
            "base_orientation": np.zeros(4),
            "joint_position": np.zeros(7),
        }
