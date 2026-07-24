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

from cloud_robotics_sim.backend import ArticulationBackend, EntityBackend, SceneBackend
from cloud_robotics_sim.utils.genesis_compat import is_genesis_scene

logger = logging.getLogger(__name__)

_is_genesis_scene = is_genesis_scene


# Franka Panda dimensions
_FRANKA_JOINTS = 7
_FRANKA_GRIPPER = 1
_FRANKA_OBS_DIM = _FRANKA_JOINTS * 3 + 2  # joints + velocities + target + gripper
_FRANKA_ACTION_DIM = _FRANKA_JOINTS + _FRANKA_GRIPPER

# UR5 dimensions
_UR5_JOINTS = 6
_UR5_OBS_DIM = _UR5_JOINTS * 3  # joints + velocities + target
_UR5_ACTION_DIM = _UR5_JOINTS

# Mobile manipulator dimensions
_MOBILE_BASE_DOF = 2
_MOBILE_OBS_DIM = 30  # Placeholder: base + arm + gripper + targets
_MOBILE_ACTION_DIM = _MOBILE_BASE_DOF + _FRANKA_ACTION_DIM


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
        self.entity: EntityBackend | ArticulationBackend | Any | None = None
        self.scene: SceneBackend | gs.Scene | None = None
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
    def spawn(
        self,
        scene: SceneBackend | gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn the robot in the scene.

        Args:
            scene: The backend scene or native Genesis gs.Scene.
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
        self._obs_dim = _FRANKA_OBS_DIM
        self._action_dim = _FRANKA_ACTION_DIM

    def spawn(
        self,
        scene: SceneBackend | gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn Franka Panda in the scene."""
        self.scene = scene
        pos = position or self.config.base_position

        # Prefer an explicit model path from the embodiment config; fall back to
        # the Genesis built-in / current-directory lookup for backwards compatibility.
        model_path = self.config.urdf_path or "franka_emika_panda/panda.xml"

        if _is_genesis_scene(scene):
            try:
                morph = gs.morphs.MJCF(file=model_path, pos=pos)
                self.entity = scene.add_entity(morph)
            except Exception as e:
                logger.warning(f"Failed to load MJCF Franka from '{model_path}': {e}")
                self._create_procedural_franka(pos)
        else:
            backend = scene.backend if hasattr(scene, "backend") else None
            if backend is None:
                raise RuntimeError("Scene backend is not available for spawning robots")

            try:
                self.entity = backend.load_mjcf(file=model_path, pos=pos)
                scene.add_articulation(self.entity)
            except Exception as e:
                logger.warning(f"Failed to load MJCF Franka from '{model_path}': {e}")
                # Fallback to procedural creation
                self._create_procedural_franka(pos)

        self._initialize_cameras()
        logger.info(f"Franka Panda spawned at {pos}")
        return self

    def _create_procedural_franka(self, position: tuple[float, float, float]) -> None:
        """Create a simplified procedural Franka."""
        # Simplified base representation. Mark it fixed so the placeholder
        # does not participate in unstable rigid-body contact dynamics when
        # the real MJCF asset is unavailable.
        if self.scene is None:
            raise RuntimeError("Scene backend is not available")

        if _is_genesis_scene(self.scene):
            morph = gs.morphs.Box(size=(0.2, 0.2, 0.1), pos=position)
            self.entity = self.scene.add_entity(morph)
        else:
            backend = self.scene.backend if hasattr(self.scene, "backend") else None
            if backend is None:
                raise RuntimeError("Scene backend is not available")
            self.entity = backend.create_box(
                size=(0.2, 0.2, 0.1),
                pos=position,
                static=True,
                name="procedural_franka",
            )
            self.scene.add_entity(self.entity)

    def reset(self) -> None:
        """Reset joint positions and velocities."""
        entity = self.entity
        if entity is None:
            return
        if hasattr(entity, "n_qs") and entity.n_qs > 0 and hasattr(entity, "set_qpos"):
            # Reset to home configuration. The MJCF Franka has two
            # independent finger DOFs, so the qpos size must match the
            # entity rather than the 8-dim action space.
            home_qpos = np.zeros(entity.n_qs)
            entity.set_qpos(home_qpos)

    def apply_action(self, action: np.ndarray) -> None:
        """Apply joint position targets.

        Args:
            action: 8-dimensional vector [7 joints, gripper].
        """
        entity = self.entity
        if (
            entity is None
            or not hasattr(entity, "n_dofs")
            or entity.n_dofs <= 0
            or not hasattr(entity, "control_dofs_position")
        ):
            return
        scaled_action = action * self.config.action_scale
        arm_targets = scaled_action[:_FRANKA_JOINTS]
        gripper_target = scaled_action[_FRANKA_JOINTS]
        # The real MJCF Franka exposes two finger DOFs driven by a
        # single tendon. Expand the scalar gripper command to both.
        if hasattr(entity, "n_qs") and entity.n_qs == _FRANKA_ACTION_DIM + 1:
            targets = np.concatenate([arm_targets, np.full(2, gripper_target)])
        else:
            targets = np.concatenate([arm_targets, np.array([gripper_target])])
        entity.control_dofs_position(targets)

    def get_observation(self) -> dict:
        """Get current robot state."""
        obs = {
            "joint_position": np.zeros(7),
            "joint_velocity": np.zeros(7),
            "gripper_width": np.array([0.04]),
            "target_joint_position": np.zeros(7),
        }

        entity = self.entity
        if entity is not None and hasattr(entity, "get_qpos"):
            obs["joint_position"] = entity.get_qpos()[:7]
            if hasattr(entity, "get_qvel"):
                obs["joint_velocity"] = entity.get_qvel()[:7]

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
        self._obs_dim = _UR5_OBS_DIM
        self._action_dim = _UR5_ACTION_DIM

    def spawn(
        self,
        scene: SceneBackend | gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn UR5 in the scene."""
        self.scene = scene
        pos = position or self.config.base_position
        model_path = self.config.urdf_path or "ur5/ur5.urdf"

        if _is_genesis_scene(scene):
            try:
                morph = gs.morphs.URDF(file=model_path, pos=pos)
                self.entity = scene.add_entity(morph)
            except Exception as e:
                logger.warning(f"Failed to load URDF UR5: {e}")
                self._create_procedural_ur5(pos)
        else:
            backend = scene.backend if hasattr(scene, "backend") else None
            if backend is None:
                raise RuntimeError("Scene backend is not available for spawning robots")

            try:
                self.entity = backend.load_urdf(file=model_path, pos=pos)
                scene.add_articulation(self.entity)
            except Exception as e:
                logger.warning(f"Failed to load URDF UR5: {e}")
                self._create_procedural_ur5(pos)

        self._initialize_cameras()
        logger.info(f"UR5 spawned at {pos}")
        return self

    def _create_procedural_ur5(self, position: tuple[float, float, float]) -> None:
        """Create simplified UR5 representation."""
        if self.scene is None:
            raise RuntimeError("Scene backend is not available")

        if _is_genesis_scene(self.scene):
            morph = gs.morphs.Box(size=(0.18, 0.18, 0.12), pos=position)
            self.entity = self.scene.add_entity(morph)
        else:
            backend = self.scene.backend if hasattr(self.scene, "backend") else None
            if backend is None:
                raise RuntimeError("Scene backend is not available")
            self.entity = backend.create_box(
                size=(0.18, 0.18, 0.12),
                pos=position,
                name="procedural_ur5",
            )
            self.scene.add_entity(self.entity)

    def reset(self) -> None:
        """Reset to home position."""
        entity = self.entity
        if entity is None:
            return
        if hasattr(entity, "n_qs") and entity.n_qs > 0 and hasattr(entity, "set_qpos"):
            entity.set_qpos(np.zeros(6))

    def apply_action(self, action: np.ndarray) -> None:
        """Apply joint position targets."""
        entity = self.entity
        if (
            entity is None
            or not hasattr(entity, "n_dofs")
            or entity.n_dofs <= 0
            or not hasattr(entity, "control_dofs_position")
        ):
            return
        scaled_action = action * self.config.action_scale
        entity.control_dofs_position(scaled_action)

    def get_observation(self) -> dict:
        """Get current robot state."""
        obs = {
            "joint_position": np.zeros(6),
            "joint_velocity": np.zeros(6),
            "target_joint_position": np.zeros(6),
        }

        entity = self.entity
        if entity is not None and hasattr(entity, "get_qpos"):
            obs["joint_position"] = entity.get_qpos()[:6]
            if hasattr(entity, "get_qvel"):
                obs["joint_velocity"] = entity.get_qvel()[:6]

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
        self._obs_dim = _MOBILE_OBS_DIM
        self._action_dim = _MOBILE_ACTION_DIM

    def spawn(
        self,
        scene: SceneBackend | gs.Scene,
        position: tuple | None = None,
    ) -> RobotEmbodiment:
        """Spawn mobile manipulator."""
        self.scene = scene
        pos = position or self.config.base_position

        if _is_genesis_scene(scene):
            morph = gs.morphs.Box(size=(0.6, 0.4, 0.2), pos=pos)
            self.entity = scene.add_entity(morph)
        else:
            backend = scene.backend if hasattr(scene, "backend") else None
            if backend is None:
                raise RuntimeError("Scene backend is not available for spawning robots")

            # Create mobile base
            self.entity = backend.create_box(
                size=(0.6, 0.4, 0.2),
                pos=pos,
                name="mobile_base",
            )
            scene.add_entity(self.entity)

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
