"""Dual-arm ALOHA-AgileX embodiment for RoboTwin replay."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cloud_robotics_sim.backend import (
    ArticulationBackend,
    SceneBackend,
)
from cloud_robotics_sim.core.embodiment import EmbodimentConfig, RobotEmbodiment
from cloud_robotics_sim.utils.genesis_compat import is_genesis_scene

logger = logging.getLogger(__name__)

_LEFT_ARM_JOINTS = [f"fl_joint{i}" for i in range(1, 7)]
_LEFT_GRIPPER_JOINTS = ["fl_joint7", "fl_joint8"]
_RIGHT_ARM_JOINTS = [f"fr_joint{i}" for i in range(1, 7)]
_RIGHT_GRIPPER_JOINTS = ["fr_joint7", "fr_joint8"]

# Action mapping: [L_arm(6), L_grip, R_arm(6), R_grip]
_ACTION_LEFT_ARM_SLICE = slice(0, 6)
_ACTION_LEFT_GRIPPER_IDX = 6
_ACTION_RIGHT_ARM_SLICE = slice(7, 13)
_ACTION_RIGHT_GRIPPER_IDX = 13

_OBS_DIM = 14 * 3  # joint_pos + joint_vel + target for each action DoF
_ACTION_DIM = 14


@dataclass
class AlohaAgileXConfig(EmbodimentConfig):
    """Configuration for the ALOHA-AgileX dual-arm embodiment.

    Attributes:
        name: Robot identifier.
        urdf_path: Path to arx5_description_isaac.urdf.
        base_position: Initial base position.
        left_arm_joints: Names of the left arm joints.
        left_gripper_joints: Names of the left gripper joints (base + mimic).
        right_arm_joints: Names of the right arm joints.
        right_gripper_joints: Names of the right gripper joints (base + mimic).
        action_scale: Scaling factor for actions.
    """

    name: str = "aloha_agilex"
    urdf_path: str | None = None
    left_arm_joints: list[str] = field(default_factory=lambda: list(_LEFT_ARM_JOINTS))
    left_gripper_joints: list[str] = field(
        default_factory=lambda: list(_LEFT_GRIPPER_JOINTS)
    )
    right_arm_joints: list[str] = field(default_factory=lambda: list(_RIGHT_ARM_JOINTS))
    right_gripper_joints: list[str] = field(
        default_factory=lambda: list(_RIGHT_GRIPPER_JOINTS)
    )


class AlohaAgileX(RobotEmbodiment):
    """ALOHA-AgileX dual-arm robot for RoboTwin tasks.

    The URDF exposes 38 DoFs but only 16 are driven for tabletop manipulation:
    left arm 6 + left gripper 2 (mimic) + right arm 6 + right gripper 2 (mimic).
    The action space is 14-D to match RoboTwin:
    ``[L_arm(6), L_grip, R_arm(6), R_grip]``.
    """

    def __init__(self, config: AlohaAgileXConfig | None = None) -> None:
        super().__init__(config or AlohaAgileXConfig())
        self.config: AlohaAgileXConfig
        self._obs_dim = _OBS_DIM
        self._action_dim = _ACTION_DIM
        self._joint_name_to_action_index: dict[str, int] = {}
        self._dofs_idx_local: list[int] = []
        self._last_action = np.zeros(_ACTION_DIM, dtype=np.float64)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def spawn(
        self,
        scene: SceneBackend | Any,
        position: tuple | None = None,
    ) -> AlohaAgileX:
        """Spawn the ALOHA-AgileX robot in the scene."""
        self.scene = scene
        pos = position or self.config.base_position

        urdf_path = self.config.urdf_path
        if not urdf_path:
            raise ValueError("AlohaAgileX requires a urdf_path in its config")

        if is_genesis_scene(scene):
            raise RuntimeError(
                "AlohaAgileX must be spawned through the backend abstraction, "
                "not a raw Genesis gs.Scene"
            )

        backend = scene.backend if hasattr(scene, "backend") else None
        if backend is None:
            raise RuntimeError("Scene backend is not available for spawning robots")

        self.entity = backend.load_urdf(
            file=urdf_path,
            pos=pos,
            fixed=True,
        )
        scene.add_articulation(self.entity)
        self._build_joint_mapping()
        self._initialize_cameras()
        logger.info(f"AlohaAgileX spawned at {pos} with {len(self._dofs_idx_local)} driven DoFs")
        return self

    def _build_joint_mapping(self) -> None:
        """Map action indices to backend joint dofs_idx_local."""
        entity = self.entity
        if entity is None or not isinstance(entity, ArticulationBackend):
            raise RuntimeError("Robot entity is not a valid articulation")

        joint_names = entity.get_joint_names()
        if not joint_names:
            logger.warning("Cannot inspect joints; falling back to identity action mapping")
            self._joint_name_to_action_index = {}
            return

        driven_joint_order: list[tuple[str, int]] = []
        for joint_name in joint_names:
            action_index = self._joint_name_to_action_index_for(joint_name)
            if action_index is None:
                continue
            dofs_idx = entity.get_joint_dofs_idx_local(joint_name)
            # A gripper base + mimic share the same scalar action.
            for local_dof in dofs_idx:
                driven_joint_order.append((joint_name, local_dof))
                self._joint_name_to_action_index[f"{joint_name}_{local_dof}"] = action_index

        self._dofs_idx_local = [idx for _, idx in driven_joint_order]
        logger.debug(
            f"ALOHA-AgileX driven DoFs: {self._dofs_idx_local} "
            f"({len(self._dofs_idx_local)} total)"
        )

    def _joint_name_to_action_index_for(self, joint_name: str) -> int | None:
        """Return the action index corresponding to a driven joint name."""
        if joint_name in self.config.left_arm_joints:
            return self.config.left_arm_joints.index(joint_name)
        if joint_name in self.config.right_arm_joints:
            return 7 + self.config.right_arm_joints.index(joint_name)
        if joint_name in self.config.left_gripper_joints:
            return _ACTION_LEFT_GRIPPER_IDX
        if joint_name in self.config.right_gripper_joints:
            return _ACTION_RIGHT_GRIPPER_IDX
        return None

    def reset(self) -> None:
        """Reset the robot to a neutral configuration."""
        entity = self.entity
        if entity is None:
            return
        if hasattr(entity, "n_qs") and entity.n_qs > 0 and hasattr(entity, "set_qpos"):
            entity.set_qpos(np.zeros(entity.n_qs, dtype=np.float64))
        self._last_action = np.zeros(_ACTION_DIM, dtype=np.float64)

    def apply_action(self, action: np.ndarray) -> None:
        """Apply a 14-D action vector to the robot.

        Args:
            action: Normalized action vector [L_arm(6), L_grip, R_arm(6), R_grip].
        """
        entity = self.entity
        if entity is None or not hasattr(entity, "control_dofs_position"):
            return

        action_arr = np.asarray(action, dtype=np.float64).flatten()
        if action_arr.shape != (_ACTION_DIM,):
            raise ValueError(f"Expected action shape ({_ACTION_DIM},), got {action_arr.shape}")

        self._last_action = action_arr.copy()

        if not self._dofs_idx_local:
            # Fallback: assume the first n_dofs directly correspond to action order.
            target = self._expand_action_to_dof_targets(action_arr)
            entity.control_dofs_position(target)
            return

        target_values: list[float] = []
        action_index_for_dof: list[int] = []
        for joint_key, action_index in self._joint_name_to_action_index.items():
            joint_name, _ = joint_key.rsplit("_", 1)
            # Skip duplicate mimic entries; we already have one target per local dof.
            target_values.append(float(action_arr[action_index]))
            action_index_for_dof.append(action_index)

        # Build dofs_idx_local ordered by the driven joints.
        dofs_idx_local = self._dofs_idx_local
        target = np.array(target_values, dtype=np.float64)
        entity.control_dofs_position(target, dofs_idx_local=dofs_idx_local)

    def _expand_action_to_dof_targets(self, action: np.ndarray) -> np.ndarray:
        """Expand a 14-D action to a full DoF target array (fallback)."""
        entity = self.entity
        n_dofs = getattr(entity, "n_dofs", _ACTION_DIM)
        targets = np.zeros(n_dofs, dtype=np.float64)

        targets[:6] = action[_ACTION_LEFT_ARM_SLICE]
        if n_dofs > 6:
            targets[6] = action[_ACTION_LEFT_GRIPPER_IDX]
        if n_dofs > 7:
            targets[7:13] = action[_ACTION_RIGHT_ARM_SLICE]
        if n_dofs > 13:
            targets[13] = action[_ACTION_RIGHT_GRIPPER_IDX]
        return targets

    def get_observation(self) -> dict:
        """Collect proprioceptive observations."""
        entity = self.entity
        if entity is None or not isinstance(entity, ArticulationBackend):
            return {
                "joint_position": np.zeros(_ACTION_DIM, dtype=np.float64),
                "joint_velocity": np.zeros(_ACTION_DIM, dtype=np.float64),
                "target_joint_position": self._last_action.copy(),
            }

        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)

        # Map full qpos to 14-D action-space observation.
        joint_position = self._map_dof_to_action_space(qpos)
        joint_velocity = self._map_dof_to_action_space(qvel)

        return {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "target_joint_position": self._last_action.copy(),
        }

    def _map_dof_to_action_space(self, values: np.ndarray) -> np.ndarray:
        """Map full DoF values to the 14-D action space.

        For arm joints we take the joint value directly. For gripper joints we
        average the base + mimic values to produce a single gripper width.
        """
        result = np.zeros(_ACTION_DIM, dtype=np.float64)
        if not self._dofs_idx_local:
            n = min(len(values), _ACTION_DIM)
            result[:n] = values[:n]
            return result

        counts = np.zeros(_ACTION_DIM, dtype=np.int64)
        for joint_key, action_index in self._joint_name_to_action_index.items():
            _, local_dof_str = joint_key.rsplit("_", 1)
            local_dof = int(local_dof_str)
            if local_dof < len(values):
                result[action_index] += values[local_dof]
                counts[action_index] += 1

        # Average gripper values across base + mimic.
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero]
        return result

    def get_action_space(self) -> dict:
        """Return the action space specification."""
        return {
            "low": -1.0,
            "high": 1.0,
            "shape": (_ACTION_DIM,),
            "dtype": "float32",
        }
