"""Retargeting: human hand/object motion -> robot joint-space trajectory."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .data import DemoSequence, RobotTrajectory

UR3_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ALLEGRO_HAND_JOINTS = [f"joint_{i}.0" for i in range(16)]

# Active joints for the Sharpa Wave 22-DOF hand after the merge normalisation
# in ``core.assets`` (prefix ``{side}_hand_``).
SHARPA_HAND_JOINTS = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]

SUPPORTED_HAND_TYPES = {"allegro": ALLEGRO_HAND_JOINTS, "sharpa": SHARPA_HAND_JOINTS}


def _hand_joint_names(hand_type: str) -> list[str]:
    """Return the ordered list of actuated hand joint names for a hand type."""
    joints = SUPPORTED_HAND_TYPES.get(hand_type)
    if joints is None:
        raise ValueError(f"Unknown hand_type: {hand_type}. Supported: {list(SUPPORTED_HAND_TYPES)}")
    return joints


def get_joint(robot, name: str):
    """Look up a joint by name, raising a clear error if missing."""
    joint = robot.get_joint(name)
    if joint is None:
        raise RuntimeError(f"Joint {name} not found in robot")
    return joint


def _flatten_indices(dofs_idx_local: list[list[int]] | list[int]) -> list[int]:
    """Flatten a list of per-joint DOF indices into a single list."""
    flat: list[int] = []
    for item in dofs_idx_local:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat.extend(int(i) for i in item)
        else:
            flat.append(int(item))
    return flat


class Retargeter(ABC):
    """Base class for retargeting a ``DemoSequence`` to a ``RobotTrajectory``."""

    @abstractmethod
    def retarget(self, demo: DemoSequence) -> RobotTrajectory:
        """Return a ``RobotTrajectory`` suitable for the robot hardware."""
        ...


class IKRetargeter(Retargeter):
    """Retarget using Genesis built-in IK for arms and a heuristic hand map.

    Args:
        robot: a built Genesis ``RigidEntity`` that contains the merged
            dual-UR3 + dual-Allegro robot.
        arm_ee_link: end-effector link name for each arm (default ``tool0``).
        hand_open_degrees: default open-hand joint value in degrees.
    """

    def __init__(
        self,
        robot: Any,
        arm_ee_link: str = "tool0",
        hand_open_degrees: float = 10.0,
        hand_type: str = "allegro",
    ) -> None:
        self.robot = robot
        self.arm_ee_link = arm_ee_link
        self.hand_open_degrees = hand_open_degrees
        self.hand_joint_names = _hand_joint_names(hand_type)

        self.left_arm_dofs = self._find_arm_dofs("left")
        self.right_arm_dofs = self._find_arm_dofs("right")
        self.left_hand_dofs = self._find_hand_dofs("left_hand")
        self.right_hand_dofs = self._find_hand_dofs("right_hand")

        self.left_ee = robot.get_link(f"left_{arm_ee_link}")
        self.right_ee = robot.get_link(f"right_{arm_ee_link}")

    def _find_arm_dofs(self, side: str) -> list[int]:
        names = [f"{side}_{j}" for j in UR3_ARM_JOINTS]
        idxs = []
        for name in names:
            joint = get_joint(self.robot, name)
            idxs.append(_flatten_indices([joint.dofs_idx_local]))
        return [i for sub in idxs for i in sub]

    def _find_hand_dofs(self, side: str) -> list[int]:
        names = [f"{side}_{j}" for j in self.hand_joint_names]
        idxs = []
        for name in names:
            joint = get_joint(self.robot, name)
            idxs.append(_flatten_indices([joint.dofs_idx_local]))
        return [i for sub in idxs for i in sub]

    def _solve_arm_ik(
        self,
        ee_link: Any,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dofs: list[int],
    ) -> np.ndarray:
        """Solve IK for one arm and return its 6-DOF joint angles."""
        q, err = self.robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=dofs,
            return_error=True,
            max_samples=20,
            max_solver_iters=15,
            pos_tol=0.001,
            rot_tol=0.01,
        )
        # q may be a torch tensor or numpy array; convert to numpy.
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        q = np.asarray(q, dtype=np.float32).flatten()
        return q[dofs]

    def _hand_q_from_demo(
        self,
        hand_traj,
        n_frames: int,
    ) -> np.ndarray:
        """Map human hand joints to the robot hand joints.

        If the demo provides at least as many joints as the hand has DOFs, copy
        the first ``n_hand_dofs`` values directly.  Otherwise return a default
        open-hand pose.
        """
        n_hand_dofs = len(self.hand_joint_names)
        if hand_traj.joints is not None and hand_traj.joints.shape[1] >= n_hand_dofs:
            return np.asarray(hand_traj.joints[:, :n_hand_dofs], dtype=np.float32)
        return np.full(
            (n_frames, n_hand_dofs),
            np.deg2rad(self.hand_open_degrees),
            dtype=np.float32,
        )

    def retarget(self, demo: DemoSequence) -> RobotTrajectory:
        n = len(demo)
        left_arm_q = np.zeros((n, 6), dtype=np.float32)
        right_arm_q = np.zeros((n, 6), dtype=np.float32)
        left_hand_q = self._hand_q_from_demo(demo.left_hand, n)
        right_hand_q = self._hand_q_from_demo(demo.right_hand, n)

        # Start from neutral arm configuration.
        neutral = np.zeros(self.robot.n_dofs, dtype=np.float32)
        self.robot.set_qpos(neutral)

        for i in range(n):
            # Left arm IK.
            left_arm_q[i] = self._solve_arm_ik(
                self.left_ee,
                demo.left_hand.wrist_positions[i],
                demo.left_hand.wrist_orientations[i],
                self.left_arm_dofs,
            )
            # Apply result so the next solve starts close to the solution.
            q = np.asarray(self.robot.get_qpos().cpu().numpy().flatten(), dtype=np.float32)
            q[self.left_arm_dofs] = left_arm_q[i]
            self.robot.set_qpos(q)

            # Right arm IK.
            right_arm_q[i] = self._solve_arm_ik(
                self.right_ee,
                demo.right_hand.wrist_positions[i],
                demo.right_hand.wrist_orientations[i],
                self.right_arm_dofs,
            )
            q = np.asarray(self.robot.get_qpos().cpu().numpy().flatten(), dtype=np.float32)
            q[self.right_arm_dofs] = right_arm_q[i]
            self.robot.set_qpos(q)

        timestamps = demo.object_trajectory.timestamps
        return RobotTrajectory(
            left_arm_q=left_arm_q,
            right_arm_q=right_arm_q,
            left_hand_q=left_hand_q,
            right_hand_q=right_hand_q,
            timestamps=timestamps,
        )


class SamplingRetargeter(Retargeter):
    """Placeholder for a sampling-based (MPPI-style) retargeter.

    The original do-as-i-do paper uses a MuJoCo-Warp-based MPPI sampler with
    warmup, domain randomization and transition rewards.  This placeholder
    simply delegates to ``IKRetargeter`` while keeping the same interface, so
    a real sampler can be dropped in later without changing the pipeline.
    """

    def __init__(self, robot: Any, **kwargs: Any) -> None:
        self.ik_retargeter = IKRetargeter(robot, **kwargs)

    def retarget(self, demo: DemoSequence) -> RobotTrajectory:
        warnings.warn(
            "SamplingRetargeter currently delegates to IKRetargeter. "
            "Replace it with a real MPPI sampler for dynamics-aware retargeting."
        )
        return self.ik_retargeter.retarget(demo)


def build_retargeter(config: dict[str, Any], robot: Any) -> Retargeter:
    """Factory for retargeters."""
    ret_cfg = config.get("retargeting", {})
    method = ret_cfg.get("method", "ik")
    hand_type = config.get("robot", {}).get("hand_type", "allegro")
    kwargs = {
        "hand_open_degrees": ret_cfg.get("hand_open_degrees", 10.0),
        "hand_type": hand_type,
    }
    if method == "ik":
        return IKRetargeter(robot, **kwargs)
    if method == "sampling":
        return SamplingRetargeter(robot, **kwargs)
    raise ValueError(f"Unknown retargeting method: {method}")
