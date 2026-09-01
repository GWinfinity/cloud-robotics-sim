"""Unitree G1 robot configuration for Genesis physics engine.

Maps MuJoCo G1 constants to Genesis-compatible equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# G1 joint groups and their PD gains / torque limits
# Ported from wbc_lab/robots/g1/actuators.py

G1_JOINT_NAMES: tuple[str, ...] = (
    # Left leg
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_NUM_JOINTS = len(G1_JOINT_NAMES)  # 29

# Body names for motion tracking
G1_ANCHOR_BODY_NAME = "torso_link"
G1_MOTION_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "torso_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)
G1_FOOT_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
G1_EE_TERMINATION_BODY_NAMES: tuple[str, ...] = (
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# Default joint positions (home keyframe)
G1_HOME_QPOS: tuple[float, ...] = (
    # Left leg
    0.0, 0.0, -0.26, 0.6, -0.34, 0.0,
    # Right leg
    0.0, 0.0, -0.26, 0.6, -0.34, 0.0,
    # Waist
    0.0, 0.0, 0.0,
    # Left arm
    0.0, 0.2, 0.0, -1.57, 0.0, 0.0, 0.0,
    # Right arm
    0.0, -0.2, 0.0, -1.57, 0.0, 0.0, 0.0,
)

# PD gains per joint group
G1_KP: dict[str, float] = {
    "hip": 200.0,
    "knee": 200.0,
    "ankle": 40.0,
    "waist": 200.0,
    "shoulder": 100.0,
    "elbow": 100.0,
    "wrist": 40.0,
}

G1_KD: dict[str, float] = {
    "hip": 10.0,
    "knee": 10.0,
    "ankle": 4.0,
    "waist": 10.0,
    "shoulder": 5.0,
    "elbow": 5.0,
    "wrist": 2.0,
}

# Torque limits (Nm) per group
G1_TORQUE_LIMIT: dict[str, float] = {
    "hip": 88.0,
    "knee": 139.0,
    "ankle": 25.0,
    "waist": 88.0,
    "shoulder": 25.0,
    "elbow": 25.0,
    "wrist": 12.0,
}

# Left-right joint symmetry pairs for motion mirroring
G1_SYMMETRY_PAIRS: tuple[tuple[int, int, float], ...] = (
    (0, 6, 1.0),    # hip_yaw: L <-> R
    (1, 7, -1.0),   # hip_roll: sign flip
    (2, 8, 1.0),    # hip_pitch
    (3, 9, 1.0),    # knee
    (4, 10, 1.0),   # ankle_pitch
    (5, 11, -1.0),  # ankle_roll: sign flip
    (15, 22, 1.0),  # shoulder_pitch
    (16, 23, -1.0), # shoulder_roll: sign flip
    (17, 24, 1.0),  # shoulder_yaw
    (18, 25, 1.0),  # elbow
    (19, 26, -1.0), # wrist_roll: sign flip
    (20, 27, 1.0),  # wrist_pitch
    (21, 28, -1.0), # wrist_yaw: sign flip
)


def _group_for_joint(name: str) -> str:
    """Resolve joint name to its PD-gain group."""
    for group in ("ankle", "wrist", "hip", "knee", "waist", "shoulder", "elbow"):
        if group in name:
            return group
    return "hip"


@dataclass(frozen=True)
class G1RobotConfig:
    """Unitree G1 robot configuration for Genesis WBC env."""

    num_joints: int = G1_NUM_JOINTS
    joint_names: tuple[str, ...] = G1_JOINT_NAMES
    anchor_body: str = G1_ANCHOR_BODY_NAME
    motion_bodies: tuple[str, ...] = G1_MOTION_BODY_NAMES
    foot_bodies: tuple[str, ...] = G1_FOOT_BODY_NAMES
    home_qpos: tuple[float, ...] = G1_HOME_QPOS
    symmetry_pairs: tuple[tuple[int, int, float], ...] = G1_SYMMETRY_PAIRS
    action_scale: float = 0.25
    urdf_path: str | None = None

    @property
    def num_dofs(self) -> int:
        return self.num_joints

    def kp_array(self) -> list[float]:
        """Per-joint Kp values."""
        return [G1_KP[_group_for_joint(j)] for j in self.joint_names]

    def kd_array(self) -> list[float]:
        """Per-joint Kd values."""
        return [G1_KD[_group_for_joint(j)] for j in self.joint_names]

    def torque_limits(self) -> list[float]:
        """Per-joint torque limits (Nm)."""
        return [G1_TORQUE_LIMIT[_group_for_joint(j)] for j in self.joint_names]
