"""Motion mirroring for sagittal-plane augmentation.

Ported from wbc_lab/motion/motion_mirror.py  -mirrors motion clips
across the sagittal plane (y -> -y) by swapping left/right joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MirrorConfig:
    """Configuration for motion mirroring."""

    # (left_idx, right_idx, scale) pairs for joint mirroring
    joint_pairs: tuple[tuple[int, int, float], ...] = ()
    # (left_idx, right_idx) pairs for body name swapping
    body_pairs: tuple[tuple[int, int], ...] = ()


def mirror_joint_array(
    data: np.ndarray,
    pairs: tuple[tuple[int, int, float], ...],
) -> np.ndarray:
    """Mirror a (T, J) joint array by swapping left/right.

    Args:
        data: (T, J) joint positions or velocities.
        pairs: (left_idx, right_idx, scale) tuples.

    Returns:
        Mirrored copy of data.
    """
    mirrored = data.copy()
    for left, right, scale in pairs:
        mirrored[:, left] = data[:, right] * scale
        mirrored[:, right] = data[:, left] * scale
    return mirrored


def mirror_body_arrays(
    pos: np.ndarray,
    quat: np.ndarray | None = None,
    body_pairs: tuple[tuple[int, int], ...] = (),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Mirror body positions/rotations across sagittal plane.

    Flips y-coordinate and swaps left/right body pairs.

    Args:
        pos: (T, B, 3) body positions.
        quat: (T, B, 4) body quaternions (optional).
        body_pairs: (left_idx, right_idx) tuples for body swapping.

    Returns:
        (mirrored_pos, mirrored_quat)
    """
    mir_pos = pos.copy()
    mir_pos[:, :, 1] *= -1  # flip y

    for left, right in body_pairs:
        mir_pos[:, left, :] = pos[:, right, :].copy()
        mir_pos[:, right, :] = pos[:, left, :].copy()
        mir_pos[:, left, 1] *= -1
        mir_pos[:, right, 1] *= -1

    mir_quat = None
    if quat is not None:
        mir_quat = quat.copy()
        # Flip quaternion y/z components for sagittal mirror
        mir_quat[:, :, 1] *= -1  # negate y
        for left, right in body_pairs:
            mir_quat[:, left, :] = quat[:, right, :].copy()
            mir_quat[:, right, :] = quat[:, left, :].copy()
            mir_quat[:, left, 1] *= -1
            mir_quat[:, right, 1] *= -1

    return mir_pos, mir_quat


def mirror_motion_log(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray | None = None,
    joint_pairs: tuple[tuple[int, int, float], ...] = (),
    body_pairs: tuple[tuple[int, int], ...] = (),
) -> dict[str, np.ndarray]:
    """Mirror a complete motion log.

    Args:
        joint_pos: (T, J) joint positions.
        joint_vel: (T, J) joint velocities.
        body_pos: (T, B, 3) body positions.
        body_quat: (T, B, 4) body quaternions (optional).
        joint_pairs: Joint mirror pairs.
        body_pairs: Body mirror pairs.

    Returns:
        Dict with mirrored arrays.
    """
    mir_jp = mirror_joint_array(joint_pos, joint_pairs)
    mir_jv = mirror_joint_array(joint_vel, joint_pairs)
    mir_bp, mir_bq = mirror_body_arrays(body_pos, body_quat, body_pairs)

    result = {
        "joint_pos": mir_jp,
        "joint_vel": mir_jv,
        "body_pos_w": mir_bp,
    }
    if mir_bq is not None:
        result["body_quat_w"] = mir_bq
    return result
