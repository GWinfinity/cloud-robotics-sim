"""SE(3) helpers compatible with Genesis quaternion convention [x, y, z, w]."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


EPS = 1e-9


def pos_quat_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous matrix from position and quaternion.

    Args:
        pos: array shape (3,)
        quat: array shape (4,) in [x, y, z, w] order

    Returns:
        4x4 homogeneous transform.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def matrix_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and quaternion from 4x4 matrix.

    Returns:
        (pos, quat) with quat in [x, y, z, w].
    """
    T = np.asarray(T, dtype=float)
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Right-multiply two SE(3) transforms: A @ B."""
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float)


def interpolate(T_start: np.ndarray, T_end: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate translation and SLERP rotation.

    Args:
        alpha: interpolation parameter in [0, 1].
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    p0, q0 = matrix_to_pos_quat(T_start)
    p1, q1 = matrix_to_pos_quat(T_end)

    # Shortest-path quaternion sign flip.
    if np.dot(q0, q1) < 0.0:
        q0 = -q0

    p = p0 + alpha * (p1 - p0)
    r0 = R.from_quat(q0)
    r1 = R.from_quat(q1)
    # Small-angle safe interpolation using rotation vectors.
    rot_vec = (1.0 - alpha) * r0.as_rotvec() + alpha * r1.as_rotvec()
    q = R.from_rotvec(rot_vec).as_quat()
    return pos_quat_to_matrix(p, q)


def transform_vector(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the rotational part of T."""
    return (T[:3, :3] @ np.asarray(v)).reshape(3)


def translation_matrix(delta: np.ndarray) -> np.ndarray:
    """Pure translation transform."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(delta)
    return T


def clip_translation(T: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp the translation part of a transform to max_norm."""
    T = np.asarray(T, dtype=float).copy()
    p = T[:3, 3]
    norm = float(np.linalg.norm(p))
    if norm > max_norm:
        T[:3, 3] = p * (max_norm / (norm + EPS))
    return T


def identity() -> np.ndarray:
    return np.eye(4, dtype=float)
