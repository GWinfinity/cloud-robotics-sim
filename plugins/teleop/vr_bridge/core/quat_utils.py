"""Minimal quaternion helpers (wxyz order, matching Genesis conventions)."""

from __future__ import annotations

import numpy as np


def qnormalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion, guarding against zero norm."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in wxyz order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def qinv(q: np.ndarray) -> np.ndarray:
    """Inverse of a (unit) quaternion."""
    q = qnormalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def qfix_sign(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the sign of ``q`` so it lies on the same hemisphere as reference.

    Keeps quaternion time-series continuous for filtering and deltas.
    """
    if float(np.dot(q, reference)) < 0.0:
        return -np.asarray(q, dtype=np.float64)
    return np.asarray(q, dtype=np.float64)


def nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Normalised linear interpolation between two unit quaternions."""
    b = qfix_sign(np.asarray(b, dtype=np.float64), a)
    return qnormalize((1.0 - t) * a + t * b)
