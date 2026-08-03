"""HiFi-UMI-2K (LeRobot v3-style) trajectory loader and math utilities.

HiFi-UMI-2K (arXiv:2607.25895) stores frame-level Parquet tables with
20-D bimanual state/action vectors:

    observation.state = [right_10d, left_10d]
    per hand          = [x, y, z, rot6d_0..rot6d_5, gripper_angle_rad]

The 6D rotation block uses the *first two rows* of the rotation matrix
(see ``meta/info.json["state_layout"]``). All poses are expressed in a
shared world frame with +Z aligned with gravity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloud_robotics_sim.utils.genesis_compat import matrix_to_quaternion

# ---------------------------------------------------------------------------
# Rotation math
# ---------------------------------------------------------------------------


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (first two *rows* of R) to rotation matrices.

    Args:
        rot6d: (..., 6) array with [row0(3), row1(3)] of the rotation matrix.

    Returns:
        (..., 3, 3) rotation matrices (orthonormalized via Gram-Schmidt).
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    r1 = rot6d[..., 0:3]
    r2 = rot6d[..., 3:6]
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = r2 - np.sum(r1 * r2, axis=-1, keepdims=True) * r1
    r2 = r2 / np.linalg.norm(r2, axis=-1, keepdims=True)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=-2)


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (first two rows) to quaternion(s) in wxyz order."""
    rot = rot6d_to_matrix(rot6d)
    quat = np.asarray(matrix_to_quaternion(rot), dtype=np.float64)
    # Normalize and fix sign for continuity-friendly downstream use.
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (wxyz), broadcastable."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions) in wxyz order."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    """Rotation angle of unit quaternion(s) in degrees."""
    w = np.clip(np.abs(q[..., 0]), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between batched quaternions (wxyz).

    Args:
        q0: (N, 4) unit quaternions, segment starts.
        q1: (N, 4) unit quaternions, segment ends.
        alpha: (N,) interpolation parameter in [0, 1].
    """
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1)
    # Flip sign for shortest path.
    sign = np.where(dot < 0.0, -1.0, 1.0)
    q1 = q1 * sign[:, None]
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Fall back to lerp when quaternions are nearly identical.
    small = sin_theta < 1e-8
    a = np.where(
        small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / np.maximum(sin_theta, 1e-12)
    )
    b = np.where(small, alpha, np.sin(alpha * theta) / np.maximum(sin_theta, 1e-12))
    out = a[:, None] * q0 + b[:, None] * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HandTrajectory:
    """Single-hand trajectory in the shared HiFi-UMI world frame."""

    pos: np.ndarray  # (N, 3) meters
    quat: np.ndarray  # (N, 4) wxyz
    gripper: np.ndarray  # (N,) radians
    valid: np.ndarray  # (N,) bool — per-frame validity for this hand


@dataclass
class UMIEpisode:
    """One HiFi-UMI episode (both hands)."""

    episode_index: int
    task: str
    fps: float
    timestamps: np.ndarray  # (N,) seconds, episode-relative
    frame_valid: np.ndarray  # (N,) bool — valid.frame column
    right: HandTrajectory
    left: HandTrajectory

    @property
    def duration_s(self) -> float:
        return (
            float(self.timestamps[-1] - self.timestamps[0])
            if len(self.timestamps)
            else 0.0
        )


# ---------------------------------------------------------------------------
# Loading (LeRobot v3-style layout)
# ---------------------------------------------------------------------------


def load_info(data_root: str | Path) -> dict:
    """Load meta/info.json from a dataset part directory."""
    info_path = Path(data_root) / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_task_names(data_root: str | Path) -> list[str]:
    """Load task strings ordered by task_index from meta/tasks.parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(data_root) / "meta" / "tasks.parquet")
    names = table.column("task").to_pylist()
    idx = table.column("task_index").to_pylist()
    ordered = [""] * len(names)
    for name, i in zip(names, idx):
        ordered[int(i)] = str(name)
    return ordered


def _split_hand(
    state: np.ndarray, valid: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block = state[:, offset : offset + 10]
    mask = valid[:, offset : offset + 10].all(axis=1)
    pos = block[:, 0:3].astype(np.float64)
    quat = rot6d_to_quat_wxyz(block[:, 3:9])
    gripper = block[:, 9].astype(np.float64)
    return pos, quat, gripper, mask


def load_episode(
    data_root: str | Path, episode_index: int, data_file: str | Path | None = None
) -> UMIEpisode:
    """Load one episode from a HiFi-UMI-2K part directory.

    Args:
        data_root: path to a ``chunk-XXXX/part-YYYY`` directory.
        episode_index: episode identifier within the shard.
        data_file: optional explicit path to the frame-level parquet file.
    """
    import pyarrow.parquet as pq

    root = Path(data_root)
    if data_file is None:
        candidates = sorted(root.glob("data/chunk-*/file-*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No frame parquet found under {root / 'data'}")
        data_file = candidates[0]
    info = load_info(root)
    fps = float(info["fps"])
    task_names = load_task_names(root)

    cols = [
        "observation.state",
        "observation.state_valid",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        "valid.frame",
    ]
    table = pq.read_table(
        data_file, columns=cols, filters=[("episode_index", "=", episode_index)]
    )
    if table.num_rows == 0:
        raise ValueError(f"Episode {episode_index} not found in {data_file}")
    state = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    state_valid = np.stack(table.column("observation.state_valid").to_pylist())
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    frame_valid = np.asarray(table.column("valid.frame").to_pylist(), dtype=bool)
    task_idx = int(table.column("task_index").to_pylist()[0])

    rp, rq, rg, rv = _split_hand(state, state_valid, 0)
    lp, lq, lg, lv = _split_hand(state, state_valid, 10)
    return UMIEpisode(
        episode_index=episode_index,
        task=(
            task_names[task_idx]
            if 0 <= task_idx < len(task_names)
            else f"task_{task_idx}"
        ),
        fps=fps,
        timestamps=timestamps,
        frame_valid=frame_valid,
        right=HandTrajectory(pos=rp, quat=rq, gripper=rg, valid=rv),
        left=HandTrajectory(pos=lp, quat=lq, gripper=lg, valid=lv),
    )


# ---------------------------------------------------------------------------
# Trajectory processing
# ---------------------------------------------------------------------------


def resample_trajectory(
    hand: HandTrajectory, timestamps: np.ndarray, rate_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a hand trajectory to a uniform rate (linear + slerp).

    Returns:
        new_timestamps, pos (M, 3), quat (M, 4 wxyz), gripper (M,)
    """
    t = np.asarray(timestamps, dtype=np.float64)
    duration = t[-1] - t[0]
    n_out = max(2, int(round(duration * rate_hz)) + 1)
    t_out = np.linspace(t[0], t[-1], n_out)

    pos = np.stack([np.interp(t_out, t, hand.pos[:, i]) for i in range(3)], axis=1)
    gripper = np.interp(t_out, t, hand.gripper)

    idx = np.searchsorted(t, t_out, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)
    t0, t1 = t[idx], t[idx + 1]
    alpha = np.where(t1 > t0, (t_out - t0) / np.maximum(t1 - t0, 1e-12), 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    quat = slerp(hand.quat[idx], hand.quat[idx + 1], alpha)
    return t_out, pos, quat, gripper


def recenter_to_workspace(
    pos: np.ndarray, quat: np.ndarray, anchor_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a trajectory so its first pose lands at ``anchor_pos``.

    Rotation is untouched; the arbitrary world origin of HiFi-UMI episodes
    is absorbed by the translation.
    """
    anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
    shift = anchor_pos - pos[0]
    return pos + shift, quat
