"""Grasp-candidate loading from RoboTwin ``model_dataN.json`` files.

RoboTwin object assets ship per-class ``model_dataN.json`` files whose
``contact_points_pose`` entry lists candidate grasp poses (4x4 homogeneous
matrices in the object frame). This module parses them into
:class:`GraspCandidate` objects and scores them for a given scene.

Bypass note (plan deviation, 2026-09-26): W4 of ROBODOJO_P0_PLAN.md says
skills read graspable regions from the W3 annotation layer, but W3 is
suspended under the AnyTask route — RoboTwin's shipped contact points are an
equivalent data source and keep the skill layer unblocked.

Python 3.9 compatible (``from __future__ import annotations``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """One grasp pose candidate in the *object frame*.

    Attributes:
        position: Contact point ``(3,)`` (object units, pre-scale).
        approach: Unit approach direction ``(3,)`` — the grasp pose matrix's
            z-axis by default (see ``approach_axis`` in the loader).
        matrix: Full 4x4 grasp pose (object frame) for callers needing the
            full orientation.
    """

    position: np.ndarray
    approach: np.ndarray
    matrix: np.ndarray


def _matrix_to_candidate(
    matrix: Sequence[Sequence[float]], approach_axis: int
) -> GraspCandidate:
    m = np.asarray(matrix, dtype=np.float64)
    rotation = m[:3, :3]
    approach = rotation[:, approach_axis]
    norm = np.linalg.norm(approach)
    if norm < 1e-9:
        approach = np.array([0.0, 0.0, 1.0])
    else:
        approach = approach / norm
    return GraspCandidate(position=m[:3, 3].copy(), approach=approach, matrix=m)


def load_grasp_candidates(
    model_data_path: str | Path,
    approach_axis: int = 2,
) -> List[GraspCandidate]:
    """Parse ``contact_points_pose`` entries from a model_data JSON file.

    Args:
        model_data_path: Path to ``model_dataN.json``.
        approach_axis: Which rotation-matrix column (0/1/2) is the approach
            direction. RoboTwin contact poses default to z-axis (2).

    Raises:
        FileNotFoundError: Missing file.
        ValueError: Malformed content.
    """
    path = Path(model_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"model_data not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("contact_points_pose") or []
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{path} has no contact_points_pose entries")
    return [_matrix_to_candidate(p, approach_axis) for p in poses]


def first_model_data(object_dir: str | Path) -> Optional[Path]:
    """Return the first (lowest-index) model_data JSON in an object directory."""
    directory = Path(object_dir)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("model_data*.json"), key=_model_data_index)
    return candidates[0] if candidates else None


def _model_data_index(path: Path) -> int:
    stem = path.stem  # model_data12
    try:
        return int(stem.rsplit("model_data", 1)[1])
    except (IndexError, ValueError):
        return 0


def candidate_in_world(
    candidate: GraspCandidate,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    scale: float | Sequence[float] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a candidate into the world frame.

    Args:
        candidate: Object-frame candidate.
        object_pos: Object world position ``(3,)``.
        object_quat: Object world orientation ``(w, x, y, z)``.
        scale: Uniform or per-axis model scale applied to the local
            position (RoboTwin ``scale`` field).

    Returns:
        ``(world_position (3,), world_approach (3,))``.
    """
    scale_arr = np.asarray(scale, dtype=np.float64)
    if scale_arr.ndim == 0:
        scale_arr = np.full(3, float(scale_arr))
    local = candidate.position * scale_arr
    world_pos = _quat_rotate(object_quat, local) + np.asarray(
        object_pos, dtype=np.float64
    )
    world_approach = _quat_rotate(object_quat, candidate.approach)
    norm = np.linalg.norm(world_approach)
    if norm > 1e-9:
        world_approach = world_approach / norm
    return world_pos, world_approach


def score_candidates(
    candidates_world: Sequence[Tuple[np.ndarray, np.ndarray]],
    ee_pos: np.ndarray,
    prefer_axis: Optional[np.ndarray] = None,
) -> List[int]:
    """Order candidate indices best-first.

    Score = approach alignment with ``prefer_axis`` (default: world +Z,
    i.e. top-down preference) minus a small reachability penalty
    (distance to the EE). Pure numpy, deterministic.
    """
    if prefer_axis is None:
        prefer = np.array([0.0, 0.0, 1.0])
    else:
        prefer = np.asarray(prefer_axis, dtype=np.float64)
        norm = np.linalg.norm(prefer)
        prefer = prefer / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    ee = np.asarray(ee_pos, dtype=np.float64)

    def _score(index: int) -> float:
        pos, approach = candidates_world[index]
        alignment = float(np.dot(approach, prefer))
        reach = float(np.linalg.norm(pos - ee))
        return alignment - 0.1 * reach

    return sorted(range(len(candidates_world)), key=_score, reverse=True)


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = (float(v) for v in quat)
    u = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
