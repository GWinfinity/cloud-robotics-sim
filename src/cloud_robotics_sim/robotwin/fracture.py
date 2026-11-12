"""Pre-fractured rigid-body helpers for household brittle objects.

Genesis owns all rigid contact and fragment dynamics. This module only creates
watertight Voronoi pieces offline, converts Genesis contact forces to impulse,
and applies the one-way intact-to-fragments transition. The runtime rebuilds
the scene at fracture because Genesis does not support deleting rigid entities
from an already built scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def contact_impulse(
    contacts: Mapping[str, Any],
    dt: float,
    geom_start: int,
    geom_end: int,
) -> float:
    """Return the net contact impulse on an entity for one single-env step."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if "force_a" not in contacts or "force_b" not in contacts:
        return 0.0
    force_a = _array(contacts["force_a"]).reshape(-1, 3)
    force_b = _array(contacts["force_b"]).reshape(-1, 3)
    if len(force_a) == 0:
        return 0.0
    geom_a = _array(contacts["geom_a"]).reshape(-1)
    self_is_a = (geom_a >= geom_start) & (geom_a < geom_end)
    forces = np.where(self_is_a[:, None], force_a, force_b)
    if "valid_mask" in contacts:
        valid = _array(contacts["valid_mask"]).reshape(-1).astype(bool)
        forces = forces[valid]
    if len(forces) == 0:
        return 0.0
    return float(np.linalg.norm(forces.sum(axis=0)) * dt)


@dataclass
class FractureTrigger:
    """Stateful, single-shot impulse threshold with a peak measurement."""

    impulse_threshold: float = 0.08
    peak_impulse: float = 0.0
    fracture_step: int | None = None
    _step: int = 0

    def __post_init__(self) -> None:
        """Validate the positive physical impulse threshold."""
        if self.impulse_threshold <= 0.0:
            raise ValueError("impulse_threshold must be positive")

    @property
    def broken(self) -> bool:
        return self.fracture_step is not None

    def update(
        self,
        contacts: Mapping[str, Any],
        dt: float,
        geom_start: int,
        geom_end: int,
    ) -> bool:
        """Consume one contact snapshot; return True only on the transition."""
        self._step += 1
        impulse = contact_impulse(contacts, dt, geom_start, geom_end)
        self.peak_impulse = max(self.peak_impulse, impulse)
        if self.broken or impulse < self.impulse_threshold:
            return False
        self.fracture_step = self._step
        return True


def fragment_velocities(
    linear_velocity: Any,
    angular_velocity: Any,
    fragment_centers: Any,
) -> np.ndarray:
    """Inherit rigid twist at each fragment center: v + omega x r."""
    linear = _array(linear_velocity).reshape(3)
    angular = _array(angular_velocity).reshape(3)
    centers = _array(fragment_centers).reshape(-1, 3)
    return linear + np.cross(np.broadcast_to(angular, centers.shape), centers)


def _voronoi_cell(seed: np.ndarray, seeds: np.ndarray, bounds: np.ndarray):
    """Construct one bounded 3D Voronoi cell as a convex mesh."""
    import trimesh
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    others = seeds[np.any(seeds != seed, axis=1)]
    normals = 2.0 * (others - seed)
    offsets = np.sum(seed**2) - np.sum(others**2, axis=1)
    lower, upper = bounds
    box_halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0, lower[0]],
            [1.0, 0.0, 0.0, -upper[0]],
            [0.0, -1.0, 0.0, lower[1]],
            [0.0, 1.0, 0.0, -upper[1]],
            [0.0, 0.0, -1.0, lower[2]],
            [0.0, 0.0, 1.0, -upper[2]],
        ]
    )
    halfspaces = np.vstack([np.column_stack([normals, offsets]), box_halfspaces])
    intersections = HalfspaceIntersection(halfspaces, seed).intersections
    hull = ConvexHull(intersections)
    cell = trimesh.Trimesh(vertices=intersections, faces=hull.simplices, process=True)
    trimesh.repair.fix_winding(cell)
    trimesh.repair.fix_normals(cell)
    if not cell.is_volume:
        raise RuntimeError("Voronoi cell is not a positive watertight volume")
    return cell


def default_cup_seeds(mesh: Any) -> np.ndarray:
    """Twelve deterministic seeds distributed around the wall and base."""
    radius_x, height, radius_z = mesh.extents
    radius = min(radius_x, radius_z) / 2.0
    wall_r = radius - 0.001
    wall_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    wall = np.column_stack(
        [
            wall_r * np.cos(wall_angles),
            np.full(8, height * 0.58),
            wall_r * np.sin(wall_angles),
        ]
    )
    base_angles = wall_angles[::2]
    base = np.column_stack(
        [
            radius * 0.4 * np.cos(base_angles),
            np.full(4, height * 0.025),
            radius * 0.4 * np.sin(base_angles),
        ]
    )
    return np.vstack([wall, base])


def voronoi_fragments(mesh: Any, seeds: Any = None) -> list:
    """Intersect a watertight mesh with bounded Voronoi cells via Manifold."""
    try:
        import manifold3d  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Voronoi fragment generation requires the optional 'fracture' extra; "
            "run `uv sync --extra fracture`"
        ) from exc
    import trimesh

    if not mesh.is_watertight or mesh.volume <= 0.0:
        raise ValueError("fracture input must be a watertight positive-volume mesh")
    seed_array = (
        default_cup_seeds(mesh)
        if seeds is None
        else np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    )
    if len(seed_array) < 2:
        raise ValueError("at least two Voronoi seeds are required")
    if len(np.unique(seed_array, axis=0)) != len(seed_array):
        raise ValueError("Voronoi seeds must be unique")
    bounds = mesh.bounds + np.array([[-1e-6], [1e-6]])

    fragments = []
    for seed in seed_array:
        cell = _voronoi_cell(seed, seed_array, bounds)
        fragment = trimesh.boolean.intersection(
            [mesh, cell], engine="manifold", check_volume=True
        )
        if fragment.is_empty or fragment.volume <= 1e-12:
            continue
        fragment.remove_unreferenced_vertices()
        if not fragment.is_watertight:
            raise RuntimeError("Manifold returned a non-watertight fragment")
        fragments.append(fragment)
    if len(fragments) < 2:
        raise RuntimeError("Voronoi split produced fewer than two fragments")
    volume_error = abs(sum(f.volume for f in fragments) - mesh.volume) / mesh.volume
    if volume_error > 0.01:
        raise RuntimeError(f"fragment volume conservation error: {volume_error:.2%}")
    return fragments
