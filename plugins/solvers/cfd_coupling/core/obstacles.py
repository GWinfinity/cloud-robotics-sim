"""Solid-mask rasterization helpers for the CFD obstacle feature.

Genesis-free: NumPy + (lazily imported) trimesh only. A solid mask is a
boolean ``(nx, ny, nz)`` array on the CFD cell centres; ``True`` marks a
cell inside an immersed obstacle (no-slip wall for the flow solver).

The typical workflow for an external CAD model (e.g. exported from CATIA)
is::

    import trimesh
    from plugins.solvers.cfd_coupling.core.obstacles import mask_from_mesh

    mesh = trimesh.load("wing.stl")          # metres, model coordinates
    mask = mask_from_mesh(mesh, domain, cells,
                         transform=... )      # optional world placement
    cfd.set_solid_mask(mask)
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["cell_centers", "mask_from_box", "mask_from_mesh", "combine_masks"]


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "Obstacle rasterization requires trimesh; "
            "install it or pass a precomputed solid mask instead."
        ) from exc
    return trimesh


def cell_centers(
    domain: tuple[float, float, float],
    cells: tuple[int, int, int],
) -> np.ndarray:
    """World-space centre coordinates of every cell, shape ``cells + (3,)``."""
    lx, ly, lz = domain
    nx, ny, nz = cells
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * (lx / nx)
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * (ly / ny)
    zs = (np.arange(nz, dtype=np.float64) + 0.5) * (lz / nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def mask_from_box(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    domain: tuple[float, float, float],
    cells: tuple[int, int, int],
) -> np.ndarray:
    """Analytic solid mask of an axis-aligned box (tests' ground truth)."""
    centers = cell_centers(domain, cells)
    c = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(size, dtype=np.float64)
    return np.all(np.abs(centers - c) <= half, axis=-1)


def mask_from_mesh(
    mesh: Any,
    domain: tuple[float, float, float],
    cells: tuple[int, int, int],
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize a triangle mesh into a cell-centre solid mask.

    Parameters
    ----------
    mesh :
        A ``trimesh.Trimesh`` in model coordinates (metres).
    domain, cells :
        CFD grid definition (same values as ``CFDOptions.domain/cells``).
    transform :
        Optional 4x4 homogeneous matrix placing the mesh into the CFD world
        frame (applied on top of the mesh's own coordinates).

    Containment test: exact winding/ray-cast containment for watertight
    meshes; signed distance < h/2 otherwise; axis-aligned bounding box as the
    last-resort fallback so that open (non-watertight) sheets still block
    flow. Only cells near the mesh AABB are queried.
    """
    trimesh = _require_trimesh()
    m = mesh.copy()
    if transform is not None:
        m.apply_transform(np.asarray(transform, dtype=np.float64))

    h = min(L / n for L, n in zip(domain, cells, strict=True))
    centers = cell_centers(domain, cells)
    flat = centers.reshape(-1, 3)

    mn, mx = np.asarray(m.bounds, dtype=np.float64)
    near = np.all((flat >= mn - h) & (flat <= mx + h), axis=-1)
    mask = np.zeros(cells, dtype=bool)
    pts = flat[near]
    if pts.size == 0:
        return mask

    if m.is_watertight:
        inside = m.contains(pts)
    else:
        try:
            sd = trimesh.proximity.signed_distance(m, pts)
            inside = np.asarray(sd) < 0.5 * h
        except Exception:
            inside = np.all((pts >= mn) & (pts <= mx), axis=-1)
    mask.reshape(-1)[near] = np.asarray(inside, dtype=bool)
    return mask


def combine_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    """Union of several masks; ``None`` entries are skipped."""
    present = [m for m in masks if m is not None]
    if not present:
        return None
    out = np.asarray(present[0], dtype=bool).copy()
    for m in present[1:]:
        out |= np.asarray(m, dtype=bool)
    return out
