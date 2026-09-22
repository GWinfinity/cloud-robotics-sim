"""Solid-mask rasterization helpers for the CFD obstacle feature.

Genesis-free: NumPy + (lazily imported) trimesh only. A solid mask is a
boolean ``(nx, ny, nz)`` array on the CFD cell centres; ``True`` marks a
cell inside an immersed obstacle (no-slip wall for the flow solver).

Also hosts the surface-force integration used for aerodynamic loads:
pressure forces from the (specific) pressure field on mask boundary faces
plus viscous wall-shear forces from the cell-centred velocity.

The typical workflow for an external CAD model (e.g. exported from CATIA)
is::

    import trimesh
    from plugins.solvers.cfd_coupling.core.obstacles import mask_from_mesh

    mesh = trimesh.load("wing.stl")          # metres, model coordinates
    mask = mask_from_mesh(mesh, domain, cells,
                         transform=... )      # optional world placement
    cfd.set_solid_mask(mask)

Assemblies: a STEP file (AP203/AP214, incl. multi-solid assemblies exported
from CATIA) is parsed through cadquery (OpenCASCADE) into one mesh per
solid::

    from plugins.solvers.cfd_coupling.core.obstacles import meshes_from_step

    parts = meshes_from_step("airframe.stp")   # one trimesh per solid
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "cell_centers",
    "combine_masks",
    "mask_from_box",
    "mask_from_mesh",
    "mask_from_step",
    "meshes_from_step",
    "surface_forces",
]


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


# --------------------------------------------------------------------- #
# STEP assemblies (CATIA / any OCC-supported CAD export)
# --------------------------------------------------------------------- #
def meshes_from_step(
    path: str | Path,
    tolerance: float = 1.0e-3,
    angular_tolerance: float = 0.1,
) -> list[Any]:
    """Parse a STEP file into one watertight-ish mesh per solid.

    Multi-solid files (CATIA products / assemblies exported as STEP with
    multiple roots) yield one ``trimesh.Trimesh`` per solid, already in the
    STEP world frame. Tessellation tolerance in model units (metres).

    Requires the optional ``cadquery`` package (OpenCASCADE bindings)::

        pip install cadquery
    """
    trimesh = _require_trimesh()
    try:
        from cadquery import importers
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "STEP parsing requires the optional 'cadquery' package "
            "(OpenCASCADE Python bindings); install it with "
            "'pip install cadquery', or export STL from your CAD tool "
            "and use mask_from_mesh instead. CATPart/CATProduct are "
            "proprietary and cannot be parsed directly - export STEP "
            "or STL from CATIA."
        ) from exc

    shape = importers.importStep(str(path)).val()
    meshes = []
    for solid in shape.Solids():
        verts, faces = solid.tessellate(tolerance, angular_tolerance)
        v = np.array([[vec.x, vec.y, vec.z] for vec in verts], dtype=np.float64)
        meshes.append(
            trimesh.Trimesh(vertices=v, faces=np.asarray(faces), process=True)
        )
    if not meshes:
        raise ValueError(f"no solids found in STEP file: {path}")
    return meshes


def mask_from_step(
    path: str | Path,
    domain: tuple[float, float, float],
    cells: tuple[int, int, int],
    tolerance: float = 1.0e-3,
    angular_tolerance: float = 0.1,
    transform: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rasterize every solid of a STEP assembly.

    Returns ``(union_mask, part_masks)`` where ``part_masks`` has one entry
    per solid (for per-part force breakdown).
    """
    parts = [
        mask_from_mesh(m, domain, cells, transform=transform)
        for m in meshes_from_step(path, tolerance, angular_tolerance)
    ]
    union = combine_masks(*parts)
    assert union is not None
    return union, parts


# --------------------------------------------------------------------- #
# Surface-force integration (aerodynamic loads on a solid mask)
# --------------------------------------------------------------------- #
def surface_forces(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    h: float,
    rho: float,
    nu: float,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Integrate fluid forces on an immersed solid (Newton, world axes).

    Pressure: each fluid cell adjacent to a solid neighbour pushes the solid
    with ``rho * p[c] * h^2`` along the fluid->solid direction (``p`` is the
    solver's specific pressure p/rho, hence the density factor).

    Viscous wall shear: the boundary face is a no-slip wall (face velocity
    pinned to 0) at distance h/2 from the fluid cell centre, so
    ``tau = rho*nu * U_tan / (h/2)``; the tangential cell-centred velocity
    ``U_tan`` (normal component removed) drags the solid along.

    For a uniform pressure field the net pressure force is exactly zero
    (closed surface); for a linear ramp ``p = gx*x`` it equals
    ``-grad(p) * V_solid`` - both are pinned in the tests.
    """
    mask = np.asarray(mask, dtype=bool)
    fluid = ~mask
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    area = h * h
    fp = np.zeros(3)
    fv = np.zeros(3)
    shear = rho * nu * (2.0 / h) * area
    for dim in range(3):
        n = mask.shape[dim]
        for off, sign in ((1, 1.0), (-1, -1.0)):
            f_sl = [slice(None)] * 3
            s_sl = [slice(None)] * 3
            if off == 1:
                f_sl[dim] = slice(0, n - 1)  # fluid cell
                s_sl[dim] = slice(1, n)  # solid neighbour
            else:
                f_sl[dim] = slice(1, n)
                s_sl[dim] = slice(0, n - 1)
            pair = fluid[tuple(f_sl)] & mask[tuple(s_sl)]
            if not pair.any():
                continue
            fp[dim] += sign * float((rho * p[tuple(f_sl)][pair] * area).sum())
            vel = np.stack(
                [
                    uc[tuple(f_sl)][pair],
                    vc[tuple(f_sl)][pair],
                    wc[tuple(f_sl)][pair],
                ],
                axis=-1,
            )
            vel[:, dim] = 0.0  # tangential component only
            fv += shear * vel.sum(axis=0)
    return {
        "pressure": fp,
        "viscous": fv,
        "total": fp + fv,
    }
