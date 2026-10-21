"""Static-scene SDF baking: triangle soups -> one narrow-band level set grid.

Shared foundation for the static three-way classification (see AGENTS.md):
contact surfaces are substituted by primitives, non-contact scenery is
dropped from the physics scene and baked here into a single queryable SDF
(:class:`~cloud_robotics_sim.vdb.XuvdbVolume`). Dynamic bodies and articulated
furniture never pass through this module.

The bake is a thin orchestration over xuvdb's ``mesh_to_level_set`` (exact
point-to-triangle distance + global +X ray parity; watertight meshes assumed
for correct interior signs). All meshes are transformed into the world frame
first and merged into one triangle soup, so parity is resolved globally.

Caching follows the project convention (``CRS_SIM_CACHE_DIR`` /
``outputs/sim_cache`` via :func:`cloud_robotics_sim.utils.cache_cleanup.get_cache_dir`):
the cache key hashes input paths, mtimes, poses and bake parameters; a hit
reloads the ``.xuvdb`` grid instead of re-baking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cloud_robotics_sim.utils.cache_cleanup import get_cache_dir
from cloud_robotics_sim.vdb.core.vdb_xuvdb import (
    XuvdbVolume,
    _require_xuvdb,
    load_xuvdb,
    save_xuvdb,
)

__all__ = [
    "bake_static_scene",
    "cache_key_for_bake",
    "default_vdb_cache_dir",
    "meshes_to_sdf",
]

logger = logging.getLogger(__name__)


def default_vdb_cache_dir() -> Path:
    """Default on-disk cache root for baked SDF grids (``<sim_cache>/vdb``)."""
    return get_cache_dir() / "vdb"


def meshes_to_sdf(
    entries: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    voxel_size: float,
    band: float = 3.0,
    name: str = "static_scene",
) -> XuvdbVolume:
    """Merge world-frame triangle soups into one narrow-band level set grid.

    Args:
        entries: ``(vertices (n, 3), faces (m, 3))`` pairs, already in world
            coordinates. Meshes must be watertight for correct interior signs.
        voxel_size: Isotropic voxel edge length (world units).
        band: Half-width of the narrow band, in voxels.
        name: Grid name.

    Returns:
        The baked :class:`XuvdbVolume` (``grid_class="level set"``,
        ``background = band * voxel_size``).
    """
    x = _require_xuvdb()
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for v, f in entries:
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"expected (n, 3) vertices, got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"expected (m, 3) faces, got {f.shape}")
        if f.size == 0:
            logger.debug("skipping mesh entry with no faces (%d verts)", len(v))
            continue
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not verts:
        raise ValueError("no mesh entries to bake")
    soup_v = np.concatenate(verts, axis=0)
    soup_f = np.concatenate(faces, axis=0)
    grid = x.mesh_to_level_set(
        soup_v,
        soup_f,
        float(voxel_size),
        band=float(band),
        origin_world=(0.0, 0.0, 0.0),
        name=name,
    )
    return XuvdbVolume(grid=grid)


def _load_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - trimesh is a core dep
        raise ImportError(
            "bake_static_scene requires trimesh (core dependency); "
            f"original error: {exc}"
        ) from exc
    return trimesh


def _mesh_to_world_arrays(
    path: Path, transform: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh file and return world-frame ``(vertices, faces)``."""
    trimesh = _load_trimesh()
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        # Concatenate with world transforms baked in (dump() is deprecated).
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if transform is not None:
        t = np.asarray(transform, dtype=np.float64)
        if t.shape != (4, 4):
            raise ValueError(f"expected (4, 4) transform, got {t.shape}")
        v = v @ t[:3, :3].T + t[:3, 3]
    return v, f


def cache_key_for_bake(
    mesh_files: Sequence[str | Path],
    world_transforms: Sequence[np.ndarray] | None,
    *,
    voxel_size: float,
    band: float,
    name: str,
) -> str:
    """Stable cache key: input paths, mtimes, poses and bake parameters."""
    hasher = hashlib.sha1()
    hasher.update(
        json.dumps(
            {
                "version": 1,
                "voxel_size": float(voxel_size),
                "band": float(band),
                "name": str(name),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for i, p in enumerate(mesh_files):
        path = Path(p)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = None
        t = None
        if world_transforms is not None:
            t = np.asarray(world_transforms[i], dtype=np.float64).round(9).tolist()
        hasher.update(json.dumps([str(path.resolve()), mtime, t]).encode("utf-8"))
    return hasher.hexdigest()[:16]


def bake_static_scene(
    mesh_files: Sequence[str | Path],
    *,
    world_transforms: Sequence[np.ndarray] | None = None,
    voxel_size: float,
    band: float = 3.0,
    cache_dir: str | Path | None = None,
    name: str = "static_scene",
) -> XuvdbVolume:
    """Bake mesh files into a single SDF grid, with on-disk caching.

    Args:
        mesh_files: Mesh files (GLB/STL/OBJ/...) loadable by trimesh, in any
            unit the caller's world frame uses (units are *not* converted here).
        world_transforms: Optional per-file 4x4 world transforms (identity
            when omitted). Vertices are transformed before merging.
        voxel_size: Isotropic voxel edge length (world units).
        band: Narrow-band half-width, in voxels.
        cache_dir: Cache directory (default :func:`default_vdb_cache_dir`).
            A cache hit reloads the ``.xuvdb`` file and skips the bake.
        name: Grid name (also part of the cache key).

    Returns:
        The baked (or cached) :class:`XuvdbVolume`.
    """
    files = [Path(p) for p in mesh_files]
    if not files:
        raise ValueError("no mesh files to bake")
    if world_transforms is not None and len(world_transforms) != len(files):
        raise ValueError(
            f"world_transforms length {len(world_transforms)} != "
            f"{len(files)} mesh files"
        )
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(f"mesh file not found: {p}")

    cache_root = Path(cache_dir) if cache_dir is not None else default_vdb_cache_dir()
    key = cache_key_for_bake(
        files, world_transforms, voxel_size=voxel_size, band=band, name=name
    )
    cache_file = cache_root / f"{name}-{key}.xuvdb"
    if cache_file.is_file():
        logger.info("static SDF cache hit: %s", cache_file)
        volumes = load_xuvdb(cache_file)
        if len(volumes) != 1:
            raise ValueError(f"expected 1 grid in {cache_file}, got {len(volumes)}")
        return volumes[0]

    t0 = time.perf_counter()
    entries = [
        _mesh_to_world_arrays(
            p, None if world_transforms is None else world_transforms[i]
        )
        for i, p in enumerate(files)
    ]
    vol = meshes_to_sdf(entries, voxel_size=voxel_size, band=band, name=name)
    elapsed = time.perf_counter() - t0
    report = vol.memory_report()
    logger.info(
        "baked static SDF %s in %.1fs (%d leaves, %d active voxels, %.1f MB leaf buffers)",
        name,
        elapsed,
        report["n_leaves"],
        report["active_voxels"],
        report["leaf_bytes"] / 1e6,
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    save_xuvdb(cache_file, [vol], compress=True)
    return vol
