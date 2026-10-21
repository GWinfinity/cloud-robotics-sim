# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""WFC layout -> static SDF bake (offline; no Genesis required).

Reuses :meth:`ChineseHomeScene.build_trimesh` (the same whole-home mesh that
:meth:`ChineseHomeScene.export_glb` writes) and feeds it to the shared
static-scene foundation (:func:`cloud_robotics_sim.vdb.meshes_to_sdf`). The
result is a query-only level set of walls/floors/furniture — a unified SDF
for planners and the agent layer, complementing the Genesis primitive scene
rather than replacing it.

The optional ShapeKernel bridge (:func:`shapekernel_prims_to_sdf`) converts
parametric LEAP71-style shapes (lattices, base shapes, modulations) into the
same meter-frame SDF via ``vox_construct`` + exact uniform scaling of the
level set (SDF values, voxel size, origin and background all scale linearly).
ShapeKernel works in millimeters; the bridge divides by ``mm_to_m``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

try:  # Relative import inside the installed plugin package.
    from .assets.furniture_cn import TrimeshBackend
    from .core.rules_cn import TILES
    from .scene import ChineseHomeScene
except ImportError:  # Plugin dir itself on sys.path.
    from assets.furniture_cn import TrimeshBackend  # type: ignore[no-redef]
    from core.rules_cn import TILES  # type: ignore[no-redef]
    from scene import ChineseHomeScene  # type: ignore[no-redef]

from cloud_robotics_sim.vdb.core.static_scene import (  # noqa: E402
    default_vdb_cache_dir,
    meshes_to_sdf,
)
from cloud_robotics_sim.vdb.core.vdb_xuvdb import XuvdbVolume, load_xuvdb, save_xuvdb

__all__ = [
    "bake_layout",
    "bake_room_variant",
    "bake_structure",
    "rescale_grid",
    "shapekernel_prims_to_sdf",
]

logger = logging.getLogger(__name__)

LayoutLike = Union[str, Path, dict, Any]


def _layout_dict(layout_like: LayoutLike) -> dict:
    """Coerce a layout reference to its replayable JSON dict."""
    if isinstance(layout_like, ChineseHomeScene):
        return layout_like.layout.to_dict()
    if isinstance(layout_like, dict):
        return layout_like
    path = Path(layout_like)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bake_layout(
    layout: LayoutLike,
    *,
    voxel_size: float = 0.05,
    band: float = 3.0,
    cache_dir: Union[str, Path, None] = None,
    name: Optional[str] = None,
) -> XuvdbVolume:
    """Bake a WFC Chinese-home layout into a query-only SDF volume.

    Args:
        layout: A :class:`ChineseHomeScene`, a layout JSON path, or a layout
            dict (seed-replayable, see :meth:`ChineseHomeScene.export_layout`).
        voxel_size: Isotropic voxel edge length in meters.
        band: Narrow-band half-width, in voxels.
        cache_dir: On-disk cache root (default the project vdb cache). The
            cache key hashes the layout JSON and bake parameters.
        name: Grid/cache name (default derived from the layout seed).

    Returns:
        The baked (or cached) :class:`XuvdbVolume`.
    """
    layout_dict = _layout_dict(layout)
    seed = layout_dict.get("seed", "x")
    grid_name = name or f"wfc-{seed}"

    cache_root = Path(cache_dir) if cache_dir is not None else default_vdb_cache_dir()
    key = hashlib.sha1(
        json.dumps(
            {
                "version": 1,
                "layout": layout_dict,
                "voxel_size": float(voxel_size),
                "band": float(band),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_root / f"{grid_name}-{key}.xuvdb"
    if cache_file.is_file():
        logger.info("layout SDF cache hit: %s", cache_file)
        volumes = load_xuvdb(cache_file)
        if len(volumes) != 1:
            raise ValueError(f"expected 1 grid in {cache_file}, got {len(volumes)}")
        return volumes[0]

    home = (
        layout
        if isinstance(layout, ChineseHomeScene)
        else ChineseHomeScene(layout=layout_dict)
    )
    # Concatenate with world transforms baked in (dump() is deprecated).
    mesh = home.build_trimesh().to_geometry()
    vol = meshes_to_sdf(
        [(np.asarray(mesh.vertices), np.asarray(mesh.faces))],
        voxel_size=voxel_size,
        band=band,
        name=grid_name,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    save_xuvdb(cache_file, [vol], compress=True)
    return vol


def _prims_to_sdf(
    prims: Sequence[Any], *, voxel_size: float, band: float, name: str
) -> XuvdbVolume:
    """Bake world-frame prims into one narrow-band level set grid."""
    backend = TrimeshBackend()
    for prim in prims:
        backend.add(prim)
    # Concatenate with world transforms baked in (dump() is deprecated).
    mesh = backend.scene.to_geometry()
    return meshes_to_sdf(
        [(np.asarray(mesh.vertices), np.asarray(mesh.faces))],
        voxel_size=voxel_size,
        band=band,
        name=name,
    )


def bake_structure(
    layout: LayoutLike,
    *,
    voxel_size: float = 0.05,
    band: float = 3.0,
    cache_dir: Union[str, Path, None] = None,
    name: Optional[str] = None,
) -> XuvdbVolume:
    """Bake the home shell (walls/floors/slabs, no furniture) into an SDF.

    This is the fixed half of room-content domain randomization: the shell
    is baked once per layout, per-room furniture variants are baked by
    :func:`bake_room_variant`, and query layers compose them by elementwise
    min (exact SDF union).
    """
    layout_dict = _layout_dict(layout)
    seed = layout_dict.get("seed", "x")
    grid_name = name or f"wfc-{seed}-structure"

    cache_root = Path(cache_dir) if cache_dir is not None else default_vdb_cache_dir()
    key = hashlib.sha1(
        json.dumps(
            {
                "version": 1,
                "kind": "structure",
                "layout": layout_dict,
                "voxel_size": float(voxel_size),
                "band": float(band),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_root / f"{grid_name}-{key}.xuvdb"
    if cache_file.is_file():
        logger.info("structure SDF cache hit: %s", cache_file)
        volumes = load_xuvdb(cache_file)
        if len(volumes) != 1:
            raise ValueError(f"expected 1 grid in {cache_file}, got {len(volumes)}")
        return volumes[0]

    home = (
        layout
        if isinstance(layout, ChineseHomeScene)
        else ChineseHomeScene(layout=layout_dict)
    )
    vol = _prims_to_sdf(
        home.structure_prims(), voxel_size=voxel_size, band=band, name=grid_name
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    save_xuvdb(cache_file, [vol], compress=True)
    return vol


def bake_room_variant(
    layout: LayoutLike,
    cell_index: int,
    tile: str,
    *,
    voxel_size: float = 0.05,
    band: float = 3.0,
    cache_dir: Union[str, Path, None] = None,
    name: Optional[str] = None,
) -> XuvdbVolume:
    """Bake one room's furniture content (dressed as ``tile``) into an SDF.

    The grid is the level set of *that room's furniture only*, positioned at
    the cell's world location — walls/floors stay in the structure bake.
    Query layers compose by elementwise min (SDF union), so any number of
    per-room variants combine without re-baking the shell.

    Args:
        layout: A :class:`ChineseHomeScene`, a layout JSON path, or a dict.
        cell_index: Index into ``layout.cells`` (same numbering as
            ``ChineseHomeScene.furniture_prims(tile_overrides=...)``).
        tile: Room type to dress the cell as (see ``rules_cn.TILES``).
        voxel_size: Isotropic voxel edge length in meters.
        band: Narrow-band half-width, in voxels.
        cache_dir: On-disk cache root (default the project vdb cache). The
            cache key hashes the layout JSON, cell index, tile and params.
        name: Grid/cache name (default derived from seed/cell/tile).

    Raises:
        ValueError: Unknown tile, out-of-range cell index, or a tile that
            produces no furniture prims for this cell.
    """
    if tile not in TILES:
        raise ValueError(f"unknown tile {tile!r}; available: {sorted(TILES)}")

    layout_dict = _layout_dict(layout)
    seed = layout_dict.get("seed", "x")
    grid_name = name or f"wfc-{seed}-cell{cell_index}-{tile}"

    cache_root = Path(cache_dir) if cache_dir is not None else default_vdb_cache_dir()
    key = hashlib.sha1(
        json.dumps(
            {
                "version": 1,
                "kind": "room_variant",
                "layout": layout_dict,
                "cell_index": int(cell_index),
                "tile": tile,
                "voxel_size": float(voxel_size),
                "band": float(band),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_root / f"{grid_name}-{key}.xuvdb"
    if cache_file.is_file():
        logger.info("room-variant SDF cache hit: %s", cache_file)
        volumes = load_xuvdb(cache_file)
        if len(volumes) != 1:
            raise ValueError(f"expected 1 grid in {cache_file}, got {len(volumes)}")
        return volumes[0]

    home = (
        layout
        if isinstance(layout, ChineseHomeScene)
        else ChineseHomeScene(layout=layout_dict)
    )
    cells = home.layout.cells
    if not 0 <= int(cell_index) < len(cells):
        raise ValueError(f"cell_index {cell_index} out of range for {len(cells)} cells")
    prims = home._cell_furniture_prims(int(cell_index), cells[int(cell_index)], tile)
    if not prims:
        raise ValueError(
            f"tile {tile!r} produced no furniture prims for cell "
            f"{cell_index}; pick a furnished variant"
        )
    vol = _prims_to_sdf(prims, voxel_size=voxel_size, band=band, name=grid_name)
    cache_root.mkdir(parents=True, exist_ok=True)
    save_xuvdb(cache_file, [vol], compress=True)
    return vol


def rescale_grid(grid: Any, factor: float) -> Any:
    """Uniformly rescale a level-set grid in place (exact for SDFs).

    Multiplies values, voxel size, origin and background by ``factor`` —
    the level set of the scaled geometry, e.g. mm -> m via ``factor=1e-3``.
    """
    f = float(factor)
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"factor must be positive, got {factor!r}")
    if f == 1.0:
        return grid
    for leaf in grid.leaves():
        leaf.values[...] = leaf.values * f
        leaf.invalidate()
    grid.background = grid.background * f
    grid.voxel_size = tuple(float(s) * f for s in grid.voxel_size)
    grid.origin_world = tuple(float(o) * f for o in grid.origin_world)
    return grid


def shapekernel_prims_to_sdf(
    shapes: Sequence[Any],
    *,
    voxel_size: float = 0.002,
    band: float = 3.0,
    mm_to_m: float = 1000.0,
    name: str = "shapekernel_scene",
) -> XuvdbVolume:
    """Convert ShapeKernel base shapes into a meter-frame SDF volume.

    Each shape must provide ``vox_construct(voxel_size=...)`` returning a
    xuvdb ``VdbGrid`` (ShapeKernel's geometry backend). The shapes are
    unioned (min) at the ShapeKernel voxel size, then uniformly rescaled to
    meters.

    Args:
        shapes: ShapeKernel shape instances (e.g. ``BaseBox``).
        voxel_size: Voxel size in the ShapeKernel (millimeter) world.
        band: Narrow-band half-width, in voxels.
        mm_to_m: Millimeters per meter (default 1000).
        name: Grid name.

    Returns:
        The meter-frame :class:`XuvdbVolume`.

    Raises:
        ImportError: If shapekernel is not installed (names the extra).
    """
    try:
        import shapekernel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "shapekernel_prims_to_sdf requires the sibling 'shapekernel' "
            "package (LEAP71 ShapeKernel Python port); install it with "
            "`uv sync --extra shapekernel` from the project root "
            f"([tool.uv.sources] points at ../shapekernel); original error: {exc}"
        ) from exc
    if not shapes:
        raise ValueError("no shapes to convert")
    merged = None
    for shape in shapes:
        grid = shape.vox_construct(voxel_size=float(voxel_size))
        if merged is None:
            merged = grid
        else:
            merged.csg(grid, "union")
    assert merged is not None
    rescale_grid(merged, 1.0 / float(mm_to_m))
    merged.name = name
    return XuvdbVolume(grid=merged)
