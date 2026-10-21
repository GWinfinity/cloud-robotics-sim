# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Room-content domain randomization over baked WFC SDF grids (offline).

The fixed home shell (walls/floors) is baked once by
:func:`bake_structure`; each randomized room's furniture is baked as
independent narrow-band grids by :func:`bake_room_variant`, positioned at
the cells' world locations. :class:`RandomizedRoomSDF` packs every grid into
one xuvdb ``VolumeBatch`` and composes queries by elementwise min — the
exact SDF union of shell ∪ furniture:

* ``sample(points, ids)`` resolves every point against its per-point variant
  in a single kernel launch (occupancy / distance fields for planners).
* ``cast(origins, directions, ids)`` groups rays by variant, casts each
  group against shell + its variant, and keeps the nearest hit (depth
  cameras / lidar).

The whole layer lives **outside** the Genesis scene. Genesis' single-scene
batching is structurally homogeneous — every env shares one built scene —
but the randomization happens on the query side here, so every env in a
batched scene can "see" a different room contents through its own volume
ids, with zero scene rebuilds.

Usage::

    home = ChineseHomeScene(grid=(3, 2), seed=42)
    room = next(
        k for k, c in enumerate(home.layout.cells) if c["tile"] == "living"
    )
    rsdf = RandomizedRoomSDF.build(
        home, {room: ["kitchen", "bedroom", "study"]}
    )
    ids = np.array(
        [rsdf.id_for(room, "kitchen"), rsdf.id_for(room, "bedroom")]
    )
    d = rsdf.sample(points, ids)            # per-point room content
    t, hit, _ = rsdf.cast(origins, dirs, ids)

Thin-wall caveat (same as the static-scene bake): walls 1-2 voxels thick
lose their negative interiors to trilinear sampling — occupancy judgements
must census leaf values, not random linear samples. Ray casting from
outside is unaffected (the first surface crossing is what matters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

try:  # Relative import inside the installed plugin package.
    from .core.rules_cn import TILES
    from .scene import ChineseHomeScene
    from .sdf_bake import (
        LayoutLike,
        _layout_dict,
        bake_room_variant,
        bake_structure,
    )
except ImportError:  # Plugin dir itself on sys.path.
    from core.rules_cn import TILES  # type: ignore[no-redef]
    from scene import ChineseHomeScene  # type: ignore[no-redef]
    from sdf_bake import (  # type: ignore[no-redef]
        LayoutLike,
        _layout_dict,
        bake_room_variant,
        bake_structure,
    )

from cloud_robotics_sim.vdb.core.vdb_xuvdb import (  # noqa: E402
    XuvdbVolume,
    _require_xuvdb,
)
from cloud_robotics_sim.vdb.ray_caster import SdfRayCaster  # noqa: E402

__all__ = ["RandomizedRoomSDF"]


class RandomizedRoomSDF:
    """Fixed WFC shell + per-room furniture variants, composed as one SDF.

    Volume id convention: ``0`` is always the structure shell; ``1..N`` are
    room variants in sorted-``(cell_index, tile)`` order — resolve ids with
    :meth:`id_for` instead of hard-coding.
    """

    STRUCTURE_ID = 0

    def __init__(
        self,
        grids: Sequence[XuvdbVolume],
        id_map: Mapping[tuple[int, str], int],
        cells: Mapping[int, Sequence[str]],
    ) -> None:
        if len(grids) < 2:
            raise ValueError("need a structure grid plus at least one variant")
        self._xuvdb = _require_xuvdb()
        self._grids = list(grids)
        self._batch = self._xuvdb.VolumeBatch([g.grid for g in self._grids])
        self._casters = [SdfRayCaster(g) for g in self._grids]
        self._id_map = dict(id_map)
        self.cells = {int(k): list(v) for k, v in cells.items()}

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        layout: LayoutLike,
        variants: Mapping[int, Sequence[str]],
        *,
        voxel_size: float = 0.05,
        band: float = 3.0,
        cache_dir: Union[str, Path, None] = None,
        name: Optional[str] = None,
    ) -> "RandomizedRoomSDF":
        """Bake (or cache-load) the shell and variants, then pack the batch.

        Args:
            layout: WFC home defining the fixed shell and cell geometry
                (:class:`ChineseHomeScene`, layout JSON path, or dict).
            variants: Mapping of cell index (position in ``layout.cells``)
                to the room types that cell can be dressed as
                (客厅/厨房/卧室/..., see ``rules_cn.TILES``). Each entry
                bakes one grid at the cell's world position.
            voxel_size: Isotropic voxel edge length in meters.
            band: Narrow-band half-width, in voxels.
            cache_dir: On-disk cache root forwarded to the bake functions
                (each grid is keyed by layout JSON + cell + tile + params).
            name: Grid/cache name prefix (default derived from the seed).

        Raises:
            ValueError: Empty ``variants``, unknown tile, or out-of-range
                cell index.
        """
        if not variants:
            raise ValueError("no room variants given")
        layout_dict = _layout_dict(layout)
        home = (
            layout
            if isinstance(layout, ChineseHomeScene)
            else ChineseHomeScene(layout=layout_dict)
        )
        cells = home.layout.cells
        base = name or f"wfc-{layout_dict.get('seed', 'x')}"

        structure = bake_structure(
            home,
            voxel_size=voxel_size,
            band=band,
            cache_dir=cache_dir,
            name=f"{base}-structure",
        )
        grids: list[XuvdbVolume] = [structure]
        id_map: dict[tuple[int, str], int] = {}
        cells_out: dict[int, list[str]] = {}
        for k in sorted(int(c) for c in variants):
            names = list(variants[k])
            if not names:
                raise ValueError(f"no variant names given for cell {k}")
            if not 0 <= k < len(cells):
                raise ValueError(f"cell index {k} out of range for {len(cells)} cells")
            if any(t not in TILES for t in names):
                bad = sorted(set(names) - set(TILES))
                raise ValueError(
                    f"unknown tile(s) {bad} for cell {k}; "
                    f"available: {sorted(TILES)}"
                )
            cells_out[k] = []
            for tile in names:
                vol = bake_room_variant(
                    home,
                    k,
                    tile,
                    voxel_size=voxel_size,
                    band=band,
                    cache_dir=cache_dir,
                    name=f"{base}-cell{k}-{tile}",
                )
                id_map[(k, tile)] = len(grids)
                grids.append(vol)
                cells_out[k].append(tile)
        return cls(grids, id_map, cells_out)

    # ------------------------------------------------------------------ ids

    def id_for(self, cell_index: int, tile: str) -> int:
        """Volume id of ``cell`` dressed as ``tile`` (for ``sample``/``cast``)."""
        try:
            return self._id_map[(int(cell_index), tile)]
        except KeyError:
            raise KeyError(
                f"no baked variant for cell {cell_index} as {tile!r}; "
                f"available: {sorted(self._id_map)}"
            ) from None

    @property
    def n_variants(self) -> int:
        return len(self._grids) - 1

    def memory_report(self) -> dict:
        reports = [g.memory_report() for g in self._grids]
        return {
            "n_grids": len(reports),
            "n_leaves": sum(r["n_leaves"] for r in reports),
            "active_voxels": sum(r["active_voxels"] for r in reports),
            "leaf_bytes": sum(r["leaf_bytes"] for r in reports),
            "grids": reports,
        }

    # ------------------------------------------------------------------ queries

    def sample(
        self, points: np.ndarray, ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Min-composed SDF value at world points.

        Args:
            points: ``(m, 3)`` world-frame points.
            ids: Optional per-point variant id (:meth:`id_for` values);
                ``None`` queries the bare structure shell.

        Returns:
            ``(m,)`` values — elementwise ``min(structure, variant)`` per
            point (exact SDF union of shell and the point's room content).
        """
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        zeros = np.zeros(len(pts), dtype=np.int32)
        struct_vals = np.asarray(self._batch.sample(pts, zeros), dtype=np.float64)
        if ids is None:
            return struct_vals
        vol_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        if len(vol_ids) != len(pts):
            raise ValueError(f"ids length {len(vol_ids)} != {len(pts)} points")
        if len(vol_ids) and (
            int(vol_ids.min()) < 1 or int(vol_ids.max()) >= len(self._grids)
        ):
            raise IndexError("variant id out of range; use id_for()")
        var_vals = np.asarray(self._batch.sample(pts, vol_ids), dtype=np.float64)
        return np.minimum(struct_vals, var_vals)

    def cast(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rays against structure ∪ per-point variant; nearest hit wins.

        Rays are grouped by variant id (one cast launch per distinct id
        present, plus one for the structure), so a batch of rays addressed
        to different envs' room contents stays a handful of launches.

        Args:
            origins: ``(n, 3)`` world-frame ray origins.
            directions: ``(n, 3)`` world-frame directions.
            ids: Optional per-ray variant id (:meth:`id_for` values);
                ``None`` casts against the bare structure shell.

        Returns:
            ``(t, points, values)`` — ``t`` is ``(n,)`` with ``inf`` for
            misses; ``points``/``values`` are ``(n, 3)``/``(n,)`` with zeros
            for misses.
        """
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if len(o) != len(d):
            raise ValueError(
                f"origins/directions ray count mismatch: {len(o)} vs {len(d)}"
            )
        t, pts, vals = self._casters[self.STRUCTURE_ID].cast(o, d)
        if ids is None or len(o) == 0:
            return t, pts, vals
        ray_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(ray_ids) != len(o):
            raise ValueError(f"ids length {len(ray_ids)} != {len(o)} rays")
        if int(ray_ids.min()) < 1 or int(ray_ids.max()) >= len(self._casters):
            raise IndexError("variant id out of range; use id_for()")
        for vid in np.unique(ray_ids):
            mask = ray_ids == vid
            tv, pv, vv = self._casters[int(vid)].cast(o[mask], d[mask])
            better = tv < t[mask]
            if not bool(better.any()):
                continue
            tm = t[mask]
            tm[better] = tv[better]
            t[mask] = tm
            pm = pts[mask]
            pm[better] = pv[better]
            pts[mask] = pm
            vm = vals[mask]
            vm[better] = vv[better]
            vals[mask] = vm
        return t, pts, vals
