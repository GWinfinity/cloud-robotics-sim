"""Batch SDF ray casting against a baked static-scene grid.

Thin, stateless-per-call wrapper over xuvdb's kernel-side DDA
(:meth:`xuvdb.GpuVolume.ray_surface_hit`): one kernel launch carries every
ray, misses come back as ``t=inf`` in a regular ``(n,)`` array alongside
``(n, 3)`` hit points and values — ready for numpy/torch downstream without
per-ray ``None`` handling.

Constraints inherited from xuvdb kernels (validated eagerly with clear
messages): float32 scalar grids, axis-aligned only. Points and directions
are world-frame; per-env transforms are the caller's job (transform the rays
into the bake frame, see the static-scene three-way classification in
AGENTS.md).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cloud_robotics_sim.vdb.core.vdb_xuvdb import XuvdbVolume, _require_xuvdb

__all__ = ["SdfRayCaster"]

_TYPE_CODES = {0: "f32", 1: "f64", 2: "vec3f", 3: "f16"}


class SdfRayCaster:
    """Batch ray caster over a :class:`XuvdbVolume` level set grid.

    Args:
        volume: The SDF grid to cast against (bake frame = world frame).

    Raises:
        TypeError: If ``volume`` is not an :class:`XuvdbVolume`.
        ValueError: If the grid violates kernel constraints (non-f32, vector
            valued, or rotated).
    """

    def __init__(self, volume: XuvdbVolume) -> None:
        if not isinstance(volume, XuvdbVolume):
            raise TypeError(f"expected XuvdbVolume, got {type(volume)!r}")
        grid = volume.grid
        value_shape = getattr(grid, "value_shape", ())
        if value_shape != ():
            raise ValueError(
                f"SdfRayCaster supports scalar grids only, got value_shape {value_shape}"
            )
        type_code = int(getattr(grid, "type_code", 0))
        if type_code != 0:
            raise ValueError(
                f"SdfRayCaster supports float32 grids only, got {_TYPE_CODES.get(type_code, type_code)}; "
                "bake in f32 (default)"
            )
        if not bool(getattr(grid, "is_axis_aligned", True)):
            raise ValueError(
                "SdfRayCaster supports axis-aligned grids only (xuvdb kernel "
                "constraint); bake in the world frame instead of using rotation"
            )
        self._xuvdb = _require_xuvdb()
        self._gpu = self._xuvdb.GpuVolume(grid)

    @property
    def volume(self) -> XuvdbVolume:
        return XuvdbVolume(grid=self._gpu.grid)

    def cast(
        self,
        origins: Sequence[Sequence[float]] | np.ndarray,
        directions: Sequence[Sequence[float]] | np.ndarray,
        *,
        tmax: float | None = None,
        isovalue: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast ``n`` rays in one kernel launch.

        Args:
            origins: ``(n, 3)`` world-frame ray origins.
            directions: ``(n, 3)`` world-frame directions (need not be unit;
                zero directions are rejected by the kernel).
            tmax: Optional per-batch march bound (default derived from the
                grid bbox).
            isovalue: Level-set value to intersect (default ``0.0``).

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
        if len(o) == 0:
            empty_t = np.empty((0,), dtype=np.float64)
            empty_p = np.empty((0, 3), dtype=np.float64)
            return empty_t, empty_p, empty_t.copy()
        hits = self._gpu.ray_surface_hit(o, d, tmax=tmax, isovalue=float(isovalue))
        n = len(o)
        t = np.full(n, np.inf, dtype=np.float64)
        points = np.zeros((n, 3), dtype=np.float64)
        values = np.zeros(n, dtype=np.float64)
        for i, hit in enumerate(hits):
            if hit is None:
                continue
            ht, hp, hv = hit
            t[i] = float(ht)
            points[i] = np.asarray(hp, dtype=np.float64)
            values[i] = float(hv)
        return t, points, values
