"""xuvdb backend for the block-sparse volume (see :mod:`cloud_robotics_sim.vdb`).

This module is a thin adapter over the optional `xuvdb`_ package (太虚 —
editable sparse voxel volumes built on quadrants kernels). It is import-safe
without the package: ``xuvdb`` is only imported lazily inside functions, so
``import cloud_robotics_sim.vdb`` works in any environment; instantiating
:class:`XuvdbVolume` raises a descriptive :class:`ImportError` naming the
extra (same convention as ``plugins/solvers/cfd_coupling/core/obstacles.py``).

What the xuvdb backend adds over the torch prototype:

* **Editable SDF/CSG stamping** — ``stamp_sphere`` (min-union, 1-Lipschitz),
  ``fill_box``, ``csg()``, ``union_spheres`` particle level sets.
* **Native file interop** — ``save_xuvdb``/``load_xuvdb`` (``.xuvdb`` v3,
  zlib + CRC32) and ``write_openvdb``/``read_openvdb`` (real OpenVDB ``.vdb``
  streams readable by Houdini/Blender, no pyopenvdb needed).
* **Field queries** — trilinear/quadratic sampling, gradients, DDA
  ``ray_surface_hit`` (empty-leaf skipping + bisection refinement),
  particle splat (``scatter_particles``).

Conversion to/from :class:`TorchSparseVolume` goes through a dense numpy
window (``to_dense`` ↔ ``VdbGrid.from_dense``): not zero-copy, but immune to
the differing in-leaf linear orders (xuvdb leaves are z-fastest) and to the
block-vs-leaf topology mismatch. Conversions are offline/batch paths, so the
copy is acceptable (KISS).

.. _xuvdb: https://pypi.org/project/xuvdb/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cloud_robotics_sim.vdb.core.vdb_torch import TorchSparseVolume

__all__ = [
    "XuvdbVolume",
    "from_xuvdb",
    "load_xuvdb",
    "read_openvdb",
    "save_xuvdb",
    "to_xuvdb",
    "write_openvdb",
]

_XUVDB_EXTRA_HINT = "pip install cloud-robotics-sim[xuvdb]"


def _require_xuvdb() -> Any:
    """Import and return the ``xuvdb`` module, with an extra-naming error."""
    try:
        import xuvdb
    except ImportError as exc:  # pragma: no cover - exercised w/o extra
        raise ImportError(
            "The xuvdb volume backend requires the optional 'xuvdb' package "
            f"({_XUVDB_EXTRA_HINT}); original error: {exc}"
        ) from exc
    return xuvdb


def _vec3(v: float | Sequence[float]) -> tuple[float, float, float]:
    """Normalize a scalar or length-3 sequence to a float 3-tuple."""
    if isinstance(v, (int, float)):
        return (float(v), float(v), float(v))
    seq = tuple(float(x) for x in v)
    if len(seq) != 3:
        raise ValueError(f"expected scalar or length-3 vector, got {v!r}")
    return (seq[0], seq[1], seq[2])


class XuvdbVolume:
    """Sparse volume backed by :class:`xuvdb.VdbGrid`.

    Unlike :class:`TorchSparseVolume` the domain is *unbounded* (root hash
    table, no ``nx``/``ny``/``nz``); ``nx``/``ny``/``nz`` are accepted as an
    optional domain hint so ``create_volume(..., nx=..., ny=..., nz=...)``
    works uniformly across backends.

    Value ops (``fill``/``scale``/``add_const``/``reduce_*``) operate on the
    dense leaf buffers, matching the torch backend's block-level sparsity
    (no per-voxel active mask on either side for these ops).
    """

    def __init__(
        self,
        background: float = 0.0,
        voxel_size: float | Sequence[float] = 1.0,
        leaf_log2: int = 4,
        name: str = "grid",
        grid_class: str = "unknown",
        grid: Any | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
    ) -> None:
        self._xuvdb = _require_xuvdb()
        if grid is None:
            grid = self._xuvdb.VdbGrid(
                background=float(background),
                voxel_size=_vec3(voxel_size),
                leaf_log2=int(leaf_log2),
                name=name,
                grid_class=grid_class,
            )
        self._grid = grid
        # Domain hint only (torch-backend parity for create_volume kwargs).
        self.domain_hint = (nx, ny, nz)

    # ------------------------------------------------------------------ #
    # duck-type parity with TorchSparseVolume
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return "xuvdb"

    @property
    def grid(self) -> Any:
        """The underlying :class:`xuvdb.VdbGrid`."""
        return self._grid

    @property
    def background(self) -> float:
        return float(self._grid.background)

    @background.setter
    def background(self, value: float) -> None:
        self._grid.background = float(value)

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        return _vec3(self._grid.voxel_size)

    @property
    def name(self) -> str:
        return str(self._grid.name)

    @property
    def leaf_log2(self) -> int:
        return int(self._grid.leaf_log2)

    @property
    def n_active_blocks(self) -> int:
        """Number of allocated leaves (xuvdb has no separate block pool)."""
        return int(self._grid.leaf_count)

    @property
    def active_voxel_count(self) -> int:
        return int(self._grid.active_voxel_count)

    def count_active_voxels(self) -> int:
        return self.active_voxel_count

    @classmethod
    def from_dense(
        cls,
        arr: np.ndarray,
        *,
        voxel_size: float | Sequence[float] = 1.0,
        origin: Sequence[int] = (0, 0, 0),
        background: float = 0.0,
        grid_class: str = "unknown",
        name: str = "grid",
        leaf_log2: int = 4,
    ) -> "XuvdbVolume":
        """Build from a dense array; leaves with non-background values."""
        x = _require_xuvdb()
        grid = x.VdbGrid.from_dense(
            np.asarray(arr, dtype=np.float32),
            origin=tuple(int(v) for v in origin),
            voxel_size=_vec3(voxel_size),
            background=float(background),
            name=name,
            grid_class=grid_class,
            leaf_log2=int(leaf_log2),
        )
        return cls(grid=grid, nx=arr.shape[0], ny=arr.shape[1], nz=arr.shape[2])

    def to_dense(self, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Dense export; returns ``(values, ijk_min)`` (may be negative)."""
        dense, ijk_min = self._grid.to_dense(pad=int(pad))
        return np.asarray(dense, dtype=np.float32), np.asarray(ijk_min)

    def fill(self, value: float) -> None:
        """Fill all allocated leaf buffers (block-level, like the torch pool)."""
        for leaf in self._grid.leaves():
            leaf.values[...] = float(value)
            leaf.invalidate()

    def scale(self, a: float) -> None:
        """In-place ``self *= a`` on all leaf buffers."""
        for leaf in self._grid.leaves():
            leaf.values[...] = leaf.values * float(a)
            leaf.invalidate()

    def add_const(self, v: float) -> None:
        """In-place ``self += v`` on all leaf buffers."""
        for leaf in self._grid.leaves():
            leaf.values[...] = leaf.values + float(v)
            leaf.invalidate()

    def reduce_sum(self) -> float:
        if self._grid.leaf_count == 0:
            return 0.0
        return float(sum(leaf.values.sum() for leaf in self._grid.leaves()))

    def reduce_max(self) -> float:
        if self._grid.leaf_count == 0:
            return float("-inf")
        return float(max(leaf.values.max() for leaf in self._grid.leaves()))

    def memory_report(self) -> dict:
        leaf_dim = 2**self.leaf_log2
        bbox = self._grid.bbox()
        n_leaves = self._grid.leaf_count
        return {
            "backend": self.backend,
            "name": self.name,
            "leaf_log2": self.leaf_log2,
            "leaf_dim": leaf_dim,
            "n_leaves": n_leaves,
            "active_voxels": self.active_voxel_count,
            "leaf_bytes": n_leaves * leaf_dim**3 * 4,
            "bbox_ijk": (np.asarray(bbox[0]).tolist(), np.asarray(bbox[1]).tolist()),
            "voxel_size": self.voxel_size,
        }

    # ------------------------------------------------------------------ #
    # xuvdb-native sparse editing / queries (thin passthroughs)
    # ------------------------------------------------------------------ #
    def stamp_sphere(
        self,
        center: Sequence[float],
        radius: float,
        value: float | None = None,
        band: float = 3.0,
    ) -> "XuvdbVolume":
        """SDF-stamp a sphere (min-union against the existing field)."""
        self._grid.stamp_sphere(tuple(center), float(radius), value=value, band=band)
        return self

    def fill_box(
        self,
        ijk_min: Sequence[int],
        ijk_max: Sequence[int],
        value: float,
        active: bool = True,
    ) -> "XuvdbVolume":
        self._grid.fill_box(tuple(ijk_min), tuple(ijk_max), float(value), active=active)
        return self

    def prune(self) -> "XuvdbVolume":
        """Drop background-valued leaves (constant-tile collapse)."""
        self._grid.prune()
        return self

    def csg(self, other: "XuvdbVolume", op: str) -> "XuvdbVolume":
        """CSG combine with another volume: ``union`` / ``intersect`` / ``subtract``."""
        if not isinstance(other, XuvdbVolume):
            raise TypeError(f"expected XuvdbVolume, got {type(other)!r}")
        self._grid.csg(other.grid, op)
        return self

    def union_spheres(
        self,
        centers: np.ndarray,
        radius: float | np.ndarray,
        band: float = 3.0,
    ) -> "XuvdbVolume":
        """Particle level set: union of per-particle spheres."""
        self._grid.union_spheres(
            np.asarray(centers, dtype=np.float32), radius, band=band
        )
        return self

    def scatter_particles(
        self,
        points: np.ndarray,
        h: float | None = None,
        weights: float | np.ndarray = 1.0,
        kernel: str = "cubic",
    ) -> "XuvdbVolume":
        """SPH-style kernel-density splat of particles into a fog volume."""
        self._grid.scatter_particles(
            np.asarray(points, dtype=np.float32), h=h, weights=weights, kernel=kernel
        )
        return self

    # ----- sampling ---------------------------------------------------- #
    def sample_linear(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._grid.sample_linear(np.asarray(points, dtype=np.float32))
        )

    def sample_nearest(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._grid.sample_nearest(np.asarray(points, dtype=np.float32))
        )

    def sample_quadratic(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._grid.sample_quadratic(np.asarray(points, dtype=np.float32))
        )

    def sample_gradient(self, points: np.ndarray, order: int = 1) -> np.ndarray:
        return np.asarray(
            self._grid.sample_gradient(
                np.asarray(points, dtype=np.float32), order=order
            )
        )

    def ray_surface_hit(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        tmax: float | None = None,
        isovalue: float = 0.0,
    ) -> Any:
        """DDA ray vs the zero level set; ``(t, point, value)`` or ``None``."""
        return self._xuvdb.ray_surface_hit(
            self._grid,
            tuple(origin),
            tuple(direction),
            tmax=tmax,
            isovalue=isovalue,
        )

    def set_value(self, ijk: Sequence[int], value: float, active: bool = True) -> None:
        self._grid.set_value(tuple(int(v) for v in ijk), float(value), active=active)

    def get_value(self, ijk: Sequence[int]) -> float:
        return float(self._grid.get_value(tuple(int(v) for v in ijk)))

    def copy(self) -> "XuvdbVolume":
        return XuvdbVolume(grid=self._grid.copy())


# ---------------------------------------------------------------------- #
# torch <-> xuvdb conversion (dense round-trip; see module docstring)
# ---------------------------------------------------------------------- #
def to_xuvdb(
    vol: "TorchSparseVolume",
    *,
    voxel_size: float | Sequence[float] = 1.0,
    name: str = "grid",
    grid_class: str = "unknown",
    leaf_log2: int = 4,
) -> XuvdbVolume:
    """Convert a :class:`TorchSparseVolume` to :class:`XuvdbVolume`."""
    dense = vol.to_dense()
    return XuvdbVolume.from_dense(
        dense,
        voxel_size=voxel_size,
        background=vol.background,
        grid_class=grid_class,
        name=name,
        leaf_log2=leaf_log2,
    )


def from_xuvdb(xvol: XuvdbVolume, *, block_size: int = 8) -> "TorchSparseVolume":
    """Convert an :class:`XuvdbVolume` to :class:`TorchSparseVolume`.

    The dense export window becomes the torch volume's logical domain
    (starting at index ``(0, 0, 0)``); the source ``ijk_min`` offset is
    dropped — retrieve it from ``xvol.to_dense()`` if the index frame
    matters.
    """
    from cloud_robotics_sim.vdb.core.vdb_torch import TorchSparseVolume

    if not isinstance(xvol, XuvdbVolume):
        raise TypeError(f"expected XuvdbVolume, got {type(xvol)!r}")
    dense, _ijk_min = xvol.to_dense()

    # TorchSparseVolume assumes the logical domain is a multiple of the
    # block size; round the dense window up (background pads the rim).
    def _round_up(v: int) -> int:
        return ((int(v) + block_size - 1) // block_size) * block_size

    vol = TorchSparseVolume(
        nx=_round_up(dense.shape[0]),
        ny=_round_up(dense.shape[1]),
        nz=_round_up(dense.shape[2]),
        block_size=block_size,
        background=xvol.background,
    )
    vol.import_window((0, 0, 0), dense)
    return vol


# ---------------------------------------------------------------------- #
# file interop
# ---------------------------------------------------------------------- #
def save_xuvdb(
    path: str | Path, volumes: Sequence[XuvdbVolume], compress: bool = True
) -> Path:
    """Write volumes to an ``.xuvdb`` file (zlib + CRC32 when ``compress``)."""
    x = _require_xuvdb()
    if not volumes:
        raise ValueError("no volumes to save")
    for v in volumes:
        if not isinstance(v, XuvdbVolume):
            raise TypeError(f"expected XuvdbVolume, got {type(v)!r}")
    path = Path(path)
    x.save(str(path), [v.grid for v in volumes], compress=compress)
    return path


def load_xuvdb(path: str | Path) -> list[XuvdbVolume]:
    """Read all grids from an ``.xuvdb`` file."""
    x = _require_xuvdb()
    grids = x.load(str(Path(path)))
    return [XuvdbVolume(grid=g) for g in grids]


def write_openvdb(
    path: str | Path, volumes: Sequence[XuvdbVolume], blosc: bool = False
) -> Path:
    """Export volumes as a real OpenVDB ``.vdb`` stream (Houdini/Blender)."""
    x = _require_xuvdb()
    if not volumes:
        raise ValueError("no volumes to export")
    for v in volumes:
        if not isinstance(v, XuvdbVolume):
            raise TypeError(f"expected XuvdbVolume, got {type(v)!r}")
    path = Path(path)
    x.write_vdb(str(path), [v.grid for v in volumes], blosc=blosc)
    return path


def read_openvdb(path: str | Path, grid_name: str | None = None) -> XuvdbVolume:
    """Read one grid from an OpenVDB ``.vdb`` file (native reader).

    ``xuvdb.read_vdb`` returns a list of grids; ``grid_name`` selects which
    one to wrap (first grid when omitted).
    """
    x = _require_xuvdb()
    grids = x.read_vdb(str(Path(path)), grid_name=grid_name)
    if not isinstance(grids, (list, tuple)):
        grids = [grids]
    if not grids:
        raise ValueError(f"no grids found in {path}")
    return XuvdbVolume(grid=grids[0])
