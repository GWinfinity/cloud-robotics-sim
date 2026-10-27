"""Bake xuvdb level sets into Genesis's dense ``SDFGrid`` so edited fields drive contacts.

The rigid contact solver is built around Genesis's ``SDFGrid`` representation: a
dense signed-distance lattice in the mesh frame, with a per-node gradient and
closest-vertex index. This module converts a ``xuvdb.VdbGrid`` into exactly that
representation, so an xuvdb-authored or xuvdb-edited field (a cavity stamped
with CSG, an eroded face) drives the stock contact pipeline without any change
to Genesis's kernels.

Two fill modes govern the field outside the xuvdb narrow band:

* ``fill="mesh"`` (default) computes the exact mesh signed distance with
  trimesh outside the band and overlays xuvdb where its mask is active. The
  exact distance deep inside and far outside the object is preserved, so the
  bake deviates from the stock SDF only near the surface edits.
* ``fill="background"`` samples xuvdb everywhere (its background becomes the
  outside field and the interior fill, if any, the inside field). Requires the
  grid to carry an interior fill (``xuvdb.mesh_to_level_set`` fills it by
  default).

The bake changes the field, not the geometry. Genesis culls candidate pairs by
the collision mesh's axis-aligned bounding box before it ever queries the SDF,
so a level set whose surface leaves that box (an inflated object) is never
sampled and other objects pass through it. Contact-relevant edits must stay
inside the mesh's own extent — carve a cavity, erode a face — rather than grow
the shape. The lattice is also fixed by the descriptor: nodes are the mesh
bounding box expanded by the material's margin, so detail finer than one cell
is lost.

The baked grid is cached on disk under the project cache root
(``CRS_SIM_CACHE_DIR`` / ``outputs/sim_cache`` -> ``vdb/genesis``), keyed on the
mesh, the descriptor, the fill mode and the content of the xuvdb grid (see
:func:`grid_fingerprint`), so rebuilding a scene — or running another one with
the same asset — reads the grid instead of baking it again.

Integration is a monkey-patch of ``genesis.engine.mesh.Mesh.get_sdf_grid`` (see
:func:`install_bake_bridge` / :func:`register_mesh_sdf`): meshes carry a pending
"xuvdb SDF spec" and the patched method bakes it at build time. The patch is
inert for meshes without a spec and checks the host Genesis actually exposes
the method before patching.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from pathlib import Path
from typing import Any, Sequence

import genesis as gs
import numpy as np

from cloud_robotics_sim.utils.cache_cleanup import get_cache_dir

logger = logging.getLogger(__name__)


def _np_float() -> Any:
    """Genesis's host float dtype (``gs.np_float``), falling back to float32 pre-init."""
    return getattr(gs, "np_float", np.float32)


def _np_int() -> Any:
    """Genesis's host int dtype (``gs.np_int``), falling back to int32 pre-init."""
    return getattr(gs, "np_int", np.int32)


__all__ = [
    "GENESIS_CONTACT_CACHE_VERSION",
    "bake_sdf_grid",
    "cached_closest_verts",
    "clear_mesh_sdf",
    "grid_fingerprint",
    "install_bake_bridge",
    "register_mesh_sdf",
]

# Bumped when the payload or the key of any cache entry changes, forcing a
# cache miss on stale entries.
GENESIS_CONTACT_CACHE_VERSION = 1


def _default_contact_cache_dir() -> Path:
    """On-disk cache root for xuvdb-derived Genesis SDF data (``<sim_cache>/vdb/genesis``)."""
    return get_cache_dir() / "vdb" / "genesis"


def _require_xuvdb() -> Any:
    """Import and return the ``xuvdb`` module, with an extra-naming error."""
    try:
        import xuvdb
    except ImportError as exc:  # pragma: no cover - exercised w/o extra
        raise ImportError(
            "Genesis SDF baking requires the optional 'xuvdb' package "
            "(pip install cloud-robotics-sim[xuvdb]); "
            f"original error: {exc}"
        ) from exc
    return xuvdb


# --------------------------------------------------------------------------- #
# Cache helpers (genesis MeshCache buckets under the project cache root)
# --------------------------------------------------------------------------- #


def _mesh_cache(
    ext: str,
    verts: np.ndarray,
    faces: np.ndarray,
    params: Sequence[Any],
    cache_dir: str | Path | None,
) -> Any:
    """A genesis ``MeshCache`` bucket for xuvdb-derived data of one mesh."""
    from genesis.utils.cache import MeshCache

    xuvdb = _require_xuvdb()

    root = Path(cache_dir) if cache_dir is not None else _default_contact_cache_dir()
    return MeshCache(
        str(root / ext),
        ext,
        np.ascontiguousarray(verts),
        np.ascontiguousarray(faces),
        *params,
        getattr(xuvdb, "__version__", "unknown"),
        GENESIS_CONTACT_CACHE_VERSION,
        np.dtype(np.float32).str,
    )


def grid_fingerprint(grid: Any) -> str:
    """Content hash of a level set, identifying its leaves and their values.

    Leaves are hashed in their canonical order, together with the transform and
    background the field is sampled through, so two grids holding the same
    field share a fingerprint whichever way they were built, and an edit
    changes it.
    """
    from genesis.utils.cache import get_hashkey

    parts: list[np.ndarray] = [
        np.asarray(grid.voxel_size, np.float64),
        np.asarray(grid.origin_world, np.float64),
        np.asarray(grid.background, np.float64).reshape(-1),
        np.array(int(grid.leaf_log2), np.int64),
        np.array(int(grid.type_code), np.int64),
    ]
    for leaf in grid.leaves():
        parts.append(np.ascontiguousarray(leaf.origin, dtype=np.int64))
        parts.append(np.ascontiguousarray(leaf.values))
        parts.append(np.packbits(np.ascontiguousarray(leaf.active)))
    return str(get_hashkey(parts))


def cached_closest_verts(
    grid: Any,
    verts: np.ndarray,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Closest local vertex index for every allocated voxel of a level set.

    The table is leaf-major and z-fastest, one entry per voxel of the packed
    volume. Its KD-tree query is cached on disk (keyed on the grid content and
    the mesh), so attaching the same volume to another scene costs a file read.
    """
    import scipy.spatial

    verts = np.ascontiguousarray(verts, dtype=np.float64)
    cache = _mesh_cache(
        "cv", verts, np.zeros((0, 3), np.int32), (grid_fingerprint(grid),), cache_dir
    )
    closest = cache.load()
    if closest is not None:
        return np.asarray(closest, dtype=np.int32)

    dim = int(grid.leaf_dim)
    local = np.stack(
        np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    tree = scipy.spatial.cKDTree(verts)
    chunks: list[np.ndarray] = []
    for leaf in grid.leaves():
        index = leaf.origin[None, :] + local
        world = grid.index_to_world(index)
        _, closest = tree.query(world, k=1)
        chunks.append(np.asarray(closest, dtype=np.int32))
    if not chunks:
        out = np.full(0, -1, np.int32)
    else:
        out = np.concatenate(chunks)
    cache.save(out)
    return out


# --------------------------------------------------------------------------- #
# Bake: xuvdb VdbGrid -> genesis SDFGrid
# --------------------------------------------------------------------------- #


def _frame_from_description(
    sdf: Any, verts: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
    """Replicate Genesis's mesh->SDF-node transform and node positions for a description."""
    _n = np.asarray(sdf.n_nodes, dtype=np.int64).reshape(-1)
    n_nodes: tuple[int, int, int] = (int(_n[0]), int(_n[1]), int(_n[2]))
    cell_size = np.asarray(sdf.cell_size, dtype=np.float64)
    lower = np.asarray(verts, dtype=np.float64).min(axis=0)
    upper = np.asarray(verts, dtype=np.float64).max(axis=0)
    center = (upper + lower) / 2.0
    grid_size = (np.asarray(n_nodes) - 1) * cell_size
    voxel_lower = center - grid_size / 2.0
    voxel_upper = center + grid_size / 2.0
    axes = [np.linspace(voxel_lower[i], voxel_upper[i], n_nodes[i]) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape((-1, 3))

    t_mesh_to_centered = np.eye(4)
    t_mesh_to_centered[:3, 3] = -center
    t_centered_to_sdf = np.eye(4)
    t_centered_to_sdf[:3, :3] *= (np.asarray(n_nodes) - 1) / grid_size
    t_centered_to_sdf[:3, 3] = (np.asarray(n_nodes) - 1) / 2
    t_mesh_to_sdf = t_centered_to_sdf @ t_mesh_to_centered
    return points, n_nodes, t_mesh_to_sdf


def _mesh_distance(
    points: np.ndarray, verts: np.ndarray, faces: np.ndarray, has_grad: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance, analytic gradient and closest vertex index at points.

    Sign convention matches Genesis's SDF (positive outside). The signed
    distance and closest point come from trimesh's AABB-tree proximity queries;
    the gradient is the unit direction to the closest surface point, with the
    face normal as the on-surface fallback.
    """
    import trimesh

    tmesh = trimesh.Trimesh(
        vertices=np.asarray(verts), faces=np.asarray(faces), process=False
    )
    # trimesh reports positive-inside; Genesis uses positive-outside.
    sd = -np.asarray(trimesh.proximity.signed_distance(tmesh, points), dtype=np.float64)
    closest_points, dist, closest_faces = trimesh.proximity.closest_point(tmesh, points)
    closest_faces = np.asarray(closest_faces, dtype=np.int64)

    if has_grad:
        face_normals = np.cross(
            verts[faces[:, 1]] - verts[faces[:, 0]],
            verts[faces[:, 2]] - verts[faces[:, 0]],
        )
        norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
        face_normals = face_normals / np.maximum(norms, 1e-12)
        offset = points - closest_points
        dist_col = np.linalg.norm(offset, axis=-1, keepdims=True)
        grad = np.where(
            dist_col > 1e-12,
            np.sign(sd)[:, None] * offset / np.maximum(dist_col, 1e-12),
            face_normals[closest_faces],
        )
    else:
        grad = np.zeros_like(points)

    vert_ids = faces[closest_faces]
    vert_dist = np.linalg.norm(verts[vert_ids] - points[:, None, :], axis=-1)
    closest_vert = vert_ids[np.arange(len(points)), np.argmin(vert_dist, axis=-1)]
    return sd, grad, closest_vert.astype(np.int64, copy=False)


def _xuvdb_sample(
    grid: Any, points: np.ndarray, need_grad: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Sample an xuvdb grid, preferring the packed device kernel and falling back to host.

    ``valid`` marks samples whose gradient is meaningful (xuvdb 1.3.x reports a
    zero normal outside the narrow band; there is no ``return_valid`` kwarg).
    """
    xuvdb = _require_xuvdb()
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    try:
        volume = xuvdb.GpuVolume(grid)
        values = np.asarray(
            volume.sample(points, linear=True), dtype=np.float32
        ).reshape(-1)
        if need_grad:
            grad = np.asarray(volume.sdf_normal(points), dtype=np.float32).reshape(
                -1, 3
            )
            valid = (np.linalg.norm(grad, axis=1) > 0.0).astype(np.int32)
        else:
            grad, valid = None, None
    except Exception:
        # Host fallback: xuvdb 1.3.x host samplers take/return scalars per
        # point, so loop the batch.
        values = np.array(
            [grid.sample_linear(p) for p in points], dtype=np.float32
        ).reshape(-1)
        if need_grad:
            grad = np.array(
                [grid.sample_gradient(p) for p in points], dtype=np.float32
            ).reshape(-1, 3)
            valid = np.ones(len(points), dtype=np.int32)
        else:
            grad, valid = None, None
    return values, grad, valid


def _active_at_nodes(grid: Any, points: np.ndarray) -> np.ndarray:
    """Xuvdb active-mask lookup at the lattice node nearest to each point."""
    index = np.rint(grid.world_to_index(np.asarray(points, dtype=np.float64))).astype(
        np.int64
    )
    _, active = grid.probe_batch(index)
    return np.asarray(active)


def bake_sdf_grid(
    grid: Any,
    sdf: Any,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
    fill: str = "mesh",
    closest_verts: np.ndarray | None = None,
    chunk_size: int = 1_000_000,
    cache: bool = True,
    cache_dir: str | Path | None = None,
) -> Any:
    """Convert an xuvdb ``VdbGrid`` into a Genesis ``SDFGrid`` matching ``sdf``.

    Args:
        grid: ``xuvdb.VdbGrid`` authored in the mesh frame (``origin_world``
            is interpreted as the mesh-frame origin and the rotation must be
            identity). Requires the ``xuvdb`` extra.
        sdf: ``SignedDistanceFieldDescription`` of the Genesis geometry. Its
            node count and anisotropic cell size define the output lattice.
        verts:
            Collision-mesh vertices of the geometry, in its local frame.
        faces:
            Collision-mesh faces of the geometry, in its local frame.
        fill: ``"mesh"`` computes the exact signed distance/gradient outside
            the xuvdb band and overlays xuvdb where its mask is active.
            ``"background"`` uses xuvdb everywhere (requires interior fill).
        closest_verts: Optional local vertex-index array for every voxel of the
            packed grid (leaf-major, z-fastest). When omitted and
            ``fill="mesh"``, a KD-tree of ``verts`` supplies it.
        chunk_size: Points sampled per chunk, bounding temporaries on large
            lattices (``fill="background"`` path).
        cache: Whether to load/save the baked grid from the on-disk cache.
            The entry is keyed on the mesh, the descriptor, the fill mode and
            the content of ``grid``.
        cache_dir: Cache root (default ``<sim_cache>/vdb/genesis``).

    Returns:
        A genesis ``engine.mesh.SDFGrid`` named tuple.
    """
    _require_xuvdb()
    if grid.is_vec or grid.type_code != 0:
        raise TypeError("SDF baking supports float32 scalar xuvdb grids only")
    if not grid.is_axis_aligned:
        raise ValueError(
            "SDF baking requires an axis-aligned xuvdb grid (identity rotation)"
        )
    if fill not in ("mesh", "background"):
        raise ValueError("fill must be 'mesh' or 'background'")

    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    has_grad = bool(sdf.has_grad)

    disk_cache = None
    if cache:
        disk_cache = _mesh_cache(
            "bsdf",
            verts,
            faces,
            (
                tuple(int(v) for v in np.asarray(sdf.n_nodes)),
                tuple(float(v) for v in np.asarray(sdf.cell_size)),
                has_grad,
                fill,
                grid_fingerprint(grid),
                (
                    None
                    if closest_verts is None
                    else np.ascontiguousarray(closest_verts, dtype=np.int32)
                ),
            ),
            cache_dir,
        )
        baked = disk_cache.load()
        if baked is not None:
            return baked

    points, n_nodes, t_mesh_to_sdf = _frame_from_description(sdf, verts)

    values = np.empty(len(points), dtype=_np_float())
    grads = np.zeros((len(points), 3), dtype=_np_float())
    closest: np.ndarray | None = None

    if fill == "mesh":
        sd, exact_grad, exact_closest = _mesh_distance(points, verts, faces, has_grad)
        values[:] = sd
        if has_grad:
            grads[:] = exact_grad
        closest = exact_closest
        active_ids = np.nonzero(_active_at_nodes(grid, points))[0]
        if len(active_ids):
            xv_val, xv_grad, xv_valid = _xuvdb_sample(
                grid, points[active_ids], has_grad
            )
            values[active_ids] = xv_val
            if has_grad:
                assert xv_grad is not None and xv_valid is not None
                good = np.asarray(xv_valid).reshape(-1) > 0
                grads[active_ids[good]] = np.asarray(xv_grad)[good]
    else:
        # Sample in chunks to keep the temporaries bounded on large lattices.
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            xv_val, xv_grad, _ = _xuvdb_sample(grid, points[start:stop], has_grad)
            values[start:stop] = xv_val
            if has_grad:
                grads[start:stop] = xv_grad

    if closest is None:
        if closest_verts is None:
            # No geometry to associate with a vertex: index 0 is the only
            # valid in-range choice, and the SDF normal path does not consume
            # closest vertices.
            closest = np.zeros(len(points), dtype=np.int64)
        else:
            closest = np.asarray(closest_verts, dtype=np.int64).reshape(-1)

            if len(closest) != len(points):
                raise ValueError(
                    f"closest_verts must have {len(points)} entries, got {len(closest)}"
                )

    if not has_grad:
        grads[:] = 0.0

    from genesis.engine.mesh import SDFGrid

    sdf_grid = SDFGrid(
        value=values.reshape(n_nodes).astype(_np_float(), copy=False),
        grad=grads.reshape((*n_nodes, 3)).astype(_np_float(), copy=False),
        closest_verts_idx=closest.reshape(n_nodes).astype(_np_int(), copy=False),
        max_value=float(np.max(values)),
        T_mesh_to_sdf=np.asarray(t_mesh_to_sdf, dtype=_np_float()),
    )
    if disk_cache is not None:
        disk_cache.save(sdf_grid)
    return sdf_grid


# --------------------------------------------------------------------------- #
# Build-time bridge: pending xuvdb specs honored by Mesh.get_sdf_grid
# --------------------------------------------------------------------------- #

_BRIDGE_INSTALLED = False


def clear_mesh_sdf(mesh: Any) -> None:
    """Remove a pending xuvdb SDF override and drop Genesis's cached SDF for ``mesh``."""
    if hasattr(mesh, "_crs_xuvdb_sdf_spec"):
        delattr(mesh, "_crs_xuvdb_sdf_spec")
    try:
        from genesis.engine.mesh import _TMESH_CACHE

        _TMESH_CACHE.release(mesh)
    except Exception:  # pragma: no cover - defensive against genesis internals
        logger.debug("could not release genesis tmesh cache", exc_info=True)


def register_mesh_sdf(
    mesh: Any,
    grid: Any,
    *,
    fill: str = "mesh",
    fill_mesh: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Make ``mesh`` use ``grid`` for its Genesis SDF at the next build.

    ``mesh`` is a ``genesis.engine.mesh.Mesh`` (``geom.desc.mesh``). ``fill_mesh``
    is an optional ``(verts, faces)`` pair used for the exact out-of-band field
    and closest vertices; it defaults to the mesh's own vertices and faces.

    The grid must be authored in the mesh's local frame, and contact-relevant
    edits must stay inside the mesh's axis-aligned bounding box (Genesis culls
    SDF pairs by that box before sampling the field).

    Returns the stored spec (also attached as ``mesh._crs_xuvdb_sdf_spec``).
    """
    install_bake_bridge()
    spec = {
        "grid": grid,
        "fill": fill,
        "fill_mesh": fill_mesh,
    }
    mesh._crs_xuvdb_sdf_spec = spec
    # Drop Genesis's cached SDF/tmesh for this mesh so the next build re-bakes
    # through the bridge; the spec itself must survive until then.
    try:
        from genesis.engine.mesh import _TMESH_CACHE

        _TMESH_CACHE.release(mesh)
    except Exception:  # pragma: no cover - defensive against genesis internals
        logger.debug("could not release genesis tmesh cache", exc_info=True)
    return spec


def install_bake_bridge() -> bool:
    """Patch ``genesis.engine.mesh.Mesh.get_sdf_grid`` to honor pending xuvdb specs.

    The patch dispatches to :func:`bake_sdf_grid` only for meshes carrying a
    spec registered by :func:`register_mesh_sdf`; every other mesh takes the
    stock path unchanged. Returns whether the patch was (or already had been)
    installed.
    """
    global _BRIDGE_INSTALLED
    if _BRIDGE_INSTALLED:
        return True
    try:
        from genesis.engine.mesh import Mesh
    except ImportError:  # pragma: no cover - genesis is a core dependency
        logger.warning("genesis-world not importable; bake bridge not installed")
        return False
    if not hasattr(Mesh, "get_sdf_grid"):
        logger.warning(
            "genesis Mesh does not expose get_sdf_grid; bake bridge not installed"
        )
        return False

    original = Mesh.get_sdf_grid

    @functools.wraps(original)
    def get_sdf_grid(self: Any, sdf: Any) -> Any:
        spec = getattr(self, "_crs_xuvdb_sdf_spec", None)
        if spec is None:
            return original(self, sdf)
        if spec["fill_mesh"] is None:
            verts, faces = self.verts, self.faces
        else:
            verts, faces = spec["fill_mesh"]
        return bake_sdf_grid(
            spec["grid"],
            sdf,
            verts=verts,
            faces=faces,
            fill=spec["fill"],
        )

    Mesh.get_sdf_grid = get_sdf_grid
    _BRIDGE_INSTALLED = True
    return True


@dataclasses.dataclass
class BakeBridgeStatus:
    """Diagnostics for the bake bridge (introspection/tests)."""

    installed: bool


def bridge_status() -> BakeBridgeStatus:
    """Return whether the bake bridge monkey-patch is installed."""
    return BakeBridgeStatus(installed=_BRIDGE_INSTALLED)
