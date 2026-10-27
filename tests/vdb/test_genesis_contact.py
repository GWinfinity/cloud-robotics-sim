"""xuvdb -> Genesis SDFGrid 接触桥测试（xuvdb 缺失时优雅 skip）。"""

from __future__ import annotations

import numpy as np

from tests.optional_deps import xuvdb_only

pytestmark = xuvdb_only

import trimesh  # noqa: E402

from cloud_robotics_sim.utils.genesis_compat import (  # noqa: E402
    ensure_genesis_initialized,
)
from cloud_robotics_sim.vdb.core import genesis_contact as gc  # noqa: E402


def _box_mesh(translated: bool = True) -> tuple[np.ndarray, np.ndarray]:
    box = trimesh.creation.box((0.1, 0.08, 0.06))
    if translated:
        box.apply_translation((0.03, -0.02, 0.05))
    return (
        np.asarray(box.vertices, np.float64),
        np.asarray(box.faces, np.int32),
    )


def _level_set(verts: np.ndarray, faces: np.ndarray, voxel_size: float = 0.005):
    import xuvdb

    return xuvdb.mesh_to_level_set(verts, faces, voxel_size=voxel_size, band=3.0)


def _description():
    from genesis.engine.entities.rigid_entity.description import (
        SignedDistanceFieldDescription,
    )

    return SignedDistanceFieldDescription(
        n_nodes=(41, 41, 41),
        cell_size=(0.005, 0.005, 0.005),
        has_grad=True,
    )


def _node_points(verts, sdf):
    points, _, _ = gc._frame_from_description(sdf, verts)
    return points


def _active_mask(grid, points) -> np.ndarray:
    index = np.rint(grid.world_to_index(points)).astype(np.int64)
    _, active = grid.probe_batch(index)
    return np.asarray(active, bool)


def test_grid_fingerprint_stable_and_edit_sensitive() -> None:
    """相同内容指纹一致；CSG 编辑改变指纹。"""
    verts, faces = _box_mesh()
    grid_a = _level_set(verts, faces)
    grid_b = _level_set(verts, faces)
    assert gc.grid_fingerprint(grid_a) == gc.grid_fingerprint(grid_b)

    import xuvdb

    hole = xuvdb.VdbGrid(
        dtype=np.float32,
        background=1.0,
        voxel_size=grid_a.voxel_size,
        origin_world=grid_a.origin_world,
        leaf_log2=grid_a.leaf_log2,
    )
    hole.stamp_sphere((0.03, -0.02, 0.08), 0.02, band=3.0)
    grid_a.csg(hole, "diff")
    assert gc.grid_fingerprint(grid_a) != gc.grid_fingerprint(grid_b)


def test_bake_matches_stock_out_of_band() -> None:
    """烘焙网格在 xuvdb 带外与 stock SDF 一致（精确网格距离），带内跟随 xuvdb。"""
    from genesis.engine.mesh import _get_sdf_grid

    # genesis's own SDF builder needs gs.logger, i.e. an initialized genesis.
    ensure_genesis_initialized(use_cuda=False, seed=0)
    verts, faces = _box_mesh()
    sdf = _description()
    grid = _level_set(verts, faces)
    baked = gc.bake_sdf_grid(grid, sdf, verts=verts, faces=faces, cache=False)
    stock = _get_sdf_grid(
        verts, faces, trimesh.Trimesh(verts, faces, process=False), sdf, "(test)"
    )

    assert np.allclose(stock.T_mesh_to_sdf, baked.T_mesh_to_sdf, atol=1e-6)

    points = _node_points(verts, sdf)
    active = _active_mask(grid, points)
    assert active.sum() > 100, "box level set should have an active band"

    baked_v = np.asarray(baked.value).reshape(-1)
    stock_v = np.asarray(stock.value).reshape(-1)
    cell = float(np.max(np.asarray(sdf.cell_size)))
    in_dev = np.abs(baked_v - stock_v)[active]
    out_diff = np.abs(baked_v - stock_v)[~active]
    assert (
        in_dev.max() < 0.5 * cell
    ), f"in-band deviation {in_dev.max():.4f} exceeds half a cell"
    assert (
        out_diff.max() < 1e-6
    ), f"out-of-band field must be the exact mesh distance, got {out_diff.max():.2e}"


def test_bake_overlay_deviates_from_stock_where_edited() -> None:
    """挖腔后的带内场必须偏离 stock（证明 xuvdb 内容真正接管了接触场）。"""
    ensure_genesis_initialized(use_cuda=False, seed=0)
    verts, faces = _box_mesh()
    sdf = _description()
    plain = _level_set(verts, faces)

    import xuvdb

    dented = xuvdb.mesh_to_level_set(verts, faces, voxel_size=0.005, band=3.0)
    hole = xuvdb.VdbGrid(
        dtype=np.float32,
        background=1.0,
        voxel_size=dented.voxel_size,
        origin_world=dented.origin_world,
        leaf_log2=dented.leaf_log2,
    )
    hole.stamp_sphere((0.03, -0.02, 0.08), 0.02, band=3.0)
    dented.csg(hole, "diff")

    baked_plain = gc.bake_sdf_grid(plain, sdf, verts=verts, faces=faces, cache=False)
    baked_dent = gc.bake_sdf_grid(dented, sdf, verts=verts, faces=faces, cache=False)

    points = _node_points(verts, sdf)
    active = _active_mask(dented, points)
    diff = np.abs(
        np.asarray(baked_dent.value).reshape(-1)
        - np.asarray(baked_plain.value).reshape(-1)
    )
    assert diff[active].max() > 0.005, "dent must change the field inside the band"


def test_bake_cache_reuse_and_content_invalidation(tmp_path) -> None:
    """相同输入命中缓存；网格内容或参数变化则未命中。"""
    from genesis.utils.cache import MeshCache

    verts, faces = _box_mesh()
    sdf = _description()
    grid = _level_set(verts, faces)
    kw = dict(verts=verts, faces=faces, cache_dir=tmp_path)

    first = gc.bake_sdf_grid(grid, sdf, **kw)
    second = gc.bake_sdf_grid(grid, sdf, **kw)
    assert np.array_equal(np.asarray(first.value), np.asarray(second.value))

    def n_entries() -> int:
        return sum(1 for p in tmp_path.rglob("*.bsdf"))

    assert n_entries() == 1, "repeated bake of the same grid must reuse one entry"

    # 编辑网格内容 -> 新条目
    import xuvdb

    hole = xuvdb.VdbGrid(
        dtype=np.float32,
        background=1.0,
        voxel_size=grid.voxel_size,
        origin_world=grid.origin_world,
        leaf_log2=grid.leaf_log2,
    )
    hole.stamp_sphere((0.03, -0.02, 0.08), 0.02, band=3.0)
    grid.csg(hole, "diff")
    gc.bake_sdf_grid(grid, sdf, **kw)
    assert n_entries() == 2, "an edited grid must be a different bake entry"

    # 不同描述 -> 新条目
    from genesis.engine.entities.rigid_entity.description import (
        SignedDistanceFieldDescription,
    )

    other = SignedDistanceFieldDescription(
        n_nodes=(21, 21, 21),
        cell_size=(0.01, 0.01, 0.01),
        has_grad=True,
    )
    gc.bake_sdf_grid(_level_set(verts, faces), other, **kw)
    assert n_entries() == 3, "a different descriptor must be a different entry"

    assert MeshCache is not None  # 缓存经由 genesis MeshCache


def test_cached_closest_verts_table_and_cache(tmp_path) -> None:
    """Closest-vertex 表尺寸正确、索引合法，且二次查询命中同一缓存条目。"""
    verts, faces = _box_mesh()
    grid = _level_set(verts, faces)
    closest = gc.cached_closest_verts(grid, verts, cache_dir=tmp_path)
    again = gc.cached_closest_verts(grid, verts, cache_dir=tmp_path)
    assert np.array_equal(closest, again)
    assert len(closest) == grid.n_leaves * grid.leaf_dim**3
    entries = sum(1 for p in tmp_path.rglob("*.cv"))
    assert entries == 1, "repeated lookup must reuse one cache entry"
    # 表内索引均为合法局部顶点索引
    assert closest.min() >= 0
    assert closest.max() < len(verts)


def _l_prism() -> trimesh.Trimesh:
    poly = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.2, 0.1], [0.1, 0.1], [0.1, 0.2], [0.0, 0.2]]
    )
    n = len(poly)
    verts = np.concatenate([np.c_[poly, np.full(n, 0.0)], np.c_[poly, np.full(n, 0.1)]])
    fan = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]
    tris = [(a, c, b) for a, b, c in fan]
    tris += [(a + n, b + n, c + n) for a, b, c in fan]
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, j + n), (i, j + n, i + n)]
    return trimesh.Trimesh(vertices=verts, faces=np.array(tris), process=False)


def _dented_prism_grid(voxel_size: float = 0.01, band: float = 3.0):
    """Prism level set minus a 3 cm sphere at the box's landing spot (inside AABB)."""
    import xuvdb

    prism = _l_prism()
    verts = np.asarray(prism.vertices, np.float64)
    faces = np.asarray(prism.faces, np.int32)
    grid = xuvdb.mesh_to_level_set(verts, faces, voxel_size=voxel_size, band=band)
    hole = xuvdb.VdbGrid(
        dtype=np.float32,
        background=1.0,
        voxel_size=grid.voxel_size,
        origin_world=grid.origin_world,
        leaf_log2=grid.leaf_log2,
    )
    hole.stamp_sphere((0.05, 0.15, 0.1), 0.03, band=band)
    grid.csg(hole, "diff")
    return grid


def _drop_box_onto_prism(register_grid, tmp_path) -> tuple[float, int]:
    """Drop a 4 cm box onto the L-prism; return (settled z, prism-box contact count)."""
    import genesis as gs

    ensure_genesis_initialized(use_cuda=False, seed=0)
    prism_path = tmp_path / "lprism.obj"
    tmp_path.mkdir(parents=True, exist_ok=True)
    _l_prism().export(prism_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=2),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(prism_path),
            pos=(0, 0, 0.004),
            convexify=False,
            decimate=False,
            watertighten=None,
        ),
        material=gs.materials.Rigid(),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.05, 0.15, 0.13)),
        material=gs.materials.Rigid(),
    )
    if register_grid is not None:
        geom = obj.desc.links[0].geoms[0]
        gc.register_mesh_sdf(geom.mesh, register_grid, fill="mesh")
    scene.build()
    for _ in range(400):
        scene.step()
    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    ga, gb = np.asarray(contacts["geom_a"]), np.asarray(contacts["geom_b"])
    prism_box = int(np.count_nonzero((ga == 2) & (gb == 1)))
    z = float(np.asarray(box.get_pos())[2])
    return z, prism_box


def test_register_mesh_sdf_drives_scene_contacts(tmp_path) -> None:
    """凹陷栅格烘焙后驱动真实接触：盒子落进凹坑，比平底停得更低。"""
    stock_z, stock_n = _drop_box_onto_prism(None, tmp_path / "stock")
    dent_z, dent_n = _drop_box_onto_prism(_dented_prism_grid(), tmp_path / "dent")
    assert stock_n > 0, "baseline must produce prism-box contacts"
    assert dent_n > 0, "baked field must produce prism-box contacts"
    assert stock_z - dent_z > 0.01, (
        f"box settled at {dent_z:.4f} over the dent vs {stock_z:.4f} on the flat "
        "top: the baked field is not driving contacts"
    )
