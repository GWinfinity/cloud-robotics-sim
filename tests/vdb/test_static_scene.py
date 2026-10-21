"""static_scene 底座测试：三角汤 -> SDF、缓存命中/失效、缓存键。

xuvdb / trimesh 缺失时优雅 skip（CI 只装 .[dev]）。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from tests.optional_deps import HAS_TRIMESH, xuvdb_only

pytestmark = [
    xuvdb_only,
    pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed"),
]

from cloud_robotics_sim.vdb.core.static_scene import (  # noqa: E402
    bake_static_scene,
    cache_key_for_bake,
    meshes_to_sdf,
)


def _box_soup(
    center: tuple[float, float, float],
    extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.creation.box(extents=extents)
    v = np.asarray(mesh.vertices, dtype=np.float64) + np.asarray(center)
    return v, np.asarray(mesh.faces, dtype=np.int64)


def test_meshes_to_sdf_two_boxes_signs() -> None:
    """两个封闭盒子的三角汤 -> 盒内负、盒间/远处正。"""
    vol = meshes_to_sdf(
        [
            _box_soup((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            _box_soup((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        ],
        voxel_size=0.02,
        name="two_boxes",
    )
    assert vol.grid.grid_class == "level set"
    # Box interiors are negative.
    for c in ((-0.15, 0.0, 0.0), (1.15, 0.0, 0.0)):
        d = float(np.atleast_1d(vol.sample_linear(np.array([c], dtype=np.float32)))[0])
        assert d < 0.0, f"inside box at {c}: {d}"
    # Gap between boxes and far field are positive.
    for c in ((0.5, 0.0, 0.0), (-2.0, 0.0, 0.0)):
        d = float(np.atleast_1d(vol.sample_linear(np.array([c], dtype=np.float32)))[0])
        assert d > 0.0, f"outside boxes at {c}: {d}"
    report = vol.memory_report()
    assert report["active_voxels"] > 0
    assert report["n_leaves"] > 0


def test_meshes_to_sdf_rejects_empty() -> None:
    """空条目列表必须报 ValueError。"""
    with pytest.raises(ValueError, match="no mesh entries"):
        meshes_to_sdf([], voxel_size=0.1)


def test_bake_static_scene_cache_hit_and_invalidation(tmp_path, monkeypatch) -> None:
    """二次调用命中缓存（不重烘）；改 mtime 或 pose 则缓存键变化。"""
    files = []
    for i, c in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))):
        v, f = _box_soup(c, (0.4, 0.4, 0.4))
        path = tmp_path / f"box_{i}.stl"
        trimesh.Trimesh(vertices=v, faces=f).export(path)
        files.append(path)

    calls = {"n": 0}
    import cloud_robotics_sim.vdb.core.static_scene as static_scene

    real_meshes_to_sdf = static_scene.meshes_to_sdf

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_meshes_to_sdf(*args, **kwargs)

    monkeypatch.setattr(static_scene, "meshes_to_sdf", counting)

    kw = dict(voxel_size=0.02, band=3.0, cache_dir=tmp_path / "cache", name="pair")
    v1 = bake_static_scene(files, **kw)
    assert calls["n"] == 1
    v2 = bake_static_scene(files, **kw)
    assert calls["n"] == 1, "second bake should hit the cache"
    assert v2.grid.name == v1.grid.name
    assert v2.active_voxel_count == v1.active_voxel_count

    # Touch a file -> new mtime -> cache miss.
    import os
    import time

    time.sleep(0.01)
    os.utime(files[0])
    bake_static_scene(files, **kw)
    assert calls["n"] == 2

    # A pose change also changes the cache key (no bake executed: key differ).
    t = np.eye(4)
    t[:3, 3] = (5.0, 0.0, 0.0)
    key_id = cache_key_for_bake(files, None, voxel_size=0.02, band=3.0, name="pair")
    key_moved = cache_key_for_bake(
        files, [np.eye(4), t], voxel_size=0.02, band=3.0, name="pair"
    )
    assert key_id != key_moved


def test_bake_static_scene_missing_file(tmp_path) -> None:
    """缺失网格文件报 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        bake_static_scene([tmp_path / "nope.glb"], voxel_size=0.1)


def test_cache_key_depends_on_params(tmp_path) -> None:
    """缓存键随体素尺寸/band/名称变化。"""
    v, f = _box_soup((0, 0, 0), (0.4, 0.4, 0.4))
    path = tmp_path / "b.stl"
    trimesh.Trimesh(vertices=v, faces=f).export(path)
    base = cache_key_for_bake([path], None, voxel_size=0.02, band=3.0, name="b")
    assert base != cache_key_for_bake([path], None, voxel_size=0.04, band=3.0, name="b")
    assert base != cache_key_for_bake([path], None, voxel_size=0.02, band=5.0, name="b")
    assert base != cache_key_for_bake([path], None, voxel_size=0.02, band=3.0, name="c")
