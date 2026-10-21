"""SdfRayCaster 测试：批量 DDA 命中/未命中、约束校验。

xuvdb 缺失时优雅 skip（CI 只装 .[dev]）。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optional_deps import xuvdb_only

pytestmark = xuvdb_only

from cloud_robotics_sim.vdb import XuvdbVolume  # noqa: E402
from cloud_robotics_sim.vdb.ray_caster import SdfRayCaster  # noqa: E402


def _sphere_volume() -> XuvdbVolume:
    """Level-set sphere SDF: centre (0.3, 0.2, 0.1), radius 0.25."""
    vol = XuvdbVolume(
        background=3 * 0.05,
        voxel_size=0.05,
        leaf_log2=4,
        name="ball",
        grid_class="level set",
    )
    vol.stamp_sphere((0.3, 0.2, 0.1), radius=0.25)
    return vol


def test_cast_hits_and_misses() -> None:
    """批量命中/未命中：miss 的 t 为 inf。"""
    caster = SdfRayCaster(_sphere_volume())
    origins = np.array(
        [
            [0.3, 0.2, 2.0],  # hits the sphere top
            [0.3, 0.2, 2.0],  # identical, hits as well
            [-3.0, -3.0, 2.0],  # far away, miss
        ],
        dtype=np.float64,
    )
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (3, 1)).astype(np.float64)
    t, points, values = caster.cast(origins, dirs)
    assert t.shape == (3,)
    assert points.shape == (3, 3)
    assert values.shape == (3,)
    assert t[0] == pytest.approx(2.0 - 0.35, abs=0.02)
    assert t[1] == pytest.approx(t[0])
    assert np.isinf(t[2])
    assert points[2].tolist() == [0.0, 0.0, 0.0]
    assert values[2] == 0.0
    assert values[0] == pytest.approx(0.0, abs=0.02)


def test_cast_empty_batch() -> None:
    """空批量返回空数组。"""
    caster = SdfRayCaster(_sphere_volume())
    t, points, values = caster.cast(np.zeros((0, 3)), np.zeros((0, 3)))
    assert t.shape == (0,)
    assert points.shape == (0, 3)
    assert values.shape == (0,)


def test_cast_shape_validation() -> None:
    """射线数不匹配与零方向报错。"""
    caster = SdfRayCaster(_sphere_volume())
    with pytest.raises(ValueError, match="ray count"):
        caster.cast(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="nonzero"):
        caster.cast(np.zeros((1, 3)), np.zeros((1, 3)))


def test_cast_matches_single_ray_api() -> None:
    """批量结果与 xuvdb 单射线 API 逐条一致。"""
    import xuvdb

    vol = _sphere_volume()
    caster = SdfRayCaster(vol)
    origins = np.array(
        [[0.3, 0.2, 2.0], [-1.0, 0.2, 0.1], [0.3, 0.2, -1.0]], dtype=np.float64
    )
    dirs = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    t, points, values = caster.cast(origins, dirs)
    for i in range(len(origins)):
        hit = xuvdb.ray_surface_hit(vol.grid, tuple(origins[i]), tuple(dirs[i]))
        if hit is None:
            assert np.isinf(t[i])
        else:
            ht, hp, hv = hit
            assert t[i] == pytest.approx(float(ht), abs=1e-6)
            assert values[i] == pytest.approx(float(hv), abs=1e-6)
            assert np.allclose(points[i], np.asarray(hp), atol=1e-6)


def test_constructor_rejects_non_volume() -> None:
    """非 XuvdbVolume 构造报 TypeError。"""
    with pytest.raises(TypeError):
        SdfRayCaster(object())


def test_constructor_rejects_rotated_grid() -> None:
    """带旋转 grid 构造报 ValueError。"""
    vol = _sphere_volume()
    c = np.cos(0.3)
    s = np.sin(0.3)
    vol.grid.rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    vol.grid.is_axis_aligned = False
    with pytest.raises(ValueError, match="axis-aligned"):
        SdfRayCaster(vol)
