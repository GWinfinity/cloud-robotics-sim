"""WFC layout SDF bake tests (offline; trimesh required, xuvdb required).

The ShapeKernel bridge test additionally requires the sibling shapekernel
package; it auto-skips without it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from plugins.scenes.wfc_scenes.scene import ChineseHomeScene  # noqa: E402
from plugins.scenes.wfc_scenes.sdf_bake import (  # noqa: E402
    bake_layout,
    rescale_grid,
    shapekernel_prims_to_sdf,
)

HAS_SHAPEKERNEL = importlib.util.find_spec("shapekernel") is not None
shapekernel_only = pytest.mark.skipif(
    not HAS_SHAPEKERNEL, reason="shapekernel sibling package not installed"
)

try:
    from tests.optional_deps import xuvdb_only
except ImportError:  # plugin tests also run standalone (repo root off sys.path)
    xuvdb_only = pytest.mark.skipif(
        importlib.util.find_spec("xuvdb") is None, reason="xuvdb not installed"
    )

pytestmark = [xuvdb_only]


def _small_home() -> ChineseHomeScene:
    return ChineseHomeScene(grid=(3, 2), seed=42)


def test_bake_layout_produces_sane_sdf(tmp_path) -> None:
    """布局烘焙出符号健康的 level set。"""
    home = _small_home()
    vol = bake_layout(home, voxel_size=0.1, cache_dir=tmp_path / "cache", name="home42")
    assert vol.grid.grid_class == "level set"
    report = vol.memory_report()
    assert report["active_voxels"] > 0

    # Sign census over the dense leaf buffers: walls/floors are 1-2 voxels
    # thick at this voxel size, so random *linear* samples would interpolate
    # the thin negative regions away -- assert on the voxel values directly.
    n_neg = sum(int((leaf.values < 0).sum()) for leaf in vol.grid.leaves())
    n_pos = sum(int((leaf.values > 0).sum()) for leaf in vol.grid.leaves())
    assert n_neg > 0, "no interior voxels: bake looks empty"
    assert n_pos > 0, "no exterior voxels: bake looks solid"

    # Trilinear sampling stays finite and positive in open air.
    lo, hi = report["bbox_ijk"]
    voxel = np.float32(report["voxel_size"])
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(1024, 3)).astype(np.float32) * voxel
    vals = np.atleast_1d(vol.sample_linear(pts))
    assert np.all(np.isfinite(vals))
    assert (vals > 0).sum() > 0


def test_bake_layout_cache_hit(tmp_path, monkeypatch) -> None:
    """同布局 JSON 命中缓存；seed 变化则 miss。"""
    import plugins.scenes.wfc_scenes.sdf_bake as sdf_bake

    calls = {"n": 0}
    real = sdf_bake.meshes_to_sdf

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sdf_bake, "meshes_to_sdf", counting)

    layout = _small_home().export_layout(tmp_path / "layout.json")
    kw = dict(voxel_size=0.1, cache_dir=tmp_path / "cache", name="home42")
    bake_layout(layout, **kw)
    assert calls["n"] == 1
    bake_layout(layout, **kw)
    assert calls["n"] == 1, "second bake should hit the cache"

    # Same layout JSON but a different seed field must miss the cache.
    import json

    data = json.loads(Path(layout).read_text(encoding="utf-8"))
    data["seed"] = 43
    bake_layout(data, **kw)
    assert calls["n"] == 2


def test_rescale_grid_mm_to_m() -> None:
    """mm->m 均匀缩放：值/体素/原点/背景全部线性。"""
    import xuvdb

    grid = xuvdb.VdbGrid(
        background=0.3, voxel_size=2.0, grid_class="level set", name="mm"
    )
    grid.stamp_sphere((100.0, 0.0, 0.0), radius=50.0)  # mm-world sphere
    rescale_grid(grid, 1e-3)
    assert np.allclose(grid.voxel_size, (0.002, 0.002, 0.002))
    assert grid.background == pytest.approx(0.3 * 1e-3)
    # Sphere centre now at 0.1 m; interior sample is negative at mm-scale value*1e-3.
    pts = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
    d = float(np.atleast_1d(grid.sample_linear(pts))[0])
    assert d < 0.0


@shapekernel_only
def test_shapekernel_prims_to_sdf_mm_to_m(tmp_path) -> None:
    """BaseBox mm 构造 -> 米框架 SDF，内外符号正确。"""
    from shapekernel import LocalFrame
    from shapekernel.baseshapes import BaseBox

    # BaseBox: length along local z, width x, depth y (mm here).
    box = BaseBox(LocalFrame((0, 0, 0)), length=40.0, width=100.0, depth=60.0)
    vol = shapekernel_prims_to_sdf([box], voxel_size=2.0)
    assert vol.grid.name == "shapekernel_scene"
    # Interior of the 100x60x40 mm box, now in meters (stay clear of the
    # faces: the old single-point host path truncated this batch to row 0
    # and never evaluated the corner point exactly on the surface):
    pts = np.array([[0.0, 0.0, 0.02], [0.04, 0.02, 0.015]], dtype=np.float32)
    vals = np.atleast_1d(vol.sample_linear(pts))
    assert np.all(vals < 0)
    # Outside along x:
    pts_out = np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
    assert float(np.atleast_1d(vol.sample_linear(pts_out))[0]) > 0


def test_shapekernel_prims_to_sdf_error_names_extra() -> None:
    """Shapekernel 缺席时桥函数报 ImportError 并指明 extra。"""
    if HAS_SHAPEKERNEL:
        pytest.skip("shapekernel installed; ImportError path not reachable")
    with pytest.raises(ImportError, match="shapekernel"):
        shapekernel_prims_to_sdf([object()])
