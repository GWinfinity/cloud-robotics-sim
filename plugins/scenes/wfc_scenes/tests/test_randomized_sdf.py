"""Room-content domain randomization tests (offline; xuvdb required).

No Genesis needed: structure/variant bakes are trimesh + xuvdb, and every
query goes through the packed VolumeBatch.
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

from plugins.scenes.wfc_scenes.randomized_sdf import RandomizedRoomSDF  # noqa: E402
from plugins.scenes.wfc_scenes.scene import ChineseHomeScene  # noqa: E402
from plugins.scenes.wfc_scenes.sdf_bake import (  # noqa: E402
    bake_room_variant,
    bake_structure,
)

try:
    from tests.optional_deps import xuvdb_only
except ImportError:  # plugin tests also run standalone (repo root off sys.path)
    xuvdb_only = pytest.mark.skipif(
        importlib.util.find_spec("xuvdb") is None, reason="xuvdb not installed"
    )

pytestmark = [xuvdb_only]

VARIANTS = ["kitchen", "bedroom"]


def _small_home() -> ChineseHomeScene:
    return ChineseHomeScene(grid=(3, 2), seed=42)


def _target_cell(home: ChineseHomeScene) -> int:
    """The living-room cell when present (richest furniture), else cell 0."""
    return next(
        (k for k, c in enumerate(home.layout.cells) if c["tile"] == "living"),
        0,
    )


def _batch_sample(vol, pts: np.ndarray) -> np.ndarray:
    """Batched trilinear sample.

    The host ``VdbGrid.sample_linear`` truncates to its first point
    (``...)[0]``, a documented xuvdb quirk) — queries here go through the
    kernel-side ``GpuVolume`` sampler like :class:`RandomizedRoomSDF` does.
    """
    import xuvdb

    gpu = xuvdb.GpuVolume(vol.grid)
    return np.asarray(gpu.sample(np.ascontiguousarray(pts, dtype=np.float32), order=1))


def _room_lattice(home: ChineseHomeScene, cell_index: int, n: int = 6) -> np.ndarray:
    """World-frame sample lattice filling the target room volume."""
    layout = home.layout
    cell = layout.cells[cell_index]
    cx, cy = layout.cell_center(cell["i"], cell["j"])
    hx, hy = layout.cell_half(cell["i"], cell["j"])
    xs = np.linspace(cx - 0.8 * hx, cx + 0.8 * hx, n)
    ys = np.linspace(cy - 0.8 * hy, cy + 0.8 * hy, n)
    zs = np.linspace(0.2, 1.8, n)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def home() -> ChineseHomeScene:
    """Shared 3x2 seed-42 home (module scope keeps bakes amortized)."""
    return _small_home()


@pytest.fixture(scope="module")
def cell_index(home: ChineseHomeScene) -> int:
    """Cell being randomized in the shared home."""
    return _target_cell(home)


def test_bake_structure_sign_census(home, tmp_path) -> None:
    """壳体（墙/地板）烘出符号健康的 level set。"""
    vol = bake_structure(
        home, voxel_size=0.1, cache_dir=tmp_path / "cache", name="struct"
    )
    assert vol.grid.grid_class == "level set"
    n_neg = sum(int((leaf.values < 0).sum()) for leaf in vol.grid.leaves())
    n_pos = sum(int((leaf.values > 0).sum()) for leaf in vol.grid.leaves())
    assert n_neg > 0, "no interior voxels: structure bake looks empty"
    assert n_pos > 0, "no exterior voxels: structure bake looks solid"


def test_room_variants_differ(home, cell_index, tmp_path) -> None:
    """同一 cell 换 tile：家具 SDF 必须不同（域随机化的存在性）。"""
    vols = [
        bake_room_variant(
            home,
            cell_index,
            tile,
            voxel_size=0.1,
            cache_dir=tmp_path / "cache",
            name=f"var-{tile}",
        )
        for tile in VARIANTS
    ]
    pts = _room_lattice(home, cell_index)
    a = _batch_sample(vols[0], pts)
    b = _batch_sample(vols[1], pts)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(
        a, b, atol=1e-6
    ), f"variants {VARIANTS} sampled identically — randomization is a no-op"


def test_randomized_sdf_sample_min_composition(home, cell_index, tmp_path) -> None:
    """Sample = min(结构, 变体)；ids=None 时退化为纯结构。"""
    rsdf = RandomizedRoomSDF.build(
        home,
        {cell_index: VARIANTS},
        voxel_size=0.1,
        cache_dir=tmp_path / "cache",
    )
    pts = _room_lattice(home, cell_index)
    id_a = rsdf.id_for(cell_index, VARIANTS[0])
    id_b = rsdf.id_for(cell_index, VARIANTS[1])

    bare = rsdf.sample(pts)
    struct = bake_structure(
        home, voxel_size=0.1, cache_dir=tmp_path / "cache", name="struct"
    )
    expect_bare = _batch_sample(struct, pts)
    assert np.allclose(bare, expect_bare, atol=1e-5)

    got_a = rsdf.sample(pts, np.full(len(pts), id_a))
    var_a = bake_room_variant(
        home,
        cell_index,
        VARIANTS[0],
        voxel_size=0.1,
        cache_dir=tmp_path / "cache",
        name=f"var-{VARIANTS[0]}",
    )
    expect_a = np.minimum(expect_bare, _batch_sample(var_a, pts))
    assert np.allclose(got_a, expect_a, atol=1e-5)

    # Two variants through one batched call must differ somewhere.
    both = rsdf.sample(
        np.concatenate([pts, pts]),
        np.array([id_a] * len(pts) + [id_b] * len(pts)),
    )
    assert not np.allclose(both[: len(pts)], both[len(pts) :], atol=1e-6)


def test_randomized_sdf_cast_hits_and_variants(home, cell_index, tmp_path) -> None:
    """向下射线：两种内容都命中，且至少部分射线命中不同表面。"""
    rsdf = RandomizedRoomSDF.build(
        home,
        {cell_index: VARIANTS},
        voxel_size=0.1,
        cache_dir=tmp_path / "cache",
    )
    layout = home.layout
    cell = layout.cells[cell_index]
    cx, cy = layout.cell_center(cell["i"], cell["j"])
    hx, hy = layout.cell_half(cell["i"], cell["j"])
    xs = np.linspace(cx - 0.7 * hx, cx + 0.7 * hx, 8)
    ys = np.linspace(cy - 0.7 * hy, cy + 0.7 * hy, 8)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)], axis=1)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(origins), 1))

    t_bare, _, _ = rsdf.cast(origins, dirs)
    assert np.isfinite(t_bare).all(), "structure-only cast missed the room"

    id_a = rsdf.id_for(cell_index, VARIANTS[0])
    id_b = rsdf.id_for(cell_index, VARIANTS[1])
    t_a, p_a, _ = rsdf.cast(origins, dirs, np.full(len(origins), id_a))
    t_b, p_b, _ = rsdf.cast(origins, dirs, np.full(len(origins), id_b))
    assert np.isfinite(t_a).all() and np.isfinite(t_b).all()
    assert np.allclose(p_a[:, 2], 2.5 - t_a, atol=1e-6)  # vertical rays
    assert not np.allclose(
        p_a[:, 2], p_b[:, 2], atol=1e-3
    ), "both variants hit identical surfaces — randomization is a no-op"


def test_room_variant_cache_hit(home, cell_index, tmp_path, monkeypatch) -> None:
    """同 layout/cell/tile 命中缓存；换 tile 则 miss。"""
    import plugins.scenes.wfc_scenes.sdf_bake as sdf_bake

    calls = {"n": 0}
    real = sdf_bake._prims_to_sdf

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sdf_bake, "_prims_to_sdf", counting)

    kw = dict(voxel_size=0.1, cache_dir=tmp_path / "cache")
    bake_room_variant(home, cell_index, VARIANTS[0], **kw)
    assert calls["n"] == 1
    bake_room_variant(home, cell_index, VARIANTS[0], **kw)
    assert calls["n"] == 1, "second bake should hit the cache"
    bake_room_variant(home, cell_index, VARIANTS[1], **kw)
    assert calls["n"] == 2


def test_validation_errors(home, cell_index, tmp_path) -> None:
    """非法 tile / cell / id 均给出明确错误。"""
    with pytest.raises(ValueError, match="unknown tile"):
        bake_room_variant(
            home, cell_index, "pool", voxel_size=0.1, cache_dir=tmp_path / "c"
        )
    with pytest.raises(ValueError, match="out of range"):
        bake_room_variant(
            home, 999, VARIANTS[0], voxel_size=0.1, cache_dir=tmp_path / "c"
        )
    with pytest.raises(ValueError, match="unknown tile"):
        RandomizedRoomSDF.build(home, {cell_index: ["pool"]}, cache_dir=tmp_path / "c")
    with pytest.raises(ValueError, match="no room variants"):
        RandomizedRoomSDF.build(home, {}, cache_dir=tmp_path / "c")

    rsdf = RandomizedRoomSDF.build(
        home, {cell_index: VARIANTS}, voxel_size=0.1, cache_dir=tmp_path / "c"
    )
    with pytest.raises(KeyError, match="no baked variant"):
        rsdf.id_for(cell_index, "dining")
    pts = _room_lattice(home, cell_index, n=2)
    with pytest.raises(IndexError, match="out of range"):
        rsdf.sample(pts, np.full(len(pts), 999))
    with pytest.raises(IndexError, match="out of range"):
        rsdf.cast(pts, np.tile([[0.0, 0.0, -1.0]], (len(pts), 1)), np.zeros(len(pts)))


def test_determinism(home, cell_index, tmp_path) -> None:
    """同一 layout + 变体重建：采样值完全一致。"""
    kw = dict(voxel_size=0.1, cache_dir=tmp_path / "cache")
    a = RandomizedRoomSDF.build(home, {cell_index: VARIANTS}, **kw)
    b = RandomizedRoomSDF.build(home, {cell_index: VARIANTS}, **kw)
    pts = _room_lattice(home, cell_index)
    ids = np.full(len(pts), a.id_for(cell_index, VARIANTS[0]))
    assert np.array_equal(a.sample(pts, ids), b.sample(pts, ids))
