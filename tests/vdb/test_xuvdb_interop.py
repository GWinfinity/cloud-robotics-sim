"""xuvdb 后端与互操作测试（xuvdb 缺失时优雅 skip；CI 只装 .[dev]）。"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optional_deps import xuvdb_only

pytestmark = xuvdb_only

from cloud_robotics_sim.vdb import (  # noqa: E402
    TorchSparseVolume,
    XuvdbVolume,
    create_volume,
    from_xuvdb,
    to_xuvdb,
)
from cloud_robotics_sim.vdb.core.vdb_xuvdb import (  # noqa: E402
    load_xuvdb,
    read_openvdb,
    save_xuvdb,
    write_openvdb,
)


def _sdf_volume() -> XuvdbVolume:
    """Level-set sphere SDF: distance - 0.25 (voxel_size 0.05)."""
    vol = XuvdbVolume(
        background=3 * 0.05,
        voxel_size=0.05,
        leaf_log2=4,
        name="shield",
        grid_class="level set",
    )
    vol.stamp_sphere((0.3, 0.2, 0.1), radius=0.25)
    return vol


def test_stamp_sphere_min_union_and_ray() -> None:
    """Test stamp sphere min union and ray."""
    vol = _sdf_volume()
    # Min-union: a second, smaller sphere merges into the field.
    vol.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)
    # Ray straight down onto the first sphere's top: z ≈ 0.1 + 0.25.
    hit = vol.ray_surface_hit((0.3, 0.2, 2.0), (0.0, 0.0, -1.0))
    assert hit is not None
    t, point, value = hit
    assert t == pytest.approx(2.0 - 0.35, abs=0.02)
    assert point[2] == pytest.approx(0.35, abs=0.02)
    assert value == pytest.approx(0.0, abs=0.02)
    # Sampling the centre is inside (negative SDF).
    pts = np.array([[0.3, 0.2, 0.1]], dtype=np.float32)
    assert np.atleast_1d(vol.sample_linear(pts))[0] == pytest.approx(-0.25, abs=0.02)
    grad = vol.sample_gradient(pts)
    assert grad.shape == (1, 3)


def test_value_editing_ops() -> None:
    """Test value editing ops."""
    vol = _sdf_volume()
    n_leaves = vol.n_active_blocks
    assert n_leaves >= 1
    vol.add_const(1.0)
    assert vol.reduce_max() == pytest.approx(0.15 + 1.0, abs=1e-4)
    vol.scale(2.0)
    assert vol.reduce_max() == pytest.approx((0.15 + 1.0) * 2.0, abs=1e-3)
    assert vol.n_active_blocks == n_leaves
    report = vol.memory_report()
    assert report["backend"] == "xuvdb"
    assert report["n_leaves"] == n_leaves


def test_to_xuvdb_from_xuvdb_roundtrip() -> None:
    """Test to xuvdb from xuvdb roundtrip."""
    n = 24
    idx = np.indices((n, n, n), dtype=np.float32)
    c = (n - 1) / 2.0
    dense = (np.linalg.norm(idx - c, axis=0) <= n / 4.0).astype(np.float32)
    torch_vol = TorchSparseVolume(nx=n, ny=n, nz=n, block_size=8, background=0.0)
    torch_vol.from_dense(dense, threshold=0.5)

    xvol = to_xuvdb(torch_vol, voxel_size=0.1, grid_class="fog volume")
    assert isinstance(xvol, XuvdbVolume)
    x_dense, ijk_min = xvol.to_dense()
    # xuvdb's dense export is the tight ACTIVE bbox, not the full input
    # window; compare against the same window of the torch volume.
    d0 = torch_vol.export_window(tuple(int(v) for v in ijk_min), x_dense.shape)
    np.testing.assert_allclose(x_dense, d0, atol=1e-6)

    back = from_xuvdb(xvol)
    assert isinstance(back, TorchSparseVolume)
    # The torch domain is rounded up to a block multiple; the dense window
    # starts at index 0 on both sides.
    slices = tuple(slice(0, s) for s in x_dense.shape)
    np.testing.assert_allclose(back.to_dense()[slices], x_dense, atol=1e-6)
    assert back.background == pytest.approx(0.0)


def test_xuvdb_file_roundtrip(tmp_path) -> None:
    """Test xuvdb file roundtrip."""
    vol = _sdf_volume()
    vol.prune()
    path = tmp_path / "scene.xuvdb"
    save_xuvdb(path, [vol], compress=True)
    assert path.exists()
    grids = load_xuvdb(path)
    assert len(grids) == 1
    assert grids[0].name == "shield"
    assert grids[0].leaf_log2 == 4
    d0, _ = vol.to_dense()
    d1, _ = grids[0].to_dense()
    np.testing.assert_allclose(d1, d0, atol=1e-6)


def test_openvdb_file_roundtrip(tmp_path) -> None:
    """Native .vdb interop (xuvdb's own reader/writer, no pyopenvdb)."""
    vol = _sdf_volume()
    path = tmp_path / "scene.vdb"
    write_openvdb(path, [vol])
    assert path.exists()
    # OpenVDB magic bytes "VDB " (0x56444220 little-endian).
    assert path.read_bytes()[:8] == (0x56444220).to_bytes(8, "little")
    back = read_openvdb(path, grid_name="shield")
    assert isinstance(back, XuvdbVolume)
    d0, _ = vol.to_dense()
    d1, _ = back.to_dense()
    assert d1.shape == d0.shape
    np.testing.assert_allclose(d1, d0, atol=1e-5)


def test_create_volume_xuvdb_and_auto() -> None:
    """Test create volume xuvdb and auto."""
    vol = create_volume("xuvdb", background=0.5, voxel_size=0.1, name="auto")
    assert isinstance(vol, XuvdbVolume)
    assert vol.background == pytest.approx(0.5)
    assert vol.voxel_size == (0.1, 0.1, 0.1)
    # auto: xuvdb is installed in this environment, so it wins.
    auto = create_volume("auto", nx=8, ny=8, nz=8, background=0.0)
    assert isinstance(auto, XuvdbVolume)


def test_create_volume_xuvdb_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment without the extra: ImportError names the extra."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "xuvdb" or name.startswith("xuvdb."):
            raise ImportError("No module named 'xuvdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.invalidate_caches()
    with pytest.raises(ImportError, match=r"cloud-robotics-sim\[xuvdb\]"):
        create_volume("xuvdb")
    # auto falls back to torch when xuvdb is unavailable.
    vol = create_volume("auto", nx=8, ny=8, nz=8)
    assert isinstance(vol, TorchSparseVolume)


def test_union_spheres_particle_level_set() -> None:
    """Test union spheres particle level set."""
    drops = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    vol = XuvdbVolume(background=0.15, voxel_size=0.05, grid_class="level set")
    vol.union_spheres(drops, radius=0.03)
    assert vol.active_voxel_count > 0
    # Ray straight down onto the first particle's north pole (z = +radius).
    hit = vol.ray_surface_hit((0.1, 0.0, 1.0), (0.0, 0.0, -1.0))
    assert hit is not None
    _, point, _ = hit
    assert point[2] == pytest.approx(0.03, abs=0.02)
