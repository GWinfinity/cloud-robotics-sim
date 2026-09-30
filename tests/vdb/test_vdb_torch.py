"""TorchSparseVolume / create_volume("torch") 基础测试（无需 xuvdb）。"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optional_deps import torch_only

pytestmark = torch_only

from cloud_robotics_sim.vdb import TorchSparseVolume, create_volume  # noqa: E402


def _demo_volume(n: int = 32) -> TorchSparseVolume:
    """Sphere-indicator volume: 1.0 inside radius n/8, 0.0 elsewhere.

    The small radius leaves whole 8^3 blocks inactive, so the volume is
    genuinely sparse (active_fill_ratio well below 1).
    """
    idx = np.indices((n, n, n), dtype=np.float32)
    c = (n - 1) / 2.0
    r = n / 8.0
    dense = (np.linalg.norm(idx - c, axis=0) <= r).astype(np.float32)
    vol = TorchSparseVolume(nx=n, ny=n, nz=n, block_size=8, background=0.0)
    vol.from_mask(dense > 0.5)
    # Import only the active bounding box: a whole-domain import_window
    # would activate every block (from_dense does that by design).
    coords = np.argwhere(dense > 0.5)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    vol.import_window(
        tuple(int(v) for v in lo), dense[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    )
    return vol


def test_activate_deactivate_roundtrip() -> None:
    """Test activate deactivate roundtrip."""
    vol = TorchSparseVolume(nx=16, ny=16, nz=16, block_size=8)
    assert vol.n_active_blocks == 0
    added = vol.activate_blocks(np.array([[0, 0, 0], [1, 1, 1]]))
    assert added == 2
    assert vol.n_active_blocks == 2
    # Re-activating an existing block is a no-op.
    assert vol.activate_blocks(np.array([[0, 0, 0]])) == 0
    removed = vol.deactivate_blocks(np.array([[0, 0, 0]]))
    assert removed == 1
    assert vol.n_active_blocks == 1


def test_from_mask_and_topology() -> None:
    """Test from mask and topology."""
    vol = TorchSparseVolume(nx=16, ny=16, nz=16, block_size=8)
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[2, 2, 2] = True  # block (0,0,0) only
    mask[9, 9, 9] = True  # block (1,1,1) only
    vol.from_mask(mask)
    table = vol.topology()
    assert table[0, 0, 0] >= 0
    assert table[1, 1, 1] >= 0
    assert table[0, 0, 1] == -1


def test_from_dense_to_dense_roundtrip() -> None:
    """Test from dense to dense roundtrip."""
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((24, 24, 24), dtype=np.float32)
    dense[dense < 0.5] = 0.0
    vol = TorchSparseVolume(nx=24, ny=24, nz=24, block_size=8, background=0.0)
    vol.from_dense(dense, threshold=0.0)
    out = vol.to_dense()
    np.testing.assert_allclose(out, dense, atol=1e-6)


def test_laplacian_constant_field_is_zero() -> None:
    """Test laplacian constant field is zero."""
    # Fill value == background: every neighbour (including out-of-domain
    # and inactive-block ones, which read `background`) sees 3.0, so the
    # 6-neighbour stencil vanishes everywhere.
    vol = TorchSparseVolume(nx=16, ny=16, nz=16, block_size=8, background=3.0)
    out = TorchSparseVolume(nx=16, ny=16, nz=16, block_size=8, background=3.0)
    # Activate the whole domain so to_dense has no background-only rim.
    all_blocks = np.array(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)]
    )
    vol.activate_blocks(all_blocks)
    vol.fill(3.0)
    out.activate_blocks(all_blocks)
    vol.laplacian(out)
    dense = out.to_dense()
    assert np.max(np.abs(dense)) < 1e-5


def test_laplacian_spike_matches_dense_stencil() -> None:
    """Test laplacian spike matches dense stencil."""
    n = 16
    dense = np.zeros((n, n, n), dtype=np.float32)
    dense[8, 8, 8] = 1.0
    vol = TorchSparseVolume(nx=n, ny=n, nz=n, block_size=8, background=0.0)
    vol.from_dense(dense, threshold=0.0)
    # Identical construction -> identical topology (required by laplacian).
    out = TorchSparseVolume(nx=n, ny=n, nz=n, block_size=8, background=0.0)
    out.from_dense(dense, threshold=0.0)
    vol.laplacian(out)
    got = out.to_dense()
    want = (
        np.roll(dense, 1, 0)
        + np.roll(dense, -1, 0)
        + np.roll(dense, 1, 1)
        + np.roll(dense, -1, 1)
        + np.roll(dense, 1, 2)
        + np.roll(dense, -1, 2)
        - 6.0 * dense
    )
    # Roll wraps at the domain boundary; the volume returns background (0)
    # there, so clear the wrapped rows/cols before comparing.
    want[0, :, :] = want[-1, :, :] = 0.0
    want[:, 0, :] = want[:, -1, :] = 0.0
    want[:, :, 0] = want[:, :, -1] = 0.0
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_axpy_and_reduce() -> None:
    """Test axpy and reduce."""
    x = _demo_volume()
    y = _demo_volume()
    # Identical construction order keeps the internal slot order identical
    # (axpy requires exactly the same topology, including slot layout).
    out = _demo_volume()
    out.axpy(2.0, x, 3.0, y)
    expected = 5.0 * x.to_dense()
    np.testing.assert_allclose(out.to_dense(), expected, atol=1e-5)
    assert out.reduce_sum() == pytest.approx(float(expected.sum()), rel=1e-5)
    assert out.reduce_max() == pytest.approx(5.0, rel=1e-5)


def test_memory_report() -> None:
    """Test memory report."""
    vol = _demo_volume()
    report = vol.memory_report()
    assert report["backend"] == "torch"
    assert report["active_blocks"] >= 1
    assert report["active_voxels"] > 0
    assert 0.0 < report["active_fill_ratio"] < 1.0


def test_create_volume_torch() -> None:
    """Test create volume torch."""
    vol = create_volume("torch", nx=8, ny=8, nz=8, block_size=8)
    assert isinstance(vol, TorchSparseVolume)
    assert vol.backend == "torch"


def test_create_volume_unknown_backend_raises() -> None:
    """Test create volume unknown backend raises."""
    with pytest.raises(ValueError, match="unknown vdb backend"):
        create_volume("bogus")
