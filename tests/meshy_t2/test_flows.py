"""Tests for the Voxel VAE and both flow-matching stages."""

import numpy as np
import torch

from cloud_robotics_sim.meshy_t2.flow import (
    cfg_velocity,
    euler_sample,
    interpolate,
    sample_t,
)
from cloud_robotics_sim.meshy_t2.models.mesh_flow import (
    MeshFlow,
    MeshFlowConfig,
    assign_latent_coords,
    image_token_coords,
    voxel_token_coords,
)
from cloud_robotics_sim.meshy_t2.models.voxel_flow import VoxelFlow, VoxelFlowConfig
from cloud_robotics_sim.meshy_t2.models.voxel_vae import (
    PixelShuffle3d,
    VoxelVAE,
    VoxelVAEConfig,
)
from cloud_robotics_sim.meshy_t2.sobol_ot import (
    morton_assign,
    ot_assign,
    sobol_points,
)

# ---------------------------------------------------------------------------
# Voxel VAE
# ---------------------------------------------------------------------------


def test_pixel_shuffle_3d():
    """Check pixel shuffle 3d."""
    x = torch.randn(2, 16, 4, 4, 4)
    y = PixelShuffle3d(2)(x)
    assert y.shape == (2, 2, 8, 8, 8)


def test_voxel_vae_roundtrip_and_loss():
    """Check voxel vae roundtrip and loss."""
    torch.manual_seed(0)
    cfg = VoxelVAEConfig.tiny()
    model = VoxelVAE(cfg)
    occ = torch.zeros(2, 1, cfg.res, cfg.res, cfg.res)
    occ[:, :, 8:24, 8:24, 8:24] = 1.0  # solid block
    logits, mu, logvar = model(occ)
    assert logits.shape == occ.shape
    assert mu.shape == (
        2,
        cfg.latent_channels,
        cfg.latent_res,
        cfg.latent_res,
        cfg.latent_res,
    )
    loss, parts = model.loss(occ)
    assert torch.isfinite(loss)
    assert set(parts) == {"bce", "kl"}
    loss.backward()


def test_voxel_vae_encode_deterministic():
    """Check voxel vae encode deterministic."""
    cfg = VoxelVAEConfig.tiny()
    model = VoxelVAE(cfg)
    occ = torch.rand(1, 1, cfg.res, cfg.res, cfg.res).round()
    z1 = model.encode_deterministic(occ)
    z2 = model.encode_deterministic(occ)
    torch.testing.assert_close(z1, z2)
    decoded = model.decode_occupancy(z1)
    assert decoded.shape == occ.shape


# ---------------------------------------------------------------------------
# Flow helpers
# ---------------------------------------------------------------------------


def test_interpolate_endpoints():
    """Check interpolate endpoints."""
    x0 = torch.zeros(3, 5)
    x1 = torch.ones(3, 5)
    x_t, v = interpolate(x0, x1, torch.zeros(3))
    torch.testing.assert_close(x_t, x0)
    torch.testing.assert_close(v, x1 - x0)
    x_t, _ = interpolate(x0, x1, torch.ones(3))
    torch.testing.assert_close(x_t, x1)


def test_sample_t_logitnormal_range():
    """Check sample t logitnormal range."""
    t = sample_t(1024)
    assert (t > 0).all() and (t < 1).all()


def test_euler_sample_straight_flow():
    """Check euler sample straight flow."""
    # Constant velocity v = -x1 moves x to 0 in one unit of time.
    x1 = torch.ones(1, 4)
    out = euler_sample(lambda x, t: torch.full_like(x, 1.0), x1, steps=4)
    torch.testing.assert_close(out, torch.zeros(1, 4))


def test_cfg_velocity_scales():
    """Check cfg velocity scales."""

    def vf(x, t, c):
        return x * c

    call = cfg_velocity(vf, {"c": 2.0}, {"c": 0.0}, guidance=3.0)
    x = torch.ones(2, 3)
    out = call(x, torch.zeros(2))
    torch.testing.assert_close(out, 2.0 * 3.0 * x)


# ---------------------------------------------------------------------------
# Stage I voxel flow
# ---------------------------------------------------------------------------


def test_voxel_flow_forward():
    """Check voxel flow forward."""
    torch.manual_seed(0)
    cfg = VoxelFlowConfig.tiny()
    flow = VoxelFlow(cfg)
    n = cfg.latent_res**3
    x = torch.randn(2, n, cfg.latent_channels)
    t = sample_t(2)
    img = torch.randn(2, 16, cfg.image_dim)
    v = flow(x, t, img)
    assert v.shape == x.shape
    # Unconditional branch (image dropped) also works.
    v_un = flow(x, t, None)
    assert v_un.shape == x.shape


# ---------------------------------------------------------------------------
# Stage II mesh flow
# ---------------------------------------------------------------------------


def _mesh_flow_inputs(cfg: MeshFlowConfig, batch: int = 1, slots: int = 24):
    x = torch.randn(batch, slots, cfg.flow_channels)
    t = sample_t(batch)
    coords = torch.rand(slots, 3)
    img = torch.randn(batch, 16, cfg.image_dim)
    img_coords = image_token_coords(16)
    vox = torch.randn(batch, cfg.voxel_res**3, cfg.voxel_channels)
    vox_coords = voxel_token_coords(cfg.voxel_res)
    count = torch.tensor([float(slots)] * batch)
    return x, t, coords, img, img_coords, vox, vox_coords, count


def test_mesh_flow_forward_full_conditioning():
    """Check mesh flow forward full conditioning."""
    torch.manual_seed(0)
    cfg = MeshFlowConfig.tiny()
    flow = MeshFlow(cfg)
    x, t, coords, img, ic, vox, vc, count = _mesh_flow_inputs(cfg)
    v = flow(x, t, coords, img, ic, vox, vc, count)
    assert v.shape == x.shape


def test_mesh_flow_forward_dropped_conditions():
    """Check mesh flow forward dropped conditions."""
    cfg = MeshFlowConfig.tiny()
    flow = MeshFlow(cfg)
    x, t, coords, *_ = _mesh_flow_inputs(cfg)
    v = flow(x, t, coords)  # all conditions dropped
    assert v.shape == x.shape


def test_image_token_coords_separate_slice():
    """Check image token coords separate slice."""
    coords = image_token_coords(64)
    assert coords.shape == (64, 3)
    assert (coords[:, 2] == -0.5).all()  # image tokens on their own slice


# ---------------------------------------------------------------------------
# Sobol OT
# ---------------------------------------------------------------------------


def test_sobol_points_deterministic():
    """Check sobol points deterministic."""
    a = sobol_points(32, seed=0)
    b = sobol_points(32, seed=0)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0 and a.max() < 1


def test_ot_assign_beats_random():
    """Check ot assign beats random."""
    rng = np.random.default_rng(0)
    points = rng.random((50, 3))
    candidates = points + rng.normal(scale=0.01, size=points.shape)
    assign = ot_assign(points, candidates)
    assert len(set(assign.tolist())) == 50  # a valid assignment (bijection)
    cost = ((points - candidates[assign]) ** 2).sum(-1).mean()
    rand = rng.permutation(50)
    cost_rand = ((points - candidates[rand]) ** 2).sum(-1).mean()
    assert cost < cost_rand


def test_morton_assign_valid():
    """Check morton assign valid."""
    rng = np.random.default_rng(1)
    points = rng.random((20, 3))
    candidates = sobol_points(20, seed=3)
    assign = morton_assign(points, candidates)
    assert sorted(assign.tolist()) == list(range(20))


def test_assign_latent_coords_slots():
    """Check assign latent coords slots."""
    rng = np.random.default_rng(2)
    verts = rng.random((10, 3))
    coords = assign_latent_coords(verts, num_slots=14, seed=0)
    assert coords.shape == (14, 3)
    # First 10 rows are OT-matched candidates; remaining 4 are leftovers.
    assert np.isfinite(coords).all()
