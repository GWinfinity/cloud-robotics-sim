"""End-to-end pipeline tests with tiny configs and random weights."""

import numpy as np
import torch

from cloud_robotics_sim.meshy_t2.models.image_encoder import TinyViTImageEncoder
from cloud_robotics_sim.meshy_t2.models.mesh_flow import MeshFlow, MeshFlowConfig
from cloud_robotics_sim.meshy_t2.models.mesh_vae import MeshVAE, MeshVAEConfig
from cloud_robotics_sim.meshy_t2.models.voxel_flow import VoxelFlow, VoxelFlowConfig
from cloud_robotics_sim.meshy_t2.models.voxel_vae import VoxelVAE, VoxelVAEConfig
from cloud_robotics_sim.meshy_t2.pipeline import MeshyT2Pipeline


def _tiny_pipeline() -> MeshyT2Pipeline:
    torch.manual_seed(0)
    mcfg, vcfg = MeshVAEConfig.tiny(), VoxelVAEConfig.tiny()
    fcfg, gcfg = VoxelFlowConfig.tiny(), MeshFlowConfig.tiny()
    fcfg.latent_channels, fcfg.latent_res = vcfg.latent_channels, vcfg.latent_res
    gcfg.latent_channels = mcfg.latent_channels
    gcfg.voxel_channels, gcfg.voxel_res = vcfg.latent_channels, vcfg.latent_res
    encoder = TinyViTImageEncoder(out_dim=fcfg.image_dim, image_size=64, patch_size=16)
    gcfg.image_dim = fcfg.image_dim
    return MeshyT2Pipeline(
        MeshVAE(mcfg),
        VoxelVAE(vcfg),
        VoxelFlow(fcfg),
        MeshFlow(gcfg),
        encoder,
        "cpu",
    )


def test_generate_end_to_end_budget():
    """Check generate end to end budget."""
    pipeline = _tiny_pipeline()
    image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mesh = pipeline.generate(image, num_vertices=48, steps=3, guidance=2.0)
    assert len(mesh.vertices) > 0
    assert len(mesh.vertices) <= 48  # pads are discarded
    assert len(mesh.faces) > 0
    assert np.isfinite(mesh.vertices).all()


def test_generate_face_budget_conversion():
    """Check generate face budget conversion."""
    assert MeshyT2Pipeline.vertices_for_faces(4000) == 2002


def test_checkpoint_roundtrip(tmp_path):
    """Check checkpoint roundtrip."""
    pipeline = _tiny_pipeline()
    pipeline.save(tmp_path)
    torch.manual_seed(1)
    mcfg, vcfg = MeshVAEConfig.tiny(), VoxelVAEConfig.tiny()
    fcfg, gcfg = VoxelFlowConfig.tiny(), MeshFlowConfig.tiny()
    fcfg.latent_channels, fcfg.latent_res = vcfg.latent_channels, vcfg.latent_res
    gcfg.latent_channels = mcfg.latent_channels
    gcfg.voxel_channels, gcfg.voxel_res = vcfg.latent_channels, vcfg.latent_res
    gcfg.image_dim = fcfg.image_dim
    encoder = TinyViTImageEncoder(out_dim=fcfg.image_dim, image_size=64, patch_size=16)
    loaded = MeshyT2Pipeline.from_checkpoints(
        tmp_path,
        MeshVAE(mcfg),
        VoxelVAE(vcfg),
        VoxelFlow(fcfg),
        MeshFlow(gcfg),
        encoder,
        "cpu",
    )
    for a, b in zip(
        pipeline.mesh_flow.state_dict().values(), loaded.mesh_flow.state_dict().values()
    ):
        torch.testing.assert_close(a, b)


def test_retopologize_uses_scaffold():
    """Check retopologize uses scaffold."""
    import trimesh

    pipeline = _tiny_pipeline()
    dense = trimesh.creation.icosphere(subdivisions=2)
    mesh = pipeline.retopologize(dense, num_vertices=32, steps=3, guidance=1.0)
    assert len(mesh.vertices) > 0
    assert len(mesh.vertices) <= 32
