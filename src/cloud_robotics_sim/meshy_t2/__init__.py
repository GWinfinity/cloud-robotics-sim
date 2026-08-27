"""Python reproduction of Meshy T2 (arXiv:2607.28675).

Fast native mesh generation with flow matching:

* :class:`~cloud_robotics_sim.meshy_t2.models.mesh_vae.MeshVAE` —
  nearly lossless vertex-set mesh VAE (Sec. 2.1).
* :class:`~cloud_robotics_sim.meshy_t2.models.voxel_vae.VoxelVAE` —
  dense 3D conv VAE over occupancy scaffolds (Sec. 2.2).
* :class:`~cloud_robotics_sim.meshy_t2.models.voxel_flow.VoxelFlow` —
  Stage I image-conditioned scaffold flow (Sec. 2.2).
* :class:`~cloud_robotics_sim.meshy_t2.models.mesh_flow.MeshFlow` —
  Stage II image/voxel/count-conditioned latent flow (Sec. 2.3).
* :class:`~cloud_robotics_sim.meshy_t2.pipeline.MeshyT2Pipeline` —
  coarse-to-fine image-to-mesh cascade.

The official code/weights were not released at the time of writing; this
module implements the architecture and training procedure from the
paper. Model defaults match the paper; ``*.tiny()`` configs run on CPU.
"""

from .models.mesh_flow import MeshFlow, MeshFlowConfig
from .models.mesh_vae import MeshVAE, MeshVAEConfig
from .models.voxel_flow import VoxelFlow, VoxelFlowConfig
from .models.voxel_vae import VoxelVAE, VoxelVAEConfig
from .pipeline import MeshyT2Pipeline

__all__ = [
    "MeshFlow",
    "MeshFlowConfig",
    "MeshVAE",
    "MeshVAEConfig",
    "MeshyT2Pipeline",
    "VoxelFlow",
    "VoxelFlowConfig",
    "VoxelVAE",
    "VoxelVAEConfig",
]
