"""End-to-end Meshy T2 inference pipeline.

``MeshyT2Pipeline.generate`` reproduces the coarse-to-fine cascade:

1. The reference image is encoded into patch tokens.
2. Stage I samples a Voxel VAE latent grid with the voxel flow and
   decodes/thresholds it into a ``64^3`` occupancy scaffold.
3. Stage II samples ``N`` per-vertex latent slots (existence channel
   included) conditioned on the image, the scaffold and the requested
   slot count; pads (existence <= 0) are discarded.
4. The frozen Mesh VAE decoder turns the surviving latents into
   positions, edges and faces, assembled into a ``trimesh.Trimesh``.

``retopologize`` replaces the generated scaffold with the ground-truth
voxelization of a dense input mesh (the retopology task of Sec. 3.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from .flow import cfg_velocity, euler_sample
from .geometry.assembly import assemble_mesh
from .geometry.mesh_utils import normalize_vertices, voxelize
from .models.mesh_flow import (
    MeshFlow,
    image_token_coords,
    voxel_token_coords,
)
from .sobol_ot import sobol_points


class MeshyT2Pipeline:
    """Coarse-to-fine image-to-mesh generation pipeline."""

    def __init__(
        self,
        mesh_vae,
        voxel_vae,
        voxel_flow,
        mesh_flow: MeshFlow,
        image_encoder,
        device: torch.device | str = "cpu",
    ) -> None:
        self.mesh_vae = mesh_vae.to(device).eval()
        self.voxel_vae = voxel_vae.to(device).eval()
        self.voxel_flow = voxel_flow.to(device).eval()
        self.mesh_flow = mesh_flow.to(device).eval()
        self.image_encoder = image_encoder.to(device).eval()
        self.device = device

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(
        self, image: Image.Image | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Encode an RGB image into patch tokens ``(1, M, image_dim)``."""
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(arr).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        return self.image_encoder(image.to(self.device))

    # ------------------------------------------------------------------
    # Stage I
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_scaffold(
        self,
        image_tokens: torch.Tensor | None,
        steps: int = 25,
        guidance: float = 3.0,
    ) -> torch.Tensor:
        """Sample a binary occupancy scaffold with the voxel flow.

        Args:
            image_tokens: ``(1, M, image_dim)`` image conditioning.
            steps: Euler integration steps.
            guidance: Classifier-free guidance scale.

        Returns:
            ``(1, 1, res, res, res)`` binary occupancy grid.
        """
        cfg = self.voxel_flow.cfg
        shape = (1, cfg.latent_res**3, cfg.latent_channels)
        x1 = torch.randn(shape, device=self.device)
        call = cfg_velocity(
            lambda x, t, **kw: self.voxel_flow(x, t, kw.get("image_tokens")),
            {"image_tokens": image_tokens},
            {"image_tokens": None},
            guidance,
        )
        z = euler_sample(call, x1, steps)
        z = z.reshape(
            1, cfg.latent_res, cfg.latent_res, cfg.latent_res, cfg.latent_channels
        )
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        return self.voxel_vae.decode_occupancy(z)

    # ------------------------------------------------------------------
    # Stage II
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_latents(
        self,
        num_slots: int,
        image_tokens: torch.Tensor | None,
        occupancy: torch.Tensor | None,
        steps: int = 25,
        guidance: float = 3.0,
    ) -> torch.Tensor:
        """Sample per-vertex latent tokens with the mesh flow.

        Args:
            num_slots: Requested latent slot count ``N``.
            image_tokens: Optional image conditioning.
            occupancy: Optional ``(1, 1, res, res, res)`` scaffold.
            steps: Euler integration steps.
            guidance: Classifier-free guidance scale.

        Returns:
            ``(K, C)`` latents of the slots with positive existence.
        """
        cfg = self.mesh_flow.cfg
        voxel_tokens, voxel_coords = None, None
        if occupancy is not None:
            z_v = self.voxel_vae.encode_deterministic(occupancy.to(self.device))
            voxel_tokens = z_v.flatten(2).transpose(1, 2)  # (1, K, Cv)
            voxel_coords = voxel_token_coords(z_v.shape[-1]).to(self.device)
        image_coords = (
            image_token_coords(image_tokens.shape[1]).to(self.device)
            if image_tokens is not None
            else None
        )
        latent_coords = torch.tensor(
            sobol_points(num_slots), dtype=torch.float32, device=self.device
        )
        count = torch.tensor([float(num_slots)], device=self.device)
        x1 = torch.randn(1, num_slots, cfg.flow_channels, device=self.device)

        call = cfg_velocity(
            self._mesh_flow_call,
            {
                "latent_coords": latent_coords,
                "image_tokens": image_tokens,
                "image_coords": image_coords,
                "voxel_tokens": voxel_tokens,
                "voxel_coords": voxel_coords,
                "count": count,
            },
            {
                "latent_coords": latent_coords,
                "image_tokens": None,
                "image_coords": None,
                "voxel_tokens": None,
                "voxel_coords": None,
                "count": None,
            },
            guidance,
        )
        x0 = euler_sample(call, x1, steps)[0]
        keep = x0[:, -1] > 0
        if int(keep.sum()) == 0:
            keep[int(torch.argmax(x0[:, -1]))] = True
        return x0[keep, :-1]

    def _mesh_flow_call(self, x: torch.Tensor, t: torch.Tensor, **kw) -> torch.Tensor:
        return self.mesh_flow(x, t, **kw)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> trimesh.Trimesh:
        """Decode a latent set into an assembled ``trimesh.Trimesh``."""
        cfg = self.mesh_vae.cfg
        positions, edge_emb, face_emb = self.mesh_vae.decode(z)
        verts, _edges, faces = assemble_mesh(
            positions,
            edge_emb,
            face_emb,
            self.mesh_vae.decoder.null_prev,
            self.mesh_vae.decoder.null_next,
            cfg.edge_dim,
            cfg.face_dim,
        )
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def vertices_for_faces(num_faces: int) -> int:
        """Vertex count matching a face budget via Euler ``F = 2V - 4``."""
        return num_faces // 2 + 2

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        num_vertices: int | None = None,
        num_faces: int | None = None,
        steps: int = 25,
        guidance: float = 3.0,
    ) -> trimesh.Trimesh:
        """Generate a mesh from a reference image.

        Args:
            image: Reference image (PIL / array / tensor, RGB).
            num_vertices: Requested vertex budget (slot count ``N``);
                the realized count falls in ``[N / (1 + p), N]``.
            num_faces: Alternative face budget (converted with Euler).
            steps: Euler steps per flow stage.
            guidance: Classifier-free guidance scale.

        Returns:
            Generated mesh.
        """
        if num_vertices is None:
            if num_faces is None:
                num_faces = 4000
            num_vertices = self.vertices_for_faces(num_faces)
        image_tokens = self.encode_image(image)
        occupancy = self.generate_scaffold(image_tokens, steps=steps, guidance=guidance)
        z = self.generate_latents(
            num_vertices, image_tokens, occupancy, steps=steps, guidance=guidance
        )
        return self.decode_latents(z)

    @torch.no_grad()
    def retopologize(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image | np.ndarray | torch.Tensor | None = None,
        num_vertices: int | None = None,
        steps: int = 25,
        guidance: float = 3.0,
    ) -> trimesh.Trimesh:
        """Retopologize a dense mesh using its own voxel scaffold.

        Args:
            mesh: Dense input mesh (any scale; normalized internally).
            image: Optional reference image for appearance conditioning.
            num_vertices: Requested vertex budget; defaults to ``V`` of
                the input mesh (capped by ``max_vertices``).
            steps: Euler steps for the mesh flow.
            guidance: Classifier-free guidance scale.

        Returns:
            Compact retopologized mesh in the normalized frame.
        """
        vertices, _, _ = normalize_vertices(np.asarray(mesh.vertices, dtype=np.float64))
        norm_mesh = trimesh.Trimesh(
            vertices=vertices, faces=np.asarray(mesh.faces), process=False
        )
        occupancy = torch.tensor(
            voxelize(norm_mesh, res=self.voxel_vae.cfg.res), dtype=torch.float32
        )[None, None]
        if num_vertices is None:
            num_vertices = min(len(mesh.vertices), self.mesh_flow.cfg.max_vertices)
        image_tokens = self.encode_image(image) if image is not None else None
        z = self.generate_latents(
            num_vertices, image_tokens, occupancy, steps=steps, guidance=guidance
        )
        return self.decode_latents(z)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, root: str | Path) -> None:
        """Save all component weights under ``root``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.mesh_vae.state_dict(), root / "mesh_vae.pt")
        torch.save(self.voxel_vae.state_dict(), root / "voxel_vae.pt")
        torch.save(self.voxel_flow.state_dict(), root / "voxel_flow.pt")
        torch.save(self.mesh_flow.state_dict(), root / "mesh_flow.pt")
        torch.save(self.image_encoder.state_dict(), root / "image_encoder.pt")

    @classmethod
    def from_checkpoints(
        cls,
        root: str | Path,
        mesh_vae,
        voxel_vae,
        voxel_flow,
        mesh_flow,
        image_encoder,
        device: torch.device | str = "cpu",
    ) -> "MeshyT2Pipeline":
        """Load component weights from ``root`` into fresh modules."""
        root = Path(root)
        modules = {
            "mesh_vae": mesh_vae,
            "voxel_vae": voxel_vae,
            "voxel_flow": voxel_flow,
            "mesh_flow": mesh_flow,
            "image_encoder": image_encoder,
        }
        for name, module in modules.items():
            path = root / f"{name}.pt"
            if path.exists():
                module.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
        return cls(mesh_vae, voxel_vae, voxel_flow, mesh_flow, image_encoder, device)
