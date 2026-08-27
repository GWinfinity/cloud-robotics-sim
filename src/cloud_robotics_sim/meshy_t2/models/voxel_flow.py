"""Stage I: image-conditioned voxel flow (Sec. 2.2).

A time-modulated transformer velocity field over the flattened latent
grid of the frozen Voxel VAE. The standardized posterior mean is split
into ``latent_res^3`` spatial tokens (paper: 16^3 = 4096 tokens of 8
channels); each block combines 3D-positional self-attention, cross-
attention to DINOv3 image features, AdaLN timestep modulation and a
SwiGLU feed-forward network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional

from .common import AdaLNBlock, CrossAttention, timestep_embedding


@dataclass
class VoxelFlowConfig:
    """Configuration of the Stage-I voxel flow (paper defaults)."""

    hidden: int = 1536
    depth: int = 28
    heads: int = 12
    latent_channels: int = 8
    latent_res: int = 16
    image_dim: int = 256  # projected image feature width
    cond_dim: int = 512  # AdaLN time-embedding width

    @classmethod
    def tiny(cls) -> VoxelFlowConfig:
        """Small configuration for tests and CPU smoke runs."""
        return cls(
            hidden=64,
            heads=4,
            depth=2,
            latent_channels=4,
            latent_res=8,
            image_dim=32,
            cond_dim=64,
        )


class VoxelFlow(nn.Module):
    """Transformer velocity field over the Voxel VAE latent grid."""

    def __init__(self, cfg: VoxelFlowConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VoxelFlowConfig()
        n = self.cfg.latent_res**3
        self.in_proj = nn.Linear(self.cfg.latent_channels, self.cfg.hidden)
        self.img_proj = nn.Linear(self.cfg.image_dim, self.cfg.hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
        )
        # One set of 6 modulation weights per block would be produced by
        # AdaLNBlock internally; time embedding width is cond_dim.
        self.blocks = nn.ModuleList(
            AdaLNBlock(self.cfg.hidden, self.cfg.heads, self.cfg.cond_dim, rope=True)
            for _ in range(self.cfg.depth)
        )
        self.cross = nn.ModuleList(
            CrossAttention(self.cfg.hidden, self.cfg.heads)
            for _ in range(self.cfg.depth)
        )
        self.final_norm = nn.LayerNorm(self.cfg.hidden, elementwise_affine=False)
        self.final_ada = nn.Linear(self.cfg.cond_dim, 2 * self.cfg.hidden)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.out_proj = nn.Linear(self.cfg.hidden, self.cfg.latent_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Fixed 3D grid coordinates for RoPE.
        r = self.cfg.latent_res
        axes = torch.arange(r, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1)
        self.register_buffer("grid_coords", grid.reshape(n, 3), persistent=False)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the flow velocity.

        Args:
            x_t: ``(B, latent_res^3, latent_channels)`` noisy latent tokens.
            t: ``(B,)`` timesteps.
            image_tokens: ``(B, M, image_dim)`` image features, or ``None``
                for the unconditional branch of classifier-free guidance.

        Returns:
            ``(B, latent_res^3, latent_channels)`` velocity prediction.
        """
        cond = self.time_mlp(timestep_embedding(t, self.cfg.cond_dim))
        x = self.in_proj(x_t)
        img = self.img_proj(image_tokens) if image_tokens is not None else None
        for block, cross in zip(self.blocks, self.cross, strict=True):
            x = block(x, cond, coords=self.grid_coords)
            if img is not None:
                x = x + cross(functional.layer_norm(x, x.shape[-1:]), img)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(h)
