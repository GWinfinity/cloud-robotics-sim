"""Stage II: image-and-voxel-conditioned mesh flow (Sec. 2.3).

A single-stream DiT over concatenated latent, image and voxel tokens
with a unified 3D RoPE. Latent tokens are ``C + 1`` channels: the mesh
VAE latent plus an existence channel (``+1`` real, ``-1`` pad) enabling
range-based vertex-count control. The requested slot count enters as a
Fourier embedding of ``N / N_max`` added to the time embedding. Latent
RoPE positions come from a deterministic Sobol point set, matched to
ground-truth vertices by optimal transport during training (Sobol OT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..sobol_ot import morton_assign, ot_assign, sobol_points
from .common import AdaLNBlock, FinalAdaLN, timestep_embedding


@dataclass
class MeshFlowConfig:
    """Configuration of the Stage-II mesh flow."""

    hidden: int = 1024
    depth: int = 24
    heads: int = 16
    latent_channels: int = 32  # C (existence channel added -> C + 1)
    max_vertices: int = 8192  # N_max for the count condition
    pad_ratio: float = 0.2  # p: pad tokens appended during training
    cond_dim: int = 512
    image_dim: int = 256
    voxel_channels: int = 8
    voxel_res: int = 16
    count_freqs: int = 16
    # Condition-dropout probabilities (Sec. 2.3).
    drop_image: float = 0.2
    drop_voxel: float = 0.3
    drop_count: float = 0.2
    drop_all: float = 0.05

    @property
    def flow_channels(self) -> int:
        """Flow state channels ``C + 1`` (latent + existence)."""
        return self.latent_channels + 1

    @classmethod
    def tiny(cls) -> MeshFlowConfig:
        """Small configuration for tests and CPU smoke runs."""
        return cls(
            hidden=64,
            heads=4,
            depth=2,
            latent_channels=8,
            max_vertices=256,
            cond_dim=64,
            image_dim=32,
            voxel_channels=4,
            voxel_res=8,
            count_freqs=4,
        )


class MeshFlow(nn.Module):
    """Single-stream DiT velocity field over per-vertex latent tokens."""

    def __init__(self, cfg: MeshFlowConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MeshFlowConfig()
        h = self.cfg.hidden
        self.latent_in = nn.Linear(self.cfg.flow_channels, h)
        self.image_in = nn.Linear(self.cfg.image_dim, h)
        self.voxel_in = nn.Linear(self.cfg.voxel_channels, h)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
        )
        self.count_mlp = nn.Sequential(
            nn.Linear(2 * self.cfg.count_freqs, self.cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
        )
        self.blocks = nn.ModuleList(
            AdaLNBlock(h, self.cfg.heads, self.cfg.cond_dim, rope=True)
            for _ in range(self.cfg.depth)
        )
        self.final = FinalAdaLN(h, self.cfg.cond_dim, self.cfg.flow_channels)

    def count_embedding(
        self, count: torch.Tensor | None, batch: int, device
    ) -> torch.Tensor:
        """Fourier embedding of ``N / N_max`` projected to the cond width.

        Args:
            count: ``(B,)`` requested slot counts, or ``None`` for the
                count-dropped (zeroed) branch.
            batch: Batch size.
            device: Target device.

        Returns:
            ``(B, cond_dim)`` additive condition embedding.
        """
        if count is None:
            return torch.zeros(batch, self.cfg.cond_dim, device=device)
        x = (count.float() / self.cfg.max_vertices).clamp(0, 1)
        freqs = (2.0 ** torch.arange(self.cfg.count_freqs, device=device)) * torch.pi
        ang = x.unsqueeze(-1) * freqs
        return self.count_mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_coords: torch.Tensor,
        image_tokens: torch.Tensor | None = None,
        image_coords: torch.Tensor | None = None,
        voxel_tokens: torch.Tensor | None = None,
        voxel_coords: torch.Tensor | None = None,
        count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow velocity for the latent tokens.

        Args:
            x_t: ``(B, N, C + 1)`` noisy latent tokens.
            t: ``(B,)`` timesteps.
            latent_coords: ``(N, 3)`` RoPE coordinates of latent tokens.
            image_tokens: Optional ``(B, M, image_dim)`` image features;
                omitted entirely when the condition is dropped.
            image_coords: ``(M, 3)`` image-token coordinates.
            voxel_tokens: Optional ``(B, K, voxel_channels)`` scaffold
                features; omitted entirely when dropped.
            voxel_coords: ``(K, 3)`` voxel-token coordinates.
            count: Optional ``(B,)`` requested slot counts.

        Returns:
            ``(B, N, C + 1)`` velocity prediction.
        """
        b = x_t.shape[0]
        cond = self.time_mlp(timestep_embedding(t, self.cfg.cond_dim))
        cond = cond + self.count_embedding(count, b, x_t.device)
        tokens = [self.latent_in(x_t)]
        coords = [latent_coords]
        if image_tokens is not None and image_coords is not None:
            tokens.append(self.image_in(image_tokens))
            coords.append(image_coords)
        if voxel_tokens is not None and voxel_coords is not None:
            tokens.append(self.voxel_in(voxel_tokens))
            coords.append(voxel_coords)
        x = torch.cat(tokens, dim=1)
        rope_coords = torch.cat(coords, dim=0).to(x.device)
        for block in self.blocks:
            x = block(x, cond, coords=rope_coords)
        n = x_t.shape[1]
        return self.final(x[:, :n], cond)


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------


def image_token_coords(num_tokens: int) -> torch.Tensor:
    """3D RoPE coordinates for image patch tokens.

    Patches are laid out on a square grid placed on a separate slice of
    the coordinate volume (``z = -0.5``) to avoid overlap with the 3D
    tokens (Sec. 2.3).
    """
    side = int(round(num_tokens**0.5))
    if side * side != num_tokens:
        raise ValueError(f"image token count {num_tokens} is not a square")
    axes = torch.arange(side, dtype=torch.float32) / max(side - 1, 1)
    grid = torch.stack(torch.meshgrid(axes, axes, indexing="ij"), dim=-1)
    coords = torch.cat(
        [grid.reshape(-1, 2), torch.full((side * side, 1), -0.5)], dim=-1
    )
    return coords


def voxel_token_coords(res: int) -> torch.Tensor:
    """Normalized ``[0, 1]^3`` coordinates of a ``res^3`` token grid."""
    axes = torch.arange(res, dtype=torch.float32) / max(res - 1, 1)
    grid = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def assign_latent_coords(
    vertices: np.ndarray,
    num_slots: int,
    method: str = "ot",
    seed: int = 0,
) -> np.ndarray:
    """Assign Sobol RoPE coordinates to latent slots during training.

    Real vertices are matched to Sobol candidates by optimal transport
    (or Morton-order pairing when ``method="morton"``); pad slots take the
    leftover candidates.

    Args:
        vertices: ``(V, 3)`` ground-truth vertex coordinates in
            ``[0, 1]^3`` with ``V <= num_slots``.
        num_slots: Total number of latent slots (real + pads).
        method: ``"ot"`` or ``"morton"``.
        seed: Sobol scramble seed.

    Returns:
        ``(num_slots, 3)`` float32 coordinates: the first ``V`` rows for
        real vertices (in vertex order), the rest for pads.
    """
    v = len(vertices)
    candidates = sobol_points(num_slots, dim=3, seed=seed)
    if v > 0:
        if method == "ot":
            assign = ot_assign(vertices, candidates)
        elif method == "morton":
            assign = morton_assign(vertices, candidates)
        else:
            raise ValueError(f"unknown assignment method: {method}")
    else:
        assign = np.zeros(0, dtype=np.int64)
    used = np.zeros(num_slots, dtype=bool)
    used[assign] = True
    leftover = candidates[~used]
    coords = np.concatenate([candidates[assign], leftover], axis=0)
    return coords.astype(np.float32)
