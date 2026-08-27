"""Dense 3D-convolutional Voxel VAE for occupancy scaffolds (Sec. 2.2).

Compresses a binary ``res^3`` occupancy grid (paper: 64^3) through two
stride-2 stages into a spatially factorized Gaussian posterior with
``latent_channels`` mean and log-variance channels at ``(res/4)^3``
(paper: 8 channels at 16^3). The mirrored decoder reconstructs occupancy
logits through two 3D pixel-shuffle upsampling stages. Trained with BCE
on the logits plus a small KL penalty (paper: 1e-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional


@dataclass
class VoxelVAEConfig:
    """Configuration of the Voxel VAE."""

    res: int = 64  # occupancy grid resolution
    latent_channels: int = 8
    base_channels: int = 64
    kl_weight: float = 1e-4

    @property
    def latent_res(self) -> int:
        """Latent grid resolution (``res / 4``)."""
        return self.res // 4

    @classmethod
    def tiny(cls) -> VoxelVAEConfig:
        """Small configuration for tests and CPU smoke runs."""
        return cls(res=32, latent_channels=4, base_channels=16)


class _ResBlock3d(nn.Module):
    """Residual block with GroupNorm, SiLU and zero-init second conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual update."""
        h = self.conv1(functional.silu(self.norm1(x)))
        h = self.conv2(functional.silu(self.norm2(h)))
        return x + h


class PixelShuffle3d(nn.Module):
    """3D analogue of pixel shuffle (depth-space rearrangement)."""

    def __init__(self, upscale: int = 2) -> None:
        super().__init__()
        self.upscale = upscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange ``(B, C * r^3, D, H, W)`` to ``(B, C, rD, rH, rW)``."""
        r = self.upscale
        b, c, d, h, w = x.shape
        if c % (r**3) != 0:
            raise ValueError("channels must be divisible by r^3")
        x = x.view(b, c // (r**3), r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, c // (r**3), d * r, h * r, w * r)


class VoxelVAE(nn.Module):
    """Dense 3D conv VAE over binary occupancy grids."""

    def __init__(self, cfg: VoxelVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VoxelVAEConfig()
        c = self.cfg.base_channels
        zc = self.cfg.latent_channels

        self.enc_in = nn.Conv3d(1, c, 3, padding=1)
        self.enc_res1 = nn.ModuleList([_ResBlock3d(c), _ResBlock3d(c)])
        self.enc_down1 = nn.Conv3d(c, c * 2, 3, stride=2, padding=1)
        self.enc_res2 = nn.ModuleList([_ResBlock3d(c * 2), _ResBlock3d(c * 2)])
        self.enc_down2 = nn.Conv3d(c * 2, c * 2, 3, stride=2, padding=1)
        self.enc_out = nn.Conv3d(c * 2, 2 * zc, 3, padding=1)

        self.dec_in = nn.Conv3d(zc, c * 2, 3, padding=1)
        self.dec_res1 = nn.ModuleList([_ResBlock3d(c * 2), _ResBlock3d(c * 2)])
        # Pixel-shuffle upsampling: channels / 8, resolution x2.
        self.dec_up1 = nn.Conv3d(c * 2, c * 2 * 8, 3, padding=1)
        self.dec_res2 = nn.ModuleList([_ResBlock3d(c * 2), _ResBlock3d(c * 2)])
        self.dec_up2 = nn.Conv3d(c * 2, c * 8, 3, padding=1)
        self.dec_res3 = nn.ModuleList([_ResBlock3d(c), _ResBlock3d(c)])
        self.dec_out = nn.Conv3d(c, 1, 3, padding=1)
        self.shuffle = PixelShuffle3d(2)

    def encode(self, occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an occupancy grid to posterior mean and log-variance.

        Args:
            occ: ``(B, 1, res, res, res)`` binary occupancy.

        Returns:
            ``(mu, logvar)`` each ``(B, C, res/4, res/4, res/4)``.
        """
        h = self.enc_in(occ)
        for blk in self.enc_res1:
            h = blk(h)
        h = self.enc_down1(h)
        for blk in self.enc_res2:
            h = blk(h)
        h = self.enc_down2(h)
        mu, logvar = self.enc_out(h).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent grid to occupancy logits.

        Args:
            z: ``(B, C, res/4, res/4, res/4)`` latent grid.

        Returns:
            ``(B, 1, res, res, res)`` occupancy logits.
        """
        h = self.dec_in(z)
        for blk in self.dec_res1:
            h = blk(h)
        h = self.shuffle(self.dec_up1(h))
        for blk in self.dec_res2:
            h = blk(h)
        h = self.shuffle(self.dec_up2(h))
        for blk in self.dec_res3:
            h = blk(h)
        return self.dec_out(h)

    def forward(
        self, occ: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full autoencoding pass.

        Returns:
            ``(logits, mu, logvar)``.
        """
        mu, logvar = self.encode(occ)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar

    def loss(self, occ: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """BCE reconstruction loss plus KL penalty.

        Args:
            occ: ``(B, 1, res, res, res)`` binary occupancy.

        Returns:
            ``(total, parts)`` scalar loss and logging dict.
        """
        logits, mu, logvar = self.forward(occ)
        bce = functional.binary_cross_entropy_with_logits(logits, occ)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        total = bce + self.cfg.kl_weight * kl
        return total, {"bce": float(bce), "kl": float(kl)}

    @torch.no_grad()
    def encode_deterministic(self, occ: torch.Tensor) -> torch.Tensor:
        """Posterior-mean encoding used by both generation stages."""
        mu, _ = self.encode(occ)
        return mu

    @torch.no_grad()
    def decode_occupancy(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Decode a latent grid and threshold into binary occupancy."""
        return (self.decode(z).sigmoid() > threshold).float()
