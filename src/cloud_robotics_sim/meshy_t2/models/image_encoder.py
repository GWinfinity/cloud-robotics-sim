"""Image encoders for flow conditioning.

The paper conditions both flow stages on a frozen DINOv3 backbone at
768x768 (patch 16, 2304 tokens). This reproduction defines a common
interface with:

* :class:`DINOv3ImageEncoder` — wraps a Hugging Face DINOv3 checkpoint
  when ``transformers`` and network access are available.
* :class:`TinyViTImageEncoder` — a self-contained ViT fallback so the
  full pipeline trains and runs offline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import TransformerBlock


class ImageEncoder(nn.Module):
    """Interface: images ``(B, 3, H, W)`` -> tokens ``(B, M, dim)``."""

    out_dim: int
    patch_size: int = 16

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch tokens."""
        raise NotImplementedError

    def token_grid_size(self, image_size: int) -> int:
        """Number of patches along one axis for a square input."""
        return image_size // self.patch_size


class TinyViTImageEncoder(nn.Module):
    """Small self-contained ViT fallback (offline-friendly).

    Patch embedding via strided convolution followed by pre-norm
    transformer blocks with 2D sine-cosine position embedding folded
    into a learned table.
    """

    patch_size = 16

    def __init__(
        self,
        out_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        image_size: int = 768,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.patch = nn.Conv2d(3, out_dim, patch_size, stride=patch_size)
        grid = image_size // patch_size
        self.pos = nn.Parameter(torch.randn(1, grid * grid, out_dim) * 0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(out_dim, heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` images to ``(B, (H/p)^2, dim)`` tokens."""
        x = self.patch(images).flatten(2).transpose(1, 2)
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DINOv3ImageEncoder(ImageEncoder):
    """Frozen DINOv3 backbone loaded through Hugging Face transformers.

    Requires the optional ``transformers`` dependency and network access
    on first use. Falls back is intentionally *not* automatic: pass
    ``TinyViTImageEncoder`` explicitly for offline work.
    """

    patch_size = 16

    def __init__(
        self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv3ImageEncoder requires `transformers`; "
                "install it or use TinyViTImageEncoder for offline runs."
            ) from exc
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = int(self.model.config.hidden_size)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to patch tokens with the frozen backbone."""
        with torch.no_grad():
            out = self.model(pixel_values=images)
        tokens = out.last_hidden_state
        # Drop register / CLS tokens, keep patch tokens.
        num_patches = (images.shape[-1] // self.patch_size) ** 2
        return tokens[:, -num_patches:]
