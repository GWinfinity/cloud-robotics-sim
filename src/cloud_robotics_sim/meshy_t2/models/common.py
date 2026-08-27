"""Shared building blocks for the Meshy T2 reproduction.

Implements Fourier features, 3D rotary position embeddings (RoPE),
attention layers (self / cross / graph-restricted), SwiGLU feed-forward
networks and AdaLN timestep modulation following the paper's descriptions
(Sec. 2.1--2.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def fourier_features(x: torch.Tensor, num_freqs: int = 8) -> torch.Tensor:
    """Map coordinates to sin/cos Fourier features (Tancik et al.).

    Args:
        x: ``(..., D)`` coordinates, expected roughly in ``[0, 1]``.
        num_freqs: Number of frequency bands (powers of two).

    Returns:
        ``(..., D * 2 * num_freqs)`` features including raw-scale bands.
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    angles = x.unsqueeze(-1) * freqs  # (..., D, F)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    """Sinusoidal timestep embedding as in Vaswani et al. / DiT.

    Args:
        t: ``(B,)`` timesteps.
        dim: Embedding dimension (must be even).
        max_period: Lowest frequency period.

    Returns:
        ``(B, dim)`` embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


# ---------------------------------------------------------------------------
# 3D RoPE
# ---------------------------------------------------------------------------


def _rope_axis(
    x: torch.Tensor, pos: torch.Tensor, base: float = 10_000.0
) -> torch.Tensor:
    """Apply rotary embedding to one axis slice.

    Args:
        x: ``(B, H, N, 2 * F)`` slice of the head dimension for one axis.
        pos: ``(N,)`` position of each token along this axis.
        base: RoPE frequency base.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    d = x.shape[-1]
    freqs = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = pos.float().unsqueeze(-1) * freqs  # (N, F)
    cos = angles.cos()[None, None]  # (1, 1, N, F)
    sin = angles.sin()[None, None]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


def apply_rope_3d(
    x: torch.Tensor, coords: torch.Tensor, base: float = 100.0
) -> torch.Tensor:
    """Apply 3D RoPE to attention projections.

    The head dimension is split into three contiguous chunks, one per
    spatial axis, and each chunk is rotated with the token's coordinate
    along that axis.

    Args:
        x: ``(B, H, N, D)`` query or key tensor.
        coords: ``(N, 3)`` per-token coordinates (any scale).
        base: RoPE frequency base.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    n = x.shape[2]
    if coords.shape[0] != n:
        raise ValueError(f"coords has {coords.shape[0]} tokens, expected {n}")
    d = x.shape[-1]
    chunk = d - 2 * (d // 3)
    parts = []
    start = 0
    for axis in range(3):
        width = d // 3 if axis < 2 else chunk
        width -= width % 2
        if width > 0:
            parts.append(
                _rope_axis(x[..., start : start + width], coords[:, axis], base)
            )
        start += width
    if start < d:
        parts.append(x[..., start:])
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional 3D RoPE and attention mask."""

    def __init__(self, dim: int, heads: int, rope: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention over ``x`` of shape ``(B, N, C)``.

        Args:
            x: Input tokens.
            coords: Optional ``(N, 3)`` coordinates enabling 3D RoPE.
            attn_mask: Optional boolean ``(N, N)`` mask; ``True`` = attend.

        Returns:
            ``(B, N, C)`` output tokens.
        """
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.rope:
            if coords is None:
                raise ValueError("RoPE attention requires coords")
            q = apply_rope_3d(q, coords)
            k = apply_rope_3d(k, coords)
        mask = None
        if attn_mask is not None:
            mask = attn_mask[None, None].expand(b, self.heads, n, n)
        out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class GraphAttention(nn.Module):
    """Message passing restricted to a graph edge set (masked attention).

    Used by the Mesh VAE encoder to restrict attention to ground-truth mesh
    edges (Sec. 2.1). Self-loops are always included.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, heads, rope=False)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend only over graph neighbors.

        Args:
            x: ``(B, N, C)`` vertex tokens.
            adjacency: Boolean ``(N, N)`` adjacency (self-loops added here).
            coords: Unused, kept for interface symmetry.

        Returns:
            ``(B, N, C)`` updated tokens.
        """
        del coords
        n = x.shape[1]
        mask = adjacency | torch.eye(n, dtype=torch.bool, device=x.device)
        return self.attn(x, attn_mask=mask)


class CrossAttention(nn.Module):
    """Cross-attention from query tokens into a context set."""

    def __init__(
        self, dim: int, heads: int, context_dim: int | None = None, rope: bool = False
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = rope
        context_dim = context_dim or dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        q_coords: torch.Tensor | None = None,
        kv_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from ``x`` (B, N, C) into ``context`` (B, M, C)."""
        b, n, c = x.shape
        m = context.shape[1]
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(b, m, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.heads, self.head_dim).transpose(1, 2)
        if self.rope:
            if q_coords is None or kv_coords is None:
                raise ValueError("RoPE cross-attention requires both coordinate sets")
            q = apply_rope_3d(q, q_coords)
            k = apply_rope_3d(k, kv_coords)
        out = functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020)."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or int(dim * 8 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU to ``x``."""
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(functional.silu(a) * b)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block with optional RoPE and graph masking."""

    def __init__(self, dim: int, heads: int, rope: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.attn = SelfAttention(dim, heads, rope=rope)
        self.ffn = SwiGLU(dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention + FFN residual updates."""
        shape = (self.dim,)
        x = x + self.attn(functional.layer_norm(x, shape), coords, attn_mask)
        x = x + self.ffn(functional.layer_norm(x, shape))
        return x


class AdaLNBlock(nn.Module):
    """DiT-style block: self-attention + SwiGLU with AdaLN modulation.

    The timestep (plus any additive condition embedding) produces per-block
    shift/scale/gate parameters (Peebles & Xie 2022).
    """

    def __init__(self, dim: int, heads: int, cond_dim: int, rope: bool = True) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, heads, rope=rope)
        self.ffn = SwiGLU(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply modulated attention + FFN.

        Args:
            x: ``(B, N, C)`` tokens.
            cond: ``(B, cond_dim)`` modulation embedding.
            coords: Optional ``(N, 3)`` RoPE coordinates.

        Returns:
            Updated tokens.
        """
        s1, c1, g1, s2, c2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + c1.unsqueeze(1)) + s1.unsqueeze(1)
        x = x + g1.unsqueeze(1) * self.attn(h, coords)
        h = self.norm2(x) * (1 + c2.unsqueeze(1)) + s2.unsqueeze(1)
        return x + g2.unsqueeze(1) * self.ffn(h)


class FinalAdaLN(nn.Module):
    """Final adaLN-modulated projection head for flow velocity outputs."""

    def __init__(self, dim: int, cond_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Project tokens to output channels under adaLN modulation."""
        shift, scale = self.ada(cond).chunk(2, dim=-1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
