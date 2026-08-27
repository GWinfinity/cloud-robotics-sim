"""Vertex-set Mesh VAE (Meshy T2 Sec. 2.1).

One continuous latent token per vertex. The encoder combines a sparse
voxel context (local PointNet over surface samples) with per-vertex
queries refined by one cross-attention into the context and interleaved
graph-attention (over ground-truth edges) and self-attention layers, all
with 3D RoPE. The decoder is a pure set decoder (no positional encoding):
a shared self-attention trunk followed by a vertex branch (position
regression) and a topology branch (edge and face embeddings).

Edges are scored by the Minkowski-style spacetime logit

    A_ij = ||e_i^time - e_j^time||^2 - ||e_i^space - e_j^space||^2

and faces by per-vertex halfedge successor scores

    Phi_i[p, q] = 1^T (f_i^root ⊙ f_p^prev ⊙ f_q^next)

normalized into soft permutations by log-space Sinkhorn iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import (
    CrossAttention,
    SelfAttention,
    SwiGLU,
    TransformerBlock,
    fourier_features,
)


@dataclass
class MeshVAEConfig:
    """Configuration of the vertex-set Mesh VAE.

    Defaults follow the paper's implementation details (Sec. 2.1).
    """

    width: int = 1024
    heads: int = 16
    latent_channels: int = 32
    edge_dim: int = 16  # d_e per half (space / time)
    face_dim: int = 16  # d_f per role (root / prev / next)
    enc_layers: int = 6  # pairs of graph-attention + self-attention
    dec_trunk_layers: int = 12
    dec_branch_layers: int = 4
    sinkhorn_iters: int = 10
    point_voxel_res: int = 256  # sparse voxel context resolution
    fourier_freqs: int = 8
    kl_weight: float = 0.0  # paper's objective has no KL term

    @classmethod
    def tiny(cls) -> MeshVAEConfig:
        """Small configuration for tests and CPU smoke runs."""
        return cls(
            width=64,
            heads=4,
            latent_channels=8,
            edge_dim=8,
            face_dim=8,
            enc_layers=2,
            dec_trunk_layers=2,
            dec_branch_layers=1,
            sinkhorn_iters=5,
            point_voxel_res=32,
            fourier_freqs=4,
        )


class VoxelContextEncoder(nn.Module):
    """Local PointNet pooling surface samples into a sparse voxel grid.

    Points (positions + normals) are embedded by an MLP and mean-pooled
    into the occupied cells of a ``res^3`` grid. The paper replaces
    scatter-pooling with a sorted CSR layout for speed; this reproduction
    uses ``index_add`` (functionally equivalent).
    """

    def __init__(self, width: int, res: int) -> None:
        super().__init__()
        self.res = res
        self.mlp = nn.Sequential(
            nn.Linear(6, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self, points: torch.Tensor, normals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool points into voxel features.

        Args:
            points: ``(P, 3)`` sample positions in ``[0, 1]^3``.
            normals: ``(P, 3)`` sample normals.

        Returns:
            ``(features, coords)``: ``(K, C)`` pooled features and
            ``(K, 3)`` voxel-center coordinates in ``[0, 1]^3``.
        """
        feats = self.mlp(torch.cat([points, normals], dim=-1))
        idx = (points * self.res).floor().long().clamp(0, self.res - 1)
        keys = idx[:, 0] * (self.res**2) + idx[:, 1] * self.res + idx[:, 2]
        uniq, inverse = torch.unique(keys, return_inverse=True)
        pooled = torch.zeros(
            len(uniq), feats.shape[1], device=feats.device, dtype=feats.dtype
        )
        pooled.index_add_(0, inverse, feats)
        counts = torch.zeros(len(uniq), device=feats.device, dtype=feats.dtype)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=feats.dtype))
        pooled = pooled / counts.clamp_min(1).unsqueeze(-1)
        iz = uniq % self.res
        iy = (uniq // self.res) % self.res
        ix = uniq // (self.res**2)
        cells = torch.stack([ix, iy, iz], dim=-1).to(feats.dtype)
        coords = (cells + 0.5) / self.res
        return pooled, coords


class MeshVAEEncoder(nn.Module):
    """Encoder mapping a mesh to one latent token per vertex."""

    def __init__(self, cfg: MeshVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context = VoxelContextEncoder(cfg.width, cfg.point_voxel_res)
        self.query_in = nn.Linear(3 * 2 * cfg.fourier_freqs, cfg.width)
        self.cross = CrossAttention(cfg.width, cfg.heads, rope=True)
        self.graph_layers = nn.ModuleList(
            SelfAttention(cfg.width, cfg.heads) for _ in range(cfg.enc_layers)
        )
        self.self_layers = nn.ModuleList(
            SelfAttention(cfg.width, cfg.heads, rope=True)
            for _ in range(cfg.enc_layers)
        )
        self.ffns = nn.ModuleList(SwiGLU(cfg.width) for _ in range(cfg.enc_layers))
        self.out_norm = nn.LayerNorm(cfg.width)
        self.to_latent = nn.Linear(cfg.width, 2 * cfg.latent_channels)

    def forward(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one mesh.

        Args:
            vertices: ``(V, 3)`` vertex positions in ``[0, 1]^3``.
            edges: ``(E, 2)`` ground-truth undirected edges.
            points: ``(P, 3)`` surface samples.
            normals: ``(P, 3)`` surface sample normals.

        Returns:
            ``(mu, logvar)`` latent parameters, each ``(V, C)``.
        """
        v = vertices.shape[0]
        ctx, ctx_coords = self.context(points, normals)
        x = self.query_in(fourier_features(vertices, self.cfg.fourier_freqs))
        x = x + self.cross(
            _norm(x.unsqueeze(0)), ctx.unsqueeze(0), vertices, ctx_coords
        ).squeeze(0)
        adjacency = torch.zeros(v, v, dtype=torch.bool, device=x.device)
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
        xb = x.unsqueeze(0)
        for graph, self_attn, ffn in zip(
            self.graph_layers, self.self_layers, self.ffns, strict=True
        ):
            xb = xb + graph(_norm(xb), attn_mask=adjacency)
            xb = xb + self_attn(_norm(xb), coords=vertices)
            xb = xb + ffn(_norm(xb))
        mu, logvar = self.to_latent(self.out_norm(xb.squeeze(0))).chunk(2, dim=-1)
        return mu, logvar


class MeshVAEDecoder(nn.Module):
    """Pure set decoder: positions, edge embeddings, face embeddings."""

    def __init__(self, cfg: MeshVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.latent_channels, cfg.width)
        self.trunk = nn.ModuleList(
            TransformerBlock(cfg.width, cfg.heads) for _ in range(cfg.dec_trunk_layers)
        )
        self.vertex_branch = nn.ModuleList(
            TransformerBlock(cfg.width, cfg.heads) for _ in range(cfg.dec_branch_layers)
        )
        self.topology_branch = nn.ModuleList(
            TransformerBlock(cfg.width, cfg.heads) for _ in range(cfg.dec_branch_layers)
        )
        self.pos_head = nn.Linear(cfg.width, 3)
        self.edge_head = nn.Linear(cfg.width, 2 * cfg.edge_dim)
        self.face_head = nn.Linear(cfg.width, 3 * cfg.face_dim)
        # Learnable NULL role vectors for open-boundary fans.
        self.null_prev = nn.Parameter(torch.randn(cfg.face_dim) * 0.02)
        self.null_next = nn.Parameter(torch.randn(cfg.face_dim) * 0.02)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent tokens.

        Args:
            z: ``(V, C)`` latent set (no positional encoding is used).

        Returns:
            ``(positions, edge_emb, face_emb)`` of shapes ``(V, 3)``,
            ``(V, 2 d_e)`` and ``(V, 3 d_f)``.
        """
        x = self.in_proj(z).unsqueeze(0)
        for block in self.trunk:
            x = block(x)
        hv = x
        for block in self.vertex_branch:
            hv = block(hv)
        ht = x
        for block in self.topology_branch:
            ht = block(ht)
        positions = self.pos_head(hv.squeeze(0))
        edge_emb = self.edge_head(ht.squeeze(0))
        face_emb = self.face_head(ht.squeeze(0))
        return positions, edge_emb, face_emb


def edge_logits(edge_emb: torch.Tensor, edge_dim: int) -> torch.Tensor:
    """Compute the symmetric spacetime adjacency logits (Eq. 1).

    Args:
        edge_emb: ``(V, 2 d_e)`` edge embeddings.
        edge_dim: ``d_e`` per half.

    Returns:
        ``(V, V)`` logits; an edge is predicted when ``A_ij > 0``.
    """
    space, time = edge_emb.split(edge_dim, dim=-1)
    d_time = torch.cdist(time, time) ** 2
    d_space = torch.cdist(space, space) ** 2
    return d_time - d_space


def face_scores(
    face_emb: torch.Tensor,
    null_prev: torch.Tensor,
    null_next: torch.Tensor,
    face_dim: int,
    vert_idx: torch.Tensor,
    neighbors: torch.Tensor,
) -> torch.Tensor:
    """Score successor candidates for one degree group (Eq. 4).

    Args:
        face_emb: ``(V, 3 d_f)`` face embeddings of all vertices.
        null_prev: ``(d_f,)`` learnable NULL predecessor vector.
        null_next: ``(d_f,)`` learnable NULL successor vector.
        face_dim: ``d_f`` per role.
        vert_idx: ``(B,)`` indices of the fan-center vertices.
        neighbors: ``(B, D)`` neighbor indices for vertices of degree ``D``.

    Returns:
        ``(B, D + 1, D + 1)`` logits ``Phi_i[p, q]`` where the last
        row/column corresponds to the NULL element.
    """
    root, prev, nxt = face_emb.split(face_dim, dim=-1)
    prev_nb = prev[neighbors]  # (B, D, d_f)
    next_nb = nxt[neighbors]
    b = neighbors.shape[0]
    prev_full = torch.cat([prev_nb, null_prev.view(1, 1, -1).expand(b, 1, -1)], dim=1)
    next_full = torch.cat([next_nb, null_next.view(1, 1, -1).expand(b, 1, -1)], dim=1)
    root_c = root[vert_idx]  # (B, d_f)
    return torch.einsum("bf,bpf,bqf->bpq", root_c, prev_full, next_full)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, x.shape[-1:])


class MeshVAE(nn.Module):
    """Full vertex-set mesh autoencoder (encoder + decoder)."""

    def __init__(self, cfg: MeshVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MeshVAEConfig()
        self.encoder = MeshVAEEncoder(self.cfg)
        self.decoder = MeshVAEDecoder(self.cfg)

    def encode(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        points: torch.Tensor,
        normals: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor:
        """Encode a mesh to its latent set.

        Args:
            vertices: ``(V, 3)`` positions in ``[0, 1]^3``.
            edges: ``(E, 2)`` edges.
            points: ``(P, 3)`` surface samples.
            normals: ``(P, 3)`` surface normals.
            sample: Sample from the posterior (adds noise); otherwise use
                the posterior mean, matching the deterministic usage of
                the frozen VAE in the generation stages.

        Returns:
            ``(V, C)`` latent tokens.
        """
        mu, logvar = self.encoder(vertices, edges, points, normals)
        if sample and self.cfg.kl_weight > 0:
            std = (0.5 * logvar).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def decode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent tokens to positions and topology embeddings."""
        return self.decoder(z)

    def forward(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode a mesh.

        Returns:
            ``(positions, edge_emb, face_emb, z)``.
        """
        z = self.encode(vertices, edges, points, normals, sample=True)
        positions, edge_emb, face_emb = self.decode(z)
        return positions, edge_emb, face_emb, z
