"""Loss functions for the vertex-set Mesh VAE (Sec. 2.1, Eqs. 2/5/6).

* ``vertex_loss``: MSE on regressed positions.
* ``edge_loss``: class-balanced softplus BCE on the strict upper triangle
  of the spacetime adjacency logits, computed in row chunks so no dense
  ``V x V`` supervision target is materialized.
* ``face_loss``: NLL of the Sinkhorn-normalized successor probabilities,
  evaluated grouped by vertex degree so no dense ``[V, Dmax, Dmax]``
  tensor is built.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from .models.mesh_vae import MeshVAEConfig, face_scores


def vertex_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and ground-truth positions."""
    return functional.mse_loss(pred, target)


def edge_loss(
    edge_emb: torch.Tensor,
    edge_dim: int,
    edges: torch.Tensor,
    num_vertices: int,
    neg_weight: float | None = None,
    chunk: int = 1024,
) -> torch.Tensor:
    """Class-balanced softplus BCE over the strict upper triangle (Eq. 2).

    Args:
        edge_emb: ``(V, 2 d_e)`` edge embeddings.
        edge_dim: ``d_e`` per half.
        edges: ``(E, 2)`` ground-truth edges with ``i < j``.
        num_vertices: ``V``.
        neg_weight: Weight ``lambda`` for negative pairs; ``None``
            activates class balancing ``E / (num_pairs - E)``.
        chunk: Row chunk size bounding the dense ``(chunk, V)`` logits.

    Returns:
        Scalar loss.
    """
    v = num_vertices
    num_pairs = v * (v - 1) // 2
    e = edges.shape[0]
    if neg_weight is None:
        neg_weight = e / max(num_pairs - e, 1)
    # Sorted edge keys for O(log E) membership tests.
    keys = torch.sort(edges[:, 0] * v + edges[:, 1]).values

    pos_sum = torch.zeros((), device=edge_emb.device)
    neg_sum = torch.zeros((), device=edge_emb.device)
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        a = edge_logits_chunk(edge_emb, edge_dim, start, stop)  # (r, V)
        rows = torch.arange(start, stop, device=a.device)
        cols = torch.arange(v, device=a.device)
        upper = cols.unsqueeze(0) > rows.unsqueeze(1)  # strict upper triangle
        pair_keys = rows.unsqueeze(1) * v + cols.unsqueeze(0)
        is_edge = _isin_sorted(pair_keys.reshape(-1), keys).reshape(a.shape)
        is_edge &= upper
        is_non = upper & ~is_edge
        pos_sum = pos_sum + functional.softplus(-a[is_edge]).sum()
        neg_sum = neg_sum + functional.softplus(a[is_non]).sum()
    z = e + neg_weight * max(num_pairs - e, 0)
    return (pos_sum + neg_weight * neg_sum) / max(z, 1e-8)


def edge_logits_chunk(
    edge_emb: torch.Tensor, edge_dim: int, start: int, stop: int
) -> torch.Tensor:
    """Compute rows ``start:stop`` of the spacetime adjacency logits."""
    space, time = edge_emb.split(edge_dim, dim=-1)
    t, s = time[start:stop], space[start:stop]
    d_time = t.pow(2).sum(-1, keepdim=True) + time.pow(2).sum(-1) - 2 * t @ time.T
    d_space = s.pow(2).sum(-1, keepdim=True) + space.pow(2).sum(-1) - 2 * s @ space.T
    return d_time - d_space


def _isin_sorted(x: torch.Tensor, sorted_vals: torch.Tensor) -> torch.Tensor:
    """Membership test against a sorted 1-D tensor."""
    idx = torch.searchsorted(sorted_vals, x)
    idx = idx.clamp(max=len(sorted_vals) - 1)
    return sorted_vals[idx] == x


def sinkhorn_log(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Normalize logits into a log-domain doubly stochastic matrix.

    Args:
        logits: ``(..., N, N)`` scores.
        iters: Number of Sinkhorn iterations.

    Returns:
        Log of the (approximately) doubly stochastic matrix.
    """
    log_p = logits
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p


def group_by_degree(
    neighbors: list[torch.Tensor], successors: list[torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Group per-vertex fan data by degree.

    Args:
        neighbors: Per-vertex ``(D_i,)`` neighbor indices.
        successors: Per-vertex ``(D_i + 1,)`` successor targets.

    Returns:
        List of ``(vert_idx, neighbors, targets)`` batched tensors per
        degree group, with shapes ``(B,)``, ``(B, D)``, ``(B, D + 1)``.
    """
    groups: dict[int, list[int]] = {}
    for i, nbrs in enumerate(neighbors):
        groups.setdefault(len(nbrs), []).append(i)
    out = []
    device = neighbors[0].device if neighbors else "cpu"
    for d, idx in sorted(groups.items()):
        if d == 0:
            continue
        vert_idx = torch.tensor(idx, dtype=torch.long, device=device)
        nb = torch.stack([neighbors[i] for i in idx])
        tg = torch.stack([successors[i] for i in idx])
        out.append((vert_idx, nb, tg))
    return out


def face_loss(
    face_emb: torch.Tensor,
    null_prev: torch.Tensor,
    null_next: torch.Tensor,
    face_dim: int,
    neighbors: list[torch.Tensor],
    successors: list[torch.Tensor],
    sinkhorn_iters: int,
) -> torch.Tensor:
    """Negative log-likelihood of the successor mappings (Eq. 5).

    Evaluated grouped by vertex degree; covers both the triangle-induced
    successor links and the NULL transitions.
    """
    total = torch.zeros((), device=face_emb.device)
    count = 0
    for vert_idx, nb, targets in group_by_degree(neighbors, successors):
        phi = face_scores(face_emb, null_prev, null_next, face_dim, vert_idx, nb)
        log_p = sinkhorn_log(phi, sinkhorn_iters)  # (B, D+1, D+1)
        rows = torch.arange(nb.shape[1] + 1, device=face_emb.device)
        picked = log_p[:, rows, targets]  # (B, D+1)
        total = total - picked.sum()
        count += picked.numel()
    return total / max(count, 1)


def mesh_vae_loss(
    positions: torch.Tensor,
    edge_emb: torch.Tensor,
    face_emb: torch.Tensor,
    null_prev: torch.Tensor,
    null_next: torch.Tensor,
    vertices: torch.Tensor,
    edges: torch.Tensor,
    neighbors: list[torch.Tensor],
    successors: list[torch.Tensor],
    cfg: MeshVAEConfig,
    w_vertex: float = 100.0,
    w_edge: float = 10.0,
    w_face: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of the vertex, edge and face losses (Eq. 6).

    Args:
        positions: Predicted ``(V, 3)`` vertex positions.
        edge_emb: Predicted ``(V, 2 d_e)`` edge embeddings.
        face_emb: Predicted ``(V, 3 d_f)`` face embeddings.
        null_prev: Decoder NULL predecessor vector.
        null_next: Decoder NULL successor vector.
        vertices: Ground-truth ``(V, 3)`` positions.
        edges: Ground-truth ``(E, 2)`` edges.
        neighbors: Per-vertex neighbor index tensors.
        successors: Per-vertex successor target tensors.
        cfg: :class:`MeshVAEConfig`-like object with ``edge_dim``,
            ``face_dim`` and ``sinkhorn_iters``.
        w_vertex: Vertex loss weight (paper: 100).
        w_edge: Edge loss weight (paper: 10).
        w_face: Face loss weight (paper: 10).

    Returns:
        ``(total, parts)`` with the scalar loss and a logging dict.
    """
    l_v = vertex_loss(positions, vertices)
    l_e = edge_loss(edge_emb, cfg.edge_dim, edges, vertices.shape[0])
    l_f = face_loss(
        face_emb,
        null_prev,
        null_next,
        cfg.face_dim,
        neighbors,
        successors,
        cfg.sinkhorn_iters,
    )
    total = w_vertex * l_v + w_edge * l_e + w_face * l_f
    parts = {"vertex": float(l_v), "edge": float(l_e), "face": float(l_f)}
    return total, parts
