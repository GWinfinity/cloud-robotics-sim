"""Inference-time mesh assembly (Sec. 2.1, "Mesh assembly").

Decoder outputs are converted into an explicit mesh in three steps:

1. Predicted positions become vertices; every pair with ``A_ij > 0``
   becomes an undirected edge.
2. Each soft successor matrix ``P_i`` is rounded to a hard successor
   mapping by a linear assignment constrained to form a single fan
   (cycles are merged greedily when Hungarian returns sub-cycles).
3. Oriented triangles are read off the hard successor mappings, each
   directed halfedge used at most once.

Connected components come for free: multi-part assets decode as a single
mesh whose parts are separated in the vertex-edge graph.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.mesh_vae import edge_logits, face_scores
from .halfedge import faces_from_topology


def assemble_mesh(
    positions: torch.Tensor,
    edge_emb: torch.Tensor,
    face_emb: torch.Tensor,
    null_prev: torch.Tensor,
    null_next: torch.Tensor,
    edge_dim: int,
    face_dim: int,
    edge_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble an explicit mesh from decoder outputs.

    Args:
        positions: ``(V, 3)`` predicted vertex positions.
        edge_emb: ``(V, 2 d_e)`` edge embeddings.
        face_emb: ``(V, 3 d_f)`` face embeddings.
        null_prev: Decoder NULL predecessor vector.
        null_next: Decoder NULL successor vector.
        edge_dim: ``d_e`` per half.
        face_dim: ``d_f`` per role.
        edge_threshold: Logit threshold for accepting an edge.

    Returns:
        ``(vertices, edges, faces)`` as numpy arrays with shapes
        ``(V, 3)``, ``(E, 2)`` and ``(F, 3)``.
    """
    verts = positions.detach().cpu().numpy()
    a = edge_logits(edge_emb.detach(), edge_dim)
    v = a.shape[0]
    upper = torch.triu(torch.ones(v, v, dtype=torch.bool), diagonal=1)
    ei, ej = torch.where((a > edge_threshold) & upper)
    edges = torch.stack([ei, ej], dim=-1).cpu().numpy().astype(np.int64)

    nbr_lists: list[list[int]] = [[] for _ in range(v)]
    for i, j in edges:
        nbr_lists[i].append(int(j))
        nbr_lists[j].append(int(i))

    neighbors: list[np.ndarray] = []
    successors: list[np.ndarray] = []
    for i in range(v):
        nbrs = np.asarray(sorted(nbr_lists[i]), dtype=np.int64)
        d = len(nbrs)
        if d == 0:
            neighbors.append(nbrs)
            successors.append(np.zeros(1, dtype=np.int64))
            continue
        phi = face_scores(
            face_emb,
            null_prev,
            null_next,
            face_dim,
            torch.tensor([i], device=face_emb.device),
            torch.from_numpy(nbrs)[None].to(face_emb.device),
        )[0]
        succ = _round_single_fan(phi.detach().cpu().numpy())
        neighbors.append(nbrs)
        successors.append(succ)

    faces = faces_from_topology(neighbors, successors)
    return verts, edges, faces


def _round_single_fan(phi: np.ndarray) -> np.ndarray:
    """Round a score matrix to a successor mapping forming one fan.

    Hungarian assignment gives a permutation on ``N(i) ∪ {NULL}``; if it
    decomposes into several disjoint cycles they are merged by greedily
    swapping the successor pair with the best score gain until a single
    cycle remains.

    Args:
        phi: ``(D + 1, D + 1)`` successor scores.

    Returns:
        ``(D + 1,)`` successor target indices.
    """
    from scipy.optimize import linear_sum_assignment

    _, col = linear_sum_assignment(-phi)
    perm = col.astype(np.int64)
    while True:
        cycles = _cycles(perm)
        if len(cycles) <= 1:
            return perm
        # Merge the two shortest cycles with the best swap.
        cycles.sort(key=len)
        c1, c2 = cycles[0], cycles[1]
        best = None
        best_gain = -np.inf
        for a in c1:
            for b in c2:
                gain = (
                    phi[a, perm[b]]
                    + phi[b, perm[a]]
                    - phi[a, perm[a]]
                    - phi[b, perm[b]]
                )
                if gain > best_gain:
                    best_gain = gain
                    best = (a, b)
        a, b = best  # type: ignore[misc]
        perm[a], perm[b] = perm[b], perm[a]


def _cycles(perm: np.ndarray) -> list[list[int]]:
    """Decompose a permutation into its cycles."""
    seen = np.zeros(len(perm), dtype=bool)
    cycles = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        cycle = []
        k = start
        while not seen[k]:
            seen[k] = True
            cycle.append(k)
            k = int(perm[k])
        cycles.append(cycle)
    return cycles
