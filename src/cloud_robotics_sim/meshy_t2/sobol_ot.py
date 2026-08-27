"""Sobol point sets and position-assignment strategies for latent tokens.

Reproduces Sec. 2.3 / Sec. 3.1 of Meshy T2: latent tokens are unordered,
so 3D RoPE coordinates are assigned from a deterministic Sobol point set.
During training, ground-truth vertex coordinates are matched to Sobol
candidates by an optimal-transport (Hungarian) assignment minimizing
squared Euclidean cost; at inference the Sobol points are used directly.
A cheaper Morton-order pairing baseline is provided for ablation.
"""

from __future__ import annotations

import numpy as np
import torch


def sobol_points(n: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Draw ``n`` deterministic Sobol points in ``[0, 1)^dim``.

    Args:
        n: Number of points.
        dim: Dimensionality (3 for spatial coordinates).
        seed: Scramble seed; fixed for reproducibility.

    Returns:
        ``(n, dim)`` float64 array.
    """
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(n).numpy().astype(np.float64)


def _morton_codes(points: np.ndarray, bits: int = 10) -> np.ndarray:
    """Compute Morton (Z-order) codes for points in ``[0, 1]^3``."""
    q = np.clip((points * (1 << bits)).astype(np.int64), 0, (1 << bits) - 1)
    codes = np.zeros(len(points), dtype=np.int64)
    for b in range(bits):
        codes |= ((q[:, 0] >> b) & 1) << (3 * b)
        codes |= ((q[:, 1] >> b) & 1) << (3 * b + 1)
        codes |= ((q[:, 2] >> b) & 1) << (3 * b + 2)
    return codes


def ot_assign(points: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Optimal-transport assignment from points to candidates.

    Solves the minimum-cost bipartite matching with squared Euclidean
    cost via the Hungarian algorithm.

    Args:
        points: ``(N, 3)`` query points (e.g. ground-truth vertex coords).
        candidates: ``(M, 3)`` candidate positions with ``M >= N``
            (e.g. Sobol points).

    Returns:
        ``(N,)`` int64 array: candidate index assigned to each point.
    """
    from scipy.optimize import linear_sum_assignment

    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    diff = points[:, None, :] - candidates[None, :, :]
    cost = np.einsum("nmd,nmd->nm", diff, diff)
    _, col = linear_sum_assignment(cost)
    return col.astype(np.int64)


def morton_assign(points: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Pair points and candidates by Morton order (ablation baseline).

    Both sets are sorted by Morton code and paired sequentially.

    Args:
        points: ``(N, 3)`` query points.
        candidates: ``(M, 3)`` candidates with ``M >= N``.

    Returns:
        ``(N,)`` int64 array of assigned candidate indices.
    """
    order_p = np.argsort(_morton_codes(points), kind="stable")
    order_c = np.argsort(_morton_codes(candidates), kind="stable")
    assign = np.zeros(len(points), dtype=np.int64)
    assign[order_p] = order_c[: len(points)]
    return assign
