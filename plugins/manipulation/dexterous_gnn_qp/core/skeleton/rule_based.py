"""Rule-based skeleton selection used in Phase-1 (no GNN yet)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from dexterous_gnn_qp.core.env.sim import Contact, SimState
from dexterous_gnn_qp.core.utils.config import DotDict
from dexterous_gnn_qp.core.utils.math import skew


def _point_in_convex_hull_2d(points: np.ndarray, query: np.ndarray) -> bool:
    """Check whether a 2D query point lies inside the convex hull of points.

    Returns False if the hull has fewer than 3 points.
    """
    if len(points) < 3:
        return False
    try:
        hull = ConvexHull(points)
    except Exception:
        return False
    # Express query as barycentric combination of hull vertices (least squares).
    verts = points[hull.vertices]
    A = np.vstack([verts.T, np.ones(len(verts))])
    b = np.append(query, 1.0)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return bool(np.all(coeffs >= -1e-8))


def _force_closure_rank(contacts: List[Contact], obj_com: np.ndarray, threshold: float = 1e-4) -> bool:
    """Check if the contact wrench span has full rank (6)."""
    if len(contacts) < 2:
        return False
    G = np.zeros((6, 3 * len(contacts)))
    for i, cnt in enumerate(contacts):
        r = cnt.pos - obj_com
        G[:, 3 * i : 3 * (i + 1)] = np.vstack([skew(r), np.eye(3)])
    sigma = np.linalg.svd(G, compute_uv=False)
    return float(sigma[5]) > threshold


def select_skeleton(
    state: SimState,
    previous_forces: np.ndarray | None,
    cfg: DotDict,
) -> Tuple[List[int], List[int]]:
    """Select skeleton contacts and return (skeleton_indices, edge_indices).

    The selection sorts contacts by force magnitude (using the previous full QP
    solution) and greedily adds contacts until the desired size is reached and
    optional stability / force-closure checks pass.
    """
    contacts = state.contacts
    n_c = len(contacts)
    min_size = int(cfg.skeleton.min_size)
    max_size = int(cfg.skeleton.max_size)
    target_size = int(cfg.skeleton.target_size)
    threshold = float(cfg.skeleton.force_threshold)
    support_check = bool(cfg.skeleton.support_polygon_check)
    closure_check = bool(cfg.skeleton.force_closure_check)

    all_indices = list(range(n_c))
    if n_c <= min_size:
        return all_indices, []

    # If no previous force history, fall back to using all contacts.
    if previous_forces is None or len(previous_forces) != 3 * n_c:
        return all_indices, []

    forces = previous_forces.reshape(n_c, 3)
    magnitudes = np.linalg.norm(forces, axis=1)
    order = np.argsort(-magnitudes)

    selected: List[int] = []
    for idx in order:
        if len(selected) >= max_size:
            break
        if magnitudes[idx] >= threshold or len(selected) < min_size:
            selected.append(int(idx))
        # Stop early once target size is reached and quality checks pass.
        if len(selected) >= target_size:
            sel_contacts = [contacts[i] for i in selected]
            ok = True
            if support_check:
                pts = np.array([contacts[i].pos[:2] for i in selected])
                ok = ok and _point_in_convex_hull_2d(pts, state.x_obj[:2])
            if closure_check:
                ok = ok and _force_closure_rank(sel_contacts, state.x_obj)
            if ok:
                break

    # Ensure minimum size.
    for idx in order:
        if len(selected) >= min_size:
            break
        if idx not in selected:
            selected.append(int(idx))

    selected_set = set(selected)
    edge_indices = [i for i in all_indices if i not in selected_set]
    return selected, edge_indices
