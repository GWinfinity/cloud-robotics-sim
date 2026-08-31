"""Thin wrapper around OSQP for building and solving QPs."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import osqp
import scipy.sparse as spa


def solve_osqp(
    P: np.ndarray,
    q: np.ndarray,
    A: spa.csc_matrix,
    l: np.ndarray,
    u: np.ndarray,
    warm_start: np.ndarray | None = None,
    verbose: bool = False,
    max_iter: int = 4000,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    polish: bool = True,
) -> Dict[str, Any]:
    """Build and solve a QP with OSQP.

    Returns a dict with keys: x, objective, status, solve_time, iter.
    """
    P_sparse = spa.csc_matrix(P)
    m = osqp.OSQP()
    m.setup(
        P=P_sparse,
        q=q,
        A=A,
        l=l,
        u=u,
        verbose=verbose,
        polish=polish,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        scaling=0,
    )
    if warm_start is not None:
        m.warm_start(x=warm_start)
    res = m.solve()
    return {
        "x": np.array(res.x) if res.x is not None else None,
        "y": np.array(res.y) if res.y is not None else None,
        "objective": float(res.info.obj_val) if hasattr(res.info, "obj_val") else float("nan"),
        "status": str(res.info.status),
        "solve_time": float(res.info.run_time),
        "iter": int(res.info.iter),
    }
