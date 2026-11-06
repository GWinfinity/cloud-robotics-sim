"""ADMM elastic-QP kernel tests.

References, all small enough to be exact:
* unconstrained problems have the closed form v = -P^-1 c;
* moderate-penalty problems are certified by the duality gap of the
  elastic-QP dual, g(lambda) with lambda from the ADMM duals -- for a convex
  QP a zero gap proves global optimality, no solver cross-check needed;
* hard-penalty feasible problems are compared against scipy's SLSQP on the
  equivalent hard QP (SLSQP is reliable in the feasible regime).
"""

import numpy as np
import pytest

from plugins.solvers.qd_ik.qp import QPConfig, QPData, solve


def _dual_gap(P, c, G, h, pen, v, u, z, rho):
    """Relative duality gap of the elastic QP using ADMM's dual estimate."""
    w = np.sqrt(pen)
    lam = np.clip(w * rho * (u + (w[:, None] * G) @ v - z), 0.0, pen)
    f = 0.5 * v @ P @ v + c @ v + pen @ np.maximum(G @ v - h, 0)
    Pinv = np.linalg.inv(P)
    g = -0.5 * (c + G.T @ lam) @ Pinv @ (c + G.T @ lam) - lam @ h
    return (f - g) / max(1.0, abs(f))


def test_unconstrained_closed_form():
    """Unconstrained elastic QP solves to the closed form v = -P^-1 c."""
    rng = np.random.default_rng(0)
    B, nv = 3, 7
    A = rng.normal(size=(B, nv, nv))
    P = np.einsum("bij,bik->bjk", A, A) + np.eye(nv)
    c = rng.normal(size=(B, nv))
    data = QPData(nv, 0, B, QPConfig())
    v = solve(P, c, np.zeros((B, 0, nv)), np.zeros((B, 0)), np.zeros(0), data)
    for b in range(B):
        assert np.allclose(v[b], np.linalg.solve(P[b], -c[b]), atol=1e-10)


def test_moderate_penalty_duality_gap():
    """Moderate-penalty solutions carry a small convex duality gap."""
    rng = np.random.default_rng(3)
    B, nv, m = 4, 12, 30
    A = rng.normal(size=(B, nv, nv))
    P = np.einsum("bij,bik->bjk", A, A) + np.eye(nv)
    c = rng.normal(size=(B, nv))
    G = rng.normal(size=(B, m, nv))
    h = rng.normal(size=(B, m)) * 0.5
    pen = np.full(m, 1e2)
    data = QPData(nv, m, B, QPConfig(iters=5000, rho=10.0, relax=1.5))
    v = solve(P, c, G, h, pen, data, warm=False)
    for b in range(B):
        gap = _dual_gap(
            P[b], c[b], G[b], h[b], pen, v[b], data.u[b], data.z[b], data.config.rho
        )
        assert gap < 1e-3


def test_hard_penalty_feasible_vs_slsqp():
    """Hard-penalty feasible problems match SLSQP on the equivalent hard QP."""
    pytest.importorskip("scipy")
    from scipy.optimize import minimize

    rng = np.random.default_rng(11)
    B, nv, m = 3, 20, 60
    A = rng.normal(size=(B, nv, nv))
    P = np.einsum("bij,bik->bjk", A, A) + 10.0 * np.eye(nv)
    c = rng.normal(size=(B, nv))
    G = rng.normal(scale=0.5, size=(B, m, nv))
    v_true = rng.normal(scale=0.1, size=(B, nv))
    h = np.einsum("bki,bi->bk", G, v_true) + rng.uniform(0.0, 2.0, size=(B, m))
    pen = np.full(m, 1e6)
    rho = float(np.mean(np.diagonal(P, axis1=1, axis2=2)) / pen[0])
    data = QPData(nv, m, B, QPConfig(iters=3000, rho=rho, relax=1.5))
    v = solve(P, c, G, h, pen, data, warm=False)
    for b in range(B):
        res = minimize(
            lambda x, P_=P[b], c_=c[b]: 0.5 * x @ P_ @ x + c_ @ x,
            v_true[b],
            jac=lambda x, P_=P[b], c_=c[b]: P_ @ x + c_,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x, G_=G[b], h_=h[b]: h_ - G_ @ x,
                    "jac": lambda x, G_=G[b]: -G_,
                }
            ],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        assert res.success
        assert np.max(G[b] @ v[b] - h[b]) < 1e-6
        f_admm = 0.5 * v[b] @ P[b] @ v[b] + c[b] @ v[b]
        rel = (f_admm - res.fun) / max(1.0, abs(res.fun))
        assert rel < 1e-2


def test_warm_continuation():
    """A warm-started re-solve after a small perturbation stays close to a
    fresh cold solve of the perturbed problem.
    """
    rng = np.random.default_rng(11)
    B, nv, m = 2, 20, 60
    A = rng.normal(size=(B, nv, nv))
    P = np.einsum("bij,bik->bjk", A, A) + 10.0 * np.eye(nv)
    c = rng.normal(size=(B, nv))
    G = rng.normal(scale=0.5, size=(B, m, nv))
    v_true = rng.normal(scale=0.1, size=(B, nv))
    h = np.einsum("bki,bi->bk", G, v_true) + rng.uniform(0.0, 2.0, size=(B, m))
    pen = np.full(m, 1e6)
    rho = float(np.mean(np.diagonal(P, axis1=1, axis2=2)) / pen[0])
    data = QPData(nv, m, B, QPConfig(iters=5000, rho=rho, relax=1.5))
    solve(P, c, G, h, pen, data, warm=False)
    h2 = h + rng.normal(scale=0.01, size=h.shape)
    v1 = solve(P, c, G, h2, pen, data, warm=True)
    ref = QPData(nv, m, B, QPConfig(iters=5000, rho=rho, relax=1.5))
    v1_ref = solve(P, c, G, h2, pen, ref, warm=False)
    drift = np.linalg.norm(v1 - v1_ref, axis=1) / np.linalg.norm(v1_ref, axis=1)
    assert drift.max() < 1e-2
