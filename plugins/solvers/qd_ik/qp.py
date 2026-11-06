"""Batched elastic-QP solver (ADMM), the Quadrants stand-in for ElastiQP.

Solves, independently for every row of the batch,

    min  0.5 v'Pv + c'v + penalty' t
    s.t. G v - t <= h,  t >= 0

i.e. the same elastic QP ElastiQP solves (L1-relaxed inequalities, no hard
equalities -- the constraints qd_ik ports are all inequalities).  The solver
is over-relaxed ADMM (Boyd et al. 2011, sec. 3.4.3) on the z = Gv split,
with the (z, t) subproblem in closed form:

    g     = G v
    y     = relax * g + (1 - relax) * z + u
    t_i   = max(0, y_i - h_i - penalty_i / rho)
    z_i   = min(y_i, h_i + t_i)
    u    += y_relaxed - z

and the v step through one Cholesky factorization of ``P + rho G'G`` per
instance.  Rows are equilibrated by ``w_i = sqrt(penalty_i)`` (Ruiz-style,
as ElastiQP's scaling): the kernel consumes w-weighted ``(w G, w h)`` and
unit penalties internally, so elastic duals live in ``[0, sqrt(pen)]`` and
the relaxation threshold is ``w / rho`` in scaled units -- with the raw
penalty the (z, t) closed form has a ``penalty / rho``-wide dead band and
dual ascent stalls at diff-IK's 1e6 penalties.

Fixed iteration count, warm-started from the previous solve's (v, z, u):
deterministic and batch-safe.  Accuracy is set by ``QPConfig.iters`` /
``QPConfig.rho``; crank ``iters`` up (or re-solve warm) if a downstream
check shows slack constraint residuals.

ElastiQP's DAS is an active-set method with Ruiz equilibration and KKT
tolerances; this kernel deliberately trades that sophistication for
SIMD-friendly batched regularity.  The *problem* is identical, the
*algorithm* is not -- do not expect bitwise agreement with elastiqp.
"""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

_F64_2 = qd.types.ndarray(dtype=qd.f64, ndim=2)
_F64_3 = qd.types.ndarray(dtype=qd.f64, ndim=3)


@dataclass(frozen=True)
class QPConfig:
    """ADMM knobs.

    Attributes:
        iters: ADMM iteration count per solve.
        rho: augmented-Lagrangian penalty.
        relax: over-relaxation in (1, 2); 1.0 disables.
    """

    iters: int = 1000
    rho: float = 10.0
    relax: float = 1.5


class QPData:
    """Per-batch scratch for the QP kernel: factor buffer plus the
    warm-start triple (v, z, u).  Do not share one QPData across threads.
    """

    def __init__(self, nv, m, batch, config):
        self.B = batch
        self.nv = nv
        self.m = m
        self.config = config
        self.L = np.zeros((batch, nv, nv))
        self.Ab = np.zeros((batch, nv, nv))
        self.v = np.zeros((batch, nv))
        self.z = np.zeros((batch, m))
        self.u = np.zeros((batch, m))

    def reset(self):
        """Cold start: zero the warm-start triple."""
        self.v[:] = 0.0
        self.z[:] = 0.0
        self.u[:] = 0.0


def build(nv, m, penalty, config: QPConfig):
    """Compile the ADMM kernel for one (nv, m, penalty) problem shape.

    Args:
        nv: number of decision variables
        m: number of elastic inequality rows
        penalty: (m,) per-row L1 relaxation penalties (static, becomes a field)
        config: :class:`QPConfig`

    Returns:
        the compiled kernel.  Cached on (nv, m, penalty bytes, config).
    """
    penalty = np.broadcast_to(np.asarray(penalty, dtype=np.float64), (m,)).copy()
    if np.any(penalty <= 0.0):
        raise ValueError(
            "elastic penalties must be positive; drop unconstrained rows instead"
        )
    key = ("qp", nv, m, penalty.tobytes(), config.iters, config.rho, config.relax)
    cache = _KERNEL_CACHE
    if key in cache:
        return cache[key]

    from .model import ensure_init

    ensure_init()
    # scaled-row weights w_i = sqrt(penalty_i): penalty_i * t_i == w_i * t_scaled
    f_w = qd.field(qd.f64, shape=m)
    f_w.from_numpy(np.sqrt(penalty))
    relax = float(config.relax)
    rho = float(config.rho)

    @qd.kernel
    def k_qp(
        P: _F64_3,
        c: _F64_2,
        G: _F64_3,
        GT: _F64_3,
        h: _F64_2,
        v: _F64_2,
        z: _F64_2,
        u: _F64_2,
        Ab: _F64_3,
        Lb: _F64_3,
        B: qd.i32,
        M: qd.i32,
        n_it: qd.i32,
    ):
        for b in range(B):
            # ---- A = P + rho (wG)'(wG) into Ab, then in-place Cholesky of Ab.
            # All loops here are runtime-bound: unrolled 35^3-scale IR made
            # LLVM spend ~100 s per compile, which is never worth it.
            for r in range(nv):
                for cc in range(nv):
                    Ab[b, r, cc] = P[b, r, cc]
            for r in range(nv):
                for k in range(M):
                    g = f_w[k] * G[b, k, r]
                    if g != 0.0:
                        for cc in range(nv):
                            Ab[b, r, cc] = Ab[b, r, cc] + rho * g * (
                                f_w[k] * G[b, k, cc]
                            )
            for i in range(nv):
                for j in range(i + 1):
                    s = qd.cast(Ab[b, i, j], qd.f64)
                    for k in range(j):
                        s = s - Lb[b, i, k] * Lb[b, j, k]
                    if i == j:
                        Lb[b, i, j] = qd.sqrt(s)
                    else:
                        Lb[b, i, j] = s / Lb[b, j, j]
            # ---- ADMM; (v, z, u) carry the warm start
            for _it in range(n_it):
                # rhs = -c + rho (wG)'(z - u); v-step via the factor in Lb
                for r in range(nv):
                    s = qd.cast(-c[b, r], qd.f64)
                    for k in range(M):
                        s = s + rho * GT[b, r, k] * f_w[k] * (z[b, k] - u[b, k])
                    v[b, r] = s  # forward substitution, in place
                for i in range(nv):
                    s = qd.cast(v[b, i], qd.f64)
                    for j in range(i):
                        s = s - Lb[b, i, j] * v[b, j]
                    v[b, i] = s / Lb[b, i, i]
                for i in range(nv):
                    ii = nv - 1 - i
                    s = qd.cast(v[b, ii], qd.f64)
                    for j in range(ii + 1, nv):
                        s = s - Lb[b, j, ii] * v[b, j]
                    v[b, ii] = s / Lb[b, ii, ii]
                # (z, t)-step in the scaled constraint space; the scaled
                # penalty is w, so the relaxation threshold is w / rho and
                # the relaxed dual saturates at w == sqrt(pen) (pen unscaled)
                for k in range(M):
                    g = qd.cast(0.0, qd.f64)
                    for r in range(nv):
                        g = g + (f_w[k] * G[b, k, r]) * v[b, r]
                    zold = qd.cast(z[b, k], qd.f64)
                    yr = relax * g + (1.0 - relax) * zold + u[b, k]
                    hs = f_w[k] * h[b, k]
                    t = yr - hs - f_w[k] / rho
                    if t < 0.0:
                        t = qd.cast(0.0, qd.f64)
                    znew = hs + t
                    if yr < znew:
                        znew = yr
                    u[b, k] = u[b, k] + (relax * g + (1.0 - relax) * zold) - znew
                    z[b, k] = znew

    cache[key] = k_qp
    return k_qp


_KERNEL_CACHE: dict = {}


def solve(P, c, G, h, penalty, data: QPData, warm=True):
    """Solve the batched elastic QP in place.

    Args:
        P: (B, nv, nv) symmetric positive-definite cost matrices
        c: (B, nv) cost vectors
        G: (B, m, nv) inequality rows
        h: (B, m) inequality bounds
        penalty: (m,) per-row penalties (must match the kernel's build)
        data: :class:`QPData` scratch + warm start (v, z, u)
        warm: False to cold-start (zero v, z, u first)

    Returns:
        v: (B, nv) view of data.v (valid until the next solve call)
    """
    B, nv = data.B, data.nv
    for name, arr, shape in (
        ("P", P, (B, nv, nv)),
        ("c", c, (B, nv)),
        ("G", G, (B, data.m, nv)),
        ("h", h, (B, data.m)),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, want {shape}")
    if not warm:
        data.reset()
    if data.m == 0:
        # unconstrained: v = P^-1 (-c), batched dense solve on the host
        data.v[:] = np.linalg.solve(P, -c[..., None])[..., 0]
        return data.v
    kernel = build(nv, data.m, penalty, data.config)
    GT = np.ascontiguousarray(np.swapaxes(G, 1, 2))
    kernel(
        P,
        c,
        G,
        GT,
        h,
        data.v,
        data.z,
        data.u,
        data.Ab,
        data.L,
        B,
        data.m,
        data.config.iters,
    )
    return data.v
