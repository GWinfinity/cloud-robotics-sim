"""Batch-parallel rigid body algorithms as Quadrants kernels.

v4 -- packed state buffers.

v3 moved the per-batch scratch out of Taichi fields and into ``ndarray`` kernel
arguments, which is what made one compiled kernel serve every ``Data`` and any
batch size.  The bill for that arrives per *call*: every ndarray parameter is
re-validated (dtype, ndim, strides) and its pointer extracted, at about 5.2 us
each (``bench/launch_cost.py``).  With 12 parameters RNEA paid 67 us a call and
ABA 114 us -- 12% and 7% of their wall clock at B=4096, and everything at B=1.

So the scratch is now packed into three buffers instead of eighteen:

    P  (B, nj, 4, 3)    rows of R_pi, then p_pi
    S  (B, nj, 12, 3)   motion / force state, one 3-vector per slot
    IC (B, nj, 9, 3)    inertia blocks A, B, C -- shared by CRBA and ABA,
                        which never run in the same kernel

RNEA drops to 7 parameters, ABA to 9.  The block index is a ``qd.template``
argument, so it is folded into a constant offset at trace time: the generated
code still does ``a[b, i, 4, 0]``, not ``a[b, i, k+1, 0]``.

Why three buffers and not one: the sweeps walk the tree slot by slot (all of
``V`` for every joint, then all of ``A``).  Keeping each slot's column
contiguous is what lets the prefetcher help.  A single (B, nj, 25, 3) blob
would stride 600 B per joint and drag the unused slots into L1.

Spatial algebra is unchanged from v2/v3: 3x3 blocks, ``[[A, B], [B^T, C]]`` for
an inertia, explicit cross products instead of 6x6 operators, and CRBA filled
by propagating ``F = Ic_i S_i`` up the tree (Fernbach et al. 2019, alg. 2).
"""

import numpy as np
import quadrants as qd

from .spatial import rot_axis_angle

_F64_2 = qd.types.ndarray(dtype=qd.f64, ndim=2)
_F64_3 = qd.types.ndarray(dtype=qd.f64, ndim=3)
_F64_4 = qd.types.ndarray(dtype=qd.f64, ndim=4)

# --- slot layout -----------------------------------------------------------
# P: three rows of the rotation, then the translation.
P_R = 0
P_p = 3
# S: every 6-vector the sweeps pass around.  Fixed rather than overlapped --
# sharing slots between algorithms saves 12 MB and costs a page of naming.
V1, V2, A1, A2, F1, F2, C1, C2, PA1, PA2, U1, U2 = range(12)
# IC: the three 3x3 blocks of a spatial inertia.
IC_A = 0
IC_B = 3
IC_C = 6


class Data:
    """Per-batch scratch space. Mirrors pinocchio.Data, but batch-major.

    Plain numpy buffers: no Taichi fields, so building one cannot trigger a
    compile.  Every buffer here is overwritten by the kernel that uses it.
    """

    def __init__(self, model, batch):
        self.model = model
        self.B = batch
        nj = model.nj
        B = batch
        # poses: R_pi rows + p_pi in one array, so both travel as one argument
        self.P = np.zeros((B, nj, 4, 3))
        # motion / force state, slot-major and flattened over the batch:
        # (12, B*nj, 3).  Each slot is contiguous down the tree (24-byte
        # stride) *and* the access keeps three indices, so the address is
        # base + k*stride + (bn+i)*3 + r with k a compile-time constant.
        self.S = np.zeros((12, B * nj, 3))
        # spatial inertia blocks; CRBA and ABA share this buffer because they
        # never run concurrently
        self.IC = np.zeros((B, nj, 9, 3))
        # per-joint scalars for ABA: d = S^T IA S, u = tau - S^T pA
        self.D2 = np.zeros((B, nj, 2))
        # world poses, only written by ``fk``
        self.R_o = np.zeros((B, nj, 3, 3))
        self.p_o = np.zeros((B, nj, 1, 3))
        # NB: no (B, nv, nv) buffer here.  M is an *output*, allocated per call
        # by ``crba``; parking one in Data would cost 130 MB at B=4096/nv=63 for
        # something most callers never touch.  Pass ``out=`` to reuse a buffer.

    def kernels(self):
        return self.model.kernels()

    def __repr__(self):
        """Short summary with batch and dof counts."""
        return f"<qd_rbd.Data B={self.B} nv={self.model.nv}>"


def _build(m):
    """Compile the kernel set for one model.  Cached on the model object.

    Model parameters stay in ``qd.field``s: there is one ``Model`` per tree, so
    those fields are compiled once and never change identity.
    """
    nj, nv = m.nj, m.nv
    fp, fjt, fax, fR, fpp, fiv = (
        m.f_parent,
        m.f_jtype,
        m.f_axis,
        m.f_Rpl,
        m.f_ppl,
        m.f_iv,
    )
    fA0, fB0, fmass = m.f_A0, m.f_B0, m.f_mass
    f_leaf, f_mc = m.f_leaf, m.f_mc
    f_row_full = m.f_row_full
    fg = m.f_grav

    # ------------------------------------------------ matrix / vector access
    # Three packed layouts, one accessor each: block ``k`` of a (B, nj, ., 3)
    # array is a 3x3 (rows k..k+2) or a 3-vector (slot k).  ``k`` is a
    # template argument so it lands as a constant offset in the generated code.
    @qd.func
    def _ld3(a: qd.template(), b, i, k: qd.template()):
        return qd.Matrix(
            [
                [a[b, i, k + 0, 0], a[b, i, k + 0, 1], a[b, i, k + 0, 2]],
                [a[b, i, k + 1, 0], a[b, i, k + 1, 1], a[b, i, k + 1, 2]],
                [a[b, i, k + 2, 0], a[b, i, k + 2, 1], a[b, i, k + 2, 2]],
            ],
            dt=qd.f64,
        )

    @qd.func
    def _st3(a: qd.template(), b, i, k: qd.template(), v):
        for r in range(3):
            for c in range(3):
                a[b, i, k + r, c] = v[r, c]

    @qd.func
    def _add3(a: qd.template(), b, i, k: qd.template(), v):
        for r in range(3):
            for c in range(3):
                a[b, i, k + r, c] = a[b, i, k + r, c] + v[r, c]

    @qd.func
    def _ldv(a: qd.template(), b, i, k: qd.template()):
        return qd.Vector([a[b, i, k, 0], a[b, i, k, 1], a[b, i, k, 2]], dt=qd.f64)

    @qd.func
    def _stv(a: qd.template(), b, i, k: qd.template(), v):
        for r in range(3):
            a[b, i, k, r] = v[r]

    @qd.func
    def _addv(a: qd.template(), b, i, k: qd.template(), v):
        for r in range(3):
            a[b, i, k, r] = a[b, i, k, r] + v[r]

    # S is slot-major -- (12, B, nj, 3) -- so that walking the tree for one
    # slot stays a 24-byte stride.  Interleaving the slots (B, nj, 12, 3) turns
    # the same walk into a 288-byte stride and costs 3-9% at B=4096 (measured).
    @qd.func
    def _ldvs(a: qd.template(), k: qd.template(), bn, i):
        m = bn + i
        return qd.Vector([a[k, m, 0], a[k, m, 1], a[k, m, 2]], dt=qd.f64)

    @qd.func
    def _stvs(a: qd.template(), k: qd.template(), bn, i, v):
        m = bn + i
        for r in range(3):
            a[k, m, r] = v[r]

    @qd.func
    def _addvs(a: qd.template(), k: qd.template(), bn, i, v):
        m = bn + i
        for r in range(3):
            a[k, m, r] = a[k, m, r] + v[r]

    # ---------------------------------------------------------------- helpers
    @qd.func
    def _pose(i, qi):
        """Joint-i placement relative to its parent."""
        R = fR[i]
        p = fpp[i]
        if fjt[i] == 0:  # revolute
            R = fR[i] @ rot_axis_angle(fax[i], qi)
        elif fjt[i] == 1:  # prismatic
            p = fpp[i] + fR[i] @ (fax[i] * qi)
        return R, p

    @qd.func
    def _xf_m(E, p, v1, v2):
        """Motion transform: parent -> child, given E = R^T."""
        return E @ v1, E @ (v2 - p.cross(v1))

    @qd.func
    def _xf_f(R, p, f1, f2):
        """Force transform: child -> parent."""
        n2 = R @ f2
        return R @ f1 + p.cross(n2), n2

    @qd.func
    def _inertia_mul(A, Bm, C, v1, v2):
        return A @ v1 + Bm @ v2, Bm.transpose() @ v1 + C @ v2

    @qd.func
    def _rows_cross(M, p):
        """M @ skew(p): row i of the result is M[i, :] x p."""
        R = qd.Matrix.zero(qd.f64, 3, 3)
        for i in range(3):
            w = qd.Vector([M[i, 0], M[i, 1], M[i, 2]], dt=qd.f64).cross(p)
            R[i, 0] = w[0]
            R[i, 1] = w[1]
            R[i, 2] = w[2]
        return R

    @qd.func
    def _cols_cross(M, p):
        """skew(p) @ M: column j of the result is p x M[:, j]."""
        R = qd.Matrix.zero(qd.f64, 3, 3)
        for j in range(3):
            w = p.cross(qd.Vector([M[0, j], M[1, j], M[2, j]], dt=qd.f64))
            R[0, j] = w[0]
            R[1, j] = w[1]
            R[2, j] = w[2]
        return R

    @qd.func
    def _skew(p):
        S = qd.Matrix.zero(qd.f64, 3, 3)
        S[0, 1] = -p[2]
        S[0, 2] = p[1]
        S[1, 0] = p[2]
        S[1, 2] = -p[0]
        S[2, 0] = -p[1]
        S[2, 1] = p[0]
        return S

    @qd.func
    def _blk_xf(A, Bm, C, T, p):
        """Blocks of X* [A B; B^T C] X for the placement (T, p).

        Only the three 3x3 conjugations are matrix products; the p-shift terms
        are cross products, which is a third of the flops of the dense 6x6 form.
        """
        Tt = T.transpose()
        At = T @ A @ Tt
        Bt = T @ Bm @ Tt
        Ct = T @ C @ Tt
        M1 = _rows_cross(Bt, p)  # Bt P
        M2 = _cols_cross(Ct, p)  # P Ct
        return At - M1 - M1.transpose() - _rows_cross(M2, p), Bt + M2, Ct

    @qd.func
    def _blk_xf_body(A0, mc, m, T, p):
        """X* I X for a *body* inertia, whose C block is m*1 and B is m c^."""
        Tt = T.transpose()
        At = T @ A0 @ Tt
        Bt = _skew(T @ mc)
        M1 = _rows_cross(Bt, p)
        I3 = qd.Matrix.identity(qd.f64, 3)
        pp = p.outer_product(p) - I3 * p.dot(p)
        return At - M1 - M1.transpose() - pp * m, Bt + _skew(p) * m, I3 * m

    @qd.func
    def _fk_rel(q: qd.template(), P: qd.template(), b, i):
        """Relative placement of joint i, stored for the sweeps."""
        R, p = _pose(i, q[b, fiv[i]])
        _st3(P, b, i, P_R, R)
        _stv(P, b, i, P_p, p)

    # ---------------------------------------------------------------- kernels
    @qd.kernel
    def k_fk(q: _F64_2, Ro: _F64_4, po: _F64_4, P: _F64_4, B: qd.i32):
        for b in range(B):
            _st3(Ro, b, 0, 0, qd.Matrix.identity(qd.f64, 3))
            _stv(po, b, 0, 0, qd.Vector.zero(qd.f64, 3))
            for i in range(1, nj):
                _fk_rel(q, P, b, i)
                _st3(Ro, b, i, 0, _ld3(Ro, b, fp[i], 0) @ _ld3(P, b, i, P_R))
                _stv(
                    po,
                    b,
                    i,
                    0,
                    _ldv(po, b, fp[i], 0) + _ld3(Ro, b, fp[i], 0) @ _ldv(P, b, i, P_p),
                )

    @qd.kernel
    def k_fk_rel(q: _F64_2, P: _F64_4, B: qd.i32):
        for b in range(B):
            for i in range(1, nj):
                _fk_rel(q, P, b, i)

    # ------------------------------------------------------------------ RNEA
    def _make_pass(use_accel):
        """Build one inverse-dynamics sweep.

        ``use_accel`` is a *Python* bool captured by the closure, not a kernel
        argument: the qdd terms are then traced out of the generated code for
        the bias-only variant instead of being branched around at run time.
        """

        @qd.func
        def _rnea_pass(
            b,
            q: qd.template(),
            qdin: qd.template(),
            qddin: qd.template(),
            tau: qd.template(),
            P: qd.template(),
            S: qd.template(),
        ):
            bn = b * nj
            _stvs(S, V1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, V2, bn, 0, qd.Vector.zero(qd.f64, 3))
            # gravity enters as a base acceleration of -g
            _stvs(S, A1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, A2, bn, 0, qd.Vector([-fg[0], -fg[1], -fg[2]], dt=qd.f64))
            for i in range(1, nj):
                _fk_rel(q, P, b, i)
                R = _ld3(P, b, i, P_R)
                p = _ldv(P, b, i, P_p)
                E = R.transpose()
                v1, v2 = _xf_m(E, p, _ldvs(S, V1, bn, fp[i]), _ldvs(S, V2, bn, fp[i]))
                a1, a2 = _xf_m(E, p, _ldvs(S, A1, bn, fp[i]), _ldvs(S, A2, bn, fp[i]))

                jt = fjt[i]
                ax = fax[i]
                # joint motion subspace: revolute spins, prismatic slides
                wj = qd.Vector.zero(qd.f64, 3)
                uj = qd.Vector.zero(qd.f64, 3)
                if jt == 0:
                    wj = ax * qdin[b, fiv[i]]
                    if use_accel:
                        a1 = a1 + ax * qddin[b, fiv[i]]
                elif jt == 1:
                    uj = ax * qdin[b, fiv[i]]
                    if use_accel:
                        a2 = a2 + ax * qddin[b, fiv[i]]
                v1 = v1 + wj
                v2 = v2 + uj
                # c = v x vJ  (motion cross product, no 6x6 operator)
                a1 = a1 + v1.cross(wj)
                a2 = a2 + v1.cross(uj) + v2.cross(wj)
                _stvs(S, V1, bn, i, v1)
                _stvs(S, V2, bn, i, v2)
                _stvs(S, A1, bn, i, a1)
                _stvs(S, A2, bn, i, a2)

                # f = I a + v x* (I v) in body coordinates
                Iv1 = fA0[i] @ v1 + fB0[i] @ v2
                Iv2 = fB0[i].transpose() @ v1 + fmass[i] * v2
                _stvs(
                    S,
                    F1,
                    bn,
                    i,
                    (fA0[i] @ a1 + fB0[i] @ a2 + v1.cross(Iv1) + v2.cross(Iv2)),
                )
                _stvs(
                    S,
                    F2,
                    bn,
                    i,
                    (fB0[i].transpose() @ a1 + fmass[i] * a2 + v1.cross(Iv2)),
                )
            for k in range(nj - 1):
                i = nj - 1 - k
                jt = fjt[i]
                if jt == 0:
                    tau[b, fiv[i]] = fax[i].dot(_ldvs(S, F1, bn, i))
                elif jt == 1:
                    tau[b, fiv[i]] = fax[i].dot(_ldvs(S, F2, bn, i))
                n1, n2 = _xf_f(
                    _ld3(P, b, i, P_R),
                    _ldv(P, b, i, P_p),
                    _ldvs(S, F1, bn, i),
                    _ldvs(S, F2, bn, i),
                )
                _addvs(S, F1, bn, fp[i], n1)
                _addvs(S, F2, bn, fp[i], n2)

        return _rnea_pass

    _pass_acc = _make_pass(True)
    _pass_bias = _make_pass(False)

    @qd.kernel
    def k_rnea(
        q: _F64_2,
        qdin: _F64_2,
        qddin: _F64_2,
        tau: _F64_2,
        P: _F64_4,
        S: _F64_3,
        B: qd.i32,
    ):
        for b in range(B):
            _pass_acc(b, q, qdin, qddin, tau, P, S)

    @qd.kernel
    def k_rnea_bias(
        q: _F64_2, qdin: _F64_2, tau: _F64_2, P: _F64_4, S: _F64_3, B: qd.i32
    ):
        for b in range(B):
            # qdin stands in for qddin, which is never traced in this variant
            _pass_bias(b, q, qdin, qdin, tau, P, S)

    # ------------------------------------------------------------------ CRBA
    @qd.func
    def _crba_ic(b, q: qd.template(), P: qd.template(), IC: qd.template()):
        """Relative poses + composite inertia for one instance.

        Shared by ``crba`` (which then fills M) and available to any variant
        that needs subtree sums: both want exactly this prefix.
        """
        for i in range(1, nj):
            _fk_rel(q, P, b, i)
        for i in range(nj):
            _st3(IC, b, i, IC_A, fA0[i])
            _st3(IC, b, i, IC_B, fB0[i])
            _st3(IC, b, i, IC_C, qd.Matrix.identity(qd.f64, 3) * fmass[i])
        for k in range(nj - 1):
            i = nj - 1 - k
            p_ = fp[i]
            A2 = qd.Matrix.zero(qd.f64, 3, 3)
            B2 = qd.Matrix.zero(qd.f64, 3, 3)
            C2 = qd.Matrix.zero(qd.f64, 3, 3)
            if f_leaf[i] == 1:
                A2, B2, C2 = _blk_xf_body(
                    fA0[i], f_mc[i], fmass[i], _ld3(P, b, i, P_R), _ldv(P, b, i, P_p)
                )
            else:
                A2, B2, C2 = _blk_xf(
                    _ld3(IC, b, i, IC_A),
                    _ld3(IC, b, i, IC_B),
                    _ld3(IC, b, i, IC_C),
                    _ld3(P, b, i, P_R),
                    _ldv(P, b, i, P_p),
                )
            _add3(IC, b, p_, IC_A, A2)
            _add3(IC, b, p_, IC_B, B2)
            _add3(IC, b, p_, IC_C, C2)

    @qd.kernel
    def k_crba(q: _F64_2, M: _F64_3, P: _F64_4, IC: _F64_4, B: qd.i32):
        """Composite inertia sweep + mass-matrix fill, fused.

        Fill is the force-propagation form: from every joint i, carry
        ``F = Ic_i S_i`` up the tree, one force transform and one dot product
        per (i, ancestor) pair.  That replaces an O(n^2) sweep of 6x6 matvecs
        and needs no per-pair transform composition.
        """
        for b in range(B):
            _crba_ic(b, q, P, IC)
            # ---- mass matrix
            # Joint pairs in different branches are structurally zero, so the
            # row sweep below never touches them; clear every row first instead
            # of paying for a separate full-matrix memset.  Rows that the fill
            # covers completely (every row, on a chain) skip the clearing --
            # at nv=63 that is 63 stores per instance, about as many as the
            # fill itself does.
            for i in range(1, nj):
                ii = fiv[i]
                if ii >= 0:
                    if f_row_full[i] == 0:
                        for j in range(nv):
                            M[b, ii, j] = 0.0
            for k in range(nj - 1):
                i = nj - 1 - k
                jt = fjt[i]
                ax = fax[i]
                # F = Ic_i S_i, S_i being the joint axis
                f1 = qd.Vector.zero(qd.f64, 3)
                f2 = qd.Vector.zero(qd.f64, 3)
                if jt == 0:
                    f1 = _ld3(IC, b, i, IC_A) @ ax
                    f2 = _ld3(IC, b, i, IC_B).transpose() @ ax
                elif jt == 1:
                    f1 = _ld3(IC, b, i, IC_B) @ ax
                    f2 = _ld3(IC, b, i, IC_C) @ ax
                if fiv[i] >= 0:
                    ii = fiv[i]
                    diag = qd.cast(0.0, qd.f64)
                    if jt == 0:
                        diag = ax.dot(f1)
                    elif jt == 1:
                        diag = ax.dot(f2)
                    M[b, ii, ii] = diag
                    j = i
                    while fp[j] != 0:
                        c = j
                        j = fp[j]
                        f1, f2 = _xf_f(_ld3(P, b, c, P_R), _ldv(P, b, c, P_p), f1, f2)
                        jj = fiv[j]
                        if jj >= 0:
                            val = qd.cast(0.0, qd.f64)
                            if fjt[j] == 0:
                                val = fax[j].dot(f1)
                            else:
                                val = fax[j].dot(f2)
                            M[b, ii, jj] = val
                            M[b, jj, ii] = val

    @qd.kernel
    def k_crba_apply(q: _F64_2, x: _F64_2, mx: _F64_2, P: _F64_4, S: _F64_3, B: qd.i32):
        """M(q) x without forming M.

        ``M x`` is inverse dynamics at qd = 0, g = 0, qdd = x: one forward
        velocity sweep and one backward force accumulation, exactly RNEA minus
        the v x* (I v) terms.  So this costs about one RNEA pass, while the
        filled form has to write an nv-long row per instance *on top of* the
        same two sweeps.

        Note it uses the *body* inertia, not the composite one: the sum over the
        subtree is what the backward sweep does.  (An earlier draft tried to
        summarise the tree in one forward pass with the composite inertia, which
        is wrong -- it drops the descendant columns of the row.)
        """
        for b in range(B):
            bn = b * nj
            _stvs(S, V1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, V2, bn, 0, qd.Vector.zero(qd.f64, 3))
            # ---- forward: joint velocity field with qd = x
            for i in range(1, nj):
                _fk_rel(q, P, b, i)
                E = _ld3(P, b, i, P_R).transpose()
                p = _ldv(P, b, i, P_p)
                w1, w2 = _xf_m(E, p, _ldvs(S, V1, bn, fp[i]), _ldvs(S, V2, bn, fp[i]))
                jt = fjt[i]
                ax = fax[i]
                if jt == 0:
                    w1 = w1 + ax * x[b, fiv[i]]
                elif jt == 1:
                    w2 = w2 + ax * x[b, fiv[i]]
                _stvs(S, V1, bn, i, w1)
                _stvs(S, V2, bn, i, w2)
                # body-inertia force; the subtree sum happens in the sweep below
                _stvs(S, F1, bn, i, fA0[i] @ w1 + fB0[i] @ w2)
                _stvs(S, F2, bn, i, fB0[i].transpose() @ w1 + fmass[i] * w2)
            # ---- backward: project on the axis, then hand the rest to the parent
            for k in range(nj - 1):
                i = nj - 1 - k
                jt = fjt[i]
                ax = fax[i]
                if jt == 0:
                    mx[b, fiv[i]] = ax.dot(_ldvs(S, F1, bn, i))
                elif jt == 1:
                    mx[b, fiv[i]] = ax.dot(_ldvs(S, F2, bn, i))
                n1, n2 = _xf_f(
                    _ld3(P, b, i, P_R),
                    _ldv(P, b, i, P_p),
                    _ldvs(S, F1, bn, i),
                    _ldvs(S, F2, bn, i),
                )
                _addvs(S, F1, bn, fp[i], n1)
                _addvs(S, F2, bn, fp[i], n2)

    # ------------------------------------------------------------------- ABA
    @qd.kernel
    def k_aba(
        q: _F64_2,
        qdin: _F64_2,
        tau: _F64_2,
        qdd: _F64_2,
        P: _F64_4,
        S: _F64_3,
        IC: _F64_4,
        D2: _F64_3,
        B: qd.i32,
    ):
        for b in range(B):
            bn = b * nj
            _stvs(S, V1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, V2, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, A1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, A2, bn, 0, qd.Vector([-fg[0], -fg[1], -fg[2]], dt=qd.f64))
            _stvs(S, PA1, bn, 0, qd.Vector.zero(qd.f64, 3))
            _stvs(S, PA2, bn, 0, qd.Vector.zero(qd.f64, 3))
            _st3(IC, b, 0, IC_A, qd.Matrix.zero(qd.f64, 3, 3))
            _st3(IC, b, 0, IC_B, qd.Matrix.zero(qd.f64, 3, 3))
            _st3(IC, b, 0, IC_C, qd.Matrix.zero(qd.f64, 3, 3))

            # pass 1: velocities, bias terms, body inertias
            for i in range(1, nj):
                _fk_rel(q, P, b, i)
                R = _ld3(P, b, i, P_R)
                p = _ldv(P, b, i, P_p)
                E = R.transpose()
                v1, v2 = _xf_m(E, p, _ldvs(S, V1, bn, fp[i]), _ldvs(S, V2, bn, fp[i]))
                jt = fjt[i]
                ax = fax[i]
                wj = qd.Vector.zero(qd.f64, 3)
                uj = qd.Vector.zero(qd.f64, 3)
                if jt == 0:
                    wj = ax * qdin[b, fiv[i]]
                elif jt == 1:
                    uj = ax * qdin[b, fiv[i]]
                v1 = v1 + wj
                v2 = v2 + uj
                _stvs(S, V1, bn, i, v1)
                _stvs(S, V2, bn, i, v2)
                _stvs(S, C1, bn, i, v1.cross(wj))
                _stvs(S, C2, bn, i, v1.cross(uj) + v2.cross(wj))

                Iv1 = fA0[i] @ v1 + fB0[i] @ v2
                Iv2 = fB0[i].transpose() @ v1 + fmass[i] * v2
                _stvs(S, PA1, bn, i, v1.cross(Iv1) + v2.cross(Iv2))
                _stvs(S, PA2, bn, i, v1.cross(Iv2))
                _st3(IC, b, i, IC_A, fA0[i])
                _st3(IC, b, i, IC_B, fB0[i])
                _st3(IC, b, i, IC_C, qd.Matrix.identity(qd.f64, 3) * fmass[i])

            # pass 2: articulated body inertia, backward
            for k in range(nj - 1):
                i = nj - 1 - k
                jt = fjt[i]
                ax = fax[i]
                A = _ld3(IC, b, i, IC_A)
                Bm = _ld3(IC, b, i, IC_B)
                C = _ld3(IC, b, i, IC_C)
                pa1 = _ldvs(S, PA1, bn, i)
                pa2 = _ldvs(S, PA2, bn, i)
                d_ = qd.cast(1.0, qd.f64)
                u_ = qd.cast(0.0, qd.f64)
                u1 = qd.Vector.zero(qd.f64, 3)
                u2 = qd.Vector.zero(qd.f64, 3)
                if jt == 0:
                    u1 = A @ ax
                    u2 = Bm.transpose() @ ax
                    d_ = ax.dot(u1)
                    u_ = tau[b, fiv[i]] - ax.dot(pa1)
                elif jt == 1:
                    u1 = Bm @ ax
                    u2 = C @ ax
                    d_ = ax.dot(u2)
                    u_ = tau[b, fiv[i]] - ax.dot(pa2)
                D2[b, i, 0] = d_
                D2[b, i, 1] = u_
                _stvs(S, U1, bn, i, u1)
                _stvs(S, U2, bn, i, u2)
                inv = qd.cast(1.0, qd.f64) / d_
                p_ = fp[i]
                c1 = _ldvs(S, C1, bn, i)
                c2 = _ldvs(S, C2, bn, i)
                if f_leaf[i] == 1:
                    # A leaf's IA *is* its body inertia, so the whole rank-1
                    # update can be carried through the transform instead of
                    # forming Ia first: X* (IA - U U^T/d) X
                    #   = X* IA X - (X* U)(X* U)^T / d
                    # (valid because the force transform equals X^T).
                    ac1 = fA0[i] @ c1 + fB0[i] @ c2
                    ac2 = fB0[i].transpose() @ c1 + fmass[i] * c2
                    k_ = (u_ - u1.dot(c1) - u2.dot(c2)) * inv
                    pn1 = pa1 + ac1 + u1 * k_
                    pn2 = pa2 + ac2 + u2 * k_
                    T2, B2, C2_ = _blk_xf_body(
                        fA0[i],
                        f_mc[i],
                        fmass[i],
                        _ld3(P, b, i, P_R),
                        _ldv(P, b, i, P_p),
                    )
                    t1, t2 = _xf_f(_ld3(P, b, i, P_R), _ldv(P, b, i, P_p), u1, u2)
                    _add3(IC, b, p_, IC_A, T2 - t1.outer_product(t1) * inv)
                    _add3(IC, b, p_, IC_B, B2 - t1.outer_product(t2) * inv)
                    _add3(IC, b, p_, IC_C, C2_ - t2.outer_product(t2) * inv)
                    n1, n2 = _xf_f(_ld3(P, b, i, P_R), _ldv(P, b, i, P_p), pn1, pn2)
                    _addvs(S, PA1, bn, p_, n1)
                    _addvs(S, PA2, bn, p_, n2)
                else:
                    # Ia = IA - U U^T / d
                    Aa = A - u1.outer_product(u1) * inv
                    Ba = Bm - u1.outer_product(u2) * inv
                    Ca = C - u2.outer_product(u2) * inv
                    # p^a = pA + Ia c + U (u / d)
                    ac1, ac2 = _inertia_mul(Aa, Ba, Ca, c1, c2)
                    k_ = u_ * inv
                    pn1 = pa1 + ac1 + u1 * k_
                    pn2 = pa2 + ac2 + u2 * k_
                    T2, B2, C2_ = _blk_xf(
                        Aa, Ba, Ca, _ld3(P, b, i, P_R), _ldv(P, b, i, P_p)
                    )
                    _add3(IC, b, p_, IC_A, T2)
                    _add3(IC, b, p_, IC_B, B2)
                    _add3(IC, b, p_, IC_C, C2_)
                    n1, n2 = _xf_f(_ld3(P, b, i, P_R), _ldv(P, b, i, P_p), pn1, pn2)
                    _addvs(S, PA1, bn, p_, n1)
                    _addvs(S, PA2, bn, p_, n2)

            # pass 3: accelerations, forward
            for i in range(1, nj):
                R = _ld3(P, b, i, P_R)
                p = _ldv(P, b, i, P_p)
                E = R.transpose()
                a1, a2 = _xf_m(E, p, _ldvs(S, A1, bn, fp[i]), _ldvs(S, A2, bn, fp[i]))
                a1 = a1 + _ldvs(S, C1, bn, i)
                a2 = a2 + _ldvs(S, C2, bn, i)
                # U^T a needs all six components: for a revolute joint U has a
                # non-zero linear block too (U = IA S with S = (ax, 0)).
                uTa = _ldvs(S, U1, bn, i).dot(a1) + _ldvs(S, U2, bn, i).dot(a2)
                jt = fjt[i]
                ax = fax[i]
                if jt == 0:
                    qi = (D2[b, i, 1] - uTa) / D2[b, i, 0]
                    _stvs(S, A1, bn, i, a1 + ax * qi)
                    _stvs(S, A2, bn, i, a2)
                    qdd[b, fiv[i]] = qi
                elif jt == 1:
                    qi = (D2[b, i, 1] - uTa) / D2[b, i, 0]
                    _stvs(S, A1, bn, i, a1)
                    _stvs(S, A2, bn, i, a2 + ax * qi)
                    qdd[b, fiv[i]] = qi
                else:
                    _stvs(S, A1, bn, i, a1)
                    _stvs(S, A2, bn, i, a2)

    return dict(
        fk=k_fk,
        fk_rel=k_fk_rel,
        rnea=k_rnea,
        rnea_bias=k_rnea_bias,
        crba=k_crba,
        crba_apply=k_crba_apply,
        aba=k_aba,
    )
