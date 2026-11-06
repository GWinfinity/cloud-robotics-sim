"""Batched FK and geometric Jacobian kernels.

Model parameters live in ``qd.field``s; per-batch scratch is plain numpy
buffers passed as ndarray arguments (the qd_rbd v4 packing lesson: field
identity is compile-time stable, ndarray scratch is per-Data).  All math is
f64.

Jacobian conventions, matching frax's MuJoCo-style floating base:

  v = [world linear velocity (3), body angular velocity (3), joint rates]
  Jlin maps v -> world velocity of the tracked frame origin
  Jang maps v -> world angular velocity of the tracked frame

Base columns: Jlin[:, 0:3] = I, Jlin[:, 3:6] = -skew(p - p_base) @ R_base,
Jang[:, 3:6] = R_base.
"""

import numpy as np
import quadrants as qd

from .spatial import quat_wxyz_to_rmat, rot_axis_angle, skew

_F64_2 = qd.types.ndarray(dtype=qd.f64, ndim=2)
_F64_3 = qd.types.ndarray(dtype=qd.f64, ndim=3)
_F64_4 = qd.types.ndarray(dtype=qd.f64, ndim=4)


class Data:
    """Per-batch scratch space.  Plain numpy buffers, no fields, so building
    one cannot trigger a compile.  Every buffer is overwritten by the kernel
    that uses it.
    """

    def __init__(self, model, batch, n_tracked):
        self.model = model
        self.B = batch
        nj, B = model.nj, batch
        self.Ro = np.zeros((B, nj, 3, 3))
        self.po = np.zeros((B, nj, 3))
        self.Jv = np.zeros((B, n_tracked, 3, model.nv))
        self.Jw = np.zeros((B, n_tracked, 3, model.nv))

    def __repr__(self):
        """Short summary with batch and dof counts."""
        return (
            f"<qd_ik.Data B={self.B} nj={self.model.nj} "
            f"nv={self.model.nv} tracked={self.Jv.shape[1]}>"
        )


def build(model, tracked):
    """Compile the FK/Jacobian kernel set for one (model, tracked-joints) pair.

    Args:
        model: :class:`~qd_ik.model.Model`
        tracked: sequence of joint indices whose frames get Jacobians
            (only used by the ``jac`` kernel; pass ``()`` for FK-only use)

    Returns:
        dict with ``fk`` and ``jac`` kernels.  Cached on the model.
    """
    key = ("kin", tuple(int(i) for i in tracked))
    cache = model.__dict__.setdefault("_qd_ik_kernels", {})
    if key in cache:
        return cache[key]

    nj, nv = model.nj, model.nv
    nt = len(tracked)
    floating = model.floating_base
    fp, fjt = model.f_parent, model.f_jtype
    fax, fR, fpp = model.f_axis, model.f_Rpl, model.f_ppl
    fiv, fiq = model.f_iv, model.f_iq
    # tracked joints as a field: k_jac subscripts it with a runtime loop index,
    # which a captured Python list cannot do.  Skipped for FK-only builds.
    f_tracked = None
    if nt > 0:
        f_tracked = qd.field(qd.i32, shape=nt)
        f_tracked.from_numpy(np.array(tracked, dtype=np.int32))

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
    def _ld3v(a: qd.template(), b, i):
        return qd.Matrix(
            [
                [a[b, i, 0, 0], a[b, i, 0, 1], a[b, i, 0, 2]],
                [a[b, i, 1, 0], a[b, i, 1, 1], a[b, i, 1, 2]],
                [a[b, i, 2, 0], a[b, i, 2, 1], a[b, i, 2, 2]],
            ],
            dt=qd.f64,
        )

    @qd.func
    def _st3v(a: qd.template(), b, i, v):
        for r in range(3):
            for c in range(3):
                a[b, i, r, c] = v[r, c]

    @qd.func
    def _stv(a: qd.template(), b, i, v):
        for r in range(3):
            a[b, i, r] = v[r]

    @qd.func
    def _ldv(a: qd.template(), b, i):
        return qd.Vector([a[b, i, 0], a[b, i, 1], a[b, i, 2]], dt=qd.f64)

    @qd.func
    def _rel_pose(i, qi):
        R = fR[i]
        p = fpp[i]
        if fjt[i] == 0:  # revolute
            R = fR[i] @ rot_axis_angle(fax[i], qi)
        elif fjt[i] == 1:  # prismatic
            p = fpp[i] + fR[i] @ (fax[i] * qi)
        return R, p

    @qd.func
    def _base_rot(q: qd.template(), b):
        if qd.static(floating):
            return quat_wxyz_to_rmat(
                qd.Vector([q[b, 3], q[b, 4], q[b, 5], q[b, 6]], dt=qd.f64)
            )
        else:
            return qd.Matrix.identity(qd.f64, 3)

    @qd.func
    def _base_p(q: qd.template(), b):
        if qd.static(floating):
            return qd.Vector([q[b, 0], q[b, 1], q[b, 2]], dt=qd.f64)
        else:
            return qd.Vector.zero(qd.f64, 3)

    @qd.func
    def _fk_all(q: qd.template(), Ro: qd.template(), po: qd.template(), b):
        _st3v(Ro, b, 0, _base_rot(q, b))
        _stv(po, b, 0, _base_p(q, b))
        for i in range(1, nj):
            R, p = _rel_pose(i, q[b, fiq[i]])
            _st3v(Ro, b, i, _ld3v(Ro, b, fp[i]) @ R)
            _stv(po, b, i, _ldv(po, b, fp[i]) + _ld3v(Ro, b, fp[i]) @ p)

    @qd.func
    def _frame_jac(q: qd.template(), Ro: qd.template(), po: qd.template(), b, i):
        """(Jlin, Jang): each (3, nv), maps v to (origin velocity, ang vel)."""
        Jl = qd.Matrix.zero(qd.f64, 3, nv)
        Ja = qd.Matrix.zero(qd.f64, 3, nv)
        pw = _ldv(po, b, i)
        if qd.static(floating):
            Rb = _base_rot(q, b)
            r = pw - _base_p(q, b)
            Ml = skew(r) @ Rb
            for cc in range(3):
                Jl[cc, cc] = qd.cast(1.0, qd.f64)
                for rr in range(3):
                    Jl[rr, 3 + cc] = -Ml[rr, cc]
                    Ja[rr, 3 + cc] = Rb[rr, cc]
        j = i
        while j != 0:
            iv = fiv[j]
            if iv >= 0:
                axw = _ld3v(Ro, b, j) @ fax[j]
                pj = _ldv(po, b, j)
                if fjt[j] == 0:  # revolute
                    col = axw.cross(pw - pj)
                    for rr in range(3):
                        Jl[rr, iv] = col[rr]
                        Ja[rr, iv] = axw[rr]
                else:  # prismatic
                    for rr in range(3):
                        Jl[rr, iv] = axw[rr]
            j = fp[j]
        return Jl, Ja

    @qd.kernel
    def k_fk(q: _F64_2, Ro: _F64_4, po: _F64_3, B: qd.i32):
        for b in range(B):
            _fk_all(q, Ro, po, b)

    @qd.kernel
    def k_jac(q: _F64_2, Jv: _F64_4, Jw: _F64_4, Ro: _F64_4, po: _F64_3, B: qd.i32):
        for b in range(B):
            _fk_all(q, Ro, po, b)
            for t in range(nt):
                Jl, Ja = _frame_jac(q, Ro, po, b, f_tracked[t])
                for rr in range(3):
                    for cc in range(nv):
                        Jv[b, t, rr, cc] = Jl[rr, cc]
                        Jw[b, t, rr, cc] = Ja[rr, cc]

    cache[key] = dict(fk=k_fk, jac=k_jac)
    return cache[key]


def forward_kinematics(model, data, q):
    """World-frame rotation and translation of every joint, per instance.

    Args:
        model: :class:`~qd_ik.model.Model`
        data: scratch buffers sized by batch
        q: (B, nq) configuration array

    Returns:
        (R, p): (B, nj, 3, 3) and (B, nj, 3)
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.shape != (data.B, model.nq):
        raise ValueError(f"expected q shape {(data.B, model.nq)}, got {q.shape}")
    build(model, ())["fk"](q, data.Ro, data.po, data.B)
    return data.Ro, data.po


def frame_jacobians(model, data, q, tracked):
    """Geometric Jacobians of the tracked joint frames, per instance.

    Args:
        model: :class:`~qd_ik.model.Model`
        data: scratch buffers sized by batch
        q: (B, nq) configuration array
        tracked: sequence of joint indices (must match the Data's build)

    Returns:
        (Jv, Jw): (B, nt, 3, nv) each
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.shape != (data.B, model.nq):
        raise ValueError(f"expected q shape {(data.B, model.nq)}, got {q.shape}")
    tracked = [int(i) for i in tracked]
    if data.Jv.shape[1] != len(tracked):
        raise ValueError("Data was built for a different number of tracked joints")
    build(model, tracked)["jac"](q, data.Jv, data.Jw, data.Ro, data.po, data.B)
    return data.Jv, data.Jw
