"""FK and geometric Jacobian kernel tests (numpy references + finite diffs)."""

import numpy as np

from plugins.solvers.qd_ik import kinematics as kin
from plugins.solvers.qd_ik.model import FIX, PRIS, REV, Model
from plugins.solvers.qd_ik.spatial import quat_wxyz_exp, quat_wxyz_mul


def _random_model(seed, floating):
    rng = np.random.default_rng(seed)
    nj = 8
    parents = np.array([0, 0, 1, 2, 3, 1, 5, 5], dtype=np.int32)
    jtype = np.array([FIX, REV, REV, PRIS, REV, FIX, REV, REV], dtype=np.int32)
    axis = rng.normal(size=(nj, 3))
    axis[0] = [1.0, 0, 0]
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)

    def rnd_rot():
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        return Q

    return Model(
        parents=parents,
        jtype=jtype,
        axis=axis,
        Rpl=np.stack([rnd_rot() for _ in range(nj)]),
        ppl=np.vstack([np.zeros(3), rng.normal(scale=0.2, size=(nj - 1, 3))]),
        floating_base=floating,
    )


def _quat2rmat(qw):
    w, x, y, z = qw / np.linalg.norm(qw)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _rodrigues(a, th):
    Km = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * Km + (1 - np.cos(th)) * Km @ Km


def _ref_fk(model, q):
    nj = model.nj
    Ro = np.zeros((nj, 3, 3))
    po = np.zeros((nj, 3))
    Ro[0] = _quat2rmat(q[3:7]) if model.floating_base else np.eye(3)
    po[0] = q[0:3] if model.floating_base else 0.0
    for i in range(1, nj):
        qi = q[model.iq[i]] if model.iq[i] >= 0 else 0.0
        if model.jtype[i] == REV:
            R = model.Rpl[i] @ _rodrigues(model.axis[i], qi)
            p = model.ppl[i]
        elif model.jtype[i] == PRIS:
            R = model.Rpl[i]
            p = model.ppl[i] + model.Rpl[i] @ (model.axis[i] * qi)
        else:
            R, p = model.Rpl[i], model.ppl[i]
        Ro[i] = Ro[model.parents[i]] @ R
        po[i] = po[model.parents[i]] + Ro[model.parents[i]] @ p
    return Ro, po


def test_fk_exact_both_bases():
    """FK matches a numpy reference to machine precision, both base types."""
    for floating in (False, True):
        model = _random_model(7, floating)
        rng = np.random.default_rng(1)
        q = rng.normal(scale=0.3, size=(3, model.nq))
        if floating:
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        data = kin.Data(model, 3, n_tracked=1)
        Ro, po = kin.forward_kinematics(model, data, q)
        for b in range(3):
            Rr, pr = _ref_fk(model, q[b])
            assert np.allclose(Ro[b], Rr, atol=1e-12)
            assert np.allclose(po[b], pr, atol=1e-12)


def test_jacobian_finite_differences():
    """Jacobians match central finite differences of FK."""
    eps = 1e-7
    tracked = list(range(1, 8))
    for floating in (False, True):
        model = _random_model(7, floating)
        rng = np.random.default_rng(2)
        q = rng.normal(scale=0.3, size=(2, model.nq))
        if floating:
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        data = kin.Data(model, 2, n_tracked=len(tracked))
        Jv, Jw = kin.frame_jacobians(model, data, q, tracked)
        Jv, Jw = Jv.copy(), Jw.copy()  # buffers are reused by FK calls below
        data1 = kin.Data(model, 1, n_tracked=len(tracked))
        worst = 0.0
        for b in range(2):
            for t, i in enumerate(tracked):
                Jv_fd = np.zeros((3, model.nv))
                Jw_fd = np.zeros((3, model.nv))
                for c in range(model.nv):
                    if floating and c < 3:
                        dq = np.zeros(model.nq)
                        dq[c] = eps
                        qp, qm = q[b] + dq, q[b] - dq
                    elif floating and c < 6:
                        e = np.zeros(3)
                        e[c - 3] = eps
                        qp = q[b].copy()
                        qm = q[b].copy()
                        qp[3:7] = quat_wxyz_mul(q[b, 3:7], quat_wxyz_exp(e))
                        qm[3:7] = quat_wxyz_mul(q[b, 3:7], quat_wxyz_exp(-e))
                    else:
                        dq = np.zeros(model.nq)
                        dq[(7 if floating else 0) + (c - 6 if floating else c)] = eps
                        qp, qm = q[b] + dq, q[b] - dq
                    if floating and c >= 3:
                        qp, qm = qp.copy(), qm.copy()
                        qp[3:7] /= np.linalg.norm(qp[3:7])
                        qm[3:7] /= np.linalg.norm(qm[3:7])
                    Ro_p, po_p = kin.forward_kinematics(model, data1, qp[None])
                    Ro_p, po_p = Ro_p.copy(), po_p.copy()
                    Ro_m, po_m = kin.forward_kinematics(model, data1, qm[None])
                    Jv_fd[:, c] = (po_p[0, i] - po_m[0, i]) / (2 * eps)
                    dR = (Ro_p[0, i] - Ro_m[0, i]) / (2 * eps) @ Ro_p[0, i].T
                    Jw_fd[:, c] = [dR[2, 1], dR[0, 2], dR[1, 0]]
                worst = max(
                    worst,
                    np.abs(Jv[b, t] - Jv_fd).max(),
                    np.abs(Jw[b, t] - Jw_fd).max(),
                )
        assert worst < 1e-5, worst
