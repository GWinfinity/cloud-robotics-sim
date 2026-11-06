"""DiffIKSolver: the frax.ik.solver DiffIKSolver port.

Per frame, and batched over B independent problems:

    min  sum_tasks || w (J v - v_des) ||^2 + || diag(reg + damping*cost) v ||^2
    s.t. G(q) v <= h(q)          (elastic, L1-penalized -- see constraints.py)

built and solved by two kernels: ``k_build`` (FK + Jacobians + QP data) and
``qp.solve``.  Integration (quaternion base exp map included) runs on the
host in numpy -- one line of vector math per frame, not worth a kernel.

The task/constraint set is fixed at construction and compiled into the
kernels; per-frame reference data arrives as a dict of (B, ...) numpy arrays
whose keys the tasks declare.
"""

import numpy as np
import quadrants as qd

from . import kinematics as kin_mod
from . import qp as qp_mod
from .constraints import FloorConstraint, JointLimits, SelfCollisionConstraint
from .spatial import (
    orientation_error,
    quat_wxyz_exp,
    quat_wxyz_mul,
    quat_wxyz_to_rmat,
    rot_axis_angle,
    skew,
)

_F64_1 = qd.types.ndarray(dtype=qd.f64, ndim=1)
_F64_2 = qd.types.ndarray(dtype=qd.f64, ndim=2)
_F64_3 = qd.types.ndarray(dtype=qd.f64, ndim=3)
_F64_4 = qd.types.ndarray(dtype=qd.f64, ndim=4)

_GAIN_ATTRS = ("gain_pos", "gain_rot", "gain")


def _max_gain(tasks, constraints) -> float:
    gains = []
    for term in list(tasks) + list(constraints):
        for a in _GAIN_ATTRS:
            if hasattr(term, a):
                gains.append(float(np.max(getattr(term, a))))
    if not gains:
        raise ValueError(
            "no gain found on any task or constraint; pass converge_dt explicitly"
        )
    return max(gains)


class Data:
    """Per-batch scratch for one DiffIKSolver.  Plain numpy buffers."""

    def __init__(self, solver, batch):
        self.solver = solver
        self.B = batch
        model = solver.model
        nv, m = model.nv, solver.m_total
        nj, ns = model.nj, solver.n_spheres
        nt = solver.n_tracked
        self.Ro = np.zeros((batch, nj, 3, 3))
        self.po = np.zeros((batch, nj, 3))
        self.Jv = np.zeros((batch, nt, 3, nv))
        self.Jw = np.zeros((batch, nt, 3, nv))
        self.sp = np.zeros((batch, ns, 3))
        self.SJ = np.zeros((batch, ns, 3, nv)) if ns else np.zeros((batch, 0, 3, nv))
        n_sc_all = solver.self_collision.num_pairs if solver.self_collision else 0
        self.HP = np.zeros((batch, n_sc_all))
        self.SCU = np.zeros((batch, n_sc_all))
        self.P = np.zeros((batch, nv, nv))
        self.c = np.zeros((batch, nv))
        self.G = np.zeros((batch, m, nv))
        self.h = np.zeros((batch, m))
        self.cost = np.zeros(batch)
        self.qp = qp_mod.QPData(nv, m, batch, solver.qp_config)

    def __repr__(self):
        """Short summary with batch, dof and row counts."""
        return (
            f"<qd_ik.Data B={self.B} nv={self.solver.model.nv} "
            f"m={self.solver.m_total}>"
        )


class DiffIKSolver:
    """Velocity-level differential IK with elastic CBF constraints.

    Args:
        model: :class:`~qd_ik.model.Model`
        tasks: non-empty sequence of FrameTask / JointTask
        constraints: sequence of JointLimits / FloorConstraint /
            SelfCollisionConstraint (at most one of each kind)
        regularization: Tikhonov per-dof weight, scalar or (nv,)
        damping: Levenberg-Marquardt damping on the task cost
        qp_config: :class:`~qd_ik.qp.QPConfig`
    """

    def __init__(
        self,
        model,
        tasks,
        constraints=(),
        regularization=1e-8,
        damping=0.0,
        qp_config=None,
    ):
        if not tasks:
            raise ValueError("at least one task is required")
        self.model = model
        self.tasks = list(tasks)
        self.constraints = list(constraints)
        self.damping = float(damping)
        nv = model.nv
        self.regularization = np.broadcast_to(
            np.asarray(regularization, dtype=np.float64), (nv,)
        ).copy()

        kinds = [type(c).__name__ for c in self.constraints]
        if len(kinds) != len(set(kinds)):
            raise ValueError("at most one constraint of each kind is supported")
        self.joint_limits = self._pick(self.constraints, JointLimits)
        self.floor = self._pick(self.constraints, FloorConstraint)
        self.self_collision = self._pick(self.constraints, SelfCollisionConstraint)
        if self.floor is not None and self.floor.spheres is not None:
            pass
        # all constraints must share the solver's sphere model if they use one
        self.spheres = None
        for con in (self.floor, self.self_collision):
            if con is not None:
                if self.spheres is None:
                    self.spheres = con.spheres
                elif con.spheres is not self.spheres:
                    raise ValueError(
                        "floor and self-collision must share one SphereModel"
                    )
        self.n_spheres = len(self.spheres) if self.spheres is not None else 0

        # ---- tracked frames (FrameTasks), joint-task rows, ref assembly plan
        self.n_tracked = 0
        self.frame_tasks = []
        for t in self.tasks:
            if hasattr(t, "pos_key"):  # FrameTask
                self.frame_tasks.append((self.n_tracked, t))
                self.n_tracked += t.n
        self.n_jtask = 0
        for t in self.tasks:
            if hasattr(t, "ref_key"):  # JointTask
                self.n_jtask += t.n

        # ---- task row table (compile-time arrays -> fields)
        rows = []  # (type, frame, dof, comp, w, gain, jidx, qidx)
        for tbase, t in self.frame_tasks:
            for k in range(t.n):
                for comp in range(3):
                    if t.pos_key is not None:
                        rows.append(
                            (
                                0,
                                tbase + k,
                                -1,
                                comp,
                                t.weight_pos[k],
                                t.gain_pos[k],
                                -1,
                                -1,
                            )
                        )
            for k in range(t.n):
                for comp in range(3):
                    if t.rot_key is not None:
                        rows.append(
                            (
                                1,
                                tbase + k,
                                -1,
                                comp,
                                t.weight_rot[k],
                                t.gain_rot[k],
                                -1,
                                -1,
                            )
                        )
        jt = 0
        for t in self.tasks:
            if hasattr(t, "ref_key"):
                for k, j in enumerate(t.joints):
                    rows.append(
                        (
                            2,
                            -1,
                            model.iv[j],
                            0,
                            t.weight[k],
                            t.gain[k],
                            jt + k,
                            model.iq[j],
                        )
                    )
                jt += t.n
        if not rows:
            raise ValueError("tasks contributed zero rows")
        self.n_task_rows = len(rows)
        self._row_type = np.array([r[0] for r in rows], dtype=np.int32)
        self._row_frame = np.array([r[1] for r in rows], dtype=np.int32)
        self._row_dof = np.array([r[2] for r in rows], dtype=np.int32)
        self._row_comp = np.array([r[3] for r in rows], dtype=np.int32)
        self._row_w = np.array([r[4] for r in rows])
        self._row_gain = np.array([r[5] for r in rows])
        self._row_jidx = np.array([r[6] for r in rows], dtype=np.int32)
        self._row_qi = np.array([r[7] for r in rows], dtype=np.int32)

        # ---- constraint row layout
        self.m_lim = self.joint_limits.num_rows if self.joint_limits else 0
        self.m_floor = self.floor.num_rows if self.floor else 0
        self.m_sc = self.self_collision.num_rows if self.self_collision else 0
        self.m_total = self.m_lim + self.m_floor + self.m_sc
        penalties = []
        for con in (self.joint_limits, self.floor, self.self_collision):
            if con is not None:
                penalties.append(con.penalty_vector())
        self.penalty = np.concatenate(penalties) if penalties else np.zeros(0)

        self.max_gain = _max_gain(self.tasks, self.constraints)
        if qp_config is None:
            qp_config = qp_mod.QPConfig(rho=self._auto_rho())
        self.qp_config = qp_config
        self._kernels = None

    def _auto_rho(self):
        """ADMM penalty balancing the elastic rows against the task Hessian.

        The v-step curvature is ``P + rho * pen * G'G``; for it not to drown
        the task term, rho * mean(penalty) should sit near the scale of
        diag(P) -- estimated with one Jacobian evaluation at the default
        configuration.  Measured effect on a badly-scaled diff-IK QP: rho by
        this formula lands within 10x of the best value in a 1e4 sweep.
        """
        if self.m_total == 0:
            return 10.0
        tracked = []
        for _, t in self.frame_tasks:
            tracked.extend(t.joints)
        scale = float(np.mean(self.regularization))
        if tracked:
            model = self.model
            q0 = model.default_configuration()
            kdata = kin_mod.Data(model, 1, max(len(tracked), 1))
            Jv, Jw = kin_mod.frame_jacobians(model, kdata, q0, tracked)
            off = 0
            acc = np.zeros(model.nv)
            for _, t in self.frame_tasks:
                w2 = np.zeros(t.n)
                if t.pos_key is not None:
                    w2 += t.weight_pos**2
                if t.rot_key is not None:
                    w2 += t.weight_rot**2
                for k in range(t.n):
                    acc += w2[k] * (
                        np.einsum("ri,ri->i", Jv[0, off + k], Jv[0, off + k])
                        + np.einsum("ri,ri->i", Jw[0, off + k], Jw[0, off + k])
                    )
                off += t.n
            scale += float(np.mean(acc))
        pen_mean = float(np.mean(self.penalty))
        rho = scale / pen_mean
        return float(np.clip(rho, 1e-6, 1e4))

    @staticmethod
    def _pick(constraints, kind):
        for c in constraints:
            if isinstance(c, kind):
                return c
        return None

    # ---------------------------------------------------------------- kernels
    def kernels(self):
        if self._kernels is None:
            self._kernels = _build(self)
        return self._kernels

    def data(self, batch):
        return Data(self, batch)

    # ------------------------------------------------------------------- ref
    def _assemble_ref(self, ref, B):
        """Stack per-task ref dict entries into canonical target arrays."""
        tgt_pos = np.zeros((B, self.n_tracked, 3))
        tgt_rmat = np.zeros((B, self.n_tracked, 3, 3))
        tgt_jnt = np.zeros((B, self.n_jtask))
        for tbase, t in self.frame_tasks:
            n = t.n
            if t.pos_key is not None:
                p = np.asarray(ref[t.pos_key], dtype=np.float64)
                if p.shape != (B, n, 3):
                    raise ValueError(
                        f"ref[{t.pos_key!r}] has shape {p.shape}, want {(B, n, 3)}"
                    )
                tgt_pos[:, tbase : tbase + n] = p
            if t.rot_key is not None:
                r = np.asarray(ref[t.rot_key], dtype=np.float64)
                if r.shape != (B, n, 3, 3):
                    raise ValueError(
                        f"ref[{t.rot_key!r}] has shape {r.shape}, want {(B, n, 3, 3)}"
                    )
                tgt_rmat[:, tbase : tbase + n] = r
        jt = 0
        for t in self.tasks:
            if hasattr(t, "ref_key"):
                v = np.asarray(ref[t.ref_key], dtype=np.float64)
                if v.shape != (B, t.n):
                    raise ValueError(
                        f"ref[{t.ref_key!r}] has shape {v.shape}, want {(B, t.n)}"
                    )
                tgt_jnt[:, jt : jt + t.n] = v
                jt += t.n
        return tgt_pos, tgt_rmat, tgt_jnt

    # ------------------------------------------------------------------- api
    def qp_data(self, q, ref, data: Data):
        """Build the QP at q.  Returns (P, c, G, h, cost) views into data."""
        q = np.ascontiguousarray(q, dtype=np.float64)
        B, model = data.B, self.model
        if q.shape != (B, model.nq):
            raise ValueError(f"q has shape {q.shape}, want {(B, model.nq)}")
        tgt_pos, tgt_rmat, tgt_jnt = self._assemble_ref(ref, B)
        k = self.kernels()["build"]
        k(
            q,
            tgt_pos,
            tgt_rmat,
            tgt_jnt,
            data.P,
            data.c,
            data.G,
            data.h,
            data.cost,
            data.Ro,
            data.po,
            data.Jv,
            data.Jw,
            data.sp,
            data.SJ,
            data.HP,
            data.SCU,
            B,
        )
        return data.P, data.c, data.G, data.h, data.cost

    def solve_v(self, q, ref, data: Data, warm=True):
        """Build the QP at q and solve it.  Returns v (B, nv) (data.qp.v)."""
        P, c, G, h, _ = self.qp_data(q, ref, data)
        return qp_mod.solve(P, c, G, h, self.penalty, data.qp, warm=warm)

    def cost(self, q, ref, data: Data):
        """The solver cost at q: task error plus elastic violation penalties."""
        *_, cost = self.qp_data(q, ref, data)
        return cost

    def step(self, q, ref, dt, data: Data, warm=True):
        """One diff-IK step.  Returns the integrated (B, nq) configuration."""
        self.solve_v(q, ref, data, warm=warm)
        return integrate(self.model, q, data.qp.v, dt)


def integrate(model, q, v, dt):
    """Integrate configuration with a constant velocity over dt (host, numpy).

    Quaternion base follows frax/MuJoCo: position integrates the world-frame
    linear velocity, the quaternion integrates the body-frame angular
    velocity through the exp map, actuated joints add v*dt directly.
    """
    q = np.array(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.shape[0] != v.shape[0]:
        raise ValueError("q and v batch dimensions differ")
    if np.ndim(dt) > 0:
        dt = np.asarray(dt, dtype=np.float64).reshape(-1, 1)
    if not model.floating_base:
        return q + v * dt
    q[:, 0:3] += v[:, 0:3] * dt
    dq = quat_wxyz_exp(v[:, 3:6] * dt)
    quat = quat_wxyz_mul(q[:, 3:7], dq)
    q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    q[:, 7:] += v[:, 6:] * dt
    return q


# --------------------------------------------------------------------- build
def _build(solver: DiffIKSolver):
    """Compile the fused QP-build kernel for one solver configuration."""
    model = solver.model
    nj, nv = model.nj, model.nv
    nt = solver.n_tracked
    ns = solver.n_spheres
    ntr = solver.n_task_rows
    m_lim, m_floor, m_sc = solver.m_lim, solver.m_floor, solver.m_sc
    m = solver.m_total
    floating = model.floating_base
    sc = solver.self_collision
    sc_n_all = sc.num_pairs if sc is not None else 0
    sc_topk = bool(sc is not None and sc.max_rows is not None)

    fp, fjt = model.f_parent, model.f_jtype
    fax, fR, fpp = model.f_axis, model.f_Rpl, model.f_ppl
    fiv, fiq = model.f_iv, model.f_iq

    f_tracked = None
    if nt > 0:
        tracked = []
        for _, t in solver.frame_tasks:
            tracked.extend(t.joints)
        f_tracked = qd.field(qd.i32, shape=nt)
        f_tracked.from_numpy(np.array(tracked, dtype=np.int32))

    f_rt = qd.field(qd.i32, shape=ntr)
    f_rt.from_numpy(solver._row_type)
    f_rf = qd.field(qd.i32, shape=ntr)
    f_rf.from_numpy(solver._row_frame)
    f_rd = qd.field(qd.i32, shape=ntr)
    f_rd.from_numpy(solver._row_dof)
    f_rc = qd.field(qd.i32, shape=ntr)
    f_rc.from_numpy(solver._row_comp)
    f_rw = qd.field(qd.f64, shape=ntr)
    f_rw.from_numpy(solver._row_w)
    f_rg = qd.field(qd.f64, shape=ntr)
    f_rg.from_numpy(solver._row_gain)
    f_rj = qd.field(qd.i32, shape=ntr)
    f_rj.from_numpy(solver._row_jidx)
    f_rq = qd.field(qd.i32, shape=ntr)
    f_rq.from_numpy(solver._row_qi)

    f_reg = qd.field(qd.f64, shape=nv)
    f_reg.from_numpy(solver.regularization)
    f_pen = qd.field(qd.f64, shape=m) if m else None
    if m:
        f_pen.from_numpy(solver.penalty)

    # limit-row fields: dof (v-index), q-index, bounds, upper flag
    fl = solver.joint_limits
    f_lim_dof = f_lim_q = f_lim_up = None
    f_lim_qmin = f_lim_qmax = f_lim_vmax = f_lim_g = None
    if fl is not None:
        base = 6 if floating else 0
        dofs, ups = [], []
        qmins, qmaxs, vmaxs, gains = [], [], [], []
        # upper rows first (offset 0), then lower rows -- matches layout below
        for d in fl.upper_dof:
            k = int(d) - base
            dofs.append(d)
            ups.append(1)
            qmins.append(fl.q_min[k])
            qmaxs.append(fl.q_max[k])
            vmaxs.append(fl.v_max[k])
            gains.append(fl.gain)
        for d in fl.lower_dof:
            k = int(d) - base
            dofs.append(d)
            ups.append(0)
            qmins.append(fl.q_min[k])
            qmaxs.append(fl.q_max[k])
            vmaxs.append(fl.v_max[k])
            gains.append(fl.gain)
        qidx = [int(fiq[int(np.argmax(model.iv == d))]) for d in dofs]
        f_lim_dof = qd.field(qd.i32, shape=m_lim)
        f_lim_dof.from_numpy(np.array(dofs, dtype=np.int32))
        f_lim_q = qd.field(qd.i32, shape=m_lim)
        f_lim_q.from_numpy(np.array(qidx, dtype=np.int32))
        f_lim_up = qd.field(qd.i32, shape=m_lim)
        f_lim_up.from_numpy(np.array(ups, dtype=np.int32))
        f_lim_qmin = qd.field(qd.f64, shape=m_lim)
        f_lim_qmin.from_numpy(np.array(qmins))
        f_lim_qmax = qd.field(qd.f64, shape=m_lim)
        f_lim_qmax.from_numpy(np.array(qmaxs))
        f_lim_vmax = qd.field(qd.f64, shape=m_lim)
        f_lim_vmax.from_numpy(np.array(vmaxs))
        f_lim_g = qd.field(qd.f64, shape=m_lim)
        f_lim_g.from_numpy(np.array(gains))

    # sphere fields
    f_sp_link = f_sp_off = f_sp_rad = None
    ff_sph = f_fg = f_fm = None
    fsc_a = fsc_b = fsc_tol = fsc_g = None
    if ns > 0:
        sp = solver.spheres
        f_sp_link = qd.field(qd.i32, shape=ns)
        f_sp_link.from_numpy(sp.links)
        f_sp_off = qd.Vector.field(3, qd.f64, shape=ns)
        f_sp_off.from_numpy(sp.offsets)
        f_sp_rad = qd.field(qd.f64, shape=ns)
        f_sp_rad.from_numpy(sp.radii)
    if solver.floor is not None:
        fc = solver.floor
        ff_sph = qd.field(qd.i32, shape=m_floor)
        ff_sph.from_numpy(np.asarray(fc.sphere_idxs, dtype=np.int32))
        f_fg = qd.field(qd.f64, shape=())
        f_fg.from_numpy(np.array(fc.gain))
        f_fm = qd.field(qd.f64, shape=())
        f_fm.from_numpy(np.array(fc.margin))
    if solver.self_collision is not None:
        sc_cfg = solver.self_collision
        fsc_a = qd.field(qd.i32, shape=sc_n_all)
        fsc_a.from_numpy(sc_cfg.pairs[:, 0].copy())
        fsc_b = qd.field(qd.i32, shape=sc_n_all)
        fsc_b.from_numpy(sc_cfg.pairs[:, 1].copy())
        fsc_tol = qd.field(qd.f64, shape=sc_n_all)
        fsc_tol.from_numpy(sc_cfg.tols + sc_cfg.margin)
        fsc_g = qd.field(qd.f64, shape=())
        fsc_g.from_numpy(np.array(sc.gain))

    damping = float(solver.damping)

    @qd.func
    def _ld3(a: qd.template(), b, i):
        return qd.Matrix(
            [
                [a[b, i, 0, 0], a[b, i, 0, 1], a[b, i, 0, 2]],
                [a[b, i, 1, 0], a[b, i, 1, 1], a[b, i, 1, 2]],
                [a[b, i, 2, 0], a[b, i, 2, 1], a[b, i, 2, 2]],
            ],
            dt=qd.f64,
        )

    @qd.func
    def _rel_pose(i, qi):

        R = fR[i]
        p = fpp[i]
        if fjt[i] == 0:
            R = fR[i] @ rot_axis_angle(fax[i], qi)
        elif fjt[i] == 1:
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
    def _fk_all(q: qd.template(), Ro: qd.template(), po: qd.template(), b):
        if qd.static(floating):
            _st3(Ro, b, 0, _base_rot(q, b))
            for r in range(3):
                po[b, 0, r] = q[b, r]
        else:
            _st3(Ro, b, 0, qd.Matrix.identity(qd.f64, 3))
            for r in range(3):
                po[b, 0, r] = 0.0
        for i in range(1, nj):
            R, p = _rel_pose(i, q[b, fiq[i]])
            Rp = _ld3(Ro, b, fp[i])
            _st3(Ro, b, i, Rp @ R)
            pp = qd.Vector(
                [po[b, fp[i], 0], po[b, fp[i], 1], po[b, fp[i], 2]], dt=qd.f64
            )
            pw = pp + Rp @ p
            for r in range(3):
                po[b, i, r] = pw[r]

    @qd.func
    def _st3(a: qd.template(), b, i, v):
        for r in range(3):
            for c in range(3):
                a[b, i, r, c] = v[r, c]

    @qd.func
    def _frame_jac(
        q: qd.template(),
        Ro: qd.template(),
        po: qd.template(),
        b,
        i,
        Jv: qd.template(),
        Jw: qd.template(),
        t,
    ):

        pw = qd.Vector([po[b, i, 0], po[b, i, 1], po[b, i, 2]], dt=qd.f64)
        for rr in range(3):
            for cc in range(nv):
                Jv[b, t, rr, cc] = 0.0
                Jw[b, t, rr, cc] = 0.0
        if qd.static(floating):
            Rb = _base_rot(q, b)
            pb = qd.Vector([q[b, 0], q[b, 1], q[b, 2]], dt=qd.f64)
            r = pw - pb
            Ml = skew(r) @ Rb
            for cc in range(3):
                Jv[b, t, cc, cc] = 1.0
                for rr in range(3):
                    Jv[b, t, rr, 3 + cc] = -Ml[rr, cc]
                    Jw[b, t, rr, 3 + cc] = Rb[rr, cc]
        j = i
        while j != 0:
            iv = fiv[j]
            if iv >= 0:
                axw = _ld3(Ro, b, j) @ fax[j]
                pj = qd.Vector([po[b, j, 0], po[b, j, 1], po[b, j, 2]], dt=qd.f64)
                if fjt[j] == 0:
                    col = axw.cross(pw - pj)
                    for rr in range(3):
                        Jv[b, t, rr, iv] = col[rr]
                        Jw[b, t, rr, iv] = axw[rr]
                else:
                    for rr in range(3):
                        Jv[b, t, rr, iv] = axw[rr]
            j = fp[j]

    @qd.func
    def _point_jac(
        q: qd.template(),
        Ro: qd.template(),
        po: qd.template(),
        b,
        i,
        pw,
        SJ: qd.template(),
        s,
    ):
        """Linear (3, nv) Jacobian of a point rigidly attached to joint i."""
        for rr in range(3):
            for cc in range(nv):
                SJ[b, s, rr, cc] = 0.0
        if qd.static(floating):
            Rb = _base_rot(q, b)
            pb = qd.Vector([q[b, 0], q[b, 1], q[b, 2]], dt=qd.f64)
            r = pw - pb
            Ml = skew(r) @ Rb
            for cc in range(3):
                SJ[b, s, cc, cc] = 1.0
                for rr in range(3):
                    SJ[b, s, rr, 3 + cc] = -Ml[rr, cc]
        j = i
        while j != 0:
            iv = fiv[j]
            if iv >= 0:
                axw = _ld3(Ro, b, j) @ fax[j]
                pj = qd.Vector([po[b, j, 0], po[b, j, 1], po[b, j, 2]], dt=qd.f64)
                if fjt[j] == 0:
                    col = axw.cross(pw - pj)
                    for rr in range(3):
                        SJ[b, s, rr, iv] = col[rr]
                else:
                    for rr in range(3):
                        SJ[b, s, rr, iv] = axw[rr]
            j = fp[j]

    @qd.func
    def _ld3m(a: qd.template(), b, i):
        """3x3 block of a (B, n, 3, 3) array."""
        return qd.Matrix(
            [
                [a[b, i, 0, 0], a[b, i, 0, 1], a[b, i, 0, 2]],
                [a[b, i, 1, 0], a[b, i, 1, 1], a[b, i, 1, 2]],
                [a[b, i, 2, 0], a[b, i, 2, 1], a[b, i, 2, 2]],
            ],
            dt=qd.f64,
        )

    need_SJ = (m_floor > 0) or (m_sc > 0)
    fgain = float(solver.floor.gain) if solver.floor is not None else 0.0
    fmargin = float(solver.floor.margin) if solver.floor is not None else 0.0
    sc_gain = (
        float(solver.self_collision.gain) if solver.self_collision is not None else 0.0
    )

    @qd.kernel
    def k_build(
        q: _F64_2,
        tgt_pos: _F64_3,
        tgt_rmat: _F64_4,
        tgt_jnt: _F64_2,
        P: _F64_3,
        c: _F64_2,
        G: _F64_3,
        h: _F64_2,
        cost: _F64_1,
        Ro: _F64_4,
        po: _F64_3,
        Jv: _F64_4,
        Jw: _F64_4,
        sp: _F64_3,
        SJ: _F64_4,
        HP: _F64_2,
        SCU: _F64_2,
        B: qd.i32,
    ):
        for b in range(B):
            _fk_all(q, Ro, po, b)
            if qd.static(nt > 0):
                for t in range(nt):
                    _frame_jac(q, Ro, po, b, f_tracked[t], Jv, Jw, t)
            # ---- sphere world positions (+ point jacobians when constrained)
            if qd.static(ns > 0):
                for s in range(ns):
                    lk = f_sp_link[s]
                    Rl = _ld3(Ro, b, lk)
                    off = qd.Vector(
                        [f_sp_off[s][0], f_sp_off[s][1], f_sp_off[s][2]], dt=qd.f64
                    )
                    pw = (
                        qd.Vector([po[b, lk, 0], po[b, lk, 1], po[b, lk, 2]], dt=qd.f64)
                        + Rl @ off
                    )
                    for r in range(3):
                        sp[b, s, r] = pw[r]
                    if qd.static(need_SJ):
                        _point_jac(q, Ro, po, b, lk, pw, SJ, s)
            # ---- task rows into P, c
            for r in range(nv):
                c[b, r] = 0.0
                for cc in range(nv):
                    P[b, r, cc] = 0.0
            task_err = qd.cast(0.0, qd.f64)
            for row in range(ntr):
                ty = f_rt[row]
                w2 = f_rw[row] * f_rw[row]
                jr = qd.Vector.zero(qd.f64, nv)
                vdes = qd.cast(0.0, qd.f64)
                if ty == 0:
                    fr = f_rf[row]
                    ji = f_tracked[fr]
                    comp = f_rc[row]
                    e = po[b, ji, comp] - tgt_pos[b, fr, comp]
                    vdes = -f_rg[row] * e
                    for r in range(nv):
                        jr[r] = Jv[b, fr, comp, r]
                elif ty == 1:
                    fr = f_rf[row]
                    ji = f_tracked[fr]
                    comp = f_rc[row]
                    er = orientation_error(_ld3(Ro, b, ji), _ld3m(tgt_rmat, b, fr))
                    vdes = -f_rg[row] * er[comp]
                    for r in range(nv):
                        jr[r] = Jw[b, fr, comp, r]
                else:
                    dof = f_rd[row]
                    qi = f_rq[row]
                    e = q[b, qi] - tgt_jnt[b, f_rj[row]]
                    vdes = -f_rg[row] * e
                    jr[dof] = qd.cast(1.0, qd.f64)
                task_err = task_err + w2 * vdes * vdes
                for r in range(nv):
                    c[b, r] = c[b, r] - w2 * vdes * jr[r]
                    jr_r = jr[r]
                    for cc in range(nv):
                        P[b, r, cc] = P[b, r, cc] + w2 * jr_r * jr[cc]
            for r in range(nv):
                P[b, r, r] = P[b, r, r] + f_reg[r] + damping * task_err
            cost[b] = task_err
            # ---- constraint rows into G, h (G is zeroed first: rows are
            # sparse and each section writes only its own coefficients)
            if qd.static(m > 0):
                for k in range(m):
                    for r in range(nv):
                        G[b, k, r] = 0.0
            if qd.static(m_lim > 0):
                for k in range(m_lim):
                    dof = f_lim_dof[k]
                    qi = f_lim_q[k]
                    if f_lim_up[k] == 1:
                        val = f_lim_vmax[k]
                        tmp = f_lim_g[k] * (f_lim_qmax[k] - q[b, qi])
                        if tmp < val:
                            val = tmp
                        G[b, k, dof] = 1.0
                        h[b, k] = val
                    else:
                        val = -f_lim_vmax[k]
                        tmp = f_lim_g[k] * (f_lim_qmin[k] - q[b, qi])
                        if tmp > val:
                            val = tmp
                        G[b, k, dof] = -1.0
                        h[b, k] = -val
                    if h[b, k] < 0.0:
                        cost[b] = cost[b] - f_pen[k] * h[b, k]
            if qd.static(m_floor > 0):
                for k in range(m_floor):
                    s = ff_sph[k]
                    row = m_lim + k
                    hz = sp[b, s, 2] - f_sp_rad[s] - fmargin
                    h[b, row] = fgain * hz
                    for r in range(nv):
                        G[b, row, r] = -SJ[b, s, 2, r]
                    if h[b, row] < 0.0:
                        cost[b] = cost[b] - f_pen[row] * h[b, row]
            if qd.static(m_sc > 0):
                if qd.static(sc_topk):
                    # pass 1: h of every pair (selection runs on h alone; the
                    # Jacobians are only needed for the kept rows)
                    for p in range(sc_n_all):
                        a0 = fsc_a[p]
                        b0 = fsc_b[p]
                        e0 = sp[b, a0, 0] - sp[b, b0, 0]
                        e1 = sp[b, a0, 1] - sp[b, b0, 1]
                        e2 = sp[b, a0, 2] - sp[b, b0, 2]
                        dpair = qd.sqrt(
                            e0 * e0 + e1 * e1 + e2 * e2 + qd.cast(1e-12, qd.f64)
                        )
                        HP[b, p] = dpair - (f_sp_rad[a0] + f_sp_rad[b0] + fsc_tol[p])
                        SCU[b, p] = 0.0
                    # pass 2: k rounds of argmin over the remaining pairs;
                    # ties resolve to the lowest index, so this is deterministic
                    for j in range(m_sc):
                        best = qd.cast(1e30, qd.f64)
                        bestp = qd.cast(0, qd.i32)
                        for p in range(sc_n_all):
                            if SCU[b, p] == 0.0:
                                if HP[b, p] < best:
                                    best = HP[b, p]
                                    bestp = p
                        SCU[b, bestp] = 1.0
                        a = fsc_a[bestp]
                        bb = fsc_b[bestp]
                        row = m_lim + m_floor + j
                        h[b, row] = sc_gain * HP[b, bestp]
                        # recompute the direction from the sphere offsets
                        d0 = sp[b, a, 0] - sp[b, bb, 0]
                        d1 = sp[b, a, 1] - sp[b, bb, 1]
                        d2 = sp[b, a, 2] - sp[b, bb, 2]
                        dn = qd.sqrt(
                            d0 * d0 + d1 * d1 + d2 * d2 + qd.cast(1e-12, qd.f64)
                        )
                        for r in range(nv):
                            jd = (
                                d0 * (SJ[b, a, 0, r] - SJ[b, bb, 0, r])
                                + d1 * (SJ[b, a, 1, r] - SJ[b, bb, 1, r])
                                + d2 * (SJ[b, a, 2, r] - SJ[b, bb, 2, r])
                            )
                            G[b, row, r] = -jd / dn
                        if h[b, row] < 0.0:
                            cost[b] = cost[b] - f_pen[row] * h[b, row]
                else:
                    for k in range(m_sc):
                        a = fsc_a[k]
                        bb = fsc_b[k]
                        row = m_lim + m_floor + k
                        d0 = sp[b, a, 0] - sp[b, bb, 0]
                        d1 = sp[b, a, 1] - sp[b, bb, 1]
                        d2 = sp[b, a, 2] - sp[b, bb, 2]
                        dist = qd.sqrt(
                            d0 * d0 + d1 * d1 + d2 * d2 + qd.cast(1e-12, qd.f64)
                        )
                        hpair = dist - (f_sp_rad[a] + f_sp_rad[bb] + fsc_tol[k])
                        h[b, row] = sc_gain * hpair
                        inv = qd.cast(1.0, qd.f64) / dist
                        for r in range(nv):
                            jd = (
                                d0 * (SJ[b, a, 0, r] - SJ[b, bb, 0, r])
                                + d1 * (SJ[b, a, 1, r] - SJ[b, bb, 1, r])
                                + d2 * (SJ[b, a, 2, r] - SJ[b, bb, 2, r])
                            )
                            G[b, row, r] = -jd * inv
                        if h[b, row] < 0.0:
                            cost[b] = cost[b] - f_pen[row] * h[b, row]

    return dict(build=k_build)
