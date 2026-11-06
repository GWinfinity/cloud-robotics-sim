"""Kinematics-only SOA robot model for batched differential IK.

Deliberately a sibling of ``qd_rbd.model`` rather than a reuse: retargeting
needs no inertias, but it does need a floating base, which qd_rbd's Model
does not model.  Conventions follow frax's MuJoCo-style quaternion base:

  q = [base position (3), WXYZ quaternion (4), actuated joints (n_act)]
  v = [world linear velocity (3), body angular velocity (3), joint rates]

so ``nq = 7 + n_act`` and ``nv = 6 + n_act`` with a floating base, and
``nq == nv`` without one.  Joint 0 is the universe, as in qd_rbd/pinocchio.
Only 1-dof joints (revolute / prismatic) and fixed joints are modelled.
"""

import os

import numpy as np
import quadrants as qd

REV = 0  # revolute, 1 dof
PRIS = 1  # prismatic, 1 dof
FIX = 2  # fixed, 0 dof

_INITED = False


def ensure_init(arch=None, threads=None):
    """Idempotent qd.init; thread count from ``$QD_THREADS`` or the CPU count."""
    global _INITED
    if _INITED:
        return
    if threads is None:
        threads = int(os.environ.get("QD_THREADS", "0")) or (os.cpu_count() or 1)
    qd.init(arch=arch or qd.cpu, cpu_max_num_threads=threads)
    _INITED = True


class Model:
    """Kinematic tree in structure-of-arrays form.

    Args:
        parents: (nj,) int, parent joint index; joint 0 is the universe.
        jtype: (nj,) int, one of REV / PRIS / FIX (row 0 is ignored).
        axis: (nj, 3) joint motion axis in the joint frame.
        Rpl: (nj, 3, 3) placement rotation of joint i relative to its parent
            at q = 0.
        ppl: (nj, 3) placement translation likewise.
        q_min, q_max: (n_act,) actuated joint position limits (+-inf allowed).
        v_max: (n_act,) actuated joint velocity limits (+-inf allowed,
            ``None`` for no limits).
        names: optional joint names; index 0 is the universe.
        floating_base: model a quaternion floating base on dofs 0..5.
    """

    def __init__(
        self,
        parents,
        jtype,
        axis,
        Rpl,
        ppl,
        q_min=None,
        q_max=None,
        v_max=None,
        names=None,
        floating_base=False,
        dof_order=None,
    ):
        ensure_init()
        self.parents = np.asarray(parents, dtype=np.int32)
        self.jtype = np.asarray(jtype, dtype=np.int32)
        self.axis = np.asarray(axis, dtype=np.float64).reshape(-1, 3)
        self.Rpl = np.asarray(Rpl, dtype=np.float64).reshape(-1, 3, 3)
        self.ppl = np.asarray(ppl, dtype=np.float64).reshape(-1, 3)
        self.names = list(names) if names is not None else None
        self.floating_base = bool(floating_base)

        self.nj = self.parents.shape[0]
        movable = [i for i in range(1, self.nj) if self.jtype[i] != FIX]
        if dof_order is not None:
            order = [int(i) for i in dof_order]
            if sorted(order) != sorted(movable):
                raise ValueError(
                    "dof_order must be a permutation of the movable joints"
                )
        else:
            order = movable
        self.n_act = len(order)
        iv = np.full(self.nj, -1, dtype=np.int32)
        iq = np.full(self.nj, -1, dtype=np.int32)
        base_v = 6 if self.floating_base else 0
        base_q = 7 if self.floating_base else 0
        for rank, j in enumerate(order):
            iv[j] = base_v + rank
            iq[j] = base_q + rank
        self.iv = iv
        self.iq = iq
        self.nv = base_v + self.n_act
        self.nq = base_q + self.n_act

        n = self.n_act
        self.q_min = (
            np.full(n, -np.inf)
            if q_min is None
            else np.asarray(q_min, dtype=np.float64).reshape(n)
        )
        self.q_max = (
            np.full(n, np.inf)
            if q_max is None
            else np.asarray(q_max, dtype=np.float64).reshape(n)
        )
        self.v_max = (
            np.full(n, np.inf)
            if v_max is None
            else np.asarray(v_max, dtype=np.float64).reshape(n)
        )

        # actuated dof -> joint index, ordered by dof index
        self.act_joint = np.array(
            sorted(
                (i for i in range(1, self.nj) if self.iv[i] >= 0),
                key=lambda i: self.iv[i],
            ),
            dtype=np.int32,
        )
        self.act_v0 = 6 if self.floating_base else 0

        for arr, shape in (
            (self.parents, (self.nj,)),
            (self.jtype, (self.nj,)),
            (self.axis, (self.nj, 3)),
            (self.Rpl, (self.nj, 3, 3)),
            (self.ppl, (self.nj, 3)),
        ):
            if arr.shape != shape:
                raise ValueError(f"model array has shape {arr.shape}, want {shape}")

        self.f_parent = qd.field(qd.i32, shape=self.nj)
        self.f_jtype = qd.field(qd.i32, shape=self.nj)
        self.f_axis = qd.Vector.field(3, qd.f64, shape=self.nj)
        self.f_Rpl = qd.Matrix.field(3, 3, qd.f64, shape=self.nj)
        self.f_ppl = qd.Vector.field(3, qd.f64, shape=self.nj)
        self.f_iv = qd.field(qd.i32, shape=self.nj)
        self.f_iq = qd.field(qd.i32, shape=self.nj)

        self.f_parent.from_numpy(self.parents)
        self.f_jtype.from_numpy(self.jtype)
        self.f_axis.from_numpy(self.axis)
        self.f_Rpl.from_numpy(self.Rpl)
        self.f_ppl.from_numpy(self.ppl)
        self.f_iv.from_numpy(self.iv)
        self.f_iq.from_numpy(self.iq)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_arrays(cls, **kwargs):
        """Alias for the constructor, mirroring qd_rbd's raw-array entry point."""
        return cls(**kwargs)

    @classmethod
    def from_urdf(cls, path, floating_base=False, joint_ordering=None):
        """Build a Model from a URDF (revolute/prismatic/continuous/fixed).

        Args:
            path: URDF file path.
            floating_base: attach a quaternion floating base between the
                universe and the URDF root link.
            joint_ordering: optional sequence of joint names; the actuated
                dof order follows this list (must cover all non-fixed joints).
        """
        from .urdf import parse_urdf_kinematics

        return parse_urdf_kinematics(path, floating_base, joint_ordering)

    # ---------------------------------------------------------------- helpers
    def joint_index(self, name):
        """Index of a joint by name."""
        if self.names is None:
            raise ValueError("model was built without joint names")
        return self.names.index(name)

    def default_configuration(self, batch=1):
        """(batch, nq) configuration: base at the origin, joints at zero."""
        q = np.zeros((batch, self.nq))
        if self.floating_base:
            q[:, 3] = 1.0  # identity quaternion
        return q

    def actuated_limits(self):
        """(q_min, q_max, v_max) of the actuated joints, as copies."""
        return self.q_min.copy(), self.q_max.copy(), self.v_max.copy()

    def __repr__(self):
        """Short summary with joint/dof counts."""
        base = "float" if self.floating_base else "fixed"
        return (
            f"<qd_ik.Model nj={self.nj} n_act={self.n_act} "
            f"nv={self.nv} nq={self.nq} {base}-base>"
        )
