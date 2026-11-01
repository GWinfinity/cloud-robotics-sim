"""SOA model: flatten a Pinocchio-like kinematic tree into Quadrants fields."""

import os

import numpy as np
import quadrants as qd

# pinocchio is an optional dependency: it is only needed to ingest an existing
# pinocchio.Model (``Model.from_pinocchio``).  Everything else -- including
# building a Model directly from raw arrays -- works without it, which matters
# on Windows where pinocchio ships no wheels (WSL/Linux only).  It is therefore
# imported lazily inside ``from_pinocchio``.

REV = 0  # revolute, 1 dof
PRIS = 1  # prismatic, 1 dof
FIX = 2  # fixed, 0 dof

_INITED = False


def ensure_init(arch=None, threads=None):
    """Idempotent qd.init.

    The thread count defaults to ``$QD_THREADS`` when set, else to the CPU
    count.  Quadrants only engages a thread when there is a batch slice to give
    it, so a small pool mainly limits large-B throughput -- see REPORT.md §5.4.
    """
    global _INITED
    if _INITED:
        return
    if threads is None:
        threads = int(os.environ.get("QD_THREADS", "0")) or (os.cpu_count() or 1)
    qd.init(arch=arch or qd.cpu, cpu_max_num_threads=threads)
    _INITED = True


class Model:
    """Rigid-body tree in structure-of-arrays form.

    Joint 0 is the universe/base and carries no dof, mirroring Pinocchio.
    Only 1-dof joints (revolute / prismatic) and fixed joints are modelled.
    """

    def __init__(self, parents, jtype, axis, Rpl, ppl, mass, lever, Icom, iv, gravity):
        ensure_init()
        self.parents = np.asarray(parents, dtype=np.int32)
        self.jtype = np.asarray(jtype, dtype=np.int32)
        self.axis = np.asarray(axis, dtype=np.float64).reshape(-1, 3)
        self.Rpl = np.asarray(Rpl, dtype=np.float64).reshape(-1, 3, 3)
        self.ppl = np.asarray(ppl, dtype=np.float64).reshape(-1, 3)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.lever = np.asarray(lever, dtype=np.float64).reshape(-1, 3)
        self.Icom = np.asarray(Icom, dtype=np.float64).reshape(-1, 3, 3)
        self.iv = np.asarray(iv, dtype=np.int32)
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)

        self.nj = self.parents.shape[0]
        self.nv = int(max(self.iv.max() + 1, 0))
        if self.nj > 64:
            raise ValueError("ancestor bitmap supports at most 63 joints")

        # leaf flag: a leaf's composite inertia is just its body inertia, which
        # admits a much cheaper transform (see algorithms._blk_xf_body)
        leaf = np.ones(self.nj, dtype=np.int32)
        for i in range(1, self.nj):
            leaf[self.parents[i]] = 0
        self.leaf = leaf

        # depth + ancestor bitmask (bit j set <=> j is an ancestor of i, incl. i)
        self.depth = np.zeros(self.nj, dtype=np.int32)
        mask = np.zeros(self.nj, dtype=np.uint64)
        mask[0] = np.uint64(1)
        for i in range(1, self.nj):
            self.depth[i] = self.depth[self.parents[i]] + 1
            mask[i] = mask[self.parents[i]] | np.uint64(1) << np.uint64(i)
        self.mask = mask

        # Which mass-matrix rows the CRBA fill covers *completely*.
        #
        # Row of dof i receives the diagonal, every ancestor column and every
        # descendant column (the last group through the symmetric write).  On a
        # chain that is the whole row, so even the zeroing pass is dead weight --
        # and at nv=63 it is 63 stores per instance, close to half of all the
        # writes CRBA does.  On a branched tree the row keeps its cross-branch
        # gaps and still has to be cleared.
        subtree = [0] * self.nj
        for i in range(self.nj - 1, 0, -1):
            subtree[self.parents[i]] |= subtree[i] | (1 << int(i))
        dof_bits = 0
        for i in range(self.nj):
            if self.iv[i] >= 0:
                dof_bits |= 1 << i
        row_full = np.zeros(self.nj, dtype=np.int32)
        for i in range(self.nj):
            if self.iv[i] < 0:
                continue
            covered = int(mask[i]) | subtree[i] | (1 << i)
            row_full[i] = 1 if bin(covered & dof_bits).count("1") == self.nv else 0
        self.row_full = row_full

        # 6x6 body spatial inertia, constant in the body frame
        I6 = np.zeros((self.nj, 6, 6))
        for i in range(self.nj):
            W = np.array(
                [
                    [0.0, -self.lever[i, 2], self.lever[i, 1]],
                    [self.lever[i, 2], 0.0, -self.lever[i, 0]],
                    [-self.lever[i, 1], self.lever[i, 0], 0.0],
                ]
            )
            m = self.mass[i]
            I6[i, :3, :3] = self.Icom[i] + m * W @ W.T
            I6[i, :3, 3:] = m * W
            I6[i, 3:, :3] = -m * W
            I6[i, 3:, 3:] = m * np.eye(3)
        self.I6_np = I6

        # body inertia in 3x3 blocks: [[A0, B0], [B0^T, m*1]]
        # A0 = I about the joint origin, B0 = m c^
        self.I_A0 = I6[:, :3, :3].copy()
        self.I_B0 = I6[:, :3, 3:].copy()

        self.f_parent = qd.field(qd.i32, shape=self.nj)
        self.f_jtype = qd.field(qd.i32, shape=self.nj)
        self.f_axis = qd.Vector.field(3, qd.f64, shape=self.nj)
        self.f_Rpl = qd.Matrix.field(3, 3, qd.f64, shape=self.nj)
        self.f_ppl = qd.Vector.field(3, qd.f64, shape=self.nj)
        self.f_A0 = qd.Matrix.field(3, 3, qd.f64, shape=self.nj)
        self.f_B0 = qd.Matrix.field(3, 3, qd.f64, shape=self.nj)
        self.f_mass = qd.field(qd.f64, shape=self.nj)
        self.f_iv = qd.field(qd.i32, shape=self.nj)
        self.f_mask = qd.field(qd.u64, shape=self.nj)
        self.f_depth = qd.field(qd.i32, shape=self.nj)
        self.f_leaf = qd.field(qd.i32, shape=self.nj)
        # 1 => CRBA fills this row completely, so it needs no zeroing
        self.f_row_full = qd.field(qd.i32, shape=self.nj)
        self.f_mc = qd.Vector.field(3, qd.f64, shape=self.nj)
        # gravity must live in a field: a Python float literal would be
        # folded to f32 by the compiler and spoil f64 agreement.
        self.f_grav = qd.field(qd.f64, shape=3)

        self.f_parent.from_numpy(self.parents)
        self.f_jtype.from_numpy(self.jtype)
        self.f_axis.from_numpy(self.axis)
        self.f_Rpl.from_numpy(self.Rpl)
        self.f_ppl.from_numpy(self.ppl)
        self.f_A0.from_numpy(self.I_A0)
        self.f_B0.from_numpy(self.I_B0)
        self.f_mass.from_numpy(self.mass.copy())
        self.f_iv.from_numpy(self.iv)
        self.f_mask.from_numpy(self.mask)
        self.f_depth.from_numpy(self.depth)
        self.f_leaf.from_numpy(self.leaf)
        self.f_row_full.from_numpy(self.row_full)
        self.f_mc.from_numpy(self.mass[:, None] * self.lever)
        self.f_grav.from_numpy(self.gravity)

        self._kern = None
        self._cmodel = None

    def cmodel(self):
        """C-side view of this model, built on first use (REPORT.md §12)."""
        if self._cmodel is None:
            from .ckern import CModel

            self._cmodel = CModel(self)
        return self._cmodel

    def kernels(self):
        """Compiled kernel set, built once per model.

        Nothing about a ``Data`` enters the compile, so this is the only place
        a compile can be triggered.
        """
        if self._kern is None:
            from .algorithms import _build

            self._kern = _build(self)
        return self._kern

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pinocchio(cls, pm):
        """Ingest a pinocchio.Model (1-dof joints only).

        Requires the optional ``pin`` (pinocchio) package; on Windows install
        it inside WSL, as pinocchio publishes no Windows wheels.
        """
        try:
            import pinocchio as pin
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Model.from_pinocchio needs the optional 'pin' package "
                "(pinocchio). On Windows use WSL; on Linux/macOS: "
                "pip install pin"
            ) from e
        nj = pm.njoints
        parents = np.array(list(pm.parents), dtype=np.int32)
        jtype = np.zeros(nj, dtype=np.int32)
        axis = np.zeros((nj, 3))
        Rpl = np.zeros((nj, 3, 3))
        ppl = np.zeros((nj, 3))
        mass = np.zeros(nj)
        lever = np.zeros((nj, 3))
        Icom = np.zeros((nj, 3, 3))
        iv = np.full(nj, -1, dtype=np.int32)

        # Motion subspace at q = 0 gives the joint axis in the joint frame.
        # pinocchio's Python bindings expose joints[] as the base class, so the
        # axis is recovered through the LOCAL joint Jacobian instead of `.axis`.
        d0 = pm.createData()
        pin.computeJointJacobians(pm, d0, np.zeros(pm.nq))

        for i in range(nj):
            M = pm.jointPlacements[i]
            Rpl[i] = np.array(M.rotation)
            ppl[i] = np.array(M.translation)
            mass[i] = pm.inertias[i].mass
            lever[i] = np.array(pm.inertias[i].lever)
            Icom[i] = np.array(pm.inertias[i].inertia)
            if i == 0:
                continue
            j = pm.joints[i]
            sn = j.shortname()
            if j.nv != 1:
                raise ValueError(
                    f"joint {i} ({sn}) has nv={j.nv}; only 1-dof joints supported"
                )
            J = np.array(pin.getJointJacobian(pm, d0, i, pin.LOCAL)).reshape(6, -1)[
                :, j.idx_v
            ]
            is_rev = "Prismatic" not in sn
            # pinocchio orders Jacobian rows (linear, angular);
            # Featherstone spatial vectors are (angular, linear).
            ax = J[3:] if is_rev else J[:3]
            ax = ax / np.linalg.norm(ax)
            axis[i] = ax
            jtype[i] = REV if is_rev else PRIS
            iv[i] = j.idx_v

        return cls(
            parents,
            jtype,
            axis,
            Rpl,
            ppl,
            mass,
            lever,
            Icom,
            iv,
            np.array(pm.gravity.linear),
        )

    def __repr__(self):
        """Short summary with joint/dof/depth counts."""
        return f"<qd_rbd.Model nj={self.nj} nv={self.nv} depth={self.depth.max()}>"
