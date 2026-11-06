"""Differential IK constraints (frax.ik.constraints port).

All constraints are elastic inequalities ``G v <= h`` with per-row L1
penalties (the ElastiQP contract): the QP solver holds them exactly when
feasible and relaxes them at the stated penalty otherwise.  Every constraint
is static configuration resolved at solver-build time; the per-frame data
they need is either the configuration itself (joint limits) or FK outputs
computed inside the QP-build kernel (floor / self-collision).
"""

from collections.abc import Sequence

import numpy as np

DEFAULT_PENALTY = 1e6  # frax DEFAULT_PENALTY


class SphereModel:
    """Sphere collision model: spheres rigidly attached to links.

    Args:
        model: :class:`~qd_ik.model.Model`
        links: link names or joint indices the spheres attach to.  A sphere
            on joint i rides that joint's frame (which coincides with the
            child link frame).
        offsets: (ns, 3) sphere centers in their link frames
        radii: (ns,) sphere radii
    """

    def __init__(self, model, links: Sequence, offsets, radii):
        idx = []
        for j in links:
            idx.append(model.joint_index(j) if isinstance(j, str) else int(j))
        self.model = model
        self.links = np.array(idx, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
        self.radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if not (len(self.links) == len(self.offsets) == len(self.radii)):
            raise ValueError("links, offsets and radii must agree in length")
        if np.any(self.radii <= 0.0):
            raise ValueError("sphere radii must be positive")

    def __len__(self):
        """Number of spheres in the model."""
        return len(self.links)

    @classmethod
    def from_dict(cls, model, d):
        """Build from a plain dict (e.g. loaded npz/json): links/offsets/radii."""
        return cls(model, d["links"], d["offsets"], d["radii"])

    def to_dict(self):
        names = [self.model.names[i] for i in self.links]
        return dict(links=names, offsets=self.offsets, radii=self.radii)


class JointLimits:
    """Actuated joint position/velocity limits as CBF velocity bounds::

        max(-v_max, gain * (q_min - q)) <= v <= min(v_max, gain * (q_max - q))

    Args:
        model: :class:`~qd_ik.model.Model`
        q_min, q_max, v_max: None (no limit), "model" (the model's own), or
            (n_act,) arrays overriding the model's
        gain: CBF gain (1/s)
        penalty: elastic relaxation penalty
    """

    def __init__(
        self,
        model,
        q_min="model",
        q_max="model",
        v_max="model",
        gain=10.0,
        penalty=DEFAULT_PENALTY,
    ):
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        self.model = model
        self.gain = float(gain)
        self.penalty = float(penalty)
        m_qmin, m_qmax, m_vmax = model.actuated_limits()

        def resolve(x, model_default, none_fill):
            if isinstance(x, str):
                if x != "model":
                    raise ValueError(
                        f'a limit must be an array, "model" or None, got {x!r}'
                    )
                return model_default
            if x is None:
                return np.full(model.n_act, none_fill)
            return np.asarray(x, dtype=np.float64)

        self.q_min = resolve(q_min, m_qmin, -np.inf)
        self.q_max = resolve(q_max, m_qmax, np.inf)
        self.v_max = resolve(v_max, m_vmax, np.inf)
        if not (len(self.q_min) == len(self.q_max) == len(self.v_max) == model.n_act):
            raise ValueError("limit arrays must have shape (n_act,)")

        base = 6 if model.floating_base else 0
        up_dof, lo_dof = [], []
        for k in range(model.n_act):
            if np.isfinite(self.q_max[k]) or np.isfinite(self.v_max[k]):
                up_dof.append(base + k)
            if np.isfinite(self.q_min[k]) or np.isfinite(self.v_max[k]):
                lo_dof.append(base + k)
        self.upper_dof = np.array(up_dof, dtype=np.int32)
        self.lower_dof = np.array(lo_dof, dtype=np.int32)
        self.num_rows = len(up_dof) + len(lo_dof)
        if self.num_rows == 0:
            raise ValueError("JointLimits resolved to zero rows")

    def penalty_vector(self):
        return np.full(self.num_rows, self.penalty)


class FloorConstraint:
    """Keep collision spheres above the floor z = 0.

    Row per sphere: ``-J_z v <= gain * (z - r - margin)``.

    Args:
        model: :class:`~qd_ik.model.Model`
        spheres: :class:`SphereModel`
        sphere_idxs: subset of spheres to constrain (default all)
        gain: CBF gain (1/s)
        penalty: elastic relaxation penalty
        margin: extra height kept above the floor
    """

    def __init__(
        self,
        model,
        spheres: SphereModel,
        sphere_idxs=None,
        gain=10.0,
        penalty=DEFAULT_PENALTY,
        margin=0.0,
    ):
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        self.model = model
        self.spheres = spheres
        self.sphere_idxs = (
            np.arange(len(spheres))
            if sphere_idxs is None
            else np.asarray(sphere_idxs, dtype=np.int32)
        )
        self.gain = float(gain)
        self.penalty = float(penalty)
        self.margin = float(margin)
        self.num_rows = len(self.sphere_idxs)

    def penalty_vector(self):
        return np.full(self.num_rows, self.penalty)


class SelfCollisionConstraint:
    """Keep sphere pairs apart.

    Row per pair (a, b): ``-n' (J_a - J_b) v <= gain * h`` with
    ``h = |p_a - p_b| - (r_a + r_b + tol)`` and ``n`` the pair direction.
    Pairs on the same link, or sphere pairs the caller marks excluded (e.g.
    adjacent links), are simply not listed.

    Args:
        model: :class:`~qd_ik.model.Model`
        spheres: :class:`SphereModel`
        pairs: (np, 2) sphere indices
        tols: per-pair extra separations (default 0), or a scalar
        gain: CBF gain (1/s)
        penalty: elastic relaxation penalty
        margin: additional safety distance on every pair
        max_rows: if set, keep only the ``max_rows`` closest pairs (smallest
            h) in the QP each frame, as frax's ``max_rows`` does.  Selection
            runs inside the build kernel on h alone; dropped pairs are
            typically far from contact and re-enter the kept set as they
            approach, so the relaxation is mild in practice.
    """

    def __init__(
        self,
        model,
        spheres: SphereModel,
        pairs,
        tols=0.0,
        gain=10.0,
        penalty=DEFAULT_PENALTY,
        margin=0.0,
        max_rows=None,
    ):
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        self.model = model
        self.spheres = spheres
        self.pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        self.tols = np.broadcast_to(
            np.asarray(tols, dtype=np.float64), (len(self.pairs),)
        ).copy()
        self.gain = float(gain)
        self.penalty = float(penalty)
        self.margin = float(margin)
        ns = len(spheres)
        if np.any(self.pairs < 0) or np.any(self.pairs >= ns):
            raise ValueError("self-collision pair index out of range")
        self.num_pairs = len(self.pairs)
        if max_rows is None:
            self.max_rows = None
            self.num_rows = self.num_pairs
        else:
            if max_rows <= 0:
                raise ValueError("max_rows must be positive")
            self.max_rows = int(max_rows)
            self.num_rows = min(self.max_rows, self.num_pairs)

    def penalty_vector(self):
        return np.full(self.num_rows, self.penalty)
