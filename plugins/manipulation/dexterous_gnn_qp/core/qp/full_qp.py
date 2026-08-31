"""Full QP baseline for dexterous hand grasping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.sparse as spa

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.env.sim import Contact, SimState
from dexterous_gnn_qp.core.qp.osqp_utils import solve_osqp
from dexterous_gnn_qp.core.utils.config import DotDict
from dexterous_gnn_qp.core.utils.math import skew


@dataclass
class FullQPResult:
    """Result of the full QP."""

    x: np.ndarray
    qddot: np.ndarray
    a_obj: np.ndarray
    forces: np.ndarray
    tau: np.ndarray
    status: str
    solve_time: float
    objective: float


def _build_index(n_hand: int, n_contacts: int, include_tau: bool) -> Dict[str, slice]:
    n_f = 3 * n_contacts
    idx = {
        "q": slice(0, n_hand),
        "a": slice(n_hand, n_hand + 6),
        "f": slice(n_hand + 6, n_hand + 6 + n_f),
    }
    if include_tau:
        idx["tau"] = slice(n_hand + 6 + n_f, n_hand + 6 + n_f + n_hand)
    return idx


class FullQP:
    """Full augmented-dynamics QP with all contact forces as decision variables.

    If ``cfg.qp.include_tau`` is False (recommended for Phase-1), the hand joint
    torques are computed analytically after the solve from the hand dynamics
    equation.  This removes the torque degree of freedom and makes the force
    distribution better posed.
    """

    def __init__(self, backend: DynamicsBackend, cfg: DotDict):
        self.backend = backend
        self.cfg = cfg
        self.w = cfg.qp.weights
        self.mu = float(cfg.qp.mu)
        self.include_tau = bool(cfg.qp.get("include_tau", False))
        self.max_force = cfg.qp.get("max_force_per_contact", None)
        if self.max_force is not None:
            self.max_force = float(self.max_force)

    def solve(
        self,
        state: SimState,
        qddot_des: np.ndarray,
        a_obj_des: np.ndarray,
        warm_start: np.ndarray | None = None,
    ) -> FullQPResult:
        n_h = self.backend.n_hand_dof
        contacts = state.contacts
        n_c = len(contacts)
        idx = _build_index(n_h, n_c, self.include_tau)
        n_var = n_h + 6 + 3 * n_c + (n_h if self.include_tau else 0)

        # Objective.
        P = np.eye(n_var) * 1e-8
        P[np.ix_(range(idx["q"].start, idx["q"].stop), range(idx["q"].start, idx["q"].stop))] += float(
            self.w.hand_acceleration
        ) * np.eye(idx["q"].stop - idx["q"].start)
        P[np.ix_(range(idx["a"].start, idx["a"].stop), range(idx["a"].start, idx["a"].stop))] += float(
            self.w.object_acceleration
        ) * np.eye(idx["a"].stop - idx["a"].start)
        P[np.ix_(range(idx["f"].start, idx["f"].stop), range(idx["f"].start, idx["f"].stop))] += float(
            self.w.force
        ) * np.eye(idx["f"].stop - idx["f"].start)
        if self.include_tau:
            P[np.ix_(range(idx["tau"].start, idx["tau"].stop), range(idx["tau"].start, idx["tau"].stop))] += float(
                self.w.torque
            ) * np.eye(idx["tau"].stop - idx["tau"].start)

        q = np.zeros(n_var)
        q[idx["q"]] = -float(self.w.hand_acceleration) * qddot_des
        q[idx["a"]] = -float(self.w.object_acceleration) * a_obj_des

        # Equality constraints.
        if self.include_tau:
            n_eq = n_h + 6
        else:
            n_eq = 6
        A_eq = np.zeros((n_eq, n_var))
        l_eq = np.zeros(n_eq)
        u_eq = np.zeros(n_eq)

        row = 0
        if self.include_tau:
            # Hand dynamics: M_h qddot - tau + sum J_i^T f_i = -C_h
            A_eq[:n_h, idx["q"]] = state.M_hand
            A_eq[:n_h, idx["tau"]] = -np.eye(n_h)
            for i, cnt in enumerate(contacts):
                J = self.backend.hand_jacobian(cnt)
                A_eq[:n_h, idx["f"].start + 3 * i : idx["f"].start + 3 * (i + 1)] = J.T
            l_eq[:n_h] = -state.C_hand
            u_eq[:n_h] = -state.C_hand
            row = n_h

        # Object dynamics: M_obj a - sum G_i^T f_i = -C_obj.
        # MuJoCo free-joint qvel is [linear; angular], so the contact map is [I; skew(r)].
        A_eq[row : row + 6, idx["a"]] = state.M_obj
        for i, cnt in enumerate(contacts):
            r = cnt.pos - state.x_obj
            Gt = np.vstack([np.eye(3), skew(r)])
            A_eq[row : row + 6, idx["f"].start + 3 * i : idx["f"].start + 3 * (i + 1)] = -Gt
        l_eq[row : row + 6] = -state.C_obj
        u_eq[row : row + 6] = -state.C_obj

        # Inequality: friction pyramid + optional normal-force limit.
        n_force_bounds = n_c if self.max_force is not None else 0
        n_ineq = 4 * n_c + n_force_bounds
        A_ineq = np.zeros((n_ineq, n_var))
        l_ineq = -np.inf * np.ones(n_ineq)
        u_ineq = np.zeros(n_ineq)
        row = 0
        for i, cnt in enumerate(contacts):
            f_start = idx["f"].start + 3 * i
            # n^T f >= 0
            A_ineq[row, f_start : f_start + 3] = cnt.normal
            l_ineq[row] = 0.0
            u_ineq[row] = np.inf
            row += 1
            # |t^T f| <= mu n^T f  ->  (t - mu n)^T f <= 0
            for t in (cnt.tangent1, cnt.tangent2):
                A_ineq[row, f_start : f_start + 3] = t - cnt.mu * cnt.normal
                l_ineq[row] = -np.inf
                u_ineq[row] = 0.0
                row += 1
        if self.max_force is not None:
            for i, cnt in enumerate(contacts):
                f_start = idx["f"].start + 3 * i
                A_ineq[row, f_start : f_start + 3] = cnt.normal
                l_ineq[row] = 0.0
                u_ineq[row] = self.max_force
                row += 1

        if self.include_tau and self.cfg.qp.torque_limits is not None:
            tau_lim = float(self.cfg.qp.torque_limits)
            n_lim = n_h
            A_lim = np.zeros((n_lim, n_var))
            A_lim[:, idx["tau"]] = np.eye(n_h)
            l_lim = -tau_lim * np.ones(n_h)
            u_lim = tau_lim * np.ones(n_h)
            A = spa.vstack([spa.csc_matrix(A_eq), spa.csc_matrix(A_ineq), spa.csc_matrix(A_lim)])
            l = np.concatenate([l_eq, l_ineq, l_lim])
            u = np.concatenate([u_eq, u_ineq, u_lim])
        else:
            A = spa.vstack([spa.csc_matrix(A_eq), spa.csc_matrix(A_ineq)])
            l = np.concatenate([l_eq, l_ineq])
            u = np.concatenate([u_eq, u_ineq])

        res = solve_osqp(
            P,
            q,
            A,
            l,
            u,
            warm_start=warm_start,
            verbose=False,
            max_iter=4000,
        )
        if res["x"] is None:
            raise RuntimeError(f"Full QP failed: {res['status']}")

        x = res["x"]
        qddot = x[idx["q"]]
        a_obj = x[idx["a"]]
        forces = x[idx["f"]]

        if self.include_tau:
            tau = x[idx["tau"]]
        else:
            # tau = M qddot + C + sum J_i^T f_i
            tau = state.M_hand @ qddot + state.C_hand
            for i, cnt in enumerate(contacts):
                J = self.backend.hand_jacobian(cnt)
                tau += J.T @ forces[3 * i : 3 * (i + 1)]

        return FullQPResult(
            x=x,
            qddot=qddot,
            a_obj=a_obj,
            forces=forces,
            tau=tau,
            status=res["status"],
            solve_time=res["solve_time"],
            objective=res["objective"],
        )

    @staticmethod
    def forces_at_contact(result: FullQPResult, contact: Contact) -> np.ndarray:
        """Extract the 3D world-frame force assigned to a contact."""
        return result.forces[contact.force_slice]
