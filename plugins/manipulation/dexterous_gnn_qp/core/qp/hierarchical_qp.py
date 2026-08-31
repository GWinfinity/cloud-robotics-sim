"""Hierarchical QP: optimize only skeleton contacts, fix edge forces.

Implements the full alternating-iteration scheme from the LIFT paper:
  1. Solve skeleton-layer QP
  2. Compute CoM dynamics residual
  3. Update edge forces via gradient correction
  4. Project edge forces into friction cones
  5. Repeat until convergence or max iterations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import scipy.sparse as spa

from dexterous_gnn_qp.core.backends.base import DynamicsBackend
from dexterous_gnn_qp.core.env.sim import Contact, SimState
from dexterous_gnn_qp.core.qp.osqp_utils import solve_osqp
from dexterous_gnn_qp.core.utils.config import DotDict
from dexterous_gnn_qp.core.utils.math import skew


@dataclass
class HierarchicalQPResult:
    """Result of the hierarchical QP."""

    x: np.ndarray
    qddot: np.ndarray
    a_obj: np.ndarray
    skeleton_forces: np.ndarray
    tau: np.ndarray
    full_forces: np.ndarray
    status: str
    solve_time: float
    objective: float
    # Alternating-iteration diagnostics
    n_iterations: int = 1
    residual_norm: float = 0.0
    converged: bool = True


def _index_slices(n_hand: int, n_skeleton: int, include_tau: bool) -> Dict[str, slice]:
    n_f = 3 * n_skeleton
    idx = {
        "q": slice(0, n_hand),
        "a": slice(n_hand, n_hand + 6),
        "f": slice(n_hand + 6, n_hand + 6 + n_f),
    }
    if include_tau:
        idx["tau"] = slice(n_hand + 6 + n_f, n_hand + 6 + n_f + n_hand)
    return idx


def _project_force_to_friction_cone(
    force: np.ndarray,
    normal: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    mu: float,
    kappa: float = 1.0,
) -> np.ndarray:
    """Project a 3D force onto the friction cone with optional shrinkage factor.

    The friction cone is: f_n >= 0, |f_t| <= kappa * mu * f_n
    where f_n = normal^T f, f_t = [tangent1^T f, tangent2^T f].

    If the force is already inside the cone, return it unchanged.
    Otherwise, project onto the nearest point on the cone boundary.
    """
    f_n = float(np.dot(normal, force))
    f_t1 = float(np.dot(tangent1, force))
    f_t2 = float(np.dot(tangent2, force))
    f_t = np.array([f_t1, f_t2])
    f_t_norm = float(np.linalg.norm(f_t))

    # Effective friction with shrinkage
    mu_eff = kappa * mu

    # Check if inside cone
    if f_n >= 0 and f_t_norm <= mu_eff * f_n + 1e-10:
        return np.asarray(force, dtype=float).copy()

    # Project onto cone boundary
    if f_n < 0:
        # Force pointing into contact surface -> project to zero
        if f_t_norm <= 1e-10:
            return np.zeros(3)
        # Project onto the cone edge with f_n = 0
        return f_t1 * tangent1 + f_t2 * tangent2

    # f_n >= 0 but |f_t| > mu_eff * f_n
    # Project onto the cone surface: |f_t| = mu_eff * f_n
    scale = mu_eff * f_n / (f_t_norm + 1e-10)
    f_t_proj = f_t * scale
    f_n_proj = f_n  # Keep normal component

    return f_n_proj * normal + f_t_proj[0] * tangent1 + f_t_proj[1] * tangent2


def _compute_residual(
    backend: DynamicsBackend,
    state: SimState,
    contacts: List[Contact],
    skeleton_indices: List[int],
    edge_indices: List[int],
    qddot: np.ndarray,
    a_obj: np.ndarray,
    tau: np.ndarray,
    skeleton_forces: np.ndarray,
    edge_forces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the dynamics residual for both hand and object.

    Hand dynamics:
    r_hand = M_hand @ qddot + C_hand - tau - sum_{i in S} J_i^T f_i - sum_{j in E} J_j^T f̂_j

    Object dynamics:
    r_obj = M_obj @ a_obj + C_obj - sum_{i in S} G_i^T f_i - sum_{j in E} G_j^T f̂_j

    Returns (r_hand, r_obj) tuple.
    """
    n_h = backend.n_hand_dof

    # Hand dynamics residual
    r_hand = state.M_hand @ qddot + state.C_hand - tau
    for local_i, global_i in enumerate(skeleton_indices):
        cnt = contacts[global_i]
        J = backend.hand_jacobian(cnt)
        f_i = skeleton_forces[local_i * 3 : (local_i + 1) * 3]
        r_hand -= J.T @ f_i
    for global_i in edge_indices:
        cnt = contacts[global_i]
        J = backend.hand_jacobian(cnt)
        f_j = edge_forces[global_i * 3 : (global_i + 1) * 3]
        r_hand -= J.T @ f_j

    # Object dynamics residual
    r_obj = state.M_obj @ a_obj + state.C_obj
    for local_i, global_i in enumerate(skeleton_indices):
        cnt = contacts[global_i]
        r = cnt.pos - state.x_obj
        Gt = np.vstack([np.eye(3), skew(r)])
        f_i = skeleton_forces[local_i * 3 : (local_i + 1) * 3]
        r_obj -= Gt @ f_i
    for global_i in edge_indices:
        cnt = contacts[global_i]
        r = cnt.pos - state.x_obj
        Gt = np.vstack([np.eye(3), skew(r)])
        f_j = edge_forces[global_i * 3 : (global_i + 1) * 3]
        r_obj -= Gt @ f_j

    return r_hand, r_obj


class HierarchicalQP:
    """Reduced QP that treats only skeleton contacts as decision variables.

    Supports alternating iteration between skeleton and edge layers:
      - Skeleton layer: solve QP for skeleton contacts only
      - Edge layer: update edge forces via residual-based gradient correction
      - Iterate until convergence or max iterations
    """

    def __init__(self, backend: DynamicsBackend, cfg: DotDict):
        self.backend = backend
        self.cfg = cfg
        self.w = cfg.qp.weights
        self.include_tau = bool(cfg.qp.get("include_tau", False))
        self.max_force = cfg.qp.get("max_force_per_contact", None)
        if self.max_force is not None:
            self.max_force = float(self.max_force)

        # Alternating iteration parameters
        hier_cfg = cfg.get("hierarchical", DotDict({}))
        self.max_iterations = int(hier_cfg.get("max_iterations", 1))
        self.residual_tol = float(hier_cfg.get("residual_tol", 1e-3))
        self.edge_update_gain = float(hier_cfg.get("edge_update_gain", 0.5))
        self.friction_cone_shrinkage = float(hier_cfg.get("friction_cone_shrinkage", 1.0))
        self.enable_alternating = bool(hier_cfg.get("enable_alternating", False))

    def _solve_single_iteration(
        self,
        state: SimState,
        skeleton_indices: List[int],
        edge_forces: np.ndarray,
        qddot_des: np.ndarray,
        a_obj_des: np.ndarray,
        warm_start: np.ndarray | None = None,
    ) -> HierarchicalQPResult:
        """Solve a single iteration of the hierarchical QP (skeleton layer only)."""
        n_h = self.backend.n_hand_dof
        contacts = state.contacts
        n_c = len(contacts)
        n_s = len(skeleton_indices)
        idx = _index_slices(n_h, n_s, self.include_tau)
        n_var = n_h + 6 + 3 * n_s + (n_h if self.include_tau else 0)

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
            A_eq[:n_h, idx["q"]] = state.M_hand
            A_eq[:n_h, idx["tau"]] = -np.eye(n_h)
            rhs_hand = -state.C_hand.copy()
            for local_i, global_i in enumerate(skeleton_indices):
                cnt = contacts[global_i]
                J = self.backend.hand_jacobian(cnt)
                A_eq[:n_h, idx["f"].start + 3 * local_i : idx["f"].start + 3 * (local_i + 1)] = J.T
            for global_i in range(n_c):
                if global_i in skeleton_indices:
                    continue
                cnt = contacts[global_i]
                f_edge = edge_forces[global_i * 3 : (global_i + 1) * 3]
                J = self.backend.hand_jacobian(cnt)
                rhs_hand -= J.T @ f_edge
            l_eq[:n_h] = rhs_hand
            u_eq[:n_h] = rhs_hand
            row = n_h

        # Object dynamics. Free-joint qvel is [linear; angular], so G = [I; skew(r)].
        A_eq[row : row + 6, idx["a"]] = state.M_obj
        rhs_obj = -state.C_obj.copy()
        for local_i, global_i in enumerate(skeleton_indices):
            cnt = contacts[global_i]
            r = cnt.pos - state.x_obj
            Gt = np.vstack([np.eye(3), skew(r)])
            A_eq[row : row + 6, idx["f"].start + 3 * local_i : idx["f"].start + 3 * (local_i + 1)] = -Gt
        for global_i in range(n_c):
            if global_i in skeleton_indices:
                continue
            cnt = contacts[global_i]
            f_edge = edge_forces[global_i * 3 : (global_i + 1) * 3]
            r = cnt.pos - state.x_obj
            Gt = np.vstack([np.eye(3), skew(r)])
            rhs_obj += Gt @ f_edge
        l_eq[row : row + 6] = rhs_obj
        u_eq[row : row + 6] = rhs_obj

        # Friction pyramid + optional normal-force limit for skeleton contacts.
        n_force_bounds = n_s if self.max_force is not None else 0
        n_ineq = 4 * n_s + n_force_bounds
        A_ineq = np.zeros((n_ineq, n_var))
        l_ineq = -np.inf * np.ones(n_ineq)
        u_ineq = np.zeros(n_ineq)
        row = 0
        for local_i, global_i in enumerate(skeleton_indices):
            cnt = contacts[global_i]
            f_start = idx["f"].start + 3 * local_i
            A_ineq[row, f_start : f_start + 3] = cnt.normal
            l_ineq[row] = 0.0
            u_ineq[row] = np.inf
            row += 1
            for t in (cnt.tangent1, cnt.tangent2):
                A_ineq[row, f_start : f_start + 3] = t - cnt.mu * cnt.normal
                l_ineq[row] = -np.inf
                u_ineq[row] = 0.0
                row += 1
        if self.max_force is not None:
            for local_i, global_i in enumerate(skeleton_indices):
                cnt = contacts[global_i]
                f_start = idx["f"].start + 3 * local_i
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
            raise RuntimeError(f"Hierarchical QP failed: {res['status']}")

        x = res["x"]
        qddot = x[idx["q"]]
        a_obj = x[idx["a"]]
        skeleton_forces = x[idx["f"]]

        if self.include_tau:
            tau = x[idx["tau"]]
        else:
            tau = state.M_hand @ qddot + state.C_hand
            for local_i, global_i in enumerate(skeleton_indices):
                cnt = contacts[global_i]
                J = self.backend.hand_jacobian(cnt)
                tau += J.T @ skeleton_forces[local_i * 3 : (local_i + 1) * 3]
            for global_i in range(n_c):
                if global_i in skeleton_indices:
                    continue
                cnt = contacts[global_i]
                f_edge = edge_forces[global_i * 3 : (global_i + 1) * 3]
                J = self.backend.hand_jacobian(cnt)
                tau += J.T @ f_edge

        # Reconstruct full force vector for comparison.
        full_forces = np.array(edge_forces)
        for local_i, global_i in enumerate(skeleton_indices):
            full_forces[global_i * 3 : (global_i + 1) * 3] = skeleton_forces[
                local_i * 3 : (local_i + 1) * 3
            ]

        return HierarchicalQPResult(
            x=x,
            qddot=qddot,
            a_obj=a_obj,
            skeleton_forces=skeleton_forces,
            tau=tau,
            full_forces=full_forces,
            status=res["status"],
            solve_time=res["solve_time"],
            objective=res["objective"],
        )

    def solve(
        self,
        state: SimState,
        skeleton_indices: List[int],
        edge_forces: np.ndarray | None,
        qddot_des: np.ndarray,
        a_obj_des: np.ndarray,
        warm_start: np.ndarray | None = None,
    ) -> HierarchicalQPResult:
        """Solve the hierarchical QP with optional alternating iteration.

        If enable_alternating is False (default), performs a single solve.
        Otherwise, iterates between skeleton and edge layers.
        """
        n_c = len(state.contacts)

        # Initialize edge forces
        if edge_forces is None or len(edge_forces) != 3 * n_c:
            edge_forces = np.zeros(3 * n_c)

        # Compute edge indices
        skeleton_set = set(skeleton_indices)
        edge_indices = [i for i in range(n_c) if i not in skeleton_set]

        # Single iteration mode (backward compatible)
        if not self.enable_alternating or self.max_iterations <= 1:
            result = self._solve_single_iteration(
                state, skeleton_indices, edge_forces,
                qddot_des, a_obj_des, warm_start
            )
            result.n_iterations = 1
            result.converged = True
            return result

        # Alternating iteration mode
        total_solve_time = 0.0
        best_result = None
        current_edge_forces = edge_forces.copy()

        for iteration in range(self.max_iterations):
            # Step 1: Solve skeleton-layer QP
            result = self._solve_single_iteration(
                state, skeleton_indices, current_edge_forces,
                qddot_des, a_obj_des, warm_start
            )
            total_solve_time += result.solve_time
            best_result = result

            # Step 2: Compute residual
            r_hand, r_obj = _compute_residual(
                self.backend, state, state.contacts,
                skeleton_indices, edge_indices,
                result.qddot, result.a_obj, result.tau,
                result.skeleton_forces, current_edge_forces
            )
            # Use object dynamics residual for edge force update
            # (edge forces directly affect object dynamics)
            residual = r_obj
            residual_norm = float(np.linalg.norm(residual))

            # Update result diagnostics
            result.n_iterations = iteration + 1
            result.residual_norm = residual_norm
            result.converged = residual_norm < self.residual_tol
            result.solve_time = total_solve_time

            # Check convergence
            if result.converged:
                break

            # Step 3: Update edge forces via gradient correction
            # f̂_j ← f̂_j + K_j · G_j^T · r_obj
            # where G_j is the grasp map: G = [I; skew(r)]
            # We use a simplified update: f̂_j += gain * G_j^T * r_obj
            for global_i in edge_indices:
                cnt = state.contacts[global_i]
                r = cnt.pos - state.x_obj
                G = np.vstack([np.eye(3), skew(r)])  # 6x3 matrix
                # Gradient direction: G^T @ r_obj (3x6 @ 6 = 3)
                gradient = G.T @ residual
                # Update edge force
                current_edge_forces[global_i * 3 : (global_i + 1) * 3] += (
                    self.edge_update_gain * gradient
                )

            # Step 4: Project edge forces into friction cones
            for global_i in edge_indices:
                cnt = state.contacts[global_i]
                f_start = global_i * 3
                f_edge = current_edge_forces[f_start : f_start + 3]
                f_projected = _project_force_to_friction_cone(
                    f_edge, cnt.normal, cnt.tangent1, cnt.tangent2,
                    cnt.mu, self.friction_cone_shrinkage
                )
                current_edge_forces[f_start : f_start + 3] = f_projected

        # Reconstruct final full forces with updated edge forces
        if best_result is not None:
            full_forces = current_edge_forces.copy()
            for local_i, global_i in enumerate(skeleton_indices):
                full_forces[global_i * 3 : (global_i + 1) * 3] = (
                    best_result.skeleton_forces[local_i * 3 : (local_i + 1) * 3]
                )
            best_result.full_forces = full_forces

        return best_result
