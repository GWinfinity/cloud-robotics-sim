"""Controller compiler + compliant Cartesian-to-joint mapper."""

from __future__ import annotations

import numpy as np

from .math_utils import compose, identity, matrix_to_pos_quat, pos_quat_to_matrix
from .sim_robot import FrankaSim
from .specs import ComposeSpec, ControllerProfile, StageSpec


class CartesianCompliantController:
    """Convert a composed SE(3) command into joint position targets.

    This is a simulation surrogate for the paper's controller compiler +
    impedance/admittance execution.  It:

    1. Adds admittance deflection from measured contact force.
    2. Converts the adjusted TCP pose to a hand pose using the fixed TCP offset.
    3. Solves IK with the current configuration as warm start.
    4. Sends joint position commands.
    """

    def __init__(
        self,
        robot: FrankaSim,
        tcp_offset: np.ndarray,
        profile: ControllerProfile | None = None,
    ) -> None:
        self.robot = robot
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        self.profile = profile or ControllerProfile("default")
        self._last_q = None

    def step(
        self,
        W_T_cmd: np.ndarray,
        contact_force: np.ndarray,
        task_z_in_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Execute one control tick.

        Returns:
            (q_target, info dict)
        """
        W_T_cmd = np.asarray(W_T_cmd, dtype=float)
        force = np.asarray(contact_force, dtype=float)

        # Admittance deflection: contact force pushes the command back.
        W_T_adjusted = self._apply_admittance(W_T_cmd, force, task_z_in_world)

        # Convert TCP command to hand command.
        W_T_hand_cmd = compose(W_T_adjusted, np.linalg.inv(self.tcp_offset))

        # Solve IK, warm-started from current q.
        init_q = self.robot.get_qpos()
        q_target = self.robot.ik(W_T_hand_cmd, init_qpos=init_q)

        # Send to low-level joint position controller.
        self.robot.set_joint_positions(q_target)
        self._last_q = q_target

        info = {
            "admittance_delta": (W_T_adjusted[:3, 3] - W_T_cmd[:3, 3]).copy(),
            "q_target": q_target.copy(),
        }
        return q_target, info

    def _apply_admittance(
        self,
        W_T_cmd: np.ndarray,
        force: np.ndarray,
        task_z_in_world: np.ndarray | None,
    ) -> np.ndarray:
        """Add a small pose deflection opposite to contact force."""
        if np.linalg.norm(force) < 0.1:
            return W_T_cmd

        task_z = (
            task_z_in_world
            if task_z_in_world is not None
            else np.array([0.0, 0.0, -1.0])
        )
        task_z = task_z / (np.linalg.norm(task_z) + 1e-9)

        # Project force onto task axes.
        f_axial = float(np.dot(force, task_z))
        f_lateral = force - f_axial * task_z

        delta = np.zeros(3)
        # Lateral compliance.
        mask = np.array(self.profile.force_axis, dtype=bool)
        active_lateral = np.zeros(3)
        if mask[:3].any():
            active_lateral = -self.profile.admittance_gain[:3] * f_lateral
            active_lateral = active_lateral * mask[:3]
            delta += active_lateral

        # Axial compliance if the insertion axis is force-controlled.
        if mask[2] if len(mask) >= 3 else False:
            delta += -self.profile.admittance_gain[2] * f_axial * task_z

        T_adj = identity()
        T_adj[:3, 3] = delta
        return compose(W_T_cmd, T_adj)


class ControllerCompiler:
    """Deterministic mapping from StageSpec to ControllerProfile."""

    @staticmethod
    def compile(stage: StageSpec, compose_spec: ComposeSpec) -> CartesianCompliantController:
        """Return a controller instance configured for the stage."""
        # In a full implementation this selects calibrated profiles by stage name.
        return CartesianCompliantController(
            robot=None,  # set later
            tcp_offset=identity(),
            profile=stage.profile,
        )

    @staticmethod
    def profile_library(name: str) -> ControllerProfile:
        """Built-in calibrated profiles."""
        if name == "free_space":
            return ControllerProfile(
                name="free_space",
                cartesian_stiffness=np.array([800.0, 800.0, 800.0, 80.0, 80.0, 80.0]),
                cartesian_damping=np.array([40.0, 40.0, 40.0, 4.0, 4.0, 4.0]),
                max_force=np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0]),
                admittance_gain=np.array([0.0, 0.0, 0.0]),
                force_axis=np.array([False, False, False]),
            )
        if name == "insertion":
            return ControllerProfile(
                name="insertion",
                cartesian_stiffness=np.array([400.0, 400.0, 200.0, 40.0, 40.0, 40.0]),
                cartesian_damping=np.array([20.0, 20.0, 10.0, 2.0, 2.0, 2.0]),
                max_force=np.array([10.0, 10.0, 15.0, 1.0, 1.0, 1.0]),
                admittance_gain=np.array([5e-5, 5e-5, 2e-5]),
                force_axis=np.array([True, True, True]),
            )
        if name == "contact":
            return ControllerProfile(
                name="contact",
                cartesian_stiffness=np.array([300.0, 300.0, 150.0, 30.0, 30.0, 30.0]),
                cartesian_damping=np.array([15.0, 15.0, 8.0, 1.5, 1.5, 1.5]),
                max_force=np.array([8.0, 8.0, 12.0, 1.0, 1.0, 1.0]),
                admittance_gain=np.array([1e-4, 1e-4, 5e-5]),
                force_axis=np.array([True, True, True]),
            )
        return ControllerProfile(name=name)
