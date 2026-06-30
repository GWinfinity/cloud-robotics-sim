"""Franka Panda wrapper for the CoStream simulation."""

from __future__ import annotations

import os

import genesis as gs
import numpy as np

from .math_utils import matrix_to_pos_quat, pos_quat_to_matrix


def _franka_mjcf_path() -> str:
    return os.path.join(
        os.path.dirname(gs.__file__),
        "assets/xml/franka_emika_panda/panda.xml",
    )


class FrankaSim:
    """Minimal Franka interface used by the CoStream demo.

    Provides:
      - end-effector (hand) pose queries
      - inverse kinematics
      - joint position control
      - a kinematically-followed tool (peg) for contact sensing
    """

    def __init__(
        self,
        scene: gs.Scene,
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.scene = scene
        self.entity = scene.add_entity(
            gs.morphs.MJCF(file=_franka_mjcf_path(), pos=base_pos),
        )
        self.hand: gs.engine.entities.RigidLink | None = None
        self.peg: gs.engine.entities.RigidEntity | None = None
        self.tcp_offset = np.eye(4)

    def build(self) -> None:
        """Call after scene.build() to resolve links."""
        self.hand = self.entity.get_link("hand")

    def set_tool(
        self,
        peg: gs.engine.entities.RigidEntity,
        tcp_offset: np.ndarray,
    ) -> None:
        """Attach a kinematic peg/tool and define the TCP offset."""
        self.peg = peg
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)

    def get_qpos(self) -> np.ndarray:
        return np.array(self.entity.get_qpos()).flatten()

    def get_hand_pose(self) -> np.ndarray:
        """Return 4x4 hand pose in world frame."""
        pos = np.array(self.hand.get_pos()).flatten()
        quat = np.array(self.hand.get_quat()).flatten()
        return pos_quat_to_matrix(pos, quat)

    def get_tcp_pose(self) -> np.ndarray:
        """Return 4x4 tool-center-point pose in world frame."""
        return self.get_hand_pose() @ self.tcp_offset

    def update_tool_pose(self) -> None:
        """Move the kinematic peg to track the current TCP."""
        if self.peg is None:
            return
        tcp = self.get_tcp_pose()
        pos, quat = matrix_to_pos_quat(tcp)
        self.peg.set_pos(pos)
        self.peg.set_quat(quat)

    def ik(
        self,
        target_hand_pose: np.ndarray,
        init_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve IK for the hand link."""
        target_hand_pose = np.asarray(target_hand_pose, dtype=float)
        pos, quat = matrix_to_pos_quat(target_hand_pose)
        if init_qpos is None:
            init_qpos = self.get_qpos()
        q = self.entity.inverse_kinematics(
            link=self.hand,
            pos=pos,
            quat=quat,
            init_qpos=init_qpos,
            max_samples=20,
            max_solver_iters=20,
            return_error=False,
        )
        return np.array(q).flatten()

    def set_joint_positions(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self.entity.control_dofs_position(q)

    def get_contact_force(self) -> np.ndarray:
        """Return net contact force on the tool in world frame."""
        if self.peg is None:
            return np.zeros(3)
        force = self.peg.get_links_net_contact_force()
        return np.array(force).flatten()[-3:]

    def object_in_hand_pose(self) -> np.ndarray:
        """Return tool pose expressed in the hand frame.

        In the real robot this is the GelSight/tracking estimate.  Here we
        compute it from ground-truth simulation state.
        """
        if self.peg is None:
            return np.eye(4)
        hand_T = self.get_hand_pose()
        peg_pos = np.array(self.peg.get_pos()).flatten()
        peg_quat = np.array(self.peg.get_quat()).flatten()
        peg_T = pos_quat_to_matrix(peg_pos, peg_quat)
        return np.linalg.inv(hand_T) @ peg_T
