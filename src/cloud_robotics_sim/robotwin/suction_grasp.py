"""Suction-style grasping and pick-and-place sequencing for the FR3 arm.

The ``franka_fr3_v2`` model is a bare 7-DoF arm without a gripper, so grasping
is emulated with a **suction cup** at the flange (``fr3v2_link8``):

- :class:`SuctionGrasper.attach` records the object's pose relative to the
  flange once the flange touches the object top;
- while attached, :meth:`SuctionGrasper.follow_step` kinematically binds the
  object to the flange (``set_qpos`` on the free base each sim step);
- :meth:`SuctionGrasper.detach` releases the object back to physics.

Motion helpers (:func:`goto_joints`, :func:`follow_path`) execute joint-space
motions while keeping the suction binding updated every step.

Orientation convention: the FR3 home pose already points the flange straight
down (180° about a horizontal axis), so the home EE quaternion
(:data:`GRASP_QUAT`) is reused as the top-down grasp orientation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "GRASP_QUAT",
    "HOME_QPOS",
    "GraspPhase",
    "SuctionGrasper",
    "follow_path",
    "goto_joints",
    "quat_conj",
    "quat_mul",
    "quat_rotate",
]


class GraspPhase(IntEnum):
    """Phase labels recorded per frame in grasp demonstration episodes."""

    RESET = 0
    PRE_GRASP = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    TRANSPORT = 5
    PLACE = 6
    RELEASE = 7
    RETREAT = 8


#: Callback invoked after every sim step with the current joint target.
StepHook = Callable[[np.ndarray], None]

#: FR3 home joint configuration (from the MJCF ``home`` keyframe).
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

#: EE (flange) quaternion ``(w, x, y, z)`` at the home pose — flange facing
#: straight down; reused as the top-down grasp orientation.
GRASP_QUAT = np.array([-0.0004, 0.9239, 0.3826, -0.0009])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions, ``(w, x, y, z)`` convention."""
    w1, x1, y1, z1 = (float(v) for v in q1)
    w2, x2, y2, z2 = (float(v) for v in q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions), ``(w, x, y, z)``."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``, ``(w, x, y, z)``."""
    q_v = np.concatenate([[0.0], v])
    return quat_mul(quat_mul(q, q_v), quat_conj(q))[1:]


@dataclass
class SuctionGrasper:
    """Kinematic suction binding between the FR3 flange and a free object."""

    robot: Any  # Genesis articulation entity (FR3)
    obj: Any  # Genesis entity (rigid GLB or articulated URDF object)
    ee_link_name: str = "fr3v2_link8"
    attached: bool = False
    _offset_pos: np.ndarray | None = None  # object pos in EE frame
    _offset_quat: np.ndarray | None = None  # relative rotation (wxyz)

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current EE world pose as ``(pos, quat_wxyz)`` numpy arrays."""
        link = self.robot.get_link(self.ee_link_name)
        return _to_numpy(link.get_pos()), _to_numpy(link.get_quat())

    def obj_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Object base world pose ``(pos, quat_wxyz)``."""
        qpos = _to_numpy(self.obj.get_qpos()).reshape(-1)
        return qpos[:3].copy(), qpos[3:7].copy()

    # ------------------------------------------------------------------
    # Suction lifecycle
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Bind the object to the flange at its current relative pose."""
        ee_pos, ee_quat = self.ee_pose()
        obj_pos, obj_quat = self.obj_base_pose()
        self._offset_pos = quat_rotate(quat_conj(ee_quat), obj_pos - ee_pos)
        self._offset_quat = quat_mul(quat_conj(ee_quat), obj_quat)
        self.attached = True

    def detach(self) -> None:
        """Release the object (physics takes over again)."""
        self.attached = False
        self._offset_pos = None
        self._offset_quat = None

    def follow_step(self) -> None:
        """Re-pin the object to the flange; call once per sim step."""
        if not self.attached:
            return
        assert self._offset_pos is not None and self._offset_quat is not None
        ee_pos, ee_quat = self.ee_pose()
        new_pos = ee_pos + quat_rotate(ee_quat, self._offset_pos)
        new_quat = quat_mul(ee_quat, self._offset_quat)
        qpos = _to_numpy(self.obj.get_qpos()).reshape(-1)
        qpos[:3] = new_pos
        qpos[3:7] = new_quat / np.linalg.norm(new_quat)
        self.obj.set_qpos(qpos)

    def is_holding(self, tolerance: float = 0.05) -> bool:
        """Sanity check: object is still at its expected pose under the EE.

        Compares the actual object base position with the position implied by
        the flange pose + the offset captured at :meth:`attach` (robust to
        tall objects whose origin sits far below the flange).
        """
        if not self.attached or self._offset_pos is None:
            return False
        ee_pos, ee_quat = self.ee_pose()
        expected = ee_pos + quat_rotate(ee_quat, self._offset_pos)
        obj_pos, _ = self.obj_base_pose()
        return bool(np.linalg.norm(obj_pos - expected) < tolerance)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors (possibly CUDA) or arrays to float64 numpy."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            result: np.ndarray = x.detach().cpu().numpy().astype(np.float64)
            return result
    except ImportError:  # pragma: no cover - torch always present in practice
        pass
    return np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------
# Motion helpers
# ----------------------------------------------------------------------


def follow_path(
    scene: Any,
    robot: Any,
    path: np.ndarray,
    grasper: SuctionGrasper | None = None,
    steps_per_waypoint: int = 3,
    on_step: StepHook | None = None,
) -> None:
    """Execute a joint-space path with position control + suction updates."""
    path = np.asarray(path, dtype=np.float64)
    if path.ndim == 1:
        path = path[None, :]
    for waypoint in path:
        robot.control_dofs_position(waypoint)
        for _ in range(steps_per_waypoint):
            scene.step()
            if grasper is not None:
                grasper.follow_step()
            if on_step is not None:
                on_step(waypoint)


def goto_joints(
    scene: Any,
    robot: Any,
    q_target: np.ndarray,
    grasper: SuctionGrasper | None = None,
    n_steps: int = 25,
    settle_steps: int = 5,
    on_step: StepHook | None = None,
) -> None:
    """Linearly interpolate joints from the current config to ``q_target``.

    Intended for short, collision-safe moves (vertical descend / lift) where
    calling a motion planner would be overkill.
    """
    q0 = _to_numpy(robot.get_qpos()).reshape(-1)[: len(q_target)]
    q_target = np.asarray(q_target, dtype=np.float64)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q_now = q0 + alpha * (q_target - q0)
        robot.control_dofs_position(q_now)
        scene.step()
        if grasper is not None:
            grasper.follow_step()
        if on_step is not None:
            on_step(q_now)
    robot.control_dofs_position(q_target)
    for _ in range(settle_steps):
        scene.step()
        if grasper is not None:
            grasper.follow_step()
        if on_step is not None:
            on_step(q_target)
