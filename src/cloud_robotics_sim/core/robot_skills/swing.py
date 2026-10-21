"""Joint-swing skill primitives: ``open`` / ``close`` articulated objects.

Both registered primitives share one phase script (the
:class:`SwingJointSkill` base class, itself *not* registered): approach the
handle -> descend -> close/attach -> swing the joint through an angular
sweep while the EE follows the arc (Rodrigues rotation about the joint
axis) -> open/detach -> retreat. They differ only in the default sweep
sign: ``open`` sweeps positive, ``close`` sweeps negative.

Environment requirements (optional ``SkillContext`` hooks):
``get_joint_info`` and ``drive_joint``. Missing hooks -> stage ``error``.
End-to-end use needs articulated assets plus a link-level grasp mode: the
current suction grasper binds the object *base* pose, which cannot hold a
drawer handle (see R-A in the P1 plan).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from .base import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    STATUS_SUCCESS,
    SkillContext,
    SkillPrimitive,
    SkillResult,
    _fail,
)

logger = logging.getLogger(__name__)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix (3x3) of ``angle`` rad about unit ``axis``."""
    x, y, z = np.asarray(axis, dtype=np.float64).reshape(3)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.asarray(
        np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k),
        dtype=np.float64,
    )


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of ``angle`` rad about unit ``axis``."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = (float(v) for v in np.asarray(a, dtype=np.float64).reshape(4))
    bw, bx, by, bz = (float(v) for v in np.asarray(b, dtype=np.float64).reshape(4))
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


class SwingJointSkill(SkillPrimitive):
    """Swing an articulated object's joint while holding its handle.

    Base class for the registered ``open`` / ``close`` primitives (not
    registered itself). Phases: approach (pre-grasp above the handle) ->
    descend -> close/attach -> swing (drive the joint through an angular
    sweep while the EE follows the arc: Rodrigues rotation about the joint
    axis, anchor at the joint) -> open/detach -> retreat.

    Phase/stage mapping (FAIL_STAGES): approach/descend failures ->
    ``transport_fail``; close/attach failures -> ``grasp_fail``; swing /
    release / retreat failures -> ``place_fail``; missing entity pose ->
    ``load_fail``; missing hooks (``get_joint_info`` / ``drive_joint``) ->
    ``error``; joint lookup failure / non-revolute / degenerate axis ->
    ``load_fail``.

    Precondition: nothing is attached yet (this primitive performs its own
    close/attach on the handle).
    """

    name = "swing"  # base class; only "open"/"close" are registered
    default_angle: float = 0.0

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        joint_name: str,
        angle: Optional[float] = None,
        grasp_pos: Optional[np.ndarray] = None,
        grasp_quat: Optional[np.ndarray] = None,
        standoff: float = 0.10,
        swing_steps: int = 5,
        retreat_height: float = 0.10,
    ) -> SkillResult:
        try:
            obj_pos, _ = context.get_entity_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        except Exception as exc:
            return _fail(self.name, "load_fail", f"no pose for '{object_name}': {exc}")

        get_joint_info = context.get_joint_info
        drive_joint = context.drive_joint
        if get_joint_info is None or drive_joint is None:
            missing = [
                hook
                for hook, present in (
                    ("get_joint_info", get_joint_info is not None),
                    ("drive_joint", drive_joint is not None),
                )
                if not present
            ]
            return _fail(
                self.name,
                "error",
                f"{self.name} requires the {missing[0]} hook "
                "(articulated-object environments only)",
            )

        try:
            info = get_joint_info(object_name, joint_name)
        except Exception as exc:
            return _fail(
                self.name,
                "load_fail",
                f"joint '{joint_name}' on '{object_name}': {exc}",
            )
        axis = np.asarray(info.axis, dtype=np.float64).reshape(3)
        anchor = np.asarray(info.anchor, dtype=np.float64).reshape(3)

        quat = (
            np.asarray(grasp_quat, dtype=np.float64) if grasp_quat is not None else None
        )
        # Fallback grasp point: object center lifted slightly — callers with
        # real articulated assets should pass the handle pose instead.
        grasp = (
            np.asarray(obj_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.01])
            if grasp_pos is None
            else np.asarray(grasp_pos, dtype=np.float64).reshape(3)
        )
        pre_grasp = grasp + np.array([0.0, 0.0, standoff])

        try:
            for stage, goal in (("approach", pre_grasp), ("descend", grasp)):
                traj = context.plan(goal, quat)
                context.execute_trajectory(traj)
        except Exception as exc:
            return _fail(self.name, "transport_fail", f"{stage}: {exc}")

        try:
            context.set_gripper(GRIPPER_CLOSED)
            context.attach(object_name)
        except Exception as exc:
            return _fail(self.name, "grasp_fail", str(exc))

        try:
            steps = int(swing_steps)
        except (TypeError, ValueError):
            return _fail(
                self.name, "error", f"swing_steps must be an int >= 1: {swing_steps!r}"
            )
        if steps < 1:
            return _fail(self.name, "error", "swing_steps must be >= 1")
        sweep = self.default_angle if angle is None else float(angle)

        theta0 = float(info.position)
        thetas = np.linspace(theta0, theta0 + sweep, steps + 1)[1:]
        trajectories = []
        last_goal = grasp
        last_delta = 0.0
        try:
            for theta in thetas:
                target = float(np.clip(theta, info.lower, info.upper))
                if abs(target - theta) > 1e-9:
                    logger.warning(
                        "%s: swing target %.3f clipped to %.3f (joint limits)",
                        self.name,
                        theta,
                        target,
                    )
                drive_joint(object_name, joint_name, target)
                delta = target - theta0
                last_goal = anchor + _rodrigues(axis, delta) @ (grasp - anchor)
                last_delta = delta
                goal_quat = (
                    None
                    if quat is None
                    else _quat_mul(_axis_angle_quat(axis, delta), quat)
                )
                traj = context.plan(last_goal, goal_quat)
                context.execute_trajectory(traj)
                trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "place_fail", f"swing: {exc}")

        try:
            context.set_gripper(GRIPPER_OPEN)
            context.detach(object_name)
            retreat = last_goal + np.array([0.0, 0.0, retreat_height])
            retreat_quat = (
                None
                if quat is None
                else _quat_mul(_axis_angle_quat(axis, last_delta), quat)
            )
            traj = context.plan(retreat, retreat_quat)
            context.execute_trajectory(traj)
            trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "place_fail", str(exc))

        return SkillResult(
            success=True,
            stage=STATUS_SUCCESS,
            skill=self.name,
            trajectories=trajectories,
        )


class OpenSkill(SwingJointSkill):
    """Open an articulated object (drawer/lid): positive joint sweep.

    Phase/stage mapping: see :class:`SwingJointSkill`. The sweep sign
    depends on the joint's axis direction; pass ``angle`` explicitly when
    the asset's axis convention is known.
    """

    name = "open"
    default_angle = float(np.pi) / 3.0  # +60 deg


class CloseSkill(SwingJointSkill):
    """Close an articulated object (drawer/lid): negative joint sweep.

    Phase/stage mapping: see :class:`SwingJointSkill`. The sweep sign
    depends on the joint's axis direction; pass ``angle`` explicitly when
    the asset's axis convention is known.
    """

    name = "close"
    default_angle = -float(np.pi) / 3.0  # -60 deg
