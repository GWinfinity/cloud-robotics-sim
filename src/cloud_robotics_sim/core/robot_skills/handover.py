"""Handover skill primitive: approach -> descend -> open/detach -> retreat.

Releases the attached object at a *free-space transfer pose* — the pose a
second arm (or a bin below) takes over at. Unlike :class:`PlaceSkill` the
transfer pose is not assumed to rest on a support surface, so the
standoff/retreat defaults leave more clearance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import (
    GRIPPER_OPEN,
    STATUS_SUCCESS,
    SkillContext,
    SkillPrimitive,
    SkillResult,
    _fail,
)


class HandoverSkill(SkillPrimitive):
    """Release the attached object at a free-space transfer pose.

    For bimanual transfer or dropping into a bin: unlike ``place`` the
    transfer pose is not assumed to rest on a support surface, so the
    standoff/retreat defaults leave more clearance.

    Phase/stage mapping (FAIL_STAGES): motion failures -> ``transport_fail``;
    release failures (descend/open/detach/retreat) -> ``place_fail``;
    missing entity pose -> ``load_fail``.

    Precondition: the object is already attached (run ``grasp`` first).
    """

    name = "handover"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        transfer_pos: np.ndarray,
        transfer_quat: Optional[np.ndarray] = None,
        standoff: float = 0.12,
        retreat_height: float = 0.15,
    ) -> SkillResult:
        quat = (
            np.asarray(transfer_quat, dtype=np.float64)
            if transfer_quat is not None
            else None
        )
        transfer = np.asarray(transfer_pos, dtype=np.float64).reshape(3)

        try:
            context.get_entity_pose(object_name)
        except Exception as exc:
            return _fail(self.name, "load_fail", f"no pose for '{object_name}': {exc}")

        approach = transfer + np.array([0.0, 0.0, standoff])
        retreat = transfer + np.array([0.0, 0.0, retreat_height])
        trajectories = []

        try:
            for goal in (approach, transfer):
                traj = context.plan(goal, quat)
                context.execute_trajectory(traj)
                trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "transport_fail", f"approach: {exc}")

        try:
            context.set_gripper(GRIPPER_OPEN)
            context.detach(object_name)
            traj = context.plan(retreat, quat)
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
