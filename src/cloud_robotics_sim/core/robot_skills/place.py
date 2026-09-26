"""Place skill primitive: approach -> descend -> open/detach -> retreat."""

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


class PlaceSkill(SkillPrimitive):
    """Place the currently attached object at a target pose.

    Phase/stage mapping (FAIL_STAGES): motion failures -> ``transport_fail``;
    release failures (descend/open/detach/retreat) -> ``place_fail``.
    """

    name = "place"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        place_pos: np.ndarray,
        place_quat: Optional[np.ndarray] = None,
        standoff: float = 0.10,
        retreat_height: float = 0.10,
    ) -> SkillResult:
        quat = (
            np.asarray(place_quat, dtype=np.float64) if place_quat is not None else None
        )
        place = np.asarray(place_pos, dtype=np.float64).reshape(3)
        approach = place + np.array([0.0, 0.0, standoff])
        retreat = place + np.array([0.0, 0.0, retreat_height])
        trajectories = []

        try:
            traj = context.plan(approach, quat)
            context.execute_trajectory(traj)
            trajectories.append(traj)
            traj = context.plan(place, quat)
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
