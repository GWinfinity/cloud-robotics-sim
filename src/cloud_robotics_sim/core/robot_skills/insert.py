"""Insert skill primitive: align -> descend along axis -> open/detach -> retreat.

Axis-aligned peg-in-hole insertion of the attached object. ``axis`` is the
insertion direction (from the mouth toward the final pose); the default
``[0, 0, -1]`` inserts downward.
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

#: Default insertion direction: straight down.
DEFAULT_AXIS = np.array([0.0, 0.0, -1.0])


class InsertSkill(SkillPrimitive):
    """Axis-aligned insertion (peg-in-hole) of the attached object.

    Phases: align above the mouth -> descend along ``axis`` to the insert
    pose -> open/detach -> retreat along ``axis``.

    Phase/stage mapping (FAIL_STAGES): motion failures -> ``transport_fail``;
    release failures -> ``place_fail``; missing entity pose -> ``load_fail``;
    degenerate axis (zero norm) -> ``error``.

    Precondition: the object is already attached (run ``grasp`` first).
    """

    name = "insert"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        insert_pos: np.ndarray,
        mouth_pos: Optional[np.ndarray] = None,
        axis: Optional[np.ndarray] = None,
        standoff: float = 0.08,
        retreat_height: float = 0.10,
        insert_quat: Optional[np.ndarray] = None,
    ) -> SkillResult:
        axis_arr = (
            DEFAULT_AXIS.copy()
            if axis is None
            else np.asarray(axis, dtype=np.float64).reshape(3)
        )
        norm = float(np.linalg.norm(axis_arr))
        if norm < 1e-9:
            return _fail(
                self.name, "error", f"degenerate insertion axis {axis_arr.tolist()}"
            )
        axis_arr = axis_arr / norm

        try:
            context.get_entity_pose(object_name)
        except Exception as exc:
            return _fail(self.name, "load_fail", f"no pose for '{object_name}': {exc}")

        quat = (
            np.asarray(insert_quat, dtype=np.float64)
            if insert_quat is not None
            else None
        )
        insert = np.asarray(insert_pos, dtype=np.float64).reshape(3)
        # No explicit mouth: treat the insert pose itself as the mouth, so the
        # approach matches the grasp pre-grasp pattern (offset along -axis).
        mouth = (
            insert
            if mouth_pos is None
            else np.asarray(mouth_pos, dtype=np.float64).reshape(3)
        )

        approach = mouth - axis_arr * standoff
        retreat = insert - axis_arr * retreat_height
        trajectories = []

        try:
            for stage, goal in (("align", approach), ("insert", insert)):
                traj = context.plan(goal, quat)
                context.execute_trajectory(traj)
                trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "transport_fail", f"{stage}: {exc}")

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
