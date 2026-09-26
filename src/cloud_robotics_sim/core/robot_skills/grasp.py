"""Grasp skill primitive: approach -> descend -> close/attach -> lift."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import (
    GRIPPER_CLOSED,
    STATUS_SUCCESS,
    SkillContext,
    SkillPrimitive,
    SkillResult,
    _fail,
)


class GraspSkill(SkillPrimitive):
    """Top-down grasp of a named scene object.

    Phase/stage mapping (FAIL_STAGES): planning failures -> ``plan_fail``;
    close/attach failures -> ``grasp_fail``; lift failures -> ``lift_fail``.
    """

    name = "grasp"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        grasp_pos: Optional[np.ndarray] = None,
        grasp_quat: Optional[np.ndarray] = None,
        standoff: float = 0.10,
        lift_height: float = 0.15,
    ) -> SkillResult:
        quat = (
            np.asarray(grasp_quat, dtype=np.float64) if grasp_quat is not None else None
        )
        try:
            obj_pos, _ = context.get_entity_pose(object_name)
        except Exception as exc:
            return _fail(self.name, "load_fail", f"no pose for '{object_name}': {exc}")
        if grasp_pos is None:
            # Fallback: object center lifted slightly — callers with real
            # assets should pass a candidate from grasp_points instead.
            grasp = np.asarray(obj_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.01])
        else:
            grasp = np.asarray(grasp_pos, dtype=np.float64).reshape(3)

        pre_grasp = grasp + np.array([0.0, 0.0, standoff])
        lift = grasp + np.array([0.0, 0.0, lift_height])
        trajectories = []

        for stage, goal in (("pre-grasp", pre_grasp), ("descend", grasp)):
            try:
                traj = context.plan(goal, quat)
                context.execute_trajectory(traj)
                trajectories.append(traj)
            except (
                Exception
            ) as exc:  # PlannerError, SkillExecutionError, backend errors
                return _fail(self.name, "plan_fail", f"{stage}: {exc}")

        try:
            context.set_gripper(GRIPPER_CLOSED)
            context.attach(object_name)
        except Exception as exc:
            return _fail(self.name, "grasp_fail", str(exc))

        try:
            traj = context.plan(lift, quat)
            context.execute_trajectory(traj)
            trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "lift_fail", f"lift: {exc}")

        return SkillResult(
            success=True,
            stage=STATUS_SUCCESS,
            skill=self.name,
            trajectories=trajectories,
        )
