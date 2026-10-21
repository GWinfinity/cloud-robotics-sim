"""Stack skill primitive: place the attached object on top of a support object.

Target = support xy + (support top + carried half-height). Heights come
from the ``get_entity_extents`` hook unless the explicit
``support_top_z`` / ``carried_half_height`` params are given;
``target_pos`` overrides the computed target entirely.
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


class _ResolveError(Exception):
    """Carries a prepared failure SkillResult out of the resolver helper."""

    def __init__(self, result: SkillResult) -> None:
        super().__init__(result.message)
        self.result = result


def _half_extents(context: SkillContext, name: str, skill: str) -> np.ndarray:
    """``(3,)`` half-extents of ``name`` via the hook; raises _ResolveError."""
    if context.get_entity_extents is None:
        raise _ResolveError(
            _fail(
                skill,
                "error",
                "stack requires the get_entity_extents hook or explicit "
                "support_top_z / carried_half_height params",
            )
        )
    try:
        return np.asarray(context.get_entity_extents(name), dtype=np.float64).reshape(3)
    except Exception as exc:
        raise _ResolveError(
            _fail(skill, "load_fail", f"no extents for '{name}': {exc}")
        )


def _entity_pose(context: SkillContext, name: str, skill: str) -> np.ndarray:
    """``(3,)`` world position of ``name``; raises _ResolveError."""
    try:
        pos, _ = context.get_entity_pose(name)
        return np.asarray(pos, dtype=np.float64).reshape(3)
    except Exception as exc:
        raise _ResolveError(_fail(skill, "load_fail", f"no pose for '{name}': {exc}"))


class StackSkill(SkillPrimitive):
    """Place the attached object on top of a support object.

    Phases: resolve the target (support xy + support top + carried
    half-height) -> approach -> descend -> open/detach -> retreat.

    Phase/stage mapping (FAIL_STAGES): motion failures -> ``transport_fail``;
    release failures -> ``place_fail``; missing entity pose/extents ->
    ``load_fail``; extents unavailable (no hook and no explicit params) ->
    ``error``.

    Precondition: the object is already attached (run ``grasp`` first).
    """

    name = "stack"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        support_name: str,
        target_pos: Optional[np.ndarray] = None,
        support_top_z: Optional[float] = None,
        carried_half_height: Optional[float] = None,
        place_quat: Optional[np.ndarray] = None,
        standoff: float = 0.10,
        retreat_height: float = 0.10,
    ) -> SkillResult:
        try:
            target = self._resolve_target(
                context,
                object_name,
                support_name,
                target_pos,
                support_top_z,
                carried_half_height,
            )
        except _ResolveError as exc:
            return exc.result

        quat = (
            np.asarray(place_quat, dtype=np.float64) if place_quat is not None else None
        )
        approach = target + np.array([0.0, 0.0, standoff])
        retreat = target + np.array([0.0, 0.0, retreat_height])
        trajectories = []

        try:
            for goal in (approach, target):
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

    def _resolve_target(
        self,
        context: SkillContext,
        object_name: str,
        support_name: str,
        target_pos: Optional[np.ndarray],
        support_top_z: Optional[float],
        carried_half_height: Optional[float],
    ) -> np.ndarray:
        if target_pos is not None:
            return np.asarray(target_pos, dtype=np.float64).reshape(3)
        support_pos = _entity_pose(context, support_name, self.name)
        if support_top_z is not None:
            support_top = float(support_top_z)
        else:
            support_top = float(
                support_pos[2] + _half_extents(context, support_name, self.name)[2]
            )
        if carried_half_height is not None:
            carried = float(carried_half_height)
        else:
            carried = float(_half_extents(context, object_name, self.name)[2])
        return np.array(
            [support_pos[0], support_pos[1], support_top + carried],
            dtype=np.float64,
        )
