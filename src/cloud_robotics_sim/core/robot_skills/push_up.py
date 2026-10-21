"""Push skill primitive: approach -> descend -> open gripper -> push -> retreat.

Open-gripper linear push along a direction — righting a tipped object or
sliding one across the table. The EE never attaches; the contact point
sits ``contact_offset`` behind the object surface along ``-direction``.
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

#: Default push direction: +X.
DEFAULT_DIRECTION = np.array([1.0, 0.0, 0.0])


class PushUpSkill(SkillPrimitive):
    """Open-gripper linear push along a direction (righting/sliding).

    The EE never attaches: approach above the contact point -> descend ->
    ensure gripper open -> push along ``direction`` -> retreat. The contact
    point sits ``contact_offset`` behind the object surface along
    ``-direction``; the surface offset comes from the ``get_entity_extents``
    hook (AABB support function) or the explicit ``half_extent`` param.

    Phase/stage mapping (FAIL_STAGES): all motion failures (incl. the
    gripper-open command) -> ``transport_fail``; missing entity pose or
    extents lookup failure -> ``load_fail``; degenerate direction ->
    ``error``; extents unavailable (no hook, no explicit param) -> ``error``.
    """

    name = "push_up"

    def execute(
        self,
        context: SkillContext,
        object_name: str,
        direction: Optional[np.ndarray] = None,
        push_distance: float = 0.15,
        contact_offset: float = 0.01,
        standoff: float = 0.08,
        retreat_height: float = 0.10,
        half_extent: Optional[float] = None,
    ) -> SkillResult:
        dir_vec = (
            DEFAULT_DIRECTION.copy()
            if direction is None
            else np.asarray(direction, dtype=np.float64).reshape(3)
        )
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-9:
            return _fail(
                self.name, "error", f"degenerate push direction {dir_vec.tolist()}"
            )
        dir_vec = dir_vec / norm

        try:
            obj_pos, _ = context.get_entity_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        except Exception as exc:
            return _fail(self.name, "load_fail", f"no pose for '{object_name}': {exc}")

        if half_extent is not None:
            half_along = float(half_extent)
        elif context.get_entity_extents is not None:
            try:
                half_extents = np.asarray(
                    context.get_entity_extents(object_name), dtype=np.float64
                ).reshape(3)
            except Exception as exc:
                return _fail(
                    self.name, "load_fail", f"no extents for '{object_name}': {exc}"
                )
            # AABB support function along the push direction (reduces to the
            # matching radius for axis-aligned directions).
            half_along = float(np.dot(half_extents, np.abs(dir_vec)))
        else:
            return _fail(
                self.name,
                "error",
                "push_up requires the get_entity_extents hook or an explicit "
                "half_extent param",
            )

        contact = obj_pos - dir_vec * (half_along + contact_offset)
        approach = contact + np.array([0.0, 0.0, standoff])
        push_end = contact + dir_vec * push_distance
        retreat = push_end + np.array([0.0, 0.0, retreat_height])
        trajectories = []

        try:
            for stage, goal in (
                ("approach", approach),
                ("descend", contact),
                ("open", contact),
                ("push", push_end),
                ("retreat", retreat),
            ):
                if stage == "open":
                    context.set_gripper(GRIPPER_OPEN)
                    continue
                traj = context.plan(goal, None)
                context.execute_trajectory(traj)
                trajectories.append(traj)
        except Exception as exc:
            return _fail(self.name, "transport_fail", f"{stage}: {exc}")

        return SkillResult(
            success=True,
            stage=STATUS_SUCCESS,
            skill=self.name,
            trajectories=trajectories,
        )
