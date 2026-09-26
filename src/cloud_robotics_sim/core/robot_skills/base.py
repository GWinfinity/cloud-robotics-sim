"""Robot skill primitives: grasp/place with planner routing and stage records.

Implements the W4 skill layer of ``docs/ROBODOJO_P0_PLAN.md``: two skill
primitives (``GraspSkill`` / ``PlaceSkill``) plus a minimal orchestrator that
runs an ordered skill sequence and reports failures using the seven-stage
``FAIL_STAGES`` vocabulary from ``robotwin.grasp_report`` so skill outcomes
stay comparable with grasp pipeline reports.

Design notes:

- Skills are *trajectory generators + phase scripts*: all simulator contact
  (closing, attaching, physics) goes through injected ``SkillContext`` hooks,
  keeping the primitives unit-testable without Genesis and reusable across
  embodiments.
- Motion planning goes through ``robotwin.curobo_planner.plan_with_fallback``
  (cuRobo-first, OMPL fallback) when a hierarchical planner is available.
- Grasp candidates come from RoboTwin ``model_dataN.json``
  (``contact_points_pose``) via :mod:`grasp_points` — a deliberate bypass of
  the suspended W3 annotation layer (documented plan deviation).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cloud_robotics_sim.robotwin.curobo_planner import (
    HierarchicalCuRoboPlanner,
    plan_with_fallback,
)

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"

#: Gripper command values (normalized): fully open / fully closed.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

#: Default top-down EE orientation (w, x, y, z): flange pointing down.
DOWN_QUAT = np.array([-0.0004, 0.9239, 0.3826, -0.0009])


class SkillExecutionError(RuntimeError):
    """A simulator-side skill hook (execute/gripper/attach/detach) failed."""


@dataclass
class SkillContext:
    """Everything a skill primitive needs from the runtime environment.

    All hooks are injected, so tests can substitute recordings/mocks and
    non-Genesis backends can provide their own semantics.

    Attributes:
        robot: ArticulationBackend-like object exposing ``get_qpos()``,
            ``inverse_kinematics(link, pos, quat)`` and
            ``plan_path(q_goal, num_waypoints)`` for the OMPL path.
        planner: Optional ``HierarchicalCuRoboPlanner``; when set,
            ``plan_with_fallback`` routes cuRobo-first.
        ee_link: End-effector link name for OMPL IK.
        get_entity_pose: ``(name) -> (pos (3,), quat (4, wxyz))`` of a scene
            object.
        execute_trajectory: ``(traj (T, dof)) -> None``; raise
            ``SkillExecutionError`` on failure.
        set_gripper: ``(value) -> None`` with 1.0 = open, 0.0 = closed.
        attach: ``(object_name) -> None`` bind object to the EE.
        detach: ``(object_name) -> None`` release it.
    """

    robot: Any
    planner: Optional[HierarchicalCuRoboPlanner]
    ee_link: str
    get_entity_pose: Callable[[str], Tuple[np.ndarray, np.ndarray]]
    execute_trajectory: Callable[[np.ndarray], None]
    set_gripper: Callable[[float], None]
    attach: Callable[[str], None]
    detach: Callable[[str], None]

    def plan(self, goal_pos: np.ndarray, goal_quat: Optional[np.ndarray]) -> np.ndarray:
        """Plan to an EE pose via plan_with_fallback (cuRobo first, OMPL)."""
        trajectory, _planner_name = plan_with_fallback(
            self.robot,
            goal_pos,
            goal_quat,
            ee_link=self.ee_link,
            planner=self.planner,
        )
        return np.asarray(trajectory, dtype=np.float64)


@dataclass
class SkillResult:
    """Outcome of one skill (or a full skill sequence)."""

    success: bool
    stage: str = "error"  # STATUS_SUCCESS or a FAIL_STAGES label
    message: str = ""
    skill: str = ""
    trajectories: List[np.ndarray] = field(default_factory=list)

    def to_record_dict(self) -> Dict[str, Any]:
        """Serialize for evaluation records."""
        return {
            "skill": self.skill,
            "status": self.stage,
            "message": self.message,
        }


class SkillPrimitive(ABC):
    """One named skill primitive executable against a SkillContext."""

    name: str = "abstract"

    @abstractmethod
    def execute(self, context: SkillContext, *args: Any, **params: Any) -> SkillResult:
        """Run the skill; never raises for expected failures (stages instead)."""


def _fail(skill: str, stage: str, message: str) -> SkillResult:
    return SkillResult(success=False, stage=stage, message=message, skill=skill)
