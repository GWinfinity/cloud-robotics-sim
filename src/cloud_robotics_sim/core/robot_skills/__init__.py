"""Robot skill primitives (W4): grasp/place + orchestrator.

Note: this package is the *robot manipulation* skill layer (W4 of
docs/ROBODOJO_P0_PLAN.md). It deliberately lives at ``core/robot_skills`` —
NOT ``core/skills`` — to avoid confusion with
``cloud_robotics_sim.runtime.skills``, the agent-facing skill registry
(patent demos / task scheduling).
"""

from cloud_robotics_sim.core.robot_skills.base import (
    DOWN_QUAT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    SkillContext,
    SkillExecutionError,
    SkillPrimitive,
    SkillResult,
)
from cloud_robotics_sim.core.robot_skills.grasp import GraspSkill
from cloud_robotics_sim.core.robot_skills.grasp_points import (
    GraspCandidate,
    candidate_in_world,
    first_model_data,
    load_grasp_candidates,
    score_candidates,
)
from cloud_robotics_sim.core.robot_skills.orchestrator import SkillSequence, SkillStep
from cloud_robotics_sim.core.robot_skills.place import PlaceSkill

__all__ = [
    "candidate_in_world",
    "DOWN_QUAT",
    "first_model_data",
    "GraspCandidate",
    "GraspSkill",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "load_grasp_candidates",
    "PlaceSkill",
    "score_candidates",
    "SkillContext",
    "SkillExecutionError",
    "SkillPrimitive",
    "SkillResult",
    "SkillSequence",
    "SkillStep",
]
