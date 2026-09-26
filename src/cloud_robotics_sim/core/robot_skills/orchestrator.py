"""Skill orchestrator: run ordered skill primitives as one compound task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .base import STATUS_SUCCESS, SkillContext, SkillPrimitive, SkillResult
from .grasp import GraspSkill
from .place import PlaceSkill

logger = logging.getLogger(__name__)

_PRIMITIVES: Dict[str, SkillPrimitive] = {
    GraspSkill.name: GraspSkill(),
    PlaceSkill.name: PlaceSkill(),
}


@dataclass
class SkillStep:
    """One entry in a skill sequence."""

    skill: str
    params: Dict[str, Any] = field(default_factory=dict)


class SkillSequence:
    """Execute an ordered sequence of skill primitives against a context.

    Stops at the first failure; the failing skill's stage (a FAIL_STAGES
    label) becomes the sequence stage. On full completion the sequence stage
    is ``success``.
    """

    def __init__(self, steps: List[Tuple[str, Dict[str, Any]] | SkillStep]):
        self._steps: List[SkillStep] = [
            s if isinstance(s, SkillStep) else SkillStep(skill=s[0], params=s[1])
            for s in steps
        ]
        for step in self._steps:
            if step.skill not in _PRIMITIVES:
                raise ValueError(
                    f"unknown skill '{step.skill}'; available: {sorted(_PRIMITIVES)}"
                )

    @property
    def steps(self) -> List[SkillStep]:
        """The ordered skill steps."""
        return list(self._steps)

    def execute(self, context: SkillContext) -> SkillResult:
        """Run all steps in order; return the aggregated result."""
        trajectories: List[Any] = []
        for index, step in enumerate(self._steps):
            primitive = _PRIMITIVES[step.skill]
            logger.info("skill %d/%d: %s", index + 1, len(self._steps), step.skill)
            result = primitive.execute(context, **step.params)
            trajectories.extend(result.trajectories)
            if not result.success:
                logger.warning(
                    "skill sequence failed at step %d (%s): stage=%s message=%s",
                    index + 1,
                    step.skill,
                    result.stage,
                    result.message,
                )
                return SkillResult(
                    success=False,
                    stage=result.stage,
                    message=f"step {index} ({step.skill}): {result.message}",
                    skill=step.skill,
                    trajectories=trajectories,
                )
        return SkillResult(
            success=True,
            stage=STATUS_SUCCESS,
            skill="+".join(s.skill for s in self._steps),
            trajectories=trajectories,
        )
