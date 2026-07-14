"""Improvement proposal generation for the continuous loop."""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .diagnostics import Diagnosis

logger = logging.getLogger(__name__)


@dataclass
class ImprovementProposal:
    """A candidate improvement to the simulation configuration."""

    name: str
    description: str
    target_diagnosis: str
    config_delta: dict[str, Any] = field(default_factory=dict)
    expected_improvement: dict[str, float] = field(default_factory=dict)
    priority: int = 1  # lower = more urgent

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_diagnosis": self.target_diagnosis,
            "config_delta": self.config_delta,
            "expected_improvement": self.expected_improvement,
            "priority": self.priority,
        }


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary without mutating the original."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


class ProposalGenerator:
    """Generate candidate improvements based on diagnosed problems."""

    def __init__(self, random_seed: int | None = None) -> None:
        self.rng = random.Random(random_seed)

    def generate(
        self, diagnosis: Diagnosis, baseline_config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        """Generate proposals for a given diagnosis."""
        generators = {
            "success_rate_low": self._success_rate_proposals,
            "physics_unstable": self._physics_proposals,
            "sim_slow": self._speed_proposals,
            "reward_shaping_bad": self._reward_proposals,
            "healthy": self._generalization_proposals,
        }
        fn = generators.get(diagnosis.category, self._fallback_proposals)
        return fn(diagnosis, baseline_config)

    def _success_rate_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        proposals = []
        # Looser success threshold
        proposals.append(
            ImprovementProposal(
                name="looser_success_threshold",
                description="Increase success threshold to allow easier early wins",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"task": {"success_threshold": 0.08}}},
                expected_improvement={"success_rate": 0.15},
                priority=1,
            )
        )
        # Stronger success reward
        proposals.append(
            ImprovementProposal(
                name="stronger_success_reward",
                description="Increase success reward to make task completion more salient",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"task": {"success_reward": 2.0}}},
                expected_improvement={"success_rate": 0.1},
                priority=2,
            )
        )
        # More goal-oriented shaping
        proposals.append(
            ImprovementProposal(
                name="tighter_shaping",
                description="Reduce step penalty and increase distance shaping weight",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"task": {"step_penalty": -0.0005}}},
                expected_improvement={"success_rate": 0.05, "mean_reward": 5.0},
                priority=3,
            )
        )
        return proposals

    def _physics_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        proposals = []
        # More substeps
        proposals.append(
            ImprovementProposal(
                name="more_substeps",
                description="Increase physics substeps to improve contact stability",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"simulation": {"substeps": 20}}},
                expected_improvement={
                    "physics_violations": -2.0,
                    "stability_score": 0.1,
                },
                priority=1,
            )
        )
        # Smaller dt
        proposals.append(
            ImprovementProposal(
                name="smaller_dt",
                description="Decrease simulation timestep for higher fidelity",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"simulation": {"dt": 0.005}}},
                expected_improvement={
                    "physics_violations": -1.5,
                    "stability_score": 0.08,
                },
                priority=2,
            )
        )
        # Stiffer joints
        proposals.append(
            ImprovementProposal(
                name="tighter_joint_control",
                description="Increase joint stiffness and damping for stable tracking",
                target_diagnosis=diagnosis.category,
                config_delta={
                    "environment": {
                        "robot": {"joint_stiffness": 150.0, "joint_damping": 15.0}
                    }
                },
                expected_improvement={"stability_score": 0.12},
                priority=3,
            )
        )
        return proposals

    def _speed_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        proposals = []
        # Fewer substeps
        proposals.append(
            ImprovementProposal(
                name="fewer_substeps",
                description="Reduce substeps if physics remains stable enough",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"simulation": {"substeps": 5}}},
                expected_improvement={"latency_ms": -30.0},
                priority=1,
            )
        )
        # Lower camera resolution
        proposals.append(
            ImprovementProposal(
                name="lower_resolution",
                description="Lower camera resolution for faster rendering",
                target_diagnosis=diagnosis.category,
                config_delta={
                    "environment": {"simulation": {"resolution": [320, 240]}}
                },
                expected_improvement={"latency_ms": -20.0},
                priority=2,
            )
        )
        return proposals

    def _reward_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        proposals = []
        proposals.append(
            ImprovementProposal(
                name="reduce_step_penalty",
                description="Reduce step penalty to discourage lazy behavior",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"task": {"step_penalty": -0.0001}}},
                expected_improvement={"mean_reward": 10.0},
                priority=1,
            )
        )
        proposals.append(
            ImprovementProposal(
                name="larger_success_reward",
                description="Increase sparse success reward signal",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"task": {"success_reward": 2.5}}},
                expected_improvement={"mean_reward": 15.0},
                priority=2,
            )
        )
        return proposals

    def _generalization_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        proposals = []
        # Randomize target positions
        proposals.append(
            ImprovementProposal(
                name="randomize_targets",
                description="Widen target position distribution for better generalization",
                target_diagnosis=diagnosis.category,
                config_delta={
                    "environment": {"task": {"target_position": [0.5, 0.0, 0.05]}}
                },
                expected_improvement={"success_rate": 0.05},
                priority=1,
            )
        )
        return proposals

    def _fallback_proposals(
        self, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> list[ImprovementProposal]:
        return [
            ImprovementProposal(
                name="tune_default",
                description="Minor default parameter sweep",
                target_diagnosis=diagnosis.category,
                config_delta={"environment": {"simulation": {"dt": 0.008}}},
                expected_improvement={},
                priority=5,
            )
        ]

    def apply(
        self, baseline_config: dict[str, Any], proposal: ImprovementProposal
    ) -> dict[str, Any]:
        """Apply a proposal to a baseline config and return a new config."""
        return deep_update(baseline_config, proposal.config_delta)
