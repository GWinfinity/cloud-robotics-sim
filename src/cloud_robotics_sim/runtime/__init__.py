"""Agent runtime module.

Hosts the agent-facing simulation surface:

* :mod:`.skills` — skill registry, task executor, and replay buffer;
* :mod:`.agent_hub` — MCP-style tool surface exposing the simulation
  environment to agents;
* :mod:`.queue_worker` — Redis queue worker for the Kubernetes/KEDA
  deployment;
* :mod:`.main` — the continuous improvement loop for Genesis simulations.
"""

from __future__ import annotations

from .main import ImprovementLoop, LoopConfig, make_env_from_config
from .queue_worker import TaskSpec, parse_task, run_worker
from .skills import (
    ExecutionRecord,
    ReplayBuffer,
    Skill,
    SkillError,
    SkillRegistry,
    TaskExecutor,
    default_skill_registry,
    register_skill,
)

__all__ = [
    "ExecutionRecord",
    "ImprovementLoop",
    "LoopConfig",
    "ReplayBuffer",
    "Skill",
    "SkillError",
    "SkillRegistry",
    "TaskExecutor",
    "TaskSpec",
    "default_skill_registry",
    "make_env_from_config",
    "parse_task",
    "register_skill",
    "run_worker",
]
