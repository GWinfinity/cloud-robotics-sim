"""Agent runtime module.

Hosts the continuous improvement loop for Genesis simulations and the
Redis queue worker used by the Kubernetes/KEDA deployment
(see ``deploy/kubernetes/``). The canonical entry points are exported
below for convenience.
"""

from __future__ import annotations

from .main import ImprovementLoop, LoopConfig, make_env_from_config
from .queue_worker import TaskSpec, parse_task, run_worker

__all__ = [
    "ImprovementLoop",
    "LoopConfig",
    "TaskSpec",
    "make_env_from_config",
    "parse_task",
    "run_worker",
]
