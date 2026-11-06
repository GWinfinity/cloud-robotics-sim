"""SuperMimic core: learned-world-model imitation learning, task-agnostic.

Ported from supermimic-genesis (Uni-SuperMimic lineage, Apache-2.0) and
decoupled from the G1 humanoid task: the learner consumes only
``contracts.EnvProtocol`` and ``contracts.SuperMimicMdpProtocol``. The
``toys`` package ships a minimal reference task (analytic two-joint tracker)
that implements both protocols end to end and serves as a template for real
robot tasks.
"""

from __future__ import annotations

from .contracts import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    EnvProtocol,
    MdpMirror,
    State,
    SuperMimicMdpProtocol,
    SupportsSuperMimicProtocol,
    get_supermimic_mdp,
)
from .learner import SuperMimicConfig, SuperMimicLearner
from .networks import Critic, Policy, WorldModel
from .optim import MuonAdam
from .replay import ReplayBuffer

__all__ = [
    "SUPERMIMIC_STATE_KEY",
    "SUPERMIMIC_TRANSITION_KEY",
    "Critic",
    "EnvProtocol",
    "MdpMirror",
    "MuonAdam",
    "Policy",
    "ReplayBuffer",
    "State",
    "SuperMimicConfig",
    "SuperMimicLearner",
    "SuperMimicMdpProtocol",
    "SupportsSuperMimicProtocol",
    "WorldModel",
    "get_supermimic_mdp",
]
