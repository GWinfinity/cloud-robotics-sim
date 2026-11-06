"""Contracts between the SuperMimic learner and task-owned implementations.

The learner is task-agnostic: it consumes an environment satisfying
``EnvProtocol`` and an MDP satisfying ``SuperMimicMdpProtocol``. All robot
specifics (observation layout, reference motion, reward, mirroring) live in
the MDP, so the same learner drives any floating-base robot or toy task.

State layout convention (see ``geometry.flatten_state`` / ``unflatten_state``):

    (position(3), rotation(3x3), lin_vel(3), ang_vel(3), joint_pos(J), joint_vel(J))

The flattened representation is ``13 + 2*J`` scalars (quaternion wxyz for
rotation). This convention is fixed by the world model's integrator; tasks
with a fixed base simply keep the root components constant.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

State = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

SUPERMIMIC_TRANSITION_KEY = "supermimic"
SUPERMIMIC_STATE_KEY = "supermimic_state"


class EnvProtocol(Protocol):
    """Minimal batched environment surface consumed by the learner.

    ``step`` must return an object with an ``info`` dict carrying
    ``SUPERMIMIC_TRANSITION_KEY`` (the transition just taken) and update the
    environment so that a subsequent ``step`` continues the episode; the
    next ``info`` must carry ``SUPERMIMIC_STATE_KEY`` describing the new
    current state. Both dicts are specified by ``SuperMimicMdpProtocol``
    implementations (see the toys package for a complete example).
    """

    num_envs: int
    device: torch.device
    state: Any

    def init_state(self) -> Any: ...

    def step(self, actions: torch.Tensor) -> Any: ...


class MdpMirror(Protocol):
    """Left-right mirror used for world-model data augmentation.

    Optional: an MDP exposes one as its ``mirror`` attribute when
    ``wm_mirror > 0`` is requested in the learner config.
    """

    def state(self, state: State) -> State: ...

    def action(self, action: torch.Tensor) -> torch.Tensor: ...

    def context(self, context: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class SuperMimicMdpProtocol(Protocol):
    """Differentiable analytic heads and reference data owned by a task."""

    num_joints: int
    context_dim: int
    obs_dim: int
    goal_frame_dim: int
    max_step_reward: float

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor: ...

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor: ...

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor: ...

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor: ...

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor: ...

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor: ...

    def valid_imagination_start(
        self, frames: torch.Tensor, horizon: int
    ) -> torch.Tensor: ...

    def to(self, device: str | torch.device) -> SuperMimicMdpProtocol:
        """Return a copy on ``device`` for learner-side use off the env device."""
        ...


def get_supermimic_mdp(env: Any) -> SuperMimicMdpProtocol:
    """Return the injected task MDP, failing closed for unsupported envs."""
    if not isinstance(env, SupportsSuperMimicProtocol):
        raise TypeError(
            f"{type(env).__name__} does not provide the SuperMimic MDP contract"
        )
    return env.supermimic_mdp


# ---------------------------------------------------------------------------
# Optional eprobe hooks (failure/evaluation-guided resets, BlinkMimic-plus)
# ---------------------------------------------------------------------------
#
# Enabling ``eprobe_share > 0`` in the learner config requires the task to
# provide, beyond the two core protocols:
#
# - ``env.cfg``: a reconstructable (dataclass) environment config; the
#   learner builds a probe environment via ``type(env)(replace(env.cfg,
#   num_envs=<tiles>, eval_mode=True, reset_noise=False, contact_reward=0.0,
#   fail_reset=0.0, eprobe_share=0.0, max_episode_steps=10**9, seed=<derived>,
#   camera=False))``;
# - ``env.reset_to_frames(frames)``: reset probe envs onto given reference
#   frames;
# - ``env.set_focus(focus, share)``: bias a share of training resets toward
#   the given per-bin probability distribution;
# - ``mdp.motion``: reference metadata with ``names``, ``clip_start``,
#   ``clip_end``, ``clip_id``, ``frame_bin`` (frame -> bin index), and
#   ``num_bins``.


@runtime_checkable
class SupportsSuperMimicProtocol(Protocol):
    """Environment exposing its task-owned SuperMimic implementation."""

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol: ...


__all__ = [
    "SUPERMIMIC_STATE_KEY",
    "SUPERMIMIC_TRANSITION_KEY",
    "EnvProtocol",
    "MdpMirror",
    "State",
    "SuperMimicMdpProtocol",
    "SupportsSuperMimicProtocol",
    "get_supermimic_mdp",
]
