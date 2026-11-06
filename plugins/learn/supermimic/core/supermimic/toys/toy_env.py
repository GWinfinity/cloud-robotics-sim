"""Batched analytic environment for the toy sinusoid-tracking task.

Pure PyTorch (no simulator): the joint dynamics are a context-dependent
critically-damped PD chain — ``qdd = gain * (u - q) - damping * qd``,
semi-implicitly integrated at ``CONTROL_DT`` — which gives the world model a
learnable but nontrivial target. The root is fixed. Episodes run
``clip_length`` frames from the start of the periodic reference stream (the
frame index equals the episode-local step, keeping
``valid_imagination_start`` exact); on episode end the env resamples the
dynamics context.

Implements ``EnvProtocol`` + ``SupportsSuperMimicProtocol`` (see
``supermimic.contracts``); use with ``SuperMimicLearner`` directly.
"""

from __future__ import annotations

import torch

from ..contracts import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
    SuperMimicMdpProtocol,
    SupportsSuperMimicProtocol,
)
from ..geometry import mat_to_quat
from .toy_mdp import CONTROL_DT, NUM_JOINTS, ToyMdp, reference


class ToyEnvState:
    """Step return object: the learner only reads ``info``."""

    def __init__(self, info: dict):
        self.info = info


class ToyEnv:
    """Vectorized toy tracker environment."""

    def __init__(
        self,
        num_envs: int = 16,
        clip_length: int = 48,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.mdp = ToyMdp(clip_length=clip_length)
        self.clip_length = clip_length
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self.state: ToyEnvState | None = None

    # -- protocol surface -------------------------------------------------
    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        return self.mdp

    def init_state(self) -> ToyEnvState:
        frames, context, q, qd = self._sample_episode(
            torch.arange(self.num_envs, device=self.device)
        )
        info = self._make_info(
            frames,
            context,
            q,
            qd,
            a_prev=torch.zeros(self.num_envs, NUM_JOINTS, device=self.device),
            ep_step=torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
        )
        self.state = ToyEnvState({SUPERMIMIC_STATE_KEY: info})
        return self.state

    def step(self, actions: torch.Tensor) -> ToyEnvState:
        assert self.state is not None, "init_state() must be called before step()"
        info = self.state.info[SUPERMIMIC_STATE_KEY]
        q = info["q"]
        qd = info["qd"]
        context = info["xi"]
        frames = info["f"]
        ep_step = info["ep_step"].long()
        a_prev = info["a_prev"]

        # Real dynamics: context-dependent PD chain.
        damping = context[:, 0:1]
        gain = context[:, 1:2]
        qdd = gain * (actions - q) - damping * qd
        next_qd = qd + CONTROL_DT * qdd
        next_q = q + CONTROL_DT * next_qd
        next_frames = frames + 1
        next_step = ep_step + 1
        done = next_step >= self.clip_length

        reward = self.mdp.reward(
            self._state_tuple(next_q, next_qd),
            next_frames,
            actions,
            a_prev,
        )
        term = torch.zeros_like(done, dtype=torch.bool)

        transition = {
            "s": self._flatten(q, qd),
            "s1": self._flatten(next_q, next_qd),
            "a": actions.clone(),
            "a_prev": a_prev.clone(),
            "xi": context.clone(),
            "r": reward,
            "f": frames.clone(),
            "ep_step": ep_step.to(torch.int32),
            "done": done,
            "term": term,
        }
        if bool(done.any()):
            transition["ep_len"] = next_step[done].float()
            transition["ep_ret_norm"] = reward[done] / (
                next_step[done].float() * self.mdp.max_step_reward
            )

        # Auto-reset finished episodes to a fresh window.
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            frames_r, context_r, q_r, qd_r = self._sample_episode(reset_ids)
            next_q = next_q.clone()
            next_qd = next_qd.clone()
            next_frames = next_frames.clone()
            next_q[reset_ids] = q_r
            next_qd[reset_ids] = qd_r
            next_frames[reset_ids] = frames_r
            next_step = next_step.clone()
            next_step[reset_ids] = 0
            a_next = torch.where(done.unsqueeze(-1), torch.zeros_like(actions), actions)
        else:
            a_next = actions

        new_info = self._make_info(
            next_frames, context, next_q, next_qd, a_next, next_step.to(torch.int32)
        )
        if reset_ids.numel() > 0:
            new_info["xi"] = info["xi"].clone()
            new_info["xi"][reset_ids] = context_r
        self.state = ToyEnvState(
            {SUPERMIMIC_STATE_KEY: new_info, SUPERMIMIC_TRANSITION_KEY: transition}
        )
        return self.state

    # -- internals --------------------------------------------------------
    def _sample_episode(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fresh episode: episodes always start at frame 0 of the reference
        stream so that the frame index equals the episode-local step and the
        MDP's ``valid_imagination_start`` is exact; diversity comes from the
        per-episode dynamics context.
        """
        n = ids.shape[0]
        start = torch.zeros(n, dtype=torch.long, device=self.device)
        damping = 2.0 + 6.0 * torch.rand(
            n, 1, generator=self._generator, device=self.device
        )
        gain = 20.0 + 20.0 * torch.rand(
            n, 1, generator=self._generator, device=self.device
        )
        context = torch.cat((damping, gain), -1)
        q_ref, qd_ref = reference(start)
        return start, context, q_ref.clone(), qd_ref.clone()

    def _state_tuple(self, q: torch.Tensor, qd: torch.Tensor) -> State:
        n = q.shape[0]
        position = torch.zeros(n, 3, device=self.device)
        position[:, 2] = 1.0
        rotation = torch.eye(3, device=self.device).expand(n, 3, 3)
        return (
            position,
            rotation,
            torch.zeros(n, 3, device=self.device),
            torch.zeros(n, 3, device=self.device),
            q,
            qd,
        )

    def _flatten(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        position, rotation, lin_vel, ang_vel, _, _ = self._state_tuple(q, qd)
        quat = mat_to_quat(rotation)
        return torch.cat((position, quat, lin_vel, ang_vel, q, qd), -1)

    def _make_info(
        self,
        frames: torch.Tensor,
        context: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        a_prev: torch.Tensor,
        ep_step: torch.Tensor,
    ) -> dict:
        # ``s`` is the flattened 13+2J state the learner/replay buffer store.
        return {
            "s": self._flatten(q, qd),
            "q": q,
            "qd": qd,
            "f": frames,
            "xi": context,
            "a_prev": a_prev,
            "ep_step": ep_step,
        }

    def env_state_dict(self) -> dict:
        return {
            "generator_state": self._generator.get_state(),
            "clip_length": float(self.clip_length),
        }

    def load_env_state_dict(self, state: dict) -> None:
        self._generator.set_state(state["generator_state"])

    def close(self) -> None:
        self.state = None


# Static protocol assertion (checked at import time by type checkers).
_: SupportsSuperMimicProtocol = ToyEnv

__all__ = ["ToyEnv", "ToyEnvState"]
