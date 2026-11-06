"""SuperMimic training over the injected environment and MDP protocols.

Task-agnostic port of the supermimic-genesis learner: everything robot
specific is provided by the MDP (see ``contracts.SuperMimicMdpProtocol``) —
reference motions, reward, mirroring, the goal-frame reference-joint offset
used by ``pi_ref_skip``, and (optionally) the motion metadata used by the
``eprobe`` failure-guided reset probe. The optional eprobe hooks a task must
provide are documented in ``contracts``.

Learning loop (why this beats on-policy RL on sample efficiency): a learned
world model is trained on real transitions (``wm_update``), then the policy
is improved by backpropagating discounted imagined returns through
``horizon`` world-model steps (``policy_update``), with a TD(critic) bootstrap
— Dreamer-style BPTT with UTD >> 1 instead of one gradient per env step.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from . import networks
from .contracts import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    EnvProtocol,
    State,
    get_supermimic_mdp,
)
from .geometry import flatten_state, unflatten_state
from .logging import SuperMimicLogger
from .networks import Critic, Policy, WorldModel, mlp
from .optim import MuonAdam
from .replay import ReplayBuffer


@dataclass
class SuperMimicConfig:
    """Hyperparameters for ``SuperMimicLearner`` (see field comments above)."""

    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    z_dim: int = 32
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_every: int = 50
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    # Control timestep used by the world model's drift integrator; must match
    # the task's simulation control dt.
    wm_dt: float = 0.02
    test_hidden_dims: tuple[int, ...] | None = None
    # Polyak decay for the policy sliding average used during data collection
    # (InstantTrack); 0 disables it and keeps plain SuperMimic behavior.
    ema_decay: float = 0.0
    # Clamp critic TD targets and the policy bootstrap to the reachable return
    # range +-max_step_reward/(1-gamma) (allmimic); guards against the
    # late-training critic overestimation that otherwise diverges long runs.
    value_clip: bool = False
    # Fraction of env-steps collected without exploration noise (allmimic), so
    # the world model also covers states reached by the deployed deterministic
    # policy instead of only the noisy training distribution.
    noise_free_frac: float = 0.0
    # BlinkMimic-plus / InstantTrack-style training options; all defaults keep
    # the original SuperMimic behavior and checkpoint format.
    position_obs: str = "none"
    contact_reward: float = 0.0
    pi_ref_skip: bool = False
    wm_structured: bool = False
    # World-model data augmentation fraction mirrored through the MDP's
    # ``mirror`` attribute; requires the MDP to provide one.
    wm_mirror: float = 0.0
    pi_opt: str = "adam"
    wm_opt: str = "adam"
    no_sync: bool = False
    fused_adam: bool = False
    static_compile: bool = False
    # Failure/evaluation-guided resets (BlinkMimic-plus eprobe): periodically
    # sweep the deployed policy over the reference stream and focus a share of
    # training resets on the windows before its failures. Requires the task to
    # expose the optional eprobe hooks (see contracts module docstring):
    # env.cfg (reconstructable dataclass), env.reset_to_frames(frames),
    # env.set_focus(focus, share), and mdp.motion with clip metadata.
    eprobe_share: float = 0.0
    eprobe_tile: int = 300
    eprobe_lead_min: int = 200
    eprobe_lead_max: int = 450
    eprobe_every: int = 250
    eprobe_start: int = 1250
    eprobe_win: int = 10
    eprobe_decay: float = 0.5


_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "wm_loss",
    "wm_err8",
    "pi_loss",
    "v_loss",
    "imag_rew",
    "pi_gradnorm",
    "imag_survival",
    "sim_rew",
    "ep_len",
    "ep_ret_norm",
    "term_frac",
    "it_per_s",
    "lr",
)


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SuperMimicLearner:
    """Dreamer-style learner: world model on real transitions, policy by BPTT
    through imagined rollouts, TD(critic) bootstrap. See module docstring.
    """

    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")
        if not 0.0 <= cfg.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {cfg.ema_decay}")
        if not 0.0 <= cfg.noise_free_frac <= 1.0:
            raise ValueError(
                f"noise_free_frac must be in [0, 1], got {cfg.noise_free_frac}"
            )
        if cfg.value_clip and cfg.no_critic:
            raise ValueError("value_clip requires the critic but no_critic is set")
        if cfg.value_clip and not 0.0 < cfg.gamma < 1.0:
            raise ValueError(f"value_clip requires gamma in (0, 1), got {cfg.gamma}")
        if not 0.0 <= cfg.wm_mirror <= 1.0:
            raise ValueError(f"wm_mirror must be in [0, 1], got {cfg.wm_mirror}")
        if cfg.pi_opt not in ("adam", "muon") or cfg.wm_opt not in ("adam", "muon"):
            raise ValueError("pi_opt and wm_opt must be 'adam' or 'muon'")
        if cfg.no_sync and not cfg.fused_adam:
            raise ValueError("no_sync currently requires fused_adam")
        if cfg.position_obs not in ("none", "root"):
            raise ValueError(
                f"position_obs must be 'none' or 'root', got {cfg.position_obs!r}"
            )
        if cfg.contact_reward < 0.0:
            raise ValueError(
                f"contact_reward must be nonnegative, got {cfg.contact_reward}"
            )
        if cfg.eprobe_share < 0.0 or cfg.eprobe_share > 1.0:
            raise ValueError(f"eprobe_share must be in [0, 1], got {cfg.eprobe_share}")
        if cfg.eprobe_share > 0.0:
            if cfg.eprobe_tile <= 0 or cfg.eprobe_every <= 0 or cfg.eprobe_win <= 0:
                raise ValueError(
                    "eprobe_tile, eprobe_every and eprobe_win must be positive"
                )
            if cfg.eprobe_lead_min < 0 or cfg.eprobe_lead_max < cfg.eprobe_lead_min:
                raise ValueError("invalid eprobe lead range")
            if not hasattr(env, "cfg"):
                raise TypeError("eprobe requires an environment with a reusable cfg")

        self.env = env
        self.cfg = cfg
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # HOST_BRIDGE backends keep env tensors on CPU while the learner may
        # run on an accelerator; when the two differ, the learner works on its
        # own MDP copy and bridges tensors at the env boundary. On CUDA/ROCm
        # the devices always match and nothing changes.
        self._bridged = self.device != self.env_device
        mdp = get_supermimic_mdp(env)
        if getattr(mdp, "position_obs", cfg.position_obs) != cfg.position_obs:
            raise ValueError("learner position_obs does not match the environment MDP")
        if (
            abs(
                float(getattr(mdp, "contact_reward", cfg.contact_reward))
                - cfg.contact_reward
            )
            > 1e-8
        ):
            raise ValueError(
                "learner contact_reward does not match the environment MDP"
            )
        if cfg.wm_mirror > 0.0 and getattr(mdp, "mirror", None) is None:
            raise ValueError(
                "wm_mirror requires the MDP to provide a 'mirror' attribute"
            )
        self.mdp = mdp.to(self.device) if self._bridged else mdp

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        networks.RunningNorm.no_sync = cfg.no_sync

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(
            self.device
        )
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(
                self.critic.norm.mean.numel(), cfg.test_hidden_dims, 1
            ).to(self.device)
        fused_adam = cfg.fused_adam and self.device.type == "cuda"
        self.opt_wm: MuonAdam | torch.optim.Adam
        if cfg.wm_opt == "muon":
            self.opt_wm = MuonAdam(self.wm, [self.wm.net], lr=cfg.lr, fused=fused_adam)
        else:
            self.opt_wm = torch.optim.Adam(
                self.wm.parameters(), lr=cfg.lr, fused=fused_adam
            )
        self.opt_pi: MuonAdam | torch.optim.Adam
        if cfg.pi_opt == "muon":
            self.opt_pi = MuonAdam(
                self.pi,
                [self.pi.enc, self.pi.dec],
                lr=cfg.lr,
                betas=(0.9, 0.95),
                fused=fused_adam,
            )
        else:
            self.opt_pi = torch.optim.Adam(
                self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95), fused=fused_adam
            )
        self.opt_v = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, fused=fused_adam
        )
        self.pi_ema: Policy | None = None
        if cfg.ema_decay > 0.0:
            self.pi_ema = Policy(
                self.mdp.obs_dim,
                self.mdp.goal_frame_dim * self.cfg.goal_frames,
                hist=self.cfg.hist,
                z_dim=self.cfg.z_dim,
                num_joints=self.mdp.num_joints,
                hidden=cfg.policy_hidden,
                ref_skip=self._policy_ref_skip(),
            ).to(self.device)
            if cfg.test_hidden_dims is not None:
                enc_output = self.pi_ema.enc[-1]
                assert isinstance(enc_output, nn.Linear)
                z_dim = enc_output.out_features
                self.pi_ema.enc = mlp(
                    self.mdp.goal_frame_dim * self.cfg.goal_frames,
                    cfg.test_hidden_dims,
                    z_dim,
                ).to(self.device)
                self.pi_ema.dec = mlp(
                    z_dim + self.mdp.obs_dim * self.cfg.hist,
                    cfg.test_hidden_dims,
                    self.mdp.num_joints,
                ).to(self.device)
            self.pi_ema.load_state_dict(self.pi.state_dict())

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.obs_hist = torch.zeros(
            env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device
        )
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)
        self.mirror = getattr(self.mdp, "mirror", None) if cfg.wm_mirror > 0.0 else None
        self._eprobe_env = None
        self._eprobe_counts: torch.Tensor | None = None
        self._eprobe_tiles: torch.Tensor | None = None
        self._eprobe_next: torch.Tensor | None = None

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        if self.cfg.static_compile:
            return torch.compile(function, dynamic=False)
        return torch.compile(function)

    def _policy_ref_skip(self) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        if not self.cfg.pi_ref_skip:
            return None
        missing = [
            name
            for name in ("goal_q_offset", "default_q", "action_scale")
            if not hasattr(self.mdp, name)
        ]
        if missing:
            raise TypeError(f"reference-action skip requires MDP fields {missing}")
        return (
            int(getattr(self.mdp, "goal_q_offset")),
            getattr(self.mdp, "default_q"),
            getattr(self.mdp, "action_scale"),
        )

    def _make_world_model(self) -> WorldModel:
        hidden = (
            self.cfg.test_hidden_dims
            if self.cfg.test_hidden_dims is not None
            else self.cfg.wm_hidden
        )
        return WorldModel(
            hidden=hidden,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            structured=self.cfg.wm_structured,
            dt=self.cfg.wm_dt,
        )

    def _make_policy(self) -> Policy:
        goal_dim = self.mdp.goal_frame_dim * self.cfg.goal_frames
        policy = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=self.cfg.hist,
            z_dim=self.cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=self.cfg.policy_hidden,
            ref_skip=self._policy_ref_skip(),
        )
        if self.cfg.test_hidden_dims is not None:
            enc_output = policy.enc[-1]
            assert isinstance(enc_output, nn.Linear)
            z_dim = enc_output.out_features
            policy.enc = mlp(goal_dim, self.cfg.test_hidden_dims, z_dim)
            policy.dec = mlp(
                z_dim + self.mdp.obs_dim * self.cfg.hist,
                self.cfg.test_hidden_dims,
                self.mdp.num_joints,
            )
            dec_output = policy.dec[-1]
            assert isinstance(dec_output, nn.Linear)
            with torch.no_grad():
                dec_output.weight.mul_(0.01)
                dec_output.bias.zero_()
        return policy

    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    @property
    def _behavior_policy(self) -> Policy:
        return self.pi_ema if self.pi_ema is not None else self.pi

    @torch.no_grad()
    def _update_pi_ema(self) -> None:
        if self.pi_ema is None:
            return
        decay = self.cfg.ema_decay
        parameters = dict(self.pi.named_parameters())
        ema_state = self.pi_ema.state_dict()
        with torch.no_grad():
            for name, tensor in self.pi.state_dict().items():
                if name in parameters:
                    ema_state[name].mul_(decay).add_(tensor, alpha=1.0 - decay)
                else:
                    # RunningNorm statistics track the data stream, not the
                    # optimization; keep them identical to the live policy.
                    ema_state[name].copy_(tensor)

    def _critic_input_dim(self) -> int:
        info = self._state_info()
        with torch.no_grad():
            features = self.mdp.critic_features(
                unflatten_state(info["s"]),
                info["f"],
                self.mdp.normalize_context(info["xi"]),
            )
        return int(features.shape[-1] + self.mdp.goal_frame_dim * self.cfg.goal_frames)

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        error = (
            self.wm.state_error(predicted, unflatten_state(next_states)) / std
        ) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = self.wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_frames, self.cfg.goal_frames)
        return next_state, action, next_hist, next_frames, next_goal

    def _reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mdp.reward(
            state, frames, action, previous_action
        ), self.mdp.soft_termination(state, frames)

    def _critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (self.mdp.critic_features(state, frames, normalized_context), goal), -1
        )

    def _clamp_value(self, value: torch.Tensor) -> torch.Tensor:
        """Bound critic targets and the policy bootstrap to the reachable return range.

        allmimic: without this clamp a few overestimated values propagate through
        the TD recursion, the policy follows the inflated bootstrap, and long
        runs diverge after several hours.
        """
        if not self.cfg.value_clip:
            return value
        limit = self.mdp.max_step_reward / (1.0 - self.cfg.gamma)
        return value.clamp(-limit, limit)

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise = self.cfg.action_noise * torch.randn_like(action)
        if self.cfg.noise_free_frac > 0.0:
            noisy_envs = (
                torch.rand(action.shape[0], device=action.device)
                >= self.cfg.noise_free_frac
            )
            noise = noise * noisy_envs.unsqueeze(-1)
        return torch.clamp(action + noise, -20, 20)

    def _critic_targets(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """GAE-style backwards recursion over imagined transitions, optional clipping.

        With ``value_clip`` the bootstrap and every intermediate target are clamped
        to the reachable return range so a few overestimated critic values cannot
        amplify through the recursion (allmimic).
        """
        values = self._clamp_value(values)
        target = values[-1]
        targets: list[torch.Tensor] = []
        for index in range(self.cfg.horizon - 1, -1, -1):
            target = rewards[index] + self.cfg.gamma * (1 - dones[index]) * (
                (1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target
            )
            targets.append(self._clamp_value(target))
        return torch.stack(targets[::-1])

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            state = unflatten_state(info["s"])
            obs = self.mdp.add_noise(
                self.mdp.obs(state, info["a_prev"], info["xi"], info["f"])
            )
            fresh = info["ep_step"] == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
            )
            goal = self.mdp.goal(info["f"], self.cfg.goal_frames)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = self._add_exploration_noise(
                self._behavior_policy(self.obs_hist, goal)
            )
            if self._bridged:
                step_state = self.env.step(action.to(self.env_device))
            else:
                step_state = self.env.step(action)
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            if self._bridged:
                transition = {
                    key: value.to(self.device) for key, value in transition.items()
                }
            self.buffer.add(transition)
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            self.wm.in_norm.update(
                self.wm.inputs(
                    initial,
                    transition["a"],
                    self.mdp.normalize_context(transition["xi"]),
                )
            )
            self.wm.out_norm.update(self.wm.targets(initial, next_state))
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in transition:
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())
                self.stats["term"].append(transition["term"].float().sum())
                self.stats["done"].append(transition["done"].float().sum())

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        if self.cfg.wm_mirror > 0.0:
            assert self.mirror is not None
            count = min(
                window["s"].shape[0], int(window["s"].shape[0] * self.cfg.wm_mirror)
            )
            if count > 0:
                mirrored_initial = self.mirror.state(
                    unflatten_state(window["s"][:count])
                )
                mirrored_next = self.mirror.state(unflatten_state(window["s1"][:count]))
                window["s"][:count] = flatten_state(mirrored_initial)
                window["s1"][:count] = flatten_state(mirrored_next)
                window["a"][:count] = self.mirror.action(window["a"][:count])
                window["xi"][:count] = self.mirror.context(window["xi"][:count])
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _sample_valid_starts(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts = []
        histories = []
        count = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(
            parts: list[dict[str, torch.Tensor]],
        ) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        initial_goal = self.mdp.goal(frames, self.cfg.goal_frames)
        goal = initial_goal
        initial_state = state
        states, actions, previous_actions, all_frames, goals = [], [], [], [], []
        try:
            for _ in range(self.cfg.horizon):
                previous_actions.append(previous_action)
                state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                states.append(state)
                actions.append(previous_action)
                all_frames.append(frames)
                goals.append(goal)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = cast(
                State,
                tuple(flatten([item[index] for item in states]) for index in range(6)),
            )
            flat_frames = flatten(all_frames)
            rewards, dones = self._reward_term_fn(
                all_states, flat_frames, flatten(actions), flatten(previous_actions)
            )
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat(
                (torch.ones(1, batch_size, device=self.device), alive[:-1]), 0
            )
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            if not self.cfg.no_critic:
                bootstrap = self._clamp_value(
                    self.critic(
                        self._critic_input_fn(state, frames, normalized_context, goal)
                    )
                )
                objective = (
                    objective
                    + (self.cfg.gamma**self.cfg.horizon) * survival * bootstrap
                )
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if torch.isfinite(grad_norm):
                self.opt_pi.step()
                self._update_pi_ema()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            with torch.no_grad():
                critic_states = cast(
                    State,
                    tuple(
                        torch.cat(
                            (
                                initial_state[index].unsqueeze(0),
                                torch.stack([item[index].detach() for item in states]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for index in range(6)
                    ),
                )
                critic_frames = torch.cat(
                    (start["f"].unsqueeze(0), torch.stack(all_frames)), 0
                ).flatten()
                critic_goals = torch.cat(
                    (initial_goal.unsqueeze(0), torch.stack(goals)), 0
                ).flatten(0, 1)
                critic_context = (
                    normalized_context.unsqueeze(0)
                    .expand(self.cfg.horizon + 1, -1, -1)
                    .flatten(0, 1)
                )
                critic_input = self._critic_input_fn(
                    critic_states, critic_frames, critic_context, critic_goals
                ).view(self.cfg.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_input)
            with torch.no_grad():
                values = self.critic(critic_input)
                target_values = self._critic_targets(
                    detached_rewards, detached_dones, values
                )
            flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
            flat_targets = target_values.reshape(-1)
            num_samples = flat_input.shape[0]
            minibatch_size = num_samples // self.cfg.critic_minibatches
            if minibatch_size == 0:
                raise ValueError(
                    "critic_minibatches exceeds the number of imagined transitions"
                )
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(num_samples, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    indices = permutation[
                        index * minibatch_size : (index + 1) * minibatch_size
                    ]
                    value_loss = (
                        (self.critic(flat_input[indices]) - flat_targets[indices]) ** 2
                    ).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()

        imagined_reward = (
            reward_sum / self.cfg.horizon / self.mdp.max_step_reward
        ).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def _ensure_eprobe(self) -> None:
        """Build the probe environment once, lazily, on first sweep."""
        if self._eprobe_env is not None:
            return
        if self._bridged:
            raise RuntimeError(
                "eprobe requires the learner and probe environment on one device"
            )
        env_any = cast(Any, self.env)
        motion = cast(Any, getattr(self.mdp, "motion", None))
        if motion is None:
            raise TypeError(
                "eprobe requires the MDP to expose its reference 'motion' metadata"
            )
        env_cfg = env_any.cfg
        tiles: list[int] = []
        for clip in range(len(motion.names)):
            start = int(motion.clip_start[clip])
            end = int(motion.clip_end[clip])
            frame = start
            while frame < end:
                frame = min(frame + self.cfg.eprobe_tile, end)
                tiles.append(frame)
        if not tiles:
            raise ValueError("eprobe found no motion tiles")
        probe_cfg = replace(
            env_cfg,
            num_envs=len(tiles),
            eval_mode=True,
            reset_noise=False,
            contact_reward=0.0,
            fail_reset=0.0,
            eprobe_share=0.0,
            max_episode_steps=10**9,
            seed=self.cfg.seed + 7919,
            camera=False,
        )
        make_env = cast(Any, type(self.env))
        self._eprobe_env = make_env(probe_cfg)
        self._eprobe_tiles = torch.tensor(tiles, dtype=torch.long, device=self.device)
        self._eprobe_counts = torch.zeros(motion.num_bins, device=self.device)
        following = torch.arange(motion.num_bins, device=self.device) + 1
        same_clip = (
            motion.bin_clip[following.clamp(max=motion.num_bins - 1)] == motion.bin_clip
        )
        self._eprobe_next = torch.where(
            (following < motion.num_bins) & same_clip, following, following - 1
        )

    @torch.no_grad()
    def eprobe_sweep(self) -> int:
        """Sweep deployed-policy failures and focus a share of training resets before them."""
        self._ensure_eprobe()
        assert self._eprobe_env is not None
        assert self._eprobe_tiles is not None
        assert self._eprobe_counts is not None
        assert self._eprobe_next is not None
        probe_env = self._eprobe_env
        motion = cast(Any, self.mdp.motion)
        lead = int(
            torch.randint(
                self.cfg.eprobe_lead_min,
                self.cfg.eprobe_lead_max + 1,
                (1,),
                device=self.device,
            ).item()
        )
        clip_ids = motion.clip_id[self._eprobe_tiles]
        start_frames = torch.maximum(
            self._eprobe_tiles - lead, motion.clip_start[clip_ids]
        )
        probe_env.reset_to_frames(start_frames)
        policy = self._behavior_policy
        history = None
        failed = torch.full(
            (self._eprobe_tiles.numel(),), -1, dtype=torch.long, device=self.device
        )
        finished = torch.zeros_like(failed, dtype=torch.bool)
        for _ in range(lead + self.cfg.eprobe_tile):
            current = probe_env.state.info[SUPERMIMIC_STATE_KEY]
            if bool(finished.all()):
                break
            state = unflatten_state(current["s"])
            obs = self.mdp.obs(state, current["a_prev"], current["xi"], current["f"])
            if history is None:
                history = obs.unsqueeze(1).repeat(1, self.cfg.hist, 1)
            else:
                history = torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
            goal = self.mdp.goal(current["f"], self.cfg.goal_frames)
            action = policy(history, goal)
            alive = ~finished
            step_state = probe_env.step(action)
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            nonfinite = ~torch.isfinite(action).all(-1)
            failure = alive & (transition["term"] | nonfinite)
            failed = torch.where(failure, current["f"], failed)
            finished |= alive & transition["done"]

        self._eprobe_counts.mul_(self.cfg.eprobe_decay)
        failure_frames = failed[failed >= 0]
        if failure_frames.numel() > 0:
            failure_bins = motion.frame_bin[failure_frames]
            self._eprobe_counts.index_add_(
                0, failure_bins, torch.ones_like(failure_bins, dtype=torch.float32)
            )
        focus = self._eprobe_counts.clone()
        shifted = self._eprobe_counts
        keep = self._eprobe_next != torch.arange(motion.num_bins, device=self.device)
        for _ in range(self.cfg.eprobe_win):
            shifted = shifted[self._eprobe_next] * keep.float()
            focus = focus + shifted
        if float(focus.sum()) > 0.0:
            set_focus = getattr(self.env, "set_focus", None)
            if set_focus is None:
                raise TypeError(
                    "eprobe requires the training environment to support set_focus"
                )
            set_focus(focus / focus.sum(), self.cfg.eprobe_share)
        return int(failure_frames.numel())

    def close(self) -> None:
        """Release the probe environment if eprobe was used."""
        if self._eprobe_env is not None:
            close = getattr(self._eprobe_env, "close", None)
            if callable(close):
                close()
            self._eprobe_env = None

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (
            0.03
            + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        checkpoint = {
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.pi_ema is not None:
            checkpoint["pi_ema"] = self.pi_ema.state_dict()
        env_state = getattr(self.env, "env_state_dict", None)
        if callable(env_state):
            checkpoint["env_state"] = env_state()
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def save(
        self, path: str | Path | None = None, include_optimizers: bool = True
    ) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        required = {"critic", "opt_wm", "opt_pi", "opt_v"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"SuperMimic resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_pi.load_state_dict(checkpoint["opt_pi"])
        self.opt_v.load_state_dict(checkpoint["opt_v"])
        if self.pi_ema is not None:
            if "pi_ema" not in checkpoint:
                raise ValueError(
                    f"resume with ema_decay={self.cfg.ema_decay} requires 'pi_ema' in {path}"
                )
            self.pi_ema.load_state_dict(checkpoint["pi_ema"])
        load_env_state = getattr(self.env, "load_env_state_dict", None)
        if "env_state" in checkpoint and callable(load_env_state):
            load_env_state(checkpoint["env_state"])
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        logger = SuperMimicLogger(
            env_name=type(self.env).__name__,
            backend=self.backend,
            max_iterations=self.cfg.iterations,
            wm_only_iters=self.cfg.wm_only_iters,
            start_iteration=start_iteration,
            num_envs=self.env.num_envs,
            log_dir=str(self.log_dir),
            log_backend=self.cfg.log_backend,
        )
        logger.start(status="Warming up...")
        try:
            if start_iteration == 0:
                self.collect(self.cfg.warmup_steps)
            required_steps = max(self.cfg.hist, self.cfg.wm_window)
            if start_iteration > 0:
                required_steps = max(required_steps, self.cfg.warmup_steps)
            if self.buffer.size < required_steps:
                self.collect(required_steps - self.buffer.size)
            logger.log_buffer_fill(self.buffer.size, self.cfg.buffer_steps)
            logger.start_training_timer()

            log_path = self.log_dir / "train_log.csv"
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                if start_iteration == 0:
                    log_file.write(",".join(_LOG_FIELDS) + "\n")
                train_start = time.time() - self.wall_time
                log_start = time.time()
                step_accumulator = 0.0
                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                for iteration in range(start_iteration, self.cfg.iterations):
                    if self.cfg.compile and iteration == start_iteration:
                        logger.log_status(
                            "Compiling world model (one-off, slow on first run)"
                        )
                    if self.cfg.compile and iteration == self.cfg.wm_only_iters:
                        logger.log_status(
                            "Compiling policy update (one-off, slow on first run)"
                        )
                    iteration_start = time.time()
                    learning_rate = self.lr_at(iteration)
                    for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                    step_accumulator += 1.0 / self.cfg.utd
                    collect_steps = int(step_accumulator)
                    step_accumulator -= collect_steps
                    stats_mark = len(self.stats["r"])
                    self.collect(collect_steps)
                    train_phase_start = time.time()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    if iteration >= self.cfg.wm_only_iters:
                        (
                            policy_loss,
                            value_loss,
                            imagined_reward,
                            grad_norm,
                            survival,
                        ) = self.policy_update()
                        aggregate["pi"].append(policy_loss)
                        aggregate["v"].append(value_loss)
                        aggregate["ir"].append(imagined_reward)
                        aggregate["gn"].append(grad_norm)
                        aggregate["surv"].append(survival)
                        pi_loss = float(policy_loss)
                        v_loss = float(value_loss)
                        imag_rew = float(imagined_reward)
                        pi_gradnorm = float(grad_norm)
                        imag_survival = float(survival)
                    train_time = time.time() - train_phase_start
                    self.iteration = iteration + 1
                    if (
                        self.cfg.eprobe_share > 0.0
                        and self.iteration >= self.cfg.eprobe_start
                        and (self.iteration - self.cfg.eprobe_start)
                        % self.cfg.eprobe_every
                        == 0
                    ):
                        self.eprobe_sweep()
                    self.wall_time = time.time() - train_start
                    iteration_time = time.time() - iteration_start

                    iter_rewards = self.stats["r"][stats_mark:]
                    mean_reward = self._mean(iter_rewards) if iter_rewards else None
                    ep_len = (
                        self._mean(self.stats["ep_len"])
                        if self.stats["ep_len"]
                        else None
                    )
                    ep_ret_norm = (
                        self._mean(self.stats["ep_ret"])
                        if self.stats["ep_ret"]
                        else None
                    )
                    total_done = sum(float(value) for value in self.stats["done"])
                    term_frac = (
                        sum(float(value) for value in self.stats["term"]) / total_done
                        if total_done > 0
                        else None
                    )
                    logger.log_step(
                        self.iteration,
                        wm_loss=float(wm_loss),
                        wm_err8=float(wm_error),
                        pi_loss=pi_loss,
                        v_loss=v_loss,
                        imag_rew=imag_rew,
                        imag_survival=imag_survival,
                        pi_gradnorm=pi_gradnorm,
                        lr=learning_rate,
                        sim_rew=mean_reward,
                        ep_len=ep_len,
                        ep_ret_norm=ep_ret_norm,
                        term_frac=term_frac,
                        iter_sim_steps=collect_steps * self.env.num_envs,
                        collect_time=train_phase_start - iteration_start,
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        iterations_per_second = self.cfg.log_every / (
                            time.time() - log_start
                        )
                        log_start = time.time()
                        term_frac = float(
                            sum(float(value) for value in self.stats["term"])
                            / max(sum(float(value) for value in self.stats["done"]), 1)
                        )
                        row = [
                            self.iteration,
                            self.wall_time / 3600,
                            self.sim_steps,
                            self._mean(aggregate["wm"]),
                            self._mean(aggregate["wm8"]),
                            self._mean(aggregate["pi"]),
                            self._mean(aggregate["v"]),
                            self._mean(aggregate["ir"]),
                            self._mean(aggregate["gn"]),
                            self._mean(aggregate["surv"]),
                            self._mean(self.stats["r"]),
                            self._mean(self.stats["ep_len"]),
                            self._mean(self.stats["ep_ret"]),
                            term_frac,
                            iterations_per_second,
                            learning_rate,
                        ]
                        log_file.write(
                            ",".join(
                                (
                                    f"{value:.6g}"
                                    if isinstance(value, float)
                                    else str(value)
                                )
                                for value in row
                            )
                            + "\n"
                        )
                        log_file.flush()
                        last_metrics = dict(zip(_LOG_FIELDS, row, strict=True))
                        self.stats = _empty_stats()
                        aggregate = {key: [] for key in aggregate}

                    if (
                        self.iteration % self.cfg.ckpt_every == 0
                        or self.iteration == self.cfg.iterations
                    ):
                        self.wall_time = time.time() - train_start
                        self.save()
                        numbered = self.save(
                            self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                            include_optimizers=False,
                        )
                        logger.log_save(str(numbered))

            self.wall_time = time.time() - train_start
            logger.finish()
            return {
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
                **last_metrics,
            }
        finally:
            self.close()
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner"]
