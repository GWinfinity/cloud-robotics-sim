"""rsl-rl 5.x adapter for the batched Genesis ``VecTask`` environments.

:class:`RslRlVecEnvWrapper` adapts the project's gymnasium-style 5-tuple
interface (:class:`GenesisVectorizedEnv` — ``step`` returns ``(obs, reward,
terminated, truncated, infos)``, resets are explicit) to the rsl-rl 5.x
:class:`rsl_rl.env.VecEnv` contract:

- ``get_observations()`` returns a ``tensordict.TensorDict`` with one
  observation group (``"policy"``); the train config's ``obs_groups`` maps
  actor/critic onto it.
- ``step(actions)`` returns ``(obs_td, rewards, dones, extras)`` where
  ``extras["time_outs"]`` carries the truncation mask (float) used by PPO for
  value bootstrapping on time limits, keeping it separate from true
  terminations in ``dones``.
- Auto-reset is internal: envs with ``dones`` are reset inside ``step`` and
  the returned observation is the post-reset one (rsl-rl's loop never calls
  ``reset`` between rollout steps).
- ``episode_length_buf`` is owned by the wrapper (rsl-rl rewrites it when
  ``init_at_random_ep_len=True``) and is the authoritative timeout counter —
  an env times out when ``episode_length_buf >= max_episode_length`` even if
  the inner task never reports ``truncated``.

rsl-rl is an optional dependency (the ``rl`` extra, ``rsl-rl-lib>=2.0``,
locked to 5.5.x); this module imports without it and only raising a friendly
error when the adapter is actually instantiated. The contract tests in
``tests/core/test_vec_rsl_rl.py`` exercise the wrapper against a fake
5-tuple env — no Genesis required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

try:  # torch is a hard dependency of genesis-world; guard for import hygiene
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # optional rl extra
    from rsl_rl.env import VecEnv as _RslRlVecEnv
    from rsl_rl.models import MLPModel as _RslRlMLPModel
    from rsl_rl.runners import OnPolicyRunner as _RslRlOnPolicyRunner
    from tensordict import TensorDict
except ImportError:  # rsl-rl-lib / tensordict not installed
    _RslRlVecEnv = object
    _RslRlMLPModel = None
    _RslRlOnPolicyRunner = object
    TensorDict = None

_RL_EXTRA_HINT = (
    "rsl-rl support is an optional extra: install it with "
    "`uv sync --extra rl` or `pip install cloud-robotics-sim[rl]`."
)

#: Observation group published by :class:`RslRlVecEnvWrapper` and consumed by
#: both the actor and the critic (see ``obs_groups`` in :func:`build_train_cfg`).
OBS_GROUP = "policy"


def _require_rl() -> None:
    """Raise a friendly error when the optional ``rl`` extra is missing."""
    if TensorDict is None:
        raise RuntimeError(_RL_EXTRA_HINT)


class RslRlVecEnvWrapper(_RslRlVecEnv):
    """Adapt a gymnasium-style batched env to the rsl-rl 5.x ``VecEnv`` contract.

    Args:
        env: Inner environment exposing ``num_envs``, ``num_actions``,
            ``num_observations``, ``device``, ``reset()``,
            ``reset_idx(envs_idx)`` and the 5-tuple ``step`` — exactly the
            :class:`GenesisVectorizedEnv` surface.
        max_episode_length: Timeout horizon. ``None`` reads
            ``env.task.max_episode_steps`` (the FrankaPickCubeVecTask
            default). The wrapper's ``episode_length_buf`` is authoritative
            for timeouts, so rsl-rl's ``init_at_random_ep_len`` works.
        cfg: Config dict exposed as ``self.cfg`` (rsl-rl's logger consumes
            it). Defaults to a small self-describing dict.

    Attributes:
        num_envs: Number of parallel environments.
        num_actions: Action dimension.
        max_episode_length: Scalar timeout horizon (int).
        episode_length_buf: ``(num_envs,)`` long tensor of per-env step
            counts; rewritten by rsl-rl for randomized episode lengths.
        device: Compute device of the inner env.
    """

    def __init__(
        self,
        env: Any,
        max_episode_length: int | None = None,
        cfg: dict | None = None,
    ) -> None:
        _require_rl()
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for the rsl-rl adapter")
        self.env = env
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.num_observations = int(env.num_observations)
        if max_episode_length is None:
            task = getattr(env, "task", None)
            max_episode_length = getattr(task, "max_episode_steps", None)
        if max_episode_length is None:
            raise ValueError(
                "max_episode_length is required: the inner env has no "
                "task.max_episode_steps to infer it from"
            )
        self.max_episode_length = int(max_episode_length)
        self.device = env.device
        self.cfg = cfg if cfg is not None else self._default_cfg()
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._obs: "torch.Tensor | None" = None

    def _default_cfg(self) -> dict:
        """Minimal config dict consumed by rsl-rl's logger."""
        return {
            "num_envs": self.num_envs,
            "num_actions": self.num_actions,
            "num_observations": self.num_observations,
            "max_episode_length": self.max_episode_length,
            "env_class": type(self.env).__name__,
        }

    # ------------------------------------------------------------------
    # rsl-rl VecEnv contract
    # ------------------------------------------------------------------

    def get_observations(self) -> Any:
        """Return the current observations as a TensorDict (lazy reset).

        Called by ``OnPolicyRunner`` before the first rollout; triggers a
        full inner reset if the wrapper has not observed anything yet. The
        returned TensorDict owns a clone — mutating it cannot corrupt the
        wrapper's cached state.
        """
        if self._obs is None:
            self.reset()
        assert self._obs is not None
        return TensorDict({OBS_GROUP: self._obs.clone()}, batch_size=[self.num_envs])

    def step(self, actions: Any) -> tuple[Any, "torch.Tensor", "torch.Tensor", dict]:
        """Apply actions, advance, auto-reset done envs, split time-outs.

        Returns ``(obs_td, rewards, dones, extras)`` per the rsl-rl 5.x
        contract: ``dones = terminated | time_outs`` and
        ``extras["time_outs"]`` (float) marks only the truncation half so PPO
        bootstraps values on time limits but not on success terminations.
        The observation of a done env is its post-reset observation.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for the rsl-rl adapter")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        expected = (self.num_envs, self.num_actions)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"expected actions of shape {expected}, got {tuple(actions.shape)}"
            )

        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        # Clone: the inner task rewrites its preallocated obs buffer on the
        # next step; the TensorDict handed to rsl-rl must stay stable.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).clone()
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(
            terminated, device=self.device, dtype=torch.bool
        ).reshape(-1)
        truncated = torch.as_tensor(
            truncated, device=self.device, dtype=torch.bool
        ).reshape(-1)

        self.episode_length_buf += 1
        # The wrapper's buffer is authoritative: honors rsl-rl's
        # init_at_random_ep_len even when the inner task never truncates.
        time_outs = truncated | (self.episode_length_buf >= self.max_episode_length)
        dones = terminated | time_outs

        if bool(dones.any()):
            done_idx = dones.nonzero(as_tuple=True)[0]
            reset_obs = self.env.reset_idx(done_idx)
            obs[done_idx] = torch.as_tensor(
                reset_obs, dtype=torch.float32, device=self.device
            )
            self.episode_length_buf[done_idx] = 0

        self._obs = obs

        extras_out: dict[str, Any] = {"time_outs": time_outs.float(), "log": {}}
        success = extras.get("success") if isinstance(extras, dict) else None
        if success is not None:
            success_t = torch.as_tensor(
                success, device=self.device, dtype=torch.bool
            ).reshape(-1)
            extras_out["success"] = success_t
            extras_out["log"]["/success"] = success_t.float().mean()

        obs_td = TensorDict({OBS_GROUP: obs}, batch_size=[self.num_envs])
        return obs_td, rewards, dones, extras_out

    # ------------------------------------------------------------------
    # Pass-through helpers (not part of the rsl-rl contract)
    # ------------------------------------------------------------------

    def reset(self, seeds: list[int] | None = None) -> tuple[Any, list[dict]]:
        """Reset all inner envs, zero the length buffer, return (TensorDict, infos)."""
        obs, infos = self.env.reset(seeds=seeds)
        self._obs = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).clone()
        self.episode_length_buf.zero_()
        td = TensorDict({OBS_GROUP: self._obs.clone()}, batch_size=[self.num_envs])
        return td, infos

    def reset_idx(self, envs_idx: Any) -> Any:
        """Reset a subset of inner envs and return their observations."""
        obs = self.env.reset_idx(envs_idx)
        if self._obs is not None:
            idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
            self._obs[idx] = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            )
            self.episode_length_buf[idx] = 0
        return obs

    def close(self) -> None:
        """Tear down the inner environment."""
        self.env.close()


def build_train_cfg(
    *,
    num_steps_per_env: int = 24,
    save_interval: int = 50,
    seed: int = 1,
    hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = True,
    init_std: float = 1.0,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 1.0,
    max_grad_norm: float = 1.0,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    desired_kl: float = 0.01,
    obs_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Assemble the rsl-rl 5.x ``train_cfg`` dict (pure, no rsl-rl import).

    The actor samples from a learnable Gaussian (needed by PPO's
    ``stochastic_output=True``); the critic is a deterministic value head.
    ``obs_groups`` defaults to the single :data:`OBS_GROUP` published by
    :class:`RslRlVecEnvWrapper`.
    """
    obs_groups = obs_groups or {"actor": [OBS_GROUP], "critic": [OBS_GROUP]}
    network = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [int(d) for d in hidden_dims],
        "activation": activation,
        "obs_normalization": bool(obs_normalization),
    }
    actor_cfg = {
        **network,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.GaussianDistribution",
            "init_std": float(init_std),
            "std_type": "log",
            "std_range": [0.01, 10.0],
            "learn_std": True,
        },
    }
    return {
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": int(num_learning_epochs),
            "num_mini_batches": int(num_mini_batches),
            "clip_param": float(clip_param),
            "gamma": float(gamma),
            "lam": float(lam),
            "value_loss_coef": float(value_loss_coef),
            "entropy_coef": float(entropy_coef),
            "learning_rate": float(learning_rate),
            "max_grad_norm": float(max_grad_norm),
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": float(desired_kl),
        },
        "actor": actor_cfg,
        "critic": dict(network),
        "obs_groups": obs_groups,
        "num_steps_per_env": int(num_steps_per_env),
        "save_interval": int(save_interval),
        "seed": int(seed),
        "check_for_nan": True,
    }


class CrsOnPolicyRunner(_RslRlOnPolicyRunner):
    """``OnPolicyRunner`` that stamps every checkpoint with self-description.

    rsl-rl's ``save`` writes ``alg.save() + {"iter", "infos"}``; this subclass
    merges a caller-provided ``checkpoint_info`` dict into ``infos`` so each
    ``model_*.pt`` carries everything needed to rebuild the actor and the
    task config at eval time (see :func:`make_actor_from_checkpoint`).
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        checkpoint_info: dict | None = None,
    ) -> None:
        self._crs_checkpoint_info = dict(checkpoint_info or {})
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save models plus the merged self-description into ``infos``."""
        merged = {**self._crs_checkpoint_info, **(infos or {})}
        super().save(path, infos=merged)


def make_actor_from_checkpoint(checkpoint: dict, device: str = "cpu") -> Any:
    """Rebuild the trained actor from a self-describing checkpoint.

    Reads the network architecture (hidden dims, activation, observation
    normalization, Gaussian distribution) from ``checkpoint["infos"]`` so the
    rebuilt ``MLPModel`` matches the trained state dict — including the
    observation normalizer's running statistics when normalization was on.

    Args:
        checkpoint: Dict as produced by ``torch.load`` on a
            :class:`CrsOnPolicyRunner` checkpoint.
        device: Target device for the rebuilt model.

    Returns:
        The rebuilt ``rsl_rl.models.MLPModel`` in eval mode.
    """
    _require_rl()
    if _RslRlMLPModel is None:  # pragma: no cover
        raise RuntimeError(_RL_EXTRA_HINT)
    infos = checkpoint.get("infos") or {}
    for key in ("obs_dim", "num_actions"):
        if key not in infos:
            raise ValueError(
                f"checkpoint infos are missing {key!r} — the checkpoint was "
                "not saved through CrsOnPolicyRunner and is not self-describing"
            )
    network = infos.get("network") or {}
    obs_dim = int(infos["obs_dim"])
    num_actions = int(infos["num_actions"])
    obs = TensorDict({OBS_GROUP: torch.zeros(1, obs_dim)}, batch_size=[1])
    obs_groups = {"actor": [OBS_GROUP], "critic": [OBS_GROUP]}
    actor = _RslRlMLPModel(
        obs,
        obs_groups,
        "actor",
        num_actions,
        hidden_dims=network.get("hidden_dims", [256, 256, 256]),
        activation=network.get("activation", "elu"),
        obs_normalization=network.get("obs_normalization", False),
        distribution_cfg=network.get("distribution_cfg"),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor.to(device)


__all__ = [
    "OBS_GROUP",
    "RslRlVecEnvWrapper",
    "CrsOnPolicyRunner",
    "build_train_cfg",
    "make_actor_from_checkpoint",
]
