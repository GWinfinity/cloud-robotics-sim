"""Contract tests for the rsl-rl 5.x adapter (no Genesis).

:class:`FakeGymVecEnv` mimics the ``GenesisVectorizedEnv`` surface —
``reset()`` / ``reset_idx(idx)`` / 5-tuple ``step()`` / ``num_observations`` /
``num_actions`` / ``device`` / ``task.max_episode_steps`` — with scripted
per-env dynamics so termination, truncation, and auto-reset behavior are
fully deterministic. The whole module is skipped without the optional ``rl``
extra (rsl-rl-lib + tensordict).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cloud_robotics_sim.core.vec_rsl_rl import (
    CrsOnPolicyRunner,
    RslRlVecEnvWrapper,
    build_train_cfg,
    make_actor_from_checkpoint,
)
from tests.optional_deps import rsl_rl_only

pytestmark = rsl_rl_only

try:  # tensordict ships with the rl extra; tests below never run without it
    from tensordict import TensorDict
except ImportError:  # pragma: no cover
    TensorDict = None  # type: ignore[assignment]


class FakeGymVecEnv:
    """Scripted 5-tuple batched env standing in for ``GenesisVectorizedEnv``.

    Per-step dynamics: each env's counter increments. Env ``i`` reports
    ``terminated`` once its counter reaches ``terminate_at[i]`` (a success),
    and ``truncated`` at ``truncate_at[i]``. Envs listed in ``freeze_done``
    never report either — used to prove the wrapper's ``episode_length_buf``
    is authoritative for timeouts. ``reset_idx`` records its arguments and
    stamps a fresh-episode marker into the observation.

    Observation layout (float32, ``num_observations`` wide):
    ``[env_index, counter, fresh_reset_marker, 0, ...]``.
    """

    def __init__(
        self,
        num_envs: int = 4,
        num_observations: int = 6,
        num_actions: int = 2,
        max_episode_steps: int = 10,
        terminate_at: dict[int, int] | None = None,
        truncate_at: dict[int, int] | None = None,
        freeze_done: set[int] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        self._task = type("Task", (), {"max_episode_steps": max_episode_steps})()
        self.terminate_at = terminate_at or {}
        self.truncate_at = truncate_at or {}
        self.freeze_done = freeze_done or set()
        self.counter = torch.zeros(num_envs, dtype=torch.long)
        self._fresh = torch.zeros(num_envs, dtype=torch.long)
        self.reset_calls: list[torch.Tensor] = []

    @property
    def task(self) -> object:
        return self._task

    def _obs(self) -> torch.Tensor:
        idx = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(1)
        cnt = self.counter.float().unsqueeze(1)
        fresh = self._fresh.float().unsqueeze(1)
        pad = torch.zeros(self.num_envs, self.num_observations - 3)
        return torch.cat([idx, cnt, fresh, pad], dim=1)

    def reset(self, seeds: list[int] | None = None):
        self.counter.zero_()
        self._fresh = torch.ones(self.num_envs, dtype=torch.long)
        return self._obs(), [{"seed": 0} for _ in range(self.num_envs)]

    def reset_idx(self, envs_idx) -> torch.Tensor:
        idx = torch.as_tensor(envs_idx, dtype=torch.long).reshape(-1)
        self.reset_calls.append(idx.clone())
        self.counter[idx] = 0
        self._fresh[idx] = 1
        return self._obs()[idx]

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"bad action shape {tuple(actions.shape)}")
        self.counter += 1
        self._fresh = torch.zeros(self.num_envs, dtype=torch.long)
        t = torch.full((self.num_envs,), 10**9)
        u = torch.full((self.num_envs,), 10**9)
        for i, at in self.terminate_at.items():
            t[i] = at
        for i, at in self.truncate_at.items():
            u[i] = at
        terminated = self.counter >= t
        truncated = self.counter >= u
        for i in self.freeze_done:
            terminated[i] = False
            truncated[i] = False
        reward = -self.counter.float()
        return self._obs(), reward, terminated, truncated, {"success": terminated}

    def close(self) -> None:
        pass


def _wrap(
    env: FakeGymVecEnv | None = None,
    max_episode_length: int | None = None,
    cfg: dict | None = None,
):
    env = env or FakeGymVecEnv()
    return RslRlVecEnvWrapper(env, max_episode_length=max_episode_length, cfg=cfg), env


class TestExposedAttributes:
    """The rsl-rl VecEnv surface the runner reads."""

    def test_exposes_num_envs_and_num_actions(self):
        wrapper, _ = _wrap(FakeGymVecEnv(num_envs=4, num_actions=2))
        assert wrapper.num_envs == 4
        assert wrapper.num_actions == 2

    def test_exposes_max_episode_length_scalar(self):
        wrapper, _ = _wrap(FakeGymVecEnv(max_episode_steps=10))
        assert wrapper.max_episode_length == 10
        wrapper2, _ = _wrap(FakeGymVecEnv(max_episode_steps=10), max_episode_length=7)
        assert wrapper2.max_episode_length == 7

    def test_episode_length_buf_starts_zeroed(self):
        wrapper, _ = _wrap(FakeGymVecEnv(num_envs=4))
        assert wrapper.episode_length_buf.shape == (4,)
        assert wrapper.episode_length_buf.dtype == torch.long
        assert wrapper.episode_length_buf.abs().sum() == 0

    def test_device_and_cfg_attributes(self):
        wrapper, _ = _wrap(FakeGymVecEnv(num_envs=4))
        assert wrapper.device == torch.device("cpu")
        assert wrapper.cfg["num_envs"] == 4
        custom = {"foo": 1}
        wrapper2, _ = _wrap(cfg=custom)
        assert wrapper2.cfg is custom


class TestGetObservations:
    """TensorDict contract and isolation."""

    def test_get_observations_returns_tensordict(self):
        wrapper, env = _wrap()
        assert wrapper._obs is None
        td = wrapper.get_observations()
        assert wrapper._obs is not None  # lazy full reset happened
        assert isinstance(td, TensorDict)
        assert env.counter.abs().sum() == 0

    def test_get_observations_policy_group_shape_and_batch_size(self):
        wrapper, _ = _wrap(FakeGymVecEnv(num_envs=4, num_observations=6))
        td = wrapper.get_observations()
        assert tuple(td.batch_size) == (4,)
        assert td["policy"].shape == (4, 6)

    def test_get_observations_returns_fresh_copy(self):
        wrapper, _ = _wrap()
        td1 = wrapper.get_observations()
        td1["policy"][0, 0] = 999.0
        td2 = wrapper.get_observations()
        assert td2["policy"][0, 0] != 999.0


class TestStepContract:
    """The 4-tuple rsl-rl consumes, and done/time_outs separation."""

    def test_step_returns_tensordict_reward_dones_extras(self):
        wrapper, _ = _wrap()
        wrapper.reset()
        obs_td, rewards, dones, extras = wrapper.step(torch.zeros(4, 2))
        assert isinstance(obs_td, TensorDict)
        assert obs_td["policy"].shape == (4, 6)
        assert extras["time_outs"].shape == (4,)
        assert "log" in extras

    def test_step_reward_shape_and_dtype(self):
        wrapper, _ = _wrap()
        wrapper.reset()
        _obs, rewards, dones, _extras = wrapper.step(torch.zeros(4, 2))
        assert rewards.shape == (4,)
        assert rewards.dtype == torch.float32
        assert dones.shape == (4,)
        assert dones.dtype == torch.bool

    def test_step_dones_union_of_terminated_and_truncated(self):
        wrapper, _ = _wrap(FakeGymVecEnv(terminate_at={0: 1}, truncate_at={1: 2}))
        wrapper.reset()
        _o, _r, dones, _e = wrapper.step(torch.zeros(4, 2))
        assert dones.tolist() == [True, False, False, False]
        _o, _r, dones, _e = wrapper.step(torch.zeros(4, 2))
        assert dones[1].item() is True

    def test_extras_time_outs_present_and_float(self):
        wrapper, _ = _wrap(FakeGymVecEnv(truncate_at={2: 1}))
        wrapper.reset()
        _o, _r, dones, extras = wrapper.step(torch.zeros(4, 2))
        assert extras["time_outs"].dtype == torch.float32
        assert extras["time_outs"].shape == (4,)
        assert extras["time_outs"][2].item() == 1.0
        assert dones[2].item() is True

    def test_time_outs_zero_for_pure_termination(self):
        wrapper, _ = _wrap(FakeGymVecEnv(terminate_at={0: 1}))
        wrapper.reset()
        _o, _r, dones, extras = wrapper.step(torch.zeros(4, 2))
        assert dones[0].item() is True
        assert extras["time_outs"].abs().max().item() == 0.0

    def test_step_accepts_numpy_actions_and_moves_to_device(self):
        wrapper, env = _wrap()
        wrapper.reset()
        actions = np.zeros((4, 2), dtype=np.float64)
        obs_td, rewards, dones, _extras = wrapper.step(actions)
        assert rewards.shape == (4,)
        assert obs_td["policy"].dtype == torch.float32
        assert env.counter.tolist() == [1, 1, 1, 1]


class TestAutoReset:
    """Done envs reset inside step(); returned obs is post-reset."""

    def test_terminated_env_obs_is_post_reset_obs(self):
        wrapper, env = _wrap(FakeGymVecEnv(terminate_at={0: 1}))
        wrapper.reset()
        obs_td, _r, dones, _e = wrapper.step(torch.zeros(4, 2))
        assert dones[0].item() is True
        assert len(env.reset_calls) == 1
        assert env.reset_calls[0].tolist() == [0]
        obs = obs_td["policy"]
        assert obs[0, 1].item() == 0.0  # counter channel reset to 0
        assert obs[0, 2].item() == 1.0  # fresh-episode marker set
        assert obs[1, 1].item() == 1.0  # non-done env advanced normally

    def test_non_done_envs_not_reset_by_subset_resets(self):
        wrapper, env = _wrap(FakeGymVecEnv(terminate_at={0: 1, 2: 1}))
        wrapper.reset()
        wrapper.step(torch.zeros(4, 2))
        assert env.reset_calls[0].tolist() == [0, 2]
        assert env.counter.tolist() == [0, 1, 0, 1]  # 1/3 advanced; 0/2 reset

        # A later step resets only its own done subset.
        wrapper2, env2 = _wrap(FakeGymVecEnv(truncate_at={1: 2}))
        wrapper2.reset()
        wrapper2.step(torch.zeros(4, 2))
        assert env2.reset_calls == []  # nothing done at step 1
        wrapper2.step(torch.zeros(4, 2))
        assert env2.reset_calls[-1].tolist() == [1]
        assert env2.counter[0].item() == 2  # env 0 never reset


class TestEpisodeLengthBuf:
    """The wrapper-owned step counter."""

    def test_episode_length_buf_increments_each_step(self):
        wrapper, _ = _wrap(FakeGymVecEnv(freeze_done={0, 1, 2, 3}))
        wrapper.reset()
        for expected in (1, 2, 3):
            wrapper.step(torch.zeros(4, 2))
            assert wrapper.episode_length_buf.tolist() == [expected] * 4

    def test_episode_length_buf_zeroed_on_done(self):
        wrapper, _ = _wrap(FakeGymVecEnv(terminate_at={0: 1}))
        wrapper.reset()
        wrapper.step(torch.zeros(4, 2))
        assert wrapper.episode_length_buf.tolist() == [0, 1, 1, 1]


class TestWrapperAuthoritativeTimeouts:
    """Timeouts fire from episode_length_buf even without inner truncation."""

    def test_timeout_when_buf_reaches_max_episode_length(self):
        wrapper, env = _wrap(
            FakeGymVecEnv(freeze_done={0, 1, 2, 3}), max_episode_length=3
        )
        wrapper.reset()
        wrapper.step(torch.zeros(4, 2))
        wrapper.step(torch.zeros(4, 2))
        _o, _r, dones, extras = wrapper.step(torch.zeros(4, 2))
        assert dones.tolist() == [True] * 4  # inner env never truncated
        assert extras["time_outs"].min().item() == 1.0
        assert wrapper.episode_length_buf.abs().sum() == 0
        assert env.counter.abs().sum() == 0  # all were auto-reset

    def test_randomized_episode_length_buf_respected(self):
        # rsl-rl's init_at_random_ep_len rewrites episode_length_buf.
        wrapper, _ = _wrap(
            FakeGymVecEnv(freeze_done={0, 1, 2, 3}), max_episode_length=5
        )
        wrapper.reset()
        wrapper.episode_length_buf = torch.tensor([0, 1, 2, 3])
        wrapper.step(torch.zeros(4, 2))
        wrapper.step(torch.zeros(4, 2))
        assert wrapper.episode_length_buf.tolist() == [2, 3, 4, 0]  # env 3 hit 5


class TestValidation:
    """Bad inputs fail loudly; outputs pass rsl-rl's NaN guard."""

    def test_wrong_action_shape_raises_value_error(self):
        wrapper, _ = _wrap()
        wrapper.reset()
        with pytest.raises(ValueError):
            wrapper.step(torch.zeros(4, 3))
        with pytest.raises(ValueError):
            wrapper.step(torch.zeros(3, 2))

    def test_rewards_and_dones_are_finite(self):
        from rsl_rl.utils import check_nan

        wrapper, _ = _wrap(FakeGymVecEnv(terminate_at={0: 2}, truncate_at={1: 3}))
        wrapper.reset()
        for _ in range(4):
            obs_td, rewards, dones, _extras = wrapper.step(torch.zeros(4, 2))
            check_nan(obs_td, rewards, dones)  # raises on NaN
            assert torch.isfinite(rewards).all()


class TestSelfDescribingCheckpoint:
    """CrsOnPolicyRunner stamps infos; the actor rebuilds from them."""

    def test_save_writes_self_describing_checkpoint(self, tmp_path):
        env = FakeGymVecEnv(num_envs=4, num_observations=6, num_actions=2)
        wrapper = RslRlVecEnvWrapper(env)
        train_cfg = build_train_cfg(
            num_steps_per_env=4, save_interval=10_000, seed=3, hidden_dims=[32, 32]
        )
        checkpoint_info = {
            "reward_mode": "guided",
            "obs_dim": 6,
            "num_actions": 2,
            "network": train_cfg["actor"],
        }
        runner = CrsOnPolicyRunner(
            env=wrapper,
            train_cfg=train_cfg,
            log_dir=None,
            device="cpu",
            checkpoint_info=checkpoint_info,
        )
        path = tmp_path / "model_5.pt"
        runner.save(str(path))

        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["iter"] == 0
        infos = checkpoint["infos"]
        assert infos["reward_mode"] == "guided"
        assert infos["obs_dim"] == 6
        assert infos["num_actions"] == 2
        assert infos["network"]["hidden_dims"] == [32, 32]

        actor = make_actor_from_checkpoint(checkpoint, device="cpu")
        obs_td = TensorDict({"policy": torch.zeros(4, 6)}, batch_size=[4])
        actions = actor(obs_td, stochastic_output=False)
        assert actions.shape == (4, 2)
        assert torch.isfinite(actions).all()


class TestBuildTrainCfg:
    """Pure assembly of the rsl-rl 5.x train_cfg dict."""

    def test_build_train_cfg_defaults(self):
        cfg = build_train_cfg(seed=7)
        assert cfg["algorithm"]["class_name"] == "rsl_rl.algorithms.PPO"
        assert cfg["actor"]["class_name"] == "rsl_rl.models.MLPModel"
        assert cfg["critic"]["class_name"] == "rsl_rl.models.MLPModel"
        assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}
        assert (
            cfg["actor"]["distribution_cfg"]["class_name"]
            == "rsl_rl.modules.GaussianDistribution"
        )
        assert "distribution_cfg" not in cfg["critic"]
        assert cfg["seed"] == 7
        assert cfg["num_steps_per_env"] == 24
        assert cfg["save_interval"] == 50
