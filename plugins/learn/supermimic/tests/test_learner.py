"""CPU training-loop tests for the protocol-driven SuperMimic learner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import pytest
import torch
from supermimic.contracts import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
)
from supermimic.learner import SuperMimicConfig, SuperMimicLearner


class FakeMdp:
    """Task-agnostic stand-in MDP: reference offset 15 mirrors the upstream
    G1 goal layout (3 pos + 6 rot + 3 linvel + 3 angvel), letting the
    ref-skip wiring be exercised without any robot specifics.
    """

    num_joints = 2
    context_dim = 2
    obs_dim = 6
    goal_frame_dim = 3
    max_step_reward = 0.1
    goal_q_offset = 15

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (state[0][..., :1], previous_action, context, frames.float().unsqueeze(-1)),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return frames.float().unsqueeze(-1).expand(-1, num_frames * self.goal_frame_dim)

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        tracking = self.max_step_reward * torch.exp(-(state[0] - 0.1).square().mean(-1))
        return tracking - 0.001 * (action - previous_action).square().mean(-1)

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del state
        return torch.zeros_like(frames, dtype=torch.float32)

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (state[0], state[4], normalized_context, frames.float().unsqueeze(-1)), -1
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(context)

    def valid_imagination_start(
        self, frames: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        del horizon
        return torch.ones_like(frames, dtype=torch.bool)

    def to(self, device: str | torch.device) -> "FakeMdp":
        self.moved_to = torch.device(device)
        return self


@dataclass
class FakeState:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict
    final_observation: None = None


class FakeEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.mdp = FakeMdp()
        self.state: FakeState | None = None
        self.initial_state = torch.zeros(num_envs, 13 + 2 * self.mdp.num_joints)
        self.initial_state[:, 3] = 1.0
        self.initial_frames = torch.arange(num_envs, dtype=torch.long)
        self.context = torch.full((num_envs, self.mdp.context_dim), 0.2)

    @property
    def supermimic_mdp(self) -> FakeMdp:
        return self.mdp

    def init_state(self) -> FakeState:
        info = {
            "s": self.initial_state.clone(),
            "f": self.initial_frames.clone(),
            "xi": self.context.clone(),
            "a_prev": torch.zeros(self.num_envs, self.mdp.num_joints),
            "ep_step": torch.zeros(self.num_envs, dtype=torch.int32),
        }
        self.state = FakeState(
            obs={},
            reward=torch.zeros(self.num_envs),
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            info={SUPERMIMIC_STATE_KEY: info},
        )
        return self.state

    def step(self, actions: torch.Tensor) -> FakeState:
        assert self.state is not None
        current = self.state.info[SUPERMIMIC_STATE_KEY]
        next_state = current["s"].clone()
        next_state[:, :3] += 0.001
        next_state[:, 13 : 13 + self.mdp.num_joints] += 0.01 * actions
        next_state[:, 13 + self.mdp.num_joints :] += 0.001 * actions
        next_frames = current["f"] + 1
        episode_steps = current["ep_step"] + 1
        done = episode_steps >= 3
        reward = self.mdp.reward(
            (
                next_state[:, 0:3],
                torch.eye(3).expand(self.num_envs, 3, 3),
                next_state[:, 7:10],
                next_state[:, 10:13],
                next_state[:, 13:15],
                next_state[:, 15:17],
            ),
            next_frames,
            actions,
            current["a_prev"],
        )
        transition = {
            "s": current["s"].clone(),
            "s1": next_state.clone(),
            "a": actions.clone(),
            "a_prev": current["a_prev"].clone(),
            "xi": current["xi"].clone(),
            "r": reward,
            "f": current["f"].clone(),
            "ep_step": current["ep_step"].clone(),
            "done": done,
            "term": torch.zeros_like(done),
        }
        if bool(done.any()):
            transition["ep_len"] = episode_steps[done].float()
            transition["ep_ret_norm"] = reward[done] / (
                episode_steps[done].float() * self.mdp.max_step_reward
            )
        post_reset_state = next_state.clone()
        post_reset_state[done] = self.initial_state[done]
        post_reset_action = torch.where(
            done.unsqueeze(-1), torch.zeros_like(actions), actions
        )
        post_reset_steps = torch.where(
            done, torch.zeros_like(episode_steps), episode_steps
        )
        info = {
            "s": post_reset_state,
            "f": next_frames,
            "xi": current["xi"].clone(),
            "a_prev": post_reset_action,
            "ep_step": post_reset_steps,
        }
        self.state = FakeState(
            obs={},
            reward=reward,
            terminated=done,
            truncated=torch.zeros_like(done),
            info={
                SUPERMIMIC_STATE_KEY: info,
                SUPERMIMIC_TRANSITION_KEY: transition,
            },
        )
        return self.state


def _config() -> SuperMimicConfig:
    return SuperMimicConfig(
        iterations=1,
        buffer_steps=16,
        utd=1.0,
        horizon=2,
        goal_frames=2,
        hist=2,
        wm_window=2,
        wm_batches=1,
        wm_batch=4,
        pi_batch=4,
        lr=1e-3,
        critic_epochs=1,
        critic_minibatches=2,
        warmup_steps=3,
        wm_only_iters=0,
        ckpt_every=1,
        log_every=1,
        log_backend="no_print",
        amp="none",
        compile=False,
        test_hidden_dims=(8,),
    )


def test_tf32_only_enabled_for_fp32_training(tmp_path) -> None:
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        SuperMimicLearner(FakeEnv(), _config(), tmp_path / "fp32")
        assert torch.backends.cuda.matmul.allow_tf32 is True
        SuperMimicLearner(FakeEnv(), replace(_config(), amp="bf16"), tmp_path / "bf16")
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_collect_and_updates_run_on_cpu(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)

    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = (
        learner.policy_update()
    )

    assert learner.buffer.size == 3
    assert learner.sim_steps == 6
    assert learner.obs_hist.shape == (2, 2, 6)
    for metric in (
        wm_loss,
        wm_error,
        policy_loss,
        value_loss,
        imagined_reward,
        grad_norm,
        survival,
    ):
        assert metric.ndim == 0
        assert torch.isfinite(metric)


def test_structured_muon_and_no_sync_training_options_run_on_cpu(tmp_path) -> None:
    cfg = replace(
        _config(),
        wm_hidden=(8, 8),
        policy_hidden=(8, 8),
        critic_hidden=(8, 8),
        test_hidden_dims=None,
        wm_structured=True,
        pi_opt="muon",
        wm_opt="muon",
        no_sync=True,
        fused_adam=True,
    )
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = (
        learner.policy_update()
    )

    assert learner.wm.structured is True
    assert learner.cfg.no_sync is True
    assert torch.isfinite(wm_loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(imagined_reward)
    assert torch.isfinite(grad_norm)
    assert torch.isfinite(survival)
    assert wm_error.ndim == 0
    assert survival.ndim == 0


def test_reference_skip_uses_mdp_goal_offset_and_pose(tmp_path) -> None:
    env = FakeEnv()
    env.mdp.default_q = torch.tensor([0.2, -0.1])
    env.mdp.action_scale = torch.tensor([0.4, 0.5])
    cfg = replace(_config(), pi_ref_skip=True, goal_frames=8)
    learner = SuperMimicLearner(env, cfg, tmp_path)

    assert learner.pi.ref_off == env.mdp.goal_q_offset
    learner.collect(3)
    start, _ = learner._sample_valid_starts(4)
    obs_hist = torch.zeros(4, cfg.hist, env.mdp.obs_dim)
    goal = learner.mdp.goal(start["f"], cfg.goal_frames)
    action = learner.pi(obs_hist, goal)

    assert action.shape == (4, 2)
    torch.testing.assert_close(learner.pi.ref_q0, torch.tensor([0.2, -0.1]))
    torch.testing.assert_close(learner.pi.ref_scale, torch.tensor([0.4, 0.5]))


def test_wm_mirror_requires_mdp_mirror(tmp_path) -> None:
    with pytest.raises(ValueError, match="mirror"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), wm_mirror=0.5), tmp_path / "mirror"
        )


def test_policy_imagination_backpropagates_without_changing_world_model(
    tmp_path,
) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.collect(3)
    wm_before = {
        name: value.detach().clone() for name, value in learner.wm.state_dict().items()
    }

    _, _, _, grad_norm, _ = learner.policy_update()

    assert grad_norm > 0
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in learner.pi.parameters()
    )
    for name, value in learner.wm.state_dict().items():
        torch.testing.assert_close(value, wm_before[name])


def test_matching_explicit_device_disables_bridging(tmp_path) -> None:
    env = FakeEnv()
    learner = SuperMimicLearner(env, _config(), tmp_path, device="cpu")

    assert learner._bridged is False
    assert learner.mdp is env.mdp


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bf16_compiled_training_and_checkpoint(tmp_path) -> None:
    pytest.importorskip(
        "triton", reason="torch.compile inductor on CUDA requires triton"
    )
    cfg = replace(_config(), amp="bf16", compile=True)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path, device="cuda")

    result = learner.train()

    assert result["completed"] is True
    assert learner.device.type == "cuda"
    assert learner.buffer.data["s"].device.type == "cuda"
    assert (tmp_path / "last.pt").is_file()
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["iteration"] == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_bridged_learner_moves_training_to_mps(tmp_path) -> None:
    env = FakeEnv()
    learner = SuperMimicLearner(env, _config(), tmp_path, device="mps")

    assert learner._bridged is True
    assert learner.device == torch.device("mps")
    assert learner.mdp.moved_to == torch.device("mps")

    learner.collect(3)
    wm_loss, _ = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = (
        learner.policy_update()
    )

    assert learner.buffer.data["s"].device.type == "mps"
    assert learner.obs_hist.device.type == "mps"
    for metric in (
        wm_loss,
        policy_loss,
        value_loss,
        imagined_reward,
        grad_norm,
        survival,
    ):
        assert metric.ndim == 0
        assert torch.isfinite(metric)
    # The env stays on its own device across the bridge.
    assert env.state is not None
    current = env.state.info[SUPERMIMIC_STATE_KEY]
    assert current["s"].device.type == "cpu"


def test_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    final = learner.train()

    assert final["completed"] is True
    assert final["iteration"] == 1
    checkpoint_path = tmp_path / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert set(checkpoint) == {
        "wm",
        "pi",
        "critic",
        "opt_wm",
        "opt_pi",
        "opt_v",
        "iteration",
        "wall_time",
        "sim_steps",
        "args",
        "saved_at",
    }
    small = torch.load(tmp_path / "ckpt_000001.pt", weights_only=True)
    assert set(small) == {
        "wm",
        "pi",
        "iteration",
        "wall_time",
        "sim_steps",
        "args",
        "saved_at",
    }
    with (tmp_path / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[0][:3] == ["iteration", "wall_h", "sim_steps"]
    assert rows[1][0] == "1"

    resumed = SuperMimicLearner(
        FakeEnv(), replace(cfg, iterations=2), tmp_path / "resumed"
    )
    with pytest.raises(ValueError, match="requires a full last.pt"):
        resumed.load(tmp_path / "ckpt_000001.pt")
    resumed.load(checkpoint_path)
    assert resumed.iteration == learner.iteration
    assert resumed.sim_steps == learner.sim_steps
    assert resumed.wall_time == checkpoint["wall_time"]
    for name in ("wm", "pi", "critic"):
        original_state = getattr(learner, name).state_dict()
        restored_state = getattr(resumed, name).state_dict()
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            torch.testing.assert_close(restored_state[key], original_state[key])

    resumed_final = resumed.train()
    assert resumed_final["completed"] is True
    assert resumed_final["iteration"] == 2
    assert resumed_final["sim_steps"] > final["sim_steps"]


def test_resume_rebuilds_warmup_history(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.train()
    resumed = SuperMimicLearner(
        FakeEnv(),
        replace(cfg, iterations=1, warmup_steps=6),
        tmp_path / "resumed",
    )

    resumed.train(resume=tmp_path / "last.pt")

    assert resumed.buffer.size == 6


def test_tensorboard_backend_logs_real_metric_names(tmp_path) -> None:
    pytest.importorskip("tensorboard", reason="tensorboard extra not installed")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    learner = SuperMimicLearner(
        FakeEnv(), replace(_config(), log_backend="tensorboard"), tmp_path
    )
    learner.train()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    expected = {
        "Loss/world_model",
        "Loss/world_model_err8",
        "Loss/policy",
        "Loss/value",
        "Loss/imagined_reward",
        "Loss/imagined_survival",
        "Train/grad_norm",
        "Train/learning_rate",
        "Reward/sim_step_norm",
        "Episode/length",
        "Episode/return_norm",
        "Episode/terminated_frac",
        "Perf/iteration_time",
        "Perf/collect_time",
        "Perf/learning_time",
        "Perf/it_per_s",
        "Perf/sim_steps_per_s",
    }
    assert expected <= tags


def test_no_print_backend_writes_no_events(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.train()

    assert not list(tmp_path.glob("events.out.tfevents.*"))


def test_ema_disabled_by_default(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.train()

    assert learner.pi_ema is None
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert "pi_ema" not in checkpoint

    with pytest.raises(ValueError, match="ema_decay"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), ema_decay=1.0), tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="ema_decay"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), ema_decay=-0.1), tmp_path / "bad2"
        )


def test_ema_tracks_policy_and_roundtrips(tmp_path) -> None:
    cfg = replace(_config(), ema_decay=0.5)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.collect(3)
    learner.wm_update()
    pi_before = {name: value.clone() for name, value in learner.pi.state_dict().items()}
    learner.policy_update()

    assert learner.pi_ema is not None
    ema_state = learner.pi_ema.state_dict()
    pi_state = learner.pi.state_dict()
    changed = False
    for name, tensor in pi_state.items():
        if name in dict(learner.pi.named_parameters()):
            expected = 0.5 * pi_before[name] + 0.5 * tensor
            torch.testing.assert_close(ema_state[name], expected)
            changed = changed or not torch.allclose(ema_state[name], tensor)
        else:
            torch.testing.assert_close(ema_state[name], tensor)
    assert changed

    final = learner.train()
    assert final["completed"] is True
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert "pi_ema" in checkpoint

    resumed = SuperMimicLearner(FakeEnv(), cfg, tmp_path / "resumed")
    resumed.load(tmp_path / "last.pt")
    assert resumed.pi_ema is not None
    for name, tensor in learner.pi_ema.state_dict().items():
        torch.testing.assert_close(resumed.pi_ema.state_dict()[name], tensor)


def test_resume_with_ema_requires_ema_checkpoint(tmp_path) -> None:
    SuperMimicLearner(FakeEnv(), _config(), tmp_path).train()
    resumed = SuperMimicLearner(
        FakeEnv(), replace(_config(), ema_decay=0.999), tmp_path / "resumed"
    )
    with pytest.raises(ValueError, match="pi_ema"):
        resumed.load(tmp_path / "last.pt")


def _scale_critic_output(learner: SuperMimicLearner, scale: float) -> None:
    """Make the critic output ``scale * features`` so bootstrap gradients reach the policy."""
    last = learner.critic.net[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.fill_(scale)
        last.bias.zero_()


def test_value_clip_bounds_policy_bootstrap(tmp_path) -> None:
    clipped = SuperMimicLearner(
        FakeEnv(), replace(_config(), value_clip=True), tmp_path / "clipped"
    )
    clipped.collect(3)
    for _ in range(5):
        clipped.wm_update()
    _scale_critic_output(clipped, 1e4)
    _, _, _, clipped_grad, _ = clipped.policy_update()
    assert torch.isfinite(clipped_grad)

    plain = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "plain")
    plain.collect(3)
    for _ in range(5):
        plain.wm_update()
    _scale_critic_output(plain, 1e4)
    _, _, _, plain_grad, _ = plain.policy_update()

    assert clipped_grad * 10 < plain_grad


def test_value_clip_helper_respects_mdp_bounds(tmp_path) -> None:
    learner = SuperMimicLearner(
        FakeEnv(), replace(_config(), value_clip=True), tmp_path / "clipped"
    )
    limit = learner.mdp.max_step_reward / (1.0 - learner.cfg.gamma)
    values = torch.tensor([-1e6, -limit, 0.0, limit, 1e6], dtype=torch.float64)
    clamped = learner._clamp_value(values)
    assert float(clamped.min()) >= -limit
    assert float(clamped.max()) <= limit
    torch.testing.assert_close(clamped[[1, 2, 3]], values[[1, 2, 3]])

    plain = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "plain")
    torch.testing.assert_close(plain._clamp_value(values), values)


def test_critic_targets_clamp_recursive_overestimation(tmp_path) -> None:
    horizon = 2
    learner = SuperMimicLearner(
        FakeEnv(),
        replace(_config(), value_clip=True, horizon=horizon),
        tmp_path / "clipped",
    )
    limit = learner.mdp.max_step_reward / (1.0 - learner.cfg.gamma)
    batch = 4
    rewards = torch.zeros(horizon, batch)
    dones = torch.zeros(horizon, batch)
    values = torch.full((horizon + 1, batch), 1e4)

    targets = learner._critic_targets(rewards, dones, values)

    assert targets.shape == (horizon, batch)
    assert float(targets.max()) <= limit

    plain = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "plain")
    unclipped = plain._critic_targets(rewards, dones, values)
    assert float(unclipped.min()) > limit


def test_noise_free_fraction_disables_exploration_noise(tmp_path) -> None:
    action = torch.zeros(2, 2)
    full = SuperMimicLearner(
        FakeEnv(),
        replace(_config(), action_noise=5.0, noise_free_frac=1.0),
        tmp_path / "full",
    )
    torch.manual_seed(0)
    torch.testing.assert_close(full._add_exploration_noise(action), action)

    plain = SuperMimicLearner(
        FakeEnv(), replace(_config(), action_noise=5.0), tmp_path / "plain"
    )
    torch.manual_seed(0)
    assert not torch.allclose(plain._add_exploration_noise(action), action)


def test_noise_free_fraction_selects_envs(tmp_path) -> None:
    learner = SuperMimicLearner(
        FakeEnv(num_envs=64),
        replace(_config(), action_noise=5.0, noise_free_frac=0.5),
        tmp_path,
    )
    torch.manual_seed(1)
    noisy = learner._add_exploration_noise(torch.zeros(64, 2))
    unchanged = int((noisy == 0).all(-1).sum())
    assert 0 < unchanged < 64


def test_value_clip_and_noise_free_frac_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="noise_free_frac"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), noise_free_frac=1.5), tmp_path / "frac"
        )
    with pytest.raises(ValueError, match="value_clip"):
        SuperMimicLearner(
            FakeEnv(),
            replace(_config(), value_clip=True, no_critic=True),
            tmp_path / "critic",
        )
    with pytest.raises(ValueError, match="gamma"):
        SuperMimicLearner(
            FakeEnv(),
            replace(_config(), value_clip=True, gamma=1.0),
            tmp_path / "gamma",
        )


def test_position_obs_and_contact_reward_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="position_obs"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), position_obs="bad"), tmp_path / "obs"
        )
    with pytest.raises(ValueError, match="contact_reward"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), contact_reward=-0.5), tmp_path / "contact"
        )


def test_position_obs_and_contact_reward_must_match_mdp(tmp_path) -> None:
    env = FakeEnv()
    env.mdp.position_obs = "root"
    with pytest.raises(ValueError, match="position_obs"):
        SuperMimicLearner(env, _config(), tmp_path / "obs")

    env = FakeEnv()
    env.mdp.contact_reward = 0.25
    with pytest.raises(ValueError, match="contact_reward"):
        SuperMimicLearner(env, _config(), tmp_path / "contact")


def test_eprobe_requires_env_cfg_and_valid_ranges(tmp_path) -> None:
    with pytest.raises(TypeError, match="cfg"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), eprobe_share=0.3), tmp_path / "nocfg"
        )
    with pytest.raises(ValueError, match="eprobe_share"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), eprobe_share=1.5), tmp_path / "share"
        )

    # A cfg attribute alone passes construction; the sweep itself needs the
    # full task hooks (motion metadata, reset_to_frames, set_focus).
    env = FakeEnv()
    env.cfg = type("Cfg", (), {})()
    learner = SuperMimicLearner(
        env, replace(_config(), eprobe_share=0.3), tmp_path / "cfg"
    )
    assert learner._eprobe_env is None
