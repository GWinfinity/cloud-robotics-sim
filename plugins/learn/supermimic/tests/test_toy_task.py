"""End-to-end tests for the toy reference task (ToyEnv + ToyMdp + learner)."""

from __future__ import annotations

import torch
from supermimic.contracts import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
    SupportsSuperMimicProtocol,
)
from supermimic.geometry import unflatten_state
from supermimic.learner import SuperMimicConfig, SuperMimicLearner
from supermimic.toys import NUM_JOINTS, ToyEnv, ToyMdp, reference


def _tiny_config(**overrides) -> SuperMimicConfig:
    kwargs = dict(
        iterations=3,
        buffer_steps=64,
        utd=1.0,
        horizon=4,
        goal_frames=3,
        hist=3,
        wm_window=4,
        wm_batches=1,
        wm_batch=16,
        pi_batch=16,
        lr=1e-3,
        critic_epochs=1,
        critic_minibatches=2,
        warmup_steps=6,
        wm_only_iters=0,
        ckpt_every=3,
        log_every=1,
        log_backend="no_print",
        amp="none",
        compile=False,
        test_hidden_dims=(16,),
        z_dim=4,
    )
    kwargs.update(overrides)
    return SuperMimicConfig(**kwargs)


class TestToyEnvProtocol:
    def test_env_exposes_mdp_and_state_contract(self):
        env = ToyEnv(num_envs=4)
        assert isinstance(env.supermimic_mdp, SuperMimicMdpProtocol)
        assert isinstance(env, SupportsSuperMimicProtocol)

        state = env.init_state()
        info = state.info[SUPERMIMIC_STATE_KEY]
        assert info["s"].shape == (4, 13 + 2 * NUM_JOINTS)
        assert info["f"].shape == (4,)
        assert info["xi"].shape == (4, 2)
        assert info["a_prev"].shape == (4, NUM_JOINTS)
        assert (info["ep_step"] == 0).all()

    def test_step_returns_transition_and_advances(self):
        env = ToyEnv(num_envs=4)
        env.init_state()
        out = env.step(torch.zeros(4, NUM_JOINTS))
        transition = out.info[SUPERMIMIC_TRANSITION_KEY]
        new_info = out.info[SUPERMIMIC_STATE_KEY]
        assert transition["s1"].shape == transition["s"].shape
        assert torch.isfinite(transition["r"]).all()
        # ``f``/``ep_step`` in the transition are the pre-step values.
        assert (transition["ep_step"] == 0).all()
        assert (new_info["f"] == transition["f"] + 1).all()
        assert (new_info["ep_step"] == 1).all()

    def test_episode_ends_and_resamples(self):
        env = ToyEnv(num_envs=4, clip_length=4)
        env.init_state()
        done_seen = False
        for _ in range(8):
            out = env.step(torch.zeros(4, NUM_JOINTS))
            if bool(out.info[SUPERMIMIC_TRANSITION_KEY]["done"].any()):
                done_seen = True
                break
        assert done_seen

    def test_reset_starts_on_reference(self):
        env = ToyEnv(num_envs=8)
        env.init_state()
        info = env.state.info[SUPERMIMIC_STATE_KEY]
        state = unflatten_state(info["s"])
        q_ref, qd_ref = reference(info["f"])
        torch.testing.assert_close(state[4], q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(state[5], qd_ref, atol=1e-5, rtol=1e-5)


class TestToyMdpHeads:
    def test_goal_stacks_upcoming_reference(self):
        mdp = ToyMdp()
        frames = torch.tensor([0, 1])
        goal = mdp.goal(frames, num_frames=3)
        assert goal.shape == (2, 3 * mdp.goal_frame_dim)
        # Blocks are f+1..f+3 in ascending order; the last block is f+3 and
        # the block at offset 0 (used by pi_ref_skip) is the next frame.
        q_ref_next, _ = reference(frames + 1)
        q_ref_last, _ = reference(frames + 3)
        torch.testing.assert_close(goal[:, :NUM_JOINTS], q_ref_next)
        torch.testing.assert_close(
            goal[:, -mdp.goal_frame_dim :][:, :NUM_JOINTS], q_ref_last
        )

    def test_reward_is_max_on_reference(self):
        mdp = ToyMdp()
        frames = torch.tensor([0, 5])
        q_ref, qd_ref = reference(frames)
        n = 2
        on_ref = (
            torch.zeros(n, 3),
            torch.eye(3).expand(n, 3, 3),
            torch.zeros(n, 3),
            torch.zeros(n, 3),
            q_ref,
            qd_ref,
        )
        zero = torch.zeros(n, NUM_JOINTS)
        reward = mdp.reward(on_ref, frames, zero, zero)
        assert reward.shape == (n,)
        assert (reward > 0.9 * mdp.max_step_reward).all()

    def test_valid_imagination_start_respects_clip_end(self):
        mdp = ToyMdp(clip_length=8)
        frames = torch.arange(10)
        valid = mdp.valid_imagination_start(frames, horizon=4)
        # frames 5..9 are within 4 of the clip end of their 8-frame window.
        assert valid[:5].all()
        assert not valid[5:].any()


class TestToyEndToEnd:
    def test_training_run_completes_with_finite_metrics(self, tmp_path):
        learner = SuperMimicLearner(ToyEnv(num_envs=4), _tiny_config(), tmp_path)
        result = learner.train()

        assert result["completed"] is True
        assert result["iteration"] == 3
        assert result["sim_steps"] > 0
        assert (tmp_path / "last.pt").is_file()
        assert (tmp_path / "train_log.csv").is_file()

    def test_reference_residual_policy_converges_toward_reference(self, tmp_path):
        """pi_ref_skip turns the policy into a residual on the next-frame
        reference; after a short run its raw output must be far smaller than
        the reference amplitude (the reference does the heavy lifting).
        """
        torch.manual_seed(0)
        learner = SuperMimicLearner(
            ToyEnv(num_envs=16),
            _tiny_config(
                pi_ref_skip=True,
                iterations=8,
                goal_frames=4,
                warmup_steps=16,
                buffer_steps=256,
                pi_batch=32,
                wm_batch=32,
            ),
            tmp_path,
        )
        learner.train()
        obs_hist = torch.zeros(8, learner.cfg.hist, learner.mdp.obs_dim)
        frames = torch.arange(8)
        goal = learner.mdp.goal(frames, learner.cfg.goal_frames)
        residual = learner.pi(obs_hist, goal)
        q_ref, _ = reference(frames)
        assert residual.abs().mean() < q_ref.abs().mean()
