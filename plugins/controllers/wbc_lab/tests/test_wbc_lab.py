"""Tests for WBC-Lab Controller Plugin.

Tests are designed to run without Genesis installed (mocking physics)
to validate the algorithmic components.
"""

import numpy as np
import pytest
from pathlib import Path


class TestG1RobotConfig:
    """G1 robot configuration tests."""

    def test_default_config(self):
        """Default G1 config has correct dimensions."""
        from plugins.controllers.wbc_lab.core.robots.g1_config import (
            G1RobotConfig, G1_NUM_JOINTS, G1_JOINT_NAMES,
        )

        cfg = G1RobotConfig()
        assert cfg.num_joints == 29
        assert cfg.num_dofs == 29
        assert len(cfg.joint_names) == 29
        assert len(cfg.home_qpos) == 29

    def test_pd_gains(self):
        """PD gains are non-negative and well-formed."""
        from plugins.controllers.wbc_lab.core.robots.g1_config import G1RobotConfig

        cfg = G1RobotConfig()
        kp = cfg.kp_array()
        kd = cfg.kd_array()
        limits = cfg.torque_limits()

        assert len(kp) == 29
        assert len(kd) == 29
        assert len(limits) == 29
        assert all(k > 0 for k in kp)
        assert all(k > 0 for k in kd)
        assert all(l > 0 for l in limits)

    def test_symmetry_pairs(self):
        """Symmetry pairs have valid indices."""
        from plugins.controllers.wbc_lab.core.robots.g1_config import G1RobotConfig

        cfg = G1RobotConfig()
        for left, right, scale in cfg.symmetry_pairs:
            assert 0 <= left < cfg.num_joints
            assert 0 <= right < cfg.num_joints
            assert left != right
            assert scale in (1.0, -1.0)


class TestMotionLoader:
    """Motion data loading tests."""

    def test_motion_clip_duration(self):
        """MotionClip duration is computed correctly."""
        from plugins.controllers.wbc_lab.core.motion.motion_loader import MotionClip

        clip = MotionClip(
            name="test",
            fps=30.0,
            num_frames=90,
            body_pos_w=np.zeros((90, 14, 3)),
            body_quat_w=np.zeros((90, 14, 4)),
            body_lin_vel_w=np.zeros((90, 14, 3)),
            body_ang_vel_w=np.zeros((90, 14, 3)),
            joint_pos=np.zeros((90, 29)),
            joint_vel=np.zeros((90, 29)),
            base_pos_w=np.zeros((90, 3)),
            base_quat_w=np.zeros((90, 4)),
            base_lin_vel_w=np.zeros((90, 3)),
            base_ang_vel_w=np.zeros((90, 3)),
        )
        assert clip.duration == pytest.approx(3.0)

    def test_loader_stacked_arrays(self):
        """Stacked arrays have correct shape."""
        from plugins.controllers.wbc_lab.core.motion.motion_loader import MotionClip, MotionLoader

        # Create mock clips
        clips = []
        for i in range(3):
            T = 30 + i * 10
            clips.append(MotionClip(
                name=f"clip_{i}",
                fps=30.0,
                num_frames=T,
                body_pos_w=np.random.randn(T, 14, 3).astype(np.float32),
                body_quat_w=np.random.randn(T, 14, 4).astype(np.float32),
                body_lin_vel_w=np.random.randn(T, 14, 3).astype(np.float32),
                body_ang_vel_w=np.random.randn(T, 14, 3).astype(np.float32),
                joint_pos=np.random.randn(T, 29).astype(np.float32),
                joint_vel=np.random.randn(T, 29).astype(np.float32),
                base_pos_w=np.random.randn(T, 3).astype(np.float32),
                base_quat_w=np.random.randn(T, 4).astype(np.float32),
                base_lin_vel_w=np.random.randn(T, 3).astype(np.float32),
                base_ang_vel_w=np.random.randn(T, 3).astype(np.float32),
            ))

        # Monkey-patch loader to avoid file I/O
        loader = object.__new__(MotionLoader)
        loader.clips = clips
        loader.motion_path = Path(".")

        stacked = loader.build_stacked_arrays()
        total_T = sum(c.num_frames for c in clips)
        assert stacked["body_pos_w"].shape == (total_T, 14, 3)
        assert stacked["joint_pos"].shape == (total_T, 29)
        assert len(stacked["clip_starts"]) == 4  # 3 clips + 1


class TestRewardComputer:
    """Reward computation tests."""

    def test_tracking_rewards_shape(self):
        """Tracking rewards produce correct shapes."""
        from plugins.controllers.wbc_lab.core.envs.rewards import RewardComputer

        calc = RewardComputer(num_envs=4)

        jp_error = np.random.randn(4, 29).astype(np.float32) * 0.1
        reward = calc.tracking_joint_pos(jp_error)
        assert reward.shape == (4,)
        assert np.all(np.isfinite(reward))

    def test_tracking_exp_kernel(self):
        """Exponential kernel returns values in [0, 1]."""
        from plugins.controllers.wbc_lab.core.envs.rewards import _tracking_exp

        error = np.random.randn(10, 3).astype(np.float32) * 0.5
        result = _tracking_exp(error, sigma=0.25)
        assert result.shape == (10,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_compute_all(self):
        """compute_all returns total + breakdown."""
        from plugins.controllers.wbc_lab.core.envs.rewards import RewardComputer

        calc = RewardComputer(num_envs=2)
        rewards, terms = calc.compute_all(
            joint_pos_error=np.zeros((2, 29)),
            joint_vel_error=np.zeros((2, 29)),
            anchor_pos_error=np.zeros((2, 3)),
            anchor_ori_error=np.zeros(2),
            anchor_lin_vel_error=np.zeros((2, 3)),
            anchor_ang_vel_error=np.zeros((2, 3)),
        )
        assert rewards.shape == (2,)
        assert len(terms) >= 6
        assert "tracking_joint_pos" in terms


class TestTerminationChecker:
    """Termination condition tests."""

    def test_base_height_termination(self):
        """Low base height triggers termination."""
        from plugins.controllers.wbc_lab.core.envs.terminations import (
            TerminationChecker,
        )

        checker = TerminationChecker(num_envs=2)
        terminated, truncated, info = checker.check(
            base_pos=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]),
            base_height=np.array([0.1, 1.0]),
        )
        assert terminated[0]  # too low
        assert not terminated[1]  # fine

    def test_truncation(self):
        """Episode truncates after max steps."""
        from plugins.controllers.wbc_lab.core.envs.terminations import (
            TerminationChecker,
            TerminationConfig,
        )

        cfg = TerminationConfig(max_episode_steps=10)
        checker = TerminationChecker(cfg, num_envs=1)

        for _ in range(9):
            term, trunc, _ = checker.check(base_pos=np.array([[0, 0, 1.0]]))
            assert not trunc[0]

        term, trunc, _ = checker.check(base_pos=np.array([[0, 0, 1.0]]))
        assert trunc[0]


class TestAdaptiveRsiSampler:
    """RSI sampling tests."""

    def test_sampling_returns_valid_frame(self):
        """Sampled frame is within valid range."""
        from plugins.controllers.wbc_lab.core.motion.sampling import AdaptiveRsiSampler, RsiCfg

        cfg = RsiCfg(bin_width_s=1.0)
        sampler = AdaptiveRsiSampler(cfg, clip_duration_s=10.0, fps=30.0)

        for _ in range(100):
            frame = sampler.sample_start_frame()
            assert 0 <= frame < 300

    def test_failure_update(self):
        """Failure levels update correctly."""
        from plugins.controllers.wbc_lab.core.motion.sampling import AdaptiveRsiSampler, RsiCfg

        cfg = RsiCfg(bin_width_s=1.0, alpha=0.1)
        sampler = AdaptiveRsiSampler(cfg, clip_duration_s=5.0, fps=30.0)

        # All should start at 1.0
        assert np.allclose(sampler.failure_levels, 1.0)

        # Update bin 0 with failure
        sampler.update_failure(0, 0.0)  # success = 0 failure
        expected = 0.9 * 1.0 + 0.1 * 0.0  # EMA
        assert abs(sampler.failure_levels[0] - expected) < 1e-6

    def test_save_load_state(self):
        """State save/load roundtrips correctly."""
        from plugins.controllers.wbc_lab.core.motion.sampling import AdaptiveRsiSampler, RsiCfg

        cfg = RsiCfg(bin_width_s=2.0)
        sampler = AdaptiveRsiSampler(cfg, clip_duration_s=10.0, fps=30.0)

        sampler.update_failure(0, 0.5)
        sampler.update_failure(2, 0.3)

        state = sampler.save_state()
        sampler2 = AdaptiveRsiSampler(cfg, clip_duration_s=10.0, fps=30.0)
        sampler2.load_state(state)

        assert np.allclose(sampler.failure_levels, sampler2.failure_levels)


class TestTrackingParamsExporter:
    """Export tests."""

    def test_build_params(self):
        """Building params produces valid dict."""
        from plugins.controllers.wbc_lab.core.export.tracking_params import (
            TrackingParamsExporter,
        )
        from plugins.controllers.wbc_lab.core.robots.g1_config import G1RobotConfig

        exporter = TrackingParamsExporter(G1RobotConfig())
        params = exporter.build_tracking_params(
            kp=[100.0] * 29,
            kd=[10.0] * 29,
            action_scale=0.25,
            anchor_body="torso_link",
        )
        assert params["schema_version"] == "wbc_tracking_params_v1"
        assert "joint_pd" in params
        assert "tracking" in params


class TestMotionMirror:
    """Motion mirroring tests."""

    def test_joint_mirror(self):
        """Joint mirroring swaps L/R correctly."""
        from plugins.controllers.wbc_lab.core.utils.motion_mirror import mirror_joint_array

        data = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        pairs = ((0, 1, 1.0), (2, 3, -1.0))
        mirrored = mirror_joint_array(data, pairs)
        assert mirrored[0, 0] == 2.0
        assert mirrored[0, 1] == 1.0
        assert mirrored[0, 2] == -4.0
        assert mirrored[0, 3] == -3.0

    def test_body_mirror_y_flip(self):
        """Body mirroring flips y coordinate."""
        from plugins.controllers.wbc_lab.core.utils.motion_mirror import mirror_body_arrays

        pos = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        mirrored, _ = mirror_body_arrays(pos)
        assert mirrored[0, 0, 0] == 1.0  # x unchanged
        assert mirrored[0, 0, 1] == -2.0  # y flipped
        assert mirrored[0, 0, 2] == 3.0  # z unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
