"""Tests for the real vectorized task (FrankaPickCubeVecTask).

Pure-function tests (reset-region mapping, reward) run everywhere. The real
Genesis smoke test is CPU-only, follows the one-``gs.init``-per-process
convention of the other Genesis tests, and is skipped without genesis-world.
"""

from __future__ import annotations

import pytest
import torch

from cloud_robotics_sim.core.vec_tasks import (
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    SUCCESS_THRESHOLD,
    FrankaPickCubeVecTask,
    cube_xy_from_uniform,
    pick_cube_reward,
)
from tests.optional_deps import genesis_only


class TestCubeXYFromUniform:
    """Reset-distribution mapping is pure and deterministic."""

    def test_maps_unit_square_to_init_region(self):
        u = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        xy = cube_xy_from_uniform(u)
        assert xy.shape == (3, 2)
        assert xy[0, 0] == pytest.approx(0.4)
        assert xy[0, 1] == pytest.approx(-0.2)
        assert xy[1, 0] == pytest.approx(0.6)
        assert xy[1, 1] == pytest.approx(0.2)
        assert xy[2, 0] == pytest.approx(0.5)
        assert xy[2, 1] == pytest.approx(0.0)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            cube_xy_from_uniform(torch.rand(4))
        with pytest.raises(ValueError):
            cube_xy_from_uniform(torch.rand(2, 3))

    def test_output_stays_in_region_for_random_uniforms(self):
        xy = cube_xy_from_uniform(torch.rand(512, 2))
        assert xy[:, 0].min() >= 0.4 and xy[:, 0].max() <= 0.6
        assert xy[:, 1].min() >= -0.2 and xy[:, 1].max() <= 0.2


class TestPickCubeReward:
    """Reward shaping and success mask."""

    def test_distance_reward_and_success(self):
        target = torch.tensor([0.5, 0.0, 0.1])
        cube_pos = torch.tensor(
            [
                [0.5, 0.0, 0.1],  # exactly at target -> success
                [0.5, 0.0, 0.2],  # 0.1 m away
            ]
        )
        reward, success = pick_cube_reward(cube_pos, target)
        assert reward.shape == (2,)
        assert success.tolist() == [True, False]
        assert reward[0] == pytest.approx(0.0)
        assert reward[1] == pytest.approx(-0.1)

    def test_success_threshold_boundary(self):
        target = torch.zeros(1, 3)
        eps = SUCCESS_THRESHOLD - 1e-6
        _reward, success = pick_cube_reward(torch.tensor([[eps, 0.0, 0.0]]), target)
        assert bool(success.item())


class TestTaskContract:
    """Static contract of the task (no Genesis needed)."""

    def test_dims(self):
        assert NUM_OBSERVATIONS == 24
        assert NUM_ACTIONS == 9
        assert FrankaPickCubeVecTask.num_observations == 24
        assert FrankaPickCubeVecTask.num_actions == 9

    def test_reset_before_setup_raises(self):
        task = FrankaPickCubeVecTask()
        with pytest.raises(RuntimeError):
            task.reset()


@pytest.mark.slow
@genesis_only
class TestGenesisCpuSmoke:
    """Real batched scene on CPU: build, reset, step, obs validity.

    ``gs.init`` runs once per process (module-level), matching the convention
    of the other Genesis tests; the scene itself is created per test.
    """

    @pytest.fixture(scope="class")
    def vec_env(self):
        from cloud_robotics_sim.core.vectorized import (
            GenesisVectorizedEnv,
            VecEnvConfig,
        )

        config = VecEnvConfig(num_envs=2, use_cuda=False)
        env = GenesisVectorizedEnv(config=config, task=FrankaPickCubeVecTask())
        env.initialize()
        yield env
        env.close()

    def test_reset_returns_finite_obs(self, vec_env):
        obs, infos = vec_env.reset()
        assert obs.shape == (2, NUM_OBSERVATIONS)
        assert torch.isfinite(obs).all()
        assert len(infos) == 2
        assert "seed" in infos[0]

    def test_step_shapes_and_finiteness(self, vec_env):
        vec_env.reset()
        actions = torch.zeros(2, NUM_ACTIONS)
        for _ in range(10):
            obs, reward, terminated, truncated, infos = vec_env.step(actions)
            assert obs.shape == (2, NUM_OBSERVATIONS)
            assert torch.isfinite(obs).all()
            assert reward.shape == (2,)
            assert torch.isfinite(reward).all()
            assert terminated.shape == (2,)
            assert truncated.shape == (2,)
            assert terminated.dtype == torch.bool
            assert truncated.dtype == torch.bool
            assert len(infos) == 2

    def test_partial_reset_subset(self, vec_env):
        vec_env.reset()
        obs = vec_env.reset_idx(torch.tensor([1]))
        assert obs.shape == (1, NUM_OBSERVATIONS)
        assert torch.isfinite(obs).all()

    def test_random_actions_do_not_nan(self, vec_env):
        vec_env.reset()
        gen = torch.Generator().manual_seed(0)
        for _ in range(20):
            actions = torch.randn(2, NUM_ACTIONS, generator=gen)
            obs, reward, terminated, truncated, _ = vec_env.step(actions)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
