"""Tests for Badminton Environment Plugin."""

import numpy as np
import pytest

from plugins.envs.badminton.core.curriculum import ThreeStageCurriculum
from plugins.envs.badminton.core.ekf import ShuttlecockEKF
from plugins.envs.badminton.core.rewards import compute_hit_reward, compute_landing_reward, compute_footwork_reward


class TestShuttlecockEKF:
    """EKF predictor tests."""

    def test_init(self):
        """EKF initializes."""
        ekf = ShuttlecockEKF()
        assert ekf is not None

    def test_predict(self):
        """Prediction produces state estimate."""
        ekf = ShuttlecockEKF()
        ekf.reset(initial_position=np.array([0, 0, 3]))
        ekf.predict()
        assert len(ekf.state) >= 6  # at least pos + vel

    def test_update(self):
        """Update with measurement."""
        ekf = ShuttlecockEKF()
        ekf.reset(initial_position=np.array([0, 0, 3]))
        measurement = np.array([1.0, 0.0, 2.0])
        ekf.update(measurement)
        assert len(ekf.state) >= 6


class TestThreeStageCurriculum:
    """Curriculum tests."""

    def test_stage_values(self):
        """Stage values."""
        curriculum = ThreeStageCurriculum()
        assert curriculum.current_stage == 1


class TestRewards:
    """Reward tests."""

    def test_hit_reward(self):
        """Hit reward is positive."""
        reward = compute_hit_reward(hit=True, hit_speed=15.0)
        assert reward >= 0

    def test_land_reward(self):
        """Landing in opponent court is positive."""
        reward = compute_landing_reward(
            landing_pos=np.array([2.0, 0.0, 0.0]),
            ideal_landing=np.array([2.0, 0.0, 0.0]),
            court_bounds={'x_min': -5, 'x_max': 5, 'y_min': -3, 'y_max': 3}
        )
        assert reward >= 0

    def test_footwork_reward(self):
        """Footwork reward computes."""
        reward = compute_footwork_reward(
            robot_pos=np.array([0.0, 0.0]),
            target_pos=np.array([1.0, 0.0]),
            robot_vel=np.array([0.5, 0.0])
        )
        assert isinstance(reward, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
