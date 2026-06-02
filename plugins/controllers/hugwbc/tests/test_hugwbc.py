"""Tests for HugWBC Controller Plugin."""

import numpy as np
import pytest

from plugins.controllers.hugwbc.core.envs.hugwbc_env import HugWBCEnv, TaskType
from plugins.controllers.hugwbc.core.utils.rewards import RewardComputer


class TestTaskType:
    """TaskType enum tests."""

    def test_values(self):
        """Enum values correct."""
        assert TaskType.LOCO.value == "h1_loco"
        assert TaskType.STAIRS.value == "h1_stairs"


class TestHugWBCEnv:
    """HugWBCEnv unit tests."""

    def test_env_init(self):
        """Environment initializes without crash."""
        env = HugWBCEnv()
        assert env is not None


class TestRewardComputer:
    """Reward calculation tests."""

    def test_init(self):
        """RewardComputer initializes."""
        config = {
            'rewards': {
                'tracking_lin_vel': {'weight': 1.0, 'sigma': 0.25},
                'tracking_ang_vel': {'weight': 1.0, 'sigma': 0.25},
                'lin_vel_z': {'weight': -1.0},
                'ang_vel_xy': {'weight': -1.0},
                'orientation': {'weight': -1.0},
                'dof_acc': {'weight': -1.0},
                'action_rate': {'weight': -1.0},
                'feet_air_time': {'weight': 1.0},
                'termination': {'weight': -1.0},
                'collision': {'weight': -1.0},
            }
        }
        calc = RewardComputer(config=config, num_envs=1)
        assert calc is not None

    def test_compute_rewards(self):
        """Can compute rewards."""
        config = {
            'rewards': {
                'tracking_lin_vel': {'weight': 1.0, 'sigma': 0.25},
                'tracking_ang_vel': {'weight': 1.0, 'sigma': 0.25},
                'lin_vel_z': {'weight': -1.0},
                'ang_vel_xy': {'weight': -1.0},
                'orientation': {'weight': -1.0},
                'dof_acc': {'weight': -1.0},
                'action_rate': {'weight': -1.0},
                'feet_air_time': {'weight': 1.0},
                'termination': {'weight': -1.0},
                'collision': {'weight': -1.0},
            }
        }
        calc = RewardComputer(config=config, num_envs=1)
        rewards, reward_dict = calc.compute_rewards(
            base_lin_vel=np.zeros((1, 3)),
            base_ang_vel=np.zeros((1, 3)),
            projected_gravity=np.array([[0, 0, -1]]),
            joint_pos=np.zeros((1, 19)),
            joint_vel=np.zeros((1, 19)),
            commands=np.zeros((1, 3)),
            actions=np.zeros((1, 19)),
            last_actions=np.zeros((1, 19)),
            feet_contact_forces=np.zeros((1, 4)),
            feet_air_time=np.zeros((1, 4)),
        )
        assert isinstance(rewards, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
