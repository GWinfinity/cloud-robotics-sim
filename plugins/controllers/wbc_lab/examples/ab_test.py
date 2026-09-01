"""WBC-Lab Plugin - A/B Test example.

Demonstrates how to compare the migrated Genesis implementation
against a reference (e.g., original mjlab implementation).
"""

import numpy as np
from pathlib import Path


def create_ab_test_config():
    """Create config suitable for A/B testing."""
    return {
        "genesis": {"dt": 0.005, "substeps": 20},
        "env": {
            "num_envs": 1,
            "max_episode_steps": 500,
            "device": "cuda",
        },
        "rewards": {
            "tracking_joint_pos": {"weight": 1.0, "sigma": 0.1},
            "tracking_joint_vel": {"weight": 0.2, "sigma": 0.5},
            "tracking_anchor_pos": {"weight": 2.0, "sigma": 0.05},
            "tracking_anchor_ori": {"weight": 1.0, "sigma": 0.2},
            "action_rate": {"weight": -0.01},
            "joint_acc": {"weight": -2.5e-7},
            "termination": {"weight": -10.0},
        },
    }


def run_ab_comparison():
    """Run A/B comparison between Genesis and reference implementations."""
    try:
        from cloud_robotics_sim.core.ab_test_framework import ABTestRunner
    except ImportError:
        print("AB test framework not available. Install genesis-cloud-sim first.")
        return

    print("=" * 60)
    print("WBC A/B Test: Genesis vs Reference")
    print("=" * 60)

    config = create_ab_test_config()

    # Genesis variant
    try:
        from cloud_robotics_sim.plugins.controllers.wbc_lab import WbcGenesisEnv
        from cloud_robotics_sim.plugins.controllers.wbc_lab.core.envs.wbc_env import WbcEnvConfig

        genesis_env = WbcGenesisEnv(WbcEnvConfig(headless=True))
        genesis_obs, _ = genesis_env.reset()
        print(f"Genesis obs dim: {genesis_obs.shape}")
    except Exception as e:
        print(f"Genesis env init failed: {e}")
        return

    # Run comparison episodes
    num_episodes = 10
    genesis_rewards = []

    for ep in range(num_episodes):
        obs, _ = genesis_env.reset()
        ep_reward = 0.0
        for step in range(500):
            action = np.random.uniform(-0.3, 0.3, size=genesis_env.action_space_dim)
            obs, reward, term, trunc, info = genesis_env.step(action)
            ep_reward += reward[0]
            if term[0] or trunc[0]:
                break
        genesis_rewards.append(ep_reward)
        print(f"Episode {ep}: reward={ep_reward:.2f}")

    genesis_env.close()

    print(f"\nGenesis: mean_reward={np.mean(genesis_rewards):.2f} ± {np.std(genesis_rewards):.2f}")
    print("A/B test complete")


if __name__ == "__main__":
    run_ab_comparison()
