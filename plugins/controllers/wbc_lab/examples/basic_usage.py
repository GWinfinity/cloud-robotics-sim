"""WBC-Lab Plugin - Basic usage example.

Demonstrates how to use the WBC motion tracking environment
on Genesis for Unitree G1 humanoid control.
"""

import numpy as np
from pathlib import Path

try:
    from cloud_robotics_sim.plugins.controllers.wbc_lab import (
        WbcGenesisEnv,
        G1RobotConfig,
        MotionCommandCfg,
        RewardComputer,
    )
    from cloud_robotics_sim.plugins.controllers.wbc_lab.core.envs.wbc_env import WbcEnvConfig
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.wbc_env import WbcGenesisEnv, WbcEnvConfig
    from core.robots.g1_config import G1RobotConfig
    from core.motion.motion_command import MotionCommandCfg


def example_basic_env():
    """Example 1: Basic environment (no motion tracking)."""
    print("=" * 60)
    print("Example 1: Basic WBC Environment")
    print("=" * 60)

    cfg = WbcEnvConfig(
        headless=True,
        num_envs=1,
    )

    env = WbcGenesisEnv(cfg)

    obs, info = env.reset()
    print(f"Observation dim: {obs.shape}")
    print(f"Action dim: {env.action_space_dim}")

    total_reward = 0
    for step in range(200):
        action = np.random.uniform(-0.5, 0.5, size=env.action_space_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0]

        if terminated[0] or truncated[0]:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Done\n")


def example_with_motion():
    """Example 2: Environment with motion tracking.

    Requires a motion NPZ bundle. Set motion_path to your dataset.
    """
    print("=" * 60)
    print("Example 2: WBC with Motion Tracking")
    print("=" * 60)

    motion_path = "data/g1/samples"  # Update to your path
    if not Path(motion_path).exists():
        print(f"Motion data not found at {motion_path}, skipping")
        return

    motion_cfg = MotionCommandCfg(
        motion_path=motion_path,
        anchor_body_name="torso_link",
    )

    cfg = WbcEnvConfig(
        headless=True,
        num_envs=1,
        motion_cfg=motion_cfg,
    )

    env = WbcGenesisEnv(cfg)

    obs, info = env.reset()
    print(f"Observation dim: {obs.shape}")
    print(f"Loaded {env.motion_cmd.num_clips()} clips")

    total_reward = 0
    for step in range(500):
        action = np.zeros(env.action_space_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0]

        if step % 100 == 0:
            clip_idx = info.get("clip_idx", [0])[0]
            print(f"  Step {step}: reward={reward[0]:.4f}, clip={clip_idx}")

        if terminated[0] or truncated[0]:
            obs, info = env.reset()

    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Done\n")


def example_reward_computation():
    """Example 3: Standalone reward computation (no Genesis needed)."""
    print("=" * 60)
    print("Example 3: Reward Computation")
    print("=" * 60)

    from core.envs.rewards import RewardComputer, RewardConfig

    cfg = RewardConfig()
    calc = RewardComputer(cfg, num_envs=4)

    rewards, terms = calc.compute_all(
        joint_pos_error=np.random.randn(4, 29).astype(np.float32) * 0.1,
        joint_vel_error=np.random.randn(4, 29).astype(np.float32) * 0.5,
        anchor_pos_error=np.random.randn(4, 3).astype(np.float32) * 0.01,
        anchor_ori_error=np.abs(np.random.randn(4).astype(np.float32)) * 0.1,
        anchor_lin_vel_error=np.abs(np.random.randn(4, 3).astype(np.float32)) * 0.1,
        anchor_ang_vel_error=np.abs(np.random.randn(4, 3).astype(np.float32)) * 0.1,
    )

    print(f"Total rewards: {rewards}")
    for name, val in terms.items():
        print(f"  {name}: {val}")
    print("Done\n")


def main():
    """Run all examples."""
    print("\nWBC-Lab Genesis Plugin - Examples\n")

    try:
        example_basic_env()
    except Exception as e:
        print(f"Basic env example failed: {e}\n")

    try:
        example_with_motion()
    except Exception as e:
        print(f"Motion tracking example failed: {e}\n")

    try:
        example_reward_computation()
    except Exception as e:
        print(f"Reward computation example failed: {e}\n")

    print("All examples complete!")


if __name__ == "__main__":
    main()
