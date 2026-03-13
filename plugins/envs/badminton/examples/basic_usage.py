"""
Basic Usage Example: Badminton Environment

演示如何使用羽毛球环境进行三阶段课程学习
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from cloud_robotics_sim.plugins.envs.badminton import BadmintonEnv, EKFPredictor


def demo_basic_usage():
    """演示基本使用"""
    print("=" * 60)
    print("Badminton Environment - Basic Usage Demo")
    print("=" * 60)
    
    # 创建环境 (Stage 1: 步法训练)
    print("\n1. Creating environment (Stage 1 - Footwork)...")
    env = BadmintonEnv(
        config_path=None,  # 使用默认配置
        num_envs=1,
        curriculum_stage=1,
        headless=True
    )
    
    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim: {env.num_actions}")
    
    # 重置环境
    print("\n2. Resetting environment...")
    obs = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    
    # 运行几个步骤
    print("\n3. Running simulation...")
    total_reward = 0
    
    for step in range(100):
        # 随机动作
        action = np.random.randn(env.num_actions) * 0.5
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward:.3f}, done={done}")
        
        if done:
            print(f"  Episode finished at step {step}")
            break
    
    print(f"\n  Total reward: {total_reward:.3f}")
    
    # 关闭环境
    env.close()
    print("\n4. Environment closed.")


def demo_curriculum_training():
    """演示三阶段课程学习"""
    print("\n" + "=" * 60)
    print("Three-Stage Curriculum Learning Demo")
    print("=" * 60)
    
    stages = [
        (1, "Footwork", 50),      # Stage 1: 步法
        (2, "Swing", 50),         # Stage 2: 挥拍
        (3, "Full Body", 50)      # Stage 3: 全身
    ]
    
    env = BadmintonEnv(num_envs=1, headless=True)
    
    for stage_num, stage_name, num_episodes in stages:
        print(f"\n--- Stage {stage_num}: {stage_name} ---")
        
        # 设置课程阶段
        env.set_curriculum_stage(stage_num)
        print(f"  Frozen joints: {len(env.frozen_joints)}")
        print(f"  Action dim: {env.num_actions}")
        
        # 训练几个回合
        total_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(200):
                action = np.random.randn(env.num_actions) * 0.3
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"  Average reward over {num_episodes} episodes: {avg_reward:.3f}")
    
    env.close()
    print("\nCurriculum training demo complete!")


def demo_ekf_prediction():
    """演示 EKF 预测"""
    print("\n" + "=" * 60)
    print("EKF Trajectory Prediction Demo")
    print("=" * 60)
    
    # 创建预测器
    predictor = EKFPredictor(dt=0.01)
    
    # 模拟球轨迹
    print("\n1. Simulating ball trajectory...")
    ball_positions = []
    
    # 初始状态
    ball_pos = np.array([3.0, 0.0, 2.0])  # 对方发球
    ball_vel = np.array([-8.0, 1.0, -2.0])  # 朝己方飞来
    
    for t in range(50):
        # 更新 EKF
        measurement = ball_pos + np.random.randn(3) * 0.05  # 添加噪声
        predictor.update(measurement)
        
        # 预测未来
        pred_pos, pred_vel = predictor.predict(steps=10)
        
        # 预测落点
        landing = predictor.predict_landing(court_height=0)
        
        ball_positions.append(ball_pos.copy())
        
        # 模拟物理 (简化)
        ball_vel[2] -= 9.81 * 0.01  # 重力
        ball_pos += ball_vel * 0.01
        
        if t % 10 == 0:
            print(f"  t={t*0.01:.2f}s: pos={ball_pos}, predicted_landing={landing}")
        
        if ball_pos[2] < 0:
            break
    
    print(f"\n  Total trajectory points: {len(ball_positions)}")
    print(f"  Final ball position: {ball_pos}")


def demo_custom_rewards():
    """演示自定义奖励函数"""
    print("\n" + "=" * 60)
    print("Custom Rewards Demo")
    print("=" * 60)
    
    from cloud_robotics_sim.plugins.envs.badminton import BadmintonRewards
    
    # 创建奖励计算器
    rewards = BadmintonRewards(config={
        'hit': {'weight': 10.0},
        'landing': {'weight': 5.0},
        'position': {'weight': 2.0}
    })
    
    # 模拟状态
    state = {
        'hit': True,
        'hit_speed': 15.0,
        'landing_position': np.array([2.5, 0.5]),
        'target_landing': np.array([3.0, 0.0]),
        'robot_position': np.array([-2.0, 0.5, 1.0]),
        'ball_position': np.array([-1.5, 0.3, 1.5])
    }
    
    action = np.random.randn(12) * 0.5
    
    print("\n1. Computing rewards...")
    reward_dict = rewards.compute(state, action)
    
    print(f"  Total reward: {reward_dict['total']:.3f}")
    print("  Components:")
    for key, value in reward_dict.items():
        if key != 'total':
            print(f"    {key}: {value:.3f}")


if __name__ == '__main__':
    # 运行所有演示
    demo_basic_usage()
    demo_curriculum_training()
    demo_ekf_prediction()
    demo_custom_rewards()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
