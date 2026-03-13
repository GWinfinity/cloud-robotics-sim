"""
Table Tennis Plugin - 基础使用示例

演示如何使用 table_tennis 环境进行训练和评估
"""

import numpy as np
import yaml
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.envs.table_tennis import (
        TableTennisEnv, PPO, DualPredictor, UnifiedPolicy
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.table_tennis_env import TableTennisEnv
    from core.algorithms.ppo import PPO


def create_default_config():
    """创建默认配置"""
    return {
        'genesis': {
            'dt': 0.02,
            'substeps': 4,
        },
        'env': {
            'episode_length': 1000,
            'action': {
                'delta_scale': 0.1,
            },
            'serve': {
                'velocity_range': [3.0, 6.0],
                'spin_range': [-10.0, 10.0],
            }
        },
        'robot': {
            'init_pos': [0.0, 0.0, 0.75],
        },
        'ball': {
            'mass': 0.0027,
            'radius': 0.02,
            'drag_coeff': 0.47,
            'restitution': 0.85,
            'spin_decay': 0.99,
        },
        'table': {
            'length': 2.74,
            'width': 1.525,
            'height': 0.76,
        },
        'rewards': {
            'hit': {
                'weight': 10.0,
                'bonus_speed': 2.0,
            },
            'predictive': {
                'weight': 5.0,
            },
            'footwork': {
                'weight': 2.0,
            },
            'posture': {
                'weight': 1.0,
                'upright_bonus': 1.0,
            },
            'energy_penalty': -0.01,
        },
    }


def example_basic_env():
    """示例1: 基础环境使用"""
    print("=" * 60)
    print("示例1: 基础环境使用")
    print("=" * 60)
    
    config = create_default_config()
    
    # 创建环境
    env = TableTennisEnv(
        config=config,
        num_envs=1,
        headless=True,  # 无GUI模式
        device='cuda'
    )
    
    # 重置环境
    obs = env.reset()
    print(f"观测维度: {obs.shape}")
    print(f"动作维度: {env.action_dim}")
    
    # 运行一个简短回合
    total_reward = 0
    for step in range(100):
        # 随机动作
        action = np.random.uniform(-1, 1, size=env.action_dim)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"回合结束于 step {step}")
            print(f"统计: {info['stats']}")
            break
    
    print(f"总奖励: {total_reward:.2f}")
    env.close()
    print("✓ 基础环境测试完成\n")


def example_heuristic_policy():
    """示例2: 启发式策略测试"""
    print("=" * 60)
    print("示例2: 启发式策略")
    print("=" * 60)
    
    config = create_default_config()
    env = TableTennisEnv(config, num_envs=1, headless=True)
    
    obs = env.reset()
    
    for step in range(200):
        # 简单启发式: 向球的方向移动
        ball_pos = env.ball.position
        robot_pos = env.robot.get_pos().cpu().numpy()
        
        # 计算到球的方向
        direction = ball_pos - robot_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # 构造动作 (简化的全身控制)
        action = np.zeros(env.action_dim)
        action[0:3] = direction * 0.5  # 躯干移动
        
        obs, reward, done, info = env.step(action)
        
        if info['hit']:
            print(f"Step {step}: 击球! 速度: {info['hit_speed']:.2f}")
        
        if done:
            break
    
    env.close()
    print("✓ 启发式策略测试完成\n")


def example_train_ppo():
    """示例3: PPO训练 (简化版)"""
    print("=" * 60)
    print("示例3: PPO训练配置")
    print("=" * 60)
    
    config = create_default_config()
    
    # 添加PPO配置
    config['ppo'] = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }
    
    # 创建环境和算法
    env = TableTennisEnv(config, num_envs=1, headless=True)
    ppo = PPO(env, config)
    
    print("PPO配置:")
    for k, v in config['ppo'].items():
        print(f"  {k}: {v}")
    
    # 注意: 实际训练需要更长时间
    # ppo.train(total_timesteps=10_000_000)
    
    env.close()
    print("✓ PPO配置测试完成\n")


def example_vectorized_env():
    """示例4: 并行环境"""
    print("=" * 60)
    print("示例4: 并行环境 (Vectorized)")
    print("=" * 60)
    
    config = create_default_config()
    num_envs = 4
    
    env = TableTennisEnv(config, num_envs=num_envs, headless=True)
    obs = env.reset()
    
    print(f"并行环境数: {num_envs}")
    print(f"观测形状: {obs.shape}")
    
    # 并行执行
    actions = [np.random.uniform(-1, 1, env.action_dim) for _ in range(num_envs)]
    
    # 注意: 当前环境需要逐个执行，未来支持真正的向量化
    for i in range(num_envs):
        obs_single, reward, done, info = env.step(actions[i])
        print(f"  Env {i}: reward={reward:.3f}, done={done}")
    
    env.close()
    print("✓ 并行环境测试完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Table Tennis Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查Genesis是否可用
    try:
        import genesis as gs
        print(f"Genesis 版本: {gs.__version__}")
    except ImportError:
        print("警告: Genesis 未安装，示例将以演示模式运行")
        return
    
    # 运行示例
    example_basic_env()
    example_heuristic_policy()
    example_train_ppo()
    example_vectorized_env()
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
