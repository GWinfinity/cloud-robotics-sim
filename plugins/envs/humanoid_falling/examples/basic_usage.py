"""
Humanoid Falling Plugin - 基础使用示例

演示如何使用人形机器人跌倒保护环境
"""

import numpy as np
import yaml
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.envs.humanoid_falling import (
        HumanoidFallingEnv, FallingCurriculum, PPO
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.humanoid_env import HumanoidFallingEnv
    from core.envs.curriculum import FallingCurriculum
    from core.algorithms.ppo import PPO


def create_default_config():
    """创建默认配置"""
    return {
        'genesis': {
            'dt': 0.01,
            'substeps': 10,
        },
        'env': {
            'episode_length': 500,
            'push_force_range': [50, 500],
            'push_duration': 0.1,
        },
        'robot': {
            'mjcf_path': 'assets/humanoid/humanoid.xml',
            'init_pos': [0, 0, 1.0],
        },
        'rewards': {
            'survival': {'weight': 1.0},
            'impact': {'weight': -0.1, 'threshold': 100.0},
            'triangle_structure': {'weight': 0.5},
            'head_protection': {'weight': -0.2},
            'joint_limit': {'weight': -0.1},
            'energy': {'weight': -0.01},
            'self_collision': {'weight': -0.1},
        },
        'curriculum': {
            'enabled': True,
            'stages': [
                {'name': 'basic', 'push_force': [50, 100], 'direction_range': 0},
                {'name': 'lateral', 'push_force': [100, 200], 'direction_range': 30},
                {'name': 'multi', 'push_force': [200, 300], 'direction_range': 60},
                {'name': 'extreme', 'push_force': [300, 500], 'direction_range': 180},
            ]
        }
    }


def example_basic_env():
    """示例1: 基础环境使用"""
    print("=" * 60)
    print("示例1: 基础环境使用")
    print("=" * 60)
    
    config = create_default_config()
    
    # 创建环境
    env = HumanoidFallingEnv(
        config_path=None,
        num_envs=1,
        headless=True
    )
    env.config = config
    
    # 重置环境
    obs = env.reset()
    print(f"观测维度: {env.num_obs}")
    print(f"动作维度: {env.num_actions}")
    
    # 运行一个简短回合
    total_reward = 0
    for step in range(200):
        # 随机动作
        action = np.random.uniform(-0.5, 0.5, size=env.num_actions)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"  Step {step}: reward={reward:.3f}")
        
        if done:
            print(f"回合结束于 step {step}")
            break
    
    print(f"总奖励: {total_reward:.2f}")
    env.close()
    print("✓ 基础环境测试完成\n")


def example_push_scenarios():
    """示例2: 不同推力场景"""
    print("=" * 60)
    print("示例2: 不同推力场景")
    print("=" * 60)
    
    config = create_default_config()
    env = HumanoidFallingEnv(config, num_envs=1, headless=True)
    
    scenarios = [
        ("前方轻推", [100, 0, 0]),
        ("后方轻推", [-100, 0, 0]),
        ("左侧轻推", [0, 100, 0]),
        ("右侧轻推", [0, -100, 0]),
        ("前方重推", [300, 0, 0]),
        ("斜向推力", [200, 100, 0]),
    ]
    
    for scenario_name, force in scenarios:
        print(f"\n场景: {scenario_name}")
        print(f"  推力: {force} N")
        
        obs = env.reset()
        
        # 应用推力
        env.apply_push(force=force, duration=0.1)
        
        # 观察反应
        max_impact = 0
        for step in range(150):
            action = np.zeros(env.num_actions)
            obs, reward, done, info = env.step(action)
            
            if 'contact_force' in info:
                max_impact = max(max_impact, info['contact_force'])
            
            if done:
                print(f"  回合结束于 step {step}")
                break
        
        print(f"  最大冲击: {max_impact:.2f} N")
    
    env.close()
    print("\n✓ 推力场景测试完成\n")


def example_curriculum_learning():
    """示例3: 课程学习"""
    print("=" * 60)
    print("示例3: 课程学习")
    print("=" * 60)
    
    config = create_default_config()
    env = HumanoidFallingEnv(config, num_envs=1, headless=True)
    
    # 创建课程
    curriculum = FallingCurriculum(
        env=env,
        stages=config['curriculum']['stages']
    )
    
    print("\n课程阶段:")
    for i, stage in enumerate(config['curriculum']['stages']):
        print(f"  阶段 {i+1}: {stage['name']}")
        print(f"    推力范围: {stage['push_force']} N")
        print(f"    方向变化: ±{stage['direction_range']}°")
    
    # 模拟课程进度
    print("\n模拟课程进度:")
    for episode in [0, 100, 500, 1000, 2000]:
        stage_idx = curriculum.get_stage(episode)
        stage_config = curriculum.get_config(stage_idx)
        print(f"  Episode {episode:4d}: 阶段 {stage_idx+1} "
              f"(推力: {stage_config['push_force']})")
    
    env.close()
    print("✓ 课程学习测试完成\n")


def example_triangle_structure():
    """示例4: 三角形保护结构"""
    print("=" * 60)
    print("示例4: 三角形保护结构")
    print("=" * 60)
    
    config = create_default_config()
    env = HumanoidFallingEnv(config, num_envs=1, headless=True)
    
    print("""
三角形保护结构:
  目标: 利用双臂和躯干形成三角形支撑
  
  接触点分布:
  - 左手  → 地面接触点1
  - 右手  → 地面接触点2  
  - 躯干  → 稳定重心
  
  优势:
  - 分散冲击力到双臂
  - 保护头部不直接撞击地面
  - 减少关节扭矩
""")
    
    # 测试不同姿势的冲击
    obs = env.reset()
    env.apply_push(force=[250, 50, 0], duration=0.1)
    
    contact_history = []
    for step in range(100):
        action = np.random.uniform(-0.3, 0.3, size=env.num_actions)
        obs, reward, done, info = env.step(action)
        
        if 'contact_points' in info:
            contact_history.append(len(info['contact_points']))
        
        if done:
            break
    
    if contact_history:
        avg_contacts = np.mean(contact_history)
        print(f"平均接触点数: {avg_contacts:.2f}")
        print(f"最大接触点数: {max(contact_history)}")
    
    env.close()
    print("✓ 三角形结构测试完成\n")


def example_impact_monitoring():
    """示例5: 冲击监测"""
    print("=" * 60)
    print("示例5: 冲击监测")
    print("=" * 60)
    
    config = create_default_config()
    env = HumanoidFallingEnv(config, num_envs=1, headless=True)
    
    print("\n监测部位:")
    body_parts = ['head', 'torso', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
    for part in body_parts:
        print(f"  - {part}")
    
    obs = env.reset()
    env.apply_push(force=[300, 0, 0], duration=0.1)
    
    print("\n冲击监测记录:")
    max_forces = {part: 0 for part in body_parts}
    
    for step in range(200):
        action = np.zeros(env.num_actions)
        obs, reward, done, info = env.step(action)
        
        if 'contact_forces' in info:
            for part, force in info['contact_forces'].items():
                if part in max_forces:
                    max_forces[part] = max(max_forces[part], force)
        
        if done:
            break
    
    print("\n最大冲击力:")
    for part, force in max_forces.items():
        status = "⚠ 高冲击!" if force > 200 else "✓ 正常"
        print(f"  {part:12s}: {force:6.2f} N {status}")
    
    env.close()
    print("✓ 冲击监测测试完成\n")


def example_safety_considerations():
    """示例6: 安全考虑"""
    print("=" * 60)
    print("示例6: 安全考虑")
    print("=" * 60)
    
    print("""
安全警告与注意事项:

⚠️  仿真到现实的差距 (Sim-to-Real Gap)
   - 仿真中的策略可能不适用于真实机器人
   - 需要充分的域随机化和现实测试

⚠️  物理限制
   - 真实机器人的执行器有力和速度限制
   - 关节角度范围可能不同
   - 传感器噪声和延迟

⚠️  部署前检查清单:
   □ 在仿真中充分测试
   □ 添加硬件安全限制
   □ 使用保护装备 (头盔、护具)
   □ 从极轻的推力开始测试
   □ 有紧急停止措施
   □ 逐步增加测试强度

⚠️  推荐测试流程:
   1. 纯仿真测试 (本环境)
   2. 硬件在环测试
   3. 固定平台测试
   4. 吊索保护测试
   5. 地面测试 (从轻推力开始)
""")
    
    print("✓ 安全考虑说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Humanoid Falling Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查依赖
    try:
        import genesis as gs
        print(f"Genesis 可用")
    except ImportError:
        print("警告: Genesis 未安装，示例将以演示模式运行")
        return
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("警告: PyTorch 未安装")
        return
    
    print()
    
    # 运行示例
    examples = [
        ("基础环境", example_basic_env),
        ("推力场景", example_push_scenarios),
        ("课程学习", example_curriculum_learning),
        ("三角形结构", example_triangle_structure),
        ("冲击监测", example_impact_monitoring),
        ("安全考虑", example_safety_considerations),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"{name} 示例失败: {e}\n")
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
