"""
Sim2Real Dexterous Plugin - 基础使用示例

演示如何使用 Sim-to-Real 灵巧操作环境
"""

import numpy as np
import yaml
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.sim2real import (
        DexterousManipulationEnv, TaskType,
        Real2SimTuner, PolicyDistillation
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.dexterous_env import DexterousManipulationEnv, TaskType
    from core.algorithms.real2sim_tuning import Real2SimTuner
    from core.models.policy_distillation import PolicyDistillation


def create_default_config():
    """创建默认配置"""
    return {
        'genesis': {
            'dt': 0.02,
            'substeps': 4,
        },
        'robot': {
            'num_joints': 54,  # 双臂 + 双手 + 躯干
            'hand_dofs': 24,   # 每手12 DOF
            'arm_dofs': 14,    # 每臂7 DOF
        },
        'env': {
            'episode_length': 500,
            'observation': {
                'image_size': [224, 224],
                'use_point_cloud': True,
                'use_proprioception': True,
                'history_length': 3,
            }
        },
        'real2sim': {
            'tunable_params': ['friction', 'mass_scale', 'damping', 'kp', 'kd'],
            'optimization': {
                'method': 'bayesian',
                'iterations': 100,
            }
        },
        'distillation': {
            'temperature': 4.0,
            'alpha': 0.5,
        }
    }


def example_basic_env():
    """示例1: 基础环境使用"""
    print("=" * 60)
    print("示例1: 基础环境使用")
    print("=" * 60)
    
    config = create_default_config()
    
    # 创建环境
    env = DexterousManipulationEnv(
        config=config,
        task_name='grasp_and_reach',
        num_envs=1,
        headless=True
    )
    
    # 重置环境
    obs = env.reset()
    print(f"观测维度: {obs.shape if hasattr(obs, 'shape') else 'multi-modal'}")
    print(f"动作维度: {env.action_dim}")
    print(f"任务: {env.task_name}")
    
    # 运行一个简短回合
    total_reward = 0
    for step in range(100):
        # 随机动作
        action = np.random.uniform(-0.5, 0.5, size=env.action_dim)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward:.3f}")
        
        if done:
            print(f"回合结束于 step {step}")
            break
    
    print(f"总奖励: {total_reward:.2f}")
    env.close()
    print("✓ 基础环境测试完成\n")


def example_different_tasks():
    """示例2: 不同任务类型"""
    print("=" * 60)
    print("示例2: 不同任务类型")
    print("=" * 60)
    
    config = create_default_config()
    
    tasks = [
        ('grasp_and_reach', '抓取并伸展'),
        ('box_lift', '箱体提升'),
        ('bimanual_handover', '双手交接'),
    ]
    
    for task_id, task_name in tasks:
        print(f"\n任务: {task_id} ({task_name})")
        try:
            env = DexterousManipulationEnv(
                config=config,
                task_name=task_id,
                num_envs=1,
                headless=True
            )
            obs = env.reset()
            
            # 运行几步
            for _ in range(50):
                action = np.zeros(env.action_dim)
                obs, reward, done, info = env.step(action)
                if done:
                    break
            
            env.close()
            print(f"  ✓ {task_name} 测试通过")
        except Exception as e:
            print(f"  ⚠ {task_name} 测试失败: {e}")
    
    print("\n✓ 多任务测试完成\n")


def example_real2sim_tuning():
    """示例3: Real-to-Sim 调优"""
    print("=" * 60)
    print("示例3: Real-to-Sim 调优")
    print("=" * 60)
    
    config = create_default_config()
    env = DexterousManipulationEnv(
        config=config,
        task_name='grasp_and_reach',
        num_envs=1,
        headless=True
    )
    
    # 模拟真实世界数据
    print("\n生成模拟真实轨迹...")
    real_trajectories = []
    for _ in range(10):
        traj = []
        env.reset()
        for _ in range(50):
            action = np.random.uniform(-0.3, 0.3, size=env.action_dim)
            obs, reward, done, info = env.step(action)
            traj.append({
                'joint_pos': env.robot.get_dofs_position().cpu().numpy(),
                'joint_vel': env.robot.get_dofs_velocity().cpu().numpy(),
            })
            if done:
                break
        real_trajectories.append(traj)
    
    print(f"真实轨迹数量: {len(real_trajectories)}")
    
    # 创建调优器
    print("\n创建 Real-to-Sim 调优器...")
    tuner = Real2SimTuner(
        env=env,
        real_trajectories=real_trajectories
    )
    
    print("可调参数:")
    for param in config['real2sim']['tunable_params']:
        print(f"  - {param}")
    
    # 注意: 实际调优需要更多迭代
    print("\n调优配置:")
    print(f"  方法: {config['real2sim']['optimization']['method']}")
    print(f"  迭代: {config['real2sim']['optimization']['iterations']}")
    
    env.close()
    print("✓ Real-to-Sim 调优配置测试完成\n")


def example_policy_distillation():
    """示例4: 策略蒸馏"""
    print("=" * 60)
    print("示例4: 策略蒸馏配置")
    print("=" * 60)
    
    config = create_default_config()
    
    print("\n分而治之策略蒸馏:")
    print("  阶段1: 训练单任务专家")
    print("    - Expert 1: Grasp-and-Reach")
    print("    - Expert 2: Box Lift")
    print("    - Expert 3: Bimanual Handover")
    print("\n  阶段2: 策略蒸馏")
    print("    - 学生网络: 多任务策略")
    print("    - 蒸馏温度: {}".format(config['distillation']['temperature']))
    print("    - 蒸馏权重: {}".format(config['distillation']['alpha']))
    
    print("\n蒸馏损失:")
    print("  L_total = α * L_distill + (1-α) * L_task")
    print("  L_distill = KL(π_teacher || π_student) * T^2")
    
    print("\n✓ 策略蒸馏配置测试完成\n")


def example_hybrid_observation():
    """示例5: 混合观测"""
    print("=" * 60)
    print("示例5: 混合物体表示")
    print("=" * 60)
    
    config = create_default_config()
    
    print("\n多模态输入:")
    print("  1. 视觉 (RGB Image)")
    print(f"     尺寸: {config['env']['observation']['image_size']}")
    print("     编码器: ResNet-18")
    print("     输出: 512-dim")
    
    print("\n  2. 点云 (Point Cloud)")
    print("     编码器: PointNet")
    print("     输出: 256-dim")
    
    print("\n  3. 本体感觉 (Proprioception)")
    print("     内容: 关节位置/速度")
    print("     输出: 128-dim")
    
    print("\n融合层:")
    print("  输入: 896-dim (512+256+128)")
    print("  MLP: 896 → 512 → 256")
    print("  输出: 256-dim 混合表示")
    
    print("\n✓ 混合观测配置测试完成\n")


def example_sim2real_pipeline():
    """示例6: Sim-to-Real 完整流程"""
    print("=" * 60)
    print("示例6: Sim-to-Real 完整流程")
    print("=" * 60)
    
    print("""
Sim-to-Real 流程:

1. Real-to-Sim 调优
   └── 收集真实世界轨迹
   └── 优化仿真参数 (摩擦、质量、阻尼)
   └── 验证仿真与真实匹配度

2. 单任务专家训练
   └── 在调优后的仿真中训练
   └── 每个任务训练一个专家策略
   └── 保存专家检查点

3. 策略蒸馏
   └── 加载所有专家策略
   └── 训练多任务学生策略
   └── 使用蒸馏损失保持性能

4. 域随机化
   └── 随机化视觉 (亮度、对比度)
   └── 随机化物理 (摩擦、质量)
   └── 随机化动力学 (KP、KD)

5. 部署到真实机器人
   └── 加载蒸馏后的策略
   └── Zero-shot 部署
   └── 可选: Few-shot 在线适应
""")
    
    print("✓ Sim-to-Real 流程说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Sim2Real Dexterous Plugin - 使用示例")
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
        ("多任务", example_different_tasks),
        ("Real-to-Sim调优", example_real2sim_tuning),
        ("策略蒸馏", example_policy_distillation),
        ("混合观测", example_hybrid_observation),
        ("Sim-to-Real流程", example_sim2real_pipeline),
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
