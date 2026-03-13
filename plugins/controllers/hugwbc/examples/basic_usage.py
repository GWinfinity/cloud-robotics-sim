"""
HugWBC Plugin - 基础使用示例

演示如何使用 HugWBC 进行人形机器人全身控制
"""

import numpy as np
import yaml
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.controllers.hugwbc import (
        HugWBCEnv, TaskType, PPO
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.hugwbc_env import HugWBCEnv, TaskType
    from core.algorithms.ppo import PPO


def create_default_config():
    """创建默认配置"""
    return {
        'task': 'h1_loco',
        'robot': {
            'mjcf_path': 'assets/h1/h1.xml',
            'init_pos': [0, 0, 1.0],
        },
        'env': {
            'episode_length': 1000,
            'gait_frequency': 1.25,
            'command_ranges': {
                'lin_vel_x': [-1.0, 2.0],
                'lin_vel_y': [-0.5, 0.5],
                'ang_vel_yaw': [-1.0, 1.0],
            }
        },
        'rewards': {
            'tracking_lin_vel': {'weight': 2.0, 'sigma': 0.25},
            'tracking_ang_vel': {'weight': 0.5, 'sigma': 0.25},
            'lin_vel_z': {'weight': -2.0},
            'orientation': {'weight': -1.0},
            'feet_air_time': {'weight': 1.0},
            'collision': {'weight': -1.0},
            'action_rate': {'weight': -0.01},
            'torque': {'weight': -0.0001},
        },
        'domain_rand': {
            'push_robots': True,
            'push_interval_s': 15.0,
            'push_vel_xy': 1.0,
            'randomize_friction': True,
            'friction_range': [0.2, 1.25],
        }
    }


def example_basic_env():
    """示例1: 基础环境使用"""
    print("=" * 60)
    print("示例1: 基础环境使用")
    print("=" * 60)
    
    config = create_default_config()
    
    # 创建环境
    env = HugWBCEnv(
        task="h1_loco",
        config_path=None,
        num_envs=1,
        headless=True,
        device='cuda'
    )
    env.config = config  # 使用默认配置
    
    # 重置环境
    obs = env.reset()
    print(f"观测维度: {obs.shape}")
    print(f"动作维度: {env.num_actions}")
    
    # 设置速度命令: 前进 1.0 m/s
    env.commands = np.array([1.0, 0.0, 0.0])
    print(f"速度命令: vx=1.0, vy=0.0, yaw_rate=0.0")
    
    # 运行一个简短回合
    total_reward = 0
    for step in range(200):
        # 随机动作
        action = np.random.uniform(-0.5, 0.5, size=env.num_actions)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"回合结束于 step {step}")
            break
    
    print(f"总奖励: {total_reward:.2f}")
    print("✓ 基础环境测试完成\n")


def example_different_tasks():
    """示例2: 不同任务类型"""
    print("=" * 60)
    print("示例2: 不同任务类型")
    print("=" * 60)
    
    tasks = [
        ("h1_loco", "平地行走"),
        ("h1_stairs", "上下楼梯"),
        ("h1_terrain", "复杂地形"),
    ]
    
    for task_id, task_name in tasks:
        print(f"\n任务: {task_id} ({task_name})")
        try:
            env = HugWBCEnv(task=task_id, num_envs=1, headless=True)
            obs = env.reset()
            
            # 设置不同的速度命令
            if task_id == "h1_loco":
                env.commands = np.array([1.0, 0.0, 0.0])  # 前进
            elif task_id == "h1_stairs":
                env.commands = np.array([0.5, 0.0, 0.0])  # 慢速前进
            else:
                env.commands = np.array([0.8, 0.2, 0.1])  # 综合运动
            
            # 运行几步
            for _ in range(50):
                action = np.zeros(env.num_actions)
                obs, reward, done, info = env.step(action)
                if done:
                    break
            
            env.close()
            print(f"  ✓ {task_name} 测试通过")
        except Exception as e:
            print(f"  ⚠ {task_name} 测试失败: {e}")
    
    print("\n✓ 多任务测试完成\n")


def example_command_tracking():
    """示例3: 命令跟踪测试"""
    print("=" * 60)
    print("示例3: 命令跟踪")
    print("=" * 60)
    
    env = HugWBCEnv(task="h1_loco", num_envs=1, headless=True)
    obs = env.reset()
    
    # 测试不同的速度命令
    commands = [
        ("前进", [1.0, 0.0, 0.0]),
        ("后退", [-0.5, 0.0, 0.0]),
        ("左移", [0.0, 0.3, 0.0]),
        ("右移", [0.0, -0.3, 0.0]),
        ("左转", [0.0, 0.0, 0.5]),
        ("右转", [0.0, 0.0, -0.5]),
        ("综合", [0.8, 0.2, 0.3]),
    ]
    
    for cmd_name, cmd_values in commands:
        env.commands = np.array(cmd_values)
        print(f"\n命令: {cmd_name} ({cmd_values})")
        
        # 运行一段时间
        for step in range(100):
            action = np.random.uniform(-0.3, 0.3, size=env.num_actions)
            obs, reward, done, info = env.step(action)
            
            if done:
                obs = env.reset()
                break
    
    env.close()
    print("\n✓ 命令跟踪测试完成\n")


def example_gait_phases():
    """示例4: 步态相位"""
    print("=" * 60)
    print("示例4: 步态相位控制")
    print("=" * 60)
    
    env = HugWBCEnv(task="h1_loco", num_envs=1, headless=True)
    obs = env.reset()
    env.commands = np.array([1.0, 0.0, 0.0])
    
    # 不同的步态频率
    frequencies = [1.0, 1.25, 1.5, 2.0]
    
    for freq in frequencies:
        env.gait_frequency = freq
        print(f"\n步态频率: {freq} Hz")
        
        # 运行几步观察步态
        phase_history = []
        for step in range(100):
            action = np.zeros(env.num_actions)
            obs, reward, done, info = env.step(action)
            
            # 记录相位
            phase = env.gait_phase[0]
            phase_history.append(phase)
            
            if done:
                break
        
        print(f"  相位范围: [{min(phase_history):.2f}, {max(phase_history):.2f}]")
    
    env.close()
    print("\n✓ 步态相位测试完成\n")


def example_heuristic_controller():
    """示例5: 启发式控制器"""
    print("=" * 60)
    print("示例5: 启发式PD控制器")
    print("=" * 60)
    
    env = HugWBCEnv(task="h1_loco", num_envs=1, headless=True)
    obs = env.reset()
    env.commands = np.array([1.0, 0.0, 0.0])
    
    # 简单的PD控制参数
    kp = 1.0
    kd = 0.1
    
    last_error = np.zeros(env.num_actions)
    
    for step in range(500):
        # 获取当前关节位置
        joint_pos = env.robot.get_dofs_position().cpu().numpy()
        joint_vel = env.robot.get_dofs_velocity().cpu().numpy()
        
        # 目标位置 (默认姿势 + 简单正弦摆动)
        target_pos = env.default_joint_pos + 0.1 * np.sin(step * 0.1)
        
        # PD控制
        error = target_pos - joint_pos
        error_deriv = (error - last_error) / 0.02  # dt = 0.02
        action = kp * error + kd * error_deriv
        
        last_error = error
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        if step % 100 == 0:
            print(f"  Step {step}: reward={reward:.3f}")
        
        if done:
            print(f"  回合结束于 step {step}")
            break
    
    env.close()
    print("✓ 启发式控制器测试完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("HugWBC Plugin - 使用示例")
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
    try:
        example_basic_env()
    except Exception as e:
        print(f"基础环境示例失败: {e}\n")
    
    try:
        example_different_tasks()
    except Exception as e:
        print(f"多任务示例失败: {e}\n")
    
    try:
        example_command_tracking()
    except Exception as e:
        print(f"命令跟踪示例失败: {e}\n")
    
    try:
        example_gait_phases()
    except Exception as e:
        print(f"步态相位示例失败: {e}\n")
    
    try:
        example_heuristic_controller()
    except Exception as e:
        print(f"启发式控制器示例失败: {e}\n")
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
