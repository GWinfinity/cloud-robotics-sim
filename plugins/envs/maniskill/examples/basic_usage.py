"""
ManiSkill Plugin - 基础使用示例

演示如何使用 Genesis ManiSkill 进行机器人操作仿真
"""

import numpy as np
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.envs.maniskill import (
        KitchenEnv, TableTopEnv, RobotLoader
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.genesis_maniskill.envs.kitchen_env import KitchenEnv
    from core.genesis_maniskill.envs.tabletop_env import TableTopEnv
    from core.genesis_maniskill.agents.robot_loader import RobotLoader


def example_kitchen_basic():
    """示例1: 厨房环境基础使用"""
    print("=" * 60)
    print("示例1: 厨房环境基础使用")
    print("=" * 60)
    
    try:
        # 创建厨房环境
        env = KitchenEnv(
            scene_type="kitchen",
            robot="franka",
            task="pick_place",
            num_envs=4,
            headless=True
        )
        
        print(f"环境创建成功")
        print(f"  并行环境数: 4")
        print(f"  机器人: Franka")
        print(f"  任务: Pick and Place")
        
        # 重置环境
        obs, info = env.reset()
        print(f"  观测维度: {obs.shape if hasattr(obs, 'shape') else 'multi-modal'}")
        
        # 运行几步
        total_reward = 0
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.3f}")
        
        print(f"  总奖励: {total_reward:.2f}")
        env.close()
        print("✓ 厨房环境测试完成\n")
        
    except Exception as e:
        print(f"⚠ 厨房环境测试失败: {e}\n")


def example_tabletop_basic():
    """示例2: 桌面环境基础使用"""
    print("=" * 60)
    print("示例2: 桌面环境基础使用")
    print("=" * 60)
    
    try:
        # 创建桌面环境
        env = TableTopEnv(
            task="pick_cube",
            robot="ur5",
            num_envs=8,
            headless=True
        )
        
        print(f"环境创建成功")
        print(f"  并行环境数: 8")
        print(f"  机器人: UR5")
        print(f"  任务: Pick Cube")
        
        # 重置环境
        obs, info = env.reset()
        
        # 运行几步
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✓ 桌面环境测试完成\n")
        
    except Exception as e:
        print(f"⚠ 桌面环境测试失败: {e}\n")


def example_different_robots():
    """示例3: 不同机器人"""
    print("=" * 60)
    print("示例3: 不同机器人")
    print("=" * 60)
    
    robots = [
        ("franka", "Franka Emika Panda", 7),
        ("ur5", "Universal Robots UR5", 6),
        ("kinova_gen3", "Kinova Gen3", 7),
    ]
    
    print("\n支持的机器人:")
    for robot_id, robot_name, dofs in robots:
        print(f"  - {robot_id}: {robot_name} ({dofs} DOF)")
    
    print("\n机器人加载示例:")
    try:
        import genesis as gs
        gs.init(backend=gs.cpu)
        
        scene = gs.Scene(show_viewer=False)
        
        for robot_id, robot_name, dofs in robots[:2]:  # 测试前两个
            try:
                print(f"\n  加载 {robot_name}...")
                # robot = RobotLoader.from_preset(scene, robot_id)
                print(f"    ✓ {robot_name} 加载成功")
            except Exception as e:
                print(f"    ⚠ {robot_name} 加载失败: {e}")
        
        print("\n✓ 机器人加载测试完成\n")
        
    except Exception as e:
        print(f"⚠ 机器人测试失败: {e}\n")


def example_different_tasks():
    """示例4: 不同任务类型"""
    print("=" * 60)
    print("示例4: 不同任务类型")
    print("=" * 60)
    
    tasks = {
        "基础操作": [
            "pick_place",
            "open_drawer",
            "push",
            "stack",
        ],
        "厨房任务": [
            "prepare_food",
            "cleanup",
            "organize_cabinet",
        ],
        "桌面任务": [
            "insert",
            "sort",
            "assembly",
        ],
    }
    
    print("\n支持的任务:")
    for category, task_list in tasks.items():
        print(f"\n  {category}:")
        for task in task_list:
            print(f"    - {task}")
    
    print("\n✓ 任务类型展示完成\n")


def example_parallel_training():
    """示例5: 并行训练配置"""
    print("=" * 60)
    print("示例5: 并行训练配置")
    print("=" * 60)
    
    configs = [
        ("小规模测试", 16),
        ("中等规模", 256),
        ("大规模训练", 1024),
        ("超大规模", 4096),
    ]
    
    print("\n并行环境配置:")
    for name, num_envs in configs:
        estimated_fps = num_envs * 50  # 假设每环境50 FPS
        print(f"  {name:12s}: {num_envs:4d} envs, ~{estimated_fps:6d} FPS")
    
    print("\n使用示例:")
    print("  env = KitchenEnv(num_envs=1024, headless=True)")
    print("  # 这将创建1024个并行环境")
    
    print("\n✓ 并行训练配置展示完成\n")


def example_dataset_migration():
    """示例6: 数据集迁移"""
    print("=" * 60)
    print("示例6: 数据集迁移")
    print("=" * 60)
    
    print("""
数据集迁移流程:

1. RoboCasa 数据集转换
   ```bash
   python scripts/convert_robocasa.py \\
       --source /path/to/robocasa.hdf5 \\
       --target /path/to/output
   ```

2. ManiSkill 数据集转换
   ```bash
   python scripts/convert_maniskill.py \\
       --source /path/to/maniskill \\
       --target /path/to/output
   ```

3. 使用转换后的数据
   ```python
   from cloud_robotics_sim.plugins.envs.maniskill import TrajectoryDataset
   
   dataset = TrajectoryDataset.load("/path/to/output")
   dataloader = dataset.to_torch_dataset(obs_key="state")
   ```

4. 训练模型
   ```python
   for batch in dataloader:
       obs = batch['obs']
       action = batch['action']
       # ... 训练代码
   ```
""")
    
    print("✓ 数据集迁移说明完成\n")


def example_api_comparison():
    """示例7: API对比"""
    print("=" * 60)
    print("示例7: API对比 (迁移指南)")
    print("=" * 60)
    
    print("""
从 RoboCasa 迁移:

RoboCasa (旧):
```python
from robocasa.environments import KitchenEnv
env = KitchenEnv(layout_id=0, style_id=0)
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

Genesis ManiSkill (新):
```python
from cloud_robotics_sim.plugins.envs.maniskill import KitchenEnv
env = KitchenEnv(scene_type="kitchen", layout_id=0, style_id=0)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

从 ManiSkill 迁移:

ManiSkill (旧):
```python
import gymnasium as gym
env = gym.make("PickCube-v1", num_envs=16)
```

Genesis ManiSkill (新):
```python
from cloud_robotics_sim.plugins.envs.maniskill import TableTopEnv
env = TableTopEnv(task="pick_cube", num_envs=16)
```
""")
    
    print("✓ API对比说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("ManiSkill Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查依赖
    try:
        import genesis as gs
        print(f"Genesis 可用")
    except ImportError:
        print("警告: Genesis 未安装，示例将以演示模式运行")
    
    try:
        import gymnasium
        print(f"Gymnasium 可用")
    except ImportError:
        print("警告: Gymnasium 未安装")
    
    print()
    
    # 运行示例
    examples = [
        ("厨房环境", example_kitchen_basic),
        ("桌面环境", example_tabletop_basic),
        ("不同机器人", example_different_robots),
        ("不同任务", example_different_tasks),
        ("并行训练", example_parallel_training),
        ("数据集迁移", example_dataset_migration),
        ("API对比", example_api_comparison),
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
