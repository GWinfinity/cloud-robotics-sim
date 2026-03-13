# hugwbc Plugin

来源: [hugwbc-genesis](../../hugwbc-genesis)

HugWBC 在 Genesis 物理引擎中的实现 - 统一的人形机器人全身控制器。

## 核心功能

### 1. 统一全身控制
- **多任务支持**: 平地行走、上下楼梯、复杂地形、交互任务
- **全身协调**: 手臂摆动 + 腿部运动
- **实时控制**: 50Hz 控制频率

### 2. 非对称 Actor-Critic
```
Actor (策略网络)
├── 输入: 机器人本体感觉观测
├── 观测维度:  proprioception + commands
└── 输出: 关节目标位置

Critic (价值网络)
├── 输入: Actor观测 + 特权信息
├── 特权信息: 地形高度、接触力、摩擦力
└── 输出: 状态价值估计
```

### 3. 命令跟踪
支持速度命令：
- `vx`: 前进/后退速度 [-1.0, 2.0] m/s
- `vy`: 侧移速度 [-0.5, 0.5] m/s  
- `yaw_rate`: 偏航角速度 [-1.0, 1.0] rad/s

### 4. 步态控制
基于相位周期的步态生成：
- 支持多种步态: 行走、跑步、跳跃
- 自适应步频: 根据速度命令调整
- 接触状态估计: 基于相位预测足端接触

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.controllers.hugwbc import HugWBCEnv, TaskType

# 创建环境
env = HugWBCEnv(
    task="h1_loco",      # 任务类型
    num_envs=1,          # 环境数量
    headless=False,      # 是否显示GUI
    device='cuda'        # 计算设备
)

# 重置环境
obs = env.reset()

# 设置速度命令 [vx, vy, yaw_rate]
env.commands = np.array([1.0, 0.0, 0.0])  # 前进 1.0 m/s

# 运行回合
for step in range(1000):
    # 获取动作 (使用策略或随机)
    action = policy(obs)  # 或 env.action_space.sample()
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

### 训练策略

```python
from cloud_robotics_sim.plugins.controllers.hugwbc import HugWBCEnv, PPO

# 创建环境
env = HugWBCEnv(task="h1_loco", num_envs=4096, headless=True)

# 创建PPO算法
ppo = PPO(
    env=env,
    learning_rate=1e-3,
    n_steps=24,
    batch_size=16384,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95
)

# 训练
ppo.train(total_timesteps=100_000_000)
```

## 算法原理

### 网络架构

```
Actor Network
├── Input: obs_dim (约 50 维)
├── MLP: [obs_dim] → [512] → [256] → [128]
├── Activation: ELU
└── Output: action_dim (关节目标位置)

Critic Network
├── Input: obs_dim + privileged_dim (约 200 维)
├── MLP: [input_dim] → [512] → [256] → [128]
├── Activation: ELU
└── Output: 1 (状态价值)
```

### 观测空间

**Actor 观测 (proprioception)**:
| 分量 | 维度 | 说明 |
|------|------|------|
| 角速度 | 3 | 机体角速度 |
| 重力向量 | 3 | 投影到机体坐标系 |
| 速度命令 | 3 | [vx, vy, yaw_rate] |
| 关节位置 | 19 | 相对于默认位置 |
| 关节速度 | 19 | 关节角速度 |
| 上一动作 | 19 | 上一帧动作 |
| 相位 | 2 | [sin(phase), cos(phase)] |
| **总计** | **68** | - |

**Critic 特权观测**:
| 分量 | 维度 | 说明 |
|------|------|------|
| Actor观测 | 68 | 完整复制 |
| 基座速度 | 3 | 线速度 (直接测量) |
| 角速度 | 3 | 角速度 |
| 地形高度 | 100 | 周围地形采样 |
| 接触力 | 4 | 足端接触力 |
| 摩擦力 | 1 | 地面摩擦系数 |
| **总计** | **179** | - |

### 奖励设计

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| tracking_lin_vel | 2.0 | 线速度跟踪 |
| tracking_ang_vel | 0.5 | 角速度跟踪 |
| lin_vel_z | -2.0 | 垂直速度惩罚 |
| orientation | -1.0 | 姿态偏离惩罚 |
| feet_air_time | 1.0 | 离地时间奖励 |
| collision | -1.0 | 碰撞惩罚 |
| action_rate | -0.01 | 动作变化惩罚 |
| torque | -0.0001 | 力矩惩罚 |

## 配置参数

### 任务配置

```yaml
# h1_loco.yaml - 平地行走
task: h1_loco
robot:
  mjcf_path: "assets/h1/h1.xml"
  init_pos: [0, 0, 1.0]

env:
  episode_length: 1000
  gait_frequency: 1.25  # Hz
  
  command_ranges:
    lin_vel_x: [-1.0, 2.0]
    lin_vel_y: [-0.5, 0.5]
    ang_vel_yaw: [-1.0, 1.0]

rewards:
  tracking_lin_vel:
    weight: 2.0
    sigma: 0.25
  tracking_ang_vel:
    weight: 0.5
    sigma: 0.25
```

### 训练参数

```yaml
training:
  num_envs: 4096
  num_steps: 24
  learning_rate: 1e-3
  
  # PPO参数
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  num_epochs: 5
  batch_size: 16384
  
  # 域随机化
  domain_rand: true
  push_robots: true
  randomize_friction: true
```

## 性能指标

| 任务 | 跟踪误差 | 成功率 | 训练时间 |
|------|----------|--------|----------|
| h1_loco | < 0.1 m/s | > 95% | ~4h |
| h1_stairs | < 0.15 m/s | > 90% | ~6h |
| h1_terrain | < 0.2 m/s | > 85% | ~8h |

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## 任务类型

```python
from cloud_robotics_sim.plugins.controllers.hugwbc import TaskType

# 可用任务
TaskType.LOCO        # 平地行走
TaskType.STAIRS      # 上下楼梯
TaskType.TERRAIN     # 复杂地形
TaskType.INTERACTION # 交互任务
```

## 扩展指南

### 添加新任务

1. 创建配置文件 `configs/my_task.yaml`
2. 在环境代码中添加任务逻辑
3. 运行训练

```python
# 使用新任务
env = HugWBCEnv(task="my_task", config_path="configs/my_task.yaml")
```

### 使用自定义机器人

```python
config = {
    'robot': {
        'mjcf_path': 'path/to/my_robot.xml',
        'joint_names': ['joint1', 'joint2', ...],
        'init_pos': [0, 0, 1.0]
    }
}
env = HugWBCEnv(task="custom", config=config)
```

## 引用

```bibtex
@inproceedings{xue2025hugwbc,
  title={HugWBC: A Unified and General Humanoid Whole-Body Controller 
         for Versatile Locomotion},
  author={Xue, Yufei and Dong, Wentao and Liu, Minghuan and 
          Zhang, Weinan and Pang, Jiangmiao},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 hugwbc-genesis 迁移到 genesis-cloud-sim plugins
- **2026-02-05**: 初始版本，适配 Genesis 物理引擎
