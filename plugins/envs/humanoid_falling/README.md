# humanoid_falling Plugin

来源: [genesis-humanoid-falling](../../genesis-humanoid-falling)

基于 Genesis 引擎实现的人形机器人跌倒保护，复现论文 "Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning"。

## 核心功能

### 1. 自我保护策略学习
训练人形机器人在被外力推倒时，学会通过形成保护性姿势来减少冲击损伤。

### 2. 三角形保护结构
通过奖励函数引导机器人形成三角形支撑结构：
```
三角形结构
├── 两个接触点 (双手)
└── 一个顶点 (躯干/臀部)
    
优势:
- 分散冲击力
- 保护关键关节
- 减少头部碰撞
```

### 3. 课程学习 (Curriculum Learning)
逐步增加训练难度：
| 阶段 | 推力范围 | 方向变化 | 目标 |
|------|----------|----------|------|
| 1 | 50-100N | 固定 | 学会基本平衡 |
| 2 | 100-200N | ±30° | 学会侧面保护 |
| 3 | 200-300N | ±60° | 学会多方向保护 |
| 4 | 300-500N | 任意 | 学会极端情况 |

### 4. 冲击监测
实时监测关键部位受力：
- 头部 (避免脑震荡)
- 躯干 (保护内脏)
- 关节 (避免超限)

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.envs.humanoid_falling import HumanoidFallingEnv

# 创建环境
env = HumanoidFallingEnv(
    config_path='configs/train_config.yaml',
    num_envs=1,
    headless=False
)

# 重置环境
obs = env.reset()

# 运行回合
for step in range(500):
    # 随机动作或使用策略
    action = policy(obs)  # 或 env.action_space.sample()
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    # 监测冲击
    if info['impact_detected']:
        print(f"冲击检测! 力: {info['contact_force']:.2f}N")
    
    if done:
        obs = env.reset()
```

### 使用课程学习

```python
from cloud_robotics_sim.plugins.envs.humanoid_falling import (
    HumanoidFallingEnv, FallingCurriculum
)

# 创建环境和课程
env = HumanoidFallingEnv(config, num_envs=1)
curriculum = FallingCurriculum(
    env=env,
    stages=[
        {'push_force': [50, 100], 'direction_range': 0},
        {'push_force': [100, 200], 'direction_range': 30},
        {'push_force': [200, 300], 'direction_range': 60},
        {'push_force': [300, 500], 'direction_range': 180},
    ]
)

# 训练循环
for episode in range(10000):
    # 根据进度调整课程难度
    curriculum.update(episode)
    
    obs = env.reset()
    # ... 训练代码
```

### 训练策略

```python
from cloud_robotics_sim.plugins.envs.humanoid_falling import PPO

# 创建PPO算法
ppo = PPO(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# 训练
ppo.train(total_timesteps=10_000_000)
```

## 算法原理

### 网络架构

```
策略网络 (Actor)
├── 输入: 观测 (约 100 维)
│   ├── 关节位置 (19)
│   ├── 关节速度 (19)
│   ├── 躯干姿态 (6)
│   ├── 外力信息 (6)
│   └── 历史动作 (19)
├── MLP: [100] → [512] → [256] → [128]
├── 激活: ELU
└── 输出: 动作 (19维关节目标)

价值网络 (Critic)
├── 输入: 完整观测 (含特权信息)
├── MLP: [150] → [512] → [256] → [128]
└── 输出: 状态价值
```

### 奖励设计

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| 存活奖励 | +1.0 | 每步存活奖励 |
| 冲击惩罚 | -0.1 | 高冲击力惩罚 |
| 三角形奖励 | +0.5 | 形成三角形支撑 |
| 头部保护 | -0.2 | 头部接触惩罚 |
| 关节超限 | -0.1 | 关节角度超限 |
| 能量效率 | -0.01 | 动作平滑性 |
| 自碰撞 | -0.1 | 自碰撞惩罚 |

### 课程学习机制

```python
# 课程阶段定义
stages = [
    # 阶段1: 基础平衡
    {
        'push_force': [50, 100],
        'push_direction': 'random_horizontal',
        'success_threshold': 0.8,
    },
    # 阶段2: 侧面跌倒
    {
        'push_force': [100, 200],
        'push_direction': 'lateral',
        'success_threshold': 0.7,
    },
    # 阶段3: 多方向
    {
        'push_force': [200, 300],
        'push_direction': 'any',
        'success_threshold': 0.6,
    },
    # 阶段4: 极端情况
    {
        'push_force': [300, 500],
        'push_direction': 'any',
        'success_threshold': 0.5,
    },
]
```

## 配置参数

### 环境配置

```yaml
# train_config.yaml
genesis:
  dt: 0.01
  substeps: 10

env:
  episode_length: 500
  push_force_range: [50, 500]  # 牛顿
  push_duration: 0.1  # 秒
  
robot:
  mjcf_path: 'assets/humanoid/humanoid.xml'
  init_pos: [0, 0, 1.0]
  
observation:
  include_joint_pos: true
  include_joint_vel: true
  include_imu: true
  include_contact: true
  history_length: 3
```

### 奖励配置

```yaml
rewards:
  survival:
    weight: 1.0
  
  impact:
    weight: -0.1
    threshold: 100.0  # 冲击力阈值(N)
  
  triangle_structure:
    weight: 0.5
    min_contact_points: 2
  
  head_protection:
    weight: -0.2
    head_link_name: 'head'
  
  joint_limit:
    weight: -0.1
  
  energy:
    weight: -0.01
  
  self_collision:
    weight: -0.1
```

### 课程配置

```yaml
curriculum:
  enabled: true
  stages:
    - name: 'basic_balance'
      push_force: [50, 100]
      direction_range: 0
      success_threshold: 0.8
      
    - name: 'lateral_fall'
      push_force: [100, 200]
      direction_range: 30
      success_threshold: 0.7
      
    - name: 'multi_direction'
      push_force: [200, 300]
      direction_range: 60
      success_threshold: 0.6
      
    - name: 'extreme_cases'
      push_force: [300, 500]
      direction_range: 180
      success_threshold: 0.5
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 存活率 (轻度推力) | > 90% |
| 存活率 (中度推力) | > 75% |
| 存活率 (重度推力) | > 60% |
| 平均冲击减少 | 40% |
| 头部碰撞避免 | > 85% |

## 可视化

```python
# 可视化跌倒保护策略
env = HumanoidFallingEnv(config, headless=False)
obs = env.reset()

# 应用推力
env.apply_push(force=[200, 0, 0], duration=0.1)

# 观察保护动作
for _ in range(200):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    
    if info['contact_forces']:
        print(f"接触力: {info['contact_forces']}")
```

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## 安全考虑

⚠️ **警告**: 本项目训练的策略仅用于仿真环境。在部署到真实机器人前：

1. 充分测试策略安全性
2. 添加硬件安全限制
3. 使用保护装备
4. 渐进式测试 (从轻推力开始)

## 引用

```bibtex
@article{humanoid_falling_2024,
  title={Discovering Self-Protective Falling Policy for Humanoid Robot 
         via Deep Reinforcement Learning},
  journal={Robotics and Automation},
  year={2024}
}
```

## Changelog

- **2026-03-13**: 从 genesis-humanoid-falling 迁移到 genesis-cloud-sim plugins
- **2026-01-20**: 初始版本，实现跌倒保护策略学习
