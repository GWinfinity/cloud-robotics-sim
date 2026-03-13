# sim2real_dexterous Plugin

来源: [genesis-sim2real-dexterous](../../genesis-sim2real-dexterous)

基于 Genesis 引擎实现的 Sim-to-Real 灵巧双手操作，复现论文 "Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids"。

## 核心功能

### 1. Real-to-Sim 自动调优
- **系统识别**: 从真实世界轨迹识别仿真参数
- **参数优化**: 自动调整摩擦、质量、阻尼等参数
- **域适配**: 缩小仿真与现实差距

### 2. 通用奖励公式
```python
# 通用奖励设计
reward = w_contact * r_contact +      # 接触奖励
         w_object * r_object_goal +    # 物体目标奖励
         w_hand * r_hand_pose +        # 手部姿态奖励
         w_reg * r_regularization      # 正则化奖励
```

### 3. 分而治之策略蒸馏
```
单任务专家训练
├── Expert 1: Grasp-and-Reach
├── Expert 2: Box Lift
└── Expert 3: Bimanual Handover
         ↓
    策略蒸馏
         ↓
通用多任务策略
├── 共享视觉编码器
├── 任务特定头
└── 知识蒸馏损失
```

### 4. 混合物体表示
- **视觉**: RGB图像 (224x224)
- **点云**: 物体几何信息
- **本体感觉**: 关节位置/速度
- **物体状态**: 位置、姿态、速度

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.sim2real import (
    DexterousManipulationEnv, TaskType
)

# 创建环境
env = DexterousManipulationEnv(
    config=config,
    task_name='grasp_and_reach',
    num_envs=1,
    headless=False
)

# 重置环境
obs = env.reset()

# 运行回合
for step in range(500):
    # 随机动作或使用策略
    action = policy(obs)  # 54维动作
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

### Real-to-Sim 调优

```python
from cloud_robotics_sim.plugins.sim2real import Real2SimTuner

# 创建调优器
tuner = Real2SimTuner(
    env=env,
    real_trajectories=real_data
)

# 自动调优仿真参数
optimized_params = tuner.tune(
    parameters=['friction', 'mass', 'damping'],
    method='bayesian_optimization',
    iterations=100
)
```

### 策略蒸馏

```python
from cloud_robotics_sim.plugins.sim2real import PolicyDistillation

# 加载单任务专家
experts = {
    'grasp_and_reach': torch.load('expert_grasp.pt'),
    'box_lift': torch.load('expert_lift.pt'),
    'bimanual_handover': torch.load('expert_handover.pt')
}

# 创建蒸馏器
distillation = PolicyDistillation(
    student_policy=multi_task_policy,
    teacher_policies=experts
)

# 蒸馏训练
distillation.train(
    env=env,
    iterations=100000,
    temperature=4.0,
    alpha=0.5  # 蒸馏损失权重
)
```

## 算法原理

### 网络架构

```
混合物体编码器
├── 视觉编码器 (ResNet-18)
│   ├── 输入: RGB (3x224x224)
│   └── 输出: 512-dim 特征
├── 点云编码器 (PointNet)
│   ├── 输入: 点云 (Nx3)
│   └── 输出: 256-dim 特征
├── 本体感觉编码
│   ├── 输入: 关节状态
│   └── 输出: 128-dim 特征
└── 融合层
    ├── 拼接所有特征
    └── MLP: 896 → 512 → 256

策略网络
├── 输入: 混合物体表示 (256-dim)
├── MLP: [256] → [512] → [256] → [128]
└── 输出: 动作 (54-dim)
    ├── 左臂 (7)
    ├── 右臂 (7)
    ├── 左手 (12)
    ├── 右手 (12)
    └── 躯干 (16)
```

### Real-to-Sim 调优

**目标**: 最小化仿真与现实轨迹差异

```python
# 损失函数
L = || traj_real - traj_sim(params) ||^2 + λ * regularization(params)

# 可调参数
params = {
    'friction': [0.1, 2.0],
    'mass_scale': [0.8, 1.2],
    'damping': [0.01, 0.1],
    'kp': [10, 100],
    'kd': [0.1, 10]
}
```

### 通用奖励设计

适用于所有操作任务的奖励公式：

| 奖励项 | 公式 | 说明 |
|--------|------|------|
| 接触奖励 | `r_contact = I_contact * f_normal` | 鼓励稳定接触 |
| 物体目标 | `r_object = -\|obj - goal\|^2` | 物体到达目标 |
| 手部姿态 | `r_hand = -\|hand - hand_target\|^2` | 手部姿态匹配 |
| 能量效率 | `r_energy = -\|torque\|^2` | 减少能量消耗 |

### 策略蒸馏

**蒸馏损失**:
```python
L_distill = KL(π_teacher || π_student) * T^2
L_task = L_policy(student)  # 策略梯度损失
L_total = α * L_distill + (1-α) * L_task
```

## 配置参数

### 环境配置

```yaml
# dexterous_config.yaml
robot:
  num_joints: 54  # 双臂 + 双手 + 躯干
  hand_dofs: 24   # 每手12 DOF
  arm_dofs: 14    # 每臂7 DOF

env:
  episode_length: 500
  dt: 0.02
  substeps: 4

observation:
  image_size: [224, 224]
  use_point_cloud: true
  use_proprioception: true
  history_length: 3
```

### Real-to-Sim 配置

```yaml
real2sim:
  tunable_params:
    - friction
    - mass_scale
    - damping
    - kp
    - kd
  
  optimization:
    method: bayesian  # 或 gradient, evolutionary
    iterations: 100
    num_samples: 50
```

### 蒸馏配置

```yaml
distillation:
  temperature: 4.0
  alpha: 0.5  # 蒸馏损失权重
  batch_size: 256
  learning_rate: 3e-4
  
  tasks:
    - grasp_and_reach
    - box_lift
    - bimanual_handover
```

## 支持任务

| 任务 | 描述 | 难度 |
|------|------|------|
| grasp_and_reach | 抓取物体并伸展手臂 | ⭐⭐ |
| box_lift | 双手协作提升箱体 | ⭐⭐⭐ |
| bimanual_handover | 双手之间交接物体 | ⭐⭐⭐⭐ |

## 性能指标

| 指标 | Sim | Real |
|------|-----|------|
| Grasp-and-Reach | 95% | 88% |
| Box Lift | 92% | 82% |
| Bimanual Handover | 88% | 75% |

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## Sim-to-Real 流程

```
1. Real-to-Sim 调优
   └── 优化仿真参数匹配真实世界

2. 单任务专家训练
   ├── 在调优后的仿真中训练
   └── 每个任务一个专家策略

3. 策略蒸馏
   └── 合并专家为通用策略

4. 域随机化
   └── 提高策略鲁棒性

5. 部署到真实机器人
   └── Zero-shot 或 Few-shot 适应
```

## 引用

```bibtex
@article{lin2025sim2real,
  title={Sim-to-Real Reinforcement Learning for Vision-Based 
         Dexterous Manipulation on Humanoids},
  author={Lin, Toru and Sachdev, Kartik and Fan, Malik and 
          Jitendra, Malik and Zhu, Yuke},
  journal={arXiv preprint arXiv:2502.20396},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 genesis-sim2real-dexterous 迁移到 genesis-cloud-sim plugins
- **2026-02-01**: 初始版本，实现三种灵巧操作任务
