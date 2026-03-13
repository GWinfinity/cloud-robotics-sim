# wbm_embrace Plugin

来源: [genesis-wbm-embrace](../../genesis-wbm-embrace)

基于 Genesis 引擎实现的 WBM Embrace，人形机器人全身操作大型物体。

## 核心功能

### 1. 人类运动先验 (Human Motion Prior)
利用预训练的大规模人类运动数据：
- 生成运动学自然、物理可行的全身运动模式
- 教师-学生架构蒸馏知识
- 提高运动的自然性和稳定性

```
人类运动数据集
├── 全身运动捕捉数据
├── 多样物体交互
└── 自然运动模式
         ↓
Motion Prior VAE
├── 编码器: 人类姿态 → 潜在空间
├── 解码器: 潜在向量 → 人形姿态
└── 损失: 重建 + 物理可行性
         ↓
教师-学生蒸馏
├── 教师: Motion Prior (冻结)
└── 学生: 机器人策略
```

### 2. NSDF (Neural Signed Distance Field)
神经符号距离场提供准确的几何感知：
- 隐式表面表示
- 连续距离查询
- 提高长时程任务中的接触意识

### 3. 全身拥抱策略
协调控制手臂和躯干：
- 稳定的多接触交互
- 增强载荷能力
- 解决传统抓取的稳定性限制

## 快速开始

### 预训练运动先验

```python
from cloud_robotics_sim.plugins.controllers.wbm_embrace import MotionPrior

# 创建运动先验模型
motion_prior = MotionPrior(
    human_pose_dim=72,
    robot_pose_dim=29,
    latent_dim=32
)

# 在人类运动数据上预训练
for batch in human_motion_dataloader:
    human_pose = batch['pose']
    
    # 重建损失
    reconstructed = motion_prior(human_pose)
    loss = mse_loss(reconstructed, human_pose)
    
    loss.backward()
    optimizer.step()

# 保存
motion_prior.save('checkpoints/motion_prior.pt')
```

### 训练拥抱策略

```python
from cloud_robotics_sim.plugins.controllers.wbm_embrace import (
    EmbraceEnv, TeacherStudentPolicy, NSDF
)

# 创建环境
env = EmbraceEnv(
    object_type='box',
    object_size=[0.5, 0.3, 0.4],
    num_envs=64
)

# 创建NSDF表示
nsdf = NSDF(object_mesh='box.obj')

# 创建策略
policy = TeacherStudentPolicy(
    motion_prior_path='checkpoints/motion_prior.pt',
    use_nsdf=True
)

# 训练循环
for episode in range(10000):
    obs = env.reset()
    for step in range(500):
        # NSDF提供几何感知
        distance_field = nsdf.query(env.robot_pose)
        
        # 策略输出动作
        action = policy(obs, distance_field)
        
        # 执行
        obs, reward, done, info = env.step(action)
```

## 算法原理

### 运动先验VAE

```
编码器: q(z|x)
├── 输入: 人类姿态 x (72维)
├── MLP: [72] → [256] → [128]
└── 输出: 潜在分布 (μ, σ)

解码器: p(x|z)
├── 输入: 潜在向量 z (32维)
├── MLP: [32] → [128] → [256]
└── 输出: 重建姿态 x̂ (72维)

损失函数:
L = ||x - x̂||² + KL(q(z|x) || p(z))
```

### 教师-学生架构

```
教师网络 (冻结):
├── 预训练的 Motion Prior
├── 提供人类运动参考
└── 不直接用于控制

学生网络 (可训练):
├── 机器人策略 π(s)
├── 输入: 机器人观测
├── 输出: 机器人动作
└── 蒸馏目标: 模仿教师的运动模式

蒸馏损失:
L = α * ||π(s) - teacher_motion||² + β * L_task
```

### NSDF几何感知

```
神经距离场:
SDF(x) = d  (点x到物体表面的距离)

查询网络:
├── 输入: 查询点 p (3D坐标)
├── MLP: [3] → [64] → [64] → [1]
└── 输出: 有符号距离

应用:
- 接触检测: SDF < threshold
- 距离场梯度: ∇SDF (表面法向)
- 碰撞避免: 最小化负SDF
```

### 全身拥抱控制器

```
观测:
├── 机器人本体感觉 (关节位置/速度)
├── 目标物体状态 (位置/姿态)
├── NSDF距离场
└── 接触力信息

策略网络:
├── 输入: 观测 (约150维)
├── MLP: [150] → [512] → [256] → [128]
├── 输出: 全身动作 (29维)
└── 激活: Tanh

动作分解:
├── 左臂 (7维)
├── 右臂 (7维)
├── 躯干 (8维)
└── 头部 (2维) + 腿部 (5维)
```

## 配置参数

### 运动先验配置

```yaml
# motion_prior_config.yaml
vae:
  human_pose_dim: 72
  robot_pose_dim: 29
  latent_dim: 32
  hidden_dims: [256, 128]
  
training:
  dataset: human_motion_amass
  batch_size: 256
  learning_rate: 3e-4
  num_epochs: 100
```

### 拥抱策略配置

```yaml
# embrace_config.yaml
environment:
  object_types: ['box', 'cylinder', 'sphere']
  object_mass_range: [5, 50]  # kg
  num_envs: 64
  episode_length: 500

nsdf:
  resolution: 128
  query_points: 1000
  
policy:
  obs_dim: 150
  action_dim: 29
  hidden_dims: [512, 256]
  
training:
  motion_prior_weight: 0.5
  task_reward_weight: 1.0
  contact_reward_weight: 0.3
```

## 支持物体类型

| 类型 | 尺寸范围 | 质量范围 | 难度 |
|------|----------|----------|------|
| Box | 0.3-1.0m | 5-30kg | ⭐⭐ |
| Cylinder | 0.3-0.6m | 5-40kg | ⭐⭐⭐ |
| Sphere | 0.2-0.5m | 5-35kg | ⭐⭐⭐⭐ |
| Irregular | 混合 | 10-50kg | ⭐⭐⭐⭐⭐ |

## 性能指标

| 指标 | 数值 |
|------|------|
| 拥抱成功率 | > 85% |
| 物体稳定性 | < 0.05m 位移 |
| 平均接触点数 | 8-12 |
| 载荷能力提升 | 3x vs 单臂 |

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## 与传统抓取对比

| 特性 | 传统抓取 | WBM Embrace |
|------|----------|-------------|
| 接触点数 | 1-2 | 8-12 |
| 最大载荷 | 5-10kg | 30-50kg |
| 稳定性 | 低 | 高 |
| 物体形状限制 | 高 | 低 |
| 控制复杂度 | 低 | 高 |

## 引用

```bibtex
@article{zheng2025embracing,
  title={Embracing Bulky Objects with Humanoid Robots: 
         Whole-Body Manipulation with Reinforcement Learning},
  author={Zheng, Chunxin and Chen, Kai and Bi, Zhihai and 
          Li, Yulin and Pan, Zhou and Jinni and Haoang and Ma, Jun},
  journal={arXiv preprint arXiv:2509.13534},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 genesis-wbm-embrace 迁移到 genesis-cloud-sim plugins
- **2026-02-18**: 初始版本，实现全身拥抱操作
