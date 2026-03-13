# slac Plugin

来源: [genesis-slac](../../genesis-slac)

基于 Genesis 引擎实现的 SLAC (Simulation-Pretrained Latent Action Space)，用于高自由度机器人全身真实世界强化学习。

## 核心功能

### 1. 潜在动作空间 (Latent Action Space)
将高维原始动作压缩到低维潜在空间：
```
原始动作空间: 50-100维 (全身关节)
    ↓ VAE编码器
潜在动作空间: 4-16维 (技能表示)
    ↓ VAE解码器
原始动作空间: 50-100维
```

**优势**:
- 促进时间抽象 (Temporal Abstraction)
- 实现安全探索
- 减少真实世界样本需求

### 2. 无监督技能发现 (Skill Discovery)
使用 DIAYN 风格的多样性驱动学习潜在动作空间：
- 最大化状态覆盖
- 促进技能解耦 (Disentanglement)
- 无需任务奖励

### 3. 真实世界下游学习
使用预训练的潜在动作空间作为动作接口：
- 潜在动作 → 原始动作 (通过VAE解码)
- 在潜在空间学习具体任务
- < 1小时真实世界交互

## 快速开始

### 阶段1: 预训练潜在动作空间 (仿真)

```python
from cloud_robotics_sim.plugins.controllers.slac import (
    SLACPretrainer, LatentActionVAE, SkillDiscovery
)

# 创建环境
env = MobileManipulatorEnv(num_envs=256)

# 创建模型
latent_action_vae = LatentActionVAE(
    action_dim=env.action_dim,  # 50-100维
    latent_dim=8,               # 压缩到8维
    obs_dim=env.obs_dim
)

skill_discovery = SkillDiscovery(
    latent_dim=8,
    num_skills=16
)

# 预训练
pretrainer = SLACPretrainer(
    latent_action_model=latent_action_vae,
    skill_discovery=skill_discovery,
    config=config
)

pretrainer.pretrain(env, num_iterations=100000)

# 保存预训练模型
torch.save(latent_action_vae.state_dict(), 'slac_pretrained.pt')
```

### 阶段2: 真实世界下游任务学习

```python
from cloud_robotics_sim.plugins.controllers.slac import (
    DownstreamPolicy, LatentActionController
)

# 加载预训练的潜在动作VAE
latent_action_vae = LatentActionVAE(action_dim=50, latent_dim=8)
latent_action_vae.load_state_dict(torch.load('slac_pretrained.pt'))

# 创建下游策略 (在潜在空间学习)
downstream_policy = DownstreamPolicy(
    obs_dim=env.obs_dim,
    latent_action_dim=8  # 使用潜在动作!
)

# 创建控制器
controller = LatentActionController(
    latent_action_vae=latent_action_vae,
    downstream_policy=downstream_policy
)

# 真实世界训练
for episode in range(100):
    obs = real_world_env.reset()
    for step in range(100):
        # 下游策略输出潜在动作
        latent_action = downstream_policy.get_action(obs)
        
        # VAE解码为原始动作
        action = controller.get_action(obs, latent_action)
        
        # 执行动作
        next_obs, reward, done, info = real_world_env.step(action)
        
        # 更新策略 (使用真实世界数据)
        downstream_policy.update(obs, latent_action, reward, next_obs)
```

## 算法原理

### 潜在动作VAE架构

```
VAE Encoder (动作 → 潜在空间)
├── 输入: 原始动作 a_t (50-100维)
├── MLP: [action_dim] → [256] → [128]
├── 输出: 潜在动作 z_t 的均值和方差
└── 重参数化: z_t = μ + σ * ε

VAE Decoder (潜在空间 → 动作)
├── 输入: 潜在动作 z_t (4-16维) + 观测 o_t
├── MLP: [latent_dim + obs_dim] → [128] → [256]
└── 输出: 重建动作 â_t (50-100维)

损失函数:
L_VAE = ||a_t - â_t||² + β * KL(q(z|a) || p(z))
```

### 技能发现 (DIAYN风格)

```
目标: 学习多样化的潜在动作技能

判别器 D(s, z):
├── 输入: 状态s + 潜在动作z
├── 输出: 预测的技能标签
└── 损失: 最大化互信息 I(s; z)

策略 π(z):
├── 输入: 观测
├── 输出: 潜在动作z
└── 目标: 最大化多样性 -log D(s, z)

总损失:
L = L_VAE + α * L_DIAYN
```

### 下游策略学习

```
潜在动作策略 π_θ(o):
├── 输入: 观测 o (100-200维)
├── MLP: [obs_dim] → [256] → [128]
└── 输出: 潜在动作 z (4-16维)

VAE解码器 (冻结):
├── 输入: 潜在动作 z + 观测 o
└── 输出: 原始动作 a (50-100维)

训练:
- 只在潜在空间更新策略
- VAE参数固定
- 使用任何RL算法 (SAC, PPO等)
```

## 配置参数

### 预训练配置

```yaml
# pretrain_config.yaml
latent_action:
  action_dim: 50          # 原始动作维度
  latent_dim: 8           # 潜在动作维度
  hidden_dims: [256, 128]
  beta: 0.001             # KL散度权重

skill_discovery:
  num_skills: 16          # 技能数量
  diversity_weight: 0.1   # 多样性权重
  
training:
  num_envs: 256
  num_iterations: 100000
  learning_rate: 3e-4
  batch_size: 256
```

### 下游任务配置

```yaml
# finetune_config.yaml
downstream_policy:
  obs_dim: 150
  latent_action_dim: 8
  hidden_dims: [256, 128]
  
rl_algorithm: sac         # sac, ppo, etc.

real_world:
  max_episodes: 100
  episode_length: 100
  safety_threshold: 0.9   # 安全动作阈值
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 原始动作维度 | 50-100 |
| 潜在动作维度 | 4-16 |
| 压缩比 | 6x-25x |
| 预训练时间 | 2-4小时 (仿真) |
| 真实世界学习 | < 1小时 |
| 任务成功率 | > 85% |

## 使用场景

### 适合使用SLAC的场景:
- 高自由度全身机器人
- 真实世界样本昂贵
- 需要安全探索
- 多任务学习

### 不适合的场景:
- 低自由度机器人 (< 10 DOF)
- 仿真到现实差距小
- 大量真实数据可用

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## Sim-to-Real 流程

```
阶段1: 仿真预训练 (离线)
├── 低保真仿真器
├── 无监督技能发现
└── 潜在动作空间学习
         ↓
阶段2: 真实世界微调 (在线)
├── 加载预训练VAE
├── 冻结VAE参数
├── 只学习下游策略
└── 潜在空间探索
```

## 与其他方法对比

| 方法 | 预训练 | 样本效率 | 安全性 | 适用性 |
|------|--------|----------|--------|--------|
| 标准RL | 无 | 低 | 低 | 仿真/真实 |
| 模仿学习 | 演示 | 高 | 中 | 真实 |
| SLAC | 仿真无监督 | 高 | 高 | 真实 |

## 引用

```bibtex
@article{hu2025slac,
  title={SLAC: Simulation-Pretrained Latent Action Space 
         for Whole-Body Real-World RL},
  author={Hu, Jiaheng and Stone, Peter and Mart{\'i}n-Mart{\'i}n, Roberto},
  journal={arXiv preprint arXiv:2506.04147},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 genesis-slac 迁移到 genesis-cloud-sim plugins
- **2026-02-10**: 初始版本，实现SLAC三阶段训练
