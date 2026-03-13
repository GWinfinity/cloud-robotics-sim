# Residual RL Plugin

来源: [genesis-residual-rl](https://github.com/Genesis-Embodied-AI/Genesis)

## 核心功能

残差强化学习 (Residual RL) - 安全地微调行为克隆策略。

**核心思想:**
```
最终动作 = BC策略(观测) + α × 残差网络(观测)

BC策略 (冻结):
  - 预训练的行为克隆策略
  - 提供合理的基准动作
  - 保证安全性

残差网络 (可训练):
  - 轻量级网络 (参数量少)
  - 输出受限: [-ε, ε]
  - 学习小幅修正
```

**优势:**
- ✅ **安全性**: BC 提供基础保障，残差只做小修正
- ✅ **样本效率**: 残差网络小，学习快
- ✅ **稳定性**: 输出范围受限，避免危险动作

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.controllers.residual_rl import (
    ResidualNetwork, CombinedPolicy, ResidualSAC, BCPolicy
)

# 1. 加载预训练的 BC 策略
bc_policy = BCPolicy(obs_dim=100, action_dim=10)
bc_policy.load('bc_policy.pt')

# 2. 创建残差网络
residual_net = ResidualNetwork(
    obs_dim=100,
    action_dim=10,
    residual_scale=0.2  # 输出限制 [-0.2, 0.2]
)

# 3. 组合策略
policy = CombinedPolicy(
    bc_policy=bc_policy,
    residual_net=residual_net,
    freeze_bc=True  # 冻结 BC
)

# 4. 使用
obs = env.reset()
action = policy(obs)['action']  # BC动作 + 残差
```

### 训练残差策略

```python
from cloud_robotics_sim.plugins.controllers.residual_rl import ResidualSAC

# 创建训练器
trainer = ResidualSAC(
    combined_policy=policy,
    q_network1=QNetwork(100, 10),
    q_network2=QNetwork(100, 10),
    lr_residual=3e-4
)

# 训练循环
for episode in range(1000):
    obs = env.reset()
    
    for step in range(500):
        # 选择动作
        action = trainer.select_action(obs)
        
        # 执行
        next_obs, reward, done, _ = env.step(action)
        
        # 存储经验
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        # 更新 (只更新残差网络!)
        if step % 4 == 0:
            batch = replay_buffer.sample(256)
            losses = trainer.update(batch)
        
        obs = next_obs
```

## 算法原理

### 残差学习框架

```
策略表示:
π_θ(a|s) = π_BC(a|s) + π_residual(s; φ)

其中:
- π_BC: 预训练 BC 策略 (冻结参数)
- π_residual: 残差网络 (可训练参数 φ)
- ||π_residual(s; φ)|| ≤ ε (输出约束)

优化目标 (SAC 风格):
J(φ) = E[min(Q1(s, a), Q2(s, a)) - α log π_θ(a|s)]
其中 a = π_BC(s) + π_residual(s; φ)

关键: 只更新 φ (残差参数)，BC 参数冻结
```

### 残差网络架构

```python
class ResidualNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, residual_scale=0.2):
        # 轻量级 MLP
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 输出头
        self.residual_head = nn.Linear(64, action_dim)
        
        # 输出限制 (关键!)
        self.residual_scale = residual_scale
        self.output_activation = lambda x: torch.tanh(x) * residual_scale
        
        # 小权重初始化 (确保初始残差≈0)
        self._init_weights(gain=0.01)
```

### 组合策略

```python
class CombinedPolicy(nn.Module):
    def forward(self, obs):
        # BC 策略输出 (无梯度)
        with torch.no_grad():
            bc_action = self.bc_policy(obs)['action']
        
        # 残差网络输出
        residual = self.residual_net(obs)['residual']
        
        # 最终动作
        final_action = bc_action + residual
        final_action = torch.clamp(final_action, -1, 1)
        
        return {
            'bc_action': bc_action,
            'residual': residual,
            'action': final_action
        }
```

## 配置参数

### 残差网络

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `obs_dim` | int | - | 观测维度 |
| `action_dim` | int | - | 动作维度 |
| `hidden_dims` | list | [256, 128, 64] | 隐藏层维度 |
| `residual_scale` | float | 0.2 | 输出范围限制 ε |
| `use_layer_norm` | bool | True | 使用 LayerNorm |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `lr_residual` | float | 3e-4 | 残差网络学习率 |
| `lr_q` | float | 3e-4 | Q 网络学习率 |
| `gamma` | float | 0.99 | 折扣因子 |
| `tau` | float | 0.005 | 软更新系数 |
| `batch_size` | int | 256 | 批次大小 |

### ε (residual_scale) 调优

```python
# 训练阶段逐步增大 ε (课程学习)
def get_residual_scale(episode):
    if episode < 100:
        return 0.1   # 初始: 小范围探索
    elif episode < 500:
        return 0.2   # 中期: 增大改进空间
    else:
        return 0.3   # 后期: 允许更大调整
```

## 文件结构

```
residual_rl/
├── core/
│   ├── residual_network.py    # 残差网络 + 组合策略 + Residual SAC
│   ├── bc_policy.py           # BC 策略基类
│   └── vision_encoder.py      # 视觉编码器
├── configs/
│   └── default.yaml           # 默认配置
├── examples/
│   ├── basic_usage.py         # 基础使用示例
│   └── ab_test.py             # A/B 测试示例
├── README.md
└── plugin.yaml
```

## 使用示例

### 示例 1: Sim-to-Real 微调

```python
# 场景: 仿真训练的策略在真实环境表现不佳

# Step 1: 加载仿真训练的 BC 策略
bc_policy = BCPolicy.load('sim_bc_policy.pt')

# Step 2: 创建残差网络
residual_net = ResidualNetwork(obs_dim, action_dim, residual_scale=0.1)

# Step 3: 在真实环境微调残差
policy = CombinedPolicy(bc_policy, residual_net)
trainer = ResidualSAC(policy, q1, q2)

for episode in range(500):  # 少量样本即可
    obs = real_env.reset()
    # ... 训练 ...
```

### 示例 2: 人类演示优化

```python
# 场景: 人类演示训练 BC，但动作不够流畅

# Step 1: 从演示训练 BC
bc_policy = train_bc_from_demonstrations(demos)

# Step 2: 用 Residual RL 优化
residual_net = ResidualNetwork(obs_dim, action_dim)
policy = CombinedPolicy(bc_policy, residual_net)

# Step 3: 自博弈优化
for episode in range(1000):
    # 现在策略 = 人类风格 + RL优化
    # 结果: 超越人类表现
```

### 示例 3: 安全探索

```python
# 场景: 在危险环境 (如悬崖边行走)

# 使用保守的残差范围
residual_net = ResidualNetwork(
    obs_dim, action_dim,
    residual_scale=0.05  # 非常小的修正
)

# BC 提供安全动作，残差只做微调
# 即使 RL 探索，也不会产生危险动作
```

## 关键实现细节

### 1. 小权重初始化

```python
def _init_weights(self, gain=0.01):
    """确保初始残差接近 0"""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
```

### 2. 梯度裁剪

```python
# 防止残差过大
torch.nn.utils.clip_grad_norm_(
    residual_net.parameters(), max_norm=1.0
)
```

### 3. BC 策略冻结

```python
for param in bc_policy.parameters():
    param.requires_grad = False
```

## 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 残差始终接近 0 | 初始化太小 | 检查 gain，适当增大 ε |
| 训练不稳定 | 学习率太高 | 降低 lr_residual，增加梯度裁剪 |
| 性能不如纯 BC | 探索不足 | 增大 ε，检查奖励函数 |
| BC 动作被覆盖 | ε 太大 | 减小 residual_scale |
| Q 值爆炸 | 目标网络更新慢 | 增大 tau |

## 参考

- Paper: "Residual Reinforcement Learning for Robot Control" (Johannink et al., 2019 ICRA)
- Related: "Residual Policy Learning" (Silver et al., 2018)

## Changelog

- 2024-03-15: 从 genesis-residual-rl 迁移
- 添加完整的文档和示例
- 添加 A/B 测试支持
