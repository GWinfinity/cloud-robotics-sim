# 残差强化学习 (Residual RL)

## 1. 算法原理

### 1.1 论文引用

**核心论文**: "Residual Reinforcement Learning for Robot Control" (Johannink et al., 2019 ICRA)

**相关论文**:
- "Residual Policy Learning" (Silver et al., 2018)
- "Leaving RL Track: Residual RL for Fine-tuning" (Zhu et al., 2024)

### 1.2 核心思想

#### 问题背景
传统行为克隆 (BC) 存在以下问题：
- **分布偏移**: 训练数据之外表现差
- **复合误差**: 小误差累积导致失败
- **无法改进**: 静态策略不能自我优化

纯 RL 又存在：
- **样本效率低**: 需要大量交互
- **探索危险**: 随机动作可能损坏机器人
- **收敛慢**: 从零开始学习困难

#### 残差学习框架

```
最终动作 = BC策略(观测) + α × 残差网络(观测)

           ┌─────────────────┐
观测 ─────→│   BC 策略        │──→ BC动作 (冻结)
           │   (预训练)       │
           └─────────────────┘
                    │
           ┌────────▼─────────┐
           │   残差网络        │──→ 残差动作 (可训练)
           │   (轻量级)       │     范围受限: [-ε, ε]
           └─────────────────┘
                    │
                    ▼
              最终动作 = BC动作 + 残差动作
```

**关键优势**:
1. **安全性**: BC 提供基础安全动作，残差只做小修正
2. **样本效率**: 残差网络小，学习快
3. **稳定性**: 残差范围受限，避免危险动作

#### 数学公式

```
策略表示:
π_θ(a|s) = π_BC(a|s) + π_residual(s; φ)

其中:
- π_BC: 预训练 BC 策略 (冻结参数)
- π_residual: 残差网络 (可训练参数 φ)
- ||π_residual(s; φ)|| ≤ ε (输出约束)

优化目标 (SAC风格):
J(φ) = E[min(Q1(s, a), Q2(s, a)) - α log π_θ(a|s)]
其中 a = π_BC(s) + π_residual(s; φ)

只更新 φ (残差参数)，BC 参数冻结
```

### 1.3 架构设计

```
┌────────────────────────────────────────────────────────────┐
│                      组合策略                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  BC 策略 (冻结)                                      │   │
│  │  - 可以是任何预训练策略                                │   │
│  │  - 行为克隆、专家演示、甚至硬编码规则                    │   │
│  │  - 提供合理的基准动作                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  残差网络 (可训练)                                    │   │
│  │  - 小型 MLP (如: obs→256→128→64→action)               │   │
│  │  - 输出限制: tanh * ε (ε通常 0.1-0.3)                 │   │
│  │  - LayerNorm 提高稳定性                              │   │
│  │  - Xavier 初始化 (小增益 0.01)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  最终动作 = BC动作 + 残差动作                          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                     残差 SAC 训练                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  双 Q 网络 (Critic)                                  │   │
│  │  - 评估最终动作的价值                                │   │
│  │  - 只更新 Q 网络参数 (非残差)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  策略更新                                            │   │
│  │  - 只更新残差网络参数 φ                              │   │
│  │  - BC 策略保持冻结                                   │   │
│  │  - 最大化 Q 值 + 熵奖励                              │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 使用场景

### 2.1 适用任务

| 任务类型 | 适用性 | 说明 |
|---------|-------|------|
| **操作微调** | ⭐⭐⭐⭐⭐ | 在 BC 基础上微调抓取、放置 |
| **行走优化** | ⭐⭐⭐⭐ | 优化 BC 步态的能效和稳定性 |
| **接触-rich任务** | ⭐⭐⭐⭐⭐ | 插销、装配等需要精细接触的任务 |
| **视觉伺服** | ⭐⭐⭐⭐ | BC 开环控制 + RL 视觉反馈 |
| **多任务迁移** | ⭐⭐⭐ | 同一 BC 策略 + 不同残差适应不同任务 |

### 2.2 典型应用模式

#### 模式 1: 模拟器 → 真实世界
```
Sim 训练 BC → Real 收集数据 → Residual RL 微调
```

#### 模式 2: 人类演示 → 自动改进
```
Human demo → BC 克隆 → Residual RL 超人类表现
```

#### 模式 3: 硬编码 → 学习型
```
Hard-coded baseline → Residual RL 自适应优化
```

### 2.3 限制与注意事项

#### BC 质量要求
- BC 策略不能太烂，至少能完成 30-50% 的任务
- 残差只能做小修正，不能纠正重大错误
- 如果 BC 完全错误方向，残差难以挽救

#### 残差范围限制
- ε 太小: 改进空间有限
- ε 太大: 可能产生危险动作，失去安全性
- 建议从 ε=0.1 开始，逐步增大

#### 探索挑战
- 残差受限意味着探索空间小
- 对于需要大改变的场景不适用
- 可以结合课程学习逐步增大 ε

### 2.4 与其他方法对比

| 方法 | 样本效率 | 安全性 | 最终性能 | 适用场景 |
|-----|---------|-------|---------|---------|
| **Residual RL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 有合理BC基线 |
| **纯 BC** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 数据充足无分布偏移 |
| **纯 RL** | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 数据充足无安全约束 |
| **BC + RL** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | BC预初始化RL |

---

## 3. 代码示例

### 3.1 最小可运行示例

```python
import torch
import torch.nn as nn
import numpy as np
import genesis as gs

# ==================== 残差网络定义 ====================

class ResidualNetwork(nn.Module):
    """
    轻量级残差网络
    学习对 BC 策略的小幅修正
    """
    def __init__(
        self,
        obs_dim: int = 512,
        action_dim: int = 7,
        hidden_dims: list = [256, 128, 64],
        residual_scale: float = 0.2,  # ε: 残差输出范围限制
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.residual_scale = residual_scale
        
        # 特征提取网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 残差输出头
        self.residual_head = nn.Linear(prev_dim, action_dim)
        
        # 小权重初始化确保初始残差接近0
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重为小值，确保初始残差接近0"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> dict:
        """
        前向传播
        
        Returns:
            dict with keys:
                - 'residual': 残差动作 (范围受限)
                - 'mean': 用于 SAC 的对数概率计算
                - 'log_std': 动作标准差
        """
        features = self.feature_net(obs)
        
        # 输出残差 (限制范围)
        residual = self.residual_head(features)
        residual = torch.tanh(residual) * self.residual_scale
        
        return {
            'residual': residual,
            'mean': residual,  # SAC 兼容性
            'log_std': torch.ones_like(residual) * -2.0  # 小的探索噪声
        }


class CombinedPolicy(nn.Module):
    """
    组合策略 = BC策略 + 残差网络
    """
    def __init__(
        self,
        bc_policy: nn.Module,
        residual_net: ResidualNetwork,
        freeze_bc: bool = True
    ):
        super().__init__()
        self.bc_policy = bc_policy
        self.residual_net = residual_net
        
        # 冻结 BC 策略
        if freeze_bc:
            for param in self.bc_policy.parameters():
                param.requires_grad = False
    
    def forward(self, obs: torch.Tensor) -> dict:
        # BC 策略输出 (无梯度)
        with torch.no_grad():
            bc_output = self.bc_policy(obs)
            bc_action = bc_output['action']
        
        # 残差网络输出
        residual_output = self.residual_net(obs)
        residual = residual_output['residual']
        
        # 最终动作
        final_action = bc_action + residual
        final_action = torch.clamp(final_action, -1, 1)
        
        return {
            'bc_action': bc_action,
            'residual': residual,
            'action': final_action,  # 兼容 RL 接口
            'mean': final_action,
            'log_std': residual_output['log_std']
        }


# ==================== 残差 SAC 训练器 ====================

class ResidualSAC:
    """
    残差 SAC 训练器
    只更新残差网络，BC 策略保持冻结
    """
    def __init__(
        self,
        combined_policy: CombinedPolicy,
        q_network1: nn.Module,
        q_network2: nn.Module,
        lr_residual: float = 3e-4,
        lr_q: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # 组合策略 (只训练残差部分)
        self.policy = combined_policy.to(device)
        
        # Q 网络 (双 Q)
        self.q1 = q_network1.to(device)
        self.q2 = q_network2.to(device)
        
        # 目标 Q 网络
        self.q1_target = type(q_network1)(
            q_network1.obs_dim, q_network1.action_dim
        ).to(device)
        self.q2_target = type(q_network2)(
            q_network2.obs_dim, q_network2.action_dim
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 优化器 (关键: 只优化残差网络参数!)
        residual_params = list(combined_policy.residual_net.parameters())
        self.policy_optimizer = torch.optim.Adam(
            residual_params, lr=lr_residual
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_q)
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            output = self.policy(obs_t)
            return output['action'].cpu().numpy()[0]
    
    def update(self, batch: dict) -> dict:
        """
        更新网络
        
        Args:
            batch: 包含 'obs', 'action', 'reward', 'next_obs', 'done'
        
        Returns:
            损失字典
        """
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        
        # ===== 更新 Q 网络 =====
        with torch.no_grad():
            # 下一个观测的策略输出
            next_output = self.policy(next_obs)
            next_action = next_output['action']
            
            # 目标 Q 值
            next_q1 = self.q1_target(next_obs, next_action)
            next_q2 = self.q2_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # 当前 Q 值
        current_q1 = self.q1(obs, action)
        current_q2 = self.q2(obs, action)
        
        # Q 损失
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ===== 更新残差网络 (关键!) =====
        output = self.policy(obs)
        new_action = output['action']
        
        new_q1 = self.q1(obs, new_action)
        new_q2 = self.q2(obs, new_action)
        new_q = torch.min(new_q1, new_q2)
        
        # 策略损失: 最大化 Q 值
        policy_loss = -new_q.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪 (稳定性)
        torch.nn.utils.clip_grad_norm_(
            self.policy.residual_net.parameters(), max_norm=1.0
        )
        
        self.policy_optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'residual_norm': output['residual'].norm().item()
        }
    
    def _soft_update_target(self):
        """软更新目标 Q 网络"""
        for param, target_param in zip(
            self.q1.parameters(), self.q1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        for param, target_param in zip(
            self.q2.parameters(), self.q2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


# ==================== Genesis 环境集成示例 ====================

class ResidualRLAgent:
    """
    在 Genesis 环境中使用残差 RL 的完整示例
    """
    def __init__(
        self,
        bc_checkpoint_path: str,
        obs_dim: int,
        action_dim: int,
        device: str = 'cuda'
    ):
        self.device = device
        
        # 加载 BC 策略
        self.bc_policy = self._load_bc_policy(bc_checkpoint_path)
        
        # 创建残差网络
        self.residual_net = ResidualNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            residual_scale=0.2
        )
        
        # 组合策略
        self.combined_policy = CombinedPolicy(
            bc_policy=self.bc_policy,
            residual_net=self.residual_net,
            freeze_bc=True
        )
        
        # 创建 Q 网络
        self.q1 = QNetwork(obs_dim, action_dim)
        self.q2 = QNetwork(obs_dim, action_dim)
        
        # 训练器
        self.trainer = ResidualSAC(
            combined_policy=self.combined_policy,
            q_network1=self.q1,
            q_network2=self.q2,
            device=device
        )
        
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def act(self, obs: np.ndarray, use_residual: bool = True) -> np.ndarray:
        """
        执行动作
        
        Args:
            obs: 当前观测
            use_residual: 是否使用残差 (训练时用，评估时可只用BC)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            output = self.combined_policy(obs_t)
            
            if use_residual:
                return output['action'].cpu().numpy()[0]
            else:
                return output['bc_action'].cpu().numpy()[0]
    
    def train_step(self, batch_size: int = 256) -> dict:
        """执行一步训练"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size)
        return self.trainer.update(batch)


def train_residual_rl():
    """完整的训练流程"""
    # 创建环境
    env = create_genesis_env()
    
    # 创建智能体
    agent = ResidualRLAgent(
        bc_checkpoint_path='bc_policy.pt',
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )
    
    # 训练循环
    for episode in range(1000):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(500):
            # 选择动作
            action = agent.act(obs, use_residual=True)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # 训练
            if step % 4 == 0:  # 每4步更新一次
                losses = agent.train_step()
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

---

## 4. 超参数指南

### 4.1 网络架构参数

| 参数 | 推荐值 | 调优建议 |
|-----|-------|---------|
| `hidden_dims` | [256, 128, 64] | 任务复杂↑则网络↑，但训练慢 |
| `residual_scale` (ε) | 0.1-0.3 | 安全优先用 0.1，需要大改进用 0.3 |
| `use_layer_norm` | True | 强烈推荐，提高稳定性 |

#### ε (residual_scale) 调优

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

### 4.2 训练参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `lr_residual` | 3e-4 | 残差网络学习率 |
| `lr_q` | 3e-4 | Q 网络学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 软更新系数 |
| `batch_size` | 256 | 批次大小 |
| `update_freq` | 4 | 每4个环境步更新一次 |

### 4.3 关键设计选择

#### 1. BC 策略冻结 vs 微调
```python
# 推荐: 完全冻结 (安全性)
freeze_bc = True

# 进阶: 后期微调 (性能优先)
if episode > 800:
    # 解冻 BC 最后几层
    for param in bc_policy.tail_layers.parameters():
        param.requires_grad = True
```

#### 2. 残差网络初始化
```python
# 关键: 小增益确保初始残差≈0
nn.init.xavier_uniform_(m.weight, gain=0.01)

# 对比: 正常初始化 (不推荐用于残差)
nn.init.xavier_uniform_(m.weight, gain=1.0)  # 残差可能太大
```

#### 3. 梯度裁剪
```python
# 必须添加，防止残差过大
torch.nn.utils.clip_grad_norm_(
    residual_net.parameters(), max_norm=1.0
)
```

### 4.4 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| **残差始终接近0** | 初始化太小或BC太好/太坏 | 检查BC性能，增大ε，调整初始化增益 |
| **训练不稳定** | 学习率太高或残差太大 | 降低lr，减小ε，增加梯度裁剪 |
| **性能不如纯BC** | 探索不足或奖励设计问题 | 增大ε，检查奖励函数，增加探索噪声 |
| **Q值爆炸** | 目标网络更新太慢 | 增大tau，或检查reward缩放 |
| **BC动作被完全覆盖** | ε太大或残差网络太深 | 减小ε，使用更小的残差网络 |

### 4.5 性能监控指标

```python
# 训练中应监控的关键指标
metrics = {
    'residual_norm': output['residual'].norm().item(),
    'residual_ratio': (residual_norm / bc_action_norm).item(),
    'bc_action_norm': bc_action.norm().item(),
    'final_action_norm': final_action.norm().item(),
    'q1_loss': q1_loss.item(),
    'policy_loss': policy_loss.item()
}

# 理想情况:
# - residual_norm: 0.1-0.5 (与ε相关)
# - residual_ratio: 0.05-0.2 (残差是BC的5-20%)
# - q1_loss: 稳定下降
# - policy_loss: 缓慢下降 (不能为负太多)
```

---

## 5. 进阶技巧

### 5.1 自适应残差范围
```python
class AdaptiveResidualNetwork(nn.Module):
    """根据不确定性自适应调整残差范围"""
    def forward(self, obs):
        features = self.feature_net(obs)
        residual = self.residual_head(features)
        
        # 预测不确定性
        uncertainty = self.uncertainty_head(features)
        
        # 不确定性高时减小残差
        adaptive_scale = self.base_scale * torch.sigmoid(-uncertainty)
        
        residual = torch.tanh(residual) * adaptive_scale
        return residual
```

### 5.2 多模态残差
```python
# 不同任务使用不同残差头
class MultiModalResidual(nn.Module):
    def __init__(self, num_modes):
        self.residual_heads = nn.ModuleList([
            ResidualHead() for _ in range(num_modes)
        ])
        self.mode_selector = ModeSelector()
    
    def forward(self, obs):
        mode = self.mode_selector(obs)
        residual = sum(w * head(obs) for w, head in zip(mode, self.residual_heads))
        return residual
```

### 5.3 残差正则化
```python
# 添加 L2 正则鼓励小残差
loss = policy_loss + lambda_residual * (residual ** 2).mean()
```
