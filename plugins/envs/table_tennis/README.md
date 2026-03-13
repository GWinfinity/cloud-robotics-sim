# table_tennis Plugin

来源: [genesis-table-tennis](../../genesis-table-tennis)

基于 Genesis 引擎实现的人形机器人乒乓球环境，复现论文 "Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation"

## 核心功能

### 1. 统一全身控制
- **机器人**: Unitree G1 (29 DOF)
- **控制目标**: 手臂击球 + 腿部步法协调
- **策略类型**: 端到端统一策略

### 2. 双预测器架构
```
学习预测器 (Policy使用)
├── 输入: 最近球位置历史
├── 输出: 未来球状态估计  
└── 作用: 增强策略观测，实现主动决策

物理预测器 (训练使用)
├── 输入: 当前球状态 + 物理模型
├── 输出: 精确未来轨迹
└── 作用: 构建密集预测奖励
```

### 3. 预测增强奖励
- 击球奖励
- 落点奖励 (基于预测)
- 步法奖励
- 姿态奖励
- 能量效率惩罚

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.envs.table_tennis import TableTennisEnv
import yaml

# 加载配置
with open('configs/table_tennis.yaml') as f:
    config = yaml.safe_load(f)

# 创建环境
env = TableTennisEnv(
    config=config,
    num_envs=1,
    headless=False
)

# 重置环境
obs = env.reset()

# 运行回合
for step in range(1000):
    # 随机动作 (或策略推理)
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

### 训练

```python
from cloud_robotics_sim.plugins.envs.table_tennis import PPO, TableTennisEnv

# 创建环境和算法
env = TableTennisEnv(config)
ppo = PPO(env, config)

# 训练
ppo.train(total_timesteps=10_000_000)
```

## 算法原理

### 双预测器协同

1. **学习预测器**
   - 轻量级神经网络
   - 实时推理 (policy forward pass)
   - 提供球轨迹的隐式表示

2. **物理预测器**
   - 基于物理模型的数值积分
   - 精确计算未来状态
   - 用于奖励计算 (仅训练时使用)

### 统一策略架构

```
观测 (106维)
├── 球状态: 位置(3) + 速度(3) = 6维
├── 预测特征: 历史轨迹编码 = 60维
└── 本体感觉: 关节位置(20) + 速度(20) = 40维

策略网络
├── MLP Encoder: 106 → 256
├── LSTM (可选): 时序建模
└── 输出: 29维关节目标位置
```

### 奖励设计

```python
# 击球奖励
r_hit = w_hit * I_hit + w_speed * v_hit

# 预测增强奖励 (核心创新)
prediction = physics_predictor(state)
r_predictive = w_pred * prediction.hit_probability

# 步法奖励
r_footwork = w_foot * exp(-distance_to_contact)

# 姿态奖励
r_posture = w_posture * upright_bonus

# 总奖励
reward = r_hit + r_predictive + r_footwork + r_posture - energy_penalty
```

## 配置参数

### 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 1 | 并行环境数 |
| `episode_length` | 1000 | 最大回合长度 |
| `dt` | 0.02 | 仿真时间步 |

### 球物理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ball.mass` | 0.0027 | 球质量 (kg) |
| `ball.radius` | 0.02 | 球半径 (m) |
| `ball.drag_coeff` | 0.47 | 阻力系数 |
| `ball.restitution` | 0.85 | 反弹系数 |

### 奖励权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rewards.hit.weight` | 10.0 | 击球基础奖励 |
| `rewards.predictive.weight` | 5.0 | 预测增强奖励 |
| `rewards.footwork.weight` | 2.0 | 步法奖励 |
| `rewards.posture.weight` | 1.0 | 姿态奖励 |
| `rewards.energy_penalty` | -0.01 | 能量惩罚系数 |

## 性能指标

| 指标 | 数值 |
|------|------|
| 击球率 (Hit Rate) | ≥ 96% |
| 成功率 (Success Rate) | ≥ 92% |
| 连续击球 | 可连续多回合 |
| 部署 | Zero-shot 到 Booster T1 |

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架
- `train.py` - 训练脚本
- `eval.py` - 评估脚本

## 引用

```bibtex
@article{hu2025tabletennis,
  title={Towards Versatile Humanoid Table Tennis: 
         Unified Reinforcement Learning with Prediction Augmentation},
  author={Hu, Muqun and Chen, Wenxi and Li, Wenjing and others},
  journal={arXiv preprint arXiv:2509.21690},
  year={2025}
}
```

## Changelog

- **2026-03-13**: 从 genesis-table-tennis 迁移到 genesis-cloud-sim plugins
- **2026-02-09**: 初始版本，实现双预测器架构
