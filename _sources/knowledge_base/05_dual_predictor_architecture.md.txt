# 双预测器架构 (Dual Predictor Architecture)

## 1. 算法原理

### 1.1 论文引用

**核心论文**: "Towards Versatile Humanoid Table Tennis: Unified RL with Prediction Augmentation" (2024)

**相关论文**:
- "FIFA: Future-aware Inverse Dynamics for Long-horizon Prediction"
- "Deep Learning for Real-time Physical Prediction"

### 1.2 核心思想

#### 问题背景
单预测器系统面临的问题：
- **纯学习预测器**: 对分布外数据泛化差，但计算快
- **纯物理预测器**: 精确但需要完整状态，计算慢
- **单一预测器**: 难以同时满足精度和速度要求

#### 双预测器设计理念

```
┌─────────────────────────────────────────────────────────────┐
│                    双预测器设计原则                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  学习预测器 (Learned Predictor)                              │
│  ├── 用途: 策略观测增强                                     │
│  ├── 输入: 历史轨迹 (部分观测)                               │
│  ├── 输出: 快速未来估计                                     │
│  └── 特点: 鲁棒、快速、端到端                               │
│                                                             │
│  物理预测器 (Physics Predictor)                              │
│  ├── 用途: 奖励计算                                         │
│  ├── 输入: 完整状态 + 物理模型                               │
│  ├── 输出: 精确轨迹                                         │
│  └── 特点: 精确、可解释、一致                               │
│                                                             │
│  核心洞见:                                                  │
│  - 策略不需要完美的预测，只需要有用的预测                     │
│  - 奖励计算需要精确的预测来提供可靠的信号                     │
│  - 两者解耦可以分别优化                                      │
└─────────────────────────────────────────────────────────────┘
```

#### 数学形式

```
学习预测器 (快速/鲁棒):
ẑ_{t+k} = f_θ(z_t, z_{t-1}, ..., z_{t-H})
其中 z_t 是历史观测特征

物理预测器 (精确/一致):
ŝ_{t+k} = g(s_t, a_t, a_{t+1}, ..., a_{t+k-1}; Φ)
其中 s_t 是完整物理状态, Φ 是物理参数

组合:
- Policy 输入: [obs_t, ẑ_{t+1}, ..., ẑ_{t+K}]
- 奖励计算: r_t = h(ŝ_{t+1}, ..., ŝ_{t+K}, goal)
```

### 1.3 架构详解

#### 学习预测器

```python
class LearnedPredictor(nn.Module):
    """
    学习预测器
    
    输入: 历史轨迹 (位置序列)
    输出: 多步未来预测
    """
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, horizon=10):
        super().__init__()
        
        # LSTM 编码历史
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # MLP 解码未来
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * input_dim)  # 预测未来 K 步
        )
    
    def forward(self, trajectory_history):
        """
        Args:
            trajectory_history: [B, H, 3] 历史位置序列
        Returns:
            predictions: [B, K, 3] 未来 K 步预测
        """
        # LSTM 编码
        lstm_out, _ = self.lstm(trajectory_history)
        hidden = lstm_out[:, -1]  # 取最后时刻
        
        # 解码预测
        predictions = self.decoder(hidden)
        predictions = predictions.view(-1, self.horizon, self.input_dim)
        
        return predictions
```

#### 物理预测器

```python
class PhysicsPredictor:
    """
    物理预测器
    
    使用数值积分模拟未来轨迹
    """
    def __init__(self, dt=0.01, horizon=50):
        self.dt = dt
        self.horizon = horizon
    
    def predict(self, initial_state, policy_actions, physics_model):
        """
        使用物理模型预测轨迹
        
        Args:
            initial_state: 初始状态 (位置、速度、旋转等)
            policy_actions: 策略动作序列 [horizon, action_dim]
            physics_model: 物理模型 (重力、阻力、碰撞等)
        
        Returns:
            trajectory: 预测的完整状态序列
        """
        states = [initial_state]
        state = initial_state.copy()
        
        for action in policy_actions:
            # 应用动作 (如机器人击球)
            state = self.apply_action(state, action)
            
            # 物理仿真步
            state = physics_model.step(state, self.dt)
            
            states.append(state.copy())
        
        return np.array(states)
    
    def get_prediction_for_reward(self, ball_state, robot_state):
        """
        为奖励计算生成预测
        
        预测球轨迹并计算关键指标:
        - 击球概率
        - 最佳击球点
        - 落点预测
        """
        # 数值积分预测球轨迹
        trajectory = self.simulate_ball_trajectory(ball_state)
        
        # 计算机器人可到达区域
        reachable = self.compute_reachable_region(robot_state)
        
        # 找到最佳击球点
        hit_point, hit_prob = self.find_optimal_hit_point(trajectory, reachable)
        
        # 预测落点
        landing_point = self.predict_landing(trajectory)
        
        return {
            'hit_probability': hit_prob,
            'contact_point': hit_point,
            'landing_point': landing_point,
            'trajectory': trajectory
        }
```

### 1.4 完整系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练时 (Training)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   环境状态 s_t                                                      │
│        │                                                            │
│        ├──→ [学习预测器] ──→ ẑ_{t+1:t+K} ──→ 策略观测 (Policy Input) │
│        │                  (快速、鲁棒)                                │
│        │                                                            │
│        └──→ [物理预测器] ──→ ŝ_{t+1:t+K} ──→ 奖励计算 (Reward)      │
│                       (精确、一致)                                    │
│                                                                     │
│   奖励设计:                                                         │
│   - 击球概率奖励: 基于物理预测的 hit_probability                    │
│   - 步法奖励: 基于 contact_point 计算距离                           │
│   - 落点奖励: 基于 landing_point 计算精度                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        推理时 (Inference)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   环境状态 s_t                                                      │
│        │                                                            │
│        └──→ [学习预测器] ──→ ẑ_{t+1:t+K}                            │
│                          (快速推理，< 1ms)                            │
│                             ↓                                       │
│                        策略网络                                      │
│                             ↓                                       │
│                        动作 a_t                                      │
│                                                                     │
│   注意: 推理时不需要物理预测器，保证实时性                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 使用场景

### 2.1 适用任务

| 任务类型 | 适用性 | 说明 |
|---------|-------|------|
| **乒乓球对打** | ⭐⭐⭐⭐⭐ | 原始设计目标，高速球预测 |
| **羽毛球击打** | ⭐⭐⭐⭐⭐ | 复杂空气动力学，需要精确预测 |
| **棒球击打** | ⭐⭐⭐⭐ | 高速投球，反应时间短 |
| **足球守门** | ⭐⭐⭐⭐ | 需要预测射门方向 |
| **避障移动** | ⭐⭐⭐ | 预测动态障碍物轨迹 |
| **多智能体** | ⭐⭐⭐⭐ | 预测其他智能体行为 |

### 2.2 应用条件

#### 需要满足的条件
1. **可预测的动力学**: 物体运动遵循物理规律
2. **部分可观测**: 系统无法直接获取完整状态
3. **快速推理需求**: 策略需要实时决策
4. **精确评估需求**: 奖励计算需要可靠的未来信息

#### 不适用场景
- 纯随机系统 (无法预测)
- 观测质量极差 (学习预测器无法工作)
- 物理模型未知且复杂 (物理预测器难以建模)

### 2.3 与其他预测方法对比

| 方法 | 推理速度 | 预测精度 | 泛化能力 | 适用场景 |
|-----|---------|---------|---------|---------|
| **双预测器** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 需要精度和速度兼顾 |
| **纯学习** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 观测质量高 |
| **纯物理** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 状态完全可观测 |
| **无预测** | ⭐⭐⭐⭐⭐ | - | - | 简单任务 |

---

## 3. 代码示例

### 3.1 最小可运行示例

```python
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

# ==================== 学习预测器 ====================

class LearnedTrajectoryPredictor(nn.Module):
    """
    学习轨迹预测器
    
    基于历史观测预测未来轨迹
    """
    def __init__(
        self,
        input_dim: int = 3,      # 位置 (x, y, z)
        hidden_dim: int = 128,
        num_layers: int = 2,
        history_len: int = 5,    # 历史长度
        prediction_horizon: int = 10  # 预测步数
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        
        # LSTM 编码器
        self.encoder = nn.LSTM(
            input_size=input_dim * 2,  # 位置 + 速度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, prediction_horizon * input_dim)
        )
    
    def forward(self, history_positions, history_velocities=None):
        """
        预测未来轨迹
        
        Args:
            history_positions: [B, H, 3] 历史位置
            history_velocities: [B, H, 3] 历史速度 (可选)
        
        Returns:
            predictions: [B, K, 3] 预测的未来位置
        """
        B, H, _ = history_positions.shape
        
        if history_velocities is None:
            # 计算数值速度
            history_velocities = torch.zeros_like(history_positions)
            history_velocities[:, 1:] = history_positions[:, 1:] - history_positions[:, :-1]
        
        # 拼接特征
        features = torch.cat([history_positions, history_velocities], dim=-1)
        
        # LSTM 编码
        lstm_out, (hidden, cell) = self.encoder(features)
        
        # 使用最后时刻的隐藏状态
        context = hidden[-1]  # [B, hidden_dim]
        
        # 预测未来
        pred = self.predictor(context)
        pred = pred.view(B, self.prediction_horizon, self.input_dim)
        
        # 残差连接: 假设匀速运动
        last_pos = history_positions[:, -1:, :]  # [B, 1, 3]
        last_vel = history_velocities[:, -1:, :]  # [B, 1, 3]
        
        time_steps = torch.arange(1, self.prediction_horizon + 1, 
                                   device=pred.device).float()
        time_steps = time_steps.view(1, -1, 1)  # [1, K, 1]
        
        # 基础预测: p_t = p_0 + v * t
        baseline = last_pos + last_vel * time_steps
        
        # 学习残差
        predictions = baseline + pred
        
        return predictions


# ==================== 物理预测器 ====================

class BallPhysicsPredictor:
    """
    物理球体预测器
    
    使用数值积分精确模拟球轨迹
    """
    def __init__(self, dt=0.01, gravity=-9.81, drag_coeff=0.4):
        self.dt = dt
        self.gravity = np.array([0, 0, gravity])
        self.drag_coeff = drag_coeff
    
    def predict_trajectory(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        horizon: int = 50
    ) -> np.ndarray:
        """
        预测球轨迹
        
        Args:
            initial_position: [3] 初始位置
            initial_velocity: [3] 初始速度
            horizon: 预测步数
        
        Returns:
            trajectory: [horizon+1, 3] 预测轨迹 (包含初始点)
        """
        positions = [initial_position.copy()]
        velocities = [initial_velocity.copy()]
        
        pos = initial_position.copy()
        vel = initial_velocity.copy()
        
        for _ in range(horizon):
            # 计算加速度
            speed = np.linalg.norm(vel)
            drag = -self.drag_coeff * speed * vel
            acc = self.gravity + drag
            
            # 欧拉积分
            vel = vel + acc * self.dt
            pos = pos + vel * self.dt
            
            # 地面碰撞检测
            if pos[2] < 0:
                pos[2] = 0
                vel[2] = -vel[2] * 0.8  # 反弹系数
            
            positions.append(pos.copy())
            velocities.append(vel.copy())
        
        return np.array(positions), np.array(velocities)
    
    def get_prediction_for_reward(
        self,
        ball_state: dict,
        robot_state: dict,
        court_bounds: dict = None
    ) -> dict:
        """
        为奖励计算生成预测信息
        
        Args:
            ball_state: {'position': [...], 'velocity': [...]}
            robot_state: {'position': [...], 'joint_angles': [...]}
        
        Returns:
            dict with keys:
                - 'hit_probability': 机器人能击中的概率
                - 'contact_point': 最佳击球点
                - 'time_to_contact': 到击球的时间
                - 'landing_point': 预测落点
        """
        # 预测球轨迹
        trajectory, velocities = self.predict_trajectory(
            ball_state['position'],
            ball_state['velocity'],
            horizon=100
        )
        
        # 计算机器人可到达区域
        robot_pos = robot_state['position']
        max_reach = 1.5  # 最大到达距离
        max_height = 2.5  # 最大高度
        
        # 找到可击中的点
        reachable_points = []
        for i, (pos, vel) in enumerate(zip(trajectory, velocities)):
            # 距离检查
            distance = np.linalg.norm(pos[:2] - robot_pos[:2])
            if distance < max_reach and pos[2] < max_height:
                # 球应该朝向机器人或处于可击范围
                if vel[0] < 5:  # 水平速度不太大
                    reachable_points.append((i, pos, vel))
        
        # 计算击球概率和最佳点
        if len(reachable_points) > 0:
            # 选择距离最近的点作为最佳击球点
            distances = [np.linalg.norm(p[1][:2] - robot_pos[:2]) 
                        for p in reachable_points]
            best_idx = np.argmin(distances)
            
            hit_point = reachable_points[best_idx][1]
            time_to_hit = reachable_points[best_idx][0] * self.dt
            hit_prob = np.exp(-distances[best_idx] / max_reach)
        else:
            hit_point = None
            time_to_hit = None
            hit_prob = 0.0
        
        # 预测落点
        for i, pos in enumerate(trajectory):
            if pos[2] < 0.1:  # 接近地面
                landing_point = pos[:2]
                break
        else:
            landing_point = trajectory[-1][:2]
        
        return {
            'hit_probability': hit_prob,
            'contact_point': hit_point,
            'time_to_contact': time_to_hit,
            'landing_point': landing_point,
            'trajectory': trajectory
        }


# ==================== 双预测器环境 ====================

class DualPredictorEnv:
    """
    使用双预测器架构的环境
    
    - 学习预测器: 为策略提供观测增强
    - 物理预测器: 为奖励计算提供精确预测
    """
    
    def __init__(
        self,
        use_learned_predictor: bool = True,
        use_physics_predictor: bool = True,
        history_len: int = 5,
        prediction_horizon: int = 10
    ):
        self.use_learned = use_learned_predictor
        self.use_physics = use_physics_predictor
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        
        # 初始化 Genesis
        gs.init(backend=gs.cuda)
        self.scene = gs.Scene()
        
        # 创建机器人和球
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/humanoid/humanoid.xml')
        )
        self.ball = self.scene.add_entity(gs.morphs.Sphere(radius=0.02))
        
        self.scene.build()
        
        # 初始化预测器
        if self.use_learned:
            self.learned_predictor = LearnedTrajectoryPredictor(
                history_len=history_len,
                prediction_horizon=prediction_horizon
            )
            self.learned_predictor.eval()
        
        if self.use_physics:
            self.physics_predictor = BallPhysicsPredictor()
        
        # 历史缓冲区
        self.ball_history = []
    
    def get_observation(self) -> dict:
        """获取观测 (包含学习预测器的输出)"""
        # 基础观测
        robot_obs = self._get_robot_obs()
        ball_obs = self._get_ball_obs()
        
        obs_dict = {
            'robot': robot_obs,
            'ball': ball_obs
        }
        
        # 学习预测器增强
        if self.use_learned and len(self.ball_history) >= self.history_len:
            # 准备历史
            history = np.array(self.ball_history[-self.history_len:])
            history_tensor = torch.FloatTensor(history).unsqueeze(0)
            
            with torch.no_grad():
                predictions = self.learned_predictor(history_tensor)
            
            # 添加到观测
            obs_dict['ball_predictions'] = predictions.squeeze(0).numpy()
        else:
            # 预测不可用，用零填充
            obs_dict['ball_predictions'] = np.zeros(
                (self.prediction_horizon, 3)
            )
        
        # 展平为向量 (用于 RL)
        obs_vector = self._flatten_obs(obs_dict)
        
        return obs_vector
    
    def compute_reward(self, action: np.ndarray) -> float:
        """计算奖励 (使用物理预测器)"""
        reward = 0.0
        
        if self.use_physics:
            # 获取当前状态
            ball_state = {
                'position': self.ball.get_pos().cpu().numpy(),
                'velocity': self.ball.get_vel().cpu().numpy()
            }
            robot_state = {
                'position': self.robot.get_pos().cpu().numpy()
            }
            
            # 物理预测
            pred = self.physics_predictor.get_prediction_for_reward(
                ball_state, robot_state
            )
            
            # 预测增强奖励
            # 1. 击球概率奖励
            reward += 2.0 * pred['hit_probability']
            
            # 2. 步法奖励 (如果知道击球点)
            if pred['contact_point'] is not None:
                robot_pos = robot_state['position']
                distance = np.linalg.norm(
                    pred['contact_point'][:2] - robot_pos[:2]
                )
                footwork_reward = np.exp(-distance)
                reward += 1.0 * footwork_reward
            
            # 3. 落点奖励
            target_landing = np.array([3.0, 0.0])  # 对方场地中心
            landing_dist = np.linalg.norm(
                pred['landing_point'] - target_landing
            )
            landing_reward = np.exp(-landing_dist / 2.0)
            reward += 5.0 * landing_reward
        
        # 基础奖励
        reward += self._compute_base_reward(action)
        
        return reward
    
    def step(self, action):
        """环境步进"""
        # 执行动作
        self.robot.control_dofs_position(action)
        self.scene.step()
        
        # 更新历史
        ball_pos = self.ball.get_pos().cpu().numpy()
        self.ball_history.append(ball_pos)
        
        # 限制历史长度
        if len(self.ball_history) > 100:
            self.ball_history.pop(0)
        
        # 获取观测和奖励
        obs = self.get_observation()
        reward = self.compute_reward(action)
        done = self._check_done()
        
        return obs, reward, done, {}


# ==================== 训练学习预测器 ====================

def train_learned_predictor():
    """训练学习预测器"""
    
    # 创建数据集 (使用物理模拟生成)
    dataset = TrajectoryDataset(
        num_sequences=10000,
        history_len=5,
        prediction_horizon=10
    )
    
    # 初始化预测器
    predictor = LearnedTrajectoryPredictor(
        history_len=5,
        prediction_horizon=10
    )
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    # 训练循环
    for epoch in range(100):
        total_loss = 0.0
        
        for batch in DataLoader(dataset, batch_size=32, shuffle=True):
            history = batch['history']  # [B, H, 3]
            target_future = batch['future']  # [B, K, 3]
            
            # 预测
            pred_future = predictor(history)
            
            # MSE 损失
            loss = nn.MSELoss()(pred_future, target_future)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataset):.4f}")
    
    # 保存
    torch.save(predictor.state_dict(), 'learned_predictor.pt')


# ==================== 完整训练流程 ====================

def train_with_dual_predictor():
    """使用双预测器架构训练 RL 策略"""
    
    # 创建环境
    env = DualPredictorEnv(
        use_learned_predictor=True,
        use_physics_predictor=True
    )
    
    # 创建策略 (观测包含预测信息)
    obs_dim = env.obs_dim  # 包含 ball_predictions
    policy = PPO(obs_dim=obs_dim, action_dim=env.action_dim)
    
    # 训练循环
    for episode in range(2000):
        obs = env.reset()
        episode_reward = 0.0
        
        for step in range(500):
            action = policy.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            policy.store_transition(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        policy.update()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

---

## 4. 超参数指南

### 4.1 学习预测器参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `history_len` | 5-10 | 历史序列长度 |
| `prediction_horizon` | 10-20 | 预测步数 (100-200ms) |
| `hidden_dim` | 128-256 | LSTM 隐藏层维度 |
| `num_layers` | 2 | LSTM 层数 |

#### 预测范围选择

```python
# 根据任务选择预测范围
if task == 'table_tennis':
    # 乒乓球速度快，需要短期精确预测
    history_len = 5      # 50ms 历史
    prediction_horizon = 10  # 100ms 预测
    
elif task == 'badminton':
    # 羽毛球速度慢但轨迹复杂，需要中长期预测
    history_len = 8      # 80ms 历史
    prediction_horizon = 20  # 200ms 预测
```

### 4.2 物理预测器参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `dt` | 0.001-0.01s | 仿真步长 (越小越精确) |
| `gravity` | -9.81 | 重力加速度 |
| `drag_coeff` | 0.3-0.8 | 阻力系数 (球类特定) |
| `restitution` | 0.6-0.9 | 反弹系数 |

#### 球类特定参数

```python
# 乒乓球
pingpong_params = {
    'mass': 0.0027,  # kg
    'radius': 0.02,  # m
    'drag_coeff': 0.4,
    'restitution': 0.8
}

# 羽毛球
badminton_params = {
    'mass': 0.005,  # kg
    'drag_coeff_high': 0.35,   # 高速
    'drag_coeff_low': 0.8,     # 低速
    'flip_speed': 5.0          # 翻转速度阈值
}
```

### 4.3 奖励权重

```python
# 预测增强奖励权重
reward_weights = {
    'hit_probability': 2.0,     # 击球概率奖励
    'footwork': 1.0,            # 步法到位奖励
    'landing': 5.0,             # 落点精度奖励 (通常最高)
    'base': 1.0                 # 基础奖励
}

# 预测置信度阈值
confidence_threshold = 0.5  # 预测置信度低时减小奖励
```

### 4.4 训练技巧

```python
# 学习预测器预训练
pretrain_config = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-5
}

# 预测器微调
finetune_config = {
    'finetune_freq': 1000,  # 每1000个episode微调一次
    'finetune_steps': 100,
    'lr': 1e-4  # 更低的学习率
}

# 在线适应
online_adaptation = {
    'adaptation_rate': 0.01,  # 在线适应速率
    'buffer_size': 1000       # 在线数据缓冲区
}
```

### 4.5 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| **学习预测器误差大** | 历史太短或网络容量不够 | 增加 history_len，增大 hidden_dim |
| **物理预测器与实际不符** | 物理参数错误 | 校准阻力系数、反弹系数 |
| **奖励信号弱** | 预测范围太长 | 缩短 prediction_horizon |
| **策略过度依赖预测** | 预测噪声大 | 添加预测不确定性估计，使用 dropout |
| **训练不稳定** | 两个预测器输出不一致 | 确保物理预测器使用 ground truth 初始化 |

---

## 5. 进阶技巧

### 5.1 不确定性估计
```python
class UncertaintyAwarePredictor(nn.Module):
    """输出预测和不确定性"""
    
    def forward(self, history):
        features = self.encoder(history)
        
        # 预测均值
        mean = self.mean_head(features)
        
        # 预测方差 (不确定性)
        log_var = self.var_head(features)
        
        return mean, torch.exp(log_var)

# 在观测中使用不确定性
obs = torch.cat([predictions, uncertainties], dim=-1)
```

### 5.2 多模态预测
```python
class MultiModalPredictor(nn.Module):
    """预测多种可能的未来"""
    
    def __init__(self, num_modes=3):
        self.mode_heads = nn.ModuleList([
            PredictorHead() for _ in range(num_modes)
        ])
        self.mode_weights = ModeSelector()
    
    def forward(self, history):
        # 为每种模式生成预测
        predictions = [head(history) for head in self.mode_heads]
        
        # 模式权重
        weights = self.mode_weights(history)
        
        return predictions, weights
```

### 5.3 注意力机制
```python
class AttentionPredictor(nn.Module):
    """使用注意力关注重要的历史帧"""
    
    def __init__(self):
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4
        )
    
    def forward(self, history):
        # 自注意力
        attended, weights = self.attention(history, history, history)
        
        # 使用注意力权重进行预测
        context = attended.mean(dim=1)
        prediction = self.decoder(context)
        
        return prediction, weights
```
