# 球类运动控制 (Ball Sports Control)

## 1. 算法原理

### 1.1 论文引用

**羽毛球项目**:
- 论文: "Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning" (Unitree, 2024)
- 核心技术: 三阶段课程学习 + EKF 轨迹预测

**乒乓球项目**:
- 论文: "Towards Versatile Humanoid Table Tennis: Unified RL with Prediction Augmentation"
- 核心技术: 双预测器架构 (学习预测器 + 物理预测器)

### 1.2 核心思想

#### 全身协调控制

球类运动需要人形机器人协调控制：
```
下肢: 步法控制 (Footwork)
   - 场地移动和定位
   - 平衡保持
   - 击球位置优化

上肢: 挥拍控制 (Swing)
   - 击球时机
   - 击球力度和方向
   - 拍面角度控制

核心挑战: 全身协调 (全身23-29 DOF同时控制)
```

#### 球类物理建模

**羽毛球空气动力学**:
```python
# 羽毛球独特物理特性
class ShuttlecockPhysics:
    """
    羽毛球空气动力学模型
    """
    def apply_aerodynamics(self, velocity, orientation):
        # 1. 高速时: 羽毛压缩，阻力小 (飞行远)
        # 2. 低速时: 羽毛展开，阻力大 (减速快)
        # 3. 翻转效应: 飞行中会翻转
        
        speed = np.linalg.norm(velocity)
        
        # 阻力系数随速度变化
        if speed > 10:  # 高速
            cd = 0.35  # 较小阻力
        else:  # 低速
            cd = 0.8   # 大阻力
        
        # 空气阻力
        drag_force = -0.5 * rho * cd * A * speed * velocity
        
        # 翻转力矩 (羽毛球特有)
        if speed < 5:
            flip_torque = self.calculate_flip_torque(orientation)
        
        return drag_force, flip_torque
```

**乒乓球物理**:
```python
class TableTennisPhysics:
    """
    乒乓球物理模型
    """
    def apply_physics(self, state, dt):
        # 重力
        gravity = np.array([0, 0, -9.81])
        
        # 空气阻力 (较小)
        drag = -0.01 * state.velocity * np.linalg.norm(state.velocity)
        
        # 马格努斯效应 (旋转导致)
        magnus_force = self.calculate_magnus_force(
            state.velocity, state.spin
        )
        
        # 更新状态
        acceleration = gravity + drag + magnus_force
        state.velocity += acceleration * dt
        state.position += state.velocity * dt
```

### 1.3 三阶段课程学习 (羽毛球)

```
阶段 1: 步法获取 (Footwork Acquisition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标: 学习基本的场地移动和定位
输入: 羽毛球位置 (简化)
输出: 下肢关节命令 (6 DOF)
奖励: 位置到达 + 平衡
冻结: 上肢关节

阶段 2: 挥拍生成 (Swing Generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标: 学习精确的球拍挥动动作
输入: 羽毛球轨迹 + 击球点
输出: 上肢关节命令 (6 DOF)
奖励: 击中奖励 + 击球质量
解冻: 部分下肢用于平衡

阶段 3: 任务精炼 (Task Refinement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标: 全身协调优化
输入: 完整观测 (球 + 自身状态)
输出: 全身关节命令 (12+ DOF)
奖励: 综合任务奖励
全部解冻: 全身协调控制
```

### 1.4 双预测器架构 (乒乓球)

```
┌─────────────────────────────────────────────────────────────┐
│                    双预测器系统                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  学习预测器 (Learned Predictor)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 输入: 最近球位置历史 (5-10个点)                       │   │
│  │ 网络: 小型MLP或LSTM                                    │   │
│  │ 输出: 未来球状态估计                                   │   │
│  │ 用途: Policy 观测增强                                  │   │
│  │ 特点: 快速、适应噪声、端到端学习                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│                    策略输入                                 │
│                                                             │
│  物理预测器 (Physics Predictor)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 输入: 当前球状态 + 物理模型                            │   │
│  │ 方法: 数值积分 (龙格-库塔等)                           │   │
│  │ 输出: 精确未来轨迹                                     │   │
│  │ 用途: 训练奖励计算                                     │   │
│  │ 特点: 精确、可解释、计算量稍大                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│                    奖励增强                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 使用场景

### 2.1 适用任务

| 任务类型 | 适用性 | 说明 |
|---------|-------|------|
| **羽毛球击打** | ⭐⭐⭐⭐⭐ | 三阶段课程学习成熟方案 |
| **乒乓球对打** | ⭐⭐⭐⭐⭐ | 双预测器架构高性能 |
| **网球击球** | ⭐⭐⭐⭐ | 类似羽毛球，需要更大力量 |
| **棒球击打** | ⭐⭐⭐ | 需要更高精度和力量 |
| **多球追踪** | ⭐⭐⭐ | 需要多目标处理能力 |
| **移动接球** | ⭐⭐⭐⭐ | 结合步法规划和接球 |

### 2.2 限制与注意事项

#### 感知要求
- **球位置跟踪**: 需要精确的视觉或传感器系统
- **实时性**: 球速快 (羽毛球可达 100m/s+)，需要低延迟感知
- **预测精度**: 预测误差会累积，影响击球时机

#### 物理仿真挑战
- **空气动力学**: 羽毛球的复杂空气动力学需要精确建模
- **接触建模**: 球拍-球碰撞需要高频仿真 (通常 1000Hz+)
- **摩擦系数**: 球拍表面材质影响球的旋转和反弹

#### 控制复杂度
- **全身 DOF**: 23-29 DOF 同时控制，动作空间高维
- **快速动作**: 挥拍动作需要在 0.1-0.3 秒内完成
- **精确时机**: 击球时机误差需 < 10ms

### 2.3 与其他运动控制对比

| 控制类型 | 时间要求 | 精度要求 | 全身协调 | 预测需求 |
|---------|---------|---------|---------|---------|
| **球类运动** | 极高 (100ms) | 极高 (cm级) | 高 | 高 |
| **行走** | 低 (秒级) | 中 | 高 | 低 |
| **操作** | 中 (秒级) | 高 (mm级) | 中 | 低 |
| **跳跃** | 高 (500ms) | 中 | 高 | 中 |

---

## 3. 代码示例

### 3.1 最小可运行示例

```python
import numpy as np
import genesis as gs
from typing import Dict, Tuple

# ==================== 球类物理模型 ====================

class BallPhysics:
    """通用球类物理模型"""
    
    def __init__(self, ball_type: str = 'shuttlecock'):
        self.ball_type = ball_type
        
        if ball_type == 'shuttlecock':
            self.mass = 0.005  # 5g
            self.drag_coeff_high = 0.35   # 高速阻力系数
            self.drag_coeff_low = 0.8     # 低速阻力系数
            self.flip_threshold = 5.0     # 翻转速度阈值
        elif ball_type == 'table_tennis':
            self.mass = 0.0027  # 2.7g
            self.drag_coeff = 0.4
            self.radius = 0.02  # 2cm
    
    def apply_dynamics(self, pos, vel, dt):
        """应用物理动力学"""
        speed = np.linalg.norm(vel)
        
        if self.ball_type == 'shuttlecock':
            # 变阻力系数
            cd = self.drag_coeff_high if speed > 10 else self.drag_coeff_low
            
            # 空气阻力 (与速度方向相反)
            drag = -0.5 * 1.225 * cd * 0.001 * speed * vel / self.mass
            
            # 翻转效应 (低速时)
            if speed < self.flip_threshold:
                # 添加随机翻转扰动
                flip = np.random.normal(0, 0.5, 3)
                drag += flip
        else:
            # 乒乓球: 恒定阻力
            drag = -0.5 * 1.225 * self.drag_coeff * np.pi * self.radius**2 * speed * vel / self.mass
        
        # 重力
        gravity = np.array([0, 0, -9.81])
        
        # 更新
        acceleration = gravity + drag
        new_vel = vel + acceleration * dt
        new_pos = pos + new_vel * dt
        
        return new_pos, new_vel


# ==================== EKF 轨迹预测器 ====================

class BallTrajectoryPredictor:
    """
    扩展卡尔曼滤波 (EKF) 预测器
    用于羽毛球/乒乓球轨迹预测
    """
    
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # 状态: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 0.1
        
        # 过程噪声
        self.Q = np.eye(6) * 0.01
        self.Q[3:, 3:] *= 0.1  # 速度噪声更大
        
        # 测量噪声
        self.R = np.eye(3) * 0.05
    
    def predict(self):
        """预测步骤"""
        # 状态转移 (假设匀加速)
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 预测状态
        self.state = F @ self.state
        
        # 预测协方差
        self.covariance = F @ self.covariance @ F.T + self.Q
        
        return self.state[:3], self.state[3:]
    
    def update(self, measurement):
        """更新步骤"""
        # 测量矩阵 (只观测位置)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        
        # 卡尔曼增益
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # 更新状态
        innovation = measurement - H @ self.state
        self.state = self.state + K @ innovation
        
        # 更新协方差
        I_KH = np.eye(6) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T
    
    def predict_landing(self, court_height=0):
        """预测落地点"""
        pos = self.state[:3].copy()
        vel = self.state[3:].copy()
        
        # 数值积分预测轨迹
        dt = 0.01
        while pos[2] > court_height and vel[2] < 0:
            # 简化物理
            pos += vel * dt
            vel[2] -= 9.81 * dt
        
        return pos[:2]  # 返回 (x, y) 落地点


# ==================== Genesis 环境示例 ====================

class BallSportEnv:
    """
    球类运动环境基类
    """
    
    def __init__(
        self,
        ball_type: str = 'shuttlecock',
        num_envs: int = 1,
        headless: bool = False
    ):
        self.ball_type = ball_type
        self.num_envs = num_envs
        
        # 初始化 Genesis
        gs.init(backend=gs.cuda)
        
        # 创建场景
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(6, 6, 4),
                camera_lookat=(0, 0, 1),
            ) if not headless else None,
            show_viewer=not headless
        )
        
        # 创建场地
        self._create_court()
        
        # 创建机器人 (人形)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/humanoid/humanoid.xml')
        )
        
        # 创建球
        self.ball = self._create_ball()
        
        # 预测器
        self.predictor = BallTrajectoryPredictor()
        
        # 球拍状态
        self.racket_pos = np.zeros(3)
        self.racket_vel = np.zeros(3)
        self.racket_normal = np.array([1, 0, 0])
        
        self.scene.build()
    
    def _create_court(self):
        """创建场地"""
        # 地面
        self.scene.add_entity(gs.morphs.Plane())
        
        if self.ball_type == 'badminton':
            # 羽毛球场地边界
            self.court_bounds = {
                'x_min': -6.7, 'x_max': 6.7,
                'y_min': -2.59, 'y_max': 2.59
            }
        else:
            # 乒乓球桌
            self.table_height = 0.76
    
    def _create_ball(self):
        """创建球"""
        ball_entity = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.05 if self.ball_type == 'shuttlecock' else 0.02)
        )
        return ball_entity
    
    def get_observation(self) -> np.ndarray:
        """获取观测 (包含预测信息)"""
        # 1. 机器人状态
        joint_pos = self.robot.get_dofs_position().cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()
        base_pos = self.robot.get_pos().cpu().numpy()
        
        # 2. 球状态
        ball_pos = self.ball.get_pos().cpu().numpy()
        ball_vel = self.ball.get_vel().cpu().numpy()
        
        # 3. 预测信息
        predicted_landing = self.predictor.predict_landing()
        time_to_land = self._estimate_time_to_land(ball_pos, ball_vel)
        
        # 4. 组合观测
        obs = np.concatenate([
            joint_pos.flatten()[:12],
            joint_vel.flatten()[:12],
            base_pos.flatten()[:3],
            ball_pos.flatten()[:3],
            ball_vel.flatten()[:3],
            predicted_landing.flatten()[:2],
            [time_to_land]
        ])
        
        return obs.astype(np.float32)
    
    def _estimate_time_to_land(self, pos, vel):
        """估计落地时间"""
        if vel[2] >= 0:
            return 10.0  # 上升中
        
        # 简化计算: h = v0*t - 0.5*g*t^2
        # 解: 0.5*g*t^2 - v0*t - h = 0
        h = pos[2]
        v0 = -vel[2]  # 向下为正
        g = 9.81
        
        discriminant = v0**2 + 2*g*h
        if discriminant < 0:
            return 10.0
        
        t = (v0 + np.sqrt(discriminant)) / g
        return t
    
    def check_hit(self) -> Dict:
        """检查是否击中球"""
        ball_pos = self.ball.get_pos().cpu().numpy()
        
        # 球拍-球距离
        distance = np.linalg.norm(ball_pos - self.racket_pos)
        hit_threshold = 0.15  # 15cm
        
        hit_info = {'hit': False}
        
        if distance < hit_threshold:
            # 计算击球效果
            racket_speed = np.linalg.norm(self.racket_vel)
            
            # 击球方向 (沿球拍法向)
            hit_direction = self.racket_normal
            hit_speed = max(10.0, racket_speed * 1.5)
            
            hit_info = {
                'hit': True,
                'speed': hit_speed,
                'direction': hit_direction,
                'position': self.racket_pos.copy()
            }
        
        return hit_info
    
    def compute_reward(self, hit_info: Dict) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 击中奖励
        if hit_info['hit']:
            reward += 10.0
            
            # 击球速度奖励
            reward += min(hit_info['speed'] / 20.0, 1.0) * 5.0
            
            # 落点奖励 (预测)
            predicted_landing = self.predictor.predict_landing()
            target_landing = np.array([3.0, 0.0])  # 对方场地中心
            landing_dist = np.linalg.norm(predicted_landing - target_landing)
            reward += np.exp(-landing_dist / 2.0) * 5.0
        
        # 2. 步法奖励
        ball_pos = self.ball.get_pos().cpu().numpy()
        robot_pos = self.robot.get_pos().cpu().numpy()
        distance_to_ball = np.linalg.norm(robot_pos[:2] - ball_pos[:2])
        reward += np.exp(-distance_to_ball) * 2.0
        
        # 3. 平衡奖励
        torso_height = robot_pos[2]
        if torso_height > 0.8:
            reward += 1.0
        
        return reward


# ==================== 课程学习训练 ====================

def train_with_curriculum():
    """使用课程学习训练"""
    
    # 阶段 1: 步法
    env_stage1 = BallSportEnv(curriculum_stage=1)
    env_stage1.freeze_joints(['shoulder', 'elbow', 'wrist'])  # 冻结上肢
    
    policy_stage1 = PPO(obs_dim=50, action_dim=6)  # 只控制下肢
    train(policy_stage1, env_stage1, episodes=1000)
    
    # 阶段 2: 挥拍
    env_stage2 = BallSportEnv(curriculum_stage=2)
    # 上肢解冻，部分下肢用于平衡
    
    policy_stage2 = PPO(obs_dim=60, action_dim=6)  # 只控制上肢
    # 使用阶段1的预训练权重
    policy_stage2.load_legs(policy_stage1)
    train(policy_stage2, env_stage2, episodes=1000)
    
    # 阶段 3: 全身协调
    env_stage3 = BallSportEnv(curriculum_stage=3)
    # 全部解冻
    
    policy_stage3 = PPO(obs_dim=80, action_dim=12)  # 全身控制
    # 使用阶段1和2的预训练权重
    policy_stage3.load_legs(policy_stage1)
    policy_stage3.load_arms(policy_stage2)
    train(policy_stage3, env_stage3, episodes=2000)
```

---

## 4. 超参数指南

### 4.1 物理仿真参数

| 参数 | 羽毛球 | 乒乓球 | 说明 |
|-----|-------|-------|------|
| `sim_freq` | 1000 Hz | 1000 Hz | 需要高频确保接触精度 |
| `ball_mass` | 5g | 2.7g | 实际质量 |
| `drag_coeff_high` | 0.35 | 0.4 | 高速阻力系数 |
| `drag_coeff_low` | 0.8 | - | 羽毛球特有低速大阻力 |
| `restitution` | 0.6 | 0.8 | 反弹系数 |

### 4.2 预测器参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `ekf_process_noise` | 0.01 | 过程噪声 Q |
| `ekf_measurement_noise` | 0.05 | 测量噪声 R |
| `prediction_horizon` | 1-2s | 预测时间范围 |
| `history_length` | 5-10 | 历史点数量 |

### 4.3 课程学习参数

```python
# 阶段转换条件
curriculum_config = {
    'stage1': {
        'min_success_rate': 0.7,  # 70% 成功率进入下一阶段
        'max_episodes': 1000
    },
    'stage2': {
        'min_success_rate': 0.6,
        'max_episodes': 1000
    },
    'stage3': {
        'target_success_rate': 0.9
    }
}

# 冻结关节配置
frozen_joints = {
    'stage1': ['shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r'],
    'stage2': ['hip_l', 'hip_r', 'knee_l', 'knee_r'],  # 部分冻结
    'stage3': []  # 全部解冻
}
```

### 4.4 奖励函数权重

```python
# 羽毛球奖励权重
badminton_rewards = {
    'hit': 10.0,           # 击中球
    'hit_speed': 5.0,      # 击球速度
    'landing': 5.0,        # 落点精度
    'position': 2.0,       # 位置到达
    'balance': 1.0,        # 平衡保持
    'energy': -0.0005      # 能量惩罚
}

# 乒乓球奖励权重
table_tennis_rewards = {
    'hit': 10.0,
    'rally': 2.0,          # 连续对打奖励
    'table_hit': 5.0,      # 击中对方桌面
    'footwork': 2.0,       # 步法到位
    'posture': 1.0
}
```

### 4.5 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| **经常打空** | 预测不准或时机错误 | 提高仿真频率，检查EKF参数 |
| **击球无力** | 动作幅度不够 | 增大动作范围奖励，调整PD增益 |
| **失去平衡** | 步法与挥拍不协调 | 加强课程学习阶段2-3过渡 |
| **球速过快跟踪不上** | 感知延迟 | 使用预测增强观测，提高控制频率 |
| **球轨迹不自然** | 物理参数错误 | 校准空气动力学参数 |

---

## 5. 进阶技巧

### 5.1 多球处理
```python
class MultiBallTracker:
    """多球跟踪器"""
    def __init__(self, max_balls=3):
        self.trackers = [BallTrajectoryPredictor() for _ in range(max_balls)]
        self.ball_ids = {}
    
    def update(self, detections):
        # 使用匈牙利算法匹配检测和跟踪
        matched = self.match_detections(detections)
        for det, tracker in matched:
            tracker.update(det)
```

### 5.2 自适应击球策略
```python
def select_shot_type(self, ball_state, robot_state):
    """根据情况选择击球方式"""
    time_available = self.estimate_reaction_time(ball_state)
    
    if time_available > 0.5:
        return 'smash'  # 时间充足: 扣杀
    elif time_available > 0.3:
        return 'clear'  # 中等时间: 高远球
    else:
        return 'defensive'  # 时间紧迫: 防守
```

### 5.3 对手建模
```python
class OpponentModel:
    """学习对手的击球习惯"""
    def __init__(self):
        self.shot_history = []
        self.preferred_locations = {}
    
    def predict_opponent_shot(self, opponent_position):
        """预测对手可能的回球方向"""
        # 基于历史统计
        return self.preferred_locations.get(opponent_position, default_distribution)
```
