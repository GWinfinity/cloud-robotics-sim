# Badminton Environment Plugin

来源: [genesis-humanoid-badminton](https://github.com/Genesis-Embodied-AI/Genesis)

## 核心功能

人形机器人羽毛球环境，实现：
- **三阶段课程学习**: 步法 → 挥拍 → 全身协调
- **羽毛球物理**: 空气动力学模型 (高速低阻力/低速高阻力/翻转)
- **EKF 预测**: 扩展卡尔曼滤波轨迹预测
- **全身控制**: 12+ DOF 同时控制

**架构:**
```
三阶段课程学习:
Stage 1: 步法获取 (Footwork)
  - 冻结上肢，只训练下肢移动
  - 目标: 到达最优击球位置

Stage 2: 挥拍生成 (Swing)
  - 冻结下肢，训练上肢挥拍
  - 目标: 准确击中羽毛球

Stage 3: 全身协调 (Coordination)
  - 全身关节同时控制
  - 目标: 完成完整击球动作
```

## 快速开始

```python
from cloud_robotics_sim.plugins.envs.badminton import BadmintonEnv

# 创建环境 (Stage 1: 步法训练)
env = BadmintonEnv(
    config_path="configs/stage1_footwork.yaml",
    num_envs=1,
    curriculum_stage=1
)

# 训练循环
obs = env.reset()
for step in range(1000):
    action = policy.predict(obs)
    obs, reward, done, info = env.step(action)
    
    if info['hit']:
        print(f"Hit! Speed: {info['hit_speed']:.2f} m/s")
```

## 算法原理

### 三阶段课程学习

```python
# Stage 1: 冻结上肢关节
frozen_joints = ['shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', ...]
action_dim = 6  # 只控制下肢

# Stage 2: 冻结下肢关节  
frozen_joints = ['hip_l', 'hip_r', 'knee_l', 'knee_r', ...]
action_dim = 6  # 只控制上肢

# Stage 3: 全身控制
frozen_joints = []
action_dim = 12  # 全身关节
```

### 羽毛球物理模型

```python
# 空气动力学特性
if speed > 10 m/s:    # 高速
    drag_coeff = 0.35  # 羽毛压缩，阻力小
else:                 # 低速
    drag_coeff = 0.8   # 羽毛展开，阻力大

# 翻转效应 (速度 < 5 m/s 时)
if speed < 5:
    apply_flip_torque()
```

### EKF 轨迹预测

```
状态: [x, y, z, vx, vy, vz]
观测: 球位置 (视觉/传感器)
预测: 未来轨迹用于击球决策
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `curriculum_stage` | int | 1 | 课程阶段 (1, 2, 3) |
| `num_envs` | int | 1 | 并行环境数 |
| `dt` | float | 0.01 | 仿真步长 (s) |
| `episode_length` | int | 1000 | 最大回合长度 |

### 课程阶段配置

```yaml
# Stage 1: 步法
stage1:
  frozen_joints: ['upper_body']
  rewards:
    position: {weight: 2.0}      # 位置到达奖励
    balance: {weight: 1.0}        # 平衡奖励

# Stage 2: 挥拍
stage2:
  frozen_joints: ['lower_body']
  rewards:
    hit: {weight: 10.0}           # 击中奖励
    hit_speed: {weight: 5.0}      # 击球速度奖励

# Stage 3: 全身协调
stage3:
  frozen_joints: []
  rewards:
    hit: {weight: 10.0}
    landing: {weight: 5.0}        # 落点精度奖励
    rally: {weight: 2.0}          # 连续击球奖励
```

## 奖励函数

| 奖励类型 | 权重 | 说明 |
|---------|------|------|
| `hit` | 10.0 | 击中羽毛球 |
| `hit_speed` | 5.0 | 击球速度 (鼓励大力击球) |
| `landing` | 5.0 | 落点精度 (落到对方场地) |
| `position` | 2.0 | 步法到位 (Stage 1) |
| `balance` | 1.0 | 保持平衡 |
| `energy` | -0.0005 | 能量惩罚 |

## 文件结构

```
badminton/
├── core/
│   ├── badminton_env.py       # 主环境
│   ├── shuttlecock.py         # 羽毛球物理
│   ├── curriculum.py          # 课程学习
│   ├── ekf.py                 # EKF 预测器
│   └── rewards.py             # 奖励函数
├── configs/
│   ├── stage1_footwork.yaml
│   ├── stage2_swing.yaml
│   └── stage3_full.yaml
├── examples/
│   ├── basic_usage.py
│   ├── ab_test.py
│   └── train_curriculum.py
├── README.md
└── plugin.yaml
```

## 使用示例

### 三阶段训练

```python
from cloud_robotics_sim.plugins.envs.badminton import BadmintonEnv

# Stage 1: 步法训练 (1000 episodes)
env = BadmintonEnv(curriculum_stage=1)
train(env, episodes=1000)

# Stage 2: 挥拍训练 (1000 episodes)
env.set_curriculum_stage(2)
train(env, episodes=1000)

# Stage 3: 全身协调 (2000 episodes)
env.set_curriculum_stage(3)
train(env, episodes=2000)
```

### EKF 预测

```python
from cloud_robotics_sim.plugins.envs.badminton import EKFPredictor

predictor = EKFPredictor(dt=0.01)

# 更新观测
predictor.update(ball_position)

# 预测未来轨迹
predicted_pos, predicted_vel = predictor.predict(steps=10)

# 预测落点
landing_point = predictor.predict_landing()
```

## 关键实现

### 击球检测

```python
def check_hit(self):
    # 球拍-球距离
    distance = np.linalg.norm(ball_pos - racket_pos)
    
    if distance < hit_threshold (0.15m):
        # 计算击球效果
        hit_direction = racket_normal
        hit_speed = racket_speed * 1.5
        
        # 应用击球
        ball.apply_hit(hit_direction, hit_speed)
        return True
    
    return False
```

### 课程切换条件

```python
def check_stage_transition(self):
    if self.stage == 1:
        # Stage 1 → 2: 步法成功率 > 70%
        if success_rate > 0.7:
            self.set_curriculum_stage(2)
    
    elif self.stage == 2:
        # Stage 2 → 3: 击球成功率 > 60%
        if hit_rate > 0.6:
            self.set_curriculum_stage(3)
```

## Changelog

- 2024-03-15: 从 genesis-humanoid-badminton 迁移
- 添加 EKF 预测器
- 完善课程学习系统

## 参考

- Paper: "Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning"
- EKF: 扩展卡尔曼滤波轨迹预测
- Physics: 羽毛球空气动力学模型
