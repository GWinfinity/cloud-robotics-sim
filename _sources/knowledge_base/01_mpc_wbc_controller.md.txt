# MPC + WBC 全身控制器

## 1. 算法原理

### 1.1 论文引用

**MPC (Model Predictive Control)**
- 基于人形机器人动力学模型预测未来状态
- 参考论文: "Dynamic Walking on Compliant and Uneven Terrain using DCM and MPC" (ETH Zurich)
- 开源实现: OpenLoong Dynamics Control (上海青龙机器人)

**WBC (Whole-Body Control)**
- 全身优先级控制框架
- 参考论文: "A Whole-Body Control Framework for Humanoid Robots" (Stanford)
- 零空间投影实现多任务优先级

### 1.2 核心思想

#### MPC 核心思想
```
状态: X = [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz] (12维)
输入: U = [FL_x, FL_y, FL_z, τL_x, τL_y, τL_z, FR_x, FR_y, FR_z, τR_x, τR_y, τR_z] (12维)

优化目标:
min Σ||X_t - Xd_t||²_Q + Σ||U_t||²_R

约束:
- 动力学约束: X_{t+1} = A*X_t + B*U_t
- 接触力约束: Fz >= 0 (单侧约束)
- 摩擦锥约束: ||F_xy|| <= μ*Fz
```

#### WBC 核心思想
```
任务栈 (优先级从高到低):
1. 质心动量跟踪 (最高优先级)
2. 摆动脚轨迹跟踪
3. 上身姿态保持
4. 关节极限避免 (最低优先级)

零空间投影:
τ = J1^T * F1 + N1 * J2^T * F2 + N1*N2 * J3^T * F3 + ...

其中 N = I - J^T * (J * M^{-1} * J^T)^{-1} * J * M^{-1} 是零空间投影矩阵
```

### 1.3 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   高层规划器                          │
│         (生成期望质心轨迹 Xd, 落脚点)                  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│                   MPC 控制器                          │
│  ┌───────────────────────────────────────────────┐  │
│  │  预测模型: 线性化单刚体模型                      │  │
│  │  优化问题: QP (二次规划)                        │  │
│  │  输出: 期望接触力/力矩 [FL, FR]                  │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│                   WBC 控制器                          │
│  ┌───────────────────────────────────────────────┐  │
│  │  任务1: 质心加速度 (来自MPC力/力矩)              │  │
│  │  任务2: 摆动脚位置和速度                        │  │
│  │  任务3: 上身姿态                                │  │
│  │  输出: 关节力矩 τ                               │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│                   机器人动力学                         │
└─────────────────────────────────────────────────────┘
```

---

## 2. 使用场景

### 2.1 适用任务

| 任务类型 | 适用性 | 说明 |
|---------|-------|------|
| **平地行走** | ⭐⭐⭐⭐⭐ | 最成熟应用，稳定可靠 |
| **上下楼梯** | ⭐⭐⭐⭐ | 需要准确落脚点规划 |
| **崎岖地形** | ⭐⭐⭐ | 需要地形感知和自适应 |
| **跳跃** | ⭐⭐⭐⭐ | MPC预测助力起跳和落地 |
| **搬运物体** | ⭐⭐⭐ | 需要调整质心模型参数 |
| **被推动恢复** | ⭐⭐⭐⭐⭐ | 扰动抑制能力强 |

### 2.2 限制与注意事项

#### 计算限制
- **MPC 频率**: 50-100 Hz (QP求解需要时间)
- **WBC 频率**: 200-1000 Hz
- **预测步数 N**: 10-20 步 (更多步数计算量指数增长)

#### 模型限制
- 使用**单刚体模型**简化，忽略腿的质量
- 假设接触点已知且固定
- 对非常规接触 (滑倒、碰撞) 处理能力有限

#### 调参复杂度
- 需要调整 MPC 权重矩阵 Q, R
- 需要调整 WBC 任务权重
- 步态参数 (步频、步长) 需要精心调优

### 2.3 与其他控制器对比

| 控制器 | 优势 | 劣势 | 适用场景 |
|-------|------|------|---------|
| **MPC+WBC** | 物理约束处理、预测能力 | 计算量大、调参复杂 | 复杂地形、高性能要求 |
| **RL策略** | 端到端、适应性强 | 样本效率低、难解释 | 未知地形、数据充足 |
| **PD控制** | 简单、稳定 | 无预测能力 | 简单任务、快速部署 |

---

## 3. 代码示例

### 3.1 最小可运行示例

```python
import numpy as np
import genesis as gs
from cloud_robotics_sim.controllers import MPCController, WBCController

# ==================== 初始化 ====================
gs.init(backend=gs.cuda)

# 创建场景
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=20),
    show_viewer=True
)

# 添加地面
scene.add_entity(gs.morphs.Plane())

# 添加人形机器人 (以 Unitree H1 为例)
robot = scene.add_entity(
    gs.morphs.MJCF(file='xml/humanoid/h1.xml'),
    surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0))
)

scene.build()

# ==================== 控制器初始化 ====================
# MPC 控制器
mpc = MPCController(
    dt=0.005,
    horizon=10,      # 预测步数
    mass=77.35,      # 机器人质量
    inertia=np.diag([12.61, 11.15, 2.15])  # 惯性张量
)

# WBC 控制器
wbc = WBCController(
    robot=robot,
    dt=0.005,
    kp_com=100.0,    # 质心位置比例增益
    kd_com=20.0      # 质心位置微分增益
)

# 步态调度器
gait_scheduler = GaitScheduler(
    gait_type='trot',    # 步态类型: trot/walk/bound
    frequency=1.25       # 步频 1.25 Hz
)

# ==================== 主循环 ====================
for t in range(10000):
    # 1. 获取当前状态
    base_pos = robot.get_pos().cpu().numpy()
    base_quat = robot.get_quat().cpu().numpy()
    base_vel = robot.get_vel().cpu().numpy()
    base_ang_vel = robot.get_ang().cpu().numpy()
    
    # 2. 生成期望轨迹 (简单示例: 向前行走)
    target_vel = np.array([1.0, 0.0, 0.0])  # 期望速度 1m/s
    desired_traj = generate_desired_trajectory(
        base_pos, target_vel, horizon=10, dt=0.01
    )
    
    # 3. MPC 计算期望接触力
    mpc.set_desired_trajectory(desired_traj)
    mpc.data_bus_read({
        'base_pos': base_pos,
        'base_rpy': quat_to_euler(base_quat),
        'base_lin_vel': base_vel,
        'base_ang_vel': base_ang_vel,
        'fe_l_pos_W': get_foot_position(robot, 'left'),
        'fe_r_pos_W': get_foot_position(robot, 'right'),
        'leg_state': gait_scheduler.get_leg_state(t)
    })
    mpc.cal()
    contact_forces = mpc.get_contact_forces()  # [FL, FR]
    
    # 4. WBC 计算关节力矩
    wbc.set_tasks({
        'com': {'force': contact_forces['total'], 'moment': contact_forces['moment']},
        'swing_foot_left': {'pos': desired_swing_pos_left, 'vel': desired_swing_vel_left},
        'swing_foot_right': {'pos': desired_swing_pos_right, 'vel': desired_swing_vel_right},
        'torso_orientation': {'quat': np.array([0, 0, 0, 1])}  # 保持直立
    })
    joint_torques = wbc.compute_torques()
    
    # 5. 应用控制
    robot.control_dofs_force(joint_torques)
    
    # 6. 仿真步进
    scene.step()
```

### 3.2 关键接口说明

```python
class MPCController:
    """
    MPC 控制器接口
    """
    def __init__(self, dt: float, horizon: int, mass: float, inertia: np.ndarray):
        """
        Args:
            dt: 控制周期 (秒)
            horizon: 预测步数 N
            mass: 机器人质量 (kg)
            inertia: 3x3 惯性张量 (kg·m²)
        """
        pass
    
    def set_desired_trajectory(self, Xd: np.ndarray):
        """
        设置期望轨迹
        
        Args:
            Xd: [Nx * N] 向量，包含 N 步的期望状态
                每步状态: [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz]
        """
        pass
    
    def set_weight(self, u_weight: float, L_diag: np.ndarray, K_diag: np.ndarray):
        """
        设置 MPC 权重
        
        Args:
            u_weight: 输入正则化权重 α
            L_diag: 状态权重对角线 (12维)
            K_diag: 输入权重对角线 (13维)
        """
        pass
    
    def cal(self):
        """执行 MPC 计算"""
        pass
    
    def get_contact_forces(self) -> Dict[str, np.ndarray]:
        """
        获取计算得到的接触力
        
        Returns:
            {'left': [Fx, Fy, Fz, τx, τy, τz],
             'right': [Fx, Fy, Fz, τx, τy, τz]}
        """
        pass


class WBCController:
    """
    WBC 控制器接口
    """
    def __init__(self, robot, dt: float, kp_com: float, kd_com: float):
        """
        Args:
            robot: Genesis 机器人实体
            dt: 控制周期
            kp_com: 质心位置比例增益
            kd_com: 质心位置微分增益
        """
        pass
    
    def set_tasks(self, tasks: Dict):
        """
        设置任务列表 (按优先级)
        
        Args:
            tasks: 任务字典，支持以下任务类型:
                - 'com': 质心任务 {'force': ..., 'moment': ...}
                - 'swing_foot_left/right': 摆动脚任务 {'pos': ..., 'vel': ...}
                - 'torso_orientation': 上身姿态 {'quat': ...}
                - 'joint_posture': 关节姿态 {'q': ...}
        """
        pass
    
    def compute_torques(self) -> np.ndarray:
        """
        计算关节力矩
        
        Returns:
            关节力矩数组 [n_dofs]
        """
        pass
```

---

## 4. 超参数指南

### 4.1 MPC 关键参数

#### 预测步数 (N) 和控制时域 (ch)

| 参数 | 推荐值 | 调优建议 |
|-----|-------|---------|
| N (预测步数) | 10-20 | 地形复杂↑则N↑，但计算量↑ |
| ch (控制时域) | 3-5 | 通常 < N，减小计算量 |
| dt (控制周期) | 0.005-0.01s | 越高频响应越快，但计算压力↑ |

**经验公式**: 预测时间 = N × dt 应覆盖 0.5-1.0 秒

#### 权重矩阵调优

```python
# 状态权重 L_diag (12维)
L_diag = np.array([
    1.0, 1.0, 1.0,       # 欧拉角 [roll, pitch, yaw]
    1.0, 200.0, 1.0,     # 位置 [x, y, z] - y(高度)权重通常最高
    1e-7, 1e-7, 1e-7,    # 角速度 [ωx, ωy, ωz]
    100.0, 10.0, 1.0     # 线速度 [vx, vy, vz]
])

# 输入权重 K_diag (13维)
K_diag = np.array([
    1.0, 1.0, 1.0,       # 左足力 [Fx, Fy, Fz]
    1.0, 1.0, 1.0,       # 左足力矩 [τx, τy, τz]
    1.0, 1.0, 1.0,       # 右足力 [Fx, Fy, Fz]
    1.0, 1.0, 1.0,       # 右足力矩 [τx, τy, τz]
    1.0                  # 额外项
])

# 输入正则化 α
alpha = 1e-6  # 防止输入过大，影响收敛速度
```

**调优建议**:
1. **先调位置权重**: 先确保机器人能站稳，再考虑其他
2. **高度权重最重要**: L_diag[4] (y/高度) 通常最大，防止摔倒
3. **力矩权重较小**: 通常力矩权重 < 力权重，因为力矩对质心影响小
4. **平滑性 α**: 如果输入震荡，增大 α；如果响应慢，减小 α

### 4.2 WBC 关键参数

#### 任务权重

```python
# 任务优先级 (从高到低)
task_weights = {
    'com': 1000.0,           # 质心任务 - 最高优先级
    'swing_foot': 100.0,     # 摆动脚 - 高优先级
    'torso_orientation': 10.0,  # 上身姿态 - 中优先级
    'joint_posture': 1.0,    # 关节姿态 - 低优先级
    'joint_limits': 0.1      # 关节极限 - 最低优先级
}
```

#### PD 增益

```python
# 质心 PD 增益
kp_com = 100.0   # 位置比例增益
kd_com = 20.0    # 位置微分增益

# 摆动脚 PD 增益
kp_swing = 400.0  # 需要更精确的位置跟踪
kd_swing = 40.0

# 姿态 PD 增益
kp_orientation = 100.0
kd_orientation = 20.0
```

**调优建议**:
1. **PD 增益比例**: 通常 kd ≈ 0.1-0.2 × kp
2. **摆动脚增益最高**: 需要精确跟踪轨迹
3. **避免过大 kp**: 会引起振荡和数值不稳定

### 4.3 步态参数

```python
# 步态频率
gait_frequency = 1.25  # Hz (周期 0.8s)

# 步态周期分段
duty_factor = 0.6      # 支撑相占比 (0.6 = 60%时间支撑)
swing_height = 0.08    # 摆动脚抬起高度 (m)

# 落脚点规划
foot_offset = 0.1      # 落地点超前距离 (m)
foot_width = 0.2       # 双脚横向间距 (m)
```

### 4.4 故障排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| **机器人抖动** | 控制频率太低或增益太高 | 降低 kp，增加控制频率 |
| **响应迟缓** | MPC权重太大或预测步数太少 | 减小 α，增加 N |
| **不稳定摔倒** | 高度权重不够或接触力约束失效 | 增大 L_diag[4]，检查摩擦系数 |
| **支撑脚滑动** | 摩擦系数设置不当 | 增大地面摩擦系数 |
| **QP求解失败** | 约束冲突或矩阵奇异 | 添加正则化，检查约束一致性 |

---

## 5. 进阶技巧

### 5.1 自适应 MPC
```python
# 根据速度自适应调整权重
if target_vel > 1.5:  # 高速
    L_diag[9:12] *= 2.0  # 增加速度跟踪权重
    N = 15  # 增加预测步数
```

### 5.2 软约束处理
```python
# 使用松弛变量处理硬约束
slack_weight = 1e6
# 在 QP 中添加: ||slack||^2 * slack_weight
```

### 5.3 热身策略 (Warm Start)
```python
# 使用上一帧的解作为初始猜测
mpc.set_initial_guess(previous_solution)
```
