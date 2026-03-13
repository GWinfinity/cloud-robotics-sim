# MPC + WBC Controller Plugin

来源: [openloong-dyn-control](https://github.com/OpenLoongRobot)

## 核心功能

MPC (Model Predictive Control) + WBC (Whole-Body Control) 组合控制器，用于人形机器人全身运动控制。

**架构:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    MPC      │────→│    WBC      │────→│   Robot     │
│  (高层规划)  │     │  (底层执行)  │     │  (物理系统)  │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                    │
       └──────────── Gait Scheduler ←───────┘
```

**特点:**
- **MPC**: 基于线性化单刚体模型，预测未来状态，优化接触力
- **WBC**: 零空间投影实现多任务优先级控制
- **步态调度**: 管理摆动腿时序

## 快速开始

```python
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController

# 创建控制器
controller = MPCWBCController(
    num_dofs=19,
    dt=0.005,
    gait_frequency=1.25,
    mass=77.35
)

# 每帧更新
tau = controller.update(
    base_pos=robot.get_pos(),
    base_rpy=robot.get_rpy(),
    base_lin_vel=robot.get_vel(),
    base_ang_vel=robot.get_ang(),
    left_foot_pos=left_foot_pos,
    right_foot_pos=right_foot_pos,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    target_vel=np.array([1.0, 0.0, 0.0])  # 向前 1m/s
)

# 应用力矩
robot.control_dofs_force(tau)
```

## 算法原理

### MPC 模型

状态: $X = [roll, pitch, yaw, x, y, z, \omega_x, \omega_y, \omega_z, v_x, v_y, v_z] \in \mathbb{R}^{12}$

输入: $U = [F_L, F_R] \in \mathbb{R}^{12}$ (左右脚接触力)

优化目标:
```
min Σ ||X_t - Xd_t||²_Q + Σ ||U_t||²_R

约束:
- X_{t+1} = A·X_t + B·U_t
- Fz ≥ 0 (单侧约束)
- ||F_xy|| ≤ μ·Fz (摩擦锥)
```

### WBC 任务优先级

从高到低:
1. 质心动量跟踪 (最高优先级)
2. 摆动脚轨迹跟踪
3. 上身姿态保持
4. 关节极限避免

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `mpc.horizon` | int | 10 | 预测步数 |
| `mpc.dt` | float | 0.005 | 控制周期 (s) |
| `mpc.mass` | float | 77.35 | 机器人质量 (kg) |
| `mpc.L_diag` | list | [1,1,1,1,200,1,...] | 状态权重 |
| `wbc.kp_com` | float | 100.0 | 质心位置比例增益 |
| `wbc.kd_com` | float | 20.0 | 质心位置微分增益 |
| `gait.frequency` | float | 1.25 | 步态频率 (Hz) |
| `gait.gait_type` | str | 'trot' | 步态类型 |

## 文件结构

```
mpc_wbc/
├── core/
│   ├── mpc_controller.py          # MPC 核心
│   ├── wbc_controller.py          # WBC 核心
│   ├── combined_controller.py     # 组合控制器
│   ├── gait_scheduler.py          # 步态调度器
│   └── config.py                  # 配置类
├── examples/
│   └── basic_usage.py             # 基本使用示例
├── configs/
│   └── default.yaml               # 默认配置
├── tests/
│   └── test_mpc.py                # 单元测试
├── README.md
├── plugin.yaml
└── __init__.py
```

## 使用示例

见 [examples/basic_usage.py](examples/basic_usage.py)

```bash
python examples/basic_usage.py
```

## Changelog

- 2024-03-15: 初始从 openloong-dyn-control 提取
- 修复了 QP 求解的维度问题
- 添加了配置系统

## 参考

- Paper: "Dynamic Walking on Compliant and Uneven Terrain using DCM and MPC" (ETH Zurich)
- OpenLoong: https://github.com/OpenLoongRobot
