# qd_ik — frax.ik 的 Quadrants 批量并行重构

把 [frax.ik](https://github.com/StanfordASL/frax)（Stanford ASL 的 JAX 机器人
运动学/逆运动学库）的重定向管线移植到 Quadrants kernel 上：

- **任务**：`FrameTask`（关节框架位置/朝向跟踪，含 frax 的
  `orientation_error_quat` 语义）、`JointTask`（关节空间跟踪）；
- **约束**（弹性 CBF，L1 松弛不等式，与 ElastiQP 同契约）：`JointLimits`
  （位置/速度限位）、`FloorConstraint`（地板）、`SelfCollisionConstraint`
  （球模型自碰撞）；
- **QP 求解器**：批处理 ADMM（同一弹性 QP 问题，**非** ElastiQP 的算法
  等价；ElastiQP 的 DAS 是 C++ 主动集法，这里换成 SIMD 友好的固定迭代
  ADMM + Ruiz 风格行缩放 `w = sqrt(penalty)`）；
- **浮动基座**：position + WXYZ quaternion（MuJoCo/frax 约定），
  `q = [pos(3), quat(4), joints]`，`v = [世界线速度, 体系角速度, 关节速度]`；
- **批量语义**：一次 kernel 调用处理 B 个并行 clip（或并行 IK 问题），
  对应 frax 的 `jax.vmap`/`device_vmap`，但走的是 qd_rbd 风格的
  SOA + numpy scratch 打包路线。

与 frax 的差异（设计取舍，非缺陷）：

| 项 | frax.ik | qd_ik |
|---|---|---|
| QP 求解 | ElastiQP（C++ FFI，DAS/PDAL/IPM） | 批处理 ADMM（固定迭代、热启动、`rho` 按 `diag(P)/pen` 自动标定） |
| 全局 IK 初始化 | `GlobalIKSolver` 多起点采样 | 未移植（`initial_q` + 自适应步长收敛，同 frax `global_ik_init=None` 路径） |
| 自碰撞行筛选 | 每步 top-k 最近 `max_rows` 对（`lax.top_k`） | 同款语义：build 内核内对全量 h 做 k 轮 argmin 选取（确定性，平手取小索引），只给选中行写 G/h |
| 障碍物/通用 CBF | `ObstacleConstraint` / `BarrierConstraint`（autodiff 雅可比） | 未移植（需要 autodiff；当前约束均为闭式雅可比） |
| 微分 | 全链路 JAX autodiff | 无（Quadrants kernel 手写几何雅可比） |

## 用法

```python
import numpy as np
from plugins.solvers.qd_ik import (
    DiffIKSolver, FloorConstraint, FrameTask, JointLimits, Retargeter,
    SphereModel,
)

model = Model.from_urdf("g1.urdf", floating_base=True)   # 或 Model(parents, jtype, ...)
spheres = SphereModel(model, links=[...], offsets=..., radii=...)
task = FrameTask(model, ["wrist"], gain_pos=30.0, gain_rot=30.0,
                 weight_pos=10.0, weight_rot=10.0)
solver = DiffIKSolver(
    model, [task],
    [JointLimits(model), FloorConstraint(model, spheres, gain=30.0)],
    regularization=1.0, damping=0.0,
)
rt = Retargeter(solver, preprocess, initial_q)   # 两个用户回调见下
qs = rt.retarget_clips(clips, dt=1/30)           # list[(T_i, nq)]
```

- `preprocess(frame_batch)`：把输入帧（B, ...）变成 ref dict，key 与任务
  声明的 `pos_key`/`rot_key`/`ref_key` 对应，值为 (B, ...) numpy 数组；
- `initial_q(frame_batch)`：给每个 clip 一个 (B, nq) 初始构型（通常取
  机器人默认构型 + 第一帧的根位姿）。

LAFAN1→G1 的完整配方（RETARGETING_TABLE、缩放常数、收窄的关节限位）
见 frax 的 `examples/ik/lafan1/lafan1_g1.py`；把其中的 `FrameTask`/
`JointLimits`/`FloorConstraint` 配置原样搬到本插件即可，预处理是纯 numpy
的缩放平移，不涉及 JAX。

## 数值与精度

- 全链路 f64。FK 与 numpy 参考对齐到机器精度；Jacobian 与中心差分对齐到
  1e-5（差分截断误差量级）。
- QP 精度由 `QPConfig.iters`（默认 1000）/`rho` 控制。`DiffIKSolver` 默认按
  `rho ≈ mean(diag(P)) / mean(penalty)` 自动标定（v 步曲率
  `P + rho·pen·G'G` 与任务项同量级；偏离 1e4 倍以上收敛明显变慢），
  也可显式传 `qp_config` 覆盖。`Retargeter(qp_iters_init=..., qp_iters_step=...)`
  可分别调整首帧收敛与逐帧步的迭代预算。验证方法见 `tests/test_qp.py`：
  无约束闭式解、凸对偶间隙证书、可行域 hard-QP 与 SLSQP 对照。
- **ADMM 冷启动收敛比 DAS 慢**；逐帧热启动（`warm=True` 链）是设计内的
  使用方式。首帧 `converge` 循环用自适应步长补偿单步求解误差。

## 性能（粗略）

瓶颈在 QP 的 ADMM 迭代（每迭代约 `2*m*nv` flops + 两次三角回代）。
nv=35、m≈200（G1 量级）、B 路并行、1000 迭代的热步约几十 ms/帧步
（单线程量级，未做向量化调优）。frax 单 clip CPU 约 10 万 FPS；qd_ik 的
定位是**离线批量吞吐**（多 clip 锁步并行 + f64 + 无 JAX 依赖），单 clip
延迟不是目标。

## 依赖与平台

| 依赖 | 是否必需 | 说明 |
|---|---|---|
| numpy, quadrants | 必需 | 核心 kernel |
| genesis-world | 仅 `Model.from_urdf` | 经 `genesis.ext.urdfpy` 解析 XML（惰性导入，不触发 `gs.init`，不加载网格） |
| scipy | 仅测试 | 参考解 |

## 测试

```bash
pytest plugins/solvers/qd_ik/tests -q -m "not slow"
```

无网络、无 GPU、无外部资产。G1/LAFAN1 级别的端到端需要用户提供 URDF 与
球碰撞模型（例如 bubblify 输出）；本仓库未内置 G1 资产。

## 未移植（相对 frax.ik）

- `GlobalIKSolver` 多起点全局初始化（当前走 `initial_q` + 自适应收敛）；
- `ObstacleConstraint` / `BarrierConstraint` 等通用 CBF（依赖 autodiff
  雅可比，当前约束均为闭式雅可比）。

自碰撞 `max_rows`（每步只保留最近 k 对的 top-k 筛选）已支持：选择在内核内
对全量 h 做 k 轮 argmin 完成，只给选中行写 G/h，`QPData` 的 m 相应缩小。
