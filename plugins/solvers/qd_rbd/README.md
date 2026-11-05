# qd_rbd — Pinocchio 刚体动力学的 Quadrants 批量并行重构

把 Pinocchio 的核心动力学算法（RNEA / CRBA / ABA / FK）从「C++ / 面向对象 /
单实例」重写为「Quadrants kernel / SOA / 批量并行」。数值上与 Pinocchio 4.1
对齐到 f64 机器精度；RNEA 另有一份 C11 转写后端（`csrc/rbd.c`），
B=1 时比 Quadrants 后端快约 7×，大 batch 快 1.2–1.4×。

本插件是 qd_rbd 八轮迭代存档的**第八轮**（最终版）：v4 的 scratch 打包 +
可复用输出，再加 C 后端。存档里其它目录的定性结论（实测数据见存档 REPORT.md）：

| 版本 | 结论 |
|---|---|
| v1–v3 | 历史迭代，性能严格劣于 v4 |
| v4 | C 之前的完整快照，Quadrants 路径与最终版逐字相同 |
| v6 | 负结果：模型常量打包成单 ndarray 比 v4 慢 14%，勿用 |
| **最终版（本插件）** | v4 + C 后端，RNEA 默认走 C，全部指标最优 |

## 特性

- 关节类型：revolute / prismatic（1 dof）、fixed；任意树拓扑，joint ≤ 63。
- 精度：f64（与 Pinocchio 4.1 数值对齐到机器精度）。
- 批量接口：`q/v/a/tau` 都是 `(B, nv)` numpy 数组，一次 kernel 调用算完整个 batch。
- `crba_apply`：只要 `M(q)·x` 时不落盘 `(B, nv, nv)` 的 M，快 2–4×。
- `fext=`：`rnea`/`non_linear_effects`/`aba` 支持外力，形状 `(B, nj, 6)`，
  为每个连杆的世界系质心 wrench `[force, torque]`（与仿真器接触力输出
  同约定，参考 FRAX/MJX）；内部一次 FK 扫掠转到体坐标系后由 sweep 减去，
  调用方无需自己做 FK。传 `fext` 时走 Quadrants 的 `*_fext` kernel
  （C 后端暂不支持外力）。
- `out=`：四个入口都支持输出缓冲复用（热循环里还能省 launch 开销）。
- 后端：RNEA 默认 C（`QD_RBD_C=0` 退回 Quadrants）；`crba`/`crba_apply`/`aba` 走 Quadrants。

## 用法

```python
import numpy as np
import pinocchio as pin                      # 可选；仅 from_pinocchio 需要
from plugins.solvers.qd_rbd import (
    Model, Data, rnea, non_linear_effects, crba, crba_apply, aba, forward_kinematics,
)

pm = pin.buildModelFromUrdf("robot.urdf")    # 任意 1-dof 关节模型
model = Model.from_pinocchio(pm)
data = Data(model, batch=4096)               # 纯 numpy 缓冲，创建不触发编译

q, v, a = (np.zeros((4096, pm.nv)) for _ in range(3))
tau = rnea(model, data, q, v, a)             # (4096, nv)
M   = crba(model, data, q)                   # (4096, nv, nv)
Mx  = crba_apply(model, data, q, v)          # (4096, nv)，不落盘 M，快 2–4×
qdd = aba(model, data, q, v, tau)            # (4096, nv)

fw = np.zeros((4096, model.nj, 6))           # 每个连杆的世界系外力
fw[:, -1, :3] = contact_force_world          # 末端接触力（质心处，世界系）
tau = rnea(model, data, q, v, a, fext=fw)    # tau = M a + b - Jᵀ fext
qdd = aba(model, data, q, v, tau, fext=fw)   # 前向动力学同样接受 fext
```

没有 pinocchio 时也可以直接用原始数组构造模型（`Model(parents, jtype,
axis, Rpl, ppl, mass, lever, Icom, iv, gravity)`），见
`examples/basic_usage.py`。

## 依赖与平台

| 依赖 | 是否必需 | 说明 |
|---|---|---|
| numpy, quadrants | 必需 | 核心 kernel |
| `pin`（pinocchio） | 可选 | 仅 `Model.from_pinocchio` / `qd_rbd.models` 参考模型。**Windows 无官方 wheel，请在 WSL 中使用**；Linux/macOS 直接 `pip install pin` |
| C 编译器 | 可选 | 仅 C 后端需要，缺失时自动退回 Quadrants 后端 |

C 后端开关：

- `QD_RBD_C=0`：强制退回 Quadrants 后端。
- `QD_RBD_CC`：指定编译器，可带参数，如 `QD_RBD_CC="python -m ziglang cc"`
  （ziglang 有 Windows wheel，是 Windows 上最省心的选择）。
- `QD_RBD_CFLAGS`：覆盖编译 flags（Windows 默认不带 `-fopenmp`，kernel 单线程；
  需要并行时自行提供带 OpenMP 的工具链并显式设置 flags）。
- `QD_RBD_LANES`：一次关节扫描里并行的实例数（默认 4，B < 16 时退为 1）。
- 产物缓存于 `~/.cache/qd_rbd/`（键 = 源码 + flags + 平台哈希）。

线程：`QD_THREADS` 控制 Quadrants 线程池；C 侧认 `OMP_NUM_THREADS` 或
`qd_rbd.ckern.set_threads(n)`。`Data` 是 scratch，**不要跨线程共享同一个 Data**。

## 测试

```bash
pytest plugins/solvers/qd_rbd/tests -q -m "not slow"
```

- `test_qd_rbd.py`：不依赖 pinocchio 的行为测试（直接从原始数组构造模型，
  内部一致性：RNEA↔CRBA↔ABA、FK、batch 独立性、`out=` 契约、fext
  的解析锚点/反重力抵消/RNEA↔ABA 往返）。
- `test_c_kernel.py`：C 后端 vs Quadrants 后端一致性；无 C 编译器时自动跳过。
- `test_pinocchio_equiv.py`：与 Pinocchio 4.1 的逐项数值对照（含 fext 的
  世界系→局部系换算）；未安装 pinocchio 时自动跳过（Windows 上于 WSL 中运行）。

## 参考数据（存档 REPORT.md，Linux EPYC 9K65）

| 机型 | 算法 | B | Quadrants | C | 倍率 |
|---|---|---|---|---|---|
| chain7 | RNEA | 1 | 36.3 µs | 5.0 µs | 7.3× |
| chain7 | RNEA | 4096，单线程 | 1995 µs | 1578 µs | 1.26× |
| chain7 | RNEA | 4096，4 线程 | 515 µs | 366 µs | 1.41× |

对 Pinocchio 串行的交叉点从 B≈48（Quadrants）降到 B≈5（C）。
