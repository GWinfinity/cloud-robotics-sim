# 多物理场网格求解器插件

`plugins/solvers/` 下的场求解器插件共享同一骨架：`plugin.yaml` 元数据 +
`install(scene, options)` 注入（`scene.build()` 前调用）+ options dataclass +
`@qd.data_oriented` Solver 类 + 每维一套 quadrants kernel + 自带 tests/ 与
examples/。全部适配 genesis-world 1.4.0（见各 README 的兼容性说明）。

| 插件 | 物理 | 数值方法 | 入口 |
|---|---|---|---|
| [thermal](thermal/) | 热传导（扩散方程 ∂T/∂t = α∇²T） | 显式 FTCS，Dirichlet/Neumann 边界，能量守恒的刚体-网格双向耦合 | `thermal.install` |
| [joule_heating](joule_heating/) | 焦耳热（∇·(σ∇V)=0 → Q=σ\|∇V\|²） | Jacobi 迭代（可选 direct），热源可单向注入 thermal 求解器 | `joule_heating.install` |
| [acoustics](acoustics/) | 线性声学（波动方程 p_tt = c²∇²p） | 二阶 leapfrog + CFL 校验；海绵层/Dirichlet/Neumann 边界；单极子声源、刚体振动发声、虚拟麦克风（FFT → SPL） | `acoustics.install` |

## 共同约定

- `options.dt` 缺省时取场景步长；每个求解器以自己的 `dt` 逐 substep 推进。
- 实体耦合对象在 `scene.build()` 前后都可注册（槽位在 build 时预分配，
  `max_sources` / `max_bodies` / `max_probes` 可配）。
- NumPy 出入口统一为 `get_<field>()` / `set_<field>()`；状态经
  `get_state` / `set_state` 参与 `scene.reset()` 与 checkpoint。
- 线性传播核对 `scene.requires_grad=True` 可微；源/耦合注入的 backward
  目前跳过（quadrants/Taichi AD 限制，见 `scripts/quadrants_issue_thermal_source_autodiff.md`
  与各内核中的 `TODO(MUSA/autodiff)` 注释）。
- 测试位于各插件 `tests/`（CPU 可跑，每个插件单独跑 pytest——模块级 `gs.init`
  不能同进程复用）；示例位于各插件 `examples/basic_usage.py`。

## 快速对比：声学求解器 vs 商业软件

| 能力 | 本 acoustics 插件 | Ansys Mechanical / Fluent | LS-DYNA |
|---|---|---|---|
| 控制方程 | 线性波动方程（时域） | Helmholtz（频域 FEM）+ FWH/波动方程（气动声） | 显式 FEM（时域）+ Helmholtz BEM（频域） |
| 开放边界 | 海绵层 | 无限元 / PML / 海绵层 | BEM 天然无限域 |
| 刚体振动发声 | `add_body()`（单向） | 强 FSI 双向耦合 | 强耦合 / FFT→BEM 弱耦合 |
| 远场 SPL | `probe.spectrum()/spl()` | FWH 接收器 + FFT | 场点压力输出 |
| 气动声（流动噪声） | 不支持 | FWH / 宽频源 / APE | CESE 可压缩求解 |

详见 [acoustics/README.md](acoustics/README.md) 与知识库
`docs/knowledge_base/08_ansys_acoustics_math_and_differentiability.md`。
