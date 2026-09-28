# nozzle_coupling — 一维喷管 ↔ 三维羽流双向耦合求解器插件

面向火箭发动机"喷管-羽流"环节的 1D-3D 联合仿真（例如对开源项目
https://github.com/webwizarding/MANPADS **已公布的发动机相关资产**——
`CAD Files/Rocket Nozzle.f3z` 喷管几何概念与 `Simulation/Folding Stabilized
Rocket.ork` 仿真源文件——所代表的发动机-喷管环节做数值仿真）。与
`plugins/solvers/` 下其他求解器一样通过 `install(scene, options)` 注入
`gs.Scene`；数值核心（NumPy + PyTorch）保持无 genesis 独立可跑。

> **边界声明**：上游仓库按其 SAFETY.md 不公布任何发动机数据（无推进剂、
> 无推力曲线、无尺寸/性能指标），本插件同样只是**通用可压缩流耦合仿真
> 工具**——喷管几何（喉部/出口面积）、燃烧室压强/温度/相态曲线均为用户
> 输入参数，仓库内不提供、也不推导任何真实武器发动机数据，不涉及推进剂
> 设计或性能优化。物理模型为教科书级公开方法（Sutton 准一维等熵流、
> 正激波关系、两相速度滞后修正）。

## 功能（对应需求）

1. **一维 → 三维实时传递**：一维喷管每个宏步把窗口平均的瞬态质量流量
   ṁ(t)、滞止温度、相态参数（凝结相质量分数 α、两相修正后的等效
   γ/R）作为三维入口边界条件（质量通量入口）。
2. **三维 → 一维反向作用**：三维 plenum 在喷管出口平面的静压（背压）
   回传一维模型，驱动堵塞/非堵塞判据、扩张段正激波位置（过膨胀响应）、
   亚声速节流直至反压堵死。

```
燃烧室曲线 p_c(t),T_c(t),α(t) → [1D 准定常喷管] → ṁ,u_e,T_e,α,r_eff
                                        ══ 宏步交换 ══
        出口平面背压 p_back ← [3D 滞弹性羽流/plenum] ← 受限/可变出口
```

## 两种使用模式

**Genesis 模式**（与其他 solver 插件一致）：

```python
import genesis as gs
from plugins.solvers.nozzle_coupling import install, NozzleOptions, JetOptions
from plugins.solvers.nozzle_coupling.core import NozzleCouplingOptions
from plugins.solvers.nozzle_coupling.solver import CoupledNozzleSolverOptions

gs.init(backend=gs.cpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=2e-4, substeps=1, gravity=(0,0,0)))
solver = install(scene, CoupledNozzleSolverOptions(
    nozzle=NozzleOptions(throat_area=1e-3, exit_area=3e-3,
                         chamber_pressure=130e3, chamber_temperature=600.0),
    jet=JetOptions(domain=(0.24, 0.12, 0.12), cells=(12, 6, 6)),
    coupling=NozzleCouplingOptions(macro_dt=2e-5, n_substeps=2),
))
scene.build()
for _ in range(1000):
    scene.step()                        # 每个 genesis substep = 一个耦合宏步
print(solver.coupler.backpressure)      # 3D 回传的喷管出口背压 [Pa]
print(solver.coupler.history()['mdot']) # 逐宏步 ṁ / 背压 / 出口压 / 工况 / 延迟
solver.jet.set_outlet_patch(((0.4, 0.6), (0.4, 0.6)))  # 运行中节流事件
```

- `scene.sim.nozzle_coupling_solver` 访问求解器；`solver.nozzle` /
  `solver.jet` / `solver.coupler` 直达三个核心。
- `scene.get_state()` / `scene.reset(state)` 快照/恢复 1D+3D+界面状态。
- 独立组件：`install_nozzle`（1D 喷管，固定背压选项）、`install_jet`（3D 羽流）。

**Headless 模式**（无 genesis）：直接组合 `core/` 里的 `Nozzle1D` /
`Jet3D` / `NozzleCoupler`，见 `examples/run_nozzle_exhaust.py`。

## 架构

| 模块 | 文件 | 说明 |
|------|------|------|
| 1D 喷管 | `core/nozzle1d.py` | 准一维 C-D 喷管（纯 NumPy）。燃烧室曲线/表格驱动 p_c(t),T_c(t),α(t)；按回传背压选择工况：超音速（壅塞）、扩张段正激波（迭代激波面积比，使激波后等熵增压匹配背压；低于唇口激波压力时钳位于唇口）、全亚声速（p_e=p_back，流量连续下降）、反压≥燃烧室压强堵死。两相：Sutton 混合法则等效 γ/R + 速度滞后因子 ψ=(1-α)+αφ 修正排气速度；可选推力一阶滞后 |
| 3D 求解器 | `core/jet3d.py` | 滞弹性（anelastic）弱可压 NS（纯 PyTorch，MAC 交错网格）：密度由 EOS ρ=p_amb/(R_mix T) 诊断给出，大热/冷密度比（典型排气羽流 ~10x）无声学 CFL 限制；能量方程给出热膨胀散度目标 D=(1/T)DT/Dt，变系数压力投影 ∇·(β∇p')=(∇·u*−D)/dt（β=1/ρ 冻结，CG + 缓存常系数 LU 预处理器，收敛判据含散度约束残差）；凝结相分数为被动输运标量并设置 R_mix；质量通量入口（ṁ,T0,α,r_eff → 补丁速度）；受限出口补丁产生 plenum 阻力（背压来源）；运行时 `set_outlet_patch` 节流事件 |
| 耦合器 | `core/coupler.py` | 宏步协调：1D 窗口内子循环采样 → 梯形平均（ṁ,T,α,r_eff）→ 3D 入口（带启动 ramp 与入口质量通量变化率限制）；3D 步后探针取入口补丁区平均静压 → 背压回传。Gauss-Seidel 固定点迭代 + **双向变化率限制**（背压 slew 限制 + 入口 ṁ slew 限制）：准定常喷管对背压是阶跃响应（p_back≥p_c 时 ṁ 突跳为零），无限制界面在重启尖峰下会极限环（blocked↔choked），变化率限制使耦合迭代收缩（物理对应 plenum 容积/供给系统时间常数） |

### 时间步长协调

宏步 = 3D 步长（`macro_dt`，受 3D 对流 CFL 限制：喷管出口射流为跨/超声速，
u_e 可达 10²-10³ m/s，须按 u_e·dt/h ≲ 0.5 选取）；1D 侧在窗口内
`n_substeps` 次采样（1D 为准定常 + 曲线驱动，无 CFL 限制）。Genesis 模式下
一个 substep 可重复多个宏步（`build()` 自动吸附）。

### 界面稳定性

- 每宏步 Gauss-Seidel 固定点迭代（默认 2 遍），残差记录在
  `NozzleMacroStepLog.fp_residuals`。
- 背压/入口质量通量双向 slew 限制（`backpressure_max_rate_pa_s` /
  `inlet_mdot_slew_kg_s2`）：抑制数值重启尖峰导致的界面极限环。

## 验证

测试在 `tests/`（离线数值断言，无 mock）：

- `test_nozzle1d.py`：堵塞流量 vs 等熵闭式解（1‰）；亚声速 p_e=p_back 与
  流量闭式解；激波工况（壅塞流量保持、激波位置恢复背压、唇口钳位、
  工况转移与单调性）；两相混合法则与速度滞后因子；面积-马赫关系往返。
- `test_jet3d.py`：质量通量入口 BC；投影满足膨胀散度目标（|∇·u−D| 有界）；
  瞬态质量簿记（dM/dt = ṁ_in − ṁ_out）；受限出口提高背压（冷态低动量
  射流，单调信号）；相态标量输运。
- `test_coupler.py`：燃烧室曲线瞬态跟随堵塞闭式解；**端到端反向作用**
  （运行中 `set_outlet_patch` 节流 → plenum 增压 → 回传背压上升 >5 kPa →
  1D 出口压力响应 >3 kPa）；界面通路确定性测试（注入跨工况背压，耦合
  ṁ 与解析值一致）；固定点残差收敛；交换延迟亚毫秒。
- `test_genesis_integration.py`：install/build/step/探针、
  `scene.reset(state)` 状态恢复、独立组件（无 genesis 自动跳过）。

```bash
uv run python -m pytest plugins/solvers/nozzle_coupling/tests/ -m "not slow"
uv run python plugins/solvers/nozzle_coupling/examples/run_nozzle_exhaust.py
```

示例输出稳态标定、节流事件前后的背压/出口压力信号与逐宏步交换延迟统计
（中位数亚毫秒级），历史保存到
`outputs/nozzle_coupling/nozzle_exhaust_history.npz`。**示例参数为占位值**。

## 已知边界（原型范围）

- 3D 为滞弹性低马赫模型：M≲2-3 的近场羽流可用，激波捕捉为激波管级一阶
  精度；M>3 区域仅定性。密度为诊断量（ρ=p_amb/(RT)），动量方程非守恒
  形式，质量守恒在准稳态下成立（瞬态簿记容差见测试）。
- 1D 为准定常（无喷管内波动/特征时间），燃烧室瞬态完全由用户曲线驱动；
  单喷管，无分叉。
- 喷管工况为 1D 集总（激波用正激波+等熵再压缩近似；低于唇口激波压力时
  钳位于唇口，外部压力匹配由 3D 完成）。
- 1D↔3D 面积不匹配通过补丁面积换算，无动量通量修正；热射流核心在出口
  平面压力匹配，背压探针对热超声速射流偏低（冷态/亚声速设计信号最强，
  测试已相应配置）。
- 无 quadrants/Taichi 后端（变系数压力方程不适合 float32 kernel）；
  无浸入障碍物（本期不接入 `cfd_coupling` 的 obstacles）；无 autodiff。
- 显式格式：3D 时间步受对流 CFL 限制（超声速射流需小步长）；脉冲式
  入口（无 ramp）会激发启动瞬态，耦合场景务必使用 ramp/slew 限制。
