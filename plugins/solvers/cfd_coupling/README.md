# cfd_coupling — 1D 管网 ↔ 3D CFD 双向耦合求解器插件

面向「AI+工业软件」赛题(*一维管网系统+三维 CFD 局部精细模拟的联合仿真:
时间步长协调与边界耦合*)的可运行原型。与 `plugins/solvers/` 下其他求解器
一样通过 `install(scene, options)` 注入 `gs.Scene`;数值核心(PyTorch + NumPy)
同时保持无 genesis 独立可跑。

## 两种使用模式

**Genesis 模式**(与其他 solver 插件一致,推荐):

```python
import genesis as gs
from plugins.solvers.cfd_coupling import CFDOptions, CouplingOptions, PipeOptions, install
from plugins.solvers.cfd_coupling.solver import CoupledSolverOptions  # genesis 侧选项

gs.init(backend=gs.cpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.002, substeps=2))
scene.add_entity(gs.morphs.Plane())

solver = install(scene, CoupledSolverOptions(
    pipe=PipeOptions(length=10.0, wave_speed=200.0, ...),
    cfd=CFDOptions(domain=(0.2, 0.1, 0.1), cells=(20, 10, 10), ...),
    coupling=CouplingOptions(fixed_point_iters=2, valve_closure_start=0.1),
))
scene.build()

for _ in range(100):
    scene.step()   # 每个 genesis substep = 一个耦合宏步

print(solver.coupler.plenum_head)      # 3D 回传的 plenum 背压水头
print(solver.coupler.history()['t'])   # 逐宏步交换历史与延迟
```

- `scene.sim.cfd_coupling_solver` 访问求解器;`solver.pipe` / `solver.cfd` /
  `solver.coupler` 直达三个核心。
- 一个 genesis substep = 一个耦合宏步;宏步自动吸附到 1D MOC 步网
  (`n·dx/a`),建议把场景 `dt/substeps` 配成 `dx/a` 的整数倍。
- `scene.get_state()` / `scene.reset(state)` 会经 `get_state`/`set_state`
  快照/恢复 1D+3D+界面状态(与其他插件求解器一致)。
- 独立组件:`install_pipe(scene, PipeSolverOptions(...))`(1D 水锤,
  自动子循环)、`install_cfd(scene, CFDSolverOptions(...))`(3D CFD)。

**Headless 模式**(无 genesis,CI/原型调试用):直接组合 `core/` 里的
`Pipe1D` / `CFD3D` / `Coupler`,见 `examples/run_valve_closure.py`。

```
reservoir ──[1D 管道, MOC 特征线法]──> nozzle/valve ══> 3D plenum ──> 受限出口
              dt_1d = dx/a (小)        边界交换周期 macro_dt (大)
```

## 架构

| 模块 | 文件 | 说明 |
|------|------|------|
| 1D 求解器 | `core/pipe1d.py` | MOC 水锤(Courant=1 精确),上游定水位水库,下游喷嘴/阀门边界,背压水头可时变(3D 回传)。NumPy 实现,控制逻辑天然 host 侧 |
| 3D 求解器(quadrants) | `core/cfd3d_qd.py` | **默认后端**。不可压 NS,MAC 交错网格 + 投影法,`@qd.kernel` 与 thermal/acoustics 同技术栈;预测/扩散/散度/投影/温度全部 Taichi 内核,压力 Poisson 走宿主侧缓存稀疏 LU(≤30k 单元)或 scipy CG |
| 3D 求解器(torch) | `core/cfd3d.py` | 备用后端(无 genesis 环境)。与 quadrants 版数值等价、API 鸭子类型兼容,共享同一稀疏 LU 装配(`build_poisson_csr`) |
| 耦合器 | `core/coupler.py` | 宏步协调、双向交换、Gauss-Seidel 固定点迭代、阀门事件(0D 控制逻辑)。对两个 3D 后端无差别 |

后端由 `CoupledSolverOptions.backend` / `CFDSolverOptions.backend` 选择:
`"quadrants"`(默认)、`"torch"` 或 `"auto"`(有 genesis 用 quadrants,否则 torch)。

**延迟**(20×10×10 网格,本机 CPU):quadrants **~2.0 ms/步** vs torch ~6.4 ms/步——
Taichi 内核的预测+投影链路比 30+ 次小张量 torch 算子更省,达到并超过 torch 延迟水平。

### 时间步长协调

1D MOC 步长 `dt_1d = dx/a` 由波速决定(通常毫秒级),3D CFL 步长较大。
耦合器以 `macro_dt`(= 3D 步长)为交换周期,1D 在周期内子循环
`n = macro_dt / dt_1d` 步;3D 入口速度取该窗口内喷嘴流量的梯形积分均值。

### 双向边界条件

- **1D → 3D**:喷嘴瞬态流量 Q(t)(逐 1D 子步采样)→ 3D 入口补丁均匀速度
  `U = Q / A_patch`,同时传递流体温度(被动标量对流进入 3D)。
- **3D → 1D**:plenum 静压(体积均值,排除射流核心的压力匹配区)→
  1D 喷嘴背压水头 `H_back = p_mean / (ρ g)`,参与喷嘴方程
  `Q = Cd·Av·√(2g(H − H_back))` 的闭式求解。

### 毫秒级控制逻辑同步

阀门是 0D 控制逻辑设备:`valve_closure_start` / `valve_closure_duration`
定义线性关闭事件,按绝对仿真时间在 1D 子步粒度上生效(无需与 3D 步长对齐)。

### 界面稳定性

每个宏步做 Gauss-Seidel 固定点迭代(默认 2 次,最多
`fixed_point_iters`):用上一轮 3D 回传背压重放整个宏窗口,直到界面水头
残差 `< fixed_point_tol`。迭代残差(水头与流量两个口径)记录在每个
`MacroStepLog` 中。

## 验证(对应交付物 c. 算例报告)

测试在 `tests/test_coupling.py`(离线,无 genesis,CPU 可跑):

1. **水锤 vs Joukowsky 解析解** — 快关阀,阀前峰值水头 = 稳态值 + `a·ΔV/g`,2% 容差。
2. **3D 质量守恒** — 恒流入方腔/管道,稳态 `Q_out == Q_in`,散度 < 5e-5。
3. **顶盖驱动方腔(slow)** — Re=100 稳态判据 + 中心速度量级(Ghia 参考)。
4. **耦合闭环响应** — 阀门关闭 → 1D 流量扼流;受限出口 plenum 压力随流量
   建立(> 0.2 m),关阀后背压回落(双向耦合信号)。
5. **固定点收敛** — 瞬态期间交换流量残差随迭代次数下降。
6. **交换延迟** — 纯边界传输(Python 层)亚毫秒级,见示例脚本输出。

运行:

```bash
uv run python -m pytest plugins/solvers/cfd_coupling/tests/ -m "not slow"
uv run python -m pytest plugins/solvers/cfd_coupling/tests/ -m slow   # 方腔
uv run python plugins/solvers/cfd_coupling/examples/run_valve_closure.py
```

`tests/test_coupling.py` 为无 genesis 的数值验证(torch 后端);
`tests/test_genesis_integration.py` 为 `gs.Scene` 生命周期验证 + quadrants 后端
数值对照(质量守恒、与 torch 参考一致性、背压闭环、延迟同量级断言),
无 genesis 时自动跳过。quadrants 内核编译(首次 JIT)需要 genesis 已初始化。

示例脚本输出稳态标定、Joukowsky 参考值、逐宏步交换延迟统计
(median/p95/最小交换周期),并将历史保存到
`outputs/cfd_coupling/valve_closure_history.npz`。

## 已知边界(原型范围)

- 单管 1D(无分叉管网);相态参数以入口边界参数传递,无真两相流模型。
- 3D 为单相不可压、等温(温度仅作被动标量);出口为受限补丁 + p=0 Dirichlet。
- 显式格式,需满足 CFL;quadrants 内核为 float32,散度 ~1e-4 量级。
- 1D↔3D 面积不匹配通过补丁面积比处理,未做动量通量修正(界面动量
  守恒的精化处理是后续工作)。
- 反传(autodiff)未实现:两个 3D 后端的梯度钩子均为 no-op。
