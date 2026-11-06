# 可微仿真 backward 接触 NaN：genesis-world 1.4.5 上游 bug 记录

**日期**: 2026-10-27
**来源**: 调研仓库 newton（`ISSUE_backward_contact.md`，已按上游 issue 模板整理完毕，可直接提交 Genesis-Embodied-AI/genesis-world）；本仓库在升级 1.4.5 与收编 learned-dynamics 能力时复核复现。
**结论**: genesis-world 1.4.5（quadrants 1.3.3）在「地面接触 + 基座速度注入 + `loss.backward()`」组合下必现 `INVALID_CONTACT_NAN`，属上游 solver 内部 bug，非本项目代码问题。可微训练路径需规避接触，或等待上游修复。

---

## 1. 现象与最小复现

可微刚体场景（地面接触）中，对所有 dof（含 6 个浮动基座 dof）经 `set_dofs_velocity` 注入随机初速度，短步数 rollout 后第一次 `loss.backward()` 在 rewind 路径崩溃。前向完全正常（loss 有限、逐步 `check_errno` 干净）。

```python
import torch
import genesis as gs

gs.init(seed=0)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2, requires_grad=True),
    rigid_options=gs.options.RigidOptions(max_collision_pairs=32),
    show_viewer=False,
)
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.45)))
scene.build(n_envs=8)

n = robot.n_dofs
v0 = 0.5 * torch.randn((8, n), dtype=gs.tc_float, device=gs.device)
scene.reset()
robot.set_dofs_velocity(v0)
for _ in range(5):
    robot.control_dofs_force(torch.zeros((8, n), dtype=gs.tc_float, device=gs.device))
    scene.step()
loss = (robot._solver.get_state().dofs_vel ** 2).sum()
loss.backward()   # <-- GenesisException: INVALID_CONTACT_NAN
```

## 2. 隔离矩阵（5 步 rollout + backward，CPU，seeds 0/1/2）

| 基座 dof 速度 | 地面接触 | `requires_grad` 力输入 | 第二场景 | 结果 |
|---|---|---|---|---|
| 随机 (0.5·randn) | 有 | 否 | 无 | **FAIL** |
| 随机 | 有 | 是 | 有 | **FAIL** |
| 仅电机 dof 随机 | 有 | 任意 | 任意 | PASS |
| 随机 | 悬空 (z=1.0 无接触) | — | — | PASS |
| 随机 (0.3 小幅), 4 envs | 有 | 否 | 无 | PASS |

触发三要素：**地面接触 + 较强随机基座速度 + backward rewind**。报错栈：

```
genesis/grad/bridge.py:46  inputs_grad = sim.reverse_step(ctx.step, ctx.epoch)
genesis/engine/simulator.py:358  inputs_grad_pairs = self._step_grad()
genesis/engine/simulator.py:448  self.load_ckpt()
genesis/engine/simulator.py:533  self.step(in_backward=True)
genesis/engine/simulator.py:407  self.rigid_solver.check_errno()
GenesisException: Collision detection reported a contact whose position,
normal or penetration is not finite.
```

## 3. 根因观察（代码走读，待上游确认）

1. 前向健康：每步 `check_errno` 无 `INVALID_CONTACT_NAN`，loss 有限。
2. rewind 时 `Simulator._step_grad → load_ckpt` 以 `in_backward=True` 重跑 `step()`，碰撞检测在**从 checkpoint 恢复的状态**上重运行并置位 errno（标志由碰撞核写入，非 `check_errno` 本身）。
3. 给 `RigidSolver.check_errno` 加 `self._is_backward` 门控不能抑制失败——NaN 接触在到达 raise 点之前已污染下游 rewind 计算。errno 是症状而非唯一病灶。
4. 合理怀疑：速度注入步的 checkpoint 未捕获碰撞检测所需的全部工作缓冲（collider 工作缓冲 / `requires_grad` 下 `kernel_save_adjoint_cache` 写入的伴随缓存），基座速度使接触强非定常后，恢复态上的接触重检读到不一致输入。

## 4. 对本项目的影响与规避

- **thermal/acoustics/joule 插件**：可微梯度路径已于 1.4.5 适配（见 `plugins/solvers/thermal` 等的桥接注册），无接触场景不受影响。
- **learned-dynamics / sysid 残差训练**（规划收编自 newton）：可微训练（`dof_force` 注入模式）必须在**无接触**场景进行（newton 的 `probe_nerd_sysid.py` 即将机器人悬停在 z=1.0 即为此因）。
- **规避选项**：(a) 无接触场景；(b) 仅随机化电机 dof 速度；(c) 减小扰动尺度（<0.3）。
- **跟踪**：上游相关 issue #2537（backward 挂起）、#3614（check_errno 阻塞性）、#3597/#3595（backward 提速，改动同一栈）。上游修复后应撤销相应 xfail/规避并补回归测试。

## 5. 附件

- 完整复现矩阵脚本：`repro_backward_contact_v2.py`（newton 仓库，env 开关隔离各因素）。
- 本文档对应的上游 issue 正文（英文，含 Suggested fixes 三段）在 newton 仓库 `ISSUE_backward_contact.md`，可直接提交。
