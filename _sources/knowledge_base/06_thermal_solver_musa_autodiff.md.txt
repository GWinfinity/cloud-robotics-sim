# Thermal Solver 可微热源耦合迁移备忘录

> 状态: **待后端支持后启用**  
> 涉及后端: MUSA / Quadrants (LLVM) autodiff  
> 影响范围: `plugins/solvers/thermal/core/thermal_solver.py`  
> 最后更新: 2026-08-14

---

## 1. 背景

`plugins/solvers/thermal` 已实现显式 FTCS 热扩散 + 热源双向耦合求解器，支持：

- 单场景 / `n_envs` batch 仿真
- build 前后及 `scene.step()` 后动态添加热源 (`add_source`)
- 重叠热源的全局能量守恒修正（绝对误差 ~1e-7）
- 初始温度场、扩散过程的反向梯度

但**热源参数（body temperature / position / radius / rate / capacity）的梯度**当前被主动跳过，原因是 Quadrants/Taichi autodiff 后端尚无法编译 source-coupling 相关 kernel 的 `.grad()`。

---

## 2. 当前行为与限制

### 2.1 已启用的梯度

- `ThermalSource.temperature` / 初始温度场 → 通过扩散 kernel 的反向传播完整可微。
- 扩散系数、边界条件（Dirichlet/Neumann）→ 已包含在 `substep_pre_coupling_grad` 的反向路径中。

### 2.2 暂不可微的部分

| 参数 | 当前行为 | 目标 |
|---|---|---|
| `source.temperature` | straight-through（视为常量） | 应参与反向传播 |
| `source.position` | straight-through | 应参与反向传播 |
| `source.radius` | straight-through | 应参与反向传播 |
| `source.rate` | straight-through | 应参与反向传播 |
| `source.capacity` | straight-through | 应参与反向传播 |

### 2.3 后端报错

对以下 kernel 调用 `.grad()` 时，Quadrants 1.2.0 (LLVM backend) 抛出：

```
RuntimeError: [quadrants/transforms/auto_diff/auto_diff_common.h:...] Not supported.
```

涉及的 kernel：

- `_compute_source_equilibrium_2d`
- `_compute_source_equilibrium_3d`
- `_apply_energy_correction_2d`
- `_apply_energy_correction_3d`

相关代码位置：

```python
# plugins/solvers/thermal/core/thermal_solver.py

def substep_pre_coupling_grad(self, f: int) -> None:
    ...
    if self._dim == 2:
        if has_sources:
            self._update_source_cells_2d.grad(f)
            # TODO: 待 MUSA/Quadrants autodiff 支持后重新启用：
            # self._compute_source_equilibrium_2d.grad(f)
            # self._apply_energy_correction_2d.grad(f)
        ...
```

---

## 3. 根因分析

source-coupling kernel 中使用了 Quadrants autodiff 当前不支持的多种模式：

| Kernel 代码模式 | 对 autodiff 的影响 | 当前处理 |
|---|---|---|
| `ti.atomic_add` 在循环内累加能量 | reverse-mode 尚未支持原子加梯度 | 保留原实现，仅跳过 `.grad()` |
| in-place field 写入（更新 `_T`、`_source_temperatures`） | 需要 alias-free 或显式 adjoint 处理 | 保留原实现，仅跳过 `.grad()` |
| 循环中 `break`（跳出 cell 搜索） | control-flow 反向支持不完整 | 保留原实现，仅跳过 `.grad()` |
| `int` ↔ `float` 混合 cast | 某些组合会触发 `Not supported` | 已尽量避免，但搜索/索引逻辑仍无法完全消除 |

> 注：扩散 kernel（`_step_transient_2d_interior` 等）未使用上述模式，因此反向传播正常。

---

## 4. MUSA / Quadrants 后端迁移 TODO

完成以下后端能力补齐后，即可重新启用 source-coupling 梯度：

### 4.1 必备项（启用 source 参数梯度）

- [ ] **Reverse-mode `atomic_add`**：支持在 `ti.kernel` 内对 `qd.field` 进行 `ti.atomic_add` 的反向梯度生成。
- [ ] **In-place field 写入的 adjoint**：允许 kernel 在正向过程中修改 field 元素，并在 `.grad()` 时正确传播 adjoint，不触发 alias 错误。
- [ ] **循环 `break` 的 control-flow 反向**：支持带 `break` 的动态循环在 autodiff 中的正确转换。
- [ ] **`int→float cast` 在 autodiff 路径中的鲁棒性**：确保索引/掩码计算中的 cast 不触发 `Not supported`。

### 4.2 推荐项（提升可维护性）

- [ ] 提供 autodiff 报错诊断工具：当 kernel 包含不支持的代码模式时，输出具体 kernel 名与行号。
- [ ] 文档化 Quadrants/MUSA autodiff 的白名单/黑名单代码模式，便于后续 kernel 设计时规避。

### 4.3 完整端到端可微（超出本次范围）

- [ ] **跨求解器梯度**：热损失 → 刚体自由度（temperature → rigid-body DOF）。需要 Genesis 暴露求解器间梯度通道，目前不在本次任务范围。

---

## 5. 重新启用条件与检查清单

当 MUSA/Quadrants 后端支持上述必备项后，按以下步骤启用 source-coupling 梯度：

### 5.1 取消 kernel 调用的注释

在 `thermal_solver.py` 的 `substep_pre_coupling_grad` 中：

```python
if has_sources:
    self._update_source_cells_2d.grad(f)
    self._compute_source_equilibrium_2d.grad(f)   # 取消注释
    self._apply_energy_correction_2d.grad(f)      # 取消注释
```

3D 版本同理。

### 5.2 移除 straight-through 说明

- 更新 `plugins/solvers/thermal/README.md` 中的 Differentiability 章节。
- 删除或更新本备忘录第 2.2 节中的限制说明。

### 5.3 验证测试

运行已有测试：

```bash
uv run python -m pytest plugins/solvers/thermal/tests/test_thermal.py -v
```

新增/启用以下梯度测试：

- `test_gradient_flows_to_source_temperature`
- `test_gradient_flows_to_source_position`
- `test_gradient_flows_to_source_radius`
- `test_gradient_flows_to_source_rate`
- `test_gradient_flows_to_source_capacity`
- `test_gradient_flows_to_initial_temperature_with_sources`（已存在，需确认 source 参数也参与梯度）

最小验证示例（可作为新增测试模板）：

```python
def test_gradient_flows_to_source_temperature(self):
    scene, thermal = self._make_scene(source=True)
    source = thermal.add_source(
        temperature=100.0,
        position=(0.5, 0.5),
        radius=0.1,
        rate=1.0,
        capacity=1e6,
    )
    thermal._T.from_torch(torch.zeros_like(thermal._T.to_torch()))
    thermal._T.grad.from_torch(torch.ones_like(thermal._T.to_torch()))

    scene.step()
    grad = source.grad_temperature
    assert grad is not None
    assert grad.abs().max() > 0.0
```

### 5.4 质量门禁

- [ ] `uv run python -m ruff check src/ tests/ plugins/solvers/thermal`
- [ ] `uv run python -m black --check src/ tests/ plugins/solvers/thermal`
- [ ] `uv run python -m pytest plugins/solvers/thermal/tests/test_thermal.py -v` 全部通过

---

## 6. 相关文件索引

| 文件 | 说明 |
|---|---|
| `plugins/solvers/thermal/core/thermal_solver.py` | 求解器主代码，包含 TODO 注释 |
| `plugins/solvers/thermal/tests/test_thermal.py` | 测试集，含现有梯度测试 |
| `plugins/solvers/thermal/README.md` | 用户文档，含 Differentiability 说明 |
| `src/cloud_robotics_sim/utils/genesis_compat.py` | Genesis/Quadrants 兼容性辅助 |

---

## 7. 已知环境噪音

- Windows pyglet offscreen context 在连续创建多个 `gs.Scene` 时偶发 fatal exception / access violation，与本次代码无关，重跑即可恢复。

---

## 8. 参考

- Genesis Cloud Sim `AGENTS.md`
- `plugins/solvers/thermal/README.md`
- Quadrants 1.2.0 LLVM backend autodiff 实现：`quadrants/transforms/auto_diff/`
