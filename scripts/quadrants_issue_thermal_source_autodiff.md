# Issue: Reverse-mode AD fails on thermal source-coupling kernels (`atomic_add`, in-place writes, `break`)

## Summary

The `ThermalSolver` plugin in `genesis-cloud-sim` can back-propagate through the plain diffusion step, but fails as soon as we try to differentiate the energy-conserving source-coupling kernels that exchange heat between coupled rigid/MPM/SPH/FEM bodies and the grid.

The failing kernel (`_compute_source_equilibrium_2d`) uses three patterns that Quadrants reverse-mode AD currently rejects:

1. `qd.atomic_add` into a temperature field.
2. In-place read/write on a per-source temperature field.
3. A `break` inside a distance-based loop over grid cells.

## Minimal reproduction

Upstream repository: `GWinfinity/quadrants`
Reproduction repository: `GWinfinity/cloud-robotics-sim` (commit `1da487d` or later)

```bash
# From the genesis-cloud-sim repo, with its venv active
uv run python scripts/repro_thermal_source_autodiff.py
```

The reproduction script is also pasted below for convenience:

```python
"""Minimal reproduction: Quadrants autodiff fails on thermal source coupling."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import genesis as gs
from plugins.solvers.thermal import install, ThermalOptions


gs.init(backend=gs.cpu)


class _FixedSource:
    def __init__(self, pos: tuple[float, float, float]) -> None:
        self._pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self._pos


def main() -> int:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    thermal = install(
        scene,
        ThermalOptions(
            resolution=(8, 8),
            dx=0.01,
            alpha=1e-4,
            boundary_mode="dirichlet",
            boundary_value=0.0,
            initial_temperature=0.0,
        ),
    )
    thermal.add_source(
        entity=_FixedSource((0.04, 0.04, 0.0)),
        temperature=1.0,
        radius=0.03,
        rate=10.0,
        heat_capacity=0.1,
    )
    scene.build()
    scene.step()

    # 1. Diffusion backward works.
    t_field = thermal._T
    assert t_field is not None and t_field.grad is not None
    thermal.reset_grad()
    t_grad = t_field.grad.to_numpy()
    t_grad[1] = 1.0
    t_field.grad.from_numpy(t_grad)
    thermal._step_transient_2d_interior.grad(0)
    thermal._step_transient_2d_boundary_dirichlet.grad(0)
    print("Diffusion grad OK")

    # 2. Source-coupling backward fails.
    thermal.reset_grad()
    t_grad = t_field.grad.to_numpy()
    t_grad[1] = 1.0
    t_field.grad.from_numpy(t_grad)
    thermal._compute_source_equilibrium_2d.grad(0)  # <-- raises RuntimeError
    print("Source equilibrium grad OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Actual behavior

The diffusion part prints `Diffusion grad OK`, then the source-coupling part raises:

```
RuntimeError: [quadrants/transforms/auto_diff/auto_diff_common.h:quadrants::lang::ADTransform::visit@449] Not supported.
```

## Expected behavior

`thermal._compute_source_equilibrium_2d.grad(0)` should compile and populate `thermal._source_temperatures.grad` with non-zero adjoint values, so that source temperatures/radii/rates become learnable parameters.

## Failing kernel pattern

The failing kernel is in `plugins/solvers/thermal/core/thermal_solver.py`, method `_compute_source_equilibrium_2d`. It roughly looks like:

```python
@qd.kernel
def _compute_source_equilibrium_2d(self, f: qd.i32):
    for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
        # Heat each grid cell by relaxing toward each source.
        for s in range(self._n_sources):
            dx = float(i) * self._dx - self._source_positions[i_b, s][0]
            dy = float(j) * self._dx - self._source_positions[i_b, s][1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > self._source_radii[i_b, s] * self._source_radii[i_b, s]:
                continue  # <-- break-like control flow
            # in-place update of per-source temperature
            self._source_temperatures[i_b, s] = ...  # read + write same field
            # accumulate heat into grid cell
            qd.atomic_add(self._T[f + 1, i_b, i, j], value)  # <-- atomic_add
```

Other skipped kernels use the same patterns:

- `_apply_energy_correction_2d` / `_apply_energy_correction_3d`
- `_accumulate_source_body_energy_2d` / `_accumulate_source_body_energy_3d`
- `_compute_energy_after_2d` / `_compute_energy_after_3d`
- `_gather_source_cells_2d` / `_gather_source_cells_3d`

## Environment

- OS: Windows 10/11 (also reproducible on Linux)
- Python: 3.13.12
- genesis-world: 1.3.2
- quadrants: 1.2.0 (commit `3d9af718`)
- Backend: `gs.cpu`

## Suggested fix directions

### Root-cause note

In current Quadrants, `qd.atomic_add` into a differentiable field and same-index in-place field writes are **already handled** by reverse-mode AD:

- `MakeAdjoint::visit(AtomicOpStmt)` emits the adjoint `atomic_add` into the target field's adjoint SNode or ndarray grad pointer (`quadrants/transforms/auto_diff/make_adjoint.cpp:643-706`).
- `MakeAdjoint::visit(GlobalStoreStmt)` supports same-index read/write and clears the per-iteration adjoint slot (`quadrants/transforms/auto_diff/make_adjoint.cpp:574-641`).

The actual crash comes from the **`continue`** statement inside the dynamic `for s in range(self._n_sources)` loop. Python `continue` in a range-for lowers to IR `ContinueStmt`, which `ADTransform::visit` explicitly rejects at `quadrants/transforms/auto_diff/auto_diff_common.h:406-407` (and similarly `break` → `WhileControlStmt` at lines 402-403). Therefore the fastest unblock is to remove that control flow; the other options are architectural fallbacks.

---

### Option 1 — Rewrite source kernels to avoid `continue`/`break` (recommended near-term)

Replace the `continue` with a guarded body:

```python
@qd.kernel
def _compute_source_equilibrium_2d(self, f: qd.i32):
    for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
        for s in range(self._n_sources):
            dx = float(i) * self._dx - self._source_positions[i_b, s][0]
            dy = float(j) * self._dx - self._source_positions[i_b, s][1]
            dist_sq = dx * dx + dy * dy
            r_sq = self._source_radii[i_b, s] * self._source_radii[i_b, s]
            if dist_sq <= r_sq:
                # in-place update of per-source temperature (same index read+write is OK)
                self._source_temperatures[i_b, s] = ...
                # accumulate heat into the grid cell
                qd.atomic_add(self._T[f + 1, i_b, i, j], value)
```

Why this should work:

1. `IfStmt` is supported by the AD transform.
2. `qd.atomic_add(self._T[...], value)` is differentiable as long as `self._T` has an adjoint SNode (it does when `requires_grad=True`).
3. `self._source_temperatures[i_b, s] = ...` reads and writes the **same** indices in every iteration, which `OffloadLevelGlobalCrossIterRAWChecker` allows (`quadrants/transforms/auto_diff/validation.cpp:19-166`).

Apply the same rewrite to the 3D variant and audit the other skipped kernels (`_apply_energy_correction_*`, `_accumulate_source_body_energy_*`, `_compute_energy_after_*`, `_gather_source_cells_*`) for any `continue`/`break` and replace them with structured `if` guards.

Scope: only `plugins/solvers/thermal/core/thermal_solver.py`; no Quadrants changes needed.

---

### Option 2 — Custom gradient API (`qd.ad.grad_replaced` / `qd.ad.grad_for`)

If a kernel really needs early-exit control flow (or if the rewrite hurts performance), wrap the primal step in a `@qd.ad.grad_replaced` Python function and supply a hand-written `@qd.ad.grad_for` backward. During `Tape.grad()`, Quadrants skips the primal kernel and calls the user-provided backward instead (`python/quadrants/ad/_ad.py:304-377`, `python/quadrants/lang/kernel.py:898`).

Sketch:

```python
@qd.kernel
def _compute_source_equilibrium_2d(self, f: qd.i32):
    ...

@qd.kernel
def _compute_source_equilibrium_2d_grad(self, f: qd.i32):
    # hand-written reverse pass: propagate T.grad back to
    # _source_temperatures.grad, _source_positions.grad, _source_radii.grad
    ...

@qd.ad.grad_replaced
def _compute_source_equilibrium_2d_ad(self, f: qd.i32):
    self._compute_source_equilibrium_2d(f)

@qd.ad.grad_for(_compute_source_equilibrium_2d_ad)
def _compute_source_equilibrium_2d_ad_grad(self, f: qd.i32):
    self._compute_source_equilibrium_2d_grad(f)
```

A working example using `qd.atomic_add` inside a custom-grad kernel is `tests/python/test_customized_grad.py`.

Trade-off: correct adjoints must be derived and maintained by hand, but the API is stable and available today.

---

### Option 3 — Extend Quadrants reverse-mode AD to support `ContinueStmt`/`WhileControlStmt`

The proper upstream fix, but the most invasive. Rejection points include:

- `quadrants/transforms/auto_diff/auto_diff_common.h:402-412` (`ADTransform` base rejections).
- `quadrants/transforms/auto_diff/ir_shaping.cpp:338-348` (`IndependentBlocksJudger`).
- `quadrants/transforms/auto_diff/post_adjoint_cleanup.cpp:288`.

Approaches:

1. **Pre-lower control flow before AD**: run a pass that turns `continue`/`break` in dynamic loops into predicated execution (`if cond: body` or a loop-active flag + structured `if`). This is essentially what Option 1 does at the Python level, but done automatically in the compiler.
2. **Generate true control-flow adjoints**: record which iterations executed, then replay the loop in reverse with the same active mask. This is more general but requires careful handling of local variables and adjoint stacks.

Also update `validation.cpp` and the cleanup passes to accept the new statements.

---

### Option 4 — Forward-mode AD (`needs_dual=True`)

Quadrants forward-mode already supports `atomic_add` and same-index global store (`quadrants/transforms/auto_diff/make_dual.cpp:231-306`), but it inherits the same `ContinueStmt`/`WhileControlStmt` rejection in `auto_diff_common.h`. So forward-mode alone does not solve the problem unless Option 1's rewrite is also applied. It is useful only when the number of differentiable source parameters is tiny; otherwise reverse mode is preferred.

---

## Recommended path

1. **Immediate unblock**: apply Option 1 — rewrite `_compute_source_equilibrium_2d` (and its 3D counterpart) so the distance test uses `if dist_sq <= r_sq: ...` instead of `continue`. Then remove the `xfail` from `plugins/solvers/thermal/tests/test_thermal.py::TestThermalSolver::test_gradient_flows_to_source_temperature` and verify `thermal._compute_source_equilibrium_2d.grad(0)` populates `thermal._source_temperatures.grad`.
2. **Fallback for stubborn kernels**: if any energy-correction/accumulation kernel cannot be expressed cleanly without `continue`/`break`, implement Option 2 (custom gradient) for that specific kernel.
3. **Upstream tracking**: open a Quadrants feature request for Option 3 so that natural `break`/`continue` inside autodiff loops works out of the box.

## Why this matters

Without source-coupling gradients, the thermal solver can only optimize the initial temperature field. Robotics/ML use cases (learning source placement, material properties, or controller parameters from thermal observations) require gradients to flow back through the coupled bodies.

## Related project documentation

- `plugins/solvers/thermal/README.md`
- `docs/knowledge_base/06_thermal_solver_musa_autodiff.md`
- `plugins/solvers/thermal/tests/test_thermal.py::TestThermalSolver::test_gradient_flows_to_source_temperature` (marked `xfail`)
