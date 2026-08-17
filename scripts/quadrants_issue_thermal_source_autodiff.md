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

Any one of the following would unblock us:

1. **Native reverse-mode support** for `qd.atomic_add`, in-place field writes, and loop `break`/`continue` in autodiff kernels.
2. **Custom gradient API** (`@qd.ad.grad_replaced` / `@qd.ad.grad_for`) that is stable enough to let us hand-write the adjoint of the source-coupling step.
3. **Forward-mode AD** (`needs_dual=True`) support for the same patterns, which would be sufficient when the number of design parameters (source temperatures, etc.) is small.

## Why this matters

Without source-coupling gradients, the thermal solver can only optimize the initial temperature field. Robotics/ML use cases (learning source placement, material properties, or controller parameters from thermal observations) require gradients to flow back through the coupled bodies.

## Related project documentation

- `plugins/solvers/thermal/README.md`
- `docs/knowledge_base/06_thermal_solver_musa_autodiff.md`
- `plugins/solvers/thermal/tests/test_thermal.py::TestThermalSolver::test_gradient_flows_to_source_temperature` (marked `xfail`)
