# Thermal Solver Plugin

A grid-based heat-conduction solver that runs as a plugin on top of `genesis-world==1.3.2`.

## Overview

- **Backend**: `quadrants` (the same backend used by Genesis 1.3.2).
- **Numerical method**: explicit FTCS (Forward-Time Centered-Space) for transient problems; optional steady-state Jacobi placeholder.
- **Boundary conditions**: Dirichlet or Neumann (zero flux).
- **Dimension**: 2D is fully implemented and tested; 3D kernels are present but not exhaustively tested.
- **Coupling**: Genesis entities (Rigid, MPM, SPH, PBD, FEM) can be registered as thermally coupled bodies that exchange heat with nearby grid cells in an energy-conserving way.

## Usage

```python
import genesis as gs
from plugins.solvers.thermal import install, ThermalOptions

gs.init(backend=gs.cpu)

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01))
scene.add_entity(gs.morphs.Plane())

box = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.32, 0.32, 0.02)),
    material=gs.materials.Rigid(),
)

# Install the thermal solver before scene.build()
thermal = install(
    scene,
    ThermalOptions(
        resolution=(64, 64),
        dx=0.01,
        alpha=1e-4,
        boundary_mode="dirichlet",
        boundary_value=0.0,
        grid_rho=1.0,
        grid_cp=1.0,
        initial_temperature=0.0,
    ),
)

# Couple the rigid box as a 1.0 K body with heat capacity 0.1 J/K.
thermal.add_source(
    entity=box,
    temperature=1.0,
    radius=0.05,
    rate=10.0,
    heat_capacity=0.1,
)

scene.build()

for _ in range(100):
    scene.step()

T = thermal.get_temperature()  # np.ndarray of shape (64, 64)
```

## Installation / Registration

`genesis-world 1.3.2` does not expose a public solver-registration API. This plugin therefore injects the solver into `scene.sim._solvers` before `scene.build()`. The simulator lifecycle then picks it up automatically.

## Thermal Coupling Model

`add_source(entity, temperature, radius, rate, heat_capacity)` registers a thermally coupled body. Each substep:

1. Grid cells within `radius` of the entity's current position are identified.
2. The coupled body and those cells relax toward a common equilibrium temperature
   `T_eq = (C_body*T_body + sum(C_cell*T_cell)) / (C_body + sum(C_cell))`.
3. Both body and cells are updated by the same fractional step `alpha = min(rate*dt, 1.0)`:
   `T_new = T_old + alpha * (T_eq - T_old)`.

Because body and cells move toward the **same** equilibrium by the **same** fraction, total thermal energy is conserved for a single, non-overlapping source. The cell-influence detection and temperature update are implemented as `quadrants` kernels, so the full temperature field is no longer copied to NumPy every substep.

## Features

- **Build-time independent source registration**: `add_source()` may be called before
  or after `scene.build()` (up to `ThermalOptions.max_sources`).
- **Batched environments**: fully supports `scene.build(n_envs=...)`; each environment
  maintains its own source body temperatures and evolves independently.
- **Strict energy conservation**: overlapping source influence regions conserve total
  thermal energy to machine precision via a deterministic energy-correction step.
- **Differentiability**: the thermal field remains differentiable with respect to the
  initial temperature field when `scene.requires_grad=True`. Source-coupling gradients
  (body temperature, position, etc.) are treated as constants during back-propagation
  in this version; the forward physics is still exact.

## Limitations

- Source parameters (body temperature, position, radius, rate, heat capacity) are
  treated as constants during back-propagation. Gradient w.r.t. the initial
  temperature field and the diffusion physics is fully supported.
- Cross-solver gradients (thermal loss flowing back into rigid-body DOFs) are not
  yet supported.
