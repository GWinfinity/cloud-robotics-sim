# JouleHeatingSolver

Grid-based Joule-heating (electro-thermal) solver plugin for `genesis-world 1.3.2`.

The solver solves the quasi-static electric field

```
∇ · (σ ∇V) = 0
```

on a regular Cartesian grid using Jacobi iteration, then computes the
volumetric heat source

```
Q = σ |∇V|² = |J|² / σ
```

where `J = -σ ∇V` is the current density. The heat source can either drive a
small internal transient thermal solve or be injected into the existing
`ThermalSolver` for full thermal diffusion.

## Features

- 2D and 3D regular Cartesian grids.
- Per-cell electrical conductivity `σ(x)`.
- Dirichlet voltage boundary conditions on any domain face.
- Zero-Neumann (insulated) boundaries where no Dirichlet value is set.
- Optional coupling to `plugins.solvers.thermal.ThermalSolver`.
- Batched environments (`scene.build(n_envs=...)`).
- Differentiable electric-potential solve when `scene.requires_grad=True`.

## Installation

Install before `scene.build()`:

```python
import genesis as gs
from plugins.solvers.joule_heating import install, JouleHeatingOptions

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=1))

joule = install(
    scene,
    JouleHeatingOptions(
        resolution=(64, 16),
        dx=0.01,
        sigma=5.8e7,   # copper-like
        rho=8960.0,
        cp=385.0,
        k=400.0,
        max_iter=1000,
        tol=1e-6,
    ),
)

scene.build()
```

## Basic usage

```python
# Apply 1 V on the left face and ground the right face.
joule.set_voltage_boundary("x_min", 1.0)
joule.set_voltage_boundary("x_max", 0.0)

# Optionally override the conductivity field.
import numpy as np
sigma = np.full((64, 16), 1.0e6, dtype=float)
joule.set_conductivity(sigma)

scene.step()

voltage = joule.get_voltage()       # [V]
current = joule.get_current_density()  # [A/m²]
source = joule.get_heat_source()    # [W/m³]
```

## Coupling with ThermalSolver

To inject the Joule heat source into `ThermalSolver`, set
`couple_to_thermal=True` and install the thermal solver **before** the Joule
solver:

```python
from plugins.solvers.thermal import install as install_thermal
from plugins.solvers.thermal import ThermalOptions

thermal = install_thermal(
    scene,
    ThermalOptions(
        resolution=(64, 16),
        dx=0.01,
        alpha=1e-4,
        boundary_mode="neumann",
        initial_temperature=300.0,
    ),
)

joule = install(
    scene,
    JouleHeatingOptions(
        resolution=(64, 16),
        dx=0.01,
        sigma=5.8e7,
        rho=8960.0,
        cp=385.0,
        k=400.0,
        couple_to_thermal=True,
    ),
)

scene.build()

joule.set_voltage_boundary("x_min", 10.0)
joule.set_voltage_boundary("x_max", 0.0)

for _ in range(100):
    scene.step()

print(thermal.get_temperature().max())
```

The Joule solver adds `Q * dt / (rho * cp)` to the thermal solver's next
frame. Because the simulator runs solvers in install order, installing the
thermal solver first and the Joule solver second guarantees that the heat
source is applied after the diffusion step and then checkpointed forward.

## Motor catalog integration

The project includes a robot motor catalog in
``src/cloud_robotics_sim/data/robot_motors.yaml``. You can query it from Python
to seed Joule-heating or thermal simulations:

```python
from cloud_robotics_sim.utils.motor_catalog import (
    get_motor,
    estimate_joule_power,
    estimate_resistance_from_motor,
    get_thermal_defaults,
)

motor = get_motor("hip/knee_large")
print(motor["voltage_v"], motor["current_a"], motor["power_w"])

# Estimate resistive heating at 80 A with a 0.05 Ω winding.
P_heat = estimate_joule_power(80.0, 0.05)  # 320 W

# Rough resistance estimate from catalog bounds.
R_est = estimate_resistance_from_motor("hip/knee_large")

# Default thermal properties for copper/steel/insulation.
thermal = get_thermal_defaults()
sigma_copper = thermal["copper"]["conductivity_s_per_m"]
```

## API

### `JouleHeatingOptions`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | `None` | Solver time step; defaults to `scene.sim_options.dt`. |
| `dim` | `2` | Spatial dimension (2 or 3). |
| `resolution` | `(64, 64)` | Grid resolution. |
| `dx` | `0.01` | Grid spacing [m]. |
| `sigma` | `5.8e7` | Electrical conductivity [S/m]. |
| `rho` | `8960.0` | Mass density [kg/m³]. |
| `cp` | `385.0` | Specific heat capacity [J/(kg·K)]. |
| `k` | `400.0` | Thermal conductivity [W/(m·K)]. |
| `initial_temperature` | `300.0` | Initial temperature for internal thermal solve [K]. |
| `voltage_boundary_value` | `0.0` | Default voltage used to initialize the field. |
| `max_iter` | `1000` | Jacobi iterations per substep. |
| `tol` | `1e-6` | Unused in v1 (fixed-iteration solve). |
| `couple_to_thermal` | `False` | Inject `Q` into `scene.sim.thermal_solver`. |

### `JouleHeatingSolver`

- `set_voltage_boundary(name, value)` — set Dirichlet voltage on a face.
  `name` is one of `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`.
- `set_conductivity(sigma)` — set the per-cell conductivity field.
- `set_voltage(voltage)` / `get_voltage()` — set/get the potential field [V].
- `get_current_density()` — get `J` [A/m²].
- `get_heat_source()` — get `Q` [W/m³].
- `set_temperature(temperature)` / `get_temperature()` — internal thermal
  field [K] (only when `couple_to_thermal=False`).

## Limitations

- The electric solve uses a fixed number of Jacobi iterations (`max_iter`).
  Convergence is not checked online; choose `max_iter` large enough for your
  grid.
- Dirichlet voltage boundaries are supported; current (Neumann) boundaries
  are not.
- Gradients w.r.t. boundary voltage values are straight-through in v1.
- The internal thermal solve uses simple explicit FTCS with insulated
  boundaries. For production thermal diffusion, use `couple_to_thermal=True`
  with `ThermalSolver`.
