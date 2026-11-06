# Classic Patent Simulations

The `cloud_robotics_sim.patents` package ports the interactive historical
patent demonstrators from [classic-patents.com](https://github.com/Dicklesworthstone/classic-patents.com)
into Genesis World physics. It is designed as a standalone demonstrator
framework, separate from the robotics Scene/Robot/Task composition API.

## Catalog

All 22 patents from classic-patents.com are registered and runnable. The
metadata lives in
`src/cloud_robotics_sim/patents/data/patents_catalog.yaml`.

| Patent ID | Title | Status |
| --- | --- | --- |
| **US1647** | **Electro-Magnetic Telegraph (Morse receiver)** | **full** |
| US3633 | India-Rubber Fabrics (Vulcanization) | stub |
| **US4750** | **Sewing Machine (Howe lockstitch kinematics)** | **full** |
| US6469 | Buoying Vessels Over Shoals | stub |
| **US174465** | **Improvement in Telegraphy (Bell magneto telephone)** | **full** |
| **US223898** | **Electric-Lamp (Edison carbon filament)** | **full** |
| **US381968** | **Electro-Magnetic Motor (Tesla two-phase induction)** | **full** |
| US586193 | Transmitting Electrical Signals | stub |
| **US593138** | **Electrical Transformer (Tesla coil, coupled resonance)** | **full** |
| **US821393** | **Flying-Machine (Wright Flyer)** | **full pilot** |
| **US1155986** | **Rocket Apparatus (Goddard two-stage)** | **full** |
| US1781541 | Refrigeration | stub |
| **US1773980** | **Television System (Farnsworth CRT raster)** | **full** |
| US2292387 | Secret Communication System | stub |
| **US2495429** | **Method of Treating Foodstuffs (Spencer microwave oven)** | **full** |
| **US2708656** | **Neutronic Reactor (Fermi point kinetics)** | **full** |
| US2524035 | Three-Electrode Circuit Element | stub |
| US2981877 | Semiconductor Device-and-Lead Structure | stub |
| US3541541 | X-Y Position Indicator | stub |
| US3671542 | Wholly Aromatic Polycarbonamide Filaments | stub |
| US3923554 | 3-Phase Charge-Coupled Device | stub |
| US4136359 | Microcomputer for Use with Video Display | stub |

"Stub" simulations build a minimal Genesis scene and run, but do not yet model
the underlying physics in detail. The Wright Flyer is fully implemented with a
lumped aerodynamics model.

### Wright Flyer parameters

| Parameter | Range | Default | Description |
|---|---|---|---|
| `wing_warp` | -1 .. 1 | 0 | Differential wing twist (roll control) |
| `rudder` | -1 .. 1 | 0 | Yaw control (only used when `coupled=0`) |
| `elevator` | -1 .. 1 | 0 | Canard pitch control |
| `thrust` | 0 .. 1 | 0 | Throttle |
| `wind_speed` | m/s | 0 | Ambient headwind |
| `coupled` | 0/1 | 1 | Claim 18 linkage: rudder follows `0.27 * wing_warp` |

With `coupled=1` (the historical default), the rudder is chained to the
wing-warping cradle exactly as described in Claim 18 of the patent: the
0.27 normalized ratio reproduces the original 0.45 deg/deg linkage over the
+/-15 deg warp and +/-25 deg rudder ranges.

## Quick Start

### Python API

```python
from cloud_robotics_sim.patents import create_simulation, PatentSimConfig

config = PatentSimConfig(
    patent_id="US821393",
    headless=True,
    dt=0.01,
    substeps=10,
    parameters={"thrust": 0.7, "wind_speed": 5.0},
)
sim = create_simulation("US821393", config=config)
sim.build()
sim.reset()

for _ in range(500):
    state = sim.step()
    print(state.metrics)

sim.close()
```

### CLI

```bash
# List all patents
uv run python -m cloud_robotics_sim patents --list

# Run the Wright Flyer with custom parameters
uv run python -m cloud_robotics_sim patents --run US821393 \
    --param thrust=0.7 --param wind_speed=5 --steps 500

# Run a stub patent
uv run python -m cloud_robotics_sim patents --run US223898 --steps 100
```

### Example Script

```bash
uv run python examples/patents/wright_flyer_demo.py --headless --steps 1000
```

## Implementing a New Patent Simulation

1. Copy `src/cloud_robotics_sim/patents/sims/_template.py` to a new module.
2. Replace `PATENT_ID`, metadata, class name, and title.
3. Implement `build()`, `reset()`, and `step()` using Genesis primitives and
   any patent-specific analytical model.
4. Add the patent to `data/patents_catalog.yaml`.
5. Import the new module in `src/cloud_robotics_sim/patents/sims/__init__.py`.
6. Add tests under `tests/patents/` and run the quality gates.

## Architecture

- `base.py` — `PatentSimConfig`, `PatentSimulation` ABC, `SimState`, and
  `StubPatentSimulation`.
- `registry.py` — `PatentRegistry` and decorators for discovery by ID.
- `runner.py` — convenience function for headless/recorded execution.
- `sims/` — one module per patent.
- `data/patents_catalog.yaml` — metadata for all 22 patents.

## Scope and Limitations

The original classic-patents.com site uses dedicated WebAssembly physics
kernels for aerodynamics, electromagnetics, nuclear kinetics, semiconductor
transport, and polymer mechanics. Genesis 1.3.2 is primarily a rigid/soft-body
simulator, so most patents are modeled as **illustrative physical
demonstrators** rather than first-principles replicas. The framework is
intentionally extensible: future fidelity improvements can be added
incrementally inside each simulation module.
