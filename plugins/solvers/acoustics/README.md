# Acoustics Solver Plugin

Grid-based **time-domain acoustics solver** for genesis-world 1.4.0, in the
same plugin skeleton as [`plugins/solvers/thermal`](../thermal) and
[`plugins/solvers/joule_heating`](../joule_heating).

It solves the linear acoustic wave equation on a regular Cartesian grid with
a second-order leapfrog scheme and participates in the normal `gs.Scene`
simulation lifecycle (inject before `scene.build()`, step with `scene.step()`).

## Physics

```
p^{n+1} = 2 p^n - p^{-n} + (c·dt/dx)² · ∇²p^n + source terms
```

| Feature | Implementation | Commercial analogue |
|---|---|---|
| Time integration | Explicit leapfrog, 2nd order | Fluent direct method / Mechanical transient acoustics |
| Stability | CFL check `c·dt/dx ≤ 1/√dim` | — |
| Open boundary | Sponge damping layers (quadratic profile) | Fluent sponge layer / Mechanical PML |
| Pressure release | Dirichlet `p = 0` | — |
| Rigid wall | Neumann zero-gradient (full reflection) | Mechanical rigid FSI boundary |
| Monopole sources | `add_source()` (callable or sampled signal) | Fluent FW-H source surfaces |
| Vibro-acoustic coupling | `add_body()` — rigid-body velocity → `rho·dv/dt` injection (one-way) | Mechanical FSI (one-way variant) |
| Receivers | `add_probe()` — records pressure time history | FW-H virtual microphones |
| Post-processing | `probe.spectrum()` (Hann FFT → SPL dB), `probe.spl()` (rms SPL, 2×10⁻⁵ Pa ref) | Fluent acoustics FFT/SPL reports |

Grid rule of thumb: ≥ 6–10 cells per wavelength at the highest frequency of
interest (`dx ≤ c / (10·f_max)`).

## Usage

```python
import genesis as gs
from plugins.solvers.acoustics import AcousticsOptions, install

gs.init(backend=gs.cpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=4e-6), show_viewer=False)
scene.add_entity(gs.morphs.Plane())

acoustics = install(scene, AcousticsOptions(resolution=(200, 200), dx=0.0025))

acoustics.add_source(position=(0.5, 0.5, 0.0),
                     signal=lambda t: 5.0 * __import__("math").sin(2 * 3.1416 * 200 * t))
mic = acoustics.add_probe((0.7, 0.5, 0.0))

scene.build()
for _ in range(1200):
    scene.step()

freqs, spl = mic.spectrum(dt=4e-6)
print(mic.spl(dt=4e-6))       # overall SPL (dB)
print(acoustics.get_pressure())  # current pressure field
```

Rigid-body radiation (one-way vibro-acoustics):

```python
box = scene.add_entity(gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.25, 0.25, 0.05)))
acoustics.add_body(box, radius=0.01, amplitude=50.0)  # injects rho * dv_z/dt
```

See `examples/basic_usage.py` (whistle + two microphones, rigid vs absorbing
comparison, CSV waveforms and SPL spectra under `outputs/acoustics_demo/`) and
`tests/test_acoustics.py` (propagation delay, reflection, CFL, SPL/spectrum,
body coupling, 3D).

## Limitations

- Linear acoustics only (no mean flow, no amplitude-dependent effects);
  aeroacoustics source modelling (FWH-style analogy) is out of scope.
- One-way structure → sound coupling; acoustic pressure does not push back
  on bodies (two-way FSI would require the coupling forces to be applied back
  to the rigid solver).
- The leapfrog and boundary kernels are differentiable under
  `scene.requires_grad=True`; source/body injection is host-driven and its
  backward path is skipped for now (same `TODO(MUSA/autodiff)` limitation as
  the thermal source coupling).
- Overlapping injection regions of multiple sources should be avoided (cell
  contributions are added without atomics).
- Two time levels are stored (`_p`, `_p_prev`): memory ≈ 2× the thermal
  solver for the same grid.

## Notes on genesis 1.4 compatibility

`plugins/solvers/{thermal,joule_heating,acoustics}` were adapted to
genesis-world 1.4.0 (`Solver` no longer mixes in `TimeBasedMixin`; install
defaults `dt` from `scene._sim.dt`; quadrants 1.3 rejects field aliases
inside kernels). All three plugins' tests pass on CPU.
