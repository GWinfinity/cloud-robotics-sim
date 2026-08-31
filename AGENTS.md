# Agent Development Notes

## Environment

This project uses `uv` for dependency management. The lockfile is `uv.lock`.

```bash
# Sync dependencies (including dev tools)
uv sync --extra dev

# Run commands inside the managed venv
uv run python -m pytest tests/
uv run python -m ruff check src/ tests/
uv run python -m black --check src/ tests/
uv run python -m mypy src/cloud_robotics_sim
```

On Windows, the venv interpreter is at `.venv/Scripts/python.exe`. Avoid using the system Python or external virtual environments.

## Supported Python Versions

- `requires-python = ">=3.10,<3.14"`
- CI tests Python 3.10, 3.11, and 3.12.
- Local development may use Python 3.13, but it is not yet in CI.

## Genesis Version

The project pins `genesis-world==1.3.2` (chosen for its deterministic CPU/GPU simulation, improved ill-conditioned mass-matrix robustness, and islands support). Do not use Genesis 0.4.x APIs.
Use the compatibility helpers in `src/cloud_robotics_sim/utils/genesis_compat.py` for backend selection and light creation.

## Quality Gates

- **Lint**: `ruff check src/ tests/`
- **Format**: `black --check src/ tests/` (or `black src/ tests/` to fix)
- **Type check**: `mypy src/cloud_robotics_sim`
- **Tests**: `pytest tests/`

CI treats all of these as blocking.

## Notes

- Deployment installs should use `tools/install_torch.py` to pick the right PyTorch wheels: it auto-detects Moore Threads MUSA hardware (installs `torch_musa`), CUDA, or CPU, and switches to Aliyun mirrors when the official PyTorch index is unreachable (mainland China). Docker builds expose this via the `TORCH_BACKEND` and `CHINA_MIRROR` build arguments.
- RoboTwin 2.0 assets (~4GB, gitignored under `assets/robotwin/`) are checked and downloaded automatically on first use via `src/cloud_robotics_sim/robotwin/assets.py` (HF `TianxingChen/RoboTwin2.0`). Prefetch with `python -m cloud_robotics_sim.robotwin.assets`; set `CRS_ROBOTWIN_AUTO_DOWNLOAD=0` to disable and `HF_ENDPOINT=https://hf-mirror.com` for the China mirror. Tests that hit the network are not allowed — mock `download_component`/`ensure_for_path` instead.
- `plugins/envs/sky/core/genesis/` is a vendored old Genesis 0.3.11 copy. Do not import it from core code.
- Keep `uv.lock` in sync with `pyproject.toml` by running `uv lock` after dependency changes.
- Genesis simulation cache cleanup is handled by `cloud_robotics_sim.utils.cache_cleanup`. Set `CRS_DATASET_PIPELINE=true` to stage cache into `outputs/dataset_pipeline/staging` for a downstream dataset pipeline; otherwise `env.close()` / `cloud-robotics-sim clean-cache` deletes `outputs/sim_cache` and releases Genesis runtime resources. Override paths with `CRS_SIM_CACHE_DIR` and `CRS_DATASET_PIPELINE_DIR`.

## Meshy T2 Module

`src/cloud_robotics_sim/meshy_t2/` is a pure-PyTorch reproduction of the Meshy T2 paper (arXiv:2607.28675; official code unreleased): vertex-set Mesh VAE (spacetime edge logits + halfedge successor faces with NULL extension + Sinkhorn), Voxel VAE, and a two-stage Rectified-Flow cascade (voxel scaffold flow -> per-vertex latent flow with Sobol-OT 3D RoPE positions, existence channel and vertex-count conditioning). It is torch/trimesh/scipy-only (no Genesis import). Model defaults match the paper; `*.tiny()` configs run on CPU. Train with `python -m cloud_robotics_sim.meshy_t2 train {voxel-vae,mesh-vae,voxel-flow,mesh-flow} --tiny`, generate with `... generate image.png --num-faces 4000`; tests live in `tests/meshy_t2/`. mypy `warn_return_any` is disabled for this module in `pyproject.toml` (torch stubs return `Any` pervasively).

## Simulated Devices Module (MHS-style)

`src/cloud_robotics_sim/devices/` is an MHS-inspired device abstraction layer: every simulated device exposes typed read/write primitives (`SimDevice.read`/`write`, validated against `WritePrimitive` safety bounds, raising `SafetyViolationError`), carries `compliance` GB-standard metadata, and generates an agent-facing reference file via `reference_file_yaml()` from the standards catalogs in `data/standards/` (`robot_standards_catalog.yaml`, `lab_equipment_standards.yaml`). Fault models in `faults.py` (`SensorDrift`, `SensorStuck`, `RelayStuckClosed`, `InterlockBypass`) support safety-evaluation scenarios. Devices: `MuffleFurnace` (lumped thermal, door interlock + latching over-temperature cutout per GB 5959.4-2008), `UniversalTestingMachine` (piecewise stress-strain specimen + frame compliance solved by bisection; GB/T 16491-2022 overload/fracture auto-stop, GB/T 228.1-2021 extensometer handling), `SharpaHandDevice` (Genesis-backed, vendored URDF from `plugins/do_as_i_do/assets/sharpa-urdf-usd-xml-main/` — `package://` mesh paths are rewritten to absolute paths before loading; fingertip links are merged into `*_DP` distal-phalanx links by Genesis; PFL force guard per GB/T 36008-2018). `mcp_adapter.py` exposes a `DeviceHub` over MCP-style tools (`list_tools`/`call_tool`, zero-dependency; `serve_stdio` needs the optional `mcp` package). Register new devices in `devices/__init__.py` `DEVICE_TYPES`; tests live in `tests/devices/` (Genesis hand tests take ~30s on CPU).

## Classic Patents Module

`src/cloud_robotics_sim/patents/` ports the 22 interactive patent demonstrators from classic-patents.com into Genesis. The Wright Flyer (`US821393`), Edison's Electric-Lamp (`US223898`, lumped electro-thermal model: negative-TCR carbon resistance, Stefan-Boltzmann radiation, Arrhenius filament wear with vacuum-dependent oxidation, Planck/photopic luminous flux) and Tesla's Electro-Magnetic Motor (`US381968`, double-revolving-field induction model: phase-offset-dependent forward/backward field decomposition, Kloss torque-slip curve, lumped rotor mechanics) and Morse's Electro-Magnetic Telegraph (`US1647`, RL line charging, Maxwell armature attraction with pull-in/drop-out hysteresis, key-timing Morse decoder), Tesla's Electrical Transformer (`US593138`, coupled resonant LC circuits integrated exactly with the matrix exponential, spark-gap fire/quench cycle, tuning-dependent voltage gain) and Goddard's Rocket Apparatus (`US1155986`, variable-mass Tsiolkovsky flight with exponential-atmosphere drag, hold-down clamps, hot staging and a ballistically falling booster), Fermi & Szilard's Neutronic Reactor (`US2708656`, six-group delayed-neutron point kinetics with negative-temperature self-regulation and scram tail), Bell's Telephone (`US174465`, magneto transmitter/receiver pair: diaphragm-driven variable-reluctance EMF over an RL line), Farnsworth's Television System (`US1773980`, electron gun with magnetic deflection raster scanning), Howe's Sewing Machine (`US4750`, slider-crank needle bar with shuttle lockstitch timing and a thread-tension stitch/miss/break window) and Spencer's Microwave Oven (`US2495429`, standing-wave cavity heating of a 3x3 food grid with boiling plateau and turntable uniformity) are fully implemented; the other 11 patents ship as runnable stubs under `patents/sims/` registered via `@register_patent`. Metadata lives in `patents/data/patents_catalog.yaml`; run simulations with `uv run python -m cloud_robotics_sim patents --list` / `--run US821393`. In Genesis 1.3.2, rigid bodies expose `control_dofs_force([Fx,Fy,Fz,Tx,Ty,Tz])` (world frame) instead of `apply_force`, use `rho=` (not `density=`) in `gs.materials.Rigid`, and cameras must be added before `scene.build()`.

The Wright Flyer visual is the Smithsonian NASM CC0 scan (`assets/patents/wright_flyer/smithsonian-nasm-1903-flyer.stl`, ~14 MB, gitignored; download from `https://cdn.jsdelivr.net/gh/Dicklesworthstone/classic-patents.com@main/public/models/wright-flyer/smithsonian-nasm-1903-flyer.cc0.stl` — jsDelivr mirror works where raw.githubusercontent.com stalls). Missing mesh falls back to a box. Genesis gotchas hit here: mesh rigid bodies get their mass from the *convex-decomposed* collision volume (or the convex hull for non-watertight scans with `collision=False`), which inflated the 340 kg Flyer to 8.4–17.4 t — fix by calling `entity.set_mass()` after `scene.build()`, which scales the inertia tensor by the same ratio. CoACD decomposition of the 297k-face STL takes ~2.5 min once, then is cached in `~/.cache/genesis/cvx/`. Angular velocity comes from `entity.get_ang()` (world frame) — `get_angular_velocity()` does not exist, so `hasattr` guards around it silently disable any damping that relies on it.
