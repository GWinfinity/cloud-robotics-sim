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

The `robomat` material library is an optional local sibling package (`../robomat`, not on PyPI) wired through `[tool.uv.sources]`. It lives in the `robomat` extra so that pip-based CI (`pip install -e ".[dev]"`) works; install it locally with `uv sync --extra dev --extra robomat`. All robomat imports are guarded and degrade to simulator defaults when absent.

## Supported Python Versions

- `requires-python = ">=3.10,<3.14"`
- CI tests Python 3.10, 3.11, and 3.12.
- Local development may use Python 3.13, but it is not yet in CI.

## Genesis Version

The project pins `genesis-world==1.4.0` (deterministic CPU/GPU simulation, improved ill-conditioned mass-matrix robustness, islands support, and the scene/trajectory `gs replay` archive). Do not use Genesis 0.4.x APIs.
Use the compatibility helpers in `src/cloud_robotics_sim/utils/genesis_compat.py` for backend selection, light creation, and external force application (`apply_entity_force`, which wraps the 1.4.0 `RigidLink.apply_external_force` wrench API with a legacy `apply_force` fallback).

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

## Robot Assets and Configuration

`configs/franka_pickplace.yaml` is consumed by the improvement loop via `load_sim_config` (`core/config_loader.py`), which validates the required `environment.scene/robot/task/simulation` structure. Robot model paths (`robot.urdf_path`) are resolved by `core/robot_assets.py`: explicit path -> bundled `assets_genesis/embodiments/*.urdf` (gitignored, optional local checkout) -> auto-shallow-cloned `robot-descriptions/awesome-robot-descriptions` (AtomGit mirror `https://atomgit.com/gh_mirrors/aw/awesome-robot-descriptions` first, GitHub fallback; `CRS_ROBOT_DESC_AUTO_DOWNLOAD=0` disables, `CRS_ROBOT_DESC_DIR` overrides location, prefetch with `python -m cloud_robotics_sim.core.robot_assets`) -> Genesis built-in lookup -> procedural placeholder box with a loud warning. After `spawn()`, `robot.asset_source` records which source produced the entity (`mjcf:<path>` / `urdf:<path>` / `procedural-box`).

## ManiSkill ReplicaCAD Scene Dataset

`plugins/envs/maniskill` integrates ManiSkill's ReplicaCAD scene dataset (the `ReplicaCAD_SceneManipulation-v1` scenes used by XLerobot) mirrored on ModelScope as `jessy888/ManiSkill_replica_cad_dataset` (single ~289 MB `replica_cad_dataset.zip`, Apache-2.0). `genesis_maniskill/datasets/replicacad_assets.py` (same pattern as `robotwin/assets.py`) checks/downloads/extracts it on demand to `assets/maniskill/` (gitignored): `ensure_dataset()` / `list_scenes()` / `download_dataset()` plus a CLI (`python -m genesis_maniskill.datasets.replicacad_assets`). Env vars: `CRS_REPLICACAD_ASSETS` (root override), `CRS_REPLICACAD_AUTO_DOWNLOAD=0` (disable; missing dataset then raises `FileNotFoundError`). `scenes/replica_cad_scene.py::ReplicaCADSceneBuilder` ports ManiSkill's ReplicaCADSceneBuilder semantics (RotX(+90°) Y-up→Z-up conversion, habitat xyzw → Genesis wxyz quats, DYNAMIC objects use the pre-decomposed convex collision GLB as body mesh, STATIC objects as non-convex fixed meshes, articulated furniture from `urdf/<name>/<name>.urdf`, doors skipped by default, masses applied via `set_mass()` in `finalize()` after `scene.build()`). `envs/replica_cad_env.py::ReplicaCADEnv` is the gym wrapper (num_envs=1 only; `scene.build()` is deferred to `_setup_spaces` so robot/cameras join before compilation). Example: `python plugins/envs/maniskill/examples/replica_cad_scene.py --scene apt_0 --save-img`. Note: the plugin's `datasets` subpackage uses absolute imports and is only importable under the top-level `genesis_maniskill` name with `plugins/envs/maniskill/core` on `sys.path` (as the plugin examples do); `scenes/replica_cad_scene.py` therefore imports it lazily.

## 3D Asset Generation APIs (asset_gen)

`src/cloud_robotics_sim/asset_gen/` is a stdlib-only (`urllib.request`) client layer for commercial text/image-to-3D services, with a unified `GenerationRequest`/`GenerationResult` model, a provider registry (`register_provider`/`get_provider`/`list_providers`), shared async-task polling (`base.poll_task`), and a facade (`AssetGenClient.submit_text`/`generate_image`/`generate`) that downloads the GLB plus a `.json` provenance file into a staging dir ready for `scripts/expand_object_library.py`. Built-in providers: `tripo` (Tripo AI V2 OpenAPI; V2 EOL 2026-11-01, migrate via `TRIPO_BASE_URL`), `meshy` (Meshy v2 text-to-3d / v1 image-to-3d), `hunyuan3d` (Tencent Cloud `ai3d` service, API version 2025-05-13, full TC3-HMAC-SHA256 signing implemented in-module; pro jobs by default, `extra={"rapid": True}` for the rapid tier), and `rodin` (Hyper3D v2 multipart submit + status/download). Credentials come from env vars (`MESHY_API_KEY`, `TRIPO_API_KEY`, `RODIN_API_KEY`, `TENCENTCLOUD_SECRET_ID`/`TENCENTCLOUD_SECRET_KEY`, optional `HY3D_REGION`); a project `.env` is loaded by the CLI when `python-dotenv` is installed. CLI: `python -m cloud_robotics_sim.asset_gen providers|text|image`. Tests in `tests/asset_gen/` mock all HTTP and pin the TC3 signature against Tencent's official worked example.

## Grid Field Solver Plugins (thermal / joule_heating / acoustics)

`plugins/solvers/` holds three structurally identical grid-based field solver plugins injected into a `gs.Scene` before `scene.build()` (`install(scene, options)` sets `scene.sim.<name>_solver` and appends to `scene.sim._solvers`): **thermal** (explicit FTCS conduction with energy-conserving entity coupling), **joule_heating** (Jacobi electric potential → `Q = σ|∇V|²`, optional `couple_to_thermal=True` injection), and **acoustics** (second-order leapfrog linear wave equation; absorbing sponge layers / Dirichlet p=0 / Neumann rigid boundaries; `add_source()` monopoles, `add_body()` one-way vibro-acoustic velocity injection `ρ·dv_z/dt`, `add_probe()` virtual microphones with Hann-FFT `spectrum()`/rms `spl()` post-processing at 2×10⁻⁵ Pa; CFL `c·dt/dx ≤ 1/√dim`). All three were adapted to genesis-world 1.4.0 quirks: `Solver` no longer mixes in `TimeBasedMixin` (each solver sets `_substep_dt = options.dt` in `build()`), `install()` defaults `dt` from `scene._sim.dt`, and quadrants 1.3 rejects field aliases inside kernels (`x = self._field` then `x[i]`) and cannot autodiff host-precomputed constant fields (material params must be inlined into kernels, cf. `joule_heating._heat_step_*`). Source-coupling backward paths remain skipped pending the upstream quadrants AD fixes (see `scripts/quadrants_issue_thermal_source_autodiff.md`). Tests live in each plugin's `tests/` (CPU-only; run per plugin — each module calls `gs.init`), examples in `examples/basic_usage.py`.

## Agent Runtime (skill-based)

`src/cloud_robotics_sim/runtime/skills.py` is the agent-facing skill layer: `Skill` (name + description + JSON-Schema parameters + handler), `SkillRegistry` (validate/dispatch), `TaskExecutor` (executes skills, resolves natural-language goals deterministically via token matching with per-skill parameter extractors, e.g. patent IDs, and records every run), and `ReplayBuffer` (in-memory + optional JSONL). Built-in skills: `run_patent`, `list_patents`, `list_scenes/robots/tasks`, `submit_sim_task`. Register custom skills with `@register_skill`. `runtime/agent_hub.py` exposes the same surface as zero-dependency MCP-style tools (`SimHub`, `list_tools`/`call_tool`, optional `serve_stdio` via the `mcp` package) mirroring `devices/mcp_adapter.py`; the CLI `cloud-robotics-sim agent --goal/--skill/--param/--list-skills` is a thin wrapper. `queue_worker` delegates patent execution to `skills.run_patent_task` so sync (skill) and async (KEDA queue) paths share one implementation.

## Kubernetes Deployment

`deploy/kubernetes/` contains native YAML manifests (no Helm) for queue-based autoscaling: a Redis queue Deployment, CPU/GPU sim-worker Deployments, and KEDA `ScaledObject`s that scale each worker tier from 0 based on its Redis list length (`sim-tasks-cpu` / `sim-tasks-gpu` are separate queues so one queue never scales both tiers). The worker entry point is `cloud-robotics-sim worker` (`src/cloud_robotics_sim/runtime/queue_worker.py`), which consumes JSON task specs via `BRPOP` and appends results to the `sim-results` Redis list. Task types are registered in `_TASK_HANDLERS` (currently `patent`). The `redis` client is an optional dependency behind the `k8s` extra (`pip install cloud-robotics-sim[k8s]`); it is never imported at package top level. `scripts/k8s_submit_task.py` pushes tasks onto a queue. See `docs/guides/kubernetes.md` for the full guide.

## Meshy T2 Module

`src/cloud_robotics_sim/meshy_t2/` is a pure-PyTorch reproduction of the Meshy T2 paper (arXiv:2607.28675; official code unreleased): vertex-set Mesh VAE (spacetime edge logits + halfedge successor faces with NULL extension + Sinkhorn), Voxel VAE, and a two-stage Rectified-Flow cascade (voxel scaffold flow -> per-vertex latent flow with Sobol-OT 3D RoPE positions, existence channel and vertex-count conditioning). It is torch/trimesh/scipy-only (no Genesis import). Model defaults match the paper; `*.tiny()` configs run on CPU. Train with `python -m cloud_robotics_sim.meshy_t2 train {voxel-vae,mesh-vae,voxel-flow,mesh-flow} --tiny`, generate with `... generate image.png --num-faces 4000`; tests live in `tests/meshy_t2/`. mypy `warn_return_any` is disabled for this module in `pyproject.toml` (torch stubs return `Any` pervasively).

## Simulated Devices Module (MHS-style)

`src/cloud_robotics_sim/devices/` is an MHS-inspired device abstraction layer: every simulated device exposes typed read/write primitives (`SimDevice.read`/`write`, validated against `WritePrimitive` safety bounds, raising `SafetyViolationError`), carries `compliance` GB-standard metadata, and generates an agent-facing reference file via `reference_file_yaml()` from the standards catalogs in `data/standards/` (`robot_standards_catalog.yaml`, `lab_equipment_standards.yaml`). Fault models in `faults.py` (`SensorDrift`, `SensorStuck`, `RelayStuckClosed`, `InterlockBypass`) support safety-evaluation scenarios. Devices: `MuffleFurnace` (lumped thermal, door interlock + latching over-temperature cutout per GB 5959.4-2008), `UniversalTestingMachine` (piecewise stress-strain specimen + frame compliance solved by bisection; GB/T 16491-2022 overload/fracture auto-stop, GB/T 228.1-2021 extensometer handling), `SharpaHandDevice` (Genesis-backed, vendored URDF from `plugins/do_as_i_do/assets/sharpa-urdf-usd-xml-main/` — `package://` mesh paths are rewritten to absolute paths before loading; fingertip links are merged into `*_DP` distal-phalanx links by Genesis; PFL force guard per GB/T 36008-2018). `mcp_adapter.py` exposes a `DeviceHub` over MCP-style tools (`list_tools`/`call_tool`, zero-dependency; `serve_stdio` needs the optional `mcp` package). Register new devices in `devices/__init__.py` `DEVICE_TYPES`; tests live in `tests/devices/` (Genesis hand tests take ~30s on CPU).

## Classic Patents Module

`src/cloud_robotics_sim/patents/` ports the 22 interactive patent demonstrators from classic-patents.com into Genesis. The Wright Flyer (`US821393`), Edison's Electric-Lamp (`US223898`, lumped electro-thermal model: negative-TCR carbon resistance, Stefan-Boltzmann radiation, Arrhenius filament wear with vacuum-dependent oxidation, Planck/photopic luminous flux) and Tesla's Electro-Magnetic Motor (`US381968`, double-revolving-field induction model: phase-offset-dependent forward/backward field decomposition, Kloss torque-slip curve, lumped rotor mechanics) and Morse's Electro-Magnetic Telegraph (`US1647`, RL line charging, Maxwell armature attraction with pull-in/drop-out hysteresis, key-timing Morse decoder), Tesla's Electrical Transformer (`US593138`, coupled resonant LC circuits integrated exactly with the matrix exponential, spark-gap fire/quench cycle, tuning-dependent voltage gain) and Goddard's Rocket Apparatus (`US1155986`, variable-mass Tsiolkovsky flight with exponential-atmosphere drag, hold-down clamps, hot staging and a ballistically falling booster), Fermi & Szilard's Neutronic Reactor (`US2708656`, six-group delayed-neutron point kinetics with negative-temperature self-regulation and scram tail), Bell's Telephone (`US174465`, magneto transmitter/receiver pair: diaphragm-driven variable-reluctance EMF over an RL line), Farnsworth's Television System (`US1773980`, electron gun with magnetic deflection raster scanning), Howe's Sewing Machine (`US4750`, slider-crank needle bar with shuttle lockstitch timing and a thread-tension stitch/miss/break window) and Spencer's Microwave Oven (`US2495429`, standing-wave cavity heating of a 3x3 food grid with boiling plateau and turntable uniformity) are fully implemented; the other 11 patents ship as runnable stubs under `patents/sims/` registered via `@register_patent`. Metadata lives in `patents/data/patents_catalog.yaml`; run simulations with `uv run python -m cloud_robotics_sim patents --list` / `--run US821393`. In genesis-world 1.4.0, rigid bodies expose `control_dofs_force([Fx,Fy,Fz,Tx,Ty,Tz])` (world frame) instead of `apply_force` (external forces go through `RigidLink.apply_external_force`/`apply_external_wrench` — use the `apply_entity_force` compat helper), use `rho=` (not `density=`) in `gs.materials.Rigid`, and cameras must be added before `scene.build()`.

The Wright Flyer visual is the Smithsonian NASM CC0 scan (`assets/patents/wright_flyer/smithsonian-nasm-1903-flyer.stl`, ~14 MB, gitignored; download from `https://cdn.jsdelivr.net/gh/Dicklesworthstone/classic-patents.com@main/public/models/wright-flyer/smithsonian-nasm-1903-flyer.cc0.stl` — jsDelivr mirror works where raw.githubusercontent.com stalls). Missing mesh falls back to a box. Genesis gotchas hit here: mesh rigid bodies get their mass from the *convex-decomposed* collision volume (or the convex hull for non-watertight scans with `collision=False`), which inflated the 340 kg Flyer to 8.4–17.4 t — fix by calling `entity.set_mass()` after `scene.build()`, which scales the inertia tensor by the same ratio. CoACD decomposition of the 297k-face STL takes ~2.5 min once, then is cached in `~/.cache/genesis/cvx/`. Angular velocity comes from `entity.get_ang()` (world frame) — `get_angular_velocity()` does not exist, so `hasattr` guards around it silently disable any damping that relies on it.
