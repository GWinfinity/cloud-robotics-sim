# RoboTwin Integration

This package replays demonstrations collected with the RoboTwin benchmark
(SAPIEN 3.0.0b1) inside the Genesis-backed simulation environment.

## Components

- `bridge.py`: Simulator-agnostic data structures (`RobotwinBridge`,
  `RobotwinFrame`, `ObjectAsset`, `CameraConfig`) and pickle serialization.
- `loader.py`: `RobotwinBridgeLoader` for iterating over bridge frames.
- `dual_arm_embodiment.py`: `AlohaAgileX` embodiment that maps the RoboTwin
  14-D action space to Genesis-driven DoFs.
- `replay_scene.py`: `RobotwinReplayScene` reconstructs the tabletop
  environment from a bridge.
- `replay_task.py`: `RobotwinReplayTask` performs open-loop kinematic replay
  frame by frame.
- `recorder.py`: `EpisodeRecorder` exports RoboTwin-format episodes
  (HDF5 + MP4 + Zarr for DP/DP3) with per-camera intrinsics/extrinsics and
  per-env episode splitting for batched `n_envs` collection (migration doc
  sections 7/10).
- `embodiment_config.py`: `RobotwinEmbodimentConfig` parses embodiment
  `config.yml` (PD gains, ee links, gripper joints) and merges the mimic
  mapping from `conversion_report.json`; `MimicJointMapper` replicates
  master joint targets onto expanded mimic joints in the control layer
  (migration doc sections 2/4.1).
- `curobo_planner.py`: `HierarchicalCuRoboPlanner` adapts the local
  `hierarchical_cuRobo_planner` project (workspace A* -> cuRobo IK ->
  trajopt) as the dual-arm / constrained planning backend, **replacing the
  raw cuRobo bridge** from migration doc section 5.2. `plan_with_fallback`
  routes to it when available and degrades to Genesis OMPL `plan_path`
  otherwise. Install with `pip install -e <path>/hierarchical_cuRobo_planner`
  (requires CUDA/MUSA PyTorch + cuRobo for end-to-end planning).
- `render_config.py`: three-tier render configs (`configs/render/*.yaml`,
  doc section 6.1): `rasterizer` (seed search/RL), `raytracer` (LuisaRender,
  SAPIEN RT 32spp+OIDN alignment for training data), `batch` (Madrona for
  parallel eval). `compare_render_pair` provides the simulator-independent
  PSNR/MAE reference metrics for the V4 alignment gate.
- `seed_search.py`: `batched_seed_search` implements parallel seed search
  (doc section 5.1) on scenes built with `build(n_envs=N)`: one random
  initialization per env, per-env success mask collection, early stop.

## Migration skeleton

The RoboTwin->Genesis migration P0 skeleton lives in:

- `src/cloud_robotics_sim/backend/base.py` + `backends/genesis_backend.py`:
  extended backend interface (IK, multilink IK, OMPL `plan_path`, PD gains,
  friction/mass/COM domain randomization, camera params).
- `tools/convert_assets.py`: asset conversion toolchain (mimic-joint
  expansion, `package://` mesh URI rewrite, fixed-base / inertial / mesh
  checklist, Genesis load smoke test, `conversion_report.json`; GLB object
  libraries via `--mesh-smoke` with batched scene builds).
- `examples/genesis_aloha_demo.py`: minimal end-to-end demo
  (URDF -> IK -> plan_path -> record HDF5).

```bash
# Assets: auto-downloaded on first use; or prefetch manually (HF mirror for China)
python -m cloud_robotics_sim.robotwin.assets
# Embodiments (URDF): copy tree, rewrite URDFs, Genesis load smoke test
uv run python tools/convert_assets.py assets/robotwin/embodiments/embodiments assets_genesis/embodiments --smoke-test
# RoboTwin-OD objects (GLB, no URDF): mesh load smoke test, no tree copy
uv run python tools/convert_assets.py assets/robotwin/objects/objects assets_genesis/objects --mesh-smoke --visual-sample 20
uv run python examples/genesis_aloha_demo.py --out outputs/genesis_aloha_demo
```

## Example

See `examples/robotwin_replay.py` and the full guide in
`docs/guides/robotwin_replay.md`.

```bash
uv run python examples/robotwin_replay.py --bridge data/trajectories/episode_0.bridge --headless
```

## Tests

```bash
uv run python -m pytest tests/robotwin/ -v
```
