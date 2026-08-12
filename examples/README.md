# Examples Index

Runnable demos and examples, grouped by topic. Status legend:
**current** = maintained and CI-covered · **experimental** = prototype, may
need specific assets/backends · **legacy** = kept for reference, superseded.

## RoboTwin (`robotwin/`)

| File | Status | Description |
|---|---|---|
| `home_demo.py` | current | Narrated 5-room pick-and-place demo videos: FR3 suction arm + RoboTwin-OD furniture, cuRobo/OMPL planning, moving track camera, per-room MP4 + report. `uv run python examples/robotwin/home_demo.py --room all --backend gpu --resolution 1920x1080` |
| `replay.py` | current | Replay a RoboTwin demonstration in Genesis (ALOHA-AgileX dual arm). See `docs/guides/robotwin_replay.md`. |
| `aloha_demo.py` | current | Minimal RoboTwin→Genesis migration skeleton: URDF loading, PD gains, IK + OMPL `plan_path`, HDF5 + MP4 episode export. |

## Grasping (`grasp/`)

| File | Status | Description |
|---|---|---|
| `grasp_all_objects.py` | current | Full-object grasp benchmark over the RoboTwin-OD library (FR3 suction + hierarchical cuRobo/OMPL planner); emits per-class reports and optional episode records. |
| `dengyu_multiscale_soft_grasp.py` | experimental | Multiscale soft-body grasping prototype. |

## Motor simulation (`motor_simulation/`)

Electromagnetic/thermal motor physics demos (`demo_01`–`demo_04`, or
`run_all.py` for the full sweep; PNG outputs written next to the scripts).

## Data & migration

| Path | Status | Description |
|---|---|---|
| `basic_usage.py` | current | Minimal API walkthrough (composer, scenes, tasks). |
| `costream/` | current | Co-streaming insertion demos (see its own README). |
| `hifiumi/` | current | HiFi-UMI-2K dataset loader + replay CLI (see its README/LESSONS). |
| `migration/` | current | DreamDojo / ManiSkill migration examples. |

## Legacy (`legacy/`)

`run_apartment.py` and `run_single_scene.py` — the original apartment video
demos. Superseded by `robotwin/home_demo.py`; kept for reference only.
