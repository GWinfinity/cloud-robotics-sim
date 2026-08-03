# RoboTwin Trajectory Replay in Genesis

This guide explains how to replay demonstrations collected with the
[RoboTwin](https://github.com/TianxingChen/RoboTwin) benchmark (built on
SAPIEN 3.0.0b1) inside the Genesis-backed simulation environment.

## Overview

The migration uses a **Bridge pattern**:

1. **Collect** demonstrations in the original RoboTwin/SAPIEN environment.
2. **Export** each episode to a simulator-agnostic `RobotwinBridge` pickle file.
3. **Replay** the episode in this project using `RobotwinReplayScene`,
   `AlohaAgileX`, and `RobotwinReplayTask`.

This avoids rewriting the 50+ RoboTwin tasks in Genesis all at once, while
allowing policies to be trained and evaluated in Genesis.

## The Bridge Format

A `RobotwinBridge` contains:

| Field | Description |
|-------|-------------|
| `task_name` | RoboTwin task identifier, e.g. `"move_can_pot"`. |
| `seed` | Random seed used during collection. |
| `fps` | Playback frequency. |
| `robot_urdf` | Path to the robot URDF used in the episode. |
| `table_height` | Table surface z-coordinate. |
| `object_assets` | Map from object name to `ObjectAsset` (`mesh` or `urdf`). |
| `cameras` | List of `CameraConfig` recordings. |
| `frames` | List of `RobotwinFrame` states. |

Each `RobotwinFrame` stores:

- `robot_command`: 14-D target joint position
  `[L_arm(6), L_grip, R_arm(6), R_grip]`.
- `robot_achieved_qpos`: Actual generalized positions from SAPIEN.
- `robot_base_pos` / `robot_base_quat`: Robot base pose.
- `object_states`: Per-object pose and optional joint state.
- `camera_images`: Optional recorded camera images.

Load a bridge in Python:

```python
from cloud_robotics_sim.robotwin import RobotwinBridge

bridge = RobotwinBridge.load("data/trajectories/episode_0.bridge")
print(bridge.summary())
```

## Running the Replay Example

The fastest way to see the pipeline is:

```bash
uv run python examples/robotwin_replay.py \
    --bridge data/trajectories/episode_0.bridge \
    --headless \
    --output-video outputs/robotwin_replay.mp4
```

Command-line options:

| Flag | Description |
|------|-------------|
| `--bridge PATH` | Path to a `.bridge` pickle file. If omitted, a synthetic bridge is used. |
| `--headless` | Run without the interactive viewer. |
| `--output-video PATH` | Render frames to a video file. |
| `--camera-pos x y z` | Camera position for rendering. |
| `--camera-lookat x y z` | Camera look-at point for rendering. |
| `--dt` | Simulation timestep (default `0.01`). |
| `--substeps` | Physics substeps per step (default `10`). |
| `--device cuda\|cpu` | Taichi device. |

## Composing a Replay Environment Programmatically

```python
from cloud_robotics_sim import ComposerConfig, EnvironmentComposer
from cloud_robotics_sim.robotwin import RobotwinBridge
from cloud_robotics_sim.robotwin.dual_arm_embodiment import AlohaAgileX, AlohaAgileXConfig
from cloud_robotics_sim.robotwin.replay_scene import RobotwinReplayScene
from cloud_robotics_sim.robotwin.replay_task import RobotwinReplayTask

bridge = RobotwinBridge.load("data/trajectories/episode_0.bridge")

composer = EnvironmentComposer(ComposerConfig(headless=True))
scene = RobotwinReplayScene(bridge)
robot = AlohaAgileX(AlohaAgileXConfig(urdf_path=bridge.robot_urdf))
task = RobotwinReplayTask(bridge)

env = composer.compose(scene, robot, task)
obs, info = env.reset(seed=bridge.seed)

while True:
    obs, reward, terminated, truncated, info = env.step(
        action=bridge.frames[info["frame_index"]].robot_command
    )
    if terminated or truncated:
        break

env.close()
```

## Robot: ALOHA-AgileX

The ALOHA-AgileX embodiment maps the RoboTwin 14-D action space to the
16 driven DoFs in the URDF (left/right arm 6 + 2 gripper fingers each):

```python
from cloud_robotics_sim.robotwin.dual_arm_embodiment import AlohaAgileX, AlohaAgileXConfig

robot = AlohaAgileX(
    AlohaAgileXConfig(
        urdf_path="assets/embodiments/aloha-agilex/urdf/robot.urdf",
        left_arm_joints=[f"fl_joint{i}" for i in range(1, 7)],
        right_arm_joints=[f"fr_joint{i}" for i in range(1, 7)],
    )
)
```

The full URDF exposes 38 DoFs (arms, grippers, passive wheels/suspension/mast).
Only the 16 driven joints are written by `apply_action` via
`control_dofs_position(..., dofs_idx_local=...)`.

## Scene: RobotwinReplayScene

`RobotwinReplayScene` reconstructs the tabletop environment from the bridge:

- Adds a table surface at `table_height`.
- Spawns `mesh` objects through the backend.
- Loads `urdf` objects as articulated objects.

Object poses are overwritten on every frame by `RobotwinReplayTask`, so the
initial spawn pose is only a placeholder.

## Task: RobotwinReplayTask

`RobotwinReplayTask` performs open-loop kinematic replay:

- `reset()` applies the first bridge frame.
- `step()` advances to the next frame and applies recorded states.
- The `action` argument is ignored; states come directly from the bridge.

This task is intentionally **not** a learning task. It is used to verify that
Genesis can reproduce the recorded kinematics before training policies on top
of it.

## Limitations

- **Genesis control**: Genesis currently supports `position` and `force`
  control, but not independent velocity targets. Replay uses position targets.
- **Complex URDFs**: ALOHA-AgileX and similar complex URDFs may need empty
  `<collision>` nodes cleaned before Genesis can load them.
- **Rendering gap**: Genesis and SAPIEN use different renderers. Visual policies
  trained on SAPIEN images may need domain randomization or adaptation when
  transferred to Genesis renders.
- **Assets**: The example requires the real ALOHA-AgileX URDF and object meshes.
  The repository does not include these assets; they must be obtained from the
  RoboTwin dataset or the robot description package.

## Exporting Your Own Bridge

To export a bridge from RoboTwin, instrument the SAPIEN environment to build a
`RobotwinBridge` object and call `bridge.save(path)` after each episode. A
helper module can be added to the RoboTwin side (in its own SAPIEN environment)
that converts native SAPIEN state dictionaries into `RobotwinFrame` instances.
