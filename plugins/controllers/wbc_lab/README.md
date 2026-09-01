# wbc-lab Genesis Plugin

Whole-body motion tracking controller migrated from [wbc-lab](https://github.com/wbc-lab/wbc-lab) (MuJoCo/mjlab) to Genesis physics engine.

## Overview

This plugin implements the shared WBC MDP from wbc-lab on Genesis:

- **Multi-clip motion library** -train on NPZ motion libraries, one policy generalizes across clips
- **Adaptive RSI** -similarity-weighted bin sampling for curriculum-based training
- **Tracking rewards** -joint pos/vel, anchor pos/ori, keybody, velocity tracking
- **Regularization** -action rate, torque limits, self-collision, feet slip
- **Unitree G1** -29-DOF humanoid with actuator-specific PD gains and torque limits
- **Deploy export** -ONNX policy + tracking params YAML

## Architecture

```
wbc_lab/
├── core/
-  ├── envs/
-  -  ├── wbc_env.py         # Main WBC environment (Genesis)
-  -  ├── rewards.py         # Tracking + regularization rewards
-  -  └── terminations.py    # Position/orientation/contact limits
-  ├── motion/
-  -  ├── motion_loader.py   # NPZ motion library loading
-  -  ├── motion_command.py  # Multi-clip playback + reference features
-  -  └── sampling.py        # Adaptive RSI bin sampler
-  ├── robots/
-  -  └── g1_config.py       # Unitree G1 joint/body/PD config
-  ├── export/
-  -  └── tracking_params.py # Deploy params YAML export
-  └── utils/
-      └── motion_mirror.py   # Sagittal-plane mirroring
├── configs/
-  └── default_g1.yaml        # Default G1 WBC config
├── examples/
-  ├── basic_usage.py         # Basic env + motion tracking
-  └── ab_test.py             # A/B comparison with reference
└── tests/
    └── test_wbc_lab.py      # Unit tests (no Genesis required)
```

## Usage

### Basic Environment

```python
from cloud_robotics_sim.plugins.controllers.wbc_lab import WbcGenesisEnv
from cloud_robotics_sim.plugins.controllers.wbc_lab.core.envs.wbc_env import WbcEnvConfig

env = WbcGenesisEnv(WbcEnvConfig(headless=True))
obs, info = env.reset()

for _ in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated[0] or truncated[0]:
        obs, info = env.reset()

env.close()
```

### With Motion Tracking

```python
from cloud_robotics_sim.plugins.controllers.wbc_lab import WbcGenesisEnv
from cloud_robotics_sim.plugins.controllers.wbc_lab.core.envs.wbc_env import WbcEnvConfig
from cloud_robotics_sim.plugins.controllers.wbc_lab.core.motion import MotionCommandCfg

cfg = WbcEnvConfig(
    motion_cfg=MotionCommandCfg(motion_path="data/g1/samples"),
    headless=True,
)
env = WbcGenesisEnv(cfg)
```

### Standalone Reward Computation

```python
from cloud_robotics_sim.plugins.controllers.wbc_lab import RewardComputer

calc = RewardComputer(num_envs=4)
rewards, terms = calc.compute_all(
    joint_pos_error=jp_error,
    joint_vel_error=jv_error,
    anchor_pos_error=anchor_err,
    anchor_ori_error=ori_err,
    anchor_lin_vel_error=lin_vel_err,
    anchor_ang_vel_error=ang_vel_err,
)
```

## Migration Notes

### What changed from wbc-lab

| Component | wbc-lab (MuJoCo) | wbc-genesis (Genesis) |
|-----------|---------------------|----------------------|
| Physics | MuJoCo via mjlab | Genesis 1.3.2 |
| Env API | `ManagerBasedRlEnvCfg` | `WbcGenesisEnv` (standalone) |
| Scene | `SceneCfg` | `gs.Scene` |
| Robot | MJCF + `SceneEntityCfg` | `gs.morphs.MJCF/URDF` |
| Actions | `JointPositionActionCfg` | Direct `control_dofs_position` |
| Observations | `ObservationManager` | Manual computation |
| Rewards | `RewardManager` | `RewardComputer` |
| RSI | `RsiCfg` + mjlab sampling | `AdaptiveRsiSampler` |
| Tasks | `mjlab.tasks` entry points | Plugin system |

### What was preserved

- All reward function logic (tracking + regularization)
- Adaptive RSI bin sampling algorithm
- Motion NPZ format and loading pipeline
- G1 robot configuration (joint names, PD gains, torque limits)
- Left-right symmetry pairs
- Motion mirroring
- Tracking params export format

### Known limitations

- Single-env support only (Genesis multi-env is WIP)
- No terrain generation (flat plane only)
- Simplified actuator model (no torque-speed envelope)
- No assistive wrench curriculum yet
- Deploy ONNX export not yet ported

## Dependencies

- `genesis-world>=1.3.0`
- `numpy>=1.20`
- `torch>=2.0`
- `pyyaml>=6.0`
- `scipy>=1.10`

## Source

- **Original project**: [wbc-lab](https://github.com/wbc-lab/wbc-lab)
- **Paper references**:
  - [ZEST](https://arxiv.org/abs/2602.00401)
  - [BeyondMimic](https://arxiv.org/abs/2508.08241)
  - [SONIC](https://arxiv.org/abs/2511.07820)
