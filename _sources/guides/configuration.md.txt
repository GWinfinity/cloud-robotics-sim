# Configuration Guide

Cloud Robotics Simulation Platform uses YAML configuration files for experiment management.

## Configuration Structure

```yaml
experiment:
  name: experiment_name
  seed: 42
  output_dir: ./outputs

environment:
  scene:
    type: scene_name
    # Scene-specific parameters
  
  robot:
    type: robot_name
    # Robot-specific parameters
  
  task:
    type: task_name
    # Task-specific parameters
  
  simulation:
    dt: 0.01
    substeps: 10
    headless: false
    resolution: [640, 480]

training:
  algorithm: PPO
  total_timesteps: 1000000
  # Algorithm-specific parameters

evaluation:
  num_episodes: 100
  render: true
  save_video: true

logging:
  use_wandb: false
  log_interval: 10
  save_interval: 100000
```

## Robot Model Resolution

`environment.robot.urdf_path` 可以留空（`null`），模型路径按以下优先级自动解析
（`core/robot_assets.py`）：

1. 显式路径（若存在）
2. 仓库内 `assets_genesis/embodiments/` 的捆绑 URDF（gitignored，本地可选存在）
3. 自动浅克隆的 [awesome-robot-descriptions](https://github.com/robot-descriptions/awesome-robot-descriptions)
   （默认用 AtomGit 镜像 `https://atomgit.com/gh_mirrors/aw/awesome-robot-descriptions`，
   GitHub 作为兜底；`CRS_ROBOT_DESC_AUTO_DOWNLOAD=0` 可禁用自动克隆，
   `CRS_ROBOT_DESC_DIR` 可改下载位置）
4. Genesis 内置资产查找；全部失败时降级为占位盒子并给出明显警告
   （spawn 后可通过 `robot.asset_source` 查看实际用到的资产）

预取模型：

```bash
python -m cloud_robotics_sim.core.robot_assets
```

配置通过 `cloud_robotics_sim.core.config_loader.load_sim_config()` 加载与校验
（`environment.scene/robot/task/simulation` 为必需结构），改进循环
（`runtime/main.py`）即使用该加载器消费 `configs/franka_pickplace.yaml`。

## Scene Configuration

### Empty Room

```yaml
environment:
  scene:
    type: empty_room
    size: [10.0, 10.0, 3.0]  # width, depth, height
```

### Living Room

```yaml
environment:
  scene:
    type: living_room
    # Pre-furnished, no additional config needed
```

### Custom Scene

```yaml
environment:
  scene:
    type: custom
    name: my_scene
    size: [5.0, 5.0, 3.0]
    objects:
      - type: cube
        position: [1.0, 0.0, 0.5]
        color: [0.9, 0.2, 0.2, 1.0]
```

## Robot Configuration

### Franka Panda

```yaml
environment:
  robot:
    type: franka_panda
    base_position: [0.0, 0.0, 0.0]
    base_orientation: [1.0, 0.0, 0.0, 0.0]
    joint_stiffness: 100.0
    joint_damping: 10.0
```

### UR5

```yaml
environment:
  robot:
    type: ur5
    base_position: [0.5, 0.0, 0.0]
```

## Task Configuration

### Pick and Place

```yaml
environment:
  task:
    type: pick_place
    object_name: target_cube
    target_position: [0.5, 0.0, 0.05]
    success_threshold: 0.05
    max_episode_steps: 500
```

### Navigation

```yaml
environment:
  task:
    type: navigation
    target_position: [3.0, 3.0, 0.0]
    success_threshold: 0.3
```

## Training Configuration

### PPO

```yaml
training:
  algorithm: PPO
  total_timesteps: 1000000
  
  ppo:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
```

### SAC

```yaml
training:
  algorithm: SAC
  total_timesteps: 1000000
  
  sac:
    learning_rate: 3.0e-4
    buffer_size: 1000000
    learning_starts: 10000
    batch_size: 256
    tau: 0.005
    gamma: 0.99
```

## Environment Variables

You can use environment variables in configs:

```yaml
experiment:
  output_dir: ${OUTPUT_DIR:-./outputs}
  
logging:
  use_wandb: ${USE_WANDB:-false}
```

## Loading Configurations

```python
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use with CLI
cloud-robotics-sim train --config config.yaml
```
