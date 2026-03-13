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
