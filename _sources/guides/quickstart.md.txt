# Quick Start Guide

Get started with Cloud Robotics Simulation Platform in 5 minutes.

## Hello Simulation

Create your first simulation environment:

```python
# hello_sim.py
from cloud_robotics_sim import (
    ComposerConfig,
    EnvironmentComposer,
    FrankaPanda,
    ObjectLibrary,
    PickPlaceTask,
    Scene,
    SceneConfig,
    TaskConfig,
)
import numpy as np


class SimpleScene(Scene):
    """A simple scene with a table and cube."""
    
    def _build_custom(self):
        # Add table
        self.add_object(ObjectLibrary.coffee_table(position=(0.5, 0, 0)))
        # Add graspable cube
        self.add_object(ObjectLibrary.graspable_cube(
            name="cube",
            position=(0.5, 0, 0.6),
        ))


# Create components
composer = EnvironmentComposer(ComposerConfig(headless=False))
scene = SimpleScene(SceneConfig(name="simple_room"))
robot = FrankaPanda()
task = PickPlaceTask(
    TaskConfig(max_episode_steps=100),
    object_name="cube",
    target_position=(0.5, 0.5, 0.05),
)

# Compose and run
env = composer.compose(scene, robot, task)

# Run random policy
obs, info = env.reset(seed=42)
for _ in range(100):
    action = np.random.uniform(-1, 1, size=8)
    obs, reward, done, trunc, info = env.step(action)
    
    if done or trunc:
        print(f"Episode finished! Success: {info.get('success')}")
        obs, info = env.reset()
```

Run it:
```bash
python hello_sim.py
```

## Training a Policy

### 1. Create Configuration

```yaml
# config.yaml
experiment:
  name: franka_pick
  seed: 42

environment:
  scene: empty_room
  robot: franka_panda
  task: pick_place

training:
  algorithm: PPO
  total_timesteps: 100000
```

### 2. Start Training

```bash
cloud-robotics-sim train --config config.yaml
```

### 3. Evaluate

```bash
cloud-robotics-sim eval --checkpoint outputs/franka_pick/latest.pt
```

## Using Predefined Scenes

```python
from cloud_robotics_sim.core.scenes import LivingRoom, Kitchen

# Use predefined scenes
living_room = LivingRoom()
kitchen = Kitchen()

env = composer.compose(living_room, robot, task)
```

## Vectorized Training

Train on multiple environments in parallel:

```python
from cloud_robotics_sim import VecEnvConfig, GenesisVectorizedEnv

vec_env = GenesisVectorizedEnv(VecEnvConfig(num_envs=128))
obs, info = vec_env.reset()

# Batched actions
actions = np.random.randn(128, 8)
obs, rewards, done, trunc, info = vec_env.step(actions)
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md)
- Explore [Examples](../../examples/)
- Check [API Reference](../api/core.md)
