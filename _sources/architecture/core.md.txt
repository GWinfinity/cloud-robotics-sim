# Core Components

The Core module provides the foundational building blocks for simulation environments.

## EnvironmentComposer

The `EnvironmentComposer` is the main entry point for creating simulation environments. It combines Scene, Robot, and Task components into a runnable `ComposedEnvironment`.

### Usage

```python
from cloud_robotics_sim import EnvironmentComposer, ComposerConfig

composer = EnvironmentComposer(ComposerConfig(
    headless=False,
    resolution=(1280, 720),
))

env = composer.compose(scene, robot, task)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 0.01 | Simulation timestep |
| `substeps` | int | 10 | Physics substeps |
| `headless` | bool | False | Run without GUI |
| `resolution` | tuple | (640, 480) | Camera resolution |

## Scene System

### Base Scene Class

```python
from cloud_robotics_sim import Scene, SceneConfig

class MyScene(Scene):
    def _build_custom(self):
        # Add custom objects
        self.add_object(ObjectLibrary.cube(position=(1, 0, 0)))
```

### Predefined Scenes

- **EmptyRoom**: Minimal environment for testing
- **LivingRoom**: Furnished living space
- **Kitchen**: Kitchen with appliances
- **Office**: Office environment

## Robot Embodiments

### Available Robots

| Robot | DOF | Type |
|-------|-----|------|
| FrankaPanda | 7+1 | Collaborative arm |
| UniversalRobotUR5 | 6 | Industrial arm |
| MobileManipulator | 10+ | Mobile base + arm |

### Creating Custom Robots

```python
from cloud_robotics_sim import RobotEmbodiment, EmbodimentConfig

class MyRobot(RobotEmbodiment):
    def spawn(self, scene, position=None):
        # Implementation
        pass
    
    def apply_action(self, action):
        # Implementation
        pass
```

## Task System

### Built-in Tasks

- **PickPlaceTask**: Grasp and place objects
- **NavigationTask**: Navigate to target position
- **ReachTask**: Move end-effector to target pose

### Custom Tasks

```python
from cloud_robotics_sim import Task, TaskConfig

class MyTask(Task):
    def step(self, scene, robot, action):
        # Compute reward and termination
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self.step_count >= self.config.max_episode_steps
        
        return reward, terminated, truncated, {}
```

## Registry System

The Registry pattern enables dynamic component instantiation:

```python
from cloud_robotics_sim import register_scene, default_registry

@register_scene("my_scene")
def create_my_scene(**kwargs):
    return MyScene(**kwargs)

# Later
scene = default_registry().create_scene("my_scene")
```
