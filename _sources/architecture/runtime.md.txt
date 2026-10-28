# Runtime Architecture

The Runtime Layer provides agent execution capabilities, skill management, and task orchestration.

## Overview

```
┌─────────────────────────────────────────────────────┐
│                    Agent Runtime                     │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │Skill Registry│  │Task Executor│  │Event Bus    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │        │
│         └────────────────┴────────────────┘        │
│                          │                         │
│                   ┌──────▼──────┐                  │
│                   │Replay Engine│                  │
│                   └─────────────┘                  │
└─────────────────────────────────────────────────────┘
```

## Skill Registry

The Skill Registry manages reusable robot skills that can be composed into complex behaviors.

### Skill Definition

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Skill:
    name: str
    description: str
    policy: Callable[[Observation], Action]
    preconditions: list[Callable[[State], bool]]
    postconditions: list[Callable[[State], bool]]
```

### Usage

```python
from cloud_robotics_sim.runtime import SkillRegistry

registry = SkillRegistry()

# Register a skill
@registry.register("pick_object")
def pick_object(observation):
    # Skill implementation
    return action

# Execute skill
action = registry.execute("pick_object", observation)
```

## Task Executor

The Task Executor manages high-level task execution and state machines.

### Task State Machine

```
┌─────────┐    start     ┌─────────┐
│  IDLE   │ ───────────> │ RUNNING │
└─────────┘              └────┬────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐    ┌─────────┐
        │SUCCESS  │     │ FAILURE │    │ TIMEOUT │
        └─────────┘     └─────────┘    └─────────┘
```

### Usage

```python
from cloud_robotics_sim.runtime import TaskExecutor

executor = TaskExecutor()

# Execute task with monitoring
for step in executor.execute(task, max_steps=1000):
    if step.status == TaskStatus.SUCCESS:
        print("Task completed!")
        break
```

## Event Bus

The Event Bus enables decoupled communication between components.

### Publishing Events

```python
from cloud_robotics_sim.runtime import EventBus

event_bus = EventBus()

event_bus.publish("robot.action", {
    "action": action,
    "timestamp": time.time()
})
```

### Subscribing to Events

```python
@event_bus.subscribe("robot.collision")
def on_collision(event):
    logger.warning(f"Collision detected: {event}")
```

## Replay Engine

The Replay Engine records and replays episodes for debugging and imitation learning.

### Recording

```python
from cloud_robotics_sim.runtime import ReplayEngine

engine = ReplayEngine()
engine.start_recording()

# Run episode
for step in episode:
    engine.record_step(observation, action, reward)

engine.save("episode_001.pkl")
```

### Replay

```python
# Load and replay
episode = engine.load("episode_001.pkl")
engine.replay(episode, speed=1.0)
```

## Integration with Environment

```python
from cloud_robotics_sim import EnvironmentComposer
from cloud_robotics_sim.runtime import AgentRuntime

# Create environment
env = composer.compose(scene, robot, task)

# Create runtime with skills
runtime = AgentRuntime(env, skill_registry)

# Execute goal
result = runtime.execute_goal("pick up the red cube and place it on the table")
```
