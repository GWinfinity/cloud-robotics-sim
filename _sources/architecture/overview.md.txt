# Architecture Overview

The Cloud Robotics Simulation Platform follows a three-layer architecture designed for scalability, composability, and cloud-native deployment.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │Skill Registry│  │Task Executor │  │Replay Buffer     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Environment Layer                         │
│                    Genesis Physics                          │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │  Scene   │  │   Robot    │  │         Task           │  │
│  │Component │  │Embodiment  │  │    (Goal + Reward)     │  │
│  └──────────┘  └────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Learning Layer                           │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  RL Algorithms  │  │    Imitation Learning            │ │
│  │  (PPO, SAC)     │  │    (Behavior Cloning, Diffusion) │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Runtime Layer

The Runtime Layer manages agent execution, skill composition, and task orchestration.

**Key Components:**
- **Skill Registry**: Manages reusable robot skills
- **Task Executor**: Handles task lifecycle and state management
- **Replay Engine**: Records and replays demonstrations

### 2. Environment Layer

The Environment Layer provides physics simulation through Genesis, with a composable design pattern.

**Key Components:**
- **Scene**: Defines environment layout and objects
- **RobotEmbodiment**: Encapsulates robot physics and sensors
- **Task**: Specifies goals, rewards, and termination conditions

### 3. Learning Layer

The Learning Layer integrates RL and IL algorithms for policy training.

**Key Components:**
- **RL Module**: PPO, SAC, and other RL algorithms
- **IL Module**: Behavior cloning, diffusion policies
- **Evaluator**: Policy evaluation and benchmarking

## Design Principles

### Composability

Components are designed to be mixed and matched without code changes:

```python
# Compose different scenes, robots, and tasks
env = composer.compose(
    scene=LivingRoom(),
    robot=FrankaPanda(),
    task=PickPlaceTask()
)
```

### Scalability

Vectorized environments enable massive parallelization:

```python
vec_env = GenesisVectorizedEnv(VecEnvConfig(num_envs=4096))
```

### Cloud-Native

Built for Kubernetes deployment with horizontal pod autoscaling.

## Data Flow

1. **Episode Start**: Task resets the scene and robot
2. **Step**: Robot applies actions → Physics simulation → Task computes reward
3. **Episode End**: Task checks termination → Learning algorithm updates policy
