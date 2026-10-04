# Learning Framework

The Learning Layer provides reinforcement learning (RL) and imitation learning (IL) capabilities for training robot policies.

## Overview

```
┌─────────────────────────────────────────────────────┐
│                  Learning Layer                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   RL Algorithms │  │  Imitation Learning     │  │
│  │   ├─ PPO        │  │  ├─ Behavior Cloning   │  │
│  │   ├─ SAC        │  │  ├─ DAgger             │  │
│  │   └─ TD3        │  │  └─ Diffusion Policy   │  │
│  └────────┬────────┘  └────────────┬────────────┘  │
│           │                        │               │
│           └────────┬───────────────┘               │
│                    │                               │
│             ┌──────▼──────┐                        │
│             │  Evaluator  │                        │
│             └─────────────┘                        │
└─────────────────────────────────────────────────────┘
```

## Reinforcement Learning

### PPO (Proximal Policy Optimization)

The default RL algorithm for continuous control tasks.

```python
from cloud_robotics_sim.learning.rl import PPO

ppo = PPO(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

ppo.learn(total_timesteps=1_000_000)
```

### SAC (Soft Actor-Critic)

Better for sample efficiency and continuous action spaces.

```python
from cloud_robotics_sim.learning.rl import SAC

sac = SAC(
    env=env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
)

sac.learn(total_timesteps=1_000_000)
```

## Imitation Learning

### Behavior Cloning

Supervised learning from demonstrations.

```python
from cloud_robotics_sim.learning.il import BehaviorCloning

bc = BehaviorCloning(
    env=env,
    demonstrations="demos/pick_place.pkl",
    batch_size=256,
    epochs=100,
)

bc.train()
```

### DAgger (Dataset Aggregation)

Interactive imitation learning with expert feedback.

```python
from cloud_robotics_sim.learning.il import DAgger

dagger = DAgger(
    env=env,
    expert_policy=expert,
    initial_demos="demos/initial.pkl",
    n_iterations=10,
)

dagger.train()
```

## Evaluation

### Policy Evaluation

```python
from cloud_robotics_sim.learning import Evaluator

evaluator = Evaluator(env, policy)

results = evaluator.evaluate(
    num_episodes=100,
    render=True,
    save_video=True,
)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average reward: {results['mean_reward']:.2f}")
```

### Benchmarking

```python
from cloud_robotics_sim.learning import Benchmark

benchmark = Benchmark([
    ("PPO", ppo_policy),
    ("SAC", sac_policy),
    ("BC", bc_policy),
])

results = benchmark.run(env_suite)
benchmark.report("results.html")
```

## Training with Vectorized Environments

```python
from cloud_robotics_sim import VecEnvConfig, GenesisVectorizedEnv
from cloud_robotics_sim.learning.rl import PPO

# Create vectorized environment
vec_env = GenesisVectorizedEnv(VecEnvConfig(num_envs=128))

# Train with massive parallelization
ppo = PPO(env=vec_env)
ppo.learn(total_timesteps=10_000_000)
```

## Checkpointing

```python
# Save checkpoint
policy.save("checkpoints/policy.pt")

# Load checkpoint
policy.load("checkpoints/policy.pt")
```

## TensorBoard Integration

```python
# Training metrics are automatically logged
tensorboard --logdir ./runs
```
