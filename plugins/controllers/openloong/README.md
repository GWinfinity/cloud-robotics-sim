# OpenLoong Walking Controller Plugin

This plugin provides walking parameter optimization for the OpenLoong humanoid robot, adapted from OpenEvolve's openloong_walking example.

## Overview

The OpenLoong Walking Controller enables:
- **Walking Parameter Optimization**: Evolve PD gains, MPC weights, and gait parameters
- **Multiple Evaluation Modes**: Simple heuristic or Genesis physics simulation
- **Gymnasium Compatibility**: Standard RL environment interface
- **Preset Gaits**: Stable, fast, and cautious walking presets

## Installation

### Prerequisites

```bash
pip install numpy>=1.20.0

# Optional but recommended
pip install genesis-world>=0.4.0
pip install gymnasium>=0.28.0
```

## Quick Start

### Basic Usage

```python
from genesis_cloud_sim.plugins.controllers.openloong import (
    WalkingParameters,
    evaluate_walking,
)

# Create default parameters
params = WalkingParameters()

# Evaluate (simple mode)
result = evaluate_walking(params, use_genesis=False)
print(f"Stability: {result.stability_score:.3f}")

# Evaluate with Genesis (if available)
result = evaluate_walking(params, use_genesis=True, device="cuda")
print(f"Stability: {result.stability_score:.3f}")
```

### Using Presets

```python
from genesis_cloud_sim.plugins.controllers.openloong import get_preset

# Available presets: 'default', 'stable_walk', 'fast_walk', 'cautious_walk'
params = get_preset('stable_walk')
result = evaluate_walking(params)
```

### Walking Parameters

The `WalkingParameters` class contains:

#### PD Controller Gains
```python
params.kp_base = 40.0  # Base proportional gain
params.kd_base = 4.0   # Base derivative gain
params.kp_leg = 25.0   # Leg proportional gain
params.kd_leg = 2.5    # Leg derivative gain
params.kp_knee = 30.0  # Knee proportional gain
params.kd_knee = 3.0   # Knee derivative gain
params.kp_ankle = 35.0 # Ankle proportional gain
params.kd_ankle = 3.5  # Ankle derivative gain
```

#### MPC Weights
```python
params.mpc_weight_roll = 10.0   # Roll tracking
params.mpc_weight_pitch = 10.0  # Pitch tracking
params.mpc_weight_pz = 50.0     # Height control
params.mpc_weight_vx = 10.0     # Forward velocity
```

#### Gait Parameters
```python
params.gait_period = 0.8    # Walking cycle duration (s)
params.swing_height = 0.08  # Foot lift height (m)
params.stance_ratio = 0.5   # Ground contact ratio
params.desired_velocity = 0.2  # Target speed (m/s)
```

## Gymnasium Environment

```python
from genesis_cloud_sim.plugins.controllers.openloong import make_env

# Create environment
env = make_env(use_genesis=False)

# Standard Gymnasium API
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### With Genesis Simulation

```python
env = make_env(use_genesis=True, device="cuda", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
```

## Evaluation Modes

### Simple Heuristic Evaluator

Fast evaluation without physics simulation:

```python
from genesis_cloud_sim.plugins.controllers.openloong import WalkingEvaluator

evaluator = WalkingEvaluator(use_genesis=False)
result = evaluator.evaluate(params)

print(f"Stability: {result.stability_score}")
print(f"Height: {result.final_height}")
print(f"Fall time: {result.fall_time}")
```

### Genesis Simulation Evaluator

Accurate evaluation with physics simulation:

```python
evaluator = WalkingEvaluator(use_genesis=True, device="cuda")
result = evaluator.evaluate(params)

# Full metrics
print(f"Stability: {result.stability_score}")
print(f"Height: {result.final_height}")
print(f"Fall time: {result.fall_time}")
print(f"Max roll: {result.max_roll}")
print(f"Max pitch: {result.max_pitch}")
```

## Parameter Optimization

### Mutate Parameters

```python
# Create mutated copy
mutated_params = params.mutate(
    mutation_rate=0.1,    # 10% chance per parameter
    mutation_scale=0.1,   # ±10% variation
)
```

### Compare Parameters

```python
from genesis_cloud_sim.plugins.controllers.openloong import compare_parameters

comparison = compare_parameters(params1, params2, use_genesis=False)

print(f"Better stability: {comparison['better_stability']}")
print(f"Stability difference: {comparison['stability_diff']:.3f}")
```

## Configuration File

Use YAML configuration for batch setup:

```yaml
# config.yaml
walking:
  kp_base: 45.0
  kd_base: 4.5
  gait_period: 0.75
  desired_velocity: 0.25

evaluation:
  sim_duration: 15.0
  use_genesis: true
  device: "cuda"
```

Load from file:

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

params = WalkingParameters(**config['walking'])
```

## Integration with Evolutionary Algorithms

Compatible with OpenEvolve and other EA frameworks:

```python
# Example: Simple hill climbing
best_params = WalkingParameters()
best_score = evaluate_walking(best_params).stability_score

for generation in range(100):
    # Mutate
    candidate = best_params.mutate(mutation_rate=0.2, mutation_scale=0.1)
    
    # Evaluate
    score = evaluate_walking(candidate).stability_score
    
    # Select
    if score > best_score:
        best_params = candidate
        best_score = score
        print(f"Gen {generation}: New best score {best_score:.3f}")
```

## Presets Reference

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Balanced parameters | General purpose |
| `stable_walk` | High gains, slower | Robust walking |
| `fast_walk` | Lower gains, faster | Quick locomotion |
| `cautious_walk` | Very slow, careful | Precise control |

## Robot Model Setup

### OpenLoong URDF

Place the OpenLoong URDF file in one of these locations:
- `urdf/robots/openloong/AzureLoong.urdf`
- `urdf/openloong/AzureLoong.urdf`
- `assets/openloong/AzureLoong.urdf`

Or specify the path explicitly:

```python
params = WalkingParameters(
    robot_urdf_path="/path/to/AzureLoong.urdf",
    initial_height=1.0,
)
```

### Fallback Models

If OpenLoong URDF is not available, the system falls back to:
- Genesis built-in humanoid model

## API Reference

### WalkingParameters

```python
@dataclass
class WalkingParameters:
    # PD Gains
    kp_base, kd_base: float
    kp_leg, kd_leg: float
    kp_knee, kd_knee: float
    kp_ankle, kd_ankle: float
    
    # MPC Weights
    mpc_weight_roll, mpc_weight_pitch, mpc_weight_yaw: float
    mpc_weight_px, mpc_weight_py, mpc_weight_pz: float
    mpc_weight_wx, mpc_weight_wy, mpc_weight_wz: float
    mpc_weight_vx, mpc_weight_vy, mpc_weight_vz: float
    
    # Gait
    gait_period, swing_height, stance_ratio: float
    
    # Task
    desired_velocity, torque_limit: float
    filter_alpha: float
    
    # Methods
    get_pd_gains() -> Dict[str, Tuple[float, float]]
    get_mpc_L_diag() -> np.ndarray
    get_mpc_K_diag() -> np.ndarray
    to_dict() -> Dict[str, Any]
    from_dict(dict) -> WalkingParameters
    mutate(mutation_rate, mutation_scale) -> WalkingParameters
```

### WalkingEvaluator

```python
class WalkingEvaluator:
    def __init__(self, use_genesis=False, device="cuda")
    def evaluate(params: WalkingParameters) -> EvaluationResult
```

### OpenLoongWalkingEnv

```python
class OpenLoongWalkingEnv:
    def __init__(self, params=None, use_genesis=False, device="cuda")
    def reset(seed=None, options=None) -> Tuple[obs, info]
    def step(action) -> Tuple[obs, reward, terminated, truncated, info]
    def render()
    def close()
```

## Troubleshooting

### "Genesis not available"

Install Genesis for physics simulation:
```bash
pip install genesis-world
```

Or use simple evaluator mode:
```python
evaluator = WalkingEvaluator(use_genesis=False)
```

### "Gymnasium not available"

Install for RL environment support:
```bash
pip install gymnasium
```

### Low stability scores

- Check parameter ranges are reasonable
- Try starting from `stable_walk` preset
- Use Genesis evaluator for accurate results

## References

- Original OpenEvolve: openevolve/examples/openloong_walking/
- OpenLoong Project: https://github.com/loongOpen/OpenLoong-Dyn-Control
- Genesis Simulation: https://genesis-world.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/

## License

Apache License 2.0
