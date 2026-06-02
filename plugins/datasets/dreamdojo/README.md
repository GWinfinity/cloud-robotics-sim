# DreamDojo Genesis Dataset Plugin

This plugin provides Genesis physics simulator integration for DreamDojo's world model training and inference within the genesis-cloud-sim framework.

## Overview

The DreamDojo plugin enables:
- **Online Simulation**: Generate synthetic training data in real-time using Genesis physics
- **Pre-generated Datasets**: Load and process pre-generated simulation data
- **RL Policy Integration**: Use trained policies to generate realistic trajectories
- **Multi-Robot Support**: Compatible with humanoid, Franka, UR5, G1, and GR1 robots

## Installation

### Prerequisites

```bash
pip install genesis-world>=0.4.0
```

### Plugin Installation

The plugin is automatically available when genesis-cloud-sim is installed. To verify:

```python
from genesis-cloud-sim import PluginManager

manager = PluginManager()
manager.load_plugin("datasets/dreamdojo")
```

## Quick Start

### Basic Usage

```python
from genesis_cloud_sim.plugins.datasets.dreamdojo import (
    GenesisSimulator,
    GenesisSimulatorConfig,
    GenesisRobotType,
    GenesisDataset,
)

# Create simulator
config = GenesisSimulatorConfig(
    robot_type=GenesisRobotType.HUMANOID,
    headless=True,
)
simulator = GenesisSimulator(config)
simulator.initialize()

# Create dataset
dataset = GenesisDataset(
    num_frames=81,
    robot_type="humanoid",
    use_online_sim=True,
)

# Get a sample
sample = dataset[0]
print(sample["video"].shape)  # [T, C, H, W]
print(sample["action"].shape)  # [T, action_dim]
```

### Generating Synthetic Data

```python
from genesis_cloud_sim.plugins.datasets.dreamdojo import GenesisDatasetWrapper

# Create wrapper
wrapper = GenesisDatasetWrapper(
    simulator=simulator,
    num_episodes=1000,
    episode_length=100,
)

# Generate and save dataset
wrapper.generate_dataset("datasets/genesis_synthetic/data.hdf5")
```

### Using with EnvironmentComposer

```python
from cloud_robotics_sim import EnvironmentComposer, ComposerConfig

composer = EnvironmentComposer(ComposerConfig())

# Add DreamDojo dataset as a data source
composer.add_dataset(
    plugin="dreamdojo",
    config={
        "robot_type": "franka",
        "use_online_sim": False,
        "pre_generated_path": "datasets/genesis_synthetic/data.hdf5",
    }
)
```

## Configuration

See `configs/genesis.yaml` for default configurations. Key settings:

```yaml
# Scene configuration
scene:
  sim_options:
    dt: 0.01
    substeps: 10
  viewer_options:
    res: [1280, 720]

# Robot configuration
robots:
  franka:
    action_dim: 7
    urdf_path: "urdf/robots/franka_emika_panda/panda.urdf"
```

## Supported Robots

| Robot | Type | Action Dim | Status |
|-------|------|------------|--------|
| Humanoid | MJCF | 21 | ✅ Ready |
| Franka Emika Panda | URDF | 7 | ✅ Ready |
| Universal Robots UR5 | URDF | 6 | ✅ Ready |
| Unitree G1 | URDF | 29 | ✅ Ready |
| Fourier GR1 | URDF | 32 | ✅ Ready |
| Custom | URDF | Configurable | ✅ Ready |

## Dataset Format

Pre-generated datasets are stored in HDF5 format:

```
data.hdf5
├── episode_0
│   ├── observations: [T, H, W, 3] - RGB frames
│   └── actions: [T, action_dim] - Joint actions
├── episode_1
│   ├── observations
│   └── actions
└── ...
```

## API Reference

### GenesisSimulator

Main simulator interface wrapping Genesis physics engine.

```python
class GenesisSimulator:
    def __init__(self, config: GenesisSimulatorConfig)
    def initialize() -> None
    def reset(seed: Optional[int] = None) -> None
    def step(action: np.ndarray) -> Tuple[np.ndarray, dict]
    def get_state() -> Dict[str, np.ndarray]
    def set_state(state: Dict[str, np.ndarray]) -> None
    def render(mode: str = "rgb_array") -> np.ndarray
    def close() -> None
```

### GenesisDataset

PyTorch Dataset for Genesis-generated data.

```python
class GenesisDataset(Dataset):
    def __init__(
        self,
        num_frames: int = 81,
        episode_length: int = 100,
        num_episodes: int = 1000,
        robot_type: str = "humanoid",
        simulator_config: Optional[Dict] = None,
        pre_generated_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        seed: int = 0,
        use_online_sim: bool = False,
        device: str = "cuda",
    )
```

### GenesisDatasetWrapper

High-level wrapper for dataset generation.

```python
class GenesisDatasetWrapper:
    def __init__(
        self,
        simulator: GenesisSimulator,
        num_episodes: int = 1000,
        episode_length: int = 100,
        action_dim: Optional[int] = None,
        seed: int = 0,
    )
    def generate_episode(policy: Optional[Callable] = None) -> Dict
    def generate_dataset(save_path: str, policy: Optional[Callable] = None)
```

## Integration with ManiSkill

The DreamDojo plugin is compatible with ManiSkill's data converters:

```python
from genesis_cloud_sim.plugins.envs.maniskill.core.genesis_maniskill.datasets import (
    ManiSkillConverter,
)

# Convert DreamDojo format to ManiSkill format
converter = ManiSkillConverter()
converter.convert(
    input_path="datasets/genesis_synthetic/data.hdf5",
    output_path="datasets/maniskill_format/",
)
```

## Troubleshooting

### Genesis Not Found

```python
# Check if Genesis is installed
import genesis as gs
print(gs.__version__)
```

### GPU Memory Issues

```python
# Use CPU backend
config = GenesisSimulatorConfig(
    device="cpu",
    headless=True,
)
```

### Robot URDF Not Found

Place custom URDFs in one of these locations:
- `genesis-cloud-sim/plugins/envs/sky/core/genesis/assets/urdf/`
- `~/.genesis/assets/urdf/`
- Or provide full path in `robot_urdf_path`

## Migration from DreamDojo

If you're migrating from the standalone DreamDojo project:

```python
# Old (DreamDojo)
from genesis_dreams.genesis_simulator import GenesisSimulator
from genesis_dreams.data.dataset_genesis import GenesisDataset

# New (genesis-cloud-sim)
from genesis_cloud_sim.plugins.datasets.dreamdojo import (
    GenesisSimulator,
    GenesisDataset,
)
```

APIs are largely compatible with minor adjustments for the plugin architecture.

## License

Apache License 2.0 - See LICENSE file for details.

## Acknowledgments

- Original DreamDojo genesis_dreams module
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- Cloud Robotics Sim framework
