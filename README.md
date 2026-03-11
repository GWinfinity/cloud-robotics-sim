# Cloud Robotics Simulation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A cloud-native robotics simulation platform built on [Genesis](https://genesis-world.readthedocs.io/) physics engine, designed for scalable reinforcement learning and imitation learning research.

## Features

- **Cloud-Native Architecture** - Kubernetes-native deployment with auto-scaling
- **Composable Design** - Mix-and-match Scenes, Robots, and Tasks
- **Massively Parallel** - Train on up to 4,096 environments simultaneously
- **Agent-Ready** - Built-in skill registry and task execution engine
- **Gymnasium Compatible** - Seamless integration with RL/IL libraries

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/cloud-robotics-sim.git
cd cloud-robotics-sim

# Install with pip
pip install -e ".[dev]"
```

### Basic Usage

```python
from cloud_robotics_sim import (
    EnvironmentComposer,
    ComposerConfig,
    SceneConfig,
    ObjectLibrary,
)

# Create a simple pick-and-place environment
composer = EnvironmentComposer(ComposerConfig(headless=False))

# Compose and run
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()
```

### Command Line Interface

```bash
# Training
cloud-robotics-sim train --config configs/franka_pickplace.yaml

# Evaluation
cloud-robotics-sim eval --checkpoint checkpoints/latest.pt --num-episodes 100

# Interactive Agent
cloud-robotics-sim agent --goal "pick up the red cube"

# Run Tests
cloud-robotics-sim test
```

## Architecture

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

## Project Structure

```
cloud-robotics-sim/
├── src/
│   └── cloud_robotics_sim/     # Main package
│       ├── core/               # Core components
│       │   ├── composer.py     # Environment composition
│       │   ├── scene.py        # Scene definitions
│       │   ├── embodiment.py   # Robot implementations
│       │   ├── task.py         # Task definitions
│       │   ├── registry.py     # Component registry
│       │   └── vectorized.py   # Parallel environments
│       ├── runtime/            # Agent runtime
│       └── learning/           # RL/IL frameworks
├── configs/                    # Configuration files
├── tests/                      # Test suite
├── examples/                   # Example scripts
└── docs/                       # Documentation
```

## Supported Robots

| Robot | Type | DOF | Status |
|-------|------|-----|--------|
| Franka Emika Panda | Collaborative Arm | 7+1 | ✅ Fully Supported |
| Universal Robots UR5 | Industrial Arm | 6 | ✅ Fully Supported |
| Mobile Manipulator | Mobile Base + Arm | 10+ | 🚧 Experimental |

## Supported Tasks

- **Pick and Place** - Grasp objects and place at target locations
- **Navigation** - Reach target positions while avoiding obstacles
- **Reach** - Move end-effector to target poses

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Getting Started Guide](docs/guides/quickstart.md)
- [API Reference](docs/reference/api.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Genesis World 0.4+
- PyTorch 2.0+

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/
black --check src/

# Run tests
pytest tests/ -v

# Build documentation
cd docs && make html
```

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{cloud_robotics_sim,
  title = {Cloud Robotics Simulation Platform},
  author = {Cloud Robotics Team},
  year = {2025},
  url = {https://github.com/your-org/cloud-robotics-sim}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Genesis](https://genesis-world.readthedocs.io/) - The underlying physics engine
- [LeRobot](https://github.com/huggingface/lerobot) - Inspiration for the learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface standard

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Support

- 📧 Email: support@cloudrobotics.dev
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/cloud-robotics-sim/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/cloud-robotics-sim/issues)
