# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial support for Franka Emika Panda and UR5 robots
- Composable Scene-Robot-Task architecture
- Vectorized environment support for up to 4,096 parallel simulations
- Gymnasium-compatible environment interface
- GitHub Actions CI/CD pipeline

## [2.0.0] - 2025-03-11

### Added
- Complete rewrite with modern Python packaging (pyproject.toml)
- Cloud-native deployment support (Kubernetes, Docker)
- Agent runtime with skill registry and task executor
- Reinforcement Learning (RL) framework integration
- Imitation Learning (IL) support
- Comprehensive test suite with pytest
- Full type hints with mypy compliance
- Code formatting with Black and Ruff

### Changed
- Migrated from `gencs` to `cloud_robotics_sim` package name
- Refactored to European/American open-source standards
- All documentation now in English

### Removed
- Legacy `pytorch-kinematics-ms` and `vggt` dependencies
- Old Chinese-language documentation

[unreleased]: https://github.com/GWinfinity/cloud-robotics-sim/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/GWinfinity/cloud-robotics-sim/releases/tag/v2.0.0
