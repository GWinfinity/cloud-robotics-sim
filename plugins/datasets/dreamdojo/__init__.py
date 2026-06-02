# DreamDojo Genesis Dataset Plugin for genesis-cloud-sim
# 
# This plugin provides Genesis physics simulator integration for DreamDojo's
# world model training and inference. It supports:
# - Online simulation data generation
# - Pre-generated dataset loading
# - RL policy-based data collection
# - Multiple robot types (humanoid, franka, ur5, g1, gr1)
#
# Based on DreamDojo's genesis_dreams module, adapted for genesis-cloud-sim.

__version__ = "0.1.0"
__plugin_name__ = "dreamdojo"
__plugin_type__ = "dataset"

from .core import (
    GenesisSimulator,
    GenesisSimulatorConfig,
    GenesisRobotType,
    GenesisDataset,
    GenesisRLDataset,
    GenesisDatasetWrapper,
    create_genesis_simulator,
    create_genesis_dataset,
)

__all__ = [
    "GenesisSimulator",
    "GenesisSimulatorConfig",
    "GenesisRobotType",
    "GenesisDataset",
    "GenesisRLDataset",
    "GenesisDatasetWrapper",
    "create_genesis_simulator",
    "create_genesis_dataset",
]
