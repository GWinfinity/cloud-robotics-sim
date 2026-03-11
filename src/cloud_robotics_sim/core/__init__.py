"""Core simulation components.

This module provides the foundational building blocks for constructing
robotic simulation environments with a focus on composability and scalability.
"""

from cloud_robotics_sim.core.composer import (
    ComposedEnvironment,
    ComposerConfig,
    EnvironmentComposer,
    EnvironmentVariantGenerator,
)
from cloud_robotics_sim.core.embodiment import (
    EmbodimentConfig,
    FrankaPanda,
    RobotEmbodiment,
    SensorConfig,
    UniversalRobotUR5,
)
from cloud_robotics_sim.core.registry import (
    AssetRegistry,
    RobotRegistry,
    SceneRegistry,
    TaskRegistry,
    default_registry,
    register_robot,
    register_scene,
    register_task,
)
from cloud_robotics_sim.core.scene import (
    ObjectLibrary,
    ObjectSpawn,
    Scene,
    SceneConfig,
)
from cloud_robotics_sim.core.task import (
    NavigationTask,
    PickPlaceTask,
    Task,
    TaskConfig,
)

__all__ = [
    "EnvironmentComposer",
    "ComposedEnvironment",
    "ComposerConfig",
    "EnvironmentVariantGenerator",
    "Scene",
    "SceneConfig",
    "ObjectSpawn",
    "ObjectLibrary",
    "RobotEmbodiment",
    "EmbodimentConfig",
    "SensorConfig",
    "FrankaPanda",
    "UniversalRobotUR5",
    "Task",
    "TaskConfig",
    "PickPlaceTask",
    "NavigationTask",
    "AssetRegistry",
    "SceneRegistry",
    "RobotRegistry",
    "TaskRegistry",
    "register_scene",
    "register_robot",
    "register_task",
    "default_registry",
]
