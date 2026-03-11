"""Core simulation components.

This module provides the foundational building blocks for constructing
robotic simulation environments with a focus on composability and scalability.
"""

from cloud_robotics_sim.core.composer import (
    EnvironmentComposer,
    ComposedEnvironment,
    ComposerConfig,
    EnvironmentVariantGenerator,
)

from cloud_robotics_sim.core.scene import (
    Scene,
    SceneConfig,
    ObjectSpawn,
    ObjectLibrary,
)

from cloud_robotics_sim.core.embodiment import (
    RobotEmbodiment,
    EmbodimentConfig,
    SensorConfig,
    FrankaPanda,
    UniversalRobotUR5,
)

from cloud_robotics_sim.core.task import (
    Task,
    TaskConfig,
    PickPlaceTask,
    NavigationTask,
)

from cloud_robotics_sim.core.registry import (
    AssetRegistry,
    SceneRegistry,
    RobotRegistry,
    TaskRegistry,
    register_scene,
    register_robot,
    register_task,
    default_registry,
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
