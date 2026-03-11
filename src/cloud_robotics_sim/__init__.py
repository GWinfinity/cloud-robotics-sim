"""Cloud Robotics Simulation Platform.

A cloud-native robotics simulation platform built on Genesis physics engine,
designed for scalable reinforcement learning and imitation learning research.

Example:
    >>> from cloud_robotics_sim import EnvironmentComposer, Scene, RobotEmbodiment
    >>> composer = EnvironmentComposer()
    >>> env = composer.compose(scene, robot, task)
    >>> obs, info = env.reset()
"""

__version__ = "2.0.0"
__author__ = "Cloud Robotics Team"
__license__ = "Apache-2.0"

# Core components
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

# Vectorized environments
from cloud_robotics_sim.core.vectorized import (
    VectorizedEnvironment,
    GenesisVectorizedEnv,
    VecEnvConfig,
)

__all__ = [
    # Version
    "__version__",
    # Composer
    "EnvironmentComposer",
    "ComposedEnvironment",
    "ComposerConfig",
    "EnvironmentVariantGenerator",
    # Scene
    "Scene",
    "SceneConfig",
    "ObjectSpawn",
    "ObjectLibrary",
    # Robot
    "RobotEmbodiment",
    "EmbodimentConfig",
    "SensorConfig",
    "FrankaPanda",
    "UniversalRobotUR5",
    # Task
    "Task",
    "TaskConfig",
    "PickPlaceTask",
    "NavigationTask",
    # Registry
    "AssetRegistry",
    "SceneRegistry",
    "RobotRegistry",
    "TaskRegistry",
    "register_scene",
    "register_robot",
    "register_task",
    "default_registry",
    # Vectorized
    "VectorizedEnvironment",
    "GenesisVectorizedEnv",
    "VecEnvConfig",
]
