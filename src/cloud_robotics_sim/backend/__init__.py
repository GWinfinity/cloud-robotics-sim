"""Backend abstraction layer for cloud_robotics_sim.

This package decouples the core simulation logic from concrete physics
engines (Genesis, MuJoCo-Warp-MUSA, etc.) so that the same Scene,
RobotEmbodiment, and Task code can run on multiple backends.
"""

from cloud_robotics_sim.backend.base import (
    ArticulationBackend,
    CameraBackend,
    DeformableEntityBackend,
    EntityBackend,
    RendererBackend,
    SceneBackend,
    SimulatorBackend,
)
from cloud_robotics_sim.backend.factory import (
    available_backends,
    get_backend,
    register_backend,
)
from cloud_robotics_sim.backend.types import (
    ArticulationState,
    BackendName,
    DeformableConfig,
    DeformableMaterialType,
    DeformableState,
    LightDescription,
    LightType,
    PhysicsState,
    Pose,
    RenderOutput,
    ShapeType,
    ViewerOptions,
)

__all__ = [
    # Factories
    "get_backend",
    "register_backend",
    "available_backends",
    # ABCs
    "SimulatorBackend",
    "SceneBackend",
    "EntityBackend",
    "DeformableEntityBackend",
    "ArticulationBackend",
    "RendererBackend",
    "CameraBackend",
    # Types
    "BackendName",
    "ShapeType",
    "DeformableMaterialType",
    "DeformableConfig",
    "DeformableState",
    "LightType",
    "Pose",
    "ViewerOptions",
    "PhysicsState",
    "ArticulationState",
    "RenderOutput",
    "LightDescription",
]
