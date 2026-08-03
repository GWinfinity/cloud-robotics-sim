"""Shared types and data classes for the backend abstraction layer.

This module defines backend-agnostic types used by both the core simulation
layer and the concrete backend implementations (Genesis, MT Lambda, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class BackendName(str, Enum):
    """Supported backend identifiers."""

    GENESIS = "genesis"
    MUJOCO = "mujoco"
    MT_LAMBDA = "mt_lambda"


class ShapeType(Enum):
    """Supported primitive shape types."""

    BOX = auto()
    SPHERE = auto()
    CYLINDER = auto()
    MESH = auto()
    CAPSULE = auto()


class DeformableMaterialType(str, Enum):
    """Supported deformable material models.

    These map to Genesis material families (FEM, PBD, SPH, MPM). Only a subset
    is exposed initially; backends may raise NotImplementedError for unsupported
    variants.
    """

    FEM_ELASTIC = "fem_elastic"
    FEM_CLOTH = "fem_cloth"
    PBD_ELASTIC = "pbd_elastic"
    PBD_CLOTH = "pbd_cloth"
    PBD_LIQUID = "pbd_liquid"
    SPH_LIQUID = "sph_liquid"


class LightType(Enum):
    """Supported light types."""

    AMBIENT = auto()
    DIRECTIONAL = auto()
    POINT = auto()
    SPOT = auto()


@dataclass
class Pose:
    """A 3D pose represented as position and quaternion."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        """Normalize pose arrays to float64 ndarrays."""
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.quat = np.asarray(self.quat, dtype=np.float64)


@dataclass
class ViewerOptions:
    """Backend-agnostic viewer configuration."""

    camera_pos: tuple[float, float, float] = (5.0, 5.0, 5.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution: tuple[int, int] = (640, 480)
    max_fps: int = 60
    title: str = "cloud-robotics-sim"


@dataclass
class PhysicsState:
    """Snapshot of physics state for scene save/restore."""

    time: float = 0.0
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    ctrl: np.ndarray | None = None
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticulationState:
    """Backend-agnostic articulation state container."""

    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None


@dataclass
class RenderOutput:
    """Container for renderer output."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None


@dataclass
class LightDescription:
    """Backend-agnostic light description."""

    light_type: LightType
    pos: tuple[float, float, float] | None = None
    direction: tuple[float, float, float] | None = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    cast_shadow: bool = False


@dataclass
class DeformableConfig:
    """Configuration for a deformable entity (soft body / fluid / cloth).

    Attributes:
        material: Deformable material model (e.g. FEM_ELASTIC).
        youngs_modulus: Young's modulus E (Pa). Used by elastic materials.
        poisson_ratio: Poisson's ratio nu. Used by elastic materials.
        density: Mass density (kg/m^3).
        resolution_level: Discretization level (1=coarse, 2=medium, 3=fine).
            Backends translate this into particle size, tetrahedral maxvolume,
            or solver iterations.
        solver_iterations: Optional override for solver iterations.
        region_of_interest: Optional (center, radius) describing the local
            high-detail zone. Used by multiscale error indicators to allocate
            resolution non-uniformly.
        fixed: If True, the entity is pinned in place (e.g. a fixed cloth).
    """

    material: DeformableMaterialType = DeformableMaterialType.FEM_ELASTIC
    youngs_modulus: float = 1.0e4
    poisson_ratio: float = 0.45
    density: float = 1000.0
    resolution_level: int = 2
    solver_iterations: int | None = None
    region_of_interest: tuple[tuple[float, float, float], float] | None = None
    fixed: bool = False


@dataclass
class DeformableState:
    """Backend-agnostic snapshot of a deformable entity.

    Attributes:
        positions: Vertex/particle positions as (N, 3) array.
        velocities: Vertex/particle velocities as (N, 3) array, if available.
        constrained_indices: Indices of constrained vertices, if any.
        constrained_positions: Target positions for constrained vertices.
    """

    positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    velocities: np.ndarray | None = None
    constrained_indices: np.ndarray | None = None
    constrained_positions: np.ndarray | None = None
