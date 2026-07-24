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
