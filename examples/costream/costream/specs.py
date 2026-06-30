"""Specification dataclasses for CoStream stages and composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObjectInfo:
    """Ground-truth or perceived object state."""

    name: str
    pos: np.ndarray
    quat: np.ndarray
    dims: tuple[float, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSummary:
    """Symbolic + geometric scene summary passed to behaviors."""

    objects: dict[str, ObjectInfo] = field(default_factory=dict)
    robot_qpos: np.ndarray | None = None
    instruction: str = ""


@dataclass
class ControllerProfile:
    """Calibrated compliant controller profile for a stage.

    Attributes:
        name: profile identifier, e.g. 'free_space', 'insertion'.
        cartesian_stiffness: 6-DOF stiffness (pos, rot).
        cartesian_damping: 6-DOF damping.
        max_force: force/torque limits (F_xyz, T_xyz).
        admittance_gain: how much contact force deflects the command pose.
        force_axis: boolean mask (3,) for axes under active force control.
    """

    name: str
    cartesian_stiffness: np.ndarray = field(
        default_factory=lambda: np.array([800.0, 800.0, 800.0, 80.0, 80.0, 80.0])
    )
    cartesian_damping: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 40.0, 40.0, 4.0, 4.0, 4.0])
    )
    max_force: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])
    )
    admittance_gain: np.ndarray = field(
        default_factory=lambda: np.array([1e-4, 1e-4, 1e-4])
    )
    force_axis: np.ndarray = field(
        default_factory=lambda: np.array([False, False, False])
    )


@dataclass
class StageSpec:
    """Stage-level task description produced by the policy compiler."""

    name: str
    objective: str
    task_frame: str  # object name used as anchor
    motion: str  # e.g. 'approach', 'insert', 'hold', 'home'
    duration: float  # seconds
    profile: ControllerProfile = field(default_factory=lambda: ControllerProfile(name="default"))
    relative_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    relative_goal: np.ndarray = field(default_factory=lambda: np.eye(4))
    guard_success: dict[str, Any] = field(default_factory=dict)
    guard_recovery: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposeSpec:
    """Per-stage composition rule."""

    composition_frame: str = "task"  # 'task' or 'world'
    axis_ownership: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=bool)
    )
    residual_bounds: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02])
    )
    fallback: dict[str, Any] = field(
        default_factory=lambda: {"missing_anchor": "hold", "missing_residual": "zero"}
    )
