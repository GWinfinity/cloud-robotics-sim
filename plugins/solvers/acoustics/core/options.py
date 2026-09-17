"""Configuration options for the AcousticsSolver plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class AcousticsOptions:
    """Options for the grid-based time-domain acoustics solver.

    Parameters
    ----------
    dt : float
        Time step used by the acoustics solver. Defaults to ``None``, in which
        case it is set to the simulator's sub-step size at install time.
    dim : int
        Spatial dimension: 2 or 3.
    resolution : tuple[int, ...]
        Number of grid cells in each spatial dimension, e.g. ``(128, 128)``.
    dx : float
        Grid spacing in metres. The grid extent is
        ``resolution[i] * dx`` per axis, starting at the world origin.
    c : float
        Speed of sound in the acoustic medium in m/s (air: ~343).
    rho : float
        Mass density of the acoustic medium in kg/m³ (air: ~1.225). Used by
        the rigid-body velocity coupling.
    boundary_mode : str
        One of ``"absorbing"`` (sponge damping layers, default; approximate
        open domain), ``"dirichlet"`` (p = 0, pressure-release/open) or
        ``"neumann"`` (zero normal gradient, rigid reflecting wall).
    sponge_layers : int
        Thickness of the absorbing sponge layer in grid cells. Only used with
        ``boundary_mode="absorbing"``.
    sponge_max_damping : float
        Maximum sponge damping in 1/s applied at the outermost layer. The
        damping profile grows quadratically from the inner sponge edge. The
        default (3000) absorbs most of the incident energy per pass for
        air-like media; lower values give a partially reflective boundary.
    initial_pressure : float | np.ndarray | Callable | None
        Initial pressure field in Pa. A scalar is broadcast to the whole
        grid; an array must match ``resolution``; a callable receives the
        coordinate meshgrid arrays and should return an array of the same
        shape.
    max_sources : int
        Maximum number of monopole sources. Slots are pre-allocated at build
        time so that ``add_source()`` can be called after ``scene.build()``.
    max_bodies : int
        Maximum number of rigid-body velocity couplers (vibro-acoustic,
        one-way structure -> sound).
    max_probes : int
        Maximum number of virtual microphones.
    """

    dt: float | None = None
    dim: int = 2
    resolution: tuple[int, ...] = (128, 128)
    dx: float = 0.01
    c: float = 343.0
    rho: float = 1.225
    boundary_mode: str = "absorbing"
    sponge_layers: int = 12
    sponge_max_damping: float = 3000.0
    initial_pressure: float | np.ndarray | Callable | None = 0.0
    max_sources: int = 16
    max_bodies: int = 16
    max_probes: int = 16

    def __post_init__(self) -> None:
        """Validate options."""
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if len(self.resolution) != self.dim:
            raise ValueError(
                f"resolution {self.resolution} must have length {self.dim}"
            )
        if any(int(r) < 4 for r in self.resolution):
            raise ValueError(
                f"resolution {self.resolution} too small (need >= 4 per axis)"
            )
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.boundary_mode not in {"absorbing", "dirichlet", "neumann"}:
            raise ValueError(
                "boundary_mode must be 'absorbing', 'dirichlet' or 'neumann', "
                f"got {self.boundary_mode}"
            )
        if self.sponge_layers < 0:
            raise ValueError(
                f"sponge_layers must be non-negative, got {self.sponge_layers}"
            )
        if self.sponge_layers >= min(int(r) for r in self.resolution) // 2:
            raise ValueError(
                f"sponge_layers {self.sponge_layers} too thick for resolution "
                f"{self.resolution}"
            )
        if self.sponge_max_damping < 0:
            raise ValueError(
                f"sponge_max_damping must be non-negative, got {self.sponge_max_damping}"
            )
        for name in ("max_sources", "max_bodies", "max_probes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
