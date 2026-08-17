"""Configuration options for the ThermalSolver plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ThermalOptions:
    """Options for the grid-based thermal conduction solver.

    Parameters
    ----------
    dt : float
        Time step used by the thermal solver. Defaults to ``None``, in which case
        it is set to the simulator's sub-step size at install time.
    dim : int
        Spatial dimension: 2 or 3. Only ``dim=2`` is fully implemented in v1.
    resolution : tuple[int, ...]
        Number of grid cells in each spatial dimension, e.g. ``(64, 64)``.
    dx : float
        Grid spacing in metres.
    alpha : float
        Thermal diffusivity ``k / (rho * cp)``. Either ``alpha`` or all three
        of ``k``, ``rho``, ``cp`` must be provided.
    k : float | None
        Thermal conductivity. Used together with ``rho`` and ``cp`` to compute
        ``alpha`` when ``alpha`` is not given.
    rho : float | None
        Mass density.
    cp : float | None
        Specific heat capacity.
    boundary_mode : str
        One of ``"dirichlet"`` or ``"neumann"``. ``"periodic"`` is not yet
        supported.
    boundary_value : float
        Temperature enforced on Dirichlet boundaries.
    grid_rho : float
        Mass density of the grid medium, used to compute the thermal capacity
        of each cell for two-way entity coupling.
    grid_cp : float
        Specific heat capacity of the grid medium.
    solver_type : str
        One of ``"transient"`` or ``"steady"``. Only ``"transient"`` is
        implemented in v1.
    initial_temperature : float | np.ndarray | Callable | None
        Initial temperature field. A scalar is broadcast to the whole grid; an
        array must have shape ``(resolution[0], resolution[1])`` for 2D or the
        3D equivalent; a callable receives the coordinate meshgrid arrays and
        should return an array of the same shape.
    max_sources : int
        Maximum number of thermally coupled entities. Source slots are
        pre-allocated at build time so that ``add_source()`` can be called
        after ``scene.build()``.
    """

    dt: float | None = None
    dim: int = 2
    resolution: tuple[int, ...] = (64, 64)
    dx: float = 0.01
    alpha: float | None = None
    k: float | None = None
    rho: float | None = None
    cp: float | None = None
    boundary_mode: str = "dirichlet"
    boundary_value: float = 0.0
    grid_rho: float = 1.0
    grid_cp: float = 1.0
    solver_type: str = "transient"
    initial_temperature: float | np.ndarray | Callable | None = 0.0
    max_sources: int = 16

    def __post_init__(self) -> None:
        """Validate options and derive alpha from k/rho/cp when needed."""
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if len(self.resolution) != self.dim:
            raise ValueError(
                f"resolution {self.resolution} must have length {self.dim}"
            )
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.boundary_mode not in {"dirichlet", "neumann"}:
            raise ValueError(
                f"boundary_mode must be 'dirichlet' or 'neumann', got {self.boundary_mode}"
            )
        if self.grid_rho <= 0:
            raise ValueError(f"grid_rho must be positive, got {self.grid_rho}")
        if self.grid_cp <= 0:
            raise ValueError(f"grid_cp must be positive, got {self.grid_cp}")
        if self.solver_type not in {"transient", "steady"}:
            raise ValueError(
                f"solver_type must be 'transient' or 'steady', got {self.solver_type}"
            )
        if self.max_sources < 0:
            raise ValueError(
                f"max_sources must be non-negative, got {self.max_sources}"
            )

        if self.alpha is None:
            if self.k is None or self.rho is None or self.cp is None:
                raise ValueError(
                    "Either alpha or all three of k, rho, cp must be provided"
                )
            self.alpha = float(self.k / (self.rho * self.cp))
        else:
            self.alpha = float(self.alpha)
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
