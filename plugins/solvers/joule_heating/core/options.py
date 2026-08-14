"""Configuration options for the JouleHeatingSolver plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class JouleHeatingOptions:
    """Options for the Joule-heating (electric-current + heat-source) solver.

    Parameters
    ----------
    dt : float
        Time step used by the solver. Defaults to ``None``, in which case it is
        set to the simulator's sub-step size at install time.
    dim : int
        Spatial dimension: 2 or 3.
    resolution : tuple[int, ...]
        Number of grid cells in each spatial dimension, e.g. ``(64, 64)``.
    dx : float
        Grid spacing in metres.
    sigma : float | np.ndarray
        Electrical conductivity ``σ`` in S/m. A scalar is broadcast to the whole
        grid; an array must have the grid shape.
    rho : float
        Mass density of the material in kg/m³. Used for the internal transient
        thermal solve when ``couple_to_thermal=False``.
    cp : float
        Specific heat capacity of the material in J/(kg·K). Used for the
        internal transient thermal solve when ``couple_to_thermal=False``.
    k : float
        Thermal conductivity in W/(m·K). Used for the internal transient
        thermal solve when ``couple_to_thermal=False``.
    initial_temperature : float | np.ndarray | Callable | None
        Initial temperature field for the internal thermal solve. A scalar is
        broadcast; an array must have the grid shape; a callable receives the
        coordinate meshgrid arrays.
    voltage_boundary_value : float
        Default Dirichlet voltage applied to all domain boundaries.
    max_iter : int
        Maximum Jacobi iterations for the electric-potential solve per substep.
    tol : float
        Convergence tolerance for the electric-potential solve.
    couple_to_thermal : bool
        If ``True`` and ``scene.sim.thermal_solver`` exists, the computed
        volumetric heat source ``Q`` is injected into the thermal solver instead
        of maintaining an internal temperature field.
    """

    dt: float | None = None
    dim: int = 2
    resolution: tuple[int, ...] = (64, 64)
    dx: float = 0.01
    sigma: float | np.ndarray = 5.8e7
    rho: float = 8960.0
    cp: float = 385.0
    k: float = 400.0
    initial_temperature: float | np.ndarray | Callable | None = 300.0
    voltage_boundary_value: float = 0.0
    max_iter: int = 1000
    tol: float = 1e-6
    couple_to_thermal: bool = False

    def __post_init__(self) -> None:
        """Validate options."""
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if len(self.resolution) != self.dim:
            raise ValueError(
                f"resolution {self.resolution} must have length {self.dim}"
            )
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.cp <= 0:
            raise ValueError(f"cp must be positive, got {self.cp}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
