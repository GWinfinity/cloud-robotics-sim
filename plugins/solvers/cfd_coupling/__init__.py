"""1D pipe network <-> 3D CFD bidirectional coupling prototype.

Self-contained (no genesis-world dependency): the 1D MOC pipe solver and the
3D projection-method CFD solver run headless on CPU/GPU via PyTorch, coupled
through a macro-step coordinator with sub-cycling, fixed-point iteration and
millisecond-scale valve control events.

This package answers the COMAC competition problem "1D-3D CFD coupled
simulation: time-step coordination and boundary coupling"; see README.md.
"""

__version__ = "0.1.0"
__source__ = "genesis-cloud-sim"
__author__ = "Genesis Cloud Sim Team"

from plugins.solvers.cfd_coupling.core import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    MacroStepLog,
    Pipe1D,
    PipeOptions,
)

__all__ = [
    "CFD3D",
    "CFDOptions",
    "Coupler",
    "CouplingOptions",
    "MacroStepLog",
    "Pipe1D",
    "PipeOptions",
]
