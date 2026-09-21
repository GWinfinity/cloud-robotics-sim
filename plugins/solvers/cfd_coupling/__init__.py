"""1D pipe network <-> 3D CFD bidirectional coupling solver plugin.

Dual-mode, like the other ``plugins/solvers``:

* Headless: the genesis-free numerical cores (1D MOC pipe, 3D projection CFD,
  macro-step coupler) run on CPU/GPU via PyTorch + NumPy alone.
* Genesis: ``install(scene, options)`` / ``install_pipe`` / ``install_cfd``
  inject the solvers into a ``gs.Scene`` before ``scene.build()`` so they
  integrate with ``scene.step()``, substeps, reset and checkpointing exactly
  like the thermal / joule_heating / acoustics plugins.

This package answers the COMAC competition problem "1D-3D CFD coupled
simulation: time-step coordination and boundary coupling"; see README.md.
"""

__version__ = "0.1.0"
__source__ = "genesis-cloud-sim"
__author__ = "Genesis Cloud Sim Team"

from typing import TYPE_CHECKING

from plugins.solvers.cfd_coupling.core import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    MacroStepLog,
    Pipe1D,
    PipeOptions,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

__all__ = [
    "CFD3D",
    "CFDOptions",
    "CFDSolver",
    "CFDSolverOptions",
    "CoupledCFDSolver",
    "CoupledSolverOptions",
    "Coupler",
    "CouplingOptions",
    "MacroStepLog",
    "Pipe1D",
    "PipeOptions",
    "PipeSolver",
    "PipeSolverOptions",
    "install",
    "install_cfd",
    "install_pipe",
]


def install(scene: "Scene", options=None):
    """Inject a ``CoupledCFDSolver`` into ``scene`` before ``scene.build()``.

    Thin lazy wrapper so the numerical core stays importable without
    genesis-world; see ``cfd_coupling.solver`` for the full docstring.
    """
    from plugins.solvers.cfd_coupling.solver import install as _install

    return _install(scene, options)


def install_pipe(scene: "Scene", options=None):
    """Inject a standalone 1D MOC ``PipeSolver`` into ``scene``."""
    from plugins.solvers.cfd_coupling.solver import install_pipe as _install

    return _install(scene, options)


def install_cfd(scene: "Scene", options=None):
    """Inject a standalone 3D projection ``CFDSolver`` into ``scene``."""
    from plugins.solvers.cfd_coupling.solver import install_cfd as _install

    return _install(scene, options)


# Auto-register with the project plugin manager if it is available.
try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    _pm = get_plugin_manager()
except Exception:
    pass
