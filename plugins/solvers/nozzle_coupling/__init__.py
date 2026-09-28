"""Nozzle exhaust <-> 3D jet bidirectional coupling solver plugin.

Dual-mode, like the other ``plugins/solvers``:

* Headless: the genesis-free numerical cores (quasi-1D nozzle, anelastic 3D
  jet, macro-step coupler) run on CPU/GPU via PyTorch + NumPy alone.
* Genesis: ``install(scene, options)`` / ``install_nozzle`` / ``install_jet``
  inject the solvers into a ``gs.Scene`` before ``scene.build()`` so they
  integrate with ``scene.step()``, substeps, reset and checkpointing exactly
  like the cfd_coupling / thermal / joule_heating / acoustics plugins.

Intended for simulating the engine/nozzle stage of published open-source
rocket projects (e.g. the MANPADS prototype's published nozzle CAD / OpenRocket
assets): the 1D side transfers transient mass flux, temperature and two-phase
(condensed fraction) parameters to the 3D plume, and the 3D exit-plane
backpressure acts back on the 1D nozzle (choking / shock-position / throttling
response). All geometry and thermodynamic inputs are user parameters; the
plugin contains no propellant or motor data.
"""

__version__ = "0.1.0"
__source__ = "genesis-cloud-sim"
__author__ = "Genesis Cloud Sim Team"

from typing import TYPE_CHECKING

from plugins.solvers.nozzle_coupling.core import (
    ExitState,
    Jet3D,
    JetOptions,
    Nozzle1D,
    NozzleCoupler,
    NozzleCouplingOptions,
    NozzleMacroStepLog,
    NozzleOptions,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

__all__ = [
    "CoupledNozzleSolver",
    "CoupledNozzleSolverOptions",
    "ExitState",
    "Jet3D",
    "JetOptions",
    "JetSolver",
    "JetSolverOptions",
    "Nozzle1D",
    "NozzleCoupler",
    "NozzleCouplingOptions",
    "NozzleMacroStepLog",
    "NozzleOptions",
    "NozzleSolver",
    "NozzleSolverOptions",
    "install",
    "install_jet",
    "install_nozzle",
]


def install(scene: "Scene", options=None):
    """Inject a ``CoupledNozzleSolver`` into ``scene`` before ``scene.build()``.

    Thin lazy wrapper so the numerical core stays importable without
    genesis-world; see ``nozzle_coupling.solver`` for the full docstring.
    """
    from plugins.solvers.nozzle_coupling.solver import install as _install

    return _install(scene, options)


def install_nozzle(scene: "Scene", options=None):
    """Inject a standalone quasi-1D ``NozzleSolver`` into ``scene``."""
    from plugins.solvers.nozzle_coupling.solver import (
        install_nozzle as _install,
    )

    return _install(scene, options)


def install_jet(scene: "Scene", options=None):
    """Inject a standalone 3D ``JetSolver`` into ``scene``."""
    from plugins.solvers.nozzle_coupling.solver import install_jet as _install

    return _install(scene, options)


# Auto-register with the project plugin manager if it is available.
try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    _pm = get_plugin_manager()
except Exception:
    pass
