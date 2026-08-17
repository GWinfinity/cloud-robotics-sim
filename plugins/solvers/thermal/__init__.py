"""ThermalSolver plugin for genesis-world 1.3.2.

This plugin provides a grid-based heat-conduction solver that can be injected
into a ``gs.Scene`` and stepped alongside the built-in Genesis solvers.

Example:
-------
>>> import genesis as gs
>>> from plugins.solvers.thermal import install, ThermalOptions
>>>
>>> gs.init(backend=gs.cpu)
>>> scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01))
>>> scene.add_entity(gs.morphs.Plane())
>>>
>>> thermal = install(scene, ThermalOptions(resolution=(64, 64), alpha=1e-4))
>>> scene.build()
>>>
>>> for _ in range(100):
...     scene.step()
>>> T = thermal.get_temperature()
"""

__version__ = "0.1.0"
__source__ = "genesis-cloud-sim"
__author__ = "Genesis Cloud Sim Team"

from typing import TYPE_CHECKING

import genesis as gs

from .core.options import ThermalOptions
from .core.thermal_solver import ThermalSolver, ThermalSource

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

__all__ = ["ThermalOptions", "ThermalSolver", "ThermalSource", "install"]


def install(scene: "Scene", options: ThermalOptions | None = None) -> ThermalSolver:
    """Inject a ``ThermalSolver`` into ``scene`` before ``scene.build()``.

    Parameters
    ----------
    scene : gs.Scene
        The Genesis scene to augment.
    options : ThermalOptions | None
        Solver options. ``dt`` defaults to ``scene.sim_options.dt`` if not set.

    Returns:
    -------
    ThermalSolver
        The injected solver, also available as ``scene.sim.thermal_solver``.
    """
    options = options or ThermalOptions()
    if options.dt is None:
        options = ThermalOptions(**{**options.__dict__, "dt": scene.sim_options.dt})

    solver = ThermalSolver(scene, scene.sim, options)
    scene.sim.thermal_solver = solver
    scene.sim._solvers.append(solver)
    return solver


# Auto-register with the project plugin manager if it is available.
try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    _pm = get_plugin_manager()
except Exception:
    pass
