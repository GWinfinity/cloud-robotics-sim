"""AcousticsSolver plugin for genesis-world 1.4.0.

This plugin provides a grid-based time-domain acoustic (linear wave
equation) solver that can be injected into a ``gs.Scene`` and stepped
alongside the built-in Genesis solvers.

Example:
-------
>>> import genesis as gs
>>> from plugins.solvers.acoustics import install, AcousticsOptions
>>>
>>> gs.init(backend=gs.cpu)
>>> scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-4))
>>> scene.add_entity(gs.morphs.Plane())
>>>
>>> acoustics = install(scene, AcousticsOptions(resolution=(64, 64),
...                                             dx=0.01, c=343.0))
>>> probe = acoustics.add_probe(position=(0.32, 0.32, 0.0))
>>> source = acoustics.add_source(position=(0.16, 0.16, 0.0),
...                               signal=lambda t: 10.0 * __import__("math").sin(2 * 3.1415926 * 2000 * t))
>>> scene.build()
>>>
>>> for _ in range(200):
...     scene.step()
>>> p = acoustics.get_signal(probe)
"""

__version__ = "0.1.0"
__source__ = "genesis-cloud-sim"
__author__ = "Genesis Cloud Sim Team"

from typing import TYPE_CHECKING

from .core.acoustics_solver import (
    AcousticBody,
    AcousticProbe,
    AcousticSource,
    AcousticsSolver,
)
from .core.options import AcousticsOptions

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

__all__ = [
    "AcousticBody",
    "AcousticProbe",
    "AcousticSource",
    "AcousticsOptions",
    "AcousticsSolver",
    "install",
]


def install(scene: "Scene", options: AcousticsOptions | None = None) -> AcousticsSolver:
    """Inject an ``AcousticsSolver`` into ``scene`` before ``scene.build()``.

    Parameters
    ----------
    scene : gs.Scene
        The Genesis scene to augment.
    options : AcousticsOptions | None
        Solver options. ``dt`` defaults to ``scene.sim_options.dt`` if not set.

    Returns:
    -------
    AcousticsSolver
        The injected solver, also available as ``scene.sim.acoustics_solver``.
    """
    options = options or AcousticsOptions()
    if options.dt is None:
        options = AcousticsOptions(**{**options.__dict__, "dt": scene._sim.dt})

    solver = AcousticsSolver(scene, scene.sim, options)
    scene.sim.acoustics_solver = solver
    scene.sim._solvers.append(solver)
    return solver


# Auto-register with the project plugin manager if it is available.
try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    _pm = get_plugin_manager()
except Exception:
    pass
