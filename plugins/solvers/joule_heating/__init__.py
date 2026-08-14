"""Joule-heating solver plugin for genesis-world 1.3.2."""

from __future__ import annotations

from .core import JouleHeatingOptions, JouleHeatingSolver

__all__ = ["JouleHeatingOptions", "JouleHeatingSolver", "install"]


def install(scene, options=None):
    """Install the JouleHeatingSolver into a Genesis scene.

    Must be called before ``scene.build()``.

    Parameters
    ----------
    scene : genesis.engine.scene.Scene
        The scene to install the solver into.
    options : JouleHeatingOptions | None
        Solver options. If ``None``, default options are used. If ``dt`` is not
        set, it is defaulted to ``scene.sim_options.dt``.

    Returns:
    -------
    JouleHeatingSolver
        The installed solver instance.
    """
    options = options or JouleHeatingOptions()
    if options.dt is None:
        options = JouleHeatingOptions(
            **{**options.__dict__, "dt": scene.sim_options.dt}
        )
    solver = JouleHeatingSolver(scene, scene.sim, options)
    scene.sim.joule_heating_solver = solver
    scene.sim._solvers.append(solver)
    return solver


try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    _pm = get_plugin_manager()
except Exception:
    pass
