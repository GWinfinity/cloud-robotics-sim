"""Genesis-world 1.4 integration for the nozzle_coupling solvers.

This module is the only part of the plugin that depends on genesis-world.
The numerical cores in ``core/`` stay genesis-free; these thin wrappers
inject them into ``gs.Scene`` exactly like the cfd_coupling / thermal /
joule_heating / acoustics plugins:

* ``install(scene, options)`` appends a ``CoupledNozzleSolver`` to
  ``scene.sim._solvers`` before ``scene.build()``; each genesis substep
  (``simulator.substep_pre_coupling``) advances one coupling macro step
  (1D nozzle sub-cycling + 3D jet step + fixed-point interface iteration
  with slew-limited backpressure / mass-flux exchange).
* ``install_nozzle`` / ``install_jet`` inject the standalone 1D / 3D cores.

Genesis 1.4 quirks handled here (same as the other solver plugins):

* ``Solver`` no longer mixes in ``TimeBasedMixin``: ``_substep_dt`` is set
  from ``sim.substep_dt`` in ``build()``.
* A dummy entity marker makes ``Simulator.reset()`` restore our state through
  ``get_state`` / ``set_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from genesis.engine.solvers.base_solver import Solver

from plugins.solvers.nozzle_coupling.core import (
    Jet3D,
    JetOptions,
    Nozzle1D,
    NozzleCoupler,
    NozzleCouplingOptions,
    NozzleOptions,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class _MarkerEntity:
    """Dummy marker so ``n_entities > 0`` and Simulator.reset() restores state."""


class _PluginSolverBase:
    """No-op lifecycle hooks the genesis Simulator calls on active solvers.

    Forward integration happens in ``substep_pre_coupling``; the remaining
    hooks are intentionally inert (no coupling to genesis entities, no
    back-propagation through the field solvers yet).
    """

    def process_input(self, in_backward: bool = False) -> None:
        pass

    def process_input_grad(self) -> None:
        pass

    def substep_pre_coupling_grad(self, f: int) -> None:
        pass

    def substep_post_coupling(self, f: int) -> None:
        pass

    def substep_post_coupling_grad(self, f: int) -> None:
        pass

    def save_ckpt(self, ckpt_name) -> None:
        pass

    def load_ckpt(self, ckpt_name) -> None:
        pass

    def reset_grad(self) -> None:
        pass

    def collect_output_grads(self) -> None:
        pass

    def add_grad_from_state(self, state) -> None:
        pass


@dataclass
class CoupledNozzleSolverOptions:
    """Options for the coupled nozzle <-> 3D jet plugin solver.

    Parameters
    ----------
    dt : float | None
        Reserved for interface parity with the other solver plugins; the
        effective macro step is the genesis substep snapped onto
        ``coupling.macro_dt`` (see ``build()``).
    nozzle, jet, coupling :
        Core solver options (see ``NozzleOptions`` / ``JetOptions`` /
        ``NozzleCouplingOptions``).
    """

    dt: float | None = None
    nozzle: NozzleOptions = field(default_factory=NozzleOptions)
    jet: JetOptions = field(default_factory=JetOptions)
    coupling: NozzleCouplingOptions = field(default_factory=NozzleCouplingOptions)


@dataclass
class NozzleSolverOptions:
    """Options for the standalone quasi-1D nozzle plugin solver."""

    dt: float | None = None
    nozzle: NozzleOptions = field(default_factory=NozzleOptions)
    backpressure_pa: float = 101325.0  # fixed backpressure [Pa]


@dataclass
class JetSolverOptions:
    """Options for the standalone 3D jet plugin solver."""

    dt: float | None = None
    jet: JetOptions = field(default_factory=JetOptions)
    inlet_mass_flux: float = 0.0  # prescribed inlet mass flux [kg/s]


# --------------------------------------------------------------------- #
# Coupled solver
# --------------------------------------------------------------------- #
class CoupledNozzleSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the nozzle <-> 3D jet bidirectional coupler.

    One genesis substep = one coupling macro step. Use ``solver.coupler``,
    ``solver.nozzle`` and ``solver.jet`` to reach the underlying cores
    (probes, histories, ``jet.set_outlet_patch`` throttling events, ...).
    """

    def __init__(
        self, scene: "Scene", sim: "Simulator", options: CoupledNozzleSolverOptions
    ):
        super().__init__(scene, sim, options)
        self._options = options
        self._nozzle = Nozzle1D(options.nozzle)
        self._jet = Jet3D(options.jet)
        self._coupler = NozzleCoupler(self._nozzle, self._jet, options.coupling)
        self._substep_dt = options.coupling.macro_dt

    @property
    def is_active(self) -> bool:
        return True

    @property
    def coupler(self) -> NozzleCoupler:
        return self._coupler

    @property
    def nozzle(self) -> Nozzle1D:
        return self._nozzle

    @property
    def jet(self) -> Jet3D:
        return self._jet

    def build(self) -> None:
        super().build()
        # genesis 1.4 Solver no longer provides TimeBasedMixin's _substep_dt.
        self._substep_dt = self._sim.substep_dt
        # One genesis substep = one coupling macro step; snap the substep to
        # the configured macro_dt (the 3D CFL usually requires macro_dt <=
        # substep_dt, in which case the substep is repeated).
        n = max(1, int(round(self._substep_dt / self._coupler.o.macro_dt)))
        self._n_sub = n
        self._substep_dt = n * self._coupler.o.macro_dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        for _ in range(self._n_sub):
            self._coupler.step()

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return (
            self._nozzle.get_state(),
            self._jet.get_state(),
            self._coupler.backpressure,
            self._coupler._mdot_applied,
        )

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        nozzle_state, jet_state, backpressure, mdot_applied = state
        self._nozzle.set_state(nozzle_state)
        self._jet.set_state(jet_state)
        self._coupler.backpressure = backpressure
        self._coupler._mdot_applied = mdot_applied
        self._coupler.logs.clear()


# --------------------------------------------------------------------- #
# Standalone solvers
# --------------------------------------------------------------------- #
class NozzleSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the standalone quasi-1D nozzle core."""

    def __init__(self, scene: "Scene", sim: "Simulator", options: NozzleSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._nozzle = Nozzle1D(options.nozzle)
        self._substep_dt = 1.0e-4
        self._n_sub = 1

    @property
    def is_active(self) -> bool:
        return True

    @property
    def nozzle(self) -> Nozzle1D:
        return self._nozzle

    def build(self) -> None:
        super().build()
        self._substep_dt = self._sim.substep_dt
        self._n_sub = max(1, int(round(self._substep_dt / self._nozzle.dt)))
        self._substep_dt = self._n_sub * self._nozzle.dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        for _ in range(self._n_sub):
            self._nozzle.step(
                backpressure_pa=self._options.backpressure_pa, dt=self._nozzle.dt
            )

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return self._nozzle.get_state()

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        self._nozzle.set_state(state)


class JetSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the standalone 3D jet core."""

    def __init__(self, scene: "Scene", sim: "Simulator", options: JetSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._jet = Jet3D(options.jet)
        self._jet.set_inlet(options.inlet_mass_flux)
        self._substep_dt = 0.0

    @property
    def is_active(self) -> bool:
        return True

    @property
    def jet(self) -> Jet3D:
        return self._jet

    def build(self) -> None:
        super().build()
        self._substep_dt = self._sim.substep_dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        self._jet.step(self._substep_dt)

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return self._jet.get_state()

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        self._jet.set_state(state)


# --------------------------------------------------------------------- #
# install() entry points
# --------------------------------------------------------------------- #
def install(
    scene: "Scene", options: CoupledNozzleSolverOptions | None = None
) -> CoupledNozzleSolver:
    """Inject a ``CoupledNozzleSolver`` into ``scene`` before ``scene.build()``.

    The solver is also available as ``scene.sim.nozzle_coupling_solver``.
    """
    options = options or CoupledNozzleSolverOptions()
    solver = CoupledNozzleSolver(scene, scene.sim, options)
    scene.sim.nozzle_coupling_solver = solver
    scene.sim._solvers.append(solver)
    return solver


def install_nozzle(
    scene: "Scene", options: NozzleSolverOptions | None = None
) -> NozzleSolver:
    """Inject a standalone quasi-1D ``NozzleSolver`` into ``scene``."""
    options = options or NozzleSolverOptions()
    solver = NozzleSolver(scene, scene.sim, options)
    scene.sim.nozzle1d_solver = solver
    scene.sim._solvers.append(solver)
    return solver


def install_jet(scene: "Scene", options: JetSolverOptions | None = None) -> JetSolver:
    """Inject a standalone 3D ``JetSolver`` into ``scene``."""
    options = options or JetSolverOptions()
    solver = JetSolver(scene, scene.sim, options)
    scene.sim.jet3d_solver = solver
    scene.sim._solvers.append(solver)
    return solver


__all__ = [
    "CoupledNozzleSolver",
    "CoupledNozzleSolverOptions",
    "JetSolver",
    "JetSolverOptions",
    "NozzleSolver",
    "NozzleSolverOptions",
    "install",
    "install_jet",
    "install_nozzle",
]
