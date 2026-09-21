"""Genesis-world 1.4 integration for the cfd_coupling solvers.

This module is the only part of the plugin that depends on genesis-world.
The numerical cores in ``core/`` stay genesis-free; these thin wrappers
inject them into ``gs.Scene`` exactly like the thermal / joule_heating /
acoustics plugins:

* ``install(scene, options)`` appends a ``CoupledCFDSolver`` to
  ``scene.sim._solvers`` before ``scene.build()``; each genesis substep
  (``simulator.substep_pre_coupling``) advances one coupling macro step
  (1D MOC sub-cycling + 3D CFD step + fixed-point interface iteration).
* ``install_pipe`` / ``install_cfd`` inject the standalone 1D / 3D cores.

Genesis 1.4 quirks handled here (same as the other solver plugins):

* ``Solver`` no longer mixes in ``TimeBasedMixin``: ``_substep_dt`` is set
  from ``sim.substep_dt`` in ``build()``.
* A dummy entity marker makes ``Simulator.reset()`` restore our state through
  ``get_state`` / ``set_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import genesis as gs
from genesis.engine.solvers.base_solver import Solver

from plugins.solvers.cfd_coupling.core import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    Pipe1D,
    PipeOptions,
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
        # Back-propagation through the field solvers is not supported; the
        # forward pass needs no per-substep checkpoint of torch state.
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
class CoupledSolverOptions:
    """Options for the coupled 1D-3D CFD plugin solver.

    Parameters
    ----------
    dt : float | None
        Reserved for interface parity with the other solver plugins; the
        effective macro step is the genesis substep snapped to the 1D MOC
        step grid (``macro_dt = n * dx / a``).
    pipe, cfd, coupling :
        Core solver options (see ``PipeOptions`` / ``CFDOptions`` /
        ``CouplingOptions``).
    """

    dt: float | None = None
    pipe: PipeOptions = field(default_factory=PipeOptions)
    cfd: CFDOptions = field(default_factory=CFDOptions)
    coupling: CouplingOptions = field(default_factory=CouplingOptions)


@dataclass
class PipeSolverOptions:
    """Options for the standalone 1D MOC pipe plugin solver."""

    dt: float | None = None
    pipe: PipeOptions = field(default_factory=PipeOptions)


@dataclass
class CFDSolverOptions:
    """Options for the standalone 3D CFD plugin solver."""

    dt: float | None = None
    cfd: CFDOptions = field(default_factory=CFDOptions)
    inlet_velocity: float = 0.0  # prescribed inlet velocity [m/s]


# --------------------------------------------------------------------- #
# Coupled solver
# --------------------------------------------------------------------- #
class CoupledCFDSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the 1D(MOC) <-> 3D(CFD) bidirectional coupler.

    One genesis substep = one coupling macro step. Use ``solver.coupler``,
    ``solver.pipe`` and ``solver.cfd`` to reach the underlying cores
    (probes, valve schedules, histories, ...).
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: CoupledSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._pipe = Pipe1D(options.pipe)
        self._cfd = CFD3D(options.cfd)
        self._coupler = Coupler(self._pipe, self._cfd, options.coupling)
        self._substep_dt: float = options.coupling.macro_dt

    @property
    def is_active(self) -> bool:
        return True

    @property
    def coupler(self) -> Coupler:
        return self._coupler

    @property
    def pipe(self) -> Pipe1D:
        return self._pipe

    @property
    def cfd(self) -> CFD3D:
        return self._cfd

    def build(self) -> None:
        super().build()
        # genesis 1.4 Solver no longer provides TimeBasedMixin's _substep_dt.
        self._substep_dt = self._sim.substep_dt
        # Snap the macro step to the 1D MOC step grid (Courant = 1 fixes
        # dt_1d = dx / a); prefer scene substeps that divide evenly.
        n = max(1, int(round(self._substep_dt / self._pipe.dt)))
        self._coupler.macro_dt = n * self._pipe.dt
        self._coupler.n_substeps = n
        self._substep_dt = self._coupler.macro_dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        self._coupler.step()

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return (
            self._pipe.get_state(),
            self._cfd.get_state(),
            self._coupler.plenum_head,
            self._coupler._last_q_mean,
        )

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        pipe_state, cfd_state, plenum_head, last_q_mean = state
        self._pipe.set_state(pipe_state)
        self._cfd.set_state(cfd_state)
        self._coupler.plenum_head = plenum_head
        self._coupler._last_q_mean = last_q_mean
        self._coupler.logs.clear()


# --------------------------------------------------------------------- #
# Standalone solvers
# --------------------------------------------------------------------- #
class PipeSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the standalone 1D MOC pipe core."""

    def __init__(self, scene: "Scene", sim: "Simulator", options: PipeSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._pipe = Pipe1D(options.pipe)
        self._n_sub = 1
        self._substep_dt = self._pipe.dt

    @property
    def is_active(self) -> bool:
        return True

    @property
    def pipe(self) -> Pipe1D:
        return self._pipe

    def build(self) -> None:
        super().build()
        self._substep_dt = self._sim.substep_dt
        self._n_sub = max(1, int(round(self._substep_dt / self._pipe.dt)))
        self._substep_dt = self._n_sub * self._pipe.dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        for _ in range(self._n_sub):
            self._pipe.step()

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return self._pipe.get_state()

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        self._pipe.set_state(state)


class CFDSolver(_PluginSolverBase, Solver):
    """Plugin solver wrapping the standalone 3D projection CFD core."""

    def __init__(self, scene: "Scene", sim: "Simulator", options: CFDSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._cfd = CFD3D(options.cfd)
        self._cfd.set_inlet(options.inlet_velocity)
        self._substep_dt = 0.0

    @property
    def is_active(self) -> bool:
        return True

    @property
    def cfd(self) -> CFD3D:
        return self._cfd

    def build(self) -> None:
        super().build()
        self._substep_dt = self._sim.substep_dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        self._cfd.step(self._substep_dt)

    def get_state(self, f: int):
        if not self.is_active:
            return None
        return self._cfd.get_state()

    def set_state(self, f: int, state, envs_idx=None) -> None:
        if state is None:
            return
        self._cfd.set_state(state)


# --------------------------------------------------------------------- #
# install() entry points
# --------------------------------------------------------------------- #
def install(scene: "Scene", options: CoupledSolverOptions | None = None) -> CoupledCFDSolver:
    """Inject a ``CoupledCFDSolver`` into ``scene`` before ``scene.build()``.

    The solver is also available as ``scene.sim.cfd_coupling_solver``.
    """
    options = options or CoupledSolverOptions()
    solver = CoupledCFDSolver(scene, scene.sim, options)
    scene.sim.cfd_coupling_solver = solver
    scene.sim._solvers.append(solver)
    return solver


def install_pipe(scene: "Scene", options: PipeSolverOptions | None = None) -> PipeSolver:
    """Inject a standalone ``PipeSolver`` (1D MOC water hammer)."""
    options = options or PipeSolverOptions()
    solver = PipeSolver(scene, scene.sim, options)
    scene.sim.pipe1d_solver = solver
    scene.sim._solvers.append(solver)
    return solver


def install_cfd(scene: "Scene", options: CFDSolverOptions | None = None) -> CFDSolver:
    """Inject a standalone ``CFDSolver`` (3D projection-method CFD)."""
    options = options or CFDSolverOptions()
    solver = CFDSolver(scene, scene.sim, options)
    scene.sim.cfd3d_solver = solver
    scene.sim._solvers.append(solver)
    return solver


__all__ = [
    "CFDSolver",
    "CFDSolverOptions",
    "CoupledCFDSolver",
    "CoupledSolverOptions",
    "PipeSolver",
    "PipeSolverOptions",
    "install",
    "install_cfd",
    "install_pipe",
]
