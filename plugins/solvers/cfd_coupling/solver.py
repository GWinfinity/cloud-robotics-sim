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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from genesis.engine.solvers.base_solver import Solver

from plugins.solvers.cfd_coupling.core import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    Pipe1D,
    PipeOptions,
)


def _make_cfd_core(cfd_options, backend: str):
    """Instantiate the CFD core on the requested backend.

    ``backend`` is one of ``"quadrants"`` (default: genesis-native
    ``@qd.kernel`` implementation), ``"torch"`` (pure PyTorch core, works
    without genesis) or ``"auto"`` (quadrants when genesis is importable,
    else torch). The two backends are numerically equivalent and
    duck-type-compatible with the coupler.
    """
    if backend in ("auto", "quadrants"):
        try:
            from plugins.solvers.cfd_coupling.core.cfd3d_qd import (
                QDCFD3D,
                QDCFDOptions,
            )

            if isinstance(cfd_options, QDCFDOptions):
                qd_opts = cfd_options
            else:
                qd_opts = QDCFDOptions(
                    **{
                        f.name: getattr(cfd_options, f.name, f.default)
                        for f in QDCFDOptions.__dataclass_fields__.values()
                    }
                )
            return QDCFD3D(qd_opts)
        except ImportError:
            if backend == "quadrants":
                raise
    return CFD3D(cfd_options)


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class _MarkerEntity:
    """Dummy marker so ``n_entities > 0`` and Simulator.reset() restores state."""


# --------------------------------------------------------------------- #
# Obstacle bridge: CATIA/CAD meshes and genesis entities -> solid mask
# --------------------------------------------------------------------- #
def _placement_matrix(
    position: tuple[float, float, float] | None,
    rotation_euler: tuple[float, float, float] | None,
    scale: float | tuple[float, float, float] | None,
) -> np.ndarray | None:
    """Compose the world-placement 4x4 (scale -> rotate -> translate)."""
    if position is None and rotation_euler is None and scale is None:
        return None
    from scipy.spatial.transform import Rotation

    mat = np.eye(4)
    if scale is not None:
        s = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
        mat[:3, :3] = np.diag(s)
    if rotation_euler is not None:
        mat[:3, :3] = (
            Rotation.from_euler("xyz", rotation_euler, degrees=True).as_matrix()
            @ mat[:3, :3]
        )
    if position is not None:
        mat[:3, 3] = np.asarray(position, dtype=np.float64)
    return mat


def _entity_to_trimesh(entity):
    """Merge a built genesis RigidEntity's geoms into one world-frame mesh."""
    import trimesh
    from scipy.spatial.transform import Rotation

    meshes = []
    for link in entity.links:
        for geom in link.geoms:
            tm = geom.get_trimesh()
            if tm is None or len(tm.vertices) == 0:
                continue
            pos = np.asarray(geom.get_pos(relative=False)).reshape(3)
            quat = np.asarray(geom.get_quat(relative=False)).reshape(4)  # wxyz
            mat = np.eye(4)
            mat[:3, :3] = Rotation.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()
            mat[:3, 3] = pos
            m = tm.copy()
            m.apply_transform(mat)
            meshes.append(m)
    if not meshes:
        raise ValueError(
            f"entity {entity!r} yielded no meshes; pass a file path or a "
            "trimesh.Trimesh instead"
        )
    return trimesh.util.concatenate(meshes)


def _add_obstacle_to_core(cfd, source, position, rotation_euler, scale) -> np.ndarray:
    """Rasterize ``source`` into a solid mask and union it into ``cfd``."""
    from plugins.solvers.cfd_coupling.core.obstacles import (
        combine_masks,
        mask_from_mesh,
    )

    if isinstance(source, (str, bytes, Path)):
        import trimesh

        mesh = trimesh.load(str(source))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
    elif hasattr(source, "links") and hasattr(source, "get_pos"):
        mesh = _entity_to_trimesh(source)
    elif hasattr(source, "vertices") and hasattr(source, "faces"):
        mesh = source  # already a trimesh.Trimesh
    else:
        raise TypeError(
            "source must be a mesh file path, a trimesh.Trimesh, or a built "
            f"genesis RigidEntity, got {type(source)!r}"
        )
    transform = _placement_matrix(position, rotation_euler, scale)
    mask = mask_from_mesh(mesh, tuple(cfd.o.domain), tuple(cfd.o.cells), transform)
    cfd.set_solid_mask(combine_masks(cfd.solid_mask, mask))
    return mask


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
    backend : str
        CFD core backend: ``"quadrants"`` (default, genesis-native
        ``@qd.kernel`` s), ``"torch"`` (genesis-free), or ``"auto"``.
    """

    dt: float | None = None
    pipe: PipeOptions = field(default_factory=PipeOptions)
    cfd: CFDOptions = field(default_factory=CFDOptions)
    coupling: CouplingOptions = field(default_factory=CouplingOptions)
    backend: str = "quadrants"


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
    backend: str = "quadrants"  # CFD core backend (see _make_cfd_core)


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
        self._cfd = _make_cfd_core(options.cfd, options.backend)
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
    def cfd(self):
        return self._cfd

    def add_obstacle(
        self,
        source,
        position: tuple[float, float, float] | None = None,
        rotation_euler: tuple[float, float, float] | None = None,
        scale: float | tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Add an immersed no-slip obstacle to the 3D CFD domain.

        ``source`` is a mesh file path (STL/OBJ/GLB, e.g. exported from
        CATIA), a ``trimesh.Trimesh``, or a built genesis RigidEntity.
        ``position`` / ``rotation_euler`` (degrees, xyz) / ``scale`` place
        the source into the CFD world frame (metres) on top of any
        transform the source itself carries. Returns the mask this
        obstacle contributed; masks of successive calls are unioned.

        Obstacles must stay clear of the x = 0 / x = Lx faces. The mask is
        static within a step; call again (e.g. after moving an entity) to
        re-rasterize. Requires ``scene.build()`` to have happened when
        ``source`` is an entity.
        """
        return _add_obstacle_to_core(self._cfd, source, position, rotation_euler, scale)

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
        self._cfd = _make_cfd_core(options.cfd, options.backend)
        self._cfd.set_inlet(options.inlet_velocity)
        self._substep_dt = 0.0

    @property
    def is_active(self) -> bool:
        return True

    @property
    def cfd(self):
        return self._cfd

    def add_obstacle(
        self,
        source,
        position: tuple[float, float, float] | None = None,
        rotation_euler: tuple[float, float, float] | None = None,
        scale: float | tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Add an immersed no-slip obstacle; see CoupledCFDSolver.add_obstacle."""
        return _add_obstacle_to_core(self._cfd, source, position, rotation_euler, scale)

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
def install(
    scene: "Scene", options: CoupledSolverOptions | None = None
) -> CoupledCFDSolver:
    """Inject a ``CoupledCFDSolver`` into ``scene`` before ``scene.build()``.

    The solver is also available as ``scene.sim.cfd_coupling_solver``.
    """
    options = options or CoupledSolverOptions()
    solver = CoupledCFDSolver(scene, scene.sim, options)
    scene.sim.cfd_coupling_solver = solver
    scene.sim._solvers.append(solver)
    return solver


def install_pipe(
    scene: "Scene", options: PipeSolverOptions | None = None
) -> PipeSolver:
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
