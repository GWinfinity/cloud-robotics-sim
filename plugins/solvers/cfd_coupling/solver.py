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
# Obstacle bridge: CAD meshes, STEP assemblies, entities -> solid masks
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


def _coerce_placement(result) -> np.ndarray | None:
    """Normalize a pose-provider return value to a 4x4 (or None = keep)."""
    if result is None:
        return None
    arr = np.asarray(result, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3,):  # position only
        mat = np.eye(4)
        mat[:3, 3] = arr
        return mat
    raise ValueError(
        "pose provider must return None, a (3,) position, or a (4, 4) "
        f"matrix, got array of shape {arr.shape}"
    )


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


def _resolve_sources(source) -> tuple[list, bool]:
    """Normalize ``source`` into a list of trimesh meshes.

    Returns ``(meshes, is_world_snapshot)``: STEP assemblies expand to one
    mesh per solid; lists/tuples flatten recursively; a genesis entity
    becomes a single world-frame snapshot mesh (``is_world_snapshot=True``,
    i.e. model coordinates are already world coordinates and cannot be
    re-placed by a pose provider).
    """
    import trimesh

    from plugins.solvers.cfd_coupling.core.obstacles import meshes_from_step

    if isinstance(source, (str, bytes, Path)):
        suffix = Path(str(source)).suffix.lower()
        if suffix in (".stp", ".step", ".stpz"):
            return meshes_from_step(source), False
        mesh = trimesh.load(str(source))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        return [mesh], False
    if isinstance(source, (list, tuple)):
        meshes: list = []
        world = False
        for part in source:
            sub, sub_world = _resolve_sources(part)
            meshes.extend(sub)
            world = world or sub_world
        return meshes, world
    if hasattr(source, "links") and hasattr(source, "get_pos"):
        return [_entity_to_trimesh(source)], True
    if hasattr(source, "vertices") and hasattr(source, "faces"):
        return [source], False  # already a trimesh.Trimesh
    raise TypeError(
        "source must be a mesh/STEP file path, a trimesh.Trimesh, a list of "
        f"those (assembly), or a built genesis RigidEntity, got {type(source)!r}"
    )


def _rasterize_meshes(cfd, meshes, placement) -> np.ndarray:
    """Rasterize meshes into a solid mask on the cfd grid (union)."""
    from plugins.solvers.cfd_coupling.core.obstacles import (
        combine_masks,
        mask_from_mesh,
    )

    mask = combine_masks(
        *[
            mask_from_mesh(
                m, tuple(cfd.o.domain), tuple(cfd.o.cells), transform=placement
            )
            for m in meshes
        ]
    )
    if mask is None or not mask.any():
        raise ValueError(
            "obstacle rasterized to an empty mask: check units (metres), "
            "placement and that the geometry intersects the CFD domain"
        )
    return mask


class _ObstacleMixin:
    """Obstacle registry shared by CFDSolver and CoupledCFDSolver.

    Each entry keeps its own mask so per-obstacle force breakdown is
    available; the core always holds the union mask. Tracked obstacles
    re-rasterize from their pose provider every substep (moving masks).
    """

    _cfd: object  # duck-typed CFD3D / QDCFD3D
    _obstacles: dict

    def _init_obstacles(self) -> None:
        self._obstacles = {}

    def add_obstacle(
        self,
        source,
        position: tuple[float, float, float] | None = None,
        rotation_euler: tuple[float, float, float] | None = None,
        scale: float | tuple[float, float, float] | None = None,
        name: str | None = None,
        track=None,
    ) -> np.ndarray:
        """Add an immersed no-slip obstacle to the 3D CFD domain.

        ``source`` is a mesh file path (STL/OBJ/GLB, e.g. exported from
        CATIA), a STEP file (``.stp``/``.step``, incl. multi-solid CATIA
        assemblies - one part per solid), a ``trimesh.Trimesh``, a list of
        those (assembly), or a built genesis RigidEntity. ``position`` /
        ``rotation_euler`` (degrees, xyz) / ``scale`` place the source into
        the CFD world frame (metres) on top of any transform the source
        itself carries.

        ``track`` is an optional pose provider callable for moving
        obstacles, invoked every substep; it returns None (keep pose), a
        (3,) position, or a (4, 4) matrix. Tracked obstacles must use
        file/mesh sources (model coordinates); entity snapshots are static.

        Returns the mask this obstacle contributed; masks of successive
        calls are unioned. Obstacles must stay clear of the x = 0 / x = Lx
        faces. One-way coupling only: the fluid feels the obstacle, not
        vice versa.
        """
        name = name or f"obstacle_{len(self._obstacles)}"
        if name in self._obstacles:
            raise ValueError(f"obstacle name already registered: {name!r}")
        meshes, world_snapshot = _resolve_sources(source)
        if track is not None and world_snapshot:
            raise ValueError(
                "tracked (moving) obstacles need file/mesh sources in model "
                "coordinates; a genesis entity snapshot is static - export "
                "the entity mesh and pass the file with a pose provider"
            )
        placement = (
            None
            if world_snapshot and position is None
            else _placement_matrix(position, rotation_euler, scale)
        )
        entry = {
            "meshes": meshes,
            "placement": placement,
            "track": track,
            "mask": _rasterize_meshes(self._cfd, meshes, placement),
        }
        self._obstacles[name] = entry
        self._rebuild_solid_union()
        return entry["mask"]

    def remove_obstacle(self, name: str) -> None:
        """Remove a registered obstacle and rebuild the union mask."""
        if name not in self._obstacles:
            raise KeyError(f"unknown obstacle: {name!r}")
        del self._obstacles[name]
        self._rebuild_solid_union()

    @property
    def obstacles(self) -> dict:
        """Registered obstacle masks by name (copies)."""
        return {k: v["mask"].copy() for k, v in self._obstacles.items()}

    def obstacle_forces(self) -> dict:
        """Per-obstacle fluid forces (Newton) plus a 'total' entry.

        Each value is the dict returned by the core's ``obstacle_forces``:
        pressure / viscous / total length-3 arrays in world axes.
        """
        out = {
            name: self._cfd.obstacle_forces(entry["mask"])
            for name, entry in self._obstacles.items()
        }
        total_mask = self._cfd.solid_mask
        out["total"] = (
            self._cfd.obstacle_forces(total_mask) if total_mask is not None else None
        )
        return out

    def _rebuild_solid_union(self) -> None:
        from plugins.solvers.cfd_coupling.core.obstacles import combine_masks

        self._cfd.set_solid_mask(
            combine_masks(*[e["mask"] for e in self._obstacles.values()])
        )

    def _sync_tracked_obstacles(self) -> None:
        """Re-rasterize tracked obstacles whose pose changed; rebuild union."""
        changed = False
        for entry in self._obstacles.values():
            if entry["track"] is None:
                continue
            new_placement = _coerce_placement(entry["track"]())
            old = entry["placement"]
            if (new_placement is None) != (old is None) or (
                new_placement is not None
                and not np.allclose(new_placement, old, atol=1e-12)
            ):
                entry["placement"] = new_placement
                entry["mask"] = _rasterize_meshes(
                    self._cfd, entry["meshes"], new_placement
                )
                changed = True
        if changed:
            self._rebuild_solid_union()


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
class CoupledCFDSolver(_ObstacleMixin, _PluginSolverBase, Solver):
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
        self._init_obstacles()

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
        self._sync_tracked_obstacles()
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


class CFDSolver(_ObstacleMixin, _PluginSolverBase, Solver):
    """Plugin solver wrapping the standalone 3D projection CFD core."""

    def __init__(self, scene: "Scene", sim: "Simulator", options: CFDSolverOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._cfd = _make_cfd_core(options.cfd, options.backend)
        self._cfd.set_inlet(options.inlet_velocity)
        self._substep_dt = 0.0
        self._init_obstacles()

    @property
    def is_active(self) -> bool:
        return True

    @property
    def cfd(self):
        return self._cfd

    def build(self) -> None:
        super().build()
        self._substep_dt = self._sim.substep_dt
        self._entities.append(_MarkerEntity())

    def substep_pre_coupling(self, f: int) -> None:
        self._sync_tracked_obstacles()
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
