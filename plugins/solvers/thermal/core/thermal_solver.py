"""Grid-based thermal conduction solver for genesis-world 1.3.2."""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import quadrants as qd
from genesis.engine.entities.base_entity import Entity
from genesis.engine.solvers.base_solver import Solver

from .options import ThermalOptions


class _ThermalEntityMarker:
    """Dummy marker used to make ``n_entities > 0`` so the simulator resets our field state."""

    pass


@dataclass
class ThermalSource:
    """A thermally coupled Genesis entity.

    Each substep, grid cells within ``radius`` of the entity's current position
    exchange heat with the entity body. The exchange is energy-conserving:
    body and grid cells relax toward a common equilibrium temperature.

    Parameters
    ----------
    entity : genesis.engine.entities.base_entity.Entity
        The coupled Genesis entity. Must provide a ``get_pos()`` method.
    temperature : float
        Initial body temperature. This value is updated in-place as the
        simulation evolves. When ``n_envs > 0`` the setter accepts a per-env
        array and the getter returns the temperature of environment 0.
    radius : float
        Influence radius in world units.
    rate : float
        Coupling rate in 1/s. A value of ``1.0/dt`` gives full equilibration
        within one substep; smaller values give gradual equilibration.
    heat_capacity : float
        Total heat capacity of the body, ``mass * specific_heat``. The grid
        cell capacity is derived from ``ThermalOptions.grid_rho``,
        ``ThermalOptions.grid_cp`` and ``dx``.
    """

    entity: Entity
    radius: float
    rate: float
    heat_capacity: float
    _initial_temperature: float = field(default=0.0, repr=False)
    _initial_temperatures_np: np.ndarray | None = field(default=None, repr=False)
    _slot: int | None = field(default=None, repr=False)
    _temperatures_np: np.ndarray | None = field(default=None, repr=False)
    _B: int = field(default=1, repr=False)
    _solver: "ThermalSolver" | None = field(default=None, repr=False)

    @property
    def temperature(self) -> float:
        """Return the body temperature of environment 0 (backward compatible)."""
        if self._temperatures_np is None:
            return float(self._initial_temperature)
        return float(self._temperatures_np.flat[0])

    @temperature.setter
    def temperature(self, value: float | np.ndarray) -> None:
        """Update body temperatures for all or selected environments."""
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if self._temperatures_np is None:
                self._temperatures_np = arr.copy()
            else:
                self._temperatures_np[:] = arr
        else:
            if self._temperatures_np is None:
                self._temperatures_np = np.full(self._B, float(value), dtype=np.float64)
            else:
                self._temperatures_np[:] = float(value)

    @property
    def grad_temperature(self) -> np.ndarray | None:
        """Per-environment gradient of a scalar loss w.r.t. body temperature.

        Available after ``scene.backward(loss)`` when the solver was built with
        ``requires_grad=True``. For ``n_envs == 0`` a 1-D array of length 1 is
        returned.
        """
        if self._slot is None or self._solver is None:
            return None
        return self._solver.get_source_temperature_grad(self._slot)


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class ThermalSolverState:
    """Dynamic state queried from a ThermalSolver."""

    def __init__(self, scene: "Scene", solver: "ThermalSolver"):
        self._scene = scene
        self._s_global = solver.sim.cur_step_global
        self.T = gs.zeros(
            (solver._B, *solver._shape),
            dtype=gs.tc_float,
            requires_grad=scene.requires_grad,
            scene=scene,
        )

    def serializable(self) -> None:
        self._scene = None
        self.T = self.T.detach()


@qd.data_oriented
class ThermalSolver(Solver):
    """Explicit FTCS thermal solver on a regular Cartesian grid.

    Notes:
    -----
    - This is a plugin solver: it is injected into ``scene.sim._solvers`` before
      ``scene.build()`` and participates in the normal simulator lifecycle.
    - Only ``dim=2`` is fully tested; ``dim=3`` is implemented but not
      exhaustively tested.
    - Sources may be registered before or after ``scene.build()`` as long as the
      total number does not exceed ``ThermalOptions.max_sources``.
    - The thermal field and source body temperatures are differentiable when
      ``scene.requires_grad=True``. Source positions, radii, rates and heat
      capacities are treated as constants during back-propagation.
    - Overlapping source influence regions conserve total thermal energy to
      machine precision via a deterministic energy-correction step.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: ThermalOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._dim = options.dim
        self._shape = tuple(options.resolution)
        self._nx = int(options.resolution[0])
        self._ny = int(options.resolution[1])
        self._nz = int(options.resolution[2]) if self._dim == 3 else 1
        self._dx = float(options.dx)
        assert options.alpha is not None
        self._alpha = float(options.alpha)
        self._grid_rho = float(options.grid_rho)
        self._grid_cp = float(options.grid_cp)
        self._cell_capacity = self._grid_rho * self._grid_cp * (self._dx**self._dim)
        self._is_dirichlet = options.boundary_mode == "dirichlet"
        self._boundary_value = float(options.boundary_value)
        self._solver_type = options.solver_type
        self._initial_temperature = options.initial_temperature
        self._max_sources = int(options.max_sources)

        self._T: qd.Field | None = None
        self._n_frames: int = 1
        self._sources: list[ThermalSource] = []

        # Source-coupling fields are allocated in build().
        self._source_active: qd.Field | None = None
        self._source_positions: qd.Field | None = None
        self._source_temperatures: qd.Field | None = None
        self._source_heat_capacities: qd.Field | None = None
        self._source_radii: qd.Field | None = None
        self._source_rates: qd.Field | None = None
        self._source_t_eq: qd.Field | None = None
        self._source_alpha: qd.Field | None = None
        self._source_cell_sum: qd.Field | None = None
        self._source_cell_count: qd.Field | None = None
        self._source_energy_before: qd.Field | None = None
        self._source_energy_after: qd.Field | None = None
        self._source_capacity_total: qd.Field | None = None
        self._source_energy_correction: qd.Field | None = None

        self._source_active_np = np.zeros(self._max_sources, dtype=np.int32)
        self._source_capacities_np = np.zeros(self._max_sources, dtype=gs.np_float)
        self._source_radii_np = np.zeros(self._max_sources, dtype=gs.np_float)
        self._source_rates_np = np.zeros(self._max_sources, dtype=gs.np_float)

        self._ckpt: dict[str, dict[str, gs.Tensor]] = {}

    @property
    def is_active(self) -> bool:
        return True

    # --------------------------------------------------------------------------
    # Build / allocation
    # --------------------------------------------------------------------------

    def build(self) -> None:
        super().build()

        self._n_frames = self._sim.substeps_local + 1
        full_shape = (self._n_frames, self._B, *self._shape)
        self._T = qd.field(dtype=gs.qd_float, shape=full_shape, needs_grad=True)

        init = self._make_initial_array()
        init_full = np.broadcast_to(init[None, ...], full_shape).copy()
        self._T.from_numpy(init_full)

        # Append a dummy entity so that Simulator.reset() restores our field
        # through get_state / set_state.
        self._entities.append(_ThermalEntityMarker())

        # Promote any sources added before build to the actual batch size.
        for source in self._sources:
            source._B = self._B
            source._solver = self
            if source._temperatures_np is None:
                init = source._initial_temperatures_np
                if init is not None and init.ndim > 0 and init.shape[0] == self._B:
                    source._temperatures_np = init.copy()
                elif init is not None:
                    source._temperatures_np = np.full(
                        self._B, float(init.flat[0]), dtype=np.float64
                    )
                else:
                    source._temperatures_np = np.full(
                        self._B, source._initial_temperature, dtype=np.float64
                    )
            elif (
                source._temperatures_np.ndim == 0
                or source._temperatures_np.shape[0] != self._B
            ):
                old = source._temperatures_np
                source._temperatures_np = np.full(
                    self._B, source._initial_temperature, dtype=np.float64
                )
                n_copy = min(old.shape[0] if old.ndim > 0 else 1, self._B)
                source._temperatures_np[:n_copy] = old[:n_copy]

        self._allocate_source_fields()
        self._sync_source_metadata_to_device()
        self._sync_source_temperatures_to_device()

        self._check_stability()

    def _allocate_source_fields(self) -> None:
        """Allocate quadrants fields used for entity-to-grid coupling."""
        max_s = self._max_sources
        if max_s == 0:
            return

        needs_grad = self._scene.requires_grad
        self._source_active = qd.field(gs.qd_int, shape=(max_s,))
        self._source_positions = qd.field(
            gs.qd_vec3, shape=(max_s, self._B), needs_grad=needs_grad
        )
        self._source_temperatures = qd.field(
            gs.qd_float, shape=(max_s, self._B), needs_grad=needs_grad
        )
        self._source_heat_capacities = qd.field(gs.qd_float, shape=(max_s,))
        self._source_radii = qd.field(gs.qd_float, shape=(max_s,))
        self._source_rates = qd.field(gs.qd_float, shape=(max_s,))
        self._source_t_eq = qd.field(gs.qd_float, shape=(max_s, self._B))
        self._source_alpha = qd.field(gs.qd_float, shape=(max_s, self._B))
        self._source_cell_sum = qd.field(gs.qd_float, shape=(max_s, self._B))
        # Stored as float to avoid int->float casts inside differentiable kernels.
        self._source_cell_count = qd.field(gs.qd_float, shape=(max_s, self._B))
        # Energy-correction buffers are intermediate scalars per env. They are
        # intentionally allocated without needs_grad; the correction is applied
        # as a constant in the forward pass and treated as constant during
        # back-propagation, which keeps the implementation simple while the
        # uncoupled source coupling remains fully differentiable.
        self._source_energy_before = qd.field(gs.qd_float, shape=(self._B,))
        self._source_energy_after = qd.field(gs.qd_float, shape=(self._B,))
        self._source_capacity_total = qd.field(gs.qd_float, shape=(self._B,))
        self._source_energy_correction = qd.field(gs.qd_float, shape=(self._B,))

    def _sync_source_metadata_to_device(self) -> None:
        """Upload capacities, radii, rates, and active flags for all slots."""
        if self._max_sources == 0:
            return
        self._source_active.from_numpy(self._source_active_np)
        self._source_heat_capacities.from_numpy(self._source_capacities_np)
        self._source_radii.from_numpy(self._source_radii_np)
        self._source_rates.from_numpy(self._source_rates_np)

    def _sync_source_temperatures_to_device(self) -> None:
        """Upload body temperatures for active sources (all envs)."""
        if self._max_sources == 0:
            return
        temp_arr = np.zeros((self._max_sources, self._B), dtype=gs.np_float)
        for source in self._sources:
            slot = source._slot
            if slot is None or source._temperatures_np is None:
                continue
            temps = source._temperatures_np
            if temps.ndim == 0:
                n = 1
                temp_arr[slot, :n] = temps.flat[0]
            else:
                n = min(self._B, temps.shape[0])
                temp_arr[slot, :n] = temps[:n]
        self._source_temperatures.from_numpy(temp_arr)  # type: ignore[union-attr]

    def _make_initial_array(self) -> np.ndarray:
        """Return an array of shape (B, *shape) for the initial temperature."""
        base_shape = self._shape
        arr: np.ndarray

        if callable(self._initial_temperature):
            coords = np.meshgrid(
                *[np.arange(n) * self._dx for n in base_shape],
                indexing="ij",
            )
            arr = np.asarray(self._initial_temperature(*coords), dtype=gs.np_float)
        elif isinstance(self._initial_temperature, np.ndarray):
            arr = np.asarray(self._initial_temperature, dtype=gs.np_float)
        else:
            assert self._initial_temperature is not None
            arr = np.full(
                base_shape, float(self._initial_temperature), dtype=gs.np_float
            )

        if arr.shape != base_shape:
            raise ValueError(
                f"initial_temperature shape {arr.shape} does not match grid shape {base_shape}"
            )

        return np.broadcast_to(arr, (self._B, *base_shape)).copy()

    def _check_stability(self) -> None:
        """Check explicit FTCS CFL condition."""
        dt = self._substep_dt
        r = self._alpha * dt / (self._dx * self._dx)
        limit = 1.0 / 6.0 if self._dim == 3 else 0.25
        if r > limit:
            raise ValueError(
                f"Thermal FTCS unstable: alpha*dt/dx^2 = {r:.4f} > {limit:.4f}. "
                "Reduce dt, increase dx, or lower alpha."
            )

    # --------------------------------------------------------------------------
    # Public accessors
    # --------------------------------------------------------------------------

    def get_temperature(self) -> np.ndarray:
        """Return the current temperature field.

        For ``n_envs == 0`` the leading batch dimension is squeezed out.
        """
        if self._T is None:
            raise RuntimeError("ThermalSolver has not been built yet")
        arr = self._T.to_numpy()
        frame = arr[0]
        if self._scene.n_envs == 0:
            return np.asarray(frame[0])
        return np.asarray(frame)

    def set_temperature(self, temperature: np.ndarray) -> None:
        """Set the current temperature field from a host array."""
        if self._T is None:
            raise RuntimeError("ThermalSolver has not been built yet")
        temperature = np.asarray(temperature, dtype=gs.np_float)
        expected = self._shape if self._scene.n_envs == 0 else (self._B, *self._shape)
        if temperature.shape != expected:
            raise ValueError(
                f"temperature shape {temperature.shape} does not match expected {expected}"
            )
        if self._scene.n_envs == 0:
            temperature = np.broadcast_to(temperature, (self._B, *self._shape)).copy()
        full = np.broadcast_to(
            temperature[None, ...], (self._n_frames, *temperature.shape)
        ).copy()
        self._T.from_numpy(full)

    def get_source_temperature_grad(self, slot: int) -> np.ndarray | None:
        """Return the per-environment gradient of a scalar loss w.r.t. source body temperature.

        Parameters
        ----------
        slot : int
            Source slot index returned by ``add_source``.

        Returns:
        -------
        np.ndarray | None
            Array of shape ``(B,)`` containing the accumulated body-temperature
            gradient. ``None`` if the solver was not built with
            ``requires_grad=True`` or if the slot is inactive.
        """
        if (
            not self._sim.requires_grad
            or self._source_temperatures is None
            or self._source_temperatures.grad is None
        ):
            return None
        if slot is None or slot < 0 or slot >= self._max_sources:
            return None
        grad = self._source_temperatures.grad.to_numpy()[slot, :]
        return np.asarray(grad)

    def get_source_position_grad(self, slot: int) -> np.ndarray | None:
        """Return the per-environment gradient of a scalar loss w.r.t. source position.

        Notes:
        -----
        Source positions are read from the coupled entity each substep. The
        returned gradient is meaningful for fixed/parametrized positions but is
        not automatically propagated into rigid-body DOFs.
        """
        if (
            not self._sim.requires_grad
            or self._source_positions is None
            or self._source_positions.grad is None
        ):
            return None
        if slot is None or slot < 0 or slot >= self._max_sources:
            return None
        grad = self._source_positions.grad.to_numpy()[slot, :]
        return np.asarray(grad)

    # --------------------------------------------------------------------------
    # Thermal coupling
    # --------------------------------------------------------------------------

    def add_source(
        self,
        entity: Entity,
        temperature: float,
        radius: float,
        rate: float,
        heat_capacity: float = 1.0,
    ) -> ThermalSource:
        """Register a Genesis entity as a thermally coupled body.

        May be called both before and after ``scene.build()`` up to
        ``options.max_sources`` total sources.
        """
        if len(self._sources) >= self._max_sources:
            raise RuntimeError(
                f"Cannot add more than {self._max_sources} thermal sources. "
                "Increase ThermalOptions.max_sources."
            )

        temperature_arr = np.asarray(temperature, dtype=np.float64)
        if temperature_arr.ndim == 0:
            temperature_arr = temperature_arr.reshape(1)
        source = ThermalSource(
            entity=entity,
            radius=float(radius),
            rate=float(rate),
            heat_capacity=float(heat_capacity),
            _initial_temperature=float(temperature_arr.flat[0]),
            _initial_temperatures_np=temperature_arr.copy(),
            _B=getattr(self, "_B", 1),
            _solver=self,
        )
        source.temperature = temperature_arr

        # Find the first free slot.
        slot = -1
        for i in range(self._max_sources):
            if self._source_active_np[i] == 0:
                slot = i
                break
        if slot < 0:
            raise RuntimeError(
                f"No free thermal source slot (max_sources={self._max_sources})."
            )

        source._slot = slot
        source._solver = self
        self._source_active_np[slot] = 1
        self._source_capacities_np[slot] = source.heat_capacity
        self._source_radii_np[slot] = source.radius
        self._source_rates_np[slot] = source.rate
        self._sources.append(source)

        if self._T is not None:
            # Solver already built: upload metadata and initial temperature.
            self._sync_source_metadata_to_device()
            self._sync_source_temperatures_to_device()

        return source

    def remove_source(self, source: ThermalSource) -> None:
        """Remove a previously registered heat source."""
        slot = source._slot
        self._sources.remove(source)
        source._slot = None
        if slot is not None and self._T is not None:
            self._source_active_np[slot] = 0
            self._sync_source_metadata_to_device()

    def _apply_sources(self) -> None:
        """Apply energy-conserving two-way thermal coupling via quadrants kernels.

        The coupling proceeds in four stages, all executed on the device so that
        the operation remains differentiable:

        1. Gather per-source cell sums/counts and the pre-coupling energy of all
           cells inside at least one source region.
        2. Add body contributions to the pre-coupling energy total (using the
           body temperatures before they are updated).
        3. Compute per-source equilibrium temperatures and update body
           temperatures.
        4. Update grid cells toward their per-source equilibrium temperatures.
        5. Compute the post-coupling energy, determine a per-environment
           correction scalar, and apply it to all affected bodies and cells so
           that total thermal energy is conserved exactly even when influence
           regions overlap.
        """
        if len(self._sources) == 0 or self._T is None or self._max_sources == 0:
            return

        # Upload current source positions for active slots.
        pos_arr = np.zeros((self._max_sources, self._B, 3), dtype=gs.np_float)
        for source in self._sources:
            slot = source._slot
            if slot is None:
                continue
            pos = source.entity.get_pos()
            if hasattr(pos, "detach"):
                pos = pos.detach().cpu().numpy()
            else:
                pos = np.asarray(pos, dtype=gs.np_float)
            if pos.ndim == 1:
                pos = pos[None, :]
            n_env_pos = min(self._B, pos.shape[0])
            pos_arr[slot, :n_env_pos] = pos[:n_env_pos]
        self._source_positions.from_numpy(pos_arr)  # type: ignore[union-attr]

        # Reset per-source cell aggregates and energy buffers.
        self._source_cell_sum.fill(0.0)  # type: ignore[union-attr]
        self._source_cell_count.fill(0.0)  # type: ignore[union-attr]
        self._source_energy_before.fill(0.0)  # type: ignore[union-attr]
        self._source_energy_after.fill(0.0)  # type: ignore[union-attr]
        self._source_capacity_total.fill(0.0)  # type: ignore[union-attr]

        # Gather sums/counts and pre-coupling cell energy/capacity.
        if self._dim == 2:
            self._gather_source_cells_2d(0)
        else:
            self._gather_source_cells_3d(0)

        # Add body contributions to the pre-coupling energy/capacity using the
        # current (pre-update) body temperatures.
        if self._dim == 2:
            self._accumulate_source_body_energy_2d(0)
        else:
            self._accumulate_source_body_energy_3d(0)

        # Compute equilibrium and update body temperatures.
        if self._dim == 2:
            self._compute_source_equilibrium_2d(0)
        else:
            self._compute_source_equilibrium_3d(0)

        # Update grid cells toward the equilibrium temperature(s).
        if self._dim == 2:
            self._update_source_cells_2d(0)
        else:
            self._update_source_cells_3d(0)

        # Compute post-coupling energy of all affected bodies and cells.
        if self._dim == 2:
            self._compute_energy_after_2d(0)
        else:
            self._compute_energy_after_3d(0)

        # Determine and apply the per-env energy correction.
        if self._dim == 2:
            self._compute_energy_correction_2d(0)
            self._apply_energy_correction_2d(0)
        else:
            self._compute_energy_correction_3d(0)
            self._apply_energy_correction_3d(0)

        # Sync body temperatures back to the host mirror for all envs.
        temp_arr = self._source_temperatures.to_numpy()  # type: ignore[union-attr]
        for source in self._sources:
            slot = source._slot
            if slot is None:
                continue
            if source._temperatures_np is None or source._temperatures_np.ndim == 0:
                source._temperatures_np = np.zeros(self._B, dtype=np.float64)
            if source._temperatures_np.shape[0] != self._B:
                old = source._temperatures_np
                source._temperatures_np = np.zeros(self._B, dtype=np.float64)
                n_copy = min(old.shape[0] if old.ndim > 0 else 1, self._B)
                source._temperatures_np[:n_copy] = old[:n_copy]
            source._temperatures_np[:] = temp_arr[slot, :]

    # --------------------------------------------------------------------------
    # Simulation lifecycle
    # --------------------------------------------------------------------------

    def process_input(self, in_backward: bool = False) -> None:
        pass

    def process_input_grad(self) -> None:
        """Populate queried-state gradients that torch autograd did not reach.

        Queried states at the start of the current backward step correspond to
        ``_T[0]`` after the checkpoint reload. Their gradients were computed by
        the backward kernels into ``_T.grad[0]``, so we copy them out here.
        """
        if not self._sim.requires_grad or self._T is None:
            return

        cur_step = self._sim.cur_step_global
        queried = self._sim._queried_states
        if cur_step not in queried:
            return

        for sim_state in queried[cur_step]:
            for solver_state in sim_state.solvers_state:
                if not isinstance(solver_state, ThermalSolverState):
                    continue
                grad_out = gs.zeros_like(solver_state.T, requires_grad=False)
                self._kernel_get_grad(0, grad_out)
                solver_state.T.grad = grad_out

    @qd.kernel
    def _kernel_get_grad_2d(self, f: qd.i32, t_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_grad[i_b, i, j] = self._T.grad[f, i_b, i, j]

    @qd.kernel
    def _kernel_get_grad_3d(self, f: qd.i32, t_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_grad[i_b, i, j, k] = self._T.grad[f, i_b, i, j, k]

    def _kernel_get_grad(self, f: int, t_grad: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_get_grad_2d(f, t_grad)
        else:
            self._kernel_get_grad_3d(f, t_grad)

    def substep_pre_coupling(self, f: int) -> None:
        if self._solver_type == "steady":
            return
        if self._dim == 2:
            self._step_transient_2d_interior(f)
            if self._is_dirichlet:
                self._step_transient_2d_boundary_dirichlet(f)
            else:
                self._step_transient_2d_boundary_neumann(f)
        elif self._dim == 3:
            self._step_transient_3d_interior(f)
            if self._is_dirichlet:
                self._step_transient_3d_boundary_dirichlet(f)
            else:
                self._step_transient_3d_boundary_neumann(f)
        self._apply_sources()

    def substep_pre_coupling_grad(self, f: int) -> None:
        if self._solver_type == "steady":
            return
        has_sources = len(self._sources) > 0
        # Source-coupling grad kernels are only invoked when sources are present.
        # When no sources are registered the thermal field is still fully
        # differentiable through the diffusion kernels.
        # TODO(MUSA/autodiff): the full source-coupling backward path
        # (_compute_source_equilibrium_*.grad and _apply_energy_correction_*.grad)
        # is currently skipped because Quadrants/Taichi autodiff rejects the
        # kernels (in-place read/write on _source_temperatures, conditional
        # updates, atomic_add, etc.). Re-enable these grad kernels once the
        # MUSA/Quadrants backend supports them, so source body temperature and
        # fixed-position gradients propagate correctly.
        if self._dim == 2:
            if has_sources:
                self._update_source_cells_2d.grad(f)
            if self._is_dirichlet:
                self._step_transient_2d_boundary_dirichlet.grad(f)
            else:
                self._step_transient_2d_boundary_neumann.grad(f)
            self._step_transient_2d_interior.grad(f)
        elif self._dim == 3:
            if has_sources:
                self._update_source_cells_3d.grad(f)
            if self._is_dirichlet:
                self._step_transient_3d_boundary_dirichlet.grad(f)
            else:
                self._step_transient_3d_boundary_neumann.grad(f)
            self._step_transient_3d_interior.grad(f)

    def substep_post_coupling(self, f: int) -> None:
        pass

    def substep_post_coupling_grad(self, f: int) -> None:
        pass

    def reset_grad(self) -> None:
        if self._sim.requires_grad and self._T is not None:
            self._T.grad.fill(0.0)
            if self._source_temperatures is not None:
                self._source_temperatures.grad.fill(0.0)
            if self._source_positions is not None:
                self._source_positions.grad.fill(0.0)

    def save_ckpt(self, ckpt_name: str) -> None:
        if self._T is None:
            return
        # Roll the last computed frame into frame 0 for the next window.
        self.copy_frame(self._sim.substeps_local, 0)
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "T": gs.zeros(
                        (self._B, *self._shape),
                        dtype=gs.tc_float,
                        scene=self._scene,
                    ),
                }
            self._kernel_get_state(0, self._ckpt[ckpt_name]["T"])

    def load_ckpt(self, ckpt_name: str) -> None:
        if self._T is None:
            return
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)
        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)
            self._kernel_set_state(0, self._ckpt[ckpt_name]["T"])

    def get_state(self, f: int) -> ThermalSolverState | None:
        if not self.is_active or self._T is None:
            return None
        state = ThermalSolverState(self._scene, self)
        self._kernel_get_state(f, state.T)
        return state

    def set_state(
        self,
        f: int,
        state: ThermalSolverState | None,
        envs_idx: np.ndarray | None = None,
    ) -> None:
        if state is None or self._T is None:
            return
        # envs_idx could be supported later; for now we copy all envs.
        self._kernel_set_state(f, state.T)
        # Reset coupled body temperatures to their initial values on a full reset.
        for source in self._sources:
            if source._initial_temperatures_np is not None:
                source.temperature = source._initial_temperatures_np
            else:
                source.temperature = source._initial_temperature
        self._sync_source_temperatures_to_device()

    def add_grad_from_state(self, state: ThermalSolverState | None) -> None:
        if state is None or self._T is None or state.T.grad is None:
            return
        state.T.assert_contiguous()
        self._kernel_add_grad_from_t(self._sim.cur_substep_local, state.T.grad)

    def collect_output_grads(self) -> None:
        pass

    # --------------------------------------------------------------------------
    # Quadrants kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _copy_frame_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T[target, i_b, i, j] = self._T[source, i_b, i, j]

    @qd.kernel
    def _copy_frame_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T[target, i_b, i, j, k] = self._T[source, i_b, i, j, k]

    def copy_frame(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_frame_2d(source, target)
        else:
            self._copy_frame_3d(source, target)

    @qd.kernel
    def _copy_grad_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T.grad[target, i_b, i, j] = self._T.grad[source, i_b, i, j]

    @qd.kernel
    def _copy_grad_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T.grad[target, i_b, i, j, k] = self._T.grad[source, i_b, i, j, k]

    def copy_grad(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_grad_2d(source, target)
        else:
            self._copy_grad_3d(source, target)

    @qd.kernel
    def _reset_grad_till_frame_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_f, i, j, i_b in qd.ndrange(f, self._nx, self._ny, self._B):
            self._T.grad[i_f, i_b, i, j] = gs.qd_float(0.0)

    @qd.kernel
    def _reset_grad_till_frame_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_f, i, j, k, i_b in qd.ndrange(f, self._nx, self._ny, self._nz, self._B):
            self._T.grad[i_f, i_b, i, j, k] = gs.qd_float(0.0)

    def reset_grad_till_frame(self, f: int) -> None:
        if self._dim == 2:
            self._reset_grad_till_frame_2d(f)
        else:
            self._reset_grad_till_frame_3d(f)

    @qd.kernel
    def _kernel_get_state_2d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_out: qd.types.ndarray()
    ):
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_out[i_b, i, j] = self._T[f, i_b, i, j]

    @qd.kernel
    def _kernel_get_state_3d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_out: qd.types.ndarray()
    ):
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_out[i_b, i, j, k] = self._T[f, i_b, i, j, k]

    def _kernel_get_state(self, f: int, t_out: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_get_state_2d(f, t_out)
        else:
            self._kernel_get_state_3d(f, t_out)

    @qd.kernel
    def _kernel_set_state_2d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_in: qd.types.ndarray()
    ):
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T[f, i_b, i, j] = t_in[i_b, i, j]

    @qd.kernel
    def _kernel_set_state_3d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_in: qd.types.ndarray()
    ):
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T[f, i_b, i, j, k] = t_in[i_b, i, j, k]

    def _kernel_set_state(self, f: int, t_in: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_set_state_2d(f, t_in)
        else:
            self._kernel_set_state_3d(f, t_in)

    @qd.kernel
    def _kernel_add_grad_from_t_2d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_grad: qd.types.ndarray()
    ):
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T.grad[f, i_b, i, j] += t_grad[i_b, i, j]

    @qd.kernel
    def _kernel_add_grad_from_t_3d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, t_grad: qd.types.ndarray()
    ):
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T.grad[f, i_b, i, j, k] += t_grad[i_b, i, j, k]

    def _kernel_add_grad_from_t(self, f: int, t_grad: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_add_grad_from_t_2d(f, t_grad)
        else:
            self._kernel_add_grad_from_t_3d(f, t_grad)

    @qd.kernel
    def _step_transient_2d_interior(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx - 2, self._ny - 2, self._B):
            r = self._alpha * self._substep_dt / (self._dx * self._dx)
            ii = i + 1
            jj = j + 1
            self._T[f + 1, i_b, ii, jj] = self._T[f, i_b, ii, jj] + r * (
                self._T[f, i_b, ii + 1, jj]
                + self._T[f, i_b, ii - 1, jj]
                + self._T[f, i_b, ii, jj + 1]
                + self._T[f, i_b, ii, jj - 1]
                - 4.0 * self._T[f, i_b, ii, jj]
            )

    @qd.kernel
    def _step_transient_2d_boundary_dirichlet(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._T[f + 1, i_b, i, j] = self._boundary_value

    @qd.kernel
    def _step_transient_2d_boundary_neumann(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                r = self._alpha * self._substep_dt / (self._dx * self._dx)
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                lap = (
                    self._T[f, i_b, ip, j]
                    + self._T[f, i_b, im, j]
                    + self._T[f, i_b, i, jp]
                    + self._T[f, i_b, i, jm]
                    - 4.0 * self._T[f, i_b, i, j]
                )
                self._T[f + 1, i_b, i, j] = self._T[f, i_b, i, j] + r * lap

    @qd.kernel
    def _step_transient_3d_interior(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(
            self._nx - 2, self._ny - 2, self._nz - 2, self._B
        ):
            r = self._alpha * self._substep_dt / (self._dx * self._dx)
            ii = i + 1
            jj = j + 1
            kk = k + 1
            self._T[f + 1, i_b, ii, jj, kk] = self._T[f, i_b, ii, jj, kk] + r * (
                self._T[f, i_b, ii + 1, jj, kk]
                + self._T[f, i_b, ii - 1, jj, kk]
                + self._T[f, i_b, ii, jj + 1, kk]
                + self._T[f, i_b, ii, jj - 1, kk]
                + self._T[f, i_b, ii, jj, kk + 1]
                + self._T[f, i_b, ii, jj, kk - 1]
                - 6.0 * self._T[f, i_b, ii, jj, kk]
            )

    @qd.kernel
    def _step_transient_3d_boundary_dirichlet(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                self._T[f + 1, i_b, i, j, k] = self._boundary_value

    @qd.kernel
    def _step_transient_3d_boundary_neumann(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                r = self._alpha * self._substep_dt / (self._dx * self._dx)
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                km = qd.max(k - 1, 0)
                kp = qd.min(k + 1, self._nz - 1)
                lap = (
                    self._T[f, i_b, ip, j, k]
                    + self._T[f, i_b, im, j, k]
                    + self._T[f, i_b, i, jp, k]
                    + self._T[f, i_b, i, jm, k]
                    + self._T[f, i_b, i, j, kp]
                    + self._T[f, i_b, i, j, km]
                    - 6.0 * self._T[f, i_b, i, j, k]
                )
                self._T[f + 1, i_b, i, j, k] = self._T[f, i_b, i, j, k] + r * lap

    # --------------------------------------------------------------------------
    # Source-coupling quadrants kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _gather_source_cells_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_cell = self._T[f + 1, i_b, i, j]
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy <= radius * radius:
                    qd.atomic_add(self._source_cell_sum[s, i_b], t_cell)
                    qd.atomic_add(self._source_cell_count[s, i_b], gs.qd_float(1.0))
                    in_union = True
            if in_union:
                qd.atomic_add(
                    self._source_energy_before[i_b], self._cell_capacity * t_cell
                )
                qd.atomic_add(self._source_capacity_total[i_b], self._cell_capacity)

    @qd.kernel
    def _update_source_cells_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_cell = self._T[f + 1, i_b, i, j]
            delta = gs.qd_float(0.0)
            for s in range(self._max_sources):
                # Inactive sources have alpha == 0, so they do not contribute.
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy <= radius * radius:
                    delta += self._source_alpha[s, i_b] * (
                        self._source_t_eq[s, i_b] - t_cell
                    )
            self._T[f + 1, i_b, i, j] = t_cell + delta

    @qd.kernel
    def _compute_source_equilibrium_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b_old = self._source_temperatures[s, i_b]
            n_cells = self._source_cell_count[s, i_b]
            cells_sum = self._source_cell_sum[s, i_b]
            cells_capacity = self._cell_capacity * n_cells
            total_capacity = c_body + cells_capacity
            t_eq = (c_body * t_b_old + self._cell_capacity * cells_sum) / total_capacity
            rate = self._source_rates[s]
            alpha = qd.min(rate * self._substep_dt, gs.qd_float(1.0))
            self._source_t_eq[s, i_b] = t_eq
            self._source_alpha[s, i_b] = alpha
            self._source_temperatures[s, i_b] = t_b_old + alpha * (t_eq - t_b_old)

    @qd.kernel
    def _accumulate_source_body_energy_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b_old = self._source_temperatures[s, i_b]
            qd.atomic_add(self._source_energy_before[i_b], c_body * t_b_old)
            qd.atomic_add(self._source_capacity_total[i_b], c_body)

    @qd.kernel
    def _compute_energy_after_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_cell = self._T[f + 1, i_b, i, j]
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy <= radius * radius:
                    in_union = True
            if in_union:
                qd.atomic_add(
                    self._source_energy_after[i_b], self._cell_capacity * t_cell
                )
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b = self._source_temperatures[s, i_b]
            qd.atomic_add(self._source_energy_after[i_b], c_body * t_b)

    @qd.kernel
    def _compute_energy_correction_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_b in qd.ndrange(self._B):
            c_total = self._source_capacity_total[i_b]
            if c_total > gs.qd_float(0.0):
                self._source_energy_correction[i_b] = (
                    -(self._source_energy_after[i_b] - self._source_energy_before[i_b])
                    / c_total
                )
            else:
                self._source_energy_correction[i_b] = gs.qd_float(0.0)

    @qd.kernel
    def _apply_energy_correction_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            self._source_temperatures[s, i_b] += self._source_energy_correction[i_b]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy <= radius * radius:
                    in_union = True
            if in_union:
                self._T[f + 1, i_b, i, j] += self._source_energy_correction[i_b]

    @qd.kernel
    def _gather_source_cells_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_cell = self._T[f + 1, i_b, i, j, k]
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                dz = pos[2] - qd.cast(k, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    qd.atomic_add(self._source_cell_sum[s, i_b], t_cell)
                    qd.atomic_add(self._source_cell_count[s, i_b], gs.qd_float(1.0))
                    in_union = True
            if in_union:
                qd.atomic_add(
                    self._source_energy_before[i_b], self._cell_capacity * t_cell
                )
                qd.atomic_add(self._source_capacity_total[i_b], self._cell_capacity)

    @qd.kernel
    def _update_source_cells_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_cell = self._T[f + 1, i_b, i, j, k]
            delta = gs.qd_float(0.0)
            for s in range(self._max_sources):
                # Inactive sources have alpha == 0, so they do not contribute.
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                dz = pos[2] - qd.cast(k, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    delta += self._source_alpha[s, i_b] * (
                        self._source_t_eq[s, i_b] - t_cell
                    )
            self._T[f + 1, i_b, i, j, k] = t_cell + delta

    @qd.kernel
    def _compute_source_equilibrium_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b_old = self._source_temperatures[s, i_b]
            n_cells = self._source_cell_count[s, i_b]
            cells_sum = self._source_cell_sum[s, i_b]
            cells_capacity = self._cell_capacity * n_cells
            total_capacity = c_body + cells_capacity
            t_eq = (c_body * t_b_old + self._cell_capacity * cells_sum) / total_capacity
            rate = self._source_rates[s]
            alpha = qd.min(rate * self._substep_dt, gs.qd_float(1.0))
            self._source_t_eq[s, i_b] = t_eq
            self._source_alpha[s, i_b] = alpha
            self._source_temperatures[s, i_b] = t_b_old + alpha * (t_eq - t_b_old)

    @qd.kernel
    def _accumulate_source_body_energy_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b_old = self._source_temperatures[s, i_b]
            qd.atomic_add(self._source_energy_before[i_b], c_body * t_b_old)
            qd.atomic_add(self._source_capacity_total[i_b], c_body)

    @qd.kernel
    def _compute_energy_after_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_cell = self._T[f + 1, i_b, i, j, k]
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                dz = pos[2] - qd.cast(k, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    in_union = True
            if in_union:
                qd.atomic_add(
                    self._source_energy_after[i_b], self._cell_capacity * t_cell
                )
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            c_body = self._source_heat_capacities[s]
            t_b = self._source_temperatures[s, i_b]
            qd.atomic_add(self._source_energy_after[i_b], c_body * t_b)

    @qd.kernel
    def _compute_energy_correction_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_b in qd.ndrange(self._B):
            c_total = self._source_capacity_total[i_b]
            if c_total > gs.qd_float(0.0):
                self._source_energy_correction[i_b] = (
                    -(self._source_energy_after[i_b] - self._source_energy_before[i_b])
                    / c_total
                )
            else:
                self._source_energy_correction[i_b] = gs.qd_float(0.0)

    @qd.kernel
    def _apply_energy_correction_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._source_active[s] == 0:
                continue
            self._source_temperatures[s, i_b] += self._source_energy_correction[i_b]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            in_union = False
            for s in range(self._max_sources):
                if self._source_active[s] == 0:
                    continue
                pos = self._source_positions[s, i_b]
                dx = pos[0] - qd.cast(i, gs.qd_float) * self._dx
                dy = pos[1] - qd.cast(j, gs.qd_float) * self._dx
                dz = pos[2] - qd.cast(k, gs.qd_float) * self._dx
                radius = self._source_radii[s]
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    in_union = True
            if in_union:
                self._T[f + 1, i_b, i, j, k] += self._source_energy_correction[i_b]
