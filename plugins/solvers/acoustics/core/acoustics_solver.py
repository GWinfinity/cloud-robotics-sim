"""Grid-based time-domain acoustics solver for genesis-world 1.4.6.

Solves the linear acoustic wave equation on a regular Cartesian grid with a
second-order leapfrog scheme:

    p^{n+1} = 2 p^n - p^{n-1} + (c * dt / dx)^2 * lap(p^n) + source terms

Boundary modes mirror the classical choices: absorbing sponge layers
(approximate open domain, cf. Fluent's sponge layer / Mechanical's PML),
Dirichlet p = 0 (pressure release) and Neumann zero-gradient (rigid wall).

This is a plugin solver: it is injected into ``scene.sim._solvers`` before
``scene.build()`` and participates in the normal simulator lifecycle, exactly
like ``plugins.solvers.thermal`` and ``plugins.solvers.joule_heating``.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import genesis as gs
import numpy as np
import quadrants as qd
import torch
from genesis.engine.solvers.base_solver import Solver
from genesis.grad.bridge import custom_backward
from genesis.utils.data import ArrayItem, DataKind

from .options import AcousticsOptions

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

REFERENCE_PRESSURE = 2e-5  # Pa, 0 dB SPL in air


class _AcousticsEntityMarker:
    """Dummy marker making ``n_entities > 0`` so the simulator resets our field state."""

    pass


@dataclass
class AcousticSource:
    """A monopole pressure source injected into the grid.

    Parameters
    ----------
    position : tuple[float, float, float]
        World-space position of the source (the z component is ignored in 2D).
    signal : Callable[[float], float] | np.ndarray
        Either a callable mapping simulation time (seconds) to pressure
        amplitude in Pa, or a pre-sampled per-substep array of amplitudes.
    amplitude : float
        Scalar multiplier applied to the signal.
    radius : float
        Injection radius in metres. The per-substep signal is distributed
        evenly over the grid cells inside the radius.
    """

    position: tuple[float, float, float]
    signal: Callable[[float], float] | np.ndarray
    amplitude: float = 1.0
    radius: float = 0.02
    _slot: int | None = field(default=None, repr=False)
    _solver: "AcousticsSolver | None" = field(default=None, repr=False)

    def value_at(self, t: float, dt: float) -> float:
        """Evaluate the source signal at time ``t``."""
        if callable(self.signal):
            return float(self.amplitude * float(self.signal(t)))
        arr = np.asarray(self.signal, dtype=np.float64).reshape(-1)
        idx = min(int(t / dt), arr.shape[0] - 1)
        return float(self.amplitude * arr[idx])


@dataclass
class AcousticBody:
    """One-way vibro-acoustic coupler: a rigid body radiating sound.

    Each substep the body's velocity is read from the coupled Genesis entity
    and its vertical acceleration ``dv_z / dt`` is injected as a monopole
    (``rho * dv_z / dt``), modelling a piston-like radiator. This is the
    acoustic analogue of ``ThermalSource``'s entity coupling in the thermal
    solver: structure motion drives the acoustic field, but the acoustic
    pressure does not push back on the structure (one-way coupling).
    """

    entity: "Entity"
    radius: float = 0.05
    amplitude: float = 1.0
    _slot: int | None = field(default=None, repr=False)
    _solver: "AcousticsSolver | None" = field(default=None, repr=False)
    _vel_prev: np.ndarray | None = field(default=None, repr=False)


@dataclass
class AcousticProbe:
    """A virtual microphone recording the pressure at a fixed grid point.

    Signals are harvested into a host-side history buffer at the end of every
    ``scene.step()``; retrieve them with ``AcousticsSolver.get_signal(probe)``.
    Post-processing helpers ``spectrum()`` and ``spl()`` mirror the receiver
    post-processing of Fluent's FW-H model (FFT -> SPL with a 2e-5 Pa
    reference pressure).
    """

    position: tuple[float, float, float]
    _slot: int | None = field(default=None, repr=False)
    _solver: "AcousticsSolver | None" = field(default=None, repr=False)

    def signal(self) -> np.ndarray:
        """Return the recorded pressure time history of environment 0."""
        if self._solver is None:
            raise RuntimeError("probe is not attached to a solver")
        return self._solver.get_signal(self)

    def spectrum(self, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequencies, SPL dB)`` of the recorded signal (Hann FFT)."""
        sig = self.signal()
        if sig.size < 4:
            raise RuntimeError("not enough recorded samples for an FFT")
        dt = float(dt if dt is not None else self._solver._options.dt)
        window = np.hanning(sig.size)
        spectrum = np.fft.rfft(sig * window)
        freqs = np.fft.rfftfreq(sig.size, d=dt)
        # Coherent gain correction for the Hann window (sum(w)/N = 0.5).
        amp = np.abs(spectrum) / (sig.size * 0.5)
        spl = 20.0 * np.log10(np.maximum(amp, 1e-12) / REFERENCE_PRESSURE)
        return freqs, spl

    def spl(self, dt: float | None = None) -> float:
        """Return the overall SPL of the recorded signal (rms, dB)."""
        sig = self.signal()
        rms = float(np.sqrt(np.mean(sig**2))) if sig.size else 0.0
        return 20.0 * np.log10(max(rms, 1e-12) / REFERENCE_PRESSURE)


class AcousticsSolverState:
    """Dynamic state queried from an AcousticsSolver."""

    def __init__(self, scene: "Scene", solver: "AcousticsSolver"):
        self._scene = scene
        self._s_global = solver.sim.cur_step_global
        self.P = torch.zeros(
            (solver._B, *solver._shape),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.P_prev = torch.zeros(
            (solver._B, *solver._shape),
            dtype=gs.tc_float,
            device=gs.device,
        )

    def serializable(self) -> None:
        self._scene = None
        self.P = self.P.detach()
        self.P_prev = self.P_prev.detach()


@qd.data_oriented
class AcousticsSolver(Solver):
    """Explicit leapfrog acoustic wave solver on a regular Cartesian grid.

    Notes:
    -----
    - Plugin solver injected into ``scene.sim._solvers`` before ``scene.build()``.
    - The pressure field is differentiable through the linear interior and
      boundary kernels when ``scene.requires_grad=True``. Source injection and
      rigid-body velocity coupling are forward-only for now (same
      ``TODO(MUSA/autodiff)`` limitation as the thermal source coupling).
    - Two time levels (``_p``, ``_p_prev``) are stored, so memory usage is
      roughly twice the thermal solver for the same grid.
    - Overlapping injection regions of multiple sources should be avoided:
      cell contributions are added without atomics.
    """

    # The fields hold the frames of one step in differentiable scenes and of
    # one substep otherwise; the simulator rolls them generically (see
    # 'Solver.has_frames' in genesis-world 1.4.6).
    has_frames = True
    # Gradients flow through the leapfrog and boundary kernels; source/body
    # injection and probe recording are grouped into one gradient-free
    # recorded entry (see 'custom_backward' in substep_pre_coupling), matching
    # the forward-only coupling this plugin has always documented.
    is_differentiable = True

    def __init__(self, scene: "Scene", sim: "Simulator", options: AcousticsOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._dim = options.dim
        self._shape = tuple(int(r) for r in options.resolution)
        self._nx, self._ny = self._shape[0], self._shape[1]
        self._nz = self._shape[2] if self._dim == 3 else 1
        self._dx = float(options.dx)
        self._c = float(options.c)
        self._rho = float(options.rho)
        self._boundary_mode = options.boundary_mode
        self._sponge_layers = int(options.sponge_layers)
        self._sponge_max_damping = float(options.sponge_max_damping)
        self._initial_pressure = options.initial_pressure
        self._max_sources = int(options.max_sources)
        self._max_bodies = int(options.max_bodies)
        self._max_probes = int(options.max_probes)

        self._p: qd.Field | None = None
        self._p_prev: qd.Field | None = None
        self._sig: qd.Field | None = None
        self._src_active: qd.Field | None = None
        self._src_pos: qd.Field | None = None
        self._src_cells: qd.Field | None = None
        self._src_values: qd.Field | None = None
        self._body_active: qd.Field | None = None
        self._body_pos: qd.Field | None = None
        self._body_cells: qd.Field | None = None
        self._body_values: qd.Field | None = None
        self._probe_active: qd.Field | None = None
        self._probe_cells: qd.Field | None = None
        self._n_frames: int = 1
        self._time: float = 0.0
        self._last_harvest_step: int = -1

        self._sources: list[AcousticSource] = []
        self._bodies: list[AcousticBody] = []
        self._probes: list[AcousticProbe] = []
        self._probe_history: np.ndarray | None = None

        # Host-side slot metadata mirrors (uploaded to device slots).
        ms = max(1, self._max_sources)
        mb = max(1, self._max_bodies)
        mp = max(1, self._max_probes)
        self._src_active_np = np.zeros(ms, dtype=np.int32)
        self._src_pos_np = np.zeros((ms, 3), dtype=np.float64)
        self._src_cells_np = np.ones(ms, dtype=np.int32)
        self._src_values_np = np.zeros(ms, dtype=np.float64)
        self._body_active_np = np.zeros(mb, dtype=np.int32)
        self._body_pos_np = np.zeros((mb, 3), dtype=np.float64)
        self._body_cells_np = np.ones(mb, dtype=np.int32)
        self._body_values_np = np.zeros(mb, dtype=np.float64)
        self._probe_active_np = np.zeros(mp, dtype=np.int32)
        self._probe_cells_np = np.zeros((mp, 3), dtype=np.int32)

    @property
    def is_active(self) -> bool:
        return True

    @property
    def n_frames(self) -> int:
        """Number of frames the fields hold (see 'Solver.has_frames')."""
        return self._n_frames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cell_of(self, position: tuple[float, float, float]) -> tuple[int, int, int]:
        ci = int(np.clip(int(position[0] / self._dx), 0, self._nx - 1))
        cj = int(np.clip(int(position[1] / self._dx), 0, self._ny - 1))
        ck = (
            int(np.clip(int(position[2] / self._dx), 0, self._nz - 1))
            if self._dim == 3
            else 0
        )
        return ci, cj, ck

    def _cells_in_radius(self, radius: float) -> int:
        """Injection radius expressed in whole grid cells (>= 1)."""
        return max(1, int(round(radius / self._dx)))

    def _sync_slots_to_device(self) -> None:
        if self._src_active is None:
            return
        self._src_active.from_numpy(self._src_active_np)
        self._src_pos.from_numpy(self._src_pos_np.astype(gs.np_float))
        self._src_cells.from_numpy(self._src_cells_np)
        self._src_values.from_numpy(self._src_values_np.astype(gs.np_float))
        self._body_active.from_numpy(self._body_active_np)
        self._body_pos.from_numpy(self._body_pos_np.astype(gs.np_float))
        self._body_cells.from_numpy(self._body_cells_np)
        self._body_values.from_numpy(self._body_values_np.astype(gs.np_float))
        self._probe_active.from_numpy(self._probe_active_np)
        self._probe_cells.from_numpy(self._probe_cells_np)

    # ------------------------------------------------------------------
    # Build / allocation
    # ------------------------------------------------------------------

    def build(self) -> None:
        super().build()
        # genesis 1.4.6 Solver does not mix in TimeBasedMixin; the solver
        # advances at its own options.dt per substep.
        self._substep_dt = float(self._options.dt)
        self._check_stability()

        # Differentiable scenes hold the frames of every substep of one step
        # (the backward pass replays the whole step); other scenes hold one
        # substep in two frames (see 'Solver.has_frames').
        self._n_frames = self._sim.substeps + 1 if self._sim.requires_grad else 2
        # Field layout is (frames, *grid, B), the genesis batch-last convention.
        full_shape = (self._n_frames, *self._shape, self._B)
        self._p = qd.field(dtype=gs.qd_float, shape=full_shape, needs_grad=True)
        self._p_prev = qd.field(dtype=gs.qd_float, shape=full_shape, needs_grad=True)
        self._sig = qd.field(
            dtype=gs.qd_float,
            shape=(self._n_frames, max(1, self._max_probes), self._B),
            needs_grad=False,
        )

        init = self._make_initial_array()
        init_full = np.broadcast_to(
            np.moveaxis(init, 0, -1)[None, ...], full_shape
        ).copy()
        self._p.from_numpy(init_full)
        self._p_prev.from_numpy(np.zeros(full_shape, dtype=gs.np_float))
        self._sig.from_numpy(
            np.zeros(
                (self._n_frames, max(1, self._max_probes), self._B), dtype=gs.np_float
            )
        )

        # Source / body / probe slot fields.
        ms, mb, mp = (
            max(1, self._max_sources),
            max(1, self._max_bodies),
            max(1, self._max_probes),
        )
        self._src_active = qd.field(dtype=qd.i32, shape=(ms,))
        self._src_pos = qd.field(dtype=gs.qd_float, shape=(ms, 3))
        self._src_cells = qd.field(dtype=qd.i32, shape=(ms,))
        self._src_values = qd.field(dtype=gs.qd_float, shape=(ms,))
        self._body_active = qd.field(dtype=qd.i32, shape=(mb,))
        self._body_pos = qd.field(dtype=gs.qd_float, shape=(mb, 3))
        self._body_cells = qd.field(dtype=qd.i32, shape=(mb,))
        self._body_values = qd.field(dtype=gs.qd_float, shape=(mb,))
        self._probe_active = qd.field(dtype=qd.i32, shape=(mp,))
        self._probe_cells = qd.field(dtype=qd.i32, shape=(mp, 3))
        self._sync_slots_to_device()

        self._probe_history = np.zeros((0, self._B, mp), dtype=np.float64)

        # Dummy entity so Simulator.reset() restores our field state.
        self._entities.append(_AcousticsEntityMarker())

    def _make_initial_array(self) -> np.ndarray:
        """Return an array of shape (B, *shape) for the initial pressure."""
        if callable(self._initial_pressure):
            coords = np.meshgrid(
                *[np.arange(n) * self._dx for n in self._shape], indexing="ij"
            )
            arr = np.asarray(self._initial_pressure(*coords), dtype=gs.np_float)
        elif isinstance(self._initial_pressure, np.ndarray):
            arr = np.asarray(self._initial_pressure, dtype=gs.np_float)
        else:
            assert self._initial_pressure is not None
            arr = np.full(self._shape, float(self._initial_pressure), dtype=gs.np_float)
        if arr.shape != self._shape:
            raise ValueError(
                f"initial_pressure shape {arr.shape} does not match grid shape {self._shape}"
            )
        return np.broadcast_to(arr, (self._B, *self._shape)).copy()

    def _check_stability(self) -> None:
        """Check the leapfrog CFL condition."""
        dt = self._substep_dt
        cfl = self._c * dt / self._dx
        limit = 1.0 / np.sqrt(self._dim)
        if cfl > limit:
            raise ValueError(
                f"Acoustic leapfrog unstable: c*dt/dx = {cfl:.4f} > {limit:.4f}. "
                "Reduce dt, increase dx, or lower c."
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_pressure(self) -> np.ndarray:
        """Return the current pressure field.

        For ``n_envs == 0`` the leading batch dimension is squeezed out.
        """
        if self._p is None:
            raise RuntimeError("AcousticsSolver has not been built yet")
        arr = self._p.to_numpy()
        frame = arr[0]
        if self._scene.n_envs == 0:
            return np.asarray(frame[..., 0])
        return np.asarray(np.moveaxis(frame, -1, 0))

    def set_pressure(self, pressure: np.ndarray) -> None:
        """Set the pressure field from a host array (all time levels)."""
        if self._p is None:
            raise RuntimeError("AcousticsSolver has not been built yet")
        pressure = np.asarray(pressure, dtype=gs.np_float)
        expected = self._shape if self._scene.n_envs == 0 else (self._B, *self._shape)
        if pressure.shape != expected:
            raise ValueError(
                f"pressure shape {pressure.shape} does not match expected {expected}"
            )
        if self._scene.n_envs == 0:
            pressure = np.broadcast_to(pressure, (self._B, *self._shape)).copy()
        # Field layout is (frames, *grid, B): move the batch axis last.
        full = np.broadcast_to(
            np.moveaxis(pressure, 0, -1)[None, ...],
            (self._n_frames, *self._shape, self._B),
        ).copy()
        self._p.from_numpy(full)

    def get_signal(self, probe: AcousticProbe) -> np.ndarray:
        """Return the recorded pressure history of environment 0 for a probe."""
        if self._probe_history is None or probe._slot is None:
            raise RuntimeError("AcousticsSolver has not been built yet")
        return np.asarray(self._probe_history[:, 0, probe._slot])

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def add_source(
        self,
        position: tuple[float, float, float],
        signal: Callable[[float], float] | np.ndarray,
        amplitude: float = 1.0,
        radius: float = 0.02,
    ) -> AcousticSource:
        """Register a monopole pressure source.

        May be called both before and after ``scene.build()`` up to
        ``options.max_sources`` total sources.
        """
        if len(self._sources) >= self._max_sources:
            raise RuntimeError(
                f"Cannot add more than {self._max_sources} acoustic sources. "
                "Increase AcousticsOptions.max_sources."
            )
        slot = self._first_free(self._src_active_np)
        source = AcousticSource(
            position=tuple(float(p) for p in position),
            signal=signal,
            amplitude=float(amplitude),
            radius=float(radius),
            _slot=slot,
            _solver=self,
        )
        self._src_active_np[slot] = 1
        self._src_pos_np[slot] = source.position
        self._src_cells_np[slot] = self._cells_in_radius(source.radius)
        self._sources.append(source)
        if self._p is not None:
            self._sync_slots_to_device()
        return source

    def add_body(
        self,
        entity: "Entity",
        radius: float = 0.05,
        amplitude: float = 1.0,
    ) -> AcousticBody:
        """Register a rigid body as a one-way vibro-acoustic radiator."""
        if len(self._bodies) >= self._max_bodies:
            raise RuntimeError(
                f"Cannot add more than {self._max_bodies} acoustic bodies. "
                "Increase AcousticsOptions.max_bodies."
            )
        slot = self._first_free(self._body_active_np)
        body = AcousticBody(
            entity=entity,
            radius=float(radius),
            amplitude=float(amplitude),
            _slot=slot,
            _solver=self,
        )
        self._body_active_np[slot] = 1
        try:
            self._body_pos_np[slot] = np.asarray(
                entity.get_pos(), dtype=np.float64
            ).reshape(-1)[:3]
        except Exception:
            pass  # Unbuilt entity: position is populated on the first substep.
        self._body_cells_np[slot] = self._cells_in_radius(body.radius)
        self._bodies.append(body)
        if self._p is not None:
            self._sync_slots_to_device()
        return body

    def add_probe(self, position: tuple[float, float, float]) -> AcousticProbe:
        """Register a virtual microphone at a world-space position."""
        if len(self._probes) >= self._max_probes:
            raise RuntimeError(
                f"Cannot add more than {self._max_probes} acoustic probes. "
                "Increase AcousticsOptions.max_probes."
            )
        slot = self._first_free(self._probe_active_np)
        probe = AcousticProbe(
            position=tuple(float(p) for p in position), _slot=slot, _solver=self
        )
        self._probe_active_np[slot] = 1
        self._probe_cells_np[slot] = self._cell_of(probe.position)
        self._probes.append(probe)
        if self._p is not None:
            self._sync_slots_to_device()
        return probe

    @staticmethod
    def _first_free(active_np: np.ndarray) -> int:
        for i in range(active_np.shape[0]):
            if active_np[i] == 0:
                return i
        raise RuntimeError("No free slot available.")

    # ------------------------------------------------------------------
    # Substep driving
    # ------------------------------------------------------------------

    def _update_source_values(self) -> None:
        """Evaluate source signals and body accelerations for this substep."""
        t = self._time
        dt = self._substep_dt
        for source in self._sources:
            if source._slot is not None:
                self._src_values_np[source._slot] = source.value_at(t, dt)
        for body in self._bodies:
            if body._slot is None:
                continue
            vel = np.asarray(body.entity.get_vel(), dtype=np.float64).reshape(-1)[:3]
            pos = np.asarray(body.entity.get_pos(), dtype=np.float64).reshape(-1)[:3]
            if body._vel_prev is None:
                body._vel_prev = vel.copy()
            accel = (vel - body._vel_prev) / dt
            body._vel_prev = vel.copy()
            self._body_pos_np[body._slot] = pos
            self._body_values_np[body._slot] = body.amplitude * self._rho * accel[2]
        if self._p is not None:
            if self._src_values_np.size:
                self._src_values.from_numpy(self._src_values_np.astype(gs.np_float))
            if self._body_values_np.size:
                self._body_values.from_numpy(self._body_values_np.astype(gs.np_float))
                self._body_pos.from_numpy(self._body_pos_np.astype(gs.np_float))

    # genesis-world 1.4.6 drives solver substeps through 'substep_pre_coupling'
    # / 'substep_post_coupling' alone; the backward pass of a differentiable
    # scene replays them while recording the quadrants kernel launches, which
    # 'LaunchRecorder' then reverses automatically. The explicit
    # process_input*/substep_*_grad/save_ckpt/load_ckpt hooks of genesis 1.4.5
    # no longer exist, and state checkpoints go through the 'data' protocol.

    def substep_pre_coupling(self, f: int) -> None:
        self._update_source_values()
        if self._dim == 2:
            self._step_leapfrog_2d_interior(f)
            self._apply_boundary_2d(f)
        else:
            self._step_leapfrog_3d_interior(f)
            self._apply_boundary_3d(f)
        # Source/body injection and probe recording are gradient-free; grouping
        # their launches into one recorded entry with no backward keeps the
        # autodiff tape from trying to reverse them, and the recorder still
        # restores what they overwrote.
        with custom_backward(None):
            if self._dim == 2:
                self._inject_sources_2d(f)
                self._inject_bodies_2d(f)
                self._record_probes_2d(f)
            else:
                self._inject_sources_3d(f)
                self._inject_bodies_3d(f)
                self._record_probes_3d(f)
        self._time += self._substep_dt

    def substep_post_coupling(self, f: int) -> None:
        pass

    def reset_grad(self) -> None:
        if self._sim.requires_grad and self._p is not None:
            self._p.grad.fill(0.0)
            self._p_prev.grad.fill(0.0)

    # ------------------------------------------------------------------
    # Boundary dispatch
    # ------------------------------------------------------------------

    def _apply_boundary_2d(self, f: int) -> None:
        if self._boundary_mode == "dirichlet":
            self._boundary_dirichlet_2d(f)
        elif self._boundary_mode == "neumann":
            self._boundary_neumann_2d(f)
        else:
            self._boundary_absorbing_2d(f)

    def _apply_boundary_3d(self, f: int) -> None:
        if self._boundary_mode == "dirichlet":
            self._boundary_dirichlet_3d(f)
        elif self._boundary_mode == "neumann":
            self._boundary_neumann_3d(f)
        else:
            self._boundary_absorbing_3d(f)

    # ------------------------------------------------------------------
    # Checkpoints / states (genesis-world 1.4.6 'data' protocol)
    # ------------------------------------------------------------------

    @property
    def data(self):
        """Yield the pressure fields for the generic checkpoint protocol."""
        if self._p is None:
            return
        yield ArrayItem("P", DataKind.STATE, self._p)
        yield ArrayItem("P_prev", DataKind.STATE, self._p_prev)

    def copy_frame(self, source: int, target: int) -> None:
        """Roll field frames; harvest probe history once per simulation step.

        The simulator calls this with (n_frames - 1, 0) at the end of every
        step (and, without gradients, at the end of every substep, since one
        substep fits in the two frames it holds). The step-clock guard keeps
        backward replays, which revisit the same step with a decreasing
        clock, from appending duplicate history.
        """
        if self._p is None:
            return
        if source == self._n_frames - 1 and target == 0:
            cur_step = self._sim.cur_step_global
            if cur_step > self._last_harvest_step:
                self._harvest_probes()
                self._last_harvest_step = cur_step
        self._copy_frame_p(source, target)
        self._copy_frame_p_prev(source, target)
        self._copy_frame_sig(source, target)

    def get_state(self, f: int) -> AcousticsSolverState | None:
        if not self.is_active or self._p is None:
            return None
        state = AcousticsSolverState(self._scene, self)
        # Between two steps the state sits in frame 0 (see 'Solver.has_frames').
        self._kernel_get_state(0, state.P, state.P_prev)
        # While a differentiable rollout is recorded, register the query with
        # the autograd bridge so the tensor gradients route back through
        # 'add_grad_from_state'.
        if self.is_recording:
            self.record_read((state.P, state.P_prev), (self._p, self._p_prev))
        return state

    def set_state(
        self,
        f: int,
        state: AcousticsSolverState | None,
        envs_idx: np.ndarray | None = None,
    ) -> None:
        if state is None or self._p is None:
            return
        self._kernel_set_state(0, state.P, state.P_prev)
        self._time = 0.0

    def add_grad_from_state(self, grads, selection) -> None:
        """Seed the frame-0 field adjoints with queried-state gradients.

        genesis-world 1.4.6 'ReadOperation.backward' hands the gradients of
        the tensors 'get_state' returned (None for those the loss does not
        reach); the field layout is (frames, *grid, B), so the (B, *grid)
        gradient tensors map onto frame 0 through the dedicated kernels.
        """
        if self._p is None:
            return
        for grad, kernel in zip(
            grads,
            (self._kernel_add_grad_from_p, self._kernel_add_grad_from_p_prev),
        ):
            if grad is not None and grad.numel() > 0:
                kernel(0, grad)

    # ------------------------------------------------------------------
    # Probe harvesting
    # ------------------------------------------------------------------

    def _harvest_probes(self) -> None:
        if self._sig is None or self._probe_history is None:
            return
        arr = self._sig.to_numpy()  # (n_frames, max_probes, B)
        # History layout is (steps, B, max_probes).
        new = np.moveaxis(
            np.asarray(arr[1 : self._n_frames + 1], dtype=np.float64), -1, 1
        )
        if self._probe_history.shape[0] == 0:
            self._probe_history = new.copy()
        else:
            self._probe_history = np.concatenate([self._probe_history, new], axis=0)

    # ------------------------------------------------------------------
    # Quadrants kernels: leapfrog interior
    # ------------------------------------------------------------------

    @qd.kernel
    def _step_leapfrog_2d_interior(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx - 2, self._ny - 2, self._B):
            ii = i + 1
            jj = j + 1
            r2 = (
                self._c
                * self._c
                * self._substep_dt
                * self._substep_dt
                / (self._dx * self._dx)
            )
            lap = (
                self._p[f, ii + 1, jj, i_b]
                + self._p[f, ii - 1, jj, i_b]
                + self._p[f, ii, jj + 1, i_b]
                + self._p[f, ii, jj - 1, i_b]
                - 4.0 * self._p[f, ii, jj, i_b]
            )
            self._p[f + 1, ii, jj, i_b] = (
                2.0 * self._p[f, ii, jj, i_b] - self._p_prev[f, ii, jj, i_b] + r2 * lap
            )
            self._p_prev[f + 1, ii, jj, i_b] = self._p[f, ii, jj, i_b]

    @qd.kernel
    def _step_leapfrog_3d_interior(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(
            self._nx - 2, self._ny - 2, self._nz - 2, self._B
        ):
            ii = i + 1
            jj = j + 1
            kk = k + 1
            r2 = (
                self._c
                * self._c
                * self._substep_dt
                * self._substep_dt
                / (self._dx * self._dx)
            )
            lap = (
                self._p[f, ii + 1, jj, kk, i_b]
                + self._p[f, ii - 1, jj, kk, i_b]
                + self._p[f, ii, jj + 1, kk, i_b]
                + self._p[f, ii, jj - 1, kk, i_b]
                + self._p[f, ii, jj, kk + 1, i_b]
                + self._p[f, ii, jj, kk - 1, i_b]
                - 6.0 * self._p[f, ii, jj, kk, i_b]
            )
            self._p[f + 1, ii, jj, kk, i_b] = (
                2.0 * self._p[f, ii, jj, kk, i_b]
                - self._p_prev[f, ii, jj, kk, i_b]
                + r2 * lap
            )
            self._p_prev[f + 1, ii, jj, kk, i_b] = self._p[f, ii, jj, kk, i_b]

    # ------------------------------------------------------------------
    # Boundary kernels
    # ------------------------------------------------------------------

    @qd.kernel
    def _boundary_dirichlet_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._p[f + 1, i, j, i_b] = 0.0
                self._p_prev[f + 1, i, j, i_b] = 0.0

    @qd.kernel
    def _boundary_dirichlet_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                self._p[f + 1, i, j, k, i_b] = 0.0
                self._p_prev[f + 1, i, j, k, i_b] = 0.0

    @qd.kernel
    def _boundary_neumann_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                r2 = (
                    self._c
                    * self._c
                    * self._substep_dt
                    * self._substep_dt
                    / (self._dx * self._dx)
                )
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                lap = (
                    self._p[f, ip, j, i_b]
                    + self._p[f, im, j, i_b]
                    + self._p[f, i, jp, i_b]
                    + self._p[f, i, jm, i_b]
                    - 4.0 * self._p[f, i, j, i_b]
                )
                self._p[f + 1, i, j, i_b] = (
                    2.0 * self._p[f, i, j, i_b] - self._p_prev[f, i, j, i_b] + r2 * lap
                )
                self._p_prev[f + 1, i, j, i_b] = self._p[f, i, j, i_b]

    @qd.kernel
    def _boundary_neumann_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                r2 = (
                    self._c
                    * self._c
                    * self._substep_dt
                    * self._substep_dt
                    / (self._dx * self._dx)
                )
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                km = qd.max(k - 1, 0)
                kp = qd.min(k + 1, self._nz - 1)
                lap = (
                    self._p[f, ip, j, k, i_b]
                    + self._p[f, im, j, k, i_b]
                    + self._p[f, i, jp, k, i_b]
                    + self._p[f, i, jm, k, i_b]
                    + self._p[f, i, j, kp, i_b]
                    + self._p[f, i, j, km, i_b]
                    - 6.0 * self._p[f, i, j, k, i_b]
                )
                self._p[f + 1, i, j, k, i_b] = (
                    2.0 * self._p[f, i, j, k, i_b]
                    - self._p_prev[f, i, j, k, i_b]
                    + r2 * lap
                )
                self._p_prev[f + 1, i, j, k, i_b] = self._p[f, i, j, k, i_b]

    @qd.kernel
    def _boundary_absorbing_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            d = qd.min(qd.min(i, self._nx - 1 - i), qd.min(j, self._ny - 1 - j))
            if d < self._sponge_layers:
                r2 = (
                    self._c
                    * self._c
                    * self._substep_dt
                    * self._substep_dt
                    / (self._dx * self._dx)
                )
                s = 1.0 - d / self._sponge_layers
                xi = self._sponge_max_damping * s * s
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                lap = (
                    self._p[f, ip, j, i_b]
                    + self._p[f, im, j, i_b]
                    + self._p[f, i, jp, i_b]
                    + self._p[f, i, jm, i_b]
                    - 4.0 * self._p[f, i, j, i_b]
                )
                p_next = (
                    2.0 * self._p[f, i, j, i_b] - self._p_prev[f, i, j, i_b] + r2 * lap
                )
                self._p[f + 1, i, j, i_b] = p_next / (1.0 + xi * self._substep_dt)
                self._p_prev[f + 1, i, j, i_b] = self._p[f, i, j, i_b]

    @qd.kernel
    def _boundary_absorbing_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            d = qd.min(
                qd.min(qd.min(i, self._nx - 1 - i), qd.min(j, self._ny - 1 - j)),
                qd.min(k, self._nz - 1 - k),
            )
            if d < self._sponge_layers:
                r2 = (
                    self._c
                    * self._c
                    * self._substep_dt
                    * self._substep_dt
                    / (self._dx * self._dx)
                )
                s = 1.0 - qd.cast(d, gs.qd_float) / qd.cast(
                    self._sponge_layers, gs.qd_float
                )
                xi = self._sponge_max_damping * s * s
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                km = qd.max(k - 1, 0)
                kp = qd.min(k + 1, self._nz - 1)
                lap = (
                    self._p[f, ip, j, k, i_b]
                    + self._p[f, im, j, k, i_b]
                    + self._p[f, i, jp, k, i_b]
                    + self._p[f, i, jm, k, i_b]
                    + self._p[f, i, j, kp, i_b]
                    + self._p[f, i, j, km, i_b]
                    - 6.0 * self._p[f, i, j, k, i_b]
                )
                p_next = (
                    2.0 * self._p[f, i, j, k, i_b]
                    - self._p_prev[f, i, j, k, i_b]
                    + r2 * lap
                )
                self._p[f + 1, i, j, k, i_b] = p_next / (1.0 + xi * self._substep_dt)
                self._p_prev[f + 1, i, j, k, i_b] = self._p[f, i, j, k, i_b]

    # ------------------------------------------------------------------
    # Source / body injection and probe recording
    # ------------------------------------------------------------------

    @qd.kernel
    def _inject_sources_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._src_active[s] == 1:
                ci = int(self._src_pos[s, 0] / self._dx)
                cj = int(self._src_pos[s, 1] / self._dx)
                ci = qd.min(qd.max(ci, 0), self._nx - 1)
                cj = qd.min(qd.max(cj, 0), self._ny - 1)
                r = self._src_cells[s]
                count = (2 * r + 1) * (2 * r + 1)
                val = self._src_values[s] / count
                for di, dj in qd.ndrange(2 * r + 1, 2 * r + 1):
                    ii = ci + di - r
                    jj = cj + dj - r
                    if ii >= 0 and ii < self._nx and jj >= 0 and jj < self._ny:
                        self._p[f + 1, ii, jj, i_b] += val

    @qd.kernel
    def _inject_sources_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_sources, self._B):
            if self._src_active[s] == 1:
                ci = qd.cast(self._src_pos[s, 0] / self._dx, qd.i32)
                cj = qd.cast(self._src_pos[s, 1] / self._dx, qd.i32)
                ck = int(self._src_pos[s, 2] / self._dx)
                ci = qd.min(qd.max(ci, 0), self._nx - 1)
                cj = qd.min(qd.max(cj, 0), self._ny - 1)
                ck = qd.min(qd.max(ck, 0), self._nz - 1)
                r = self._src_cells[s]
                count = (2 * r + 1) * (2 * r + 1) * (2 * r + 1)
                val = self._src_values[s] / qd.cast(count, gs.qd_float)
                for di, dj, dk in qd.ndrange(2 * r + 1, 2 * r + 1, 2 * r + 1):
                    ii = ci + di - r
                    jj = cj + dj - r
                    kk = ck + dk - r
                    if (
                        ii >= 0
                        and ii < self._nx
                        and jj >= 0
                        and jj < self._ny
                        and kk >= 0
                        and kk < self._nz
                    ):
                        self._p[f + 1, ii, jj, kk, i_b] += val

    @qd.kernel
    def _inject_bodies_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_bodies, self._B):
            if self._body_active[s] == 1:
                ci = int(self._body_pos[s, 0] / self._dx)
                cj = int(self._body_pos[s, 1] / self._dx)
                ci = qd.min(qd.max(ci, 0), self._nx - 1)
                cj = qd.min(qd.max(cj, 0), self._ny - 1)
                r = self._body_cells[s]
                count = (2 * r + 1) * (2 * r + 1)
                val = self._body_values[s] / count
                for di, dj in qd.ndrange(2 * r + 1, 2 * r + 1):
                    ii = ci + di - r
                    jj = cj + dj - r
                    if ii >= 0 and ii < self._nx and jj >= 0 and jj < self._ny:
                        self._p[f + 1, ii, jj, i_b] += val

    @qd.kernel
    def _inject_bodies_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_bodies, self._B):
            if self._body_active[s] == 1:
                ci = qd.cast(self._body_pos[s, 0] / self._dx, qd.i32)
                cj = qd.cast(self._body_pos[s, 1] / self._dx, qd.i32)
                ck = int(self._body_pos[s, 2] / self._dx)
                ci = qd.min(qd.max(ci, 0), self._nx - 1)
                cj = qd.min(qd.max(cj, 0), self._ny - 1)
                ck = qd.min(qd.max(ck, 0), self._nz - 1)
                r = self._body_cells[s]
                count = (2 * r + 1) * (2 * r + 1) * (2 * r + 1)
                val = self._body_values[s] / qd.cast(count, gs.qd_float)
                for di, dj, dk in qd.ndrange(2 * r + 1, 2 * r + 1, 2 * r + 1):
                    ii = ci + di - r
                    jj = cj + dj - r
                    kk = ck + dk - r
                    if (
                        ii >= 0
                        and ii < self._nx
                        and jj >= 0
                        and jj < self._ny
                        and kk >= 0
                        and kk < self._nz
                    ):
                        self._p[f + 1, ii, jj, kk, i_b] += val

    @qd.kernel
    def _record_probes_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_probes, self._B):
            if self._probe_active[s] == 1:
                pi = self._probe_cells[s, 0]
                pj = self._probe_cells[s, 1]
                self._sig[f + 1, s, i_b] = self._p[f + 1, pi, pj, i_b]

    @qd.kernel
    def _record_probes_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for s, i_b in qd.ndrange(self._max_probes, self._B):
            if self._probe_active[s] == 1:
                pi = self._probe_cells[s, 0]
                pj = self._probe_cells[s, 1]
                pk = self._probe_cells[s, 2]
                self._sig[f + 1, s, i_b] = self._p[f + 1, pi, pj, pk, i_b]

    # ------------------------------------------------------------------
    # Frame copies
    # ------------------------------------------------------------------

    @qd.kernel
    def _copy_frame_p_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._p[target, i, j, i_b] = self._p[source, i, j, i_b]

    @qd.kernel
    def _copy_frame_p_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._p[target, i, j, k, i_b] = self._p[source, i, j, k, i_b]

    def _copy_frame_p(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_frame_p_2d(source, target)
        else:
            self._copy_frame_p_3d(source, target)

    @qd.kernel
    def _copy_frame_p_prev_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._p_prev[target, i, j, i_b] = self._p_prev[source, i, j, i_b]

    @qd.kernel
    def _copy_frame_p_prev_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._p_prev[target, i, j, k, i_b] = self._p_prev[source, i, j, k, i_b]

    def _copy_frame_p_prev(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_frame_p_prev_2d(source, target)
        else:
            self._copy_frame_p_prev_3d(source, target)

    @qd.kernel
    def _copy_frame_sig_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, s, i_b in qd.ndrange(self._nx, self._max_probes, self._B):
            self._sig[target, s, i_b] = self._sig[source, s, i_b]

    @qd.kernel
    def _copy_frame_sig_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, s, i_b in qd.ndrange(self._nx, self._max_probes, self._B):
            self._sig[target, s, i_b] = self._sig[source, s, i_b]

    def _copy_frame_sig(self, source: int, target: int) -> None:
        # The signal buffer has no spatial axes; a 1-D copy suffices for both dims.
        if self._dim == 2:
            self._copy_frame_sig_2d(source, target)
        else:
            self._copy_frame_sig_3d(source, target)

    # ------------------------------------------------------------------
    # State get/set and grad export
    # ------------------------------------------------------------------

    @qd.kernel
    def _kernel_get_state_2d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, p_out: qd.types.ndarray(), pp_out: qd.types.ndarray()
    ):
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            p_out[i_b, i, j] = self._p[f, i, j, i_b]
            pp_out[i_b, i, j] = self._p_prev[f, i, j, i_b]

    @qd.kernel
    def _kernel_get_state_3d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, p_out: qd.types.ndarray(), pp_out: qd.types.ndarray()
    ):
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            p_out[i_b, i, j, k] = self._p[f, i, j, k, i_b]
            pp_out[i_b, i, j, k] = self._p_prev[f, i, j, k, i_b]

    def _kernel_get_state(
        self, f: int, p_out: torch.Tensor, pp_out: torch.Tensor
    ) -> None:
        if self._dim == 2:
            self._kernel_get_state_2d(f, p_out, pp_out)
        else:
            self._kernel_get_state_3d(f, p_out, pp_out)

    @qd.kernel
    def _kernel_set_state_2d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, p_in: qd.types.ndarray(), pp_in: qd.types.ndarray()
    ):
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._p[f, i, j, i_b] = p_in[i_b, i, j]
            self._p_prev[f, i, j, i_b] = pp_in[i_b, i, j]

    @qd.kernel
    def _kernel_set_state_3d(  # type: ignore[no-untyped-def]
        self, f: qd.i32, p_in: qd.types.ndarray(), pp_in: qd.types.ndarray()
    ):
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._p[f, i, j, k, i_b] = p_in[i_b, i, j, k]
            self._p_prev[f, i, j, k, i_b] = pp_in[i_b, i, j, k]

    def _kernel_set_state(
        self, f: int, p_in: torch.Tensor, pp_in: torch.Tensor
    ) -> None:
        if self._dim == 2:
            self._kernel_set_state_2d(f, p_in, pp_in)
        else:
            self._kernel_set_state_3d(f, p_in, pp_in)

    @qd.kernel
    def _kernel_add_grad_from_p_2d(self, f: qd.i32, p_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._p.grad[f, i, j, i_b] += p_grad[i_b, i, j]

    @qd.kernel
    def _kernel_add_grad_from_p_3d(self, f: qd.i32, p_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._p.grad[f, i, j, k, i_b] += p_grad[i_b, i, j, k]

    def _kernel_add_grad_from_p(self, f: int, p_grad: torch.Tensor) -> None:
        if self._dim == 2:
            self._kernel_add_grad_from_p_2d(f, p_grad)
        else:
            self._kernel_add_grad_from_p_3d(f, p_grad)

    @qd.kernel
    def _kernel_add_grad_from_p_prev_2d(self, f: qd.i32, p_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._p_prev.grad[f, i, j, i_b] += p_grad[i_b, i, j]

    @qd.kernel
    def _kernel_add_grad_from_p_prev_3d(self, f: qd.i32, p_grad: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._p_prev.grad[f, i, j, k, i_b] += p_grad[i_b, i, j, k]

    def _kernel_add_grad_from_p_prev(self, f: int, p_grad: torch.Tensor) -> None:
        if self._dim == 2:
            self._kernel_add_grad_from_p_prev_2d(f, p_grad)
        else:
            self._kernel_add_grad_from_p_prev_3d(f, p_grad)
