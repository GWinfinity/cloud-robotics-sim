"""Joule-heating solver for genesis-world 1.3.2."""

# mypy: ignore-errors
from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import quadrants as qd
from genesis.engine.solvers.base_solver import Solver

from .options import JouleHeatingOptions

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class _JouleHeatingEntityMarker:
    """Dummy marker so the simulator resets our field state."""

    pass


@qd.data_oriented
class JouleHeatingSolver(Solver):
    """Electric-current and Joule-heating source solver on a Cartesian grid.

    Solves the quasi-static electric field

        ∇·(σ ∇V) = 0

    using Jacobi iteration, then computes the volumetric heat source

        Q = σ |∇V|² .

    The heat source can either be injected into ``scene.sim.thermal_solver``
    (when ``couple_to_thermal=True``) or drive a small internal transient
    thermal solve (when ``couple_to_thermal=False``).

    Notes:
    -----
    - Only Dirichlet voltage boundaries are supported in v1.
    - Gradients through the fixed-iteration Jacobi solve are supported when
      ``scene.requires_grad=True``. Gradients w.r.t. boundary values are
      straight-through in this version.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: JouleHeatingOptions):
        super().__init__(scene, sim, options)
        self._options = options
        self._dim = options.dim
        self._shape = tuple(options.resolution)
        self._nx = int(options.resolution[0])
        self._ny = int(options.resolution[1])
        self._nz = int(options.resolution[2]) if self._dim == 3 else 1
        self._dx = float(options.dx)
        self._sigma_value = options.sigma
        self._sigma_scalar = float(np.asarray(options.sigma).flat[0])
        self._rho = float(options.rho)
        self._cp = float(options.cp)
        self._k = float(options.k)
        self._alpha = self._k / (self._rho * self._cp)
        self._voltage_boundary_value = float(options.voltage_boundary_value)
        self._max_iter = int(options.max_iter)
        self._tol = float(options.tol)
        self._couple_to_thermal = bool(options.couple_to_thermal)
        self._initial_temperature = options.initial_temperature

        self._V: qd.Field | None = None
        self._V_tmp: qd.Field | None = None
        self._Q: qd.Field | None = None
        self._J: qd.Field | None = None
        self._sigma: qd.Field | None = None
        self._T: qd.Field | None = None
        self._voltage_boundary_mask: qd.Field | None = None
        self._n_frames: int = 1

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

        needs_grad = self._scene.requires_grad
        self._V = qd.field(dtype=gs.qd_float, shape=full_shape, needs_grad=needs_grad)
        self._V_tmp = qd.field(
            dtype=gs.qd_float, shape=(self._B, *self._shape), needs_grad=needs_grad
        )
        self._Q = qd.field(dtype=gs.qd_float, shape=full_shape, needs_grad=needs_grad)
        self._J = qd.field(
            dtype=gs.qd_vec3, shape=(self._B, *self._shape), needs_grad=needs_grad
        )
        self._sigma = qd.field(
            dtype=gs.qd_float, shape=(self._B, *self._shape), needs_grad=needs_grad
        )
        self._voltage_boundary_mask = qd.field(
            dtype=gs.qd_int, shape=(self._B, *self._shape)
        )

        self._init_voltage()
        self._init_conductivity()
        self._init_voltage_boundary_mask()

        if (
            not self._couple_to_thermal
            or getattr(self._sim, "thermal_solver", None) is None
        ):
            self._T = qd.field(
                dtype=gs.qd_float, shape=full_shape, needs_grad=needs_grad
            )
            self._init_temperature()

        self._entities.append(_JouleHeatingEntityMarker())
        self._check_stability()

    def _init_voltage(self) -> None:
        """Initialize the potential field to the Dirichlet boundary value."""
        arr = np.full(
            (self._n_frames, self._B, *self._shape),
            self._voltage_boundary_value,
            dtype=gs.np_float,
        )
        self._V.from_numpy(arr)
        arr_tmp = np.full(
            (self._B, *self._shape), self._voltage_boundary_value, dtype=gs.np_float
        )
        self._V_tmp.from_numpy(arr_tmp)

    def _init_conductivity(self) -> None:
        """Broadcast conductivity to the batched grid."""
        sigma_arr = np.asarray(self._options.sigma, dtype=gs.np_float)
        if sigma_arr.shape == self._shape:
            sigma_batch = np.broadcast_to(
                sigma_arr[None, ...], (self._B, *self._shape)
            ).copy()
        elif sigma_arr.shape == (self._B, *self._shape):
            sigma_batch = sigma_arr.copy()
        else:
            sigma_batch = np.full(
                (self._B, *self._shape), float(self._sigma_scalar), dtype=gs.np_float
            )
        self._sigma.from_numpy(sigma_batch)

    def _init_voltage_boundary_mask(self) -> None:
        """Initialize the boundary mask to zero (all boundaries Neumann).

        Dirichlet nodes are marked later through ``set_voltage_boundary``.
        """
        mask = np.zeros((self._B, *self._shape), dtype=np.int32)
        self._voltage_boundary_mask.from_numpy(mask)

    def _make_initial_temperature_array(self) -> np.ndarray:
        """Return an array of shape (B, *shape) for the initial temperature."""
        base_shape = self._shape
        init = self._initial_temperature
        if callable(init):
            coords = np.meshgrid(
                *[np.arange(n) * self._dx for n in base_shape],
                indexing="ij",
            )
            arr = np.asarray(init(*coords), dtype=gs.np_float)
        elif isinstance(init, np.ndarray):
            arr = np.asarray(init, dtype=gs.np_float)
        else:
            arr = np.full(base_shape, float(init), dtype=gs.np_float)
        if arr.shape != base_shape:
            raise ValueError(
                f"initial_temperature shape {arr.shape} does not match grid shape {base_shape}"
            )
        return np.broadcast_to(arr, (self._B, *base_shape)).copy()

    def _init_temperature(self) -> None:
        arr = self._make_initial_temperature_array()
        full = np.broadcast_to(arr[None, ...], (self._n_frames, *arr.shape)).copy()
        self._T.from_numpy(full)

    def _check_stability(self) -> None:
        """Check explicit FTCS CFL for the internal thermal solve."""
        if (
            self._couple_to_thermal
            and getattr(self._sim, "thermal_solver", None) is not None
        ):
            return
        r = self._alpha * self._substep_dt / (self._dx * self._dx)
        limit = 1.0 / 6.0 if self._dim == 3 else 0.25
        if r > limit:
            raise ValueError(
                f"Joule-heating internal thermal FTCS unstable: "
                f"alpha*dt/dx^2 = {r:.4f} > {limit:.4f}. "
                "Reduce dt, increase dx, or lower k/(rho*cp)."
            )

    # --------------------------------------------------------------------------
    # Public accessors
    # --------------------------------------------------------------------------

    def _frame_or_full(self, arr: np.ndarray, squeeze_env: bool = True) -> np.ndarray:
        """Helper to return frame 0 and optionally squeeze the env dimension."""
        frame = arr[0]
        if squeeze_env and self._scene.n_envs == 0:
            return np.asarray(frame[0])
        return np.asarray(frame)

    def get_voltage(self) -> np.ndarray:
        """Return the electric potential field [V]."""
        if self._V is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._frame_or_full(self._V.to_numpy())

    def set_voltage(self, voltage: np.ndarray) -> None:
        """Set the electric potential field from a host array [V]."""
        if self._V is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        voltage = np.asarray(voltage, dtype=gs.np_float)
        expected = self._shape if self._scene.n_envs == 0 else (self._B, *self._shape)
        if voltage.shape != expected:
            raise ValueError(
                f"voltage shape {voltage.shape} does not match expected {expected}"
            )
        if self._scene.n_envs == 0:
            voltage = np.broadcast_to(voltage, (self._B, *self._shape)).copy()
        full = np.broadcast_to(
            voltage[None, ...], (self._n_frames, *voltage.shape)
        ).copy()
        self._V.from_numpy(full)

    def get_current_density(self) -> np.ndarray:
        """Return the current-density vector field [A/m²]."""
        if self._J is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        arr = self._J.to_numpy()
        if self._scene.n_envs == 0:
            return np.asarray(arr[0])
        return np.asarray(arr)

    def get_heat_source(self) -> np.ndarray:
        """Return the volumetric heat-source field [W/m³]."""
        if self._Q is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._frame_or_full(self._Q.to_numpy())

    def get_temperature(self) -> np.ndarray:
        """Return the temperature field [K] (internal solve only)."""
        if self._T is None:
            if self._couple_to_thermal:
                raise RuntimeError(
                    "Temperature is managed by scene.sim.thermal_solver when "
                    "couple_to_thermal=True"
                )
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._frame_or_full(self._T.to_numpy())

    def set_temperature(self, temperature: np.ndarray) -> None:
        """Set the temperature field for the internal thermal solve [K]."""
        if self._T is None:
            raise RuntimeError(
                "Temperature field is only available when couple_to_thermal=False"
            )
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

    # --------------------------------------------------------------------------
    # Material / boundary setters
    # --------------------------------------------------------------------------

    def set_conductivity(self, sigma: np.ndarray) -> None:
        """Set the electrical-conductivity field [S/m]."""
        if self._sigma is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        sigma = np.asarray(sigma, dtype=gs.np_float)
        expected = self._shape if self._scene.n_envs == 0 else (self._B, *self._shape)
        if sigma.shape != expected:
            raise ValueError(
                f"sigma shape {sigma.shape} does not match expected {expected}"
            )
        if self._scene.n_envs == 0:
            sigma = np.broadcast_to(sigma, (self._B, *self._shape)).copy()
        self._sigma.from_numpy(sigma)

    def set_voltage_boundary(self, name: str, value: float) -> None:
        """Apply a Dirichlet voltage to one domain face.

        Parameters
        ----------
        name : str
            One of ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``,
            ``z_max``. For 2D grids ``z_min``/``z_max`` are ignored.
        value : float
            Boundary voltage in volts.
        """
        if self._V is None or self._voltage_boundary_mask is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")

        arr = self._V.to_numpy()
        mask = self._voltage_boundary_mask.to_numpy()

        if self._dim == 2:
            slices = {
                "x_min": (slice(None), 0, slice(None)),
                "x_max": (slice(None), -1, slice(None)),
                "y_min": (slice(None), slice(None), 0),
                "y_max": (slice(None), slice(None), -1),
            }
        else:
            slices = {
                "x_min": (slice(None), 0, slice(None), slice(None)),
                "x_max": (slice(None), -1, slice(None), slice(None)),
                "y_min": (slice(None), slice(None), 0, slice(None)),
                "y_max": (slice(None), slice(None), -1, slice(None)),
                "z_min": (slice(None), slice(None), slice(None), 0),
                "z_max": (slice(None), slice(None), slice(None), -1),
            }

        if name not in slices:
            raise ValueError(f"unknown boundary name: {name}")

        sl = (slice(None),) + slices[name][1:]
        arr[(slice(None),) + sl] = float(value)
        mask[sl] = 1
        self._V.from_numpy(arr)
        self._voltage_boundary_mask.from_numpy(mask)

    # --------------------------------------------------------------------------
    # Simulation lifecycle
    # --------------------------------------------------------------------------

    def process_input(self, in_backward: bool = False) -> None:
        pass

    def process_input_grad(self) -> None:
        pass

    def substep_pre_coupling(self, f: int) -> None:
        # Solve electric potential for frame f+1.
        self._copy_v_to_tmp(f)
        for _ in range(self._max_iter):
            if self._dim == 2:
                self._jacobi_step_v_2d()
            else:
                self._jacobi_step_v_3d()
        self._copy_tmp_to_v(f)

        # Compute current density and heat source from the new potential.
        if self._dim == 2:
            self._compute_j_2d()
            self._compute_q_2d(f)
        else:
            self._compute_j_3d()
            self._compute_q_3d(f)

        # Inject heat source into thermal solver or internal thermal field.
        if self._couple_to_thermal:
            thermal = getattr(self._sim, "thermal_solver", None)
            if thermal is not None and thermal._T is not None:
                if self._dim == 2:
                    self._apply_q_to_thermal_2d(f)
                else:
                    self._apply_q_to_thermal_3d(f)
            elif self._T is not None:
                if self._dim == 2:
                    self._heat_step_2d(f)
                else:
                    self._heat_step_3d(f)
        elif self._T is not None:
            if self._dim == 2:
                self._heat_step_2d(f)
            else:
                self._heat_step_3d(f)

    def substep_pre_coupling_grad(self, f: int) -> None:
        # Back-propagate through internal thermal step or thermal injection.
        if not self._couple_to_thermal and self._T is not None:
            if self._dim == 2:
                self._heat_step_2d.grad(f)
            else:
                self._heat_step_3d.grad(f)

        # Back-propagate through Q = sigma |grad V|^2.
        if self._dim == 2:
            self._compute_q_2d.grad(f)
            self._compute_j_2d.grad()
        else:
            self._compute_q_3d.grad(f)
            self._compute_j_3d.grad()

        # Back-propagate through the fixed-iteration Jacobi solve.
        for _ in range(self._max_iter):
            if self._dim == 2:
                self._jacobi_step_v_2d.grad()
            else:
                self._jacobi_step_v_3d.grad()
        if self._dim == 2:
            self._copy_tmp_to_v_2d.grad(f)
            self._copy_v_to_tmp_2d.grad(f)
        else:
            self._copy_tmp_to_v_3d.grad(f)
            self._copy_v_to_tmp_3d.grad(f)

    def substep_post_coupling(self, f: int) -> None:
        pass

    def substep_post_coupling_grad(self, f: int) -> None:
        pass

    def reset_grad(self) -> None:
        if not self._sim.requires_grad:
            return
        if self._V is not None:
            self._V.grad.fill(0.0)
        if self._V_tmp is not None:
            self._V_tmp.grad.fill(0.0)
        if self._Q is not None:
            self._Q.grad.fill(0.0)
        if self._J is not None:
            self._J.grad.fill(0.0)
        if self._T is not None:
            self._T.grad.fill(0.0)
        if self._sigma is not None:
            self._sigma.grad.fill(0.0)

    def save_ckpt(self, ckpt_name: str) -> None:
        if self._V is None:
            return
        self.copy_frame(self._sim.substeps_local, 0)
        if self._T is not None:
            self._copy_t_frame(self._sim.substeps_local, 0)
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "V": gs.zeros(
                        (self._B, *self._shape), dtype=gs.tc_float, scene=self._scene
                    ),
                }
                if self._T is not None:
                    self._ckpt[ckpt_name]["T"] = gs.zeros(
                        (self._B, *self._shape), dtype=gs.tc_float, scene=self._scene
                    )
            self._kernel_get_state(0, self._ckpt[ckpt_name]["V"])
            if self._T is not None:
                self._kernel_get_t_state(0, self._ckpt[ckpt_name]["T"])

    def load_ckpt(self, ckpt_name: str) -> None:
        if self._V is None:
            return
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)
        if self._T is not None:
            self._copy_t_frame(0, self._sim.substeps_local)
            self._copy_t_grad(0, self._sim.substeps_local)
        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)
            self._kernel_set_state(0, self._ckpt[ckpt_name]["V"])
            if self._T is not None:
                self._kernel_set_t_state(0, self._ckpt[ckpt_name]["T"])

    def get_state(self, f: int) -> dict | None:
        if not self.is_active or self._V is None:
            return None
        state: dict[str, np.ndarray] = {"V": self._V.to_numpy()[f].copy()}
        if self._T is not None:
            state["T"] = self._T.to_numpy()[f].copy()
        return state

    def set_state(
        self,
        f: int,
        state: dict | None,
        envs_idx: np.ndarray | None = None,
    ) -> None:
        if state is None or self._V is None:
            return
        # envs_idx could be supported later; for now we copy all envs.
        arr = self._V.to_numpy()
        arr[f] = state["V"]
        self._V.from_numpy(arr)
        if self._T is not None and "T" in state:
            t_arr = self._T.to_numpy()
            t_arr[f] = state["T"]
            self._T.from_numpy(t_arr)

    def add_grad_from_state(self, state: dict | None) -> None:
        pass

    def collect_output_grads(self) -> None:
        pass

    # --------------------------------------------------------------------------
    # Quadrants kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _copy_v_to_tmp_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V_tmp[i_b, i, j] = self._V[f, i_b, i, j]

    @qd.kernel
    def _copy_v_to_tmp_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V_tmp[i_b, i, j, k] = self._V[f, i_b, i, j, k]

    def _copy_v_to_tmp(self, f: int) -> None:
        if self._dim == 2:
            self._copy_v_to_tmp_2d(f)
        else:
            self._copy_v_to_tmp_3d(f)

    @qd.kernel
    def _copy_tmp_to_v_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[f + 1, i_b, i, j] = self._V_tmp[i_b, i, j]

    @qd.kernel
    def _copy_tmp_to_v_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[f + 1, i_b, i, j, k] = self._V_tmp[i_b, i, j, k]

    def _copy_tmp_to_v(self, f: int) -> None:
        if self._dim == 2:
            self._copy_tmp_to_v_2d(f)
        else:
            self._copy_tmp_to_v_3d(f)

    @qd.kernel
    def _jacobi_step_v_2d(self):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if self._voltage_boundary_mask[i_b, i, j] == 1:
                # Dirichlet node: keep the prescribed value.
                self._V_tmp[i_b, i, j] = self._V_tmp[i_b, i, j]
            else:
                # Zero-Neumann boundaries use the boundary value as the ghost value.
                im = i - 1 if i > 0 else i
                ip = i + 1 if i < self._nx - 1 else i
                jm = j - 1 if j > 0 else j
                jp = j + 1 if j < self._ny - 1 else j
                # Face conductivities via harmonic mean of adjacent cells.
                s_c = self._sigma[i_b, i, j]
                s_ip = self._sigma[i_b, ip, j]
                s_im = self._sigma[i_b, im, j]
                s_jp = self._sigma[i_b, i, jp]
                s_jm = self._sigma[i_b, i, jm]
                sigma_x = 2.0 * s_c * s_ip / (s_c + s_ip + 1e-10)
                sigma_xm = 2.0 * s_c * s_im / (s_c + s_im + 1e-10)
                sigma_y = 2.0 * s_c * s_jp / (s_c + s_jp + 1e-10)
                sigma_ym = 2.0 * s_c * s_jm / (s_c + s_jm + 1e-10)
                denom = sigma_x + sigma_xm + sigma_y + sigma_ym + 1e-10
                self._V_tmp[i_b, i, j] = (
                    sigma_x * self._V_tmp[i_b, ip, j]
                    + sigma_xm * self._V_tmp[i_b, im, j]
                    + sigma_y * self._V_tmp[i_b, i, jp]
                    + sigma_ym * self._V_tmp[i_b, i, jm]
                ) / denom

    @qd.kernel
    def _jacobi_step_v_3d(self):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if self._voltage_boundary_mask[i_b, i, j, k] == 1:
                self._V_tmp[i_b, i, j, k] = self._V_tmp[i_b, i, j, k]
            else:
                im = i - 1 if i > 0 else i
                ip = i + 1 if i < self._nx - 1 else i
                jm = j - 1 if j > 0 else j
                jp = j + 1 if j < self._ny - 1 else j
                km = k - 1 if k > 0 else k
                kp = k + 1 if k < self._nz - 1 else k
                # Face conductivities via harmonic mean of adjacent cells.
                s_c = self._sigma[i_b, i, j, k]
                s_ip = self._sigma[i_b, ip, j, k]
                s_im = self._sigma[i_b, im, j, k]
                s_jp = self._sigma[i_b, i, jp, k]
                s_jm = self._sigma[i_b, i, jm, k]
                s_kp = self._sigma[i_b, i, j, kp]
                s_km = self._sigma[i_b, i, j, km]
                sigma_x = 2.0 * s_c * s_ip / (s_c + s_ip + 1e-10)
                sigma_xm = 2.0 * s_c * s_im / (s_c + s_im + 1e-10)
                sigma_y = 2.0 * s_c * s_jp / (s_c + s_jp + 1e-10)
                sigma_ym = 2.0 * s_c * s_jm / (s_c + s_jm + 1e-10)
                sigma_z = 2.0 * s_c * s_kp / (s_c + s_kp + 1e-10)
                sigma_zm = 2.0 * s_c * s_km / (s_c + s_km + 1e-10)
                denom = (
                    sigma_x + sigma_xm + sigma_y + sigma_ym + sigma_z + sigma_zm + 1e-10
                )
                self._V_tmp[i_b, i, j, k] = (
                    sigma_x * self._V_tmp[i_b, ip, j, k]
                    + sigma_xm * self._V_tmp[i_b, im, j, k]
                    + sigma_y * self._V_tmp[i_b, i, jp, k]
                    + sigma_ym * self._V_tmp[i_b, i, jm, k]
                    + sigma_z * self._V_tmp[i_b, i, j, kp]
                    + sigma_zm * self._V_tmp[i_b, i, j, km]
                ) / denom

    @qd.kernel
    def _compute_j_2d(self):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._J[i_b, i, j] = gs.qd_vec3(0.0, 0.0, 0.0)
            else:
                sigma = self._sigma[i_b, i, j]
                dvdx = (self._V_tmp[i_b, i + 1, j] - self._V_tmp[i_b, i - 1, j]) / (
                    2.0 * self._dx
                )
                dvdy = (self._V_tmp[i_b, i, j + 1] - self._V_tmp[i_b, i, j - 1]) / (
                    2.0 * self._dx
                )
                self._J[i_b, i, j] = gs.qd_vec3(-sigma * dvdx, -sigma * dvdy, 0.0)

    @qd.kernel
    def _compute_j_3d(self):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                self._J[i_b, i, j, k] = gs.qd_vec3(0.0, 0.0, 0.0)
            else:
                sigma = self._sigma[i_b, i, j, k]
                dvdx = (
                    self._V_tmp[i_b, i + 1, j, k] - self._V_tmp[i_b, i - 1, j, k]
                ) / (2.0 * self._dx)
                dvdy = (
                    self._V_tmp[i_b, i, j + 1, k] - self._V_tmp[i_b, i, j - 1, k]
                ) / (2.0 * self._dx)
                dvdz = (
                    self._V_tmp[i_b, i, j, k + 1] - self._V_tmp[i_b, i, j, k - 1]
                ) / (2.0 * self._dx)
                self._J[i_b, i, j, k] = gs.qd_vec3(
                    -sigma * dvdx, -sigma * dvdy, -sigma * dvdz
                )

    @qd.kernel
    def _compute_q_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._Q[f, i_b, i, j] = gs.qd_float(0.0)
            else:
                j_vec = self._J[i_b, i, j]
                jx = j_vec[0]
                jy = j_vec[1]
                sigma = self._sigma[i_b, i, j]
                # Q = |J|^2 / sigma
                self._Q[f, i_b, i, j] = (jx * jx + jy * jy) / (sigma + 1e-10)

    @qd.kernel
    def _compute_q_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
                self._Q[f, i_b, i, j, k] = gs.qd_float(0.0)
            else:
                j_vec = self._J[i_b, i, j, k]
                jx = j_vec[0]
                jy = j_vec[1]
                jz = j_vec[2]
                sigma = self._sigma[i_b, i, j, k]
                self._Q[f, i_b, i, j, k] = (jx * jx + jy * jy + jz * jz) / (
                    sigma + 1e-10
                )

    @qd.kernel
    def _apply_q_to_thermal_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            q = self._Q[f, i_b, i, j]
            dt_over_rhocp = self._substep_dt / (self._rho * self._cp)
            self._sim.thermal_solver._T[f + 1, i_b, i, j] += q * dt_over_rhocp

    @qd.kernel
    def _apply_q_to_thermal_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            q = self._Q[f, i_b, i, j, k]
            dt_over_rhocp = self._substep_dt / (self._rho * self._cp)
            self._sim.thermal_solver._T[f + 1, i_b, i, j, k] += q * dt_over_rhocp

    @qd.kernel
    def _heat_step_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_old = self._T[f, i_b, i, j]
            r = self._alpha * self._substep_dt / (self._dx * self._dx)
            heating = self._Q[f, i_b, i, j] * self._substep_dt / (self._rho * self._cp)
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                # Insulated (zero-Neumann) boundary.
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                lap = (
                    self._T[f, i_b, ip, j]
                    + self._T[f, i_b, im, j]
                    + self._T[f, i_b, i, jp]
                    + self._T[f, i_b, i, jm]
                    - 4.0 * t_old
                )
                self._T[f + 1, i_b, i, j] = t_old + r * lap + heating
            else:
                lap = (
                    self._T[f, i_b, i + 1, j]
                    + self._T[f, i_b, i - 1, j]
                    + self._T[f, i_b, i, j + 1]
                    + self._T[f, i_b, i, j - 1]
                    - 4.0 * t_old
                )
                self._T[f + 1, i_b, i, j] = t_old + r * lap + heating

    @qd.kernel
    def _heat_step_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_old = self._T[f, i_b, i, j, k]
            r = self._alpha * self._substep_dt / (self._dx * self._dx)
            heating = (
                self._Q[f, i_b, i, j, k] * self._substep_dt / (self._rho * self._cp)
            )
            if (
                i == 0
                or i == self._nx - 1
                or j == 0
                or j == self._ny - 1
                or k == 0
                or k == self._nz - 1
            ):
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
                    - 6.0 * t_old
                )
                self._T[f + 1, i_b, i, j, k] = t_old + r * lap + heating
            else:
                lap = (
                    self._T[f, i_b, i + 1, j, k]
                    + self._T[f, i_b, i - 1, j, k]
                    + self._T[f, i_b, i, j + 1, k]
                    + self._T[f, i_b, i, j - 1, k]
                    + self._T[f, i_b, i, j, k + 1]
                    + self._T[f, i_b, i, j, k - 1]
                    - 6.0 * t_old
                )
                self._T[f + 1, i_b, i, j, k] = t_old + r * lap + heating

    # --------------------------------------------------------------------------
    # State / checkpoint kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _copy_frame_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[target, i_b, i, j] = self._V[source, i_b, i, j]

    @qd.kernel
    def _copy_frame_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[target, i_b, i, j, k] = self._V[source, i_b, i, j, k]

    def copy_frame(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_frame_2d(source, target)
        else:
            self._copy_frame_3d(source, target)

    @qd.kernel
    def _copy_grad_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V.grad[target, i_b, i, j] = self._V.grad[source, i_b, i, j]

    @qd.kernel
    def _copy_grad_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V.grad[target, i_b, i, j, k] = self._V.grad[source, i_b, i, j, k]

    def copy_grad(self, source: int, target: int) -> None:
        if self._dim == 2:
            self._copy_grad_2d(source, target)
        else:
            self._copy_grad_3d(source, target)

    @qd.kernel
    def _reset_grad_till_frame_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_f, i, j, i_b in qd.ndrange(f, self._nx, self._ny, self._B):
            self._V.grad[i_f, i_b, i, j] = gs.qd_float(0.0)

    @qd.kernel
    def _reset_grad_till_frame_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i_f, i, j, k, i_b in qd.ndrange(f, self._nx, self._ny, self._nz, self._B):
            self._V.grad[i_f, i_b, i, j, k] = gs.qd_float(0.0)

    def reset_grad_till_frame(self, f: int) -> None:
        if self._dim == 2:
            self._reset_grad_till_frame_2d(f)
        else:
            self._reset_grad_till_frame_3d(f)

    @qd.kernel
    def _kernel_get_state_2d(self, f: qd.i32, v_out: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            v_out[i_b, i, j] = self._V[f, i_b, i, j]

    @qd.kernel
    def _kernel_get_state_3d(self, f: qd.i32, v_out: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            v_out[i_b, i, j, k] = self._V[f, i_b, i, j, k]

    def _kernel_get_state(self, f: int, v_out: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_get_state_2d(f, v_out)
        else:
            self._kernel_get_state_3d(f, v_out)

    @qd.kernel
    def _kernel_set_state_2d(self, f: qd.i32, v_in: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[f, i_b, i, j] = v_in[i_b, i, j]

    @qd.kernel
    def _kernel_set_state_3d(self, f: qd.i32, v_in: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[f, i_b, i, j, k] = v_in[i_b, i, j, k]

    def _kernel_set_state(self, f: int, v_in: gs.Tensor) -> None:
        if self._dim == 2:
            self._kernel_set_state_2d(f, v_in)
        else:
            self._kernel_set_state_3d(f, v_in)

    @qd.kernel
    def _copy_t_frame_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T[target, i_b, i, j] = self._T[source, i_b, i, j]

    @qd.kernel
    def _copy_t_frame_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T[target, i_b, i, j, k] = self._T[source, i_b, i, j, k]

    def _copy_t_frame(self, source: int, target: int) -> None:
        if self._T is None:
            return
        if self._dim == 2:
            self._copy_t_frame_2d(source, target)
        else:
            self._copy_t_frame_3d(source, target)

    @qd.kernel
    def _copy_t_grad_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T.grad[target, i_b, i, j] = self._T.grad[source, i_b, i, j]

    @qd.kernel
    def _copy_t_grad_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T.grad[target, i_b, i, j, k] = self._T.grad[source, i_b, i, j, k]

    def _copy_t_grad(self, source: int, target: int) -> None:
        if self._T is None:
            return
        if self._dim == 2:
            self._copy_t_grad_2d(source, target)
        else:
            self._copy_t_grad_3d(source, target)

    @qd.kernel
    def _kernel_get_t_state_2d(self, f: qd.i32, t_out: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_out[i_b, i, j] = self._T[f, i_b, i, j]

    @qd.kernel
    def _kernel_get_t_state_3d(self, f: qd.i32, t_out: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_out[i_b, i, j, k] = self._T[f, i_b, i, j, k]

    def _kernel_get_t_state(self, f: int, t_out: gs.Tensor) -> None:
        if self._T is None:
            return
        if self._dim == 2:
            self._kernel_get_t_state_2d(f, t_out)
        else:
            self._kernel_get_t_state_3d(f, t_out)

    @qd.kernel
    def _kernel_set_t_state_2d(self, f: qd.i32, t_in: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T[f, i_b, i, j] = t_in[i_b, i, j]

    @qd.kernel
    def _kernel_set_t_state_3d(self, f: qd.i32, t_in: qd.types.ndarray()):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T[f, i_b, i, j, k] = t_in[i_b, i, j, k]

    def _kernel_set_t_state(self, f: int, t_in: gs.Tensor) -> None:
        if self._T is None:
            return
        if self._dim == 2:
            self._kernel_set_t_state_2d(f, t_in)
        else:
            self._kernel_set_t_state_3d(f, t_in)
