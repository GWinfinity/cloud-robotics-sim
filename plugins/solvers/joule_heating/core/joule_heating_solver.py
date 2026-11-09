"""Joule-heating solver for genesis-world 1.3.2."""

# mypy: ignore-errors
from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import quadrants as qd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from genesis.engine.solvers.base_solver import Solver
from genesis.utils.data import ArrayItem, DataKind

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
      supported when the boundary is set after ``scene.build()`` via
      ``set_voltage_boundary`` with ``scene.requires_grad=True``.
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
        self._dt_over_dx2 = 0.0  # filled in build() once _substep_dt is known
        self._voltage_boundary_value = float(options.voltage_boundary_value)
        self._max_iter = int(options.max_iter)
        self._tol = float(options.tol)
        self._couple_to_thermal = bool(options.couple_to_thermal)
        self._solver_type = str(options.solver_type)
        self._direct_solver_cache: dict[str, object] | None = None
        self._direct_v_cache: np.ndarray | None = None
        self._initial_temperature = options.initial_temperature

        self._V: qd.Field | None = None
        self._V_tmp: qd.Field | None = None
        self._Q: qd.Field | None = None
        self._J: qd.Field | None = None
        self._sigma: qd.Field | None = None
        self._T: qd.Field | None = None
        self._voltage_boundary_mask: qd.Field | None = None
        self._boundary_voltage_field: qd.Field | None = None
        self._rho_field: qd.Field | None = None
        self._cp_field: qd.Field | None = None
        self._k_field: qd.Field | None = None
        self._n_frames: int = 1

    # The fields hold the frames of one step in differentiable scenes and of
    # one substep otherwise; the simulator rolls them generically (see
    # 'Solver.has_frames' in genesis-world 1.4.6).
    has_frames = True
    # Gradients flow through the Jacobi solve and the J/Q/thermal kernels (in
    # differentiable scenes the unfused kernels run so 'LaunchRecorder' can
    # reverse each launch; see 'substep_pre_coupling').
    is_differentiable = True

    @property
    def is_active(self) -> bool:
        return True

    @property
    def n_frames(self) -> int:
        """Number of frames the fields hold (see 'Solver.has_frames')."""
        return self._n_frames

    # --------------------------------------------------------------------------
    # Build / allocation
    # --------------------------------------------------------------------------

    def build(self) -> None:
        super().build()

        # genesis 1.4.6 Solver does not mix in TimeBasedMixin.
        self._substep_dt = self._sim.substep_dt

        # Differentiable scenes hold the frames of every substep of one step
        # (the backward pass replays the whole step); other scenes hold one
        # substep in two frames (see 'Solver.has_frames').
        self._n_frames = self._sim.substeps + 1 if self._sim.requires_grad else 2
        # Framed field layout is (frames, *grid, B), the genesis batch-last
        # convention; the unframed scratch fields keep (B, *grid).
        full_shape = (self._n_frames, *self._shape, self._B)

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
        self._boundary_voltage_field = qd.field(
            dtype=gs.qd_float, shape=(self._B, *self._shape), needs_grad=needs_grad
        )
        self._rho_field = qd.field(
            dtype=gs.qd_float, shape=(self._B,), needs_grad=needs_grad
        )
        self._cp_field = qd.field(
            dtype=gs.qd_float, shape=(self._B,), needs_grad=needs_grad
        )
        self._k_field = qd.field(
            dtype=gs.qd_float, shape=(self._B,), needs_grad=needs_grad
        )
        self._r_field = qd.field(
            dtype=gs.qd_float, shape=(self._B,), needs_grad=needs_grad
        )
        self._dt_over_rhocp_field = qd.field(
            dtype=gs.qd_float, shape=(self._B,), needs_grad=needs_grad
        )

        self._dt_over_dx2 = self._substep_dt / (self._dx * self._dx)

        self._init_voltage()
        self._init_conductivity()
        self._init_voltage_boundary_mask()
        self._init_scalar_material_params()

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
            (self._n_frames, *self._shape, self._B),
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
        The boundary-voltage field is initialized to the default boundary
        value so interior cells start from the same initial guess as ``_V``.
        """
        mask = np.zeros((self._B, *self._shape), dtype=np.int32)
        self._voltage_boundary_mask.from_numpy(mask)
        boundary_values = np.full(
            (self._B, *self._shape), self._voltage_boundary_value, dtype=gs.np_float
        )
        self._boundary_voltage_field.from_numpy(boundary_values)

    def _init_scalar_material_params(self) -> None:
        """Broadcast scalar material properties to batched differentiable fields."""
        rho_arr = np.full((self._B,), self._rho, dtype=gs.np_float)
        cp_arr = np.full((self._B,), self._cp, dtype=gs.np_float)
        k_arr = np.full((self._B,), self._k, dtype=gs.np_float)
        self._rho_field.from_numpy(rho_arr)
        self._cp_field.from_numpy(cp_arr)
        self._k_field.from_numpy(k_arr)
        self._init_thermal_constants()

    def _init_thermal_constants(self) -> None:
        """Precompute per-env thermal constants for the internal heat step."""
        rho = self._rho_field.to_numpy()
        cp = self._cp_field.to_numpy()
        k = self._k_field.to_numpy()
        r_arr = (k / (rho * cp)) * self._dt_over_dx2
        dt_over_rhocp_arr = self._substep_dt / (rho * cp)
        self._r_field.from_numpy(r_arr.astype(gs.np_float))
        self._dt_over_rhocp_field.from_numpy(dt_over_rhocp_arr.astype(gs.np_float))

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
        full = np.broadcast_to(
            np.moveaxis(arr, 0, -1)[None, ...], (self._n_frames, *self._shape, self._B)
        ).copy()
        self._T.from_numpy(full)

    def _check_stability(self) -> None:
        """Check explicit FTCS CFL for the internal thermal solve."""
        if (
            self._couple_to_thermal
            and getattr(self._sim, "thermal_solver", None) is not None
        ):
            return
        rho = float(self._rho_field.to_numpy()[0])
        cp = float(self._cp_field.to_numpy()[0])
        k = float(self._k_field.to_numpy()[0])
        alpha = k / (rho * cp)
        r = alpha * self._substep_dt / (self._dx * self._dx)
        limit = 1.0 / 6.0 if self._dim == 3 else 0.25
        if r > limit:
            raise ValueError(
                f"Joule-heating internal thermal FTCS unstable: "
                f"alpha*dt/dx^2 = {r:.4f} > {limit:.4f}. "
                "Reduce dt, increase dx, or lower k/(rho*cp)."
            )

    # --------------------------------------------------------------------------
    # Direct sparse solver (forward-only, single-env)
    # --------------------------------------------------------------------------

    def _invalidate_direct_solver_cache(self) -> None:
        """Drop cached sparse factorization and solution.

        Called when conductivity or boundary values change.
        """
        self._direct_solver_cache = None
        self._direct_v_cache = None

    def _build_linear_system_2d(self) -> tuple:
        """Build ``A`` and ``b`` for the 2D potential solve (single env).

        Returns:
        -------
        A : scipy.sparse.csr_matrix
            N×N sparse matrix, N = nx * ny.
        b : np.ndarray
            Right-hand side vector of length N.
        """
        nx, ny = self._nx, self._ny
        n = nx * ny
        mask = self._voltage_boundary_mask.to_numpy()[0]
        bv = self._boundary_voltage_field.to_numpy()[0]
        sigma = self._sigma.to_numpy()[0]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b = np.zeros(n, dtype=gs.np_float)

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                if mask[i, j] == 1:
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                    b[idx] = bv[i, j]
                    continue

                im = i - 1 if i > 0 else i
                ip = i + 1 if i < nx - 1 else i
                jm = j - 1 if j > 0 else j
                jp = j + 1 if j < ny - 1 else j

                s_c = sigma[i, j]
                s_ip = sigma[ip, j]
                s_im = sigma[im, j]
                s_jp = sigma[i, jp]
                s_jm = sigma[i, jm]
                sigma_x = 2.0 * s_c * s_ip / (s_c + s_ip + 1e-10)
                sigma_xm = 2.0 * s_c * s_im / (s_c + s_im + 1e-10)
                sigma_y = 2.0 * s_c * s_jp / (s_c + s_jp + 1e-10)
                sigma_ym = 2.0 * s_c * s_jm / (s_c + s_jm + 1e-10)
                denom = sigma_x + sigma_xm + sigma_y + sigma_ym + 1e-10

                rows.append(idx)
                cols.append(idx)
                data.append(1.0)

                rows.append(idx)
                cols.append(ip * ny + j)
                data.append(-sigma_x / denom)

                rows.append(idx)
                cols.append(im * ny + j)
                data.append(-sigma_xm / denom)

                rows.append(idx)
                cols.append(i * ny + jp)
                data.append(-sigma_y / denom)

                rows.append(idx)
                cols.append(i * ny + jm)
                data.append(-sigma_ym / denom)

        a_mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return a_mat, b

    def _build_linear_system_3d(self) -> tuple:
        """Build ``A`` and ``b`` for the 3D potential solve (single env)."""
        nx, ny, nz = self._nx, self._ny, self._nz
        n = nx * ny * nz
        mask = self._voltage_boundary_mask.to_numpy()[0]
        bv = self._boundary_voltage_field.to_numpy()[0]
        sigma = self._sigma.to_numpy()[0]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b = np.zeros(n, dtype=gs.np_float)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = (i * ny + j) * nz + k
                    if mask[i, j, k] == 1:
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0)
                        b[idx] = bv[i, j, k]
                        continue

                    im = i - 1 if i > 0 else i
                    ip = i + 1 if i < nx - 1 else i
                    jm = j - 1 if j > 0 else j
                    jp = j + 1 if j < ny - 1 else j
                    km = k - 1 if k > 0 else k
                    kp = k + 1 if k < nz - 1 else k

                    s_c = sigma[i, j, k]
                    s_ip = sigma[ip, j, k]
                    s_im = sigma[im, j, k]
                    s_jp = sigma[i, jp, k]
                    s_jm = sigma[i, jm, k]
                    s_kp = sigma[i, j, kp]
                    s_km = sigma[i, j, km]
                    sigma_x = 2.0 * s_c * s_ip / (s_c + s_ip + 1e-10)
                    sigma_xm = 2.0 * s_c * s_im / (s_c + s_im + 1e-10)
                    sigma_y = 2.0 * s_c * s_jp / (s_c + s_jp + 1e-10)
                    sigma_ym = 2.0 * s_c * s_jm / (s_c + s_jm + 1e-10)
                    sigma_z = 2.0 * s_c * s_kp / (s_c + s_kp + 1e-10)
                    sigma_zm = 2.0 * s_c * s_km / (s_c + s_km + 1e-10)
                    denom = (
                        sigma_x
                        + sigma_xm
                        + sigma_y
                        + sigma_ym
                        + sigma_z
                        + sigma_zm
                        + 1e-10
                    )

                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)

                    rows.append(idx)
                    cols.append((ip * ny + j) * nz + k)
                    data.append(-sigma_x / denom)

                    rows.append(idx)
                    cols.append((im * ny + j) * nz + k)
                    data.append(-sigma_xm / denom)

                    rows.append(idx)
                    cols.append((i * ny + jp) * nz + k)
                    data.append(-sigma_y / denom)

                    rows.append(idx)
                    cols.append((i * ny + jm) * nz + k)
                    data.append(-sigma_ym / denom)

                    rows.append(idx)
                    cols.append((i * ny + j) * nz + kp)
                    data.append(-sigma_z / denom)

                    rows.append(idx)
                    cols.append((i * ny + j) * nz + km)
                    data.append(-sigma_zm / denom)

        a_mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return a_mat, b

    def _solve_v_direct(self, f: int) -> None:
        """Solve the electric potential with a sparse direct solver.

        The sparse factorization and the converged voltage are cached and
        reused across substeps until conductivity or boundary values change.
        """
        if self._B != 1:
            raise RuntimeError(
                "direct solver is only supported for single-environment scenes"
            )

        if self._direct_v_cache is not None:
            arr = self._V.to_numpy()
            arr[f + 1, ..., 0] = self._direct_v_cache
            self._V.from_numpy(arr)
            return

        if self._direct_solver_cache is None:
            if self._dim == 2:
                a_mat, b = self._build_linear_system_2d()
            else:
                a_mat, b = self._build_linear_system_3d()
            factorization = spla.factorized(a_mat)
            self._direct_solver_cache = {
                "A": a_mat,
                "b": b,
                "factorization": factorization,
            }
        else:
            b = self._direct_solver_cache["b"].copy()
            factorization = self._direct_solver_cache["factorization"]

        v_flat = factorization(b)
        v = v_flat.reshape(self._shape)
        self._direct_v_cache = v.copy()
        arr = self._V.to_numpy()
        arr[f + 1, ..., 0] = v
        self._V.from_numpy(arr)

    # --------------------------------------------------------------------------
    # Public accessors
    # --------------------------------------------------------------------------

    def _frame_or_full(self, arr: np.ndarray, squeeze_env: bool = True) -> np.ndarray:
        """Helper to return frame 0 and optionally squeeze the env dimension."""
        frame = arr[0]
        if squeeze_env and self._scene.n_envs == 0:
            return np.asarray(frame[..., 0])
        return np.asarray(np.moveaxis(frame, -1, 0))

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
        # Field layout is (frames, *grid, B): move the batch axis last.
        full = np.broadcast_to(
            np.moveaxis(voltage, 0, -1)[None, ...],
            (self._n_frames, *self._shape, self._B),
        ).copy()
        self._V.from_numpy(full)
        self._direct_v_cache = None

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
        # Field layout is (frames, *grid, B): move the batch axis last.
        full = np.broadcast_to(
            np.moveaxis(temperature, 0, -1)[None, ...],
            (self._n_frames, *self._shape, self._B),
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
        self._invalidate_direct_solver_cache()

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
        if (
            self._V is None
            or self._voltage_boundary_mask is None
            or self._boundary_voltage_field is None
        ):
            raise RuntimeError("JouleHeatingSolver has not been built yet")

        arr = self._V.to_numpy()
        mask = self._voltage_boundary_mask.to_numpy()
        boundary_values = self._boundary_voltage_field.to_numpy()

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
        # _V is (frames, *grid, B); mask/boundary_values are (B, *grid).
        v_sl = (slice(None),) + slices[name][1:] + (slice(None),)
        arr[v_sl] = float(value)
        boundary_values[sl] = float(value)
        mask[sl] = 1
        self._V.from_numpy(arr)
        self._voltage_boundary_mask.from_numpy(mask)
        self._boundary_voltage_field.from_numpy(boundary_values)
        self._invalidate_direct_solver_cache()

    def _set_scalar_param(
        self, field: qd.Field | None, attr_name: str, value: float
    ) -> None:
        """Update a batched scalar parameter field and its cached Python value."""
        if field is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        arr = np.full((self._B,), float(value), dtype=gs.np_float)
        field.from_numpy(arr)
        setattr(self, attr_name, float(value))

    def set_rho(self, rho: float) -> None:
        """Set mass density [kg/m^3] for all environments."""
        self._set_scalar_param(self._rho_field, "_rho", rho)
        self._init_thermal_constants()

    def set_cp(self, cp: float) -> None:
        """Set specific heat capacity [J/(kg*K)] for all environments."""
        self._set_scalar_param(self._cp_field, "_cp", cp)
        self._init_thermal_constants()

    def set_k(self, k: float) -> None:
        """Set thermal conductivity [W/(m*K)] for all environments."""
        self._set_scalar_param(self._k_field, "_k", k)
        self._alpha = k / (self._rho * self._cp)
        self._init_thermal_constants()

    def get_rho(self) -> np.ndarray:
        """Return the per-environment density values [kg/m^3]."""
        if self._rho_field is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._rho_field.to_numpy().copy()

    def get_cp(self) -> np.ndarray:
        """Return the per-environment specific heat values [J/(kg*K)]."""
        if self._cp_field is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._cp_field.to_numpy().copy()

    def get_k(self) -> np.ndarray:
        """Return the per-environment thermal conductivity values [W/(m*K)]."""
        if self._k_field is None:
            raise RuntimeError("JouleHeatingSolver has not been built yet")
        return self._k_field.to_numpy().copy()

    # --------------------------------------------------------------------------
    # Simulation lifecycle
    # --------------------------------------------------------------------------

    # genesis-world 1.4.6 drives solver substeps through 'substep_pre_coupling'
    # / 'substep_post_coupling' alone; the backward pass of a differentiable
    # scene replays them while recording the quadrants kernel launches, which
    # 'LaunchRecorder' then reverses automatically. Differentiable scenes
    # therefore run the unfused kernel sequence (the fused kernels fuse the
    # same-field in-place updates that autodiff cannot reverse); plain scenes
    # keep the fused kernels. 'substep_pre_coupling_grad' below is kept as the
    # plugin's manual gradient API (used by the tests to pull gradients into
    # material fields without a scene-level backward).

    def substep_pre_coupling(self, f: int) -> None:
        # Solve electric potential for frame f+1.
        direct_path = False
        if self._solver_type == "direct":
            if self._sim.requires_grad:
                # The direct solver does not yet support autodiff; fall back
                # to the Jacobi kernels so gradients still flow.
                self._jacobi_solve_v(f)
            else:
                self._solve_v_direct(f)
                direct_path = True
        elif self._sim.requires_grad:
            self._jacobi_solve_v(f)
        else:
            # Use the fused forward kernel to avoid Python dispatch overhead
            # for each Jacobi iteration.
            if self._dim == 2:
                self._solve_v_2d(f)
            else:
                self._solve_v_3d(f)

        # Compute current density and heat source from the new potential.
        # For the direct solver path, use the fused kernel that reads from
        # _V[f+1] directly, skipping the _V_tmp copy. Differentiable scenes
        # run the unfused pair so the recorded launches reverse one by one.
        if direct_path:
            if self._dim == 2:
                self._compute_jq_from_v_2d(f)
            else:
                self._compute_jq_from_v_3d(f)
        elif self._sim.requires_grad:
            if self._dim == 2:
                self._compute_j_2d()
                self._compute_q_2d(f)
            else:
                self._compute_j_3d()
                self._compute_q_3d(f)
        else:
            if self._dim == 2:
                self._compute_jq_2d(f)
            else:
                self._compute_jq_3d(f)

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

    def _jacobi_solve_v(self, f: int) -> None:
        """Unfixed Jacobi solve: one recorded launch per iteration."""
        self._copy_v_to_tmp(f)
        for _ in range(self._max_iter):
            if self._dim == 2:
                self._jacobi_step_v_2d()
            else:
                self._jacobi_step_v_3d()
        self._copy_tmp_to_v(f)

    def substep_pre_coupling_grad(self, f: int) -> None:
        """Manually reverse one recorded substep (plugin gradient API).

        genesis-world 1.4.6 no longer calls this from the simulator (a
        scene-level backward reverses the recorded launches itself); it is
        kept so callers can seed field adjoints by hand and pull gradients
        into the material fields, exactly mirroring the recorded sequence of
        'substep_pre_coupling' in reverse order.
        """
        # Back-propagate through internal thermal step or cross-solver injection.
        if self._couple_to_thermal:
            thermal = getattr(self._sim, "thermal_solver", None)
            if thermal is not None and thermal._T is not None:
                if self._dim == 2:
                    self._apply_q_to_thermal_2d.grad(f)
                else:
                    self._apply_q_to_thermal_3d.grad(f)
        elif self._T is not None:
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
        if self._boundary_voltage_field is not None:
            self._boundary_voltage_field.grad.fill(0.0)
        if self._rho_field is not None:
            self._rho_field.grad.fill(0.0)
        if self._cp_field is not None:
            self._cp_field.grad.fill(0.0)
        if self._k_field is not None:
            self._k_field.grad.fill(0.0)
        if self._r_field is not None:
            self._r_field.grad.fill(0.0)
        if self._dt_over_rhocp_field is not None:
            self._dt_over_rhocp_field.grad.fill(0.0)

    # State snapshots round-trip through numpy host arrays keyed by field;
    # get_state does not register autograd reads, so a scene-level backward
    # never routes gradients through 'add_grad_from_state' (it stays a no-op
    # kept only for base-class signature compatibility).

    @property
    def data(self):
        """Yield the potential and temperature fields for checkpointing."""
        if self._V is None:
            return
        yield ArrayItem("V", DataKind.STATE, self._V)
        if self._T is not None:
            yield ArrayItem("T", DataKind.STATE, self._T)

    def get_state(self, f: int) -> dict | None:
        if not self.is_active or self._V is None:
            return None
        # Fields are (frames, *grid, B); state dicts keep the public (B, *grid).
        state: dict[str, np.ndarray] = {
            "V": np.moveaxis(self._V.to_numpy()[0], -1, 0).copy()
        }
        if self._T is not None:
            state["T"] = np.moveaxis(self._T.to_numpy()[0], -1, 0).copy()
        return state

    def set_state(
        self,
        f: int,
        state: dict | None,
        envs_idx: np.ndarray | None = None,
    ) -> None:
        if state is None or self._V is None:
            return
        # envs_idx could be supported later; for now we copy all envs. Between
        # two steps the state sits in frame 0 (see 'Solver.has_frames').
        arr = self._V.to_numpy()
        arr[0] = np.moveaxis(state["V"], 0, -1)
        self._V.from_numpy(arr)
        if self._T is not None and "T" in state:
            t_arr = self._T.to_numpy()
            t_arr[0] = np.moveaxis(state["T"], 0, -1)
            self._T.from_numpy(t_arr)

    def add_grad_from_state(self, grads, selection) -> None:
        return

    # --------------------------------------------------------------------------
    # Quadrants kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _copy_v_to_tmp_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V_tmp[i_b, i, j] = self._V[f, i, j, i_b]

    @qd.kernel
    def _copy_v_to_tmp_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V_tmp[i_b, i, j, k] = self._V[f, i, j, k, i_b]

    def _copy_v_to_tmp(self, f: int) -> None:
        if self._dim == 2:
            self._copy_v_to_tmp_2d(f)
        else:
            self._copy_v_to_tmp_3d(f)

    @qd.kernel
    def _copy_tmp_to_v_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[f + 1, i, j, i_b] = self._V_tmp[i_b, i, j]

    @qd.kernel
    def _copy_tmp_to_v_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[f + 1, i, j, k, i_b] = self._V_tmp[i_b, i, j, k]

    def _copy_tmp_to_v(self, f: int) -> None:
        if self._dim == 2:
            self._copy_tmp_to_v_2d(f)
        else:
            self._copy_tmp_to_v_3d(f)

    @qd.kernel
    def _solve_v_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused forward Jacobi solve: copy -> iterate -> copy back.

        This replaces the Python-level loop of ``_copy_v_to_tmp``,
        ``_jacobi_step_v_2d`` (``max_iter`` times), and ``_copy_tmp_to_v``
        with a single kernel launch, removing Python dispatch overhead for
        long rollouts. The per-iteration kernels are kept for the backward
        pass, which cannot be fused due to same-field in-place read/write
        adjoint limitations.
        """
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V_tmp[i_b, i, j] = self._V[f, i, j, i_b]
        for _ in range(self._max_iter):
            for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
                if self._voltage_boundary_mask[i_b, i, j] == 1:
                    self._V_tmp[i_b, i, j] = self._boundary_voltage_field[i_b, i, j]
                else:
                    im = i - 1 if i > 0 else i
                    ip = i + 1 if i < self._nx - 1 else i
                    jm = j - 1 if j > 0 else j
                    jp = j + 1 if j < self._ny - 1 else j
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
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[f + 1, i, j, i_b] = self._V_tmp[i_b, i, j]

    @qd.kernel
    def _jacobi_step_v_2d(self):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if self._voltage_boundary_mask[i_b, i, j] == 1:
                # Dirichlet node: enforce the boundary voltage.
                self._V_tmp[i_b, i, j] = self._boundary_voltage_field[i_b, i, j]
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
    def _solve_v_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused forward Jacobi solve for 3D (see ``_solve_v_2d``)."""
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V_tmp[i_b, i, j, k] = self._V[f, i, j, k, i_b]
        for _ in range(self._max_iter):
            for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
                if self._voltage_boundary_mask[i_b, i, j, k] == 1:
                    self._V_tmp[i_b, i, j, k] = self._boundary_voltage_field[
                        i_b, i, j, k
                    ]
                else:
                    im = i - 1 if i > 0 else i
                    ip = i + 1 if i < self._nx - 1 else i
                    jm = j - 1 if j > 0 else j
                    jp = j + 1 if j < self._ny - 1 else j
                    km = k - 1 if k > 0 else k
                    kp = k + 1 if k < self._nz - 1 else k
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
                        sigma_x
                        + sigma_xm
                        + sigma_y
                        + sigma_ym
                        + sigma_z
                        + sigma_zm
                        + 1e-10
                    )
                    self._V_tmp[i_b, i, j, k] = (
                        sigma_x * self._V_tmp[i_b, ip, j, k]
                        + sigma_xm * self._V_tmp[i_b, im, j, k]
                        + sigma_y * self._V_tmp[i_b, i, jp, k]
                        + sigma_ym * self._V_tmp[i_b, i, jm, k]
                        + sigma_z * self._V_tmp[i_b, i, j, kp]
                        + sigma_zm * self._V_tmp[i_b, i, j, km]
                    ) / denom
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[f + 1, i, j, k, i_b] = self._V_tmp[i_b, i, j, k]

    @qd.kernel
    def _jacobi_step_v_3d(self):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            if self._voltage_boundary_mask[i_b, i, j, k] == 1:
                self._V_tmp[i_b, i, j, k] = self._boundary_voltage_field[i_b, i, j, k]
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
                self._Q[f, i, j, i_b] = gs.qd_float(0.0)
            else:
                j_vec = self._J[i_b, i, j]
                jx = j_vec[0]
                jy = j_vec[1]
                sigma = self._sigma[i_b, i, j]
                # Q = |J|^2 / sigma
                self._Q[f, i, j, i_b] = (jx * jx + jy * jy) / (sigma + 1e-10)

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
                self._Q[f, i, j, k, i_b] = gs.qd_float(0.0)
            else:
                j_vec = self._J[i_b, i, j, k]
                jx = j_vec[0]
                jy = j_vec[1]
                jz = j_vec[2]
                sigma = self._sigma[i_b, i, j, k]
                self._Q[f, i, j, k, i_b] = (jx * jx + jy * jy + jz * jz) / (
                    sigma + 1e-10
                )

    @qd.kernel
    def _compute_jq_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused J and Q computation for the forward pass.

        Eliminates one kernel launch per substep by computing ``Q = |J|^2/sigma``
        directly from the voltage gradient without materializing ``_J``.
        The separate ``_compute_j_*`` / ``_compute_q_*`` kernels remain for the
        backward pass.
        """
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._J[i_b, i, j] = gs.qd_vec3(0.0, 0.0, 0.0)
                self._Q[f, i, j, i_b] = gs.qd_float(0.0)
            else:
                sigma = self._sigma[i_b, i, j]
                dvdx = (self._V_tmp[i_b, i + 1, j] - self._V_tmp[i_b, i - 1, j]) / (
                    2.0 * self._dx
                )
                dvdy = (self._V_tmp[i_b, i, j + 1] - self._V_tmp[i_b, i, j - 1]) / (
                    2.0 * self._dx
                )
                jx = -sigma * dvdx
                jy = -sigma * dvdy
                self._J[i_b, i, j] = gs.qd_vec3(jx, jy, 0.0)
                self._Q[f, i, j, i_b] = (jx * jx + jy * jy) / (sigma + 1e-10)

    @qd.kernel
    def _compute_jq_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused J and Q computation for 3D (see ``_compute_jq_2d``)."""
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
                self._Q[f, i, j, k, i_b] = gs.qd_float(0.0)
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
                jx = -sigma * dvdx
                jy = -sigma * dvdy
                jz = -sigma * dvdz
                self._J[i_b, i, j, k] = gs.qd_vec3(jx, jy, jz)
                self._Q[f, i, j, k, i_b] = (jx * jx + jy * jy + jz * jz) / (
                    sigma + 1e-10
                )

    @qd.kernel
    def _compute_jq_from_v_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused J+Q from ``_V[f+1]`` (skips ``_V_tmp`` copy for direct solver)."""
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                self._J[i_b, i, j] = gs.qd_vec3(0.0, 0.0, 0.0)
                self._Q[f, i, j, i_b] = gs.qd_float(0.0)
            else:
                sigma = self._sigma[i_b, i, j]
                dvdx = (
                    self._V[f + 1, i + 1, j, i_b] - self._V[f + 1, i - 1, j, i_b]
                ) / (2.0 * self._dx)
                dvdy = (
                    self._V[f + 1, i, j + 1, i_b] - self._V[f + 1, i, j - 1, i_b]
                ) / (2.0 * self._dx)
                jx = -sigma * dvdx
                jy = -sigma * dvdy
                self._J[i_b, i, j] = gs.qd_vec3(jx, jy, 0.0)
                self._Q[f, i, j, i_b] = (jx * jx + jy * jy) / (sigma + 1e-10)

    @qd.kernel
    def _compute_jq_from_v_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        """Fused J+Q from ``_V[f+1]`` for 3D (see ``_compute_jq_from_v_2d``)."""
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
                self._Q[f, i, j, k, i_b] = gs.qd_float(0.0)
            else:
                sigma = self._sigma[i_b, i, j, k]
                dvdx = (
                    self._V[f + 1, i + 1, j, k, i_b] - self._V[f + 1, i - 1, j, k, i_b]
                ) / (2.0 * self._dx)
                dvdy = (
                    self._V[f + 1, i, j + 1, k, i_b] - self._V[f + 1, i, j - 1, k, i_b]
                ) / (2.0 * self._dx)
                dvdz = (
                    self._V[f + 1, i, j, k + 1, i_b] - self._V[f + 1, i, j, k - 1, i_b]
                ) / (2.0 * self._dx)
                jx = -sigma * dvdx
                jy = -sigma * dvdy
                jz = -sigma * dvdz
                self._J[i_b, i, j, k] = gs.qd_vec3(jx, jy, jz)
                self._Q[f, i, j, k, i_b] = (jx * jx + jy * jy + jz * jz) / (
                    sigma + 1e-10
                )

    @qd.kernel
    def _apply_q_to_thermal_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._sim.thermal_solver._T[f + 1, i, j, i_b] += (
                self._Q[f, i, j, i_b]
                * self._substep_dt
                / (self._rho_field[i_b] * self._cp_field[i_b])
            )

    @qd.kernel
    def _apply_q_to_thermal_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._sim.thermal_solver._T[f + 1, i, j, k, i_b] += (
                self._Q[f, i, j, k, i_b]
                * self._substep_dt
                / (self._rho_field[i_b] * self._cp_field[i_b])
            )

    @qd.kernel
    def _heat_step_2d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        # NOTE: quadrants 1.3 rejects field aliases inside kernels
        # (`r_env = self._r_field`); index the fields directly instead.
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            t_old = self._T[f, i, j, i_b]
            rho = self._rho_field[i_b]
            cp = self._cp_field[i_b]
            # Inlined from _init_thermal_constants so autodiff reaches the
            # scalar material fields (host-precomputed constants would not).
            r = self._k_field[i_b] / (rho * cp) * self._dt_over_dx2
            heating = self._Q[f, i, j, i_b] * self._substep_dt / (rho * cp)
            if i == 0 or i == self._nx - 1 or j == 0 or j == self._ny - 1:
                im = qd.max(i - 1, 0)
                ip = qd.min(i + 1, self._nx - 1)
                jm = qd.max(j - 1, 0)
                jp = qd.min(j + 1, self._ny - 1)
                lap = (
                    self._T[f, ip, j, i_b]
                    + self._T[f, im, j, i_b]
                    + self._T[f, i, jp, i_b]
                    + self._T[f, i, jm, i_b]
                    - 4.0 * t_old
                )
                self._T[f + 1, i, j, i_b] = t_old + r * lap + heating
            else:
                lap = (
                    self._T[f, i + 1, j, i_b]
                    + self._T[f, i - 1, j, i_b]
                    + self._T[f, i, j + 1, i_b]
                    + self._T[f, i, j - 1, i_b]
                    - 4.0 * t_old
                )
                self._T[f + 1, i, j, i_b] = t_old + r * lap + heating

    @qd.kernel
    def _heat_step_3d(self, f: qd.i32):  # type: ignore[no-untyped-def]
        # NOTE: quadrants 1.3 rejects field aliases inside kernels
        # (`r_env = self._r_field`); index the fields directly instead.
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            t_old = self._T[f, i, j, k, i_b]
            rho = self._rho_field[i_b]
            cp = self._cp_field[i_b]
            # Inlined from _init_thermal_constants so autodiff reaches the
            # scalar material fields (host-precomputed constants would not).
            r = self._k_field[i_b] / (rho * cp) * self._dt_over_dx2
            heating = self._Q[f, i, j, k, i_b] * self._substep_dt / (rho * cp)
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
                    self._T[f, ip, j, k, i_b]
                    + self._T[f, im, j, k, i_b]
                    + self._T[f, i, jp, k, i_b]
                    + self._T[f, i, jm, k, i_b]
                    + self._T[f, i, j, kp, i_b]
                    + self._T[f, i, j, km, i_b]
                    - 6.0 * t_old
                )
                self._T[f + 1, i, j, k, i_b] = t_old + r * lap + heating
            else:
                lap = (
                    self._T[f, i + 1, j, k, i_b]
                    + self._T[f, i - 1, j, k, i_b]
                    + self._T[f, i, j + 1, k, i_b]
                    + self._T[f, i, j - 1, k, i_b]
                    + self._T[f, i, j, k + 1, i_b]
                    + self._T[f, i, j, k - 1, i_b]
                    - 6.0 * t_old
                )
                self._T[f + 1, i, j, k, i_b] = t_old + r * lap + heating

    # --------------------------------------------------------------------------
    # State / checkpoint kernels
    # --------------------------------------------------------------------------

    @qd.kernel
    def _copy_frame_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._V[target, i, j, i_b] = self._V[source, i, j, i_b]

    @qd.kernel
    def _copy_frame_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._V[target, i, j, k, i_b] = self._V[source, i, j, k, i_b]

    def copy_frame(self, source: int, target: int) -> None:
        """Roll the framed fields (the simulator drives this generically)."""
        if self._V is None:
            return
        if self._dim == 2:
            self._copy_frame_2d(source, target)
        else:
            self._copy_frame_3d(source, target)
        if self._T is not None:
            if self._dim == 2:
                self._copy_t_frame_2d(source, target)
            else:
                self._copy_t_frame_3d(source, target)

    @qd.kernel
    def _copy_t_frame_2d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, i_b in qd.ndrange(self._nx, self._ny, self._B):
            self._T[target, i, j, i_b] = self._T[source, i, j, i_b]

    @qd.kernel
    def _copy_t_frame_3d(self, source: qd.i32, target: qd.i32):  # type: ignore[no-untyped-def]
        for i, j, k, i_b in qd.ndrange(self._nx, self._ny, self._nz, self._B):
            self._T[target, i, j, k, i_b] = self._T[source, i, j, k, i_b]
