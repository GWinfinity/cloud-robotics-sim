"""Weakly-compressible 3D jet/plume solver for nozzle-exhaust coupling.

Genesis-free pure-PyTorch solver on a MAC staggered grid, structurally
mirroring ``cfd_coupling.core.cfd3d`` but for *hot compressible exhaust into
an open plenum* instead of isothermal incompressible flow.

Model: anelastic low-Mach formulation.

* Density is diagnostic from the equation of state at the (fixed)
  thermodynamic pressure ``p_ambient``: ``rho = p_ambient / (R_mix T)``.
  Large hot/cold density ratios (typical exhaust plumes) are handled
  without an acoustic CFL limit — the time step is advection/diffusion
  limited only.
* Energy equation (advection + conduction, variable ``rho``) gives the
  thermal-expansion divergence ``D = (1/T) DT/Dt``; the projection enforces
  ``div(u) = D`` through a variable-coefficient pressure solve
  ``div((1/rho) grad p') = (div(u*) - D)/dt`` with frozen ``rho``.
* Condensed-phase mass fraction ``alpha`` enters as a passive transported
  scalar and sets the mixture gas constant ``R_mix = (1 - alpha) R_gas``
  (supplied by the 1D nozzle side as ``r_eff``).
* Inlet BC (x = 0 patch) is a *mass-flux* condition:
  ``u_in = mdot / (rho_in A_patch)`` with ``rho_in`` from the stagnation
  temperature; the rest of the face is a wall.
* Outlet BC (x = Lx) is p = 0 (gauge) Dirichlet on a patch; restricting the
  patch gives the plenum a hydraulic resistance, i.e. the backpressure that
  the 3D side feeds back to the 1D nozzle.

Probes for the coupling: ``exit_plane_pressure()`` (absolute static pressure
averaged over the inlet-adjacent patch cells — the 3D -> 1D backpressure),
``inlet_mass_flow()`` / ``outlet_mass_flow()`` / ``domain_mass()`` for mass
bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from plugins.solvers.cfd_coupling.core.cfd3d import (
    _mirror_pad,
    _replicate_pad,
)


@dataclass
class JetOptions:
    """Options for the weakly-compressible 3D jet solver."""

    domain: tuple[float, float, float] = (1.5, 0.3, 0.3)  # Lx, Ly, Lz [m]
    cells: tuple[int, int, int] = (60, 12, 12)  # nx, ny, nz
    viscosity: float = 2.0e-5  # kinematic viscosity (momentum diffusion) [m^2/s]
    gas_constant: float = 320.0  # exhaust gas constant R = r_eff [J/(kg K)]
    ambient_pressure: float = 101325.0  # thermodynamic pressure [Pa]
    ambient_temperature: float = 300.0  # initial / ambient temperature [K]
    inlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (0.25, 0.75),
    )  # (y0, y1), (z0, z1) as fractions of Ly / Lz on the x=0 face
    outlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )  # subset of the x=Lx face that is open; the rest is a wall
    inlet_temperature: float = 300.0  # inlet stagnation temperature [K]
    cp: float = 1600.0  # mixture specific heat at constant pressure [J/(kg K)]
    conductivity: float = 0.08  # thermal conductivity k [W/(m K)]
    phase_diffusivity: float = 1.0e-5  # condensed-fraction scalar diffusion [m^2/s]
    cg_tol: float = 1e-4  # relative residual on the pressure rhs
    cg_constraint_tol: float = 1e-2  # absolute divergence-constraint residual
    # |div(u) - D| [1/s] the CG also enforces (dt-scaled rhs residual)
    cg_max_iters: int = 600
    pressure_solver: str = "auto"  # 'auto' (CG + cached const-coefficient LU
    # preconditioner) | 'direct' (exact sparse LU rebuilt every step, small
    # grids only) | 'cg' (plain Jacobi-preconditioned CG)
    device: str = "cpu"


class Jet3D:
    """Anelastic weakly-compressible NS solver on a uniform MAC grid."""

    def __init__(self, options: JetOptions) -> None:
        o = options
        nx, ny, nz = o.cells
        Lx, Ly, Lz = o.domain
        self.o = o
        self.device = torch.device(o.device)
        self.h = Lx / nx
        if abs(Ly / ny - self.h) > 1e-9 or abs(Lz / nz - self.h) > 1e-9:
            raise ValueError("cells/domain must give a uniform grid spacing")
        self.nx, self.ny, self.nz = nx, ny, nz

        dev = self.device
        self.u = torch.zeros(nx + 1, ny, nz, device=dev)  # x-faces
        self.v = torch.zeros(nx, ny + 1, nz, device=dev)  # y-faces
        self.w = torch.zeros(nx, ny, nz + 1, device=dev)  # z-faces
        self.p = torch.zeros(nx, ny, nz, device=dev)  # gauge pressure, cell centres
        self.T = torch.full((nx, ny, nz), o.ambient_temperature, device=dev)
        self.alpha = torch.zeros(nx, ny, nz, device=dev)  # condensed fraction

        self.t = 0.0
        self.n_steps = 0
        self._mdot = 0.0  # prescribed inlet mass flux [kg/s]
        self.T_in = o.inlet_temperature
        self.alpha_in = 0.0
        self.r_mix = o.gas_constant
        self.U_in = 0.0  # patch inlet velocity derived from the mass flux

        # Face masks on the (ny, nz) face grid.
        (y0, y1), (z0, z1) = o.inlet_patch
        yc = (torch.arange(ny, device=dev, dtype=torch.float64) + 0.5) / ny
        zc = (torch.arange(nz, device=dev, dtype=torch.float64) + 0.5) / nz
        ym = ((yc >= y0) & (yc < y1)).unsqueeze(1)
        zm = ((zc >= z0) & (zc < z1)).unsqueeze(0)
        self.inlet_mask_f = (ym & zm).to(dev)
        (y0o, y1o), (z0o, z1o) = o.outlet_patch
        ymo = ((yc >= y0o) & (yc < y1o)).unsqueeze(1)
        zmo = ((zc >= z0o) & (zc < z1o)).unsqueeze(0)
        self.outlet_mask_f = (ymo & zmo).to(dev)
        self.outlet_mask1 = self.outlet_mask_f.unsqueeze(0)  # (1, ny, nz)
        self._patch_cell_count = int(self.inlet_mask_f.sum())

        self._use_direct = o.pressure_solver == "direct"
        self._direct_lu = None
        self._const_lu = None  # constant-coefficient LU preconditioner
        self._cg_iters = 0
        self.n_lu_rebuilds = 0

        # Variable-coefficient pressure operator, cached symbolic pattern.
        # Entries: in-domain cell pairs (face coefficient beta = 1/rho), plus
        # outlet Dirichlet cells (diag + 2*beta, no off-diagonal).
        self._build_operator_pattern()

    # ------------------------------------------------------------------ #
    # Pressure-operator symbolic pattern (rebuilt numerically every step)
    # ------------------------------------------------------------------ #
    def _cell_index(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def _build_operator_pattern(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        rows: list[int] = []
        cols: list[int] = []
        pa: list[int] = []  # pair cell a (also diag cell)
        pb: list[int] = []  # pair cell b (-1 for outlet Dirichlet diag)
        dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        outlet = self.outlet_mask_f.cpu().numpy()
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    c = self._cell_index(i, j, k)
                    for di, dj, dk in dirs:
                        ii, jj, kk = i + di, j + dj, k + dk
                        if ii < nx and jj < ny and kk < nz:
                            n = self._cell_index(ii, jj, kk)
                            rows.append(c)
                            cols.append(n)
                            pa.append(c)
                            pb.append(n)
                            rows.append(n)
                            cols.append(c)
                            pa.append(n)
                            pb.append(c)
                        elif di == 1 and outlet[j, k]:
                            # x+ boundary on the open outlet patch: Dirichlet.
                            # (only the +x direction can leave the domain this
                            # way; y/z overflows above are plain Neumann walls)
                            rows.append(c)
                            cols.append(c)
                            pa.append(c)
                            pb.append(-1)
        self._op_rows = torch.tensor(rows, dtype=torch.long)
        self._op_cols = torch.tensor(cols, dtype=torch.long)
        self._op_pa = torch.tensor(pa, dtype=torch.long)
        self._op_pb = torch.tensor(pb, dtype=torch.long)  # -1 = dirichlet diag

    def set_outlet_patch(
        self, patch: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        """Change the open-outlet subset of the x=Lx face at runtime.

        Narrowing the outlet raises the plenum hydraulic resistance — the
        mechanism behind the 3D -> 1D backpressure signal (throttling event).
        Rebuilds the pressure-operator pattern and invalidates the cached
        constant-coefficient LU preconditioner.
        """
        (y0o, y1o), (z0o, z1o) = patch
        yc = (
            torch.arange(self.ny, device=self.device, dtype=torch.float64) + 0.5
        ) / self.ny
        zc = (
            torch.arange(self.nz, device=self.device, dtype=torch.float64) + 0.5
        ) / self.nz
        ymo = ((yc >= y0o) & (yc < y1o)).unsqueeze(1)
        zmo = ((zc >= z0o) & (zc < z1o)).unsqueeze(0)
        self.outlet_mask_f = (ymo & zmo).to(self.device)
        self.outlet_mask1 = self.outlet_mask_f.unsqueeze(0)
        self._build_operator_pattern()
        self._const_lu = None
        self._direct_lu = None

    # ------------------------------------------------------------------ #
    # Thermodynamics helpers
    # ------------------------------------------------------------------ #
    def density(self, temp: torch.Tensor | None = None) -> torch.Tensor:
        """Diagnostic density from the anelastic EOS [kg/m^3]."""
        t_field = self.T if temp is None else temp
        return self.o.ambient_pressure / (self.r_mix * t_field.clamp(min=1.0))

    def inlet_density(self) -> float:
        return self.o.ambient_pressure / (self.r_mix * max(self.T_in, 1.0))

    def set_inlet(
        self,
        mass_flux: float,
        stagnation_temp: float | None = None,
        condensed_fraction: float | None = None,
        r_eff: float | None = None,
    ) -> None:
        """Prescribe the inlet mass flux [kg/s] (+ stagnation T, phase, R_mix).

        This is the 1D -> 3D boundary exchange: the quasi-1D nozzle delivers
        its exit-plane mass flux, temperature and two-phase parameters, which
        are converted into a patch inlet velocity via the anelastic EOS.
        """
        self._mdot = float(mass_flux)
        if stagnation_temp is not None:
            self.T_in = float(stagnation_temp)
        if condensed_fraction is not None:
            self.alpha_in = float(np.clip(condensed_fraction, 0.0, 1.0))
        if r_eff is not None:
            self.r_mix = float(r_eff)
        self.U_in = self._mdot / (self.inlet_density() * self.inlet_patch_area())

    # ------------------------------------------------------------------ #
    # Boundary conditions / transport helpers
    # ------------------------------------------------------------------ #
    def _enforce_bc(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> None:
        u[0] = self.U_in * self.inlet_mask_f
        u[-1] = u[-1] * self.outlet_mask_f
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0

    @staticmethod
    def _upwind(
        fp: torch.Tensor, uf: torch.Tensor, vf: torch.Tensor, wf: torch.Tensor, h: float
    ) -> torch.Tensor:
        c = fp[1:-1, 1:-1, 1:-1]
        ax = torch.where(uf >= 0, c - fp[:-2, 1:-1, 1:-1], fp[2:, 1:-1, 1:-1] - c) / h
        ay = torch.where(vf >= 0, c - fp[1:-1, :-2, 1:-1], fp[1:-1, 2:, 1:-1] - c) / h
        az = torch.where(wf >= 0, c - fp[1:-1, 1:-1, :-2], fp[1:-1, 1:-1, 2:] - c) / h
        return uf * ax + vf * ay + wf * az

    @staticmethod
    def _laplacian(fp: torch.Tensor, h: float) -> torch.Tensor:
        c = fp[1:-1, 1:-1, 1:-1]
        return (
            fp[2:, 1:-1, 1:-1]
            + fp[:-2, 1:-1, 1:-1]
            + fp[1:-1, 2:, 1:-1]
            + fp[1:-1, :-2, 1:-1]
            + fp[1:-1, 1:-1, 2:]
            + fp[1:-1, 1:-1, :-2]
            - 6.0 * c
        ) / (h * h)

    def _component_pads(self, f: torch.Tensor, comp: str) -> torch.Tensor:
        if comp == "u":
            fp = _replicate_pad(f, 0)
            fp = _mirror_pad(fp, 1)
            fp = _mirror_pad(fp, 2)
        elif comp == "v":
            fp = _mirror_pad(f, 0)
            fp = _replicate_pad(fp, 1)
            fp = _mirror_pad(fp, 2)
        else:
            fp = _mirror_pad(f, 0)
            fp = _mirror_pad(fp, 1)
            fp = _replicate_pad(fp, 2)
        return fp

    def _cross_velocities(self):
        u, v, w = self.u, self.v, self.w
        vpx = _replicate_pad(v, 0)
        wpx = _replicate_pad(w, 0)
        V_u = 0.25 * (
            vpx[:-1, :-1, :] + vpx[1:, :-1, :] + vpx[:-1, 1:, :] + vpx[1:, 1:, :]
        )
        W_u = 0.25 * (
            wpx[:-1, :, :-1] + wpx[1:, :, :-1] + wpx[:-1, :, 1:] + wpx[1:, :, 1:]
        )
        upy = _mirror_pad(u, 1)
        wpy = _mirror_pad(w, 1)
        U_v = 0.25 * (
            upy[:-1, :-1, :] + upy[1:, :-1, :] + upy[:-1, 1:, :] + upy[1:, 1:, :]
        )
        W_v = 0.25 * (
            wpy[:, :-1, :-1] + wpy[:, :-1, 1:] + wpy[:, 1:, :-1] + wpy[:, 1:, 1:]
        )
        upz = _mirror_pad(u, 2)
        vpz = _mirror_pad(v, 2)
        U_w = 0.25 * (
            upz[:-1, :, :-1] + upz[1:, :, :-1] + upz[:-1, :, 1:] + upz[1:, :, 1:]
        )
        V_w = 0.25 * (vpz[:, :-1, :-1] + vpz[:, 1:, :-1] + vpz[:, :-1, 1:])
        return (u, V_u, W_u), (U_v, v, W_v), (U_w, V_w, w)

    # ------------------------------------------------------------------ #
    # Scalar transport (temperature / phase fraction)
    # ------------------------------------------------------------------ #
    def _scalar_field(
        self, f: torch.Tensor, f_in: float, dt: float, diffusivity: torch.Tensor
    ) -> torch.Tensor:
        """Advect (upwind) + diffuse one cell-centred scalar with a Dirichlet
        inlet ghost and zero-gradient elsewhere; returns the updated field.
        """
        h = self.h
        U_c = 0.5 * (self.u[:-1] + self.u[1:])
        V_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        W_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        left = torch.full_like(f[:1], f_in)
        fp = torch.cat([left, f, f[-1:]], dim=0)
        fp = _replicate_pad(fp, 1)
        fp = _replicate_pad(fp, 2)
        # Variable-coefficient diffusion in conservative face-flux form.
        dpad = torch.cat(
            [
                torch.full_like(diffusivity[:1], float(diffusivity[0].mean())),
                diffusivity,
                diffusivity[-1:],
            ],
            dim=0,
        )
        dpad = _replicate_pad(dpad, 1)
        dpad = _replicate_pad(dpad, 2)
        fc = fp[1:-1, 1:-1, 1:-1]
        flux = (
            0.5
            * (dpad[1:-1, 1:-1, 1:-1] + dpad[2:, 1:-1, 1:-1])
            * (fp[2:, 1:-1, 1:-1] - fc)
            - 0.5
            * (dpad[:-2, 1:-1, 1:-1] + dpad[1:-1, 1:-1, 1:-1])
            * (fc - fp[:-2, 1:-1, 1:-1])
            + 0.5
            * (dpad[1:-1, 1:-1, 1:-1] + dpad[1:-1, 2:, 1:-1])
            * (fp[1:-1, 2:, 1:-1] - fc)
            - 0.5
            * (dpad[1:-1, :-2, 1:-1] + dpad[1:-1, 1:-1, 1:-1])
            * (fc - fp[1:-1, :-2, 1:-1])
            + 0.5
            * (dpad[1:-1, 1:-1, 1:-1] + dpad[1:-1, 1:-1, 2:])
            * (fp[1:-1, 1:-1, 2:] - fc)
            - 0.5
            * (dpad[1:-1, 1:-1, :-2] + dpad[1:-1, 1:-1, 1:-1])
            * (fc - fp[1:-1, 1:-1, :-2])
        ) / (h * h)
        return f - dt * (self._upwind(fp, U_c, V_c, W_c, h) - flux)

    # ------------------------------------------------------------------ #
    # Pressure solve: div((1/rho) grad p) = rhs via -A p = rhs, A SPD
    # ------------------------------------------------------------------ #
    def _operator_data(
        self, inv_rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair face coefficients and per-cell diagonal of -div(beta grad)."""
        beta = inv_rho.reshape(-1)
        pa = self._op_pa.to(self.device)
        pb = self._op_pb.to(self.device)
        bf = 0.5 * (beta[pa] + beta[pb.clamp(min=0)])
        bf = torch.where(pb < 0, 2.0 * beta[pa], bf)
        diag = torch.zeros_like(beta)
        diag.index_add_(0, pa, bf)
        return bf, diag

    def _solve_pressure(
        self, rhs: torch.Tensor, inv_rho: torch.Tensor, dt: float
    ) -> torch.Tensor:
        if self._use_direct:
            return self._solve_pressure_direct(rhs, inv_rho)
        if self.o.pressure_solver == "cg":
            return self._solve_pressure_cg(rhs, inv_rho, dt, None)
        return self._solve_pressure_cg(rhs, inv_rho, dt, self._const_coefficient_lu())

    def _const_coefficient_lu(self):
        """LU of the unit-coefficient operator, cached (preconditioner only).

        The variable-coefficient operator only departs from it by the (order
        unity, slowly varying) factor 1/rho, so it is an excellent
        preconditioner and never needs rebuilding.
        """
        if self._const_lu is None:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            h2 = self.h * self.h
            n = self.nx * self.ny * self.nz
            ones = torch.ones(n, dtype=torch.float64)
            bf = 0.5 * (ones[self._op_pa] + ones[self._op_pb.clamp(min=0)])
            bf = torch.where(self._op_pb < 0, 2.0 * ones[self._op_pa], bf)
            diag = torch.zeros_like(ones)
            diag.index_add_(0, self._op_pa, bf)
            offdiag = torch.where(self._op_pb < 0, torch.zeros_like(bf), -bf) / h2
            vals = torch.cat([offdiag, diag / h2])
            rows = torch.cat([self._op_rows, torch.arange(n, dtype=torch.long)])
            cols = torch.cat([self._op_cols, torch.arange(n, dtype=torch.long)])
            mat = sp.csr_matrix(
                (vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n, n)
            )
            self._const_lu = spla.splu(mat.tocsc())
            self.n_lu_rebuilds += 1
        return self._const_lu

    def _solve_pressure_direct(
        self, rhs: torch.Tensor, inv_rho: torch.Tensor
    ) -> torch.Tensor:
        """Sparse-LU solve; the factorisation is rebuilt whenever the
        symbolic pattern is requested (rho-dependent coefficients change
        every step), still cheap at <=30k cells.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        h2 = self.h * self.h
        bf, diag = self._operator_data(inv_rho)
        n = self.nx * self.ny * self.nz
        # Dirichlet (outlet-ghost) entries contribute to the diagonal only.
        offdiag = torch.where(self._op_pb < 0, torch.zeros_like(bf), -bf) / h2
        vals = torch.cat([offdiag, diag / h2])
        rows = torch.cat([self._op_rows, torch.arange(n, dtype=torch.long)])
        cols = torch.cat([self._op_cols, torch.arange(n, dtype=torch.long)])
        mat = sp.csr_matrix(
            (vals.cpu().numpy(), (rows.numpy(), cols.numpy())), shape=(n, n)
        )
        lu = spla.splu(mat.tocsc())
        b = -(rhs.reshape(-1).cpu().numpy())
        x = lu.solve(b)
        self.p = torch.from_numpy(x.reshape(self.nx, self.ny, self.nz)).to(self.device)
        self._cg_iters = 0
        return self.p

    def _solve_pressure_cg(
        self,
        rhs: torch.Tensor,
        inv_rho: torch.Tensor,
        dt: float,
        precond=None,
    ) -> torch.Tensor:
        """CG on -div(beta grad) with warm start and an optional LU preconditioner.

        ``precond`` is a ``scipy SuperLU`` object of a nearby SPD operator;
        without one, Jacobi preconditioning is used. Convergence is judged on
        both the rhs-relative residual and the implied divergence-constraint
        residual ``|div(u_new) - D| = dt * |r_k|``.
        """
        h2 = self.h * self.h
        bf, diag = self._operator_data(inv_rho)
        pb = self._op_pb.to(self.device)
        rows = self._op_rows.to(self.device)
        cols = self._op_cols.to(self.device)
        pair_mask = pb >= 0
        p_rows, p_cols, p_bf = rows[pair_mask], cols[pair_mask], bf[pair_mask]
        d_rows, d_bf = rows[~pair_mask], bf[~pair_mask]
        minv = (h2 / diag).reshape(self.nx, self.ny, self.nz)

        def apply_op(p: torch.Tensor) -> torch.Tensor:
            pv = p.reshape(-1)
            acc = torch.zeros_like(pv)
            acc.index_add_(0, p_rows, -p_bf * pv[p_cols] / h2)
            acc.index_add_(0, p_rows, p_bf * pv[p_rows] / h2)
            # d_bf already carries the Dirichlet factor 2 (see _operator_data).
            acc.index_add_(0, d_rows, d_bf * pv[d_rows] / h2)
            return acc.reshape(p.shape)

        if precond is not None:

            def apply_minv(r: torch.Tensor) -> torch.Tensor:
                v = precond.solve(r.reshape(-1).cpu().numpy())
                out: torch.Tensor = torch.from_numpy(v).reshape(r.shape).to(self.device)
                return out

        else:

            def apply_minv(r: torch.Tensor) -> torch.Tensor:
                return r * minv

        b = -rhs
        x = self.p.clone()
        r = b - apply_op(x)
        z = apply_minv(r)
        d = z.clone()
        rz = float((r * z).sum())
        bnorm = float(b.abs().max()) + 1e-30
        r_abs_max = float(r.abs().max())
        iters = 0
        for iters in range(1, self.o.cg_max_iters + 1):
            # |div(u_new) - D| = dt * |r_k|: enforce the divergence constraint
            # absolutely; the relative criterion only floors quiescent states.
            if dt * r_abs_max < self.o.cg_constraint_tol:
                break
            if r_abs_max < 1e-6 * bnorm:
                break
            ad = apply_op(d)
            denom = float((d * ad).sum())
            if denom <= 1e-30:
                break
            alpha = rz / denom
            x = x + alpha * d
            r = r - alpha * ad
            z = apply_minv(r)
            rz_new = float((r * z).sum())
            beta = rz_new / max(rz, 1e-30)
            d = z + beta * d
            rz = rz_new
            r_abs_max = float(r.abs().max())
        self.p = x
        self._cg_iters = iters
        return x

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        """Advance one time step (explicit; dt limited by advection/diffusion)."""
        h = self.h
        nu = self.o.viscosity
        u, v, w = self.u, self.v, self.w
        self._enforce_bc(u, v, w)

        (U_u, V_u, W_u), (U_v, V_v, W_v), (U_w, V_w, W_w) = self._cross_velocities()
        u1 = u + dt * (
            -self._upwind(self._component_pads(u, "u"), U_u, V_u, W_u, h)
            + nu * self._laplacian(self._component_pads(u, "u"), h)
        )
        v1 = v + dt * (
            -self._upwind(self._component_pads(v, "v"), U_v, V_v, W_v, h)
            + nu * self._laplacian(self._component_pads(v, "v"), h)
        )
        w1 = w + dt * (
            -self._upwind(self._component_pads(w, "w"), U_w, V_w, W_w, h)
            + nu * self._laplacian(self._component_pads(w, "w"), h)
        )
        self._enforce_bc(u1, v1, w1)
        self.u, self.v, self.w = u1, v1, w1

        # Energy + phase transport and the thermal-expansion divergence target.
        rho = self.density()
        kappa = self.o.conductivity / (rho * self.o.cp)  # thermal diffusivity [m^2/s]
        T_new = self._scalar_field(self.T, self.T_in, dt, kappa)
        T_new = T_new.clamp(min=1.0)
        alpha_new = self._scalar_field(
            self.alpha,
            self.alpha_in,
            dt,
            torch.full_like(rho, self.o.phase_diffusivity),
        ).clamp(0.0, 1.0)
        D = ((T_new - self.T) / dt) / T_new  # div(u) target [1/s]
        self.T = T_new
        self.alpha = alpha_new
        self._last_D = D  # diagnostic: last projection divergence target

        # Variable-coefficient projection with the frozen density.
        inv_rho = 1.0 / self.density()
        div = (
            (self.u[1:] - self.u[:-1])
            + (self.v[:, 1:] - self.v[:, :-1])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / h
        rhs = (div - D) / dt
        p = self._solve_pressure(rhs, inv_rho, dt)

        # Correction with face beta = 0.5(1/rho_a + 1/rho_b).
        beta_c = inv_rho
        ppx = torch.cat(
            [p[:1], p, torch.where(self.outlet_mask1, -p[-1:], p[-1:])], dim=0
        )
        ppy = _replicate_pad(p, 1)
        ppz = _replicate_pad(p, 2)
        bx = 0.5 * (
            torch.cat([beta_c[:1], beta_c], dim=0)
            + torch.cat([beta_c, beta_c[-1:]], dim=0)
        )
        by = 0.5 * (
            torch.cat([beta_c[:, :1], beta_c], dim=1)
            + torch.cat([beta_c, beta_c[:, -1:]], dim=1)
        )
        bz = 0.5 * (
            torch.cat([beta_c[:, :, :1], beta_c], dim=2)
            + torch.cat([beta_c, beta_c[:, :, -1:]], dim=2)
        )
        self.u[1:] -= dt * bx[1:] * (ppx[2:] - ppx[1:-1]) / h
        self.v[:, 1:-1] -= dt * by[:, 1:-1] * (ppy[:, 2:-1] - ppy[:, 1:-2]) / h
        self.w[:, :, 1:-1] -= (
            dt * bz[:, :, 1:-1] * (ppz[:, :, 2:-1] - ppz[:, :, 1:-2]) / h
        )
        self._enforce_bc(self.u, self.v, self.w)

        self.t += dt
        self.n_steps += 1

    # ------------------------------------------------------------------ #
    # Coupling probes
    # ------------------------------------------------------------------ #
    def inlet_patch_area(self) -> float:
        return float(self.inlet_mask_f.sum()) * self.h * self.h

    def exit_plane_pressure(self) -> float:
        """Absolute static pressure at the nozzle exit plane [Pa].

        Gauge-pressure mean over the inlet-adjacent (x = h/2) patch cells plus
        the thermodynamic (ambient) pressure; this is the 3D -> 1D backpressure
        that drives the nozzle choking / shock response.
        """
        p_gauge = float((self.p[0] * self.inlet_mask_f).sum()) / max(
            self._patch_cell_count, 1
        )
        return self.o.ambient_pressure + p_gauge

    def inlet_mass_flow(self) -> float:
        """Mass flux through the inlet patch [kg/s]."""
        return (
            self.inlet_density()
            * float((self.u[0] * self.inlet_mask_f).sum())
            * self.h
            * self.h
        )

    def outlet_mass_flow(self) -> float:
        """Mass flux through the open outlet patch [kg/s]."""
        rho_e = self.density()[-1]
        return float((rho_e * self.u[-1] * self.outlet_mask_f).sum()) * self.h * self.h

    def domain_mass(self) -> float:
        """Total mass inside the domain [kg]."""
        return float(self.density().sum()) * self.h**3

    def divergence_norm(self) -> float:
        h = self.h
        div = (
            (self.u[1:] - self.u[:-1])
            + (self.v[:, 1:] - self.v[:, :-1])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / h
        return float(div.abs().max())

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "u": self.u.clone(),
            "v": self.v.clone(),
            "w": self.w.clone(),
            "p": self.p.clone(),
            "T": self.T.clone(),
            "alpha": self.alpha.clone(),
            "t": self.t,
            "n_steps": self.n_steps,
            "mdot": self._mdot,
            "T_in": self.T_in,
            "alpha_in": self.alpha_in,
            "r_mix": self.r_mix,
        }

    def set_state(self, state: dict) -> None:
        self.u, self.v, self.w, self.p, self.T, self.alpha = (
            state["u"],
            state["v"],
            state["w"],
            state["p"],
            state["T"],
            state["alpha"],
        )
        self.t = state["t"]
        self.n_steps = state["n_steps"]
        self._mdot = state["mdot"]
        self.T_in = state["T_in"]
        self.alpha_in = state["alpha_in"]
        self.r_mix = state["r_mix"]
        self.U_in = self._mdot / (self.inlet_density() * self.inlet_patch_area())
