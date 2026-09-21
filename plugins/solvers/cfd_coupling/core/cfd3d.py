"""3D incompressible Navier-Stokes solver on a MAC staggered grid.

Fractional-step (projection) method, all in PyTorch so the same code runs on
CPU and GPU:

    u* = u^n + dt * ( -upwind(u . grad u) + nu lap(u) )     (predictor)
    lap(p) = div(u*) / dt                                   (pressure Poisson)
    u^(n+1) = u* - dt * grad(p)                             (projection)

The pressure Poisson equation is pure Neumann (consistent because the net
inflow/outflow through the velocity BCs is zero for incompressible flow); the
null mode is removed by pinning p[0, 0, 0] = 0 and a Jacobi-preconditioned CG
with warm start from the previous step.

Boundary conditions
-------------------
* x = 0 face: inlet patch with prescribed velocity U_in(t) and temperature
  T_in(t); the rest of the face is a wall.
* x = Lx face: pressure Dirichlet p = 0 (gauge) - the projection then makes
  the outflow conserve mass exactly and lets the inlet-plane pressure react
  to flow changes (this is the backpressure the 1D side feels).
* y/z faces: no-slip walls.
* The pressure fed back to the 1D model is the inlet-plane mean gauge pressure
  converted to head [m] via p / (rho * g).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

G = 9.81  # gravitational acceleration [m/s^2]


def build_poisson_csr(
    nx: int,
    ny: int,
    nz: int,
    h: float,
    outlet_mask: np.ndarray,
) -> tuple[Any, np.ndarray]:
    """Assemble the negative-Laplacian operator (Neumann + outlet Dirichlet).

    Returns the CSR matrix and its diagonal. Shared by the torch and
    quadrants CFD backends: row ``(i*ny + j)*nz + k`` matches C-order
    raveling of an ``(nx, ny, nz)`` field.

    Ghost accounting per direction pair: both neighbours present -> diagonal
    2 each; a Neumann (replicate) ghost -> contributes 1; a Dirichlet ghost
    at the open outlet cells -> contributes 3 (1 + 2).
    """
    import scipy.sparse as sp

    h2 = h * h
    n = nx * ny * nz
    outlet = np.asarray(outlet_mask, dtype=bool).reshape(ny, nz)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.empty(n)

    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = idx(i, j, k)
                d = 0.0
                # x pair
                cnt = 0
                if i > 0:
                    rows.append(c)
                    cols.append(idx(i - 1, j, k))
                    vals.append(-1.0)
                    cnt += 1
                if i < nx - 1:
                    rows.append(c)
                    cols.append(idx(i + 1, j, k))
                    vals.append(-1.0)
                    cnt += 1
                elif outlet[j, k]:
                    d += 2.0  # Dirichlet ghost: 1 + 2 = 3
                d += cnt
                # y / z pairs: walls, always Neumann ghosts
                for jj, kk in ((j - 1, k), (j + 1, k), (j, k - 1), (j, k + 1)):
                    if 0 <= jj < ny and 0 <= kk < nz:
                        rows.append(c)
                        cols.append(idx(i, jj, kk))
                        vals.append(-1.0)
                        d += 1.0
                rows.append(c)
                cols.append(c)
                vals.append(d)
                diag[c] = d
    mat = sp.csr_matrix(
        (np.array(vals) / h2, (rows, cols)), shape=(n, n), dtype=np.float64
    )
    return mat, diag


def _replicate_pad(f: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad by replicating the boundary values (zero-gradient ghost)."""
    lo = f.narrow(dim, 0, 1)
    hi = f.narrow(dim, f.size(dim) - 1, 1)
    return torch.cat([lo, f, hi], dim=dim)


def _mirror_pad(f: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad by negating the boundary value (no-slip ghost: u_ghost = -u_wall)."""
    lo = -f.narrow(dim, 0, 1)
    hi = -f.narrow(dim, f.size(dim) - 1, 1)
    return torch.cat([lo, f, hi], dim=dim)


@dataclass
class CFDOptions:
    """Options for the 3D projection-method CFD solver."""

    domain: tuple[float, float, float] = (1.0, 0.5, 0.5)  # Lx, Ly, Lz [m]
    cells: tuple[int, int, int] = (48, 24, 24)  # nx, ny, nz
    viscosity: float = 1.0e-6  # kinematic viscosity [m^2/s]
    density: float = 1000.0  # fluid density [kg/m^3]
    inlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (0.25, 0.75),
    )  # (y0, y1), (z0, z1) as fractions of Ly / Lz on the x=0 face
    outlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )  # subset of the x=Lx face that is open; the rest is a wall. Restricting
    # the outlet gives the plenum a hydraulic resistance, so its pressure
    # reacts to the flow (this is the 3D -> 1D backpressure signal).
    inlet_temperature: float = 300.0  # prescribed inlet temperature [K]
    initial_temperature: float = 300.0
    thermal_diffusivity: float = 1.4e-7  # for the passive temperature field
    advect_temperature: bool = True
    cg_tol: float = 1e-4
    cg_max_iters: int = 400
    pressure_solver: str = "auto"  # 'auto' | 'direct' (sparse LU) | 'cg'
    device: str = "cpu"


class CFD3D:
    """Incompressible NS solver on a uniform MAC grid (u/v/w staggered)."""

    def __init__(self, options: CFDOptions) -> None:
        o = options
        self.o = o
        self.device = torch.device(o.device)
        nx, ny, nz = o.cells
        self.nx, self.ny, self.nz = nx, ny, nz
        Lx, Ly, Lz = o.domain
        # Uniform grid: require the cell size to be identical in all directions.
        self.h = Lx / nx
        if abs(Ly / ny - self.h) > 1e-9 or abs(Lz / nz - self.h) > 1e-9:
            raise ValueError("cells/domain must give a uniform grid spacing")

        dev = self.device
        self.u = torch.zeros(nx + 1, ny, nz, device=dev)  # x-faces
        self.v = torch.zeros(nx, ny + 1, nz, device=dev)  # y-faces
        self.w = torch.zeros(nx, ny, nz + 1, device=dev)  # z-faces
        self.p = torch.zeros(nx, ny, nz, device=dev)  # cell centres
        self.T = torch.full((nx, ny, nz), o.initial_temperature, device=dev)

        self.t = 0.0
        self.n_steps = 0
        self.U_in = 0.0  # prescribed inlet velocity [m/s]
        self.T_in = o.inlet_temperature

        # Inlet patch mask on the (ny, nz) face grid (face centres).
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

        self._cg_iters = 0
        self._direct_lu = None  # cached sparse LU factorisation
        self._use_direct = o.pressure_solver == "direct" or (
            o.pressure_solver == "auto" and nx * ny * nz <= 30000
        )

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #
    def _enforce_bc(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> None:
        """No-slip walls + prescribed inlet velocity + restricted outlet."""
        u[0] = self.U_in * self.inlet_mask_f
        u[-1] = u[-1] * self.outlet_mask_f  # non-patch outlet cells are walls
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0

    # ------------------------------------------------------------------ #
    # Advection / diffusion helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _upwind(fp: torch.Tensor, U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, h: float) -> torch.Tensor:
        """First-order upwind divergence of a padded field fp at its faces."""
        c = fp[1:-1, 1:-1, 1:-1]
        ax = torch.where(U >= 0, c - fp[:-2, 1:-1, 1:-1], fp[2:, 1:-1, 1:-1] - c) / h
        ay = torch.where(V >= 0, c - fp[1:-1, :-2, 1:-1], fp[1:-1, 2:, 1:-1] - c) / h
        az = torch.where(W >= 0, c - fp[1:-1, 1:-1, :-2], fp[1:-1, 1:-1, 2:] - c) / h
        return U * ax + V * ay + W * az

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
        """Pad a velocity component with its physical ghost rules.

        comp in {"u", "v", "w"}; tangential directions use no-slip mirror
        ghosts, the component's own staggered boundary faces are replicated.
        """
        if comp == "u":
            fp = _replicate_pad(f, 0)
            fp = _mirror_pad(fp, 1)
            fp = _mirror_pad(fp, 2)
        elif comp == "v":
            fp = _mirror_pad(f, 0)
            fp = _replicate_pad(fp, 1)
            fp = _mirror_pad(fp, 2)
        else:  # w
            fp = _mirror_pad(f, 0)
            fp = _mirror_pad(fp, 1)
            fp = _replicate_pad(fp, 2)
        return fp

    def _cross_velocities(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Interpolate cell-based velocities onto each component's faces."""
        u, v, w = self.u, self.v, self.w
        # x-padded cross components (for the u component)
        vpx = _replicate_pad(v, 0)
        wpx = _replicate_pad(w, 0)
        V_u = 0.25 * (
            vpx[:-1, :-1, :] + vpx[1:, :-1, :] + vpx[:-1, 1:, :] + vpx[1:, 1:, :]
        )
        W_u = 0.25 * (
            wpx[:-1, :, :-1] + wpx[1:, :, :-1] + wpx[:-1, :, 1:] + wpx[1:, :, 1:]
        )
        # y-padded cross components (for the v component)
        upy = _mirror_pad(u, 1)
        wpy = _mirror_pad(w, 1)
        U_v = 0.25 * (
            upy[:-1, :-1, :] + upy[1:, :-1, :] + upy[:-1, 1:, :] + upy[1:, 1:, :]
        )
        W_v = 0.25 * (
            wpy[:, :-1, :-1] + wpy[:, :-1, 1:] + wpy[:, 1:, :-1] + wpy[:, 1:, 1:]
        )
        # z-padded cross components (for the w component)
        upz = _mirror_pad(u, 2)
        vpz = _mirror_pad(v, 2)
        U_w = 0.25 * (
            upz[:-1, :, :-1] + upz[1:, :, :-1] + upz[:-1, :, 1:] + upz[1:, :, 1:]
        )
        V_w = 0.25 * (
            vpz[:, :-1, :-1] + vpz[:, 1:, :-1] + vpz[:, :-1, 1:] + vpz[:, 1:, 1:]
        )
        return (u, V_u, W_u), (U_v, v, W_v), (U_w, V_w, w)

    # ------------------------------------------------------------------ #
    # Pressure Poisson (Neumann, pinned) via preconditioned CG
    # ------------------------------------------------------------------ #
    def _poisson_rhs(self, dt: float) -> torch.Tensor:
        h = self.h
        div = (
            (self.u[1:] - self.u[:-1])
            + (self.v[:, 1:] - self.v[:, :-1])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / h
        return div / dt

    def _solve_pressure(self, rhs: torch.Tensor) -> torch.Tensor:
        if self._use_direct:
            return self._solve_pressure_direct(rhs)
        return self._solve_pressure_cg(rhs)

    def _solve_pressure_direct(self, rhs: torch.Tensor) -> torch.Tensor:
        """Exact sparse-LU solve (cached factorisation), for small grids."""
        import scipy.sparse.linalg as spla

        if self._direct_lu is None:
            mat, diag = build_poisson_csr(
                self.nx,
                self.ny,
                self.nz,
                self.h,
                self.outlet_mask_f.cpu().numpy(),
            )
            self._direct_lu = spla.splu(mat.tocsc())
            self._direct_diag = diag

        b = -(rhs.cpu().numpy().ravel())
        x = self._direct_lu.solve(b)
        self.p = torch.from_numpy(x.reshape(self.nx, self.ny, self.nz)).to(
            self.device
        )
        self._cg_iters = 0
        return self.p

    def _solve_pressure_cg(self, rhs: torch.Tensor) -> torch.Tensor:
        """Poisson solve: Neumann everywhere, Dirichlet p=0 at the outlet.

        Mixed Neumann/Dirichlet makes the operator symmetric positive definite
        (no null mode to pin), so plain Jacobi-preconditioned CG with warm
        start converges in a few tens of iterations.
        """
        h2 = self.h * self.h
        diag = torch.full_like(rhs, 6.0 / h2)
        diag[-1, :, :] = torch.where(
            self.outlet_mask_f,
            torch.full_like(self.outlet_mask_f, 7.0 / h2),
            torch.full_like(self.outlet_mask_f, 5.0 / h2),
        )
        minv = 1.0 / diag
        mask1 = self.outlet_mask1

        def A(p: torch.Tensor) -> torch.Tensor:
            """Negative Laplacian: Neumann ghosts; p=0 on the open outlet."""
            left = p.narrow(0, 0, 1)
            last = p.narrow(0, p.size(0) - 1, 1)
            right = torch.where(mask1, -last, last)  # Dirichlet on the patch
            px = torch.cat([left, p, right], dim=0)
            px = _replicate_pad(px, 1)
            px = _replicate_pad(px, 2)
            return -self._laplacian(px, self.h)

        b = -rhs
        x = self.p.clone()  # warm start
        r = b - A(x)
        z = r * minv
        d = z.clone()
        rz = float((r * z).sum())
        bnorm = float(b.abs().max()) + 1e-30
        iters = 0
        for iters in range(1, self.o.cg_max_iters + 1):
            if float(r.abs().max()) < self.o.cg_tol * bnorm:
                break
            ad = A(d)
            denom = float((d * ad).sum())
            if denom <= 1e-30:  # lost positive-definiteness; safeguard
                break
            alpha = rz / denom
            x = x + alpha * d
            r = r - alpha * ad
            z = r * minv
            rz_new = float((r * z).sum())
            beta = rz_new / max(rz, 1e-30)
            d = z + beta * d
            rz = rz_new
        self.p = x
        self._cg_iters = iters
        return x

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        """Advance one time step (explicit; dt must satisfy CFL limits)."""
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

        # Projection: grad(p) at faces; outlet ghost mirrors p=0 on the patch.
        p = self._solve_pressure(self._poisson_rhs(dt))
        p_last = p.narrow(0, p.size(0) - 1, 1)
        p_right = torch.where(self.outlet_mask1, -p_last, p_last)
        ppx = torch.cat([p.narrow(0, 0, 1), p, p_right], dim=0)
        ppy = _replicate_pad(p, 1)
        ppz = _replicate_pad(p, 2)
        self.u[1:] -= dt * (ppx[2:] - ppx[1:-1]) / h
        self.v[:, 1:-1] -= dt * (ppy[:, 2:-1] - ppy[:, 1:-2]) / h
        self.w[:, :, 1:-1] -= dt * (ppz[:, :, 2:-1] - ppz[:, :, 1:-2]) / h
        self._enforce_bc(self.u, self.v, self.w)

        if self.o.advect_temperature:
            self._advect_temperature(dt)

        self.t += dt
        self.n_steps += 1

    def _advect_temperature(self, dt: float) -> None:
        h = self.h
        U_c = 0.5 * (self.u[:-1] + self.u[1:])
        V_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        W_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        T = self.T
        # Dirichlet ghost at the inlet (x = 0), zero-gradient elsewhere.
        left = torch.full_like(T[:1], self.T_in)
        right = T[-1:]
        fp = torch.cat([left, T, right], dim=0)
        fp = _replicate_pad(fp, 1)
        fp = _replicate_pad(fp, 2)
        self.T = T - dt * (
            self._upwind(fp, U_c, V_c, W_c, h)
            - self.o.thermal_diffusivity * self._laplacian(fp, h)
        )

    # ------------------------------------------------------------------ #
    # Coupling probes
    # ------------------------------------------------------------------ #
    def set_inlet(self, velocity: float, temperature: float | None = None) -> None:
        """Prescribe inlet velocity [m/s] (and optionally temperature [K])."""
        self.U_in = float(velocity)
        if temperature is not None:
            self.T_in = float(temperature)

    def inlet_patch_area(self) -> float:
        return float(self.inlet_mask_f.sum()) * self.h * self.h

    def inlet_flow(self) -> float:
        """Integrated inflow through the inlet patch [m^3/s]."""
        return float((self.u[0] * self.inlet_mask_f).sum()) * self.h * self.h

    def outlet_flow(self) -> float:
        """Integrated outflow through the x=Lx face [m^3/s]."""
        return float(self.u[-1].sum()) * self.h * self.h

    def inlet_pressure_head(self) -> float:
        """Plenum static pressure probe, as head [m] (p / rho g).

        Volume mean over all cells: the jet core is pressure-matched and
        reads ~0, while the pressure build-up near a restricted outlet is
        localised; the volume mean isolates the chamber static pressure that
        hydraulically pushes back on the 1D nozzle.

        Note: the Poisson solve yields the specific pressure p/rho [m^2/s^2],
        so the head is p_mean / g (no extra density factor).
        """
        return float(self.p.mean()) / G

    def divergence_norm(self) -> float:
        h = self.h
        div = (
            (self.u[1:] - self.u[:-1])
            + (self.v[:, 1:] - self.v[:, :-1])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / h
        return float(div.abs().max())

    def kinetic_energy(self) -> float:
        h3 = self.h**3
        ke = 0.5 * (
            (0.5 * (self.u[:-1] + self.u[1:])) ** 2
            + (0.5 * (self.v[:, :-1] + self.v[:, 1:])) ** 2
            + (0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])) ** 2
        ).sum()
        return float(ke) * h3

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "u": self.u.clone(),
            "v": self.v.clone(),
            "w": self.w.clone(),
            "p": self.p.clone(),
            "T": self.T.clone(),
            "t": self.t,
            "n_steps": self.n_steps,
        }

    def set_state(self, state: dict) -> None:
        self.u, self.v, self.w, self.p, self.T = (
            state["u"],
            state["v"],
            state["w"],
            state["p"],
            state["T"],
        )
        self.t = state["t"]
        self.n_steps = state["n_steps"]
