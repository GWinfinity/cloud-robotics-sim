"""Quadrants/Taichi backend for the 3D projection-method CFD solver.

Numerically equivalent to the torch backend in ``cfd3d.py`` (same MAC
staggered grid, same upwind advection, same mixed Neumann/Dirichlet pressure
operator) but implemented with ``@qd.kernel`` s like the thermal /
joule_heating / acoustics plugins, so it compiles for CPU and GPU through
the genesis-world backend and carries no torch tensor state.

The pressure Poisson solve stays on the host (cached sparse LU for <= 30k
cells, scipy CG otherwise) — a direct solve costs about a millisecond on
competition-prototype grids and needs the same assembly on both backends.
Quadrants 1.3 quirks honoured: no field aliases inside kernels (fields are
always reached through ``self``), no host-precomputed constants inside
kernels (scalars are passed as kernel arguments).

Single-instance solver (no batch dimension) like the torch backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import genesis as gs
import quadrants as qd

from plugins.solvers.cfd_coupling.core.cfd3d import G, build_poisson_csr


@dataclass
class QDCFDOptions:
    """Options for the quadrants CFD backend (mirrors ``CFDOptions``)."""

    domain: tuple[float, float, float] = (1.0, 0.5, 0.5)
    cells: tuple[int, int, int] = (48, 24, 24)
    viscosity: float = 1.0e-6
    density: float = 1000.0
    inlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (0.25, 0.75),
    )
    outlet_patch: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )
    inlet_temperature: float = 300.0
    initial_temperature: float = 300.0
    thermal_diffusivity: float = 1.4e-7
    advect_temperature: bool = True
    pressure_solver: str = "auto"  # 'auto' | 'direct' (sparse LU) | 'cg'
    direct_precision: str = "float64"  # LU precision: 'float64' (default, also
    # faster here than 'float32' through SuperLU) | 'float32'
    cg_tol: float = 1e-6
    cg_max_iters: int = 800


@qd.data_oriented
class QDCFD3D:
    """Incompressible NS solver on a uniform MAC grid (quadrants backend).

    Duck-type compatible with ``CFD3D``: the coupler and the genesis solver
    wrappers work with either backend unchanged.
    """

    def __init__(self, options: QDCFDOptions):
        o = options
        self.o = o
        nx, ny, nz = o.cells
        self.nx, self.ny, self.nz = nx, ny, nz
        Lx, Ly, Lz = o.domain
        self.h = Lx / nx
        if abs(Ly / ny - self.h) > 1e-9 or abs(Lz / nz - self.h) > 1e-9:
            raise ValueError("cells/domain must give a uniform grid spacing")

        f32 = gs.qd_float
        self.u = qd.field(dtype=f32, shape=(nx + 1, ny, nz))
        self.v = qd.field(dtype=f32, shape=(nx, ny + 1, nz))
        self.w = qd.field(dtype=f32, shape=(nx, ny, nz + 1))
        self.u1 = qd.field(dtype=f32, shape=(nx + 1, ny, nz))
        self.v1 = qd.field(dtype=f32, shape=(nx, ny + 1, nz))
        self.w1 = qd.field(dtype=f32, shape=(nx, ny, nz + 1))
        self.p = qd.field(dtype=f32, shape=(nx, ny, nz))
        self.div = qd.field(dtype=f32, shape=(nx, ny, nz))
        # Temperature uses a two-frame buffer (read frame f, write frame 1-f)
        # like the thermal plugin: the advection stencil reads neighbours, so
        # an in-place update would be a data race.
        self.T = qd.field(dtype=f32, shape=(2, nx, ny, nz))
        self._t_frame = 0
        self.mask_in = qd.field(dtype=f32, shape=(ny, nz))
        self.mask_out = qd.field(dtype=f32, shape=(ny, nz))

        (y0, y1), (z0, z1) = o.inlet_patch
        yc = (np.arange(ny, dtype=np.float64) + 0.5) / ny
        zc = (np.arange(nz, dtype=np.float64) + 0.5) / nz
        ym = ((yc >= y0) & (yc < y1))[:, None]
        zm = ((zc >= z0) & (zc < z1))[None, :]
        self._mask_in_np = (ym & zm).astype(np.float64)
        (y0o, y1o), (z0o, z1o) = o.outlet_patch
        ymo = ((yc >= y0o) & (yc < y1o))[:, None]
        zmo = ((zc >= z0o) & (zc < z1o))[None, :]
        self._mask_out_np = (ymo & zmo).astype(np.float64)
        self.mask_in.from_numpy(self._mask_in_np)
        self.mask_out.from_numpy(self._mask_out_np)

        self.u.from_numpy(np.zeros((nx + 1, ny, nz)))
        self.v.from_numpy(np.zeros((nx, ny + 1, nz)))
        self.w.from_numpy(np.zeros((nx, ny, nz + 1)))
        self.u1.from_numpy(np.zeros((nx + 1, ny, nz)))
        self.v1.from_numpy(np.zeros((nx, ny + 1, nz)))
        self.w1.from_numpy(np.zeros((nx, ny, nz + 1)))
        self.p.from_numpy(np.zeros((nx, ny, nz)))
        self.div.from_numpy(np.zeros((nx, ny, nz)))
        _t0 = np.full((nx, ny, nz), o.initial_temperature)
        self.T.from_numpy(np.stack([_t0, _t0]))

        self.t = 0.0
        self.n_steps = 0
        self.U_in = 0.0
        self.T_in = o.inlet_temperature
        self._cg_iters = 0

        # Host-side pressure operator (shared assembly with the torch backend).
        self._direct_lu = None
        self._div_t = None  # zero-copy torch views of the transfer fields
        self._p_t = None
        self._direct_threshold = 30000
        self._use_direct = o.pressure_solver == "direct" or (
            o.pressure_solver == "auto" and nx * ny * nz <= self._direct_threshold
        )
        self._mat = None

    # ------------------------------------------------------------------ #
    # Ghost accessors. Each component has its own ghost rules; pad-space
    # accessors (index reaching one past a boundary) mirror the torch
    # backend's pad tensors exactly.
    # ------------------------------------------------------------------ #
    @qd.func
    def _u_own(self, i: int, j: int, k: int):  # x replicate, y/z mirror
        ii = i
        jj = j
        kk = k
        s = 1.0
        if i < 0:
            ii = 0
        if i > self.nx:
            ii = self.nx
        if j < 0:
            jj = 0
            s = -s
        if j > self.ny - 1:
            jj = self.ny - 1
            s = -s
        if k < 0:
            kk = 0
            s = -s
        if k > self.nz - 1:
            kk = self.nz - 1
            s = -s
        return s * self.u[ii, jj, kk]

    @qd.func
    def _v_own(self, i: int, j: int, k: int):  # x mirror, y replicate, z mirror
        ii = i
        jj = j
        kk = k
        s = 1.0
        if i < 0:
            ii = 0
            s = -s
        if i > self.nx - 1:
            ii = self.nx - 1
            s = -s
        if j < 0:
            jj = 0
        if j > self.ny:
            jj = self.ny
        if k < 0:
            kk = 0
            s = -s
        if k > self.nz - 1:
            kk = self.nz - 1
            s = -s
        return s * self.v[ii, jj, kk]

    @qd.func
    def _w_own(self, i: int, j: int, k: int):  # x/y mirror, z replicate
        ii = i
        jj = j
        kk = k
        s = 1.0
        if i < 0:
            ii = 0
            s = -s
        if i > self.nx - 1:
            ii = self.nx - 1
            s = -s
        if j < 0:
            jj = 0
            s = -s
        if j > self.ny - 1:
            jj = self.ny - 1
            s = -s
        if k < 0:
            kk = 0
        if k > self.nz:
            kk = self.nz
        return s * self.w[ii, jj, kk]

    @qd.func
    def _v_xpad(self, i: int, j: int, k: int):  # replicate x ghosts (pad coords)
        ii = i
        if i < 0:
            ii = 0
        if i > self.nx - 1:
            ii = self.nx - 1
        return self.v[ii, j, k]

    @qd.func
    def _w_xpad(self, i: int, j: int, k: int):  # replicate x ghosts
        ii = i
        if i < 0:
            ii = 0
        if i > self.nx - 1:
            ii = self.nx - 1
        return self.w[ii, j, k]

    @qd.func
    def _u_ypad(self, i: int, jy: int, k: int):  # mirror y ghosts (pad coords)
        val = self.u[i, jy, k]
        if jy == 0:
            val = -self.u[i, 0, k]
        if jy == self.ny + 1:
            val = -self.u[i, self.ny - 1, k]
        return val

    @qd.func
    def _w_ypad(self, i: int, jy: int, k: int):  # mirror y ghosts
        val = self.w[i, jy, k]
        if jy == 0:
            val = -self.w[i, 0, k]
        if jy == self.ny + 1:
            val = -self.w[i, self.ny - 1, k]
        return val

    @qd.func
    def _u_zpad(self, i: int, j: int, kz: int):  # mirror z ghosts (pad coords)
        val = self.u[i, j, kz]
        if kz == 0:
            val = -self.u[i, j, 0]
        if kz == self.nz + 1:
            val = -self.u[i, j, self.nz - 1]
        return val

    @qd.func
    def _v_zpad(self, i: int, j: int, kz: int):  # mirror z ghosts
        val = self.v[i, j, kz]
        if kz == 0:
            val = -self.v[i, j, 0]
        if kz == self.nz + 1:
            val = -self.v[i, j, self.nz - 1]
        return val

    @qd.func
    def _p_xpad(self, i: int, j: int, k: int):  # x pad coords 0..nx+1
        val = self.p[i - 1, j, k]
        if i == 0:
            val = self.p[0, j, k]
        if i == self.nx + 1:
            val = self.p[self.nx - 1, j, k]
            if self.mask_out[j, k] > 0.5:
                val = -val
        return val

    # ------------------------------------------------------------------ #
    # Fused predictor + BC + divergence kernel (single launch): the three
    # component predictors (each loop reads u/v/w, writes u1/v1/w1 - no
    # cross-loop hazard), then boundary conditions on the predictor fields,
    # then the divergence pre-scaled by -1/dt so the host pressure solve
    # consumes it directly (rhs of -lap p = -div/dt).
    # ------------------------------------------------------------------ #
    @qd.kernel
    def _predict_bc_div(
        self,
        dt: float,
        nu: float,
        inv_h: float,
        inv_h2: float,
        u_in: float,
        neg_inv_dt: float,
    ):
        for i, j, k in qd.ndrange(self.nx + 1, self.ny, self.nz):
            U = self.u[i, j, k]
            V = 0.25 * (
                self._v_xpad(i - 1, j, k)
                + self._v_xpad(i, j, k)
                + self._v_xpad(i - 1, j + 1, k)
                + self._v_xpad(i, j + 1, k)
            )
            W = 0.25 * (
                self._w_xpad(i - 1, j, k)
                + self._w_xpad(i, j, k)
                + self._w_xpad(i - 1, j, k + 1)
                + self._w_xpad(i, j, k + 1)
            )
            c = self._u_own(i, j, k)
            ax = 0.0
            ay = 0.0
            az = 0.0
            if U >= 0.0:
                ax = (c - self._u_own(i - 1, j, k)) * inv_h
            else:
                ax = (self._u_own(i + 1, j, k) - c) * inv_h
            if V >= 0.0:
                ay = (c - self._u_own(i, j - 1, k)) * inv_h
            else:
                ay = (self._u_own(i, j + 1, k) - c) * inv_h
            if W >= 0.0:
                az = (c - self._u_own(i, j, k - 1)) * inv_h
            else:
                az = (self._u_own(i, j, k + 1) - c) * inv_h
            lap = (
                self._u_own(i - 1, j, k)
                + self._u_own(i + 1, j, k)
                + self._u_own(i, j - 1, k)
                + self._u_own(i, j + 1, k)
                + self._u_own(i, j, k - 1)
                + self._u_own(i, j, k + 1)
                - 6.0 * c
            ) * inv_h2
            self.u1[i, j, k] = c + dt * (-(U * ax + V * ay + W * az) + nu * lap)
        for i, j, k in qd.ndrange(self.nx, self.ny + 1, self.nz):
            V = self.v[i, j, k]
            U = 0.25 * (
                self._u_ypad(i, j, k)
                + self._u_ypad(i + 1, j, k)
                + self._u_ypad(i, j + 1, k)
                + self._u_ypad(i + 1, j + 1, k)
            )
            W = 0.25 * (
                self._w_ypad(i, j, k)
                + self._w_ypad(i, j + 1, k)
                + self._w_ypad(i, j, k + 1)
                + self._w_ypad(i, j + 1, k + 1)
            )
            c = self._v_own(i, j, k)
            ax = 0.0
            ay = 0.0
            az = 0.0
            if U >= 0.0:
                ax = (c - self._v_own(i - 1, j, k)) * inv_h
            else:
                ax = (self._v_own(i + 1, j, k) - c) * inv_h
            if V >= 0.0:
                ay = (c - self._v_own(i, j - 1, k)) * inv_h
            else:
                ay = (self._v_own(i, j + 1, k) - c) * inv_h
            if W >= 0.0:
                az = (c - self._v_own(i, j, k - 1)) * inv_h
            else:
                az = (self._v_own(i, j, k + 1) - c) * inv_h
            lap = (
                self._v_own(i - 1, j, k)
                + self._v_own(i + 1, j, k)
                + self._v_own(i, j - 1, k)
                + self._v_own(i, j + 1, k)
                + self._v_own(i, j, k - 1)
                + self._v_own(i, j, k + 1)
                - 6.0 * c
            ) * inv_h2
            self.v1[i, j, k] = c + dt * (-(U * ax + V * ay + W * az) + nu * lap)
        for i, j, k in qd.ndrange(self.nx, self.ny, self.nz + 1):
            W = self.w[i, j, k]
            U = 0.25 * (
                self._u_zpad(i, j, k)
                + self._u_zpad(i + 1, j, k)
                + self._u_zpad(i, j, k + 1)
                + self._u_zpad(i + 1, j, k + 1)
            )
            V = 0.25 * (
                self._v_zpad(i, j, k)
                + self._v_zpad(i, j, k + 1)
                + self._v_zpad(i, j + 1, k)
                + self._v_zpad(i, j + 1, k + 1)
            )
            c = self._w_own(i, j, k)
            ax = 0.0
            ay = 0.0
            az = 0.0
            if U >= 0.0:
                ax = (c - self._w_own(i - 1, j, k)) * inv_h
            else:
                ax = (self._w_own(i + 1, j, k) - c) * inv_h
            if V >= 0.0:
                ay = (c - self._w_own(i, j - 1, k)) * inv_h
            else:
                ay = (self._w_own(i, j + 1, k) - c) * inv_h
            if W >= 0.0:
                az = (c - self._w_own(i, j, k - 1)) * inv_h
            else:
                az = (self._w_own(i, j, k + 1) - c) * inv_h
            lap = (
                self._w_own(i - 1, j, k)
                + self._w_own(i + 1, j, k)
                + self._w_own(i, j - 1, k)
                + self._w_own(i, j + 1, k)
                + self._w_own(i, j, k - 1)
                + self._w_own(i, j, k + 1)
                - 6.0 * c
            ) * inv_h2
            self.w1[i, j, k] = c + dt * (-(U * ax + V * ay + W * az) + nu * lap)
        # Boundary conditions on the predictor fields.
        for j, k in qd.ndrange(self.ny, self.nz):
            self.u1[0, j, k] = u_in * self.mask_in[j, k]
            if self.mask_out[j, k] < 0.5:
                self.u1[self.nx, j, k] = 0.0
        for i, k in qd.ndrange(self.nx, self.nz):
            self.v1[i, 0, k] = 0.0
            self.v1[i, self.ny, k] = 0.0
        for i, j in qd.ndrange(self.nx, self.ny):
            self.w1[i, j, 0] = 0.0
            self.w1[i, j, self.nz] = 0.0
        # Divergence of the predictor field, pre-scaled by -1/dt.
        for i, j, k in qd.ndrange(self.nx, self.ny, self.nz):
            self.div[i, j, k] = (
                (self.u1[i + 1, j, k] - self.u1[i, j, k])
                + (self.v1[i, j + 1, k] - self.v1[i, j, k])
                + (self.w1[i, j, k + 1] - self.w1[i, j, k])
            ) * (inv_h * neg_inv_dt)

    @qd.kernel
    def _project_swap_temp(
        self,
        dt: float,
        inv_h: float,
        inv_h2: float,
        kappa: float,
        u_in: float,
        t_in: float,
        f: int,
        do_t: int,
    ):
        # Face correction with the pressure gradient (x ghosts: Neumann at the
        # inlet, Dirichlet p=0 on the open outlet patch).
        for i, j, k in qd.ndrange(self.nx + 1, self.ny, self.nz):
            if i > 0:
                self.u1[i, j, k] -= dt * (
                    self._p_xpad(i + 1, j, k) - self._p_xpad(i, j, k)
                ) * inv_h
        for i, j, k in qd.ndrange(self.nx, self.ny + 1, self.nz):
            if j > 0 and j < self.ny:
                self.v1[i, j, k] -= dt * (self.p[i, j, k] - self.p[i, j - 1, k]) * inv_h
        for i, j, k in qd.ndrange(self.nx, self.ny, self.nz + 1):
            if k > 0 and k < self.nz:
                self.w1[i, j, k] -= dt * (self.p[i, j, k] - self.p[i, j, k - 1]) * inv_h
        # Swap predictor fields back into u/v/w with boundary conditions.
        for i, j, k in qd.ndrange(self.nx + 1, self.ny, self.nz):
            val = self.u1[i, j, k]
            if i == 0:
                val = u_in * self.mask_in[j, k]
            self.u[i, j, k] = val
        for i, j, k in qd.ndrange(self.nx, self.ny + 1, self.nz):
            val = self.v1[i, j, k]
            if j == 0 or j == self.ny:
                val = 0.0
            self.v[i, j, k] = val
        for i, j, k in qd.ndrange(self.nx, self.ny, self.nz + 1):
            val = self.w1[i, j, k]
            if k == 0 or k == self.nz:
                val = 0.0
            self.w[i, j, k] = val
        # Passive temperature advection (two-frame buffer: read f, write 1-f).
        f1 = 1 - f
        for i, j, k in qd.ndrange(self.nx, self.ny, self.nz):
            if do_t == 0:
                continue
            U = 0.5 * (self.u1[i, j, k] + self.u1[i + 1, j, k])
            V = 0.5 * (self.v1[i, j, k] + self.v1[i, j + 1, k])
            W = 0.5 * (self.w1[i, j, k] + self.w1[i, j, k + 1])
            c = self._t_ghost(i, j, k, t_in, f)
            ax = 0.0
            ay = 0.0
            az = 0.0
            if U >= 0.0:
                ax = (c - self._t_ghost(i - 1, j, k, t_in, f)) * inv_h
            else:
                ax = (self._t_ghost(i + 1, j, k, t_in, f) - c) * inv_h
            if V >= 0.0:
                ay = (c - self._t_ghost(i, j - 1, k, t_in, f)) * inv_h
            else:
                ay = (self._t_ghost(i, j + 1, k, t_in, f) - c) * inv_h
            if W >= 0.0:
                az = (c - self._t_ghost(i, j, k - 1, t_in, f)) * inv_h
            else:
                az = (self._t_ghost(i, j, k + 1, t_in, f) - c) * inv_h
            lap = (
                self._t_ghost(i - 1, j, k, t_in, f)
                + self._t_ghost(i + 1, j, k, t_in, f)
                + self._t_ghost(i, j - 1, k, t_in, f)
                + self._t_ghost(i, j + 1, k, t_in, f)
                + self._t_ghost(i, j, k - 1, t_in, f)
                + self._t_ghost(i, j, k + 1, t_in, f)
                - 6.0 * c
            ) * inv_h2
            self.T[f1, i, j, k] = c + dt * (-(U * ax + V * ay + W * az) + kappa * lap)

    # ------------------------------------------------------------------ #
    # Passive temperature advection helpers
    # ------------------------------------------------------------------ #
    @qd.func
    def _t_ghost(self, i: int, j: int, k: int, t_in: float, f: int):
        # Dirichlet ghost at the inlet (x = 0), zero-gradient elsewhere.
        ii = i
        jj = j
        kk = k
        val = t_in
        if i >= 0:
            if i > self.nx - 1:
                ii = self.nx - 1
            if j < 0:
                jj = 0
            if j > self.ny - 1:
                jj = self.ny - 1
            if k < 0:
                kk = 0
            if k > self.nz - 1:
                kk = self.nz - 1
            val = self.T[f, ii, jj, kk]
        return val

    # ------------------------------------------------------------------ #
    # Host-side pressure Poisson: cached sparse LU or scipy CG. The
    # divergence field arrives pre-scaled by -1/dt from the fused kernel,
    # so the right-hand side is consumed with no host arithmetic.
    # ------------------------------------------------------------------ #
    def _solve_pressure(self) -> None:
        import scipy.sparse.linalg as spla
        import torch
        from genesis.utils.misc import qd_to_torch

        if self._div_t is None:
            # Zero-copy views when genesis runs zerocopy (CPU): ~0.3 ms/step
            # cheaper than field.to_numpy() / field.from_numpy() transfers.
            self._div_t = qd_to_torch(self.div)
            self._p_t = qd_to_torch(self.p)
        if self._mat is None:
            lu_dtype = (
                np.float32 if self.o.direct_precision == "float32" else np.float64
            )
            self._mat, _ = build_poisson_csr(
                self.nx,
                self.ny,
                self.nz,
                self.h,
                self._mask_out_np,
                dtype=lu_dtype,
            )
            if self._use_direct:
                self._direct_lu = spla.splu(self._mat.tocsc())
        b = self._div_t.double().numpy().ravel()
        if self._use_direct:
            x = self._direct_lu.solve(b)
            self._cg_iters = 0
        else:
            x, info = spla.cg(
                self._mat,
                b,
                rtol=self.o.cg_tol,
                maxiter=self.o.cg_max_iters,
            )
            self._cg_iters = info if info > 0 else self.o.cg_max_iters
        self._p_t.copy_(torch.from_numpy(x.reshape(self.nx, self.ny, self.nz)))

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        """Advance one time step (explicit; dt must satisfy CFL limits).

        Two fused quadrants launches per step (predictor+BC+divergence,
        projection+swap+temperature) plus the host-side pressure solve.
        """
        o = self.o
        inv_h = 1.0 / self.h
        inv_h2 = inv_h * inv_h
        self._predict_bc_div(
            dt, o.viscosity, inv_h, inv_h2, self.U_in, -1.0 / dt
        )
        self._solve_pressure()
        self._project_swap_temp(
            dt,
            inv_h,
            inv_h2,
            o.thermal_diffusivity,
            self.U_in,
            self.T_in,
            self._t_frame,
            1 if o.advect_temperature else 0,
        )
        if o.advect_temperature:
            self._t_frame = 1 - self._t_frame
        self.t += dt
        self.n_steps += 1

    # ------------------------------------------------------------------ #
    # Coupling probes (host side; grids are prototype-sized)
    # ------------------------------------------------------------------ #
    @property
    def inlet_mask_f(self) -> np.ndarray:
        return self._mask_in_np

    @property
    def outlet_mask_f(self) -> np.ndarray:
        return self._mask_out_np

    def set_inlet(self, velocity: float, temperature: float | None = None) -> None:
        self.U_in = float(velocity)
        if temperature is not None:
            self.T_in = float(temperature)

    def inlet_patch_area(self) -> float:
        return float(self._mask_in_np.sum()) * self.h * self.h

    def inlet_flow(self) -> float:
        u0 = self.u.to_numpy()[0]
        return float((u0 * self._mask_in_np).sum()) * self.h * self.h

    def outlet_flow(self) -> float:
        return float(self.u.to_numpy()[-1].sum()) * self.h * self.h

    def inlet_pressure_head(self) -> float:
        return float(self.p.to_numpy().mean()) / G

    def divergence_norm(self) -> float:
        u = self.u.to_numpy()
        v = self.v.to_numpy()
        w = self.w.to_numpy()
        div = (
            (u[1:] - u[:-1])
            + (v[:, 1:] - v[:, :-1])
            + (w[:, :, 1:] - w[:, :, :-1])
        ) / self.h
        return float(np.abs(div).max())

    def kinetic_energy(self) -> float:
        u = self.u.to_numpy()
        v = self.v.to_numpy()
        w = self.w.to_numpy()
        h3 = self.h**3
        ke = 0.5 * (
            (0.5 * (u[:-1] + u[1:])) ** 2
            + (0.5 * (v[:, :-1] + v[:, 1:])) ** 2
            + (0.5 * (w[:, :, :-1] + w[:, :, 1:])) ** 2
        ).sum()
        return float(ke) * h3

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "u": self.u.to_numpy(),
            "v": self.v.to_numpy(),
            "w": self.w.to_numpy(),
            "p": self.p.to_numpy(),
            "T": self.T.to_numpy()[self._t_frame],
            "t": self.t,
            "n_steps": self.n_steps,
        }

    def set_state(self, state: dict) -> None:
        self.u.from_numpy(np.ascontiguousarray(state["u"]))
        self.v.from_numpy(np.ascontiguousarray(state["v"]))
        self.w.from_numpy(np.ascontiguousarray(state["w"]))
        self.p.from_numpy(np.ascontiguousarray(state["p"]))
        _t = np.ascontiguousarray(state["T"])
        self.T.from_numpy(np.stack([_t, _t]))  # both frames
        self._t_frame = 0
        self.t = state["t"]
        self.n_steps = state["n_steps"]


__all__ = ["QDCFD3D", "QDCFDOptions"]
