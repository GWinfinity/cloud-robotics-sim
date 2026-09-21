"""Offline tests for the 1D-3D CFD coupling prototype (no genesis needed).

Validation strategy (matches the competition's 算例要求):
1. Water hammer vs the Joukowsky analytical solution (1D solver alone).
2. Lid-driven cavity vs steady-state behaviour (3D solver alone).
3. Coupled pipe-plenum run: mass conservation, backpressure feedback and
   fixed-point convergence of the interface residual.
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from plugins.solvers.cfd_coupling import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    Pipe1D,
    PipeOptions,
)


# --------------------------------------------------------------------- #
# 1. Water hammer vs Joukowsky
# --------------------------------------------------------------------- #
def test_water_hammer_joukowsky() -> None:
    """Fast valve closure: peak head must match the Joukowsky rise a*dV/g."""
    a = 1000.0  # wave speed [m/s]
    L = 100.0  # length [m]
    N = 50
    pipe = Pipe1D(
        PipeOptions(
            length=L,
            diameter=0.1,
            n_reaches=N,
            wave_speed=a,
            friction=0.0,  # frictionless: Joukowsky is exact
            reservoir_head=50.0,
            discharge_head=0.0,
            nozzle_area=1.0e-3,
            discharge_coeff=0.8,
        )
    )
    v0 = pipe.Q[-1] / pipe.area
    dH_theory = a * v0 / 9.81
    h_steady = pipe.H[N]  # valve-node head before closure

    # Track the peak valve-node head within the first reflection period 2L/a.
    n_window = int(round(2 * L / a / pipe.dt))
    h_max = -np.inf
    for _ in range(n_window):
        frac = 1.0 if pipe.t < 2 * pipe.dt else 1.0e-3  # effectively instant
        pipe.step(valve_fraction=frac)
        h_max = max(h_max, pipe.H[N])

    # Joukowsky: peak = steady valve-node head + a * dV / g.
    expected = h_steady + dH_theory
    assert h_max == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------- #
# 2. Lid-driven cavity (3D solver alone, Re=100)
# --------------------------------------------------------------------- #
def _lid_driven_case(n: int = 30, steps: int = 800, dt: float = 0.01) -> CFD3D:
    """Lid-driven cavity via a prescribed top-face velocity (ramp start)."""
    cfd = CFD3D(
        CFDOptions(
            domain=(1.0, 1.0, 1.0),
            cells=(n, n, n),
            viscosity=0.01,  # Re = U*L/nu = 100
            advect_temperature=False,
        )
    )
    ke_hist: list[float] = []
    # Top wall (y = Ly) moves in +x: emulate by prescribing the ghost-row
    # velocity of the u component just below the top boundary each step.
    # The lid is ramped in over 50 steps to avoid the impulsive-start
    # corner singularity blowing up the explicit scheme.
    for i in range(steps):
        ramp = min(1.0, i / 50.0)
        cfd._enforce_bc(cfd.u, cfd.v, cfd.w)
        cfd.u[:, -1, :] = ramp  # moving lid row
        cfd.step(dt)
        ke_hist.append(cfd.kinetic_energy())
    cfd.ke_hist = ke_hist  # type: ignore[attr-defined]
    return cfd


@pytest.mark.slow
def test_lid_driven_cavity() -> None:
    cfd = _lid_driven_case()
    # (a) incompressibility
    assert cfd.divergence_norm() < 1e-5
    # (b) steady state: kinetic-energy plateau over the last 200 steps
    ke_hist = cfd.ke_hist  # type: ignore[attr-defined]
    ke_early = float(np.mean(ke_hist[-400:-200]))
    ke_late = float(np.mean(ke_hist[-200:]))
    assert ke_late > 0.0
    assert abs(ke_late - ke_early) / ke_late < 0.02
    # (c) vortex structure: +x flow in the upper half, return flow below
    u_c = 0.5 * (cfd.u[:-1] + cfd.u[1:])  # cell-centred x-velocity
    upper = float(u_c[:, 2 * cfd.ny // 3 :, :].mean())
    lower = float(u_c[:, : cfd.ny // 3, :].mean())
    assert upper > 0.05
    assert lower < 0.0
    # (d) interior velocity magnitude in the Re=100 range (Ghia: ~0.2-0.4).
    # Exclude the two rows nearest the lid (our discrete moving-lid proxy
    # over-drives them) and the side walls.
    interior = float(u_c[:, 1:-3, 1:-1].abs().max())
    assert 0.15 < interior < 0.7


def test_cfd_mass_conservation_inlet_outlet() -> None:
    """Constant inflow through a patch must equal the outflow at steady state."""
    cfd = CFD3D(
        CFDOptions(
            domain=(0.1, 0.05, 0.05),
            cells=(20, 10, 10),
            viscosity=1.0e-3,
            inlet_patch=((0.4, 0.6), (0.4, 0.6)),
            advect_temperature=False,
            cg_tol=1e-6,
        )
    )
    cfd.set_inlet(0.5)
    for _ in range(2500):
        cfd.step(0.001)
    q_in = cfd.inlet_flow()
    q_out = cfd.outlet_flow()
    assert q_in == pytest.approx(0.5 * cfd.inlet_patch_area(), rel=1e-3)
    assert q_out == pytest.approx(q_in, rel=0.05)
    assert cfd.divergence_norm() < 5e-5


# --------------------------------------------------------------------- #
# 3. Coupled pipe-plenum run
# --------------------------------------------------------------------- #
def _make_coupled_system() -> tuple[Pipe1D, CFD3D, Coupler]:
    pipe = Pipe1D(
        PipeOptions(
            length=10.0,
            diameter=0.05,
            n_reaches=50,
            wave_speed=200.0,
            friction=0.02,
            reservoir_head=30.0,
            nozzle_area=3.0e-5,
            discharge_coeff=0.8,
        )
    )
    cfd = CFD3D(
        CFDOptions(
            domain=(0.2, 0.1, 0.1),
            cells=(20, 10, 10),
            viscosity=1.0e-4,
            inlet_patch=((0.4, 0.6), (0.4, 0.6)),
            outlet_patch=((0.45, 0.55), (0.45, 0.55)),
            advect_temperature=False,
        )
    )
    coupler = Coupler(
        pipe,
        cfd,
        CouplingOptions(
            macro_dt=0.001,
            fixed_point_iters=2,
            valve_closure_start=0.1,
            valve_closure_duration=0.08,
        ),
    )
    return pipe, cfd, coupler


def test_coupler_backpressure_response() -> None:
    """Two-way coupling: plenum pressurises with flow, valve closure chokes
    the nozzle and the falling plenum backpressure feeds back to the 1D side."""
    pipe, cfd, coupler = _make_coupled_system()

    logs_pre = coupler.run(0.1)  # steady with the valve open (closure at 0.1)
    flow_pre = np.mean([log.nozzle_flow for log in logs_pre[-10:]])
    head_pre = np.mean([log.plenum_head for log in logs_pre[-10:]])

    # The restricted outlet pressurises the plenum: the 1D flow must feel a
    # non-zero backpressure at steady state.
    assert head_pre > 0.2

    logs_post = coupler.run(0.4)  # through closure (0.1-0.18 s) and decay
    flow_post_min = min(log.nozzle_flow for log in logs_post)
    head_post_min = min(log.plenum_head for log in logs_post)

    # 1D -> 3D: the closing valve chokes the nozzle flow into the plenum.
    assert flow_post_min < 0.5 * flow_pre
    # 3D -> 1D: with less flow through the restricted outlet, the plenum
    # backpressure that the 1D nozzle feels drops accordingly.
    assert head_post_min < head_pre - 0.02
    assert np.all(np.diff(coupler.history()["t"]) > 0)


def test_coupler_fixed_point_convergence() -> None:
    """During a transient, more Gauss-Seidel passes must shrink the change of
    the exchanged nozzle flow between successive passes (interface residual)."""
    coupler1 = _make_coupled_system()[2]
    coupler1.o.fixed_point_iters = 1
    coupler1.run(0.1)  # warmup; valve starts closing at 0.1 s
    coupler1.step()  # first transient step (valve still fully open)
    log1 = coupler1.step()

    coupler4 = _make_coupled_system()[2]
    coupler4.o.fixed_point_iters = 4
    coupler4.run(0.1)
    coupler4.step()
    log4 = coupler4.step()

    assert log4.fp_flow_residuals[-1] < log1.fp_flow_residuals[-1]


def test_coupler_exchange_latency_recorded() -> None:
    """Exchange latency (boundary transfer only) must be measured and small."""
    _, _, coupler = _make_coupled_system()
    logs = coupler.run(0.1)
    lat = np.array([log.exchange_latency_ms for log in logs])
    assert np.all(lat >= 0.0)
    assert np.median(lat) < 5.0  # pure Python boundary transfer, no solvers
