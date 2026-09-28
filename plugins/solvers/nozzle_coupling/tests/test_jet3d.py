"""Offline tests for the weakly-compressible 3D jet solver (no genesis).

Validation:
1. Inlet BC: the prescribed mass flux enters the domain exactly.
2. Projection consistency: div(u) matches the thermal-expansion target D.
3. Transient mass bookkeeping: d(domain_mass)/dt == mdot_in - mdot_out.
4. Backpressure response: restricting the 3D outlet patch raises the pressure
   fed back to the 1D side (the 3D -> 1D coupling signal).
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

from plugins.solvers.nozzle_coupling.core.jet3d import Jet3D, JetOptions


def _jet(outlet_patch=((0.0, 1.0), (0.0, 1.0)), **kw) -> Jet3D:
    opts = JetOptions(
        domain=(0.4, 0.16, 0.16),
        cells=(20, 8, 8),
        outlet_patch=outlet_patch,
    )
    for k, v in kw.items():
        setattr(opts, k, v)
    return Jet3D(opts)


def _drive(
    jet: Jet3D,
    steps: int,
    dt: float,
    mdot: float = 0.2,
    hot: bool = True,
) -> None:
    """Drive the jet with a short startup ramp (avoids the impulsive-start
    CFL transient) until ``steps`` total steps are taken.
    """
    for _ in range(steps):
        ramp = min(1.0, jet.t / 0.01)
        jet.set_inlet(
            mass_flux=mdot * ramp,
            stagnation_temp=300.0 + (1200.0 if hot else 0.0) * ramp,
            condensed_fraction=0.1 * ramp,
            r_eff=300.0,
        )
        jet.step(dt)


def _steady(jet: Jet3D, mdot: float, dt: float, hot: bool = True) -> None:
    """Ramp up, then continue at constant conditions until quasi-steady."""
    _drive(jet, steps=200, dt=dt, mdot=mdot, hot=hot)
    temp = 1500.0 if hot else 300.0
    jet.set_inlet(
        mass_flux=mdot, stagnation_temp=temp, condensed_fraction=0.1, r_eff=300.0
    )
    for _ in range(400):
        jet.step(dt)


# --------------------------------------------------------------------- #
# 1. Inlet mass flux BC
# --------------------------------------------------------------------- #
def test_inlet_mass_flux_bc() -> None:
    """The prescribed mass flux enters the domain exactly."""
    jet = _jet()
    _drive(jet, steps=150, dt=1.0e-4)
    assert jet.inlet_mass_flow() == pytest.approx(0.2, rel=1e-6)


# --------------------------------------------------------------------- #
# 2. Projection consistency with the expansion target
# --------------------------------------------------------------------- #
def test_projection_enforces_expansion_constraint() -> None:
    """div(u) matches the thermal-expansion target D."""
    jet = _jet()
    _drive(jet, steps=150, dt=1.0e-4)
    h = jet.h
    div = (
        (jet.u[1:] - jet.u[:-1])
        + (jet.v[:, 1:] - jet.v[:, :-1])
        + (jet.w[:, :, 1:] - jet.w[:, :, :-1])
    ) / h
    residual = (div - jet._last_D).abs().max()
    assert float(residual) < jet.o.cg_constraint_tol * 2.0


# --------------------------------------------------------------------- #
# 3. Mass bookkeeping over a transient
# --------------------------------------------------------------------- #
def test_mass_bookkeeping_transient() -> None:
    """d(domain_mass)/dt matches the net mass flux window."""
    jet = _jet()
    dt = 1.0e-4
    _steady(jet, mdot=0.2, dt=dt)  # ramp + quasi-steady state
    m0 = jet.domain_mass()
    mdot_in, mdot_out = [], []
    n_window = 100
    for _ in range(n_window):
        jet.step(dt)
        mdot_in.append(jet.inlet_mass_flow())
        mdot_out.append(jet.outlet_mass_flow())
    dm_actual = jet.domain_mass() - m0
    dm_flux = (np.mean(mdot_in) - np.mean(mdot_out)) * (n_window * dt)
    # Loose tolerance: first-order upwind + frozen-coefficient projection.
    assert dm_actual == pytest.approx(dm_flux, abs=1.0e-3)


# --------------------------------------------------------------------- #
# 4. Backpressure response to outlet restriction
# --------------------------------------------------------------------- #
def test_outlet_restriction_raises_backpressure() -> None:
    """Restricting the outlet raises the backpressure probe."""
    # Cold, low-momentum jet: the exit-plane pressure is then a monotone
    # function of the plenum resistance (hot supersonic jets are momentum-
    # dominated and pressure-matched at the core, see README).
    dt, mdot = 5.0e-5, 0.05

    open_jet = _jet(outlet_patch=((0.0, 1.0), (0.0, 1.0)))
    _steady(open_jet, mdot=mdot, dt=dt, hot=False)
    p_open = open_jet.exit_plane_pressure()

    # A narrow slot as the outlet: strongly restricted plenum.
    restricted = _jet(outlet_patch=((0.35, 0.65), (0.35, 0.65)))
    _steady(restricted, mdot=mdot, dt=dt, hot=False)
    p_restricted = restricted.exit_plane_pressure()

    assert p_restricted > p_open + 100.0  # Pa, clear coupling signal
    # The backpressure probe is absolute and gauge-consistent.
    assert p_restricted > restricted.o.ambient_pressure


def test_phase_scalar_advected_into_domain() -> None:
    """The condensed fraction is advected into the domain."""
    jet = _jet()
    _drive(jet, steps=150, dt=1.0e-4)
    # The condensed fraction enters at 0.1 and must be present downstream.
    assert float(jet.alpha.max()) > 0.05
    assert float(jet.alpha.min()) >= 0.0
