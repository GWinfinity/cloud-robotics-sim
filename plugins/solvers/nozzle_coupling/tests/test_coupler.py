"""Offline tests for the nozzle <-> 3D jet bidirectional coupler.

Validation:
1. Chamber-curve transient: the coupled mass flux tracks the choked closed
   form of the chamber pressure curve (1D -> 3D transient transfer).
2. Closed-loop backpressure response: restricting the 3D outlet raises the
   backpressure fed back to the 1D nozzle and throttles its mass flux
   (3D -> 1D reverse action).
3. Fixed-point convergence of the interface residual.
4. Exchange latency: the pure boundary transfer stays sub-millisecond.
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
import torch

from plugins.solvers.nozzle_coupling.core.coupler import (
    NozzleCoupler,
    NozzleCouplingOptions,
)
from plugins.solvers.nozzle_coupling.core.jet3d import Jet3D, JetOptions
from plugins.solvers.nozzle_coupling.core.nozzle1d import (
    Nozzle1D,
    NozzleOptions,
    _mach_from_area_ratio,
    _t_of_m,
)

P_AMB = 101325.0
GAMMA = 1.25
MW = 0.025
R_G = 8.314462618 / MW


def _nozzle(p_c: float, t_c: float, curve: bool = False) -> Nozzle1D:
    o = NozzleOptions(
        throat_area=1.0e-4,
        exit_area=3.0e-4,
        gamma=GAMMA,
        molecular_weight=MW,
        chamber_pressure=p_c,
        chamber_temperature=t_c,
    )
    if curve:
        o.curve_t = np.array([0.0, 0.005, 0.01])
        o.curve_p = np.array([P_AMB, 0.5 * (P_AMB + p_c), p_c])
        o.curve_T = np.array([300.0, 0.5 * (300.0 + t_c), t_c])
    return Nozzle1D(o)


def _jet(outlet=((0.0, 1.0), (0.0, 1.0))) -> Jet3D:
    # Small grid: the coupled tests are CPU-bound and the nozzle exit jet is
    # supersonic, so the 3D CFL dictates a small macro_dt; keep the cell
    # count low to stay fast.
    return Jet3D(
        JetOptions(
            domain=(0.24, 0.12, 0.12),
            cells=(12, 6, 6),
            outlet_patch=outlet,
        )
    )


def _coupler(nozzle: Nozzle1D, jet: Jet3D, **kw) -> NozzleCoupler:
    kw.setdefault("macro_dt", 2.0e-4)
    kw.setdefault("n_substeps", 4)
    return NozzleCoupler(nozzle, jet, NozzleCouplingOptions(**kw))


# --------------------------------------------------------------------- #
# 1. Chamber-curve transient tracks the choked closed form
# --------------------------------------------------------------------- #
def test_chamber_curve_transient_transfer() -> None:
    """Coupled mass flux tracks the choked closed form of the chamber curve."""
    # Choked but low-Mach-friendly: p_c = 2*P_amb keeps the exit jet near
    # M ~ 2 so the 3D CFL is manageable on the small test grid.
    p_c, t_c = 2.0 * P_AMB, 600.0
    nozzle = _nozzle(p_c, t_c, curve=True)
    coupler = _coupler(nozzle, _jet(), macro_dt=1.0e-5, n_substeps=2)

    n = 600  # t = 0.006 s (curve ends at 0.01 s, sampled on the rising part)
    checkpoints = {150: None, 350: None, 599: None}
    for i in range(n):
        log = coupler.step()
        if i in checkpoints:
            checkpoints[i] = log
    assert bool(torch.isfinite(coupler.jet.T).all())
    for i, log in checkpoints.items():
        if log is None:
            continue
        t = log.t
        p_curve = float(np.interp(t, nozzle.o.curve_t, nozzle.o.curve_p))
        t_curve = float(np.interp(t, nozzle.o.curve_t, nozzle.o.curve_T))
        fac = (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
        mdot_theory = 1.0e-4 * p_curve / np.sqrt(R_G * t_curve) * np.sqrt(GAMMA) * fac
        assert log.mdot == pytest.approx(mdot_theory, rel=0.05)


# --------------------------------------------------------------------- #
# 2. Closed loop: mid-run throttling of the 3D outlet pressurizes the
#    plenum and the raised backpressure drives the 1D nozzle response
# --------------------------------------------------------------------- #
def test_closed_loop_backpressure_response() -> None:
    """End-to-end reverse action: restrict the 3D outlet mid-run -> the
    plenum pressurizes -> the backpressure crosses the interface -> the 1D
    nozzle responds (normal shock pushed upstream, exit pressure rises).

    The nozzle runs choked (shock regime), where the mass flux is physically
    independent of backpressure; the nozzle response shows up in the exit
    static pressure / shock position instead. Mass-flux throttling across
    the interface is covered deterministically by
    ``test_interface_transmits_backpressure_to_nozzle`` below and by the 1D
    unit tests.
    """
    nozzle = Nozzle1D(
        NozzleOptions(
            throat_area=1.0e-3,
            exit_area=3.0e-3,
            gamma=GAMMA,
            molecular_weight=MW,
            chamber_pressure=130000.0,
            chamber_temperature=600.0,
        )
    )
    jet = _jet()
    coupler = _coupler(
        nozzle, jet, macro_dt=2.0e-5, fixed_point_iters=2, inlet_ramp_time=0.004
    )
    coupler.run(duration=0.008)  # open-outlet steady state (choked)
    p_open = coupler.backpressure
    pe_open = coupler.history()["exit_pressure_pa"][-50:].mean()

    jet.set_outlet_patch(((0.4, 0.6), (0.4, 0.6)))  # throttling event
    coupler.run(duration=0.008)

    h = coupler.history()
    assert bool(torch.isfinite(jet.T).all())
    # 3D -> 1D signal: the restricted plenum feeds back a much higher
    # backpressure through the coupled interface.
    assert coupler.backpressure > p_open + 5000.0
    # 1D nozzle response: the exit static pressure rises (the shock is pushed
    # upstream toward the throat as the backpressure climbs).
    assert h["exit_pressure_pa"][-300:].mean() > pe_open + 3000.0


# --------------------------------------------------------------------- #
# 3. Interface plumbing: the 3D-side backpressure selects the 1D regime
# --------------------------------------------------------------------- #
def test_interface_transmits_backpressure_to_nozzle() -> None:
    """Deterministic 3D -> 1D pathway: inject interface backpressures across
    the nozzle regimes and check the coupled mass flux matches the analytic
    quasi-1D value for the injected pressure.
    """
    p_c, t_c = 5.0e6, 3000.0
    nozzle = _nozzle(p_c, t_c)
    coupler = _coupler(nozzle, _jet(), fixed_point_iters=1)
    coupler.run(duration=0.004)  # settle

    g = GAMMA
    m_e_sup = _mach_from_area_ratio(3.0, g, supersonic=True)
    p_exit_sup = p_c * _t_of_m(m_e_sup, g) ** (g / (g - 1.0))
    m_e_sub = _mach_from_area_ratio(3.0, g, supersonic=False)
    p_exit_sub = p_c * _t_of_m(m_e_sub, g) ** (g / (g - 1.0))

    cases = [
        (0.5 * p_exit_sup, "supersonic"),
        (0.5 * (p_exit_sup + p_exit_sub), "shock"),
        (0.99 * p_c, "subsonic"),
        (1.01 * p_c, "blocked"),
    ]
    for p_b, regime in cases:
        coupler.backpressure = p_b
        log = coupler.step()
        st = nozzle.evaluate(backpressure_pa=p_b, t=nozzle.t)
        assert log.regime == regime
        assert log.mdot == pytest.approx(st.mdot, rel=1e-9)


# --------------------------------------------------------------------- #
# 3. Fixed-point convergence
# --------------------------------------------------------------------- #
def test_fixed_point_interface_residual() -> None:
    """Interface residuals decrease across Gauss-Seidel passes and stay small."""
    coupler = _coupler(_nozzle(2.0 * P_AMB, 600.0), _jet(), fixed_point_iters=4)
    coupler.run(duration=0.02)
    residuals = [log.fp_residuals[-1] for log in coupler.logs if log.fp_residuals]
    # Late-step interface residuals are small relative to the chamber scale.
    assert np.median(residuals[-20:]) < 0.1 * P_AMB
    # Two consecutive passes agree better than the first two (convergence).
    multi = [log.fp_residuals for log in coupler.logs if len(log.fp_residuals) >= 2]
    assert np.median([r[1] for r in multi[-20:]]) <= np.median(
        [r[0] for r in multi[-20:]]
    )


# --------------------------------------------------------------------- #
# 4. Exchange latency
# --------------------------------------------------------------------- #
def test_exchange_latency_submillisecond() -> None:
    """Pure boundary exchange stays sub-millisecond per macro step."""
    coupler = _coupler(_nozzle(2.0 * P_AMB, 600.0), _jet())
    coupler.run(duration=0.02)
    lat = coupler.history()["exchange_latency_ms"]
    # Pure boundary exchange (set_inlet + probe), excluding the 3D step.
    assert np.median(lat) < 1.0
