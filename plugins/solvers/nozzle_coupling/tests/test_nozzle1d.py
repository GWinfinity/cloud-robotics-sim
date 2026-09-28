"""Offline tests for the quasi-1D nozzle solver (no genesis needed).

Validation against closed-form compressible-flow results:
1. Choked mass flux vs the isentropic closed form.
2. Fully subsonic regime: exit pressure equals the backpressure.
3. Normal-shock regime: throat stays choked (mdot == choked value) and the
   flow turns fully subsonic once the backpressure reaches p_exit_sub.
4. Two-phase mixture rules (gamma drop + Sutton velocity-lag factor).
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

from plugins.solvers.nozzle_coupling.core.nozzle1d import (
    Nozzle1D,
    NozzleOptions,
    _area_mach_relation,
    _mach_from_area_ratio,
    _normal_shock_mach,
    _normal_shock_pressure_ratio,
    _t_of_m,
)

GAMMA = 1.25
MW = 0.025  # [kg/mol]
R_G = 8.314462618 / MW
P_C = 5.0e6  # [Pa]
T_C = 3000.0  # [K]
AT = 1.0e-4  # throat area [m^2]
EPS = 3.0  # area ratio


def _nozzle(**kw) -> Nozzle1D:
    opts = NozzleOptions(
        throat_area=AT,
        exit_area=AT * EPS,
        gamma=GAMMA,
        molecular_weight=MW,
        chamber_pressure=P_C,
        chamber_temperature=T_C,
    )
    for k, v in kw.items():
        setattr(opts, k, v)
    return Nozzle1D(opts)


# --------------------------------------------------------------------- #
# 1. Choked mass flux vs closed form
# --------------------------------------------------------------------- #
def test_choked_mdot_closed_form() -> None:
    """Choked mass flux matches the isentropic closed form."""
    n = _nozzle()
    g = GAMMA
    fac = (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    mdot_theory = AT * P_C / np.sqrt(R_G * T_C) * np.sqrt(g) * fac

    m_e = _mach_from_area_ratio(EPS, g, supersonic=True)
    p_exit_sup = P_C * _t_of_m(m_e, g) ** (g / (g - 1.0))
    st = n.evaluate(backpressure_pa=0.5 * p_exit_sup, t=0.0)

    assert st.regime == "supersonic"
    assert st.mdot == pytest.approx(mdot_theory, rel=1e-3)
    assert st.pressure == pytest.approx(p_exit_sup, rel=1e-3)
    # Supersonic exit and positive ideal velocity.
    assert st.mach > 1.0
    assert st.ideal_velocity > 0.0


# --------------------------------------------------------------------- #
# 2. Subsonic regime: p_e == p_back
# --------------------------------------------------------------------- #
def test_subsonic_exit_pressure_equals_backpressure() -> None:
    """Subsonic regime: exit pressure equals the backpressure."""
    n = _nozzle()
    g = GAMMA
    p_choked = _nozzle().choked_mdot(P_C, T_C, 0.0)

    # All above the subsonic-design pressure ratio (~0.976 for eps=3, g=1.25).
    for frac in (0.99, 0.985, 0.98):
        p_b = frac * P_C
        st = n.evaluate(backpressure_pa=p_b, t=0.0)
        assert st.regime == "subsonic"
        assert st.pressure == pytest.approx(p_b, rel=1e-9)
        assert st.mach < 1.0
        # Mass flux below the choked value and decreasing with backpressure.
        assert 0.0 < st.mdot < p_choked

    # Closed-form cross-check at one point.
    p_b = 0.99 * P_C
    st = n.evaluate(backpressure_pa=p_b, t=0.0)
    mach = np.sqrt(((p_b / P_C) ** (-(g - 1.0) / g) - 1.0) * 2.0 / (g - 1.0))
    T_e = T_C * _t_of_m(mach, g)
    rho_e = p_b / (R_G * T_e)
    u_e = mach * np.sqrt(g * R_G * T_e)
    mdot_theory = rho_e * u_e * AT * EPS
    assert st.mdot == pytest.approx(mdot_theory, rel=1e-9)
    assert st.temperature == pytest.approx(T_e, rel=1e-9)


# --------------------------------------------------------------------- #
# 3. Normal-shock regime: choked mdot preserved; subsonic transition
# --------------------------------------------------------------------- #
def test_shock_regime_and_regime_transitions() -> None:
    """Shock regime keeps the choked mdot; regime transitions are monotone."""
    n = _nozzle()
    g = GAMMA
    m_e_sup = _mach_from_area_ratio(EPS, g, supersonic=True)
    p_exit_sup = P_C * _t_of_m(m_e_sup, g) ** (g / (g - 1.0))
    m_e_sub = _mach_from_area_ratio(EPS, g, supersonic=False)
    p_exit_sub = P_C * _t_of_m(m_e_sub, g) ** (g / (g - 1.0))
    mdot_choked = n.choked_mdot(P_C, T_C, 0.0)

    # Between the two design pressures: shock in the divergent, throat choked.
    assert p_exit_sup < p_exit_sub
    st = n.evaluate(backpressure_pa=0.5 * (p_exit_sup + p_exit_sub), t=0.0)
    assert st.regime == "shock"
    assert st.mdot == pytest.approx(mdot_choked, rel=1e-9)
    assert st.pressure == pytest.approx(0.5 * (p_exit_sup + p_exit_sub), rel=1e-6)
    assert st.mach < 1.0  # post-shock subsonic exit

    # Just above p_exit_sub: flow turns subsonic; mdot below choked.
    st2 = n.evaluate(backpressure_pa=p_exit_sub * 1.001, t=0.0)
    assert st2.regime == "subsonic"
    assert st2.mdot < mdot_choked
    # mdot decreases monotonically with backpressure in the subsonic regime.
    mdots = [
        n.evaluate(backpressure_pa=f * P_C, t=0.0).mdot for f in (0.98, 0.985, 0.99)
    ]
    assert all(b < a for a, b in zip(mdots, mdots[1:]))

    # Backpressure at/above chamber pressure: blocked.
    st3 = n.evaluate(backpressure_pa=1.05 * P_C, t=0.0)
    assert st3.regime == "blocked"
    assert st3.mdot == 0.0


def test_shock_position_recoveries_exit_pressure() -> None:
    """The iterated shock position must recompress to the imposed backpressure.

    Below the lip-shock pressure the solution clamps to the lip (no in-nozzle
    normal shock exists there; the rest of the pressure match is external).
    """
    n = _nozzle()
    g = GAMMA
    m_e_sup = _mach_from_area_ratio(EPS, g, supersonic=True)
    p_exit_sup = P_C * _t_of_m(m_e_sup, g) ** (g / (g - 1.0))
    m_e_sub = _mach_from_area_ratio(EPS, g, supersonic=False)
    p_exit_sub = P_C * _t_of_m(m_e_sub, g) ** (g / (g - 1.0))

    def p_exit_after_shock(eps_s: float) -> float:
        m1 = _mach_from_area_ratio(eps_s, g, supersonic=True)
        p1 = P_C * _t_of_m(m1, g) ** (g / (g - 1.0))
        p2 = p1 * _normal_shock_pressure_ratio(m1, g)
        m2 = _normal_shock_mach(m1, g)
        m_out = _mach_from_area_ratio(
            (EPS / eps_s) * _area_mach_relation(m2, g), g, supersonic=False
        )
        return p2 * (_t_of_m(m_out, g) / _t_of_m(m2, g)) ** (g / (g - 1.0))

    p_lip = p_exit_after_shock(EPS)
    assert p_exit_sup < p_lip < p_exit_sub

    # Interior-shock band: the shock position recovers p_back exactly.
    for frac in (0.7, 0.85):
        p_b = p_lip + frac * (p_exit_sub - p_lip)
        eps_s = n._shock_position(P_C, p_b, g)
        assert 1.0 < eps_s < EPS
        assert p_exit_after_shock(eps_s) == pytest.approx(p_b, rel=1e-3)

    # Lip-clamped band: shock sits at the lip and p_exit stays at p_lip > p_b.
    for frac in (0.1, 0.5):
        p_b = p_exit_sup + frac * (p_lip - p_exit_sup)
        eps_s = n._shock_position(P_C, p_b, g)
        assert eps_s == pytest.approx(EPS, rel=1e-9)
        assert p_exit_after_shock(eps_s) == pytest.approx(p_lip, rel=1e-9)
        st = n.evaluate(backpressure_pa=p_b, t=0.0)
        assert st.regime == "shock"
        assert st.pressure == pytest.approx(p_lip, rel=1e-6)
        assert st.pressure > p_b


# --------------------------------------------------------------------- #
# 4. Two-phase mixture rules
# --------------------------------------------------------------------- #
def test_two_phase_gamma_drop_and_lag_factor() -> None:
    """Two-phase mixture rules lower gamma and the lag factor."""
    alpha = 0.2
    phi = 0.8
    n = _nozzle(condensed_fraction=alpha, particle_lag_factor=phi)
    g_eff, r_eff, psi = n.effective_properties(alpha)

    cp_g = GAMMA * R_G / (GAMMA - 1.0)
    cv_g = R_G / (GAMMA - 1.0)
    cp_m = (1 - alpha) * cp_g + alpha * n.o.particle_cp
    cv_m = (1 - alpha) * cv_g + alpha * n.o.particle_cp
    assert g_eff == pytest.approx(cp_m / cv_m, rel=1e-12)
    assert g_eff < GAMMA
    assert r_eff == pytest.approx((1 - alpha) * R_G, rel=1e-12)
    assert psi == pytest.approx((1 - alpha) + alpha * phi, rel=1e-12)

    # Velocity carries the lag factor relative to the ideal velocity.
    st = n.evaluate(backpressure_pa=1.0e5, t=0.0)
    assert st.velocity == pytest.approx(st.ideal_velocity * psi, rel=1e-9)

    # The same alpha through the chamber curve must agree.
    t = np.array([0.0, 1.0])
    n2 = _nozzle(
        curve_t=t,
        curve_p=np.array([P_C, P_C]),
        curve_T=np.array([T_C, T_C]),
        curve_alpha=np.array([alpha, alpha]),
        particle_lag_factor=phi,
    )
    st2 = n2.evaluate(backpressure_pa=1.0e5, t=0.4)
    assert st2.velocity == pytest.approx(st.velocity, rel=1e-9)
    assert st2.alpha == pytest.approx(alpha, rel=1e-12)


def test_area_mach_relation_roundtrip() -> None:
    """A/A* inversion round-trips on both branches."""
    for m in (0.3, 0.9, 1.5, 3.0, 6.0):
        eps = float(_area_mach_relation(m, GAMMA))
        branch = m >= 1.0
        assert _mach_from_area_ratio(eps, GAMMA, supersonic=branch) == pytest.approx(
            m, rel=1e-6
        )
