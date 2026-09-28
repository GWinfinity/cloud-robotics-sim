"""Quasi-1D converging-diverging nozzle flow solver (genesis-free, pure NumPy).

Textbook compressible-flow model in the spirit of Sutton, *Rocket Propulsion
Elements* (ch. 3): the chamber supplies stagnation state ``p_c(t), T_c(t)``
driven by user-provided curves/tables, the nozzle is quasi-steady isentropic
flow with a Cd, and the flow regime is selected by the backpressure fed back
from the 3D side:

* ``p_back <= p_exit_sup``  : choked throat, supersonic divergent
  (under-expanded; exit pressure is the fully-expanded supersonic value).
* ``p_exit_sub < p_back < p_exit_sup`` : normal shock inside the divergent
  section; the throat stays choked, so the mass flux equals the choked value.
  The shock area ratio is iterated so the post-shock isentropic recompression
  matches ``p_back`` at the exit.
* ``p_back >= p_exit_sub``  : fully subsonic, ``p_e = p_back``, mass flux
  drops below the choked value and goes to zero as ``p_back -> p_c``.

Two-phase exhaust (condensed particulates, e.g. metal-oxide smoke) is treated
with the standard engineering mixture rules:

* effective ratio of specific heats / gas constant from the condensed mass
  fraction ``alpha`` and a solid heat capacity (Sutton mixture rule),
* a velocity-lag correction factor ``psi = (1 - alpha) + alpha * phi``
  applied to the ideal exhaust velocity (particles move at ``phi`` times the
  gas velocity and contribute no momentum lag-free).

All geometry and thermodynamic inputs are user parameters; this module is a
generic compressible-flow component and contains no propellant or motor data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_UNIVERSAL = 8.314462618  # universal gas constant [J/(mol K)]


@dataclass
class NozzleOptions:
    """Options for the quasi-1D nozzle solver.

    Parameters
    ----------
    throat_area, exit_area :
        Throat / exit cross-section areas [m^2]; ``exit_area > throat_area``.
    gamma :
        Gas-phase ratio of specific heats [-].
    molecular_weight :
        Gas-phase molecular weight [kg/mol] (sets the gas constant).
    chamber_pressure, chamber_temperature :
        Constant chamber stagnation state [Pa], [K]; used when no curves are
        given (or as the fallback outside the curve time range).
    curve_t, curve_p, curve_T, curve_alpha :
        Optional tabulated chamber history (1-D arrays, same length, seconds /
        Pa / K / -) driving the transient; linearly interpolated, clamped
        outside the range.
    condensed_fraction :
        Condensed-phase mass fraction ``alpha`` in [0, 1) when no curve given.
    particle_cp :
        Solid-phase specific heat for the mixture rule [J/(kg K)].
    particle_lag_factor :
        Particle-to-gas velocity ratio ``phi`` in (0, 1] (velocity lag).
    discharge_coeff :
        Throat discharge coefficient ``Cd`` [-].
    thrust_lag_tau :
        Optional first-order lag time constant on the delivered mass flux and
        exhaust velocity, e.g. to mimic ignition/start transients [s];
        ``0`` disables (no internal state).
    """

    throat_area: float = 1.0e-4
    exit_area: float = 3.0e-4
    gamma: float = 1.25
    molecular_weight: float = 0.025  # [kg/mol]
    chamber_pressure: float = 5.0e6  # [Pa]
    chamber_temperature: float = 3000.0  # [K]
    curve_t: np.ndarray | None = None
    curve_p: np.ndarray | None = None
    curve_T: np.ndarray | None = None  # noqa: N815 (physics naming)
    curve_alpha: np.ndarray | None = None
    condensed_fraction: float = 0.0
    particle_cp: float = 1000.0  # [J/(kg K)]
    particle_lag_factor: float = 1.0
    discharge_coeff: float = 1.0
    thrust_lag_tau: float = 0.0  # [s]


@dataclass
class ExitState:
    """Nozzle exit-plane state for one evaluation (passing to the 3D side)."""

    mdot: float  # mass flux [kg/s]
    velocity: float  # lag-corrected exhaust (momentum) velocity [m/s]
    ideal_velocity: float  # single-phase-equivalent ideal velocity [m/s]
    temperature: float  # exit static temperature [K]
    pressure: float  # exit static pressure [Pa]
    mach: float  # exit Mach number [-]
    alpha: float  # condensed-phase mass fraction [-]
    gamma_eff: float  # two-phase effective ratio of specific heats [-]
    r_eff: float  # two-phase effective gas constant [J/(kg K)]
    regime: str  # one of "supersonic" | "shock" | "subsonic" | "blocked"


def _area_mach_relation(m, g: float):
    """Area ratio A/A* as a function of Mach (isentropic flow).

    Scalar inputs take a pure-float fast path (math.pow); array inputs use
    the vectorised NumPy path. This function is hot inside the shock-position
    bisection, so the scalar path avoids per-call ``np.asarray`` overhead.
    """
    expo = (g + 1.0) / (2.0 * (g - 1.0))
    if np.isscalar(m):
        mf = float(m)  # type: ignore[arg-type]
        out: float = (1.0 / mf) * (
            2.0 / (g + 1.0) * (1.0 + (g - 1.0) / 2.0 * mf * mf)
        ) ** expo
        return out
    m = np.asarray(m, dtype=np.float64)
    return (1.0 / m) * ((2.0 / (g + 1.0)) * (1.0 + (g - 1.0) / 2.0 * m**2)) ** expo


def _mach_from_area_ratio(eps: float, g: float, supersonic: bool) -> float:
    """Invert A/A* = eps for the sub- or supersonic branch (bisection)."""
    # A/A* decreases monotonically to 1 on the subsonic branch and increases
    # monotonically from 1 on the supersonic branch; bisection direction
    # differs between the two.
    if supersonic:
        lo, hi = 1.0 + 1.0e-12, 200.0

        def below(v: float) -> bool:
            return v < eps  # need larger M

    else:
        lo, hi = 1.0e-9, 1.0 - 1.0e-12

        def below(v: float) -> bool:
            return v > eps  # need larger M

    for _ in range(80):  # 2^-80 bracket, more than float64 precision needs
        mid = 0.5 * (lo + hi)
        if below(_area_mach_relation(mid, g)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _normal_shock_pressure_ratio(m1: float, g: float) -> float:
    """p2/p1 across a normal shock."""
    return (2.0 * g * m1**2 - (g - 1.0)) / (g + 1.0)


def _normal_shock_temperature_ratio(m1: float, g: float) -> float:
    """T2/T1 across a normal shock."""
    density_ratio = ((g + 1.0) * m1**2) / (2.0 + (g - 1.0) * m1**2)
    return _normal_shock_pressure_ratio(m1, g) / density_ratio


def _normal_shock_mach(m1: float, g: float) -> float:
    """Downstream Mach of a normal shock."""
    return float(
        np.sqrt((1.0 + (g - 1.0) / 2.0 * m1**2) / (g * m1**2 - (g - 1.0) / 2.0))
    )


def _t_of_m(m: float, g: float) -> float:
    """Isentropic T/T0 ratio at Mach ``m``."""
    return 1.0 / (1.0 + (g - 1.0) / 2.0 * m**2)


def _cp_of(g: float, r_gas: float) -> float:
    """Specific heat at constant pressure."""
    return g * r_gas / (g - 1.0)


class Nozzle1D:
    """Quasi-1D C-D nozzle with curve-driven chamber and backpressure input."""

    def __init__(self, options: NozzleOptions) -> None:
        o = options
        if o.exit_area <= o.throat_area:
            raise ValueError("exit_area must exceed throat_area")
        if not 1.0 < o.gamma <= 2.0:
            raise ValueError("gamma must be in (1, 2]")
        if not 0.0 <= o.condensed_fraction < 1.0:
            raise ValueError("condensed_fraction must be in [0, 1)")
        self.o = o
        self.eps = o.exit_area / o.throat_area
        self.R_g = R_UNIVERSAL / o.molecular_weight
        self._validate_curves()
        # Optional first-order lag state (thrust transient).
        self._lag_mdot = 0.0
        self._lag_u = 0.0
        self.t = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------------ #
    def _validate_curves(self) -> None:
        o = self.o
        curves = {
            "curve_p": o.curve_p,
            "curve_T": o.curve_T,
            "curve_alpha": o.curve_alpha,
        }
        if o.curve_t is None:
            if any(c is not None for c in curves.values()):
                raise ValueError("curve_t is required with any chamber curve")
            return
        t = np.asarray(o.curve_t, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("curve_t must be increasing with >= 2 points")
        for name, c in curves.items():
            if c is None:
                continue
            if np.asarray(c).shape != t.shape:
                raise ValueError(f"{name} must match curve_t shape")

    def _curve(self, t: float, values: np.ndarray | None, const: float) -> float:
        if values is None:
            return float(const)
        return float(np.interp(t, np.asarray(self.o.curve_t), values))

    # ------------------------------------------------------------------ #
    def chamber_state(self, t: float) -> tuple[float, float, float]:
        """Stagnation state (p_c [Pa], T_c [K], alpha [-]) at time ``t``."""
        o = self.o
        p_c = self._curve(t, o.curve_p, o.chamber_pressure)
        T_c = self._curve(t, o.curve_T, o.chamber_temperature)
        alpha = float(
            np.clip(
                self._curve(t, o.curve_alpha, o.condensed_fraction),
                0.0,
                1.0 - 1e-9,
            )
        )
        return p_c, T_c, alpha

    def effective_properties(self, alpha: float) -> tuple[float, float, float]:
        """Two-phase effective (gamma, R, lag factor psi) at fraction ``alpha``.

        Mixture rule (Sutton): per unit total mass,
        ``c_p = (1-a) c_p,g + a c_s``, ``c_v = (1-a) c_v,g + a c_s`` with the
        solid c_v = c_p (incompressible), so gamma drops below the gas value.
        """
        o = self.o
        g, R = o.gamma, self.R_g
        cp_g = g * R / (g - 1.0)
        cv_g = R / (g - 1.0)
        cp_m = (1.0 - alpha) * cp_g + alpha * o.particle_cp
        cv_m = (1.0 - alpha) * cv_g + alpha * o.particle_cp
        g_eff = cp_m / cv_m
        r_eff = (1.0 - alpha) * R
        psi = (1.0 - alpha) + alpha * o.particle_lag_factor
        return g_eff, r_eff, psi

    # ------------------------------------------------------------------ #
    def choked_mdot(self, p_c: float, temp_c: float, alpha: float) -> float:
        """Choked-throat mass flux (shock in the divergent does not change it)."""
        g, r_eff, _ = self.effective_properties(alpha)
        fac = (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        return float(
            self.o.discharge_coeff
            * self.o.throat_area
            * p_c
            / np.sqrt(r_eff * temp_c)
            * np.sqrt(g)
            * fac
        )

    def evaluate(self, backpressure_pa: float, t: float) -> ExitState:
        """Quasi-steady nozzle state for chamber state at ``t`` and ``p_back``.

        Parameters
        ----------
        backpressure_pa :
            Static pressure at the nozzle exit plane fed back from the 3D side
            [Pa] (absolute).
        t :
            Evaluation time [s] (selects the chamber curve point).
        """
        o = self.o
        p_back = max(float(backpressure_pa), 1.0)
        p_c, T_c, alpha = self.chamber_state(t)
        g, r_eff, psi = self.effective_properties(alpha)

        if p_back >= p_c:
            return ExitState(
                mdot=0.0,
                velocity=0.0,
                ideal_velocity=0.0,
                temperature=T_c,
                pressure=p_c,
                mach=0.0,
                alpha=alpha,
                gamma_eff=g,
                r_eff=r_eff,
                regime="blocked",
            )

        m_e_sup = _mach_from_area_ratio(self.eps, g, supersonic=True)
        p_exit_sup = p_c * _t_of_m(m_e_sup, g) ** (g / (g - 1.0))
        # Subsonic-branch exit pressure at the full area ratio: the boundary
        # above which the divergent flow is fully subsonic.
        m_e_sub = _mach_from_area_ratio(self.eps, g, supersonic=False)
        p_exit_sub = p_c * _t_of_m(m_e_sub, g) ** (g / (g - 1.0))

        if p_back <= p_exit_sup:
            # Supersonic divergent, choked throat.
            mdot = self.choked_mdot(p_c, T_c, alpha)
            mach = m_e_sup
            regime = "supersonic"
            t_ratio = _t_of_m(mach, g)
            T_e = T_c * t_ratio
            p_e = p_c * t_ratio ** (g / (g - 1.0))
        elif p_back <= p_exit_sub:
            # Normal shock inside the divergent; throat remains choked. The
            # shock area ratio is solved from p_back, but the in-nozzle
            # solution only exists down to the lip-shock pressure (a normal
            # shock cannot sit *inside* for lower backpressures); below that
            # the shock clamps to the lip and the remaining pressure match
            # happens outside, in the 3D plenum.
            mdot = self.choked_mdot(p_c, T_c, alpha)
            eps_s = self._shock_position(p_c, p_back, g)
            m1 = _mach_from_area_ratio(eps_s, g, supersonic=True)
            m2 = _normal_shock_mach(m1, g)
            # Post-shock isentropic deceleration on the post-shock critical
            # area A2* = A_s / G(M2): G(m_exit) = (eps/eps_s) * G(M2).
            m_exit = _mach_from_area_ratio(
                (self.eps / eps_s) * _area_mach_relation(m2, g),
                g,
                supersonic=False,
            )
            t1 = _t_of_m(m1, g)
            T1 = T_c * t1
            T2 = T1 * _normal_shock_temperature_ratio(m1, g)
            T_e = T2 * (_t_of_m(m_exit, g) / _t_of_m(m2, g))
            p1 = p_c * t1 ** (g / (g - 1.0))
            # Post-shock isentropic deceleration: pressure rises as t(M)^{g/(g-1)}.
            p_e = (
                p1
                * _normal_shock_pressure_ratio(m1, g)
                * (_t_of_m(m_exit, g) / _t_of_m(m2, g)) ** (g / (g - 1.0))
            )
            mach = m_exit
            regime = "shock"
        else:
            # Fully subsonic: exit pressure equals the backpressure.
            p_ratio = p_back / p_c
            mach = float(np.sqrt((p_ratio ** (-(g - 1.0) / g) - 1.0) * 2.0 / (g - 1.0)))
            t_ratio = _t_of_m(mach, g)
            T_e = T_c * t_ratio
            p_e = p_back
            rho_e = p_e / (r_eff * T_e)
            u_e = mach * np.sqrt(g * r_eff * T_e)
            mdot = o.discharge_coeff * rho_e * u_e * o.exit_area
            regime = "subsonic"
            return ExitState(
                mdot=float(mdot),
                velocity=float(u_e * psi),
                ideal_velocity=float(u_e),
                temperature=float(T_e),
                pressure=float(p_e),
                mach=float(mach),
                alpha=alpha,
                gamma_eff=g,
                r_eff=r_eff,
                regime=regime,
            )

        # Isentropic exit velocity from the enthalpy drop, then lag factor.
        u_id = np.sqrt(2.0 * _cp_of(g, r_eff) * (T_c - T_e)) if T_c > T_e else 0.0
        return ExitState(
            mdot=float(mdot),
            velocity=float(u_id * psi),
            ideal_velocity=float(u_id),
            temperature=float(T_e),
            pressure=float(p_e),
            mach=float(mach),
            alpha=alpha,
            gamma_eff=g,
            r_eff=r_eff,
            regime=regime,
        )

    def _shock_position(self, p_c: float, p_back: float, g: float) -> float:
        """Divergent area ratio at the normal shock, matching p_e = p_back.

        Bisection on the shock area ratio ``eps_s`` in (1, eps]: pre-shock
        supersonic branch from the throat, normal shock, then subsonic-branch
        isentropic recompression from ``eps_s`` to ``eps``.
        """

        def p_exit(eps_s: float) -> float:
            m1 = _mach_from_area_ratio(eps_s, g, supersonic=True)
            p1 = p_c * _t_of_m(m1, g) ** (g / (g - 1.0))
            p2 = p1 * _normal_shock_pressure_ratio(m1, g)
            m2 = _normal_shock_mach(m1, g)
            # Decelerate on the post-shock critical area A2* = A_s / G(M2).
            m_out = _mach_from_area_ratio(
                (self.eps / eps_s) * _area_mach_relation(m2, g),
                g,
                supersonic=False,
            )
            pe: float = p2 * (_t_of_m(m_out, g) / _t_of_m(m2, g)) ** (g / (g - 1.0))
            return pe

        lo, hi = 1.0 + 1.0e-9, self.eps - 1.0e-9
        # p_exit is maximised by a throat shock (lo side, p -> p_exit_sub)
        # and minimised by the lip shock (hi side). Below the lip-shock
        # pressure no in-nozzle normal-shock solution exists: clamp the shock
        # to the exit lip (the external pressure match then happens outside,
        # handled by the 3D side).
        if p_exit(hi) > p_back:
            return self.eps
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            # p_exit decreases as the shock moves downstream (larger eps_s):
            # too little recompression -> push the shock upstream (hi = mid).
            if p_exit(mid) < p_back:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ #
    def step(self, backpressure_pa: float, dt: float) -> ExitState:
        """Advance one step (chamber curve time + optional thrust lag).

        The quasi-steady state is evaluated at the start-of-step time; with
        ``thrust_lag_tau > 0`` the delivered mass flux and exhaust velocity
        follow a first-order lag toward the quasi-steady value.
        """
        st = self.evaluate(backpressure_pa, self.t)
        tau = self.o.thrust_lag_tau
        if tau > 0.0:
            a = min(1.0, dt / tau)
            self._lag_mdot += a * (st.mdot - self._lag_mdot)
            self._lag_u += a * (st.velocity - self._lag_u)
            st = ExitState(
                mdot=self._lag_mdot,
                velocity=self._lag_u,
                ideal_velocity=st.ideal_velocity,
                temperature=st.temperature,
                pressure=st.pressure,
                mach=st.mach,
                alpha=st.alpha,
                gamma_eff=st.gamma_eff,
                r_eff=st.r_eff,
                regime=st.regime,
            )
        self.t += dt
        self.n_steps += 1
        return st

    # ------------------------------------------------------------------ #
    @property
    def dt(self) -> float:
        """Nominal evaluation step; the coupler drives the actual stepping."""
        return 1.0e-4

    def get_state(self) -> tuple[float, int, float, float]:
        return self.t, self.n_steps, self._lag_mdot, self._lag_u

    def set_state(self, state: tuple[float, int, float, float]) -> None:
        self.t, self.n_steps, self._lag_mdot, self._lag_u = state
