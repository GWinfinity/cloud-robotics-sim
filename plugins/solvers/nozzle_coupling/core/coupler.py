"""1D nozzle <-> 3D jet bidirectional coupling coordinator.

Topology
--------
    chamber curves p_c(t), T_c(t), alpha(t)
        --> [1D quasi-steady nozzle: choked / shocked / subsonic regime]
        --> mdot(t), u_e(t), T_e(t), alpha(t), r_eff ══ macro-step exchange ══>
        [3D anelastic jet/plenum]
        --> exit-plane static pressure p_back --> back to the 1D nozzle

Exchange per macro step (period ``macro_dt``, the 3D time step):
    1D -> 3D : the nozzle state is sampled every 1D sub-step over the macro
               window; trapezoidal means of the mass flux, temperature,
               condensed fraction and effective gas constant are imposed on
               the 3D inlet patch (with a startup ramp).
    3D -> 1D : the absolute static pressure at the 3D inlet plane, returned
               to the 1D nozzle as backpressure, selects the flow regime
               (supersonic / shock-in-divergent / subsonic / blocked).

Interface stability: Gauss-Seidel fixed-point iteration per macro step — the
whole macro window is replayed with the backpressure updated from the
previous pass until the interface residual falls below ``fixed_point_tol_pa``
or ``fixed_point_iters`` is reached.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np

from .jet3d import Jet3D
from .nozzle1d import ExitState, Nozzle1D


@dataclass
class NozzleCouplingOptions:
    """Options for the nozzle <-> jet coupling coordinator."""

    macro_dt: float = 1.0e-3  # exchange period = 3D time step [s]
    n_substeps: int = 10  # 1D samples per macro window
    fixed_point_iters: int = 2  # Gauss-Seidel passes per macro step
    fixed_point_tol_pa: float = 10.0  # interface backpressure residual [Pa]
    inlet_ramp_time: float = 0.01  # startup ramp on the 3D inlet [s]
    backpressure_max_rate_pa_s: float = 1.0e6  # slew limit on the 3D -> 1D
    # backpressure exchange [Pa/s]. The quasi-steady nozzle is infinitely
    # stiff (mdot(p_back) jumps to zero at p_back = p_c), so an unbounded
    # interface limit-cycles (blocked <-> choked) whenever the 3D pressure
    # probe overshoots during a re-start (numerical spikes reach GPa/s while
    # physical plenum pressurization stays well below this limit).
    inlet_mdot_slew_kg_s2: float = 1.0e2  # slew limit on the 1D -> 3D mass
    # flux exchange [kg/s^2]. An impulsive restart of the inlet mass flux
    # (blocked -> choked within one macro step) spikes the 3D pressure probe
    # by tens of kPa; rate-limiting the inlet reproduces the physical feed-
    # system/thrust lag and keeps the coupled iteration a contraction.


@dataclass
class NozzleMacroStepLog:
    """Diagnostics of one coupled macro step."""

    t: float
    mdot: float  # final 1D mass flux [kg/s]
    backpressure_pa: float  # 3D exit-plane pressure fed back [Pa]
    exit_pressure_pa: float  # 1D nozzle exit static pressure [Pa]
    exit_mach: float  # 1D nozzle exit Mach number [-]
    regime: str  # 1D nozzle flow regime
    fp_residuals: list[float]  # |p_back_new - p_back_old| per pass [Pa]
    mdot_mean: float  # window-averaged mass flux sent to the 3D side [kg/s]
    exchange_latency_ms: float  # pure boundary-exchange time [ms]
    wall_time_ms: float  # total macro step wall time [ms]


class NozzleCoupler:
    """Bidirectional quasi-1D nozzle <-> 3D jet coupling coordinator."""

    def __init__(
        self,
        nozzle: Nozzle1D,
        jet: Jet3D,
        options: NozzleCouplingOptions,
    ) -> None:
        self.nozzle = nozzle
        self.jet = jet
        self.o = options
        if options.macro_dt <= 0.0:
            raise ValueError("macro_dt must be positive")
        if options.n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        self.dt_1d = options.macro_dt / options.n_substeps
        # Initial interface state: the 3D plenum at rest, ambient pressure.
        self.backpressure = jet.o.ambient_pressure
        self._mdot_applied = 0.0
        self.logs: list[NozzleMacroStepLog] = []

    # ------------------------------------------------------------------ #
    def step(self) -> NozzleMacroStepLog:
        """Advance one macro step (n 1D substeps + one 3D step)."""
        noz_state = self.nozzle.get_state()
        jet_state = self.jet.get_state()
        t0 = self.nozzle.t

        p_back = self.backpressure
        mdot_applied = self._mdot_applied
        residuals: list[float] = []
        exchange_s = 0.0
        wall_start = _time.perf_counter()
        mdot_mean = 0.0
        st: ExitState | None = None

        for _ in range(self.o.fixed_point_iters):
            # --- pass 1: 1D nozzle sub-cycling at the current backpressure ---
            self.nozzle.set_state(noz_state)
            # The applied inlet mass flux is an interface quantity: within one
            # macro step it is a single slew-clamped value recomputed from the
            # macro-start state each pass (deterministic across passes).
            mdot_applied = self._mdot_applied
            st0 = self.nozzle.evaluate(backpressure_pa=p_back, t=t0)
            ts = [t0]
            mdots = [st0.mdot]
            temps = [st0.temperature]
            alphas = [st0.alpha]
            rs = [st0.r_eff]
            for _ in range(self.o.n_substeps):
                st = self.nozzle.step(backpressure_pa=p_back, dt=self.dt_1d)
                ts.append(self.nozzle.t)
                mdots.append(st.mdot)
                temps.append(st.temperature)
                alphas.append(st.alpha)
                rs.append(st.r_eff)
            window = self.o.macro_dt
            ts_a = np.asarray(ts)
            mdot_mean = float(np.trapezoid(np.asarray(mdots), ts_a)) / window
            t_mean = float(np.trapezoid(np.asarray(temps), ts_a)) / window
            alpha_mean = float(np.trapezoid(np.asarray(alphas), ts_a)) / window
            r_mean = float(np.trapezoid(np.asarray(rs), ts_a)) / window

            # --- pass 2: 3D step with the window-averaged inlet state ---
            t_start = _time.perf_counter()
            ramp = min(1.0, (t0 + window) / max(self.o.inlet_ramp_time, 1e-9))
            # Slew-limited inlet (feed-system/thrust lag): an impulsive
            # restart of the mass flux would spike the 3D pressure probe.
            mdot_target = ramp * mdot_mean
            dmdt_max = self.o.inlet_mdot_slew_kg_s2 * window
            mdot_applied = float(
                np.clip(mdot_target, mdot_applied - dmdt_max, mdot_applied + dmdt_max)
            )
            self.jet.set_state(jet_state)
            self.jet.set_inlet(
                mdot_applied,
                stagnation_temp=t_mean,
                condensed_fraction=alpha_mean,
                r_eff=r_mean,
            )
            exchange_s += _time.perf_counter() - t_start
            self.jet.step(window)

            t_start = _time.perf_counter()
            p_new = self.jet.exit_plane_pressure()
            exchange_s += _time.perf_counter() - t_start

            residuals.append(abs(p_new - p_back))
            p_back = p_new
            if residuals[-1] < self.o.fixed_point_tol_pa:
                break

        self.backpressure = p_back
        self._mdot_applied = mdot_applied
        wall_s = _time.perf_counter() - wall_start
        log = NozzleMacroStepLog(
            t=t0 + self.o.macro_dt,
            mdot=float(st.mdot if st is not None else 0.0),
            backpressure_pa=p_back,
            exit_pressure_pa=float(st.pressure if st is not None else 0.0),
            exit_mach=float(st.mach if st is not None else 0.0),
            regime=st.regime if st is not None else "unknown",
            fp_residuals=residuals,
            mdot_mean=mdot_mean,
            exchange_latency_ms=exchange_s * 1.0e3,
            wall_time_ms=wall_s * 1.0e3,
        )
        self.logs.append(log)
        return log

    # ------------------------------------------------------------------ #
    def run(self, duration: float) -> list[NozzleMacroStepLog]:
        n = int(round(duration / self.o.macro_dt))
        return [self.step() for _ in range(n)]

    def history(self) -> dict[str, np.ndarray]:
        """Return the per-macro-step history as arrays."""
        return {
            "t": np.array([log.t for log in self.logs]),
            "mdot": np.array([log.mdot for log in self.logs]),
            "backpressure_pa": np.array([log.backpressure_pa for log in self.logs]),
            "exit_pressure_pa": np.array([log.exit_pressure_pa for log in self.logs]),
            "exit_mach": np.array([log.exit_mach for log in self.logs]),
            "mdot_mean": np.array([log.mdot_mean for log in self.logs]),
            "regime": np.array([log.regime for log in self.logs]),
            "exchange_latency_ms": np.array(
                [log.exchange_latency_ms for log in self.logs]
            ),
            "wall_time_ms": np.array([log.wall_time_ms for log in self.logs]),
        }
