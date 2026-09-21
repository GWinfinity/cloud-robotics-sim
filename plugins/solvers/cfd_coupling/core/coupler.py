"""1D-3D bidirectional coupling coordinator (the competition "coupler").

Topology
--------
    reservoir --[1D pipe: MOC]--> nozzle/valve --[mass + temperature]--> 3D plenum --> outlet

Exchange per macro step (period = ``macro_dt``, the 3D time step):
    1D -> 3D : nozzle discharge Q(t) sampled every 1D substep, linearly
               interpolated onto the 3D inlet as uniform patch velocity,
               plus the fluid temperature.
    3D -> 1D : mean gauge pressure on the 3D inlet plane, returned to the
               1D nozzle as backpressure head (rho*g*H).

Time-step coordination: the 1D MOC solver runs with dt_1d = dx/a (small),
sub-cycling n = macro_dt / dt_1d times per macro step.

Stability of the coupling is obtained with a fixed-point (Gauss-Seidel)
iteration per macro step: the whole macro window is replayed with the
backpressure updated from the previous pass, until the interface head
residual falls below ``fixed_point_tol`` or ``fixed_point_iters`` is reached.

The valve is a 0D control-logic device: a linear closure schedule evaluated
at absolute simulation time, i.e. millisecond-scale control actions enter
the 1D side immediately at the next substep.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np

from .cfd3d import CFD3D
from .pipe1d import Pipe1D


@dataclass
class CouplingOptions:
    """Options for the 1D-3D coupler."""

    macro_dt: float = 0.005  # exchange period = 3D time step [s]
    fixed_point_iters: int = 2  # Gauss-Seidel iterations per macro step
    fixed_point_tol: float = 1.0e-3  # interface head residual [m]
    valve_closure_start: float = 0.0  # control-logic event time (valve begins
    # closing here); between events the valve stays at its last fraction [-]
    valve_closure_duration: float | None = None  # linear closure duration [s]
    min_valve_fraction: float = 0.02  # minimum opening (leakage) [-]
    inlet_ramp_time: float = 0.05  # startup ramp on the 3D inlet velocity [s]


@dataclass
class MacroStepLog:
    """Diagnostics of one macro step."""

    t: float
    nozzle_flow: float  # final 1D nozzle discharge [m^3/s]
    plenum_head: float  # 3D inlet-plane head fed back [m]
    valve_fraction: float
    fp_residuals: list[float]  # |head_new - head_old| per Gauss-Seidel pass
    fp_flow_residuals: list[float]  # |Q_mean_new - Q_mean_old| per pass
    exchange_latency_ms: float  # pure boundary-exchange time [ms]
    wall_time_ms: float  # total macro step wall time [ms]


class Coupler:
    """Bidirectional 1D(MOC) <-> 3D(CFD) coupling coordinator."""

    def __init__(self, pipe: Pipe1D, cfd: CFD3D, options: CouplingOptions) -> None:
        self.pipe = pipe
        self.cfd = cfd
        self.o = options
        n = round(options.macro_dt / pipe.dt)
        if n < 1:
            raise ValueError("macro_dt must be >= the 1D MOC time step")
        self.macro_dt = n * pipe.dt  # snap to the 1D step grid
        self.n_substeps = n

        self.plenum_head = cfd.inlet_pressure_head()  # initial interface state
        self._last_q_mean: float | None = None
        self.logs: list[MacroStepLog] = []

    # ------------------------------------------------------------------ #
    def valve_fraction(self, t: float) -> float:
        """Valve schedule: fully open until ``closure_start``, then linear."""
        t0 = self.o.valve_closure_start
        tc = self.o.valve_closure_duration
        if tc is None or t < t0:
            return 1.0
        if t >= t0 + tc:
            return self.o.min_valve_fraction
        return max(1.0 - (t - t0) / tc, self.o.min_valve_fraction)

    # ------------------------------------------------------------------ #
    def step(self) -> MacroStepLog:
        """Advance one macro step (n 1D substeps + one 3D step)."""
        pipe_state = self.pipe.get_state()
        cfd_state = self.cfd.get_state()
        t0 = self.pipe.t

        plenum_head = self.plenum_head
        residuals: list[float] = []
        flow_residuals: list[float] = []
        q_mean_prev = self._last_q_mean
        exchange_s = 0.0
        wall_start = _time.perf_counter()

        q_profile_t = np.array([t0])
        q_profile_q = np.array([self.pipe.nozzle_discharge()])

        for _ in range(self.o.fixed_point_iters):
            # --- pass 1: 1D sub-cycling with current plenum backpressure ---
            self.pipe.set_state(pipe_state)
            ts = np.empty(self.n_substeps)
            qs = np.empty(self.n_substeps)
            for i in range(self.n_substeps):
                t = self.pipe.t
                qs[i] = self.pipe.step(
                    backpressure_head=plenum_head,
                    valve_fraction=self.valve_fraction(t),
                )
                ts[i] = self.pipe.t
            q_profile_t = np.concatenate([[t0], ts])
            q_profile_q = np.concatenate([[pipe_state[1][-1]], qs])

            # --- pass 2: 3D step with interpolated inlet mass flow ---
            t_start = _time.perf_counter()
            # Trapezoidal mean of the nozzle flow over the macro window,
            # with a startup ramp to avoid pressurising the 3D domain
            # impulsively from rest.
            q_mean = float(np.trapezoid(q_profile_q, q_profile_t)) / self.macro_dt
            ramp = min(1.0, (t0 + self.macro_dt) / max(self.o.inlet_ramp_time, 1e-9))
            u_inlet = ramp * q_mean / self.cfd.inlet_patch_area()
            self.cfd.set_state(cfd_state)
            self.cfd.set_inlet(u_inlet, self.pipe.o.fluid_temperature)
            exchange_s += _time.perf_counter() - t_start

            self.cfd.step(self.macro_dt)

            t_start = _time.perf_counter()
            new_head = self.cfd.inlet_pressure_head()
            exchange_s += _time.perf_counter() - t_start

            residuals.append(abs(new_head - plenum_head))
            if q_mean_prev is not None:
                flow_residuals.append(abs(q_mean - q_mean_prev))
            q_mean_prev = q_mean
            plenum_head = new_head
            if residuals[-1] < self.o.fixed_point_tol:
                break

        self.plenum_head = plenum_head
        self._last_q_mean = q_mean_prev
        wall_s = _time.perf_counter() - wall_start
        log = MacroStepLog(
            t=t0 + self.macro_dt,
            nozzle_flow=float(q_profile_q[-1]),
            plenum_head=plenum_head,
            valve_fraction=self.valve_fraction(t0 + self.macro_dt),
            fp_residuals=residuals,
            fp_flow_residuals=flow_residuals,
            exchange_latency_ms=exchange_s * 1.0e3,
            wall_time_ms=wall_s * 1.0e3,
        )
        self.logs.append(log)
        return log

    # ------------------------------------------------------------------ #
    def run(self, duration: float) -> list[MacroStepLog]:
        n = int(round(duration / self.macro_dt))
        return [self.step() for _ in range(n)]

    def history(self) -> dict[str, np.ndarray]:
        """Return the per-macro-step history as arrays."""
        return {
            "t": np.array([log.t for log in self.logs]),
            "nozzle_flow": np.array([log.nozzle_flow for log in self.logs]),
            "plenum_head": np.array([log.plenum_head for log in self.logs]),
            "valve_fraction": np.array([log.valve_fraction for log in self.logs]),
            "exchange_latency_ms": np.array(
                [log.exchange_latency_ms for log in self.logs]
            ),
            "wall_time_ms": np.array([log.wall_time_ms for log in self.logs]),
        }
