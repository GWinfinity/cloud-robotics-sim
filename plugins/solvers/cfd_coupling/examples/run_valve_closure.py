"""Coupled 1D(pipe) - 3D(plenum) valve-closure demonstration scenario.

Run:
    uv run python plugins/solvers/cfd_coupling/examples/run_valve_closure.py

Scenario
--------
A reservoir feeds a 1D pipe (MOC) ending in a nozzle/valve that discharges
into a 3D plenum (projection-method CFD) vented through a restricted outlet
orifice. At t = 0.1 s a control-logic event closes the valve linearly over
80 ms (millisecond-scale actuation on the 1D side).

Demonstrates the full bidirectional loop:
    1D -> 3D : nozzle flow Q(t) -> 3D inlet velocity + fluid temperature
    3D -> 1D : plenum static pressure (volume mean) -> 1D nozzle backpressure

Prints steady-state calibration, the Joukowsky reference for the closure,
per-macro-step exchange latency statistics, and saves the history to
``outputs/cfd_coupling/valve_closure_history.npz``.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from plugins.solvers.cfd_coupling import (
    CFD3D,
    CFDOptions,
    Coupler,
    CouplingOptions,
    Pipe1D,
    PipeOptions,
)


def main() -> None:
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
            fluid_temperature=320.0,  # -> passed to the 3D inlet
        )
    )
    cfd = CFD3D(
        CFDOptions(
            domain=(0.2, 0.1, 0.1),
            cells=(20, 10, 10),
            viscosity=1.0e-4,
            inlet_patch=((0.4, 0.6), (0.4, 0.6)),
            outlet_patch=((0.45, 0.55), (0.45, 0.55)),
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

    q0 = pipe.Q[-1]
    v0 = q0 / pipe.area
    print(f"Steady nozzle flow Q0 = {q0:.4e} m^3/s (V0 = {v0:.3f} m/s)")
    print(f"Joukowsky reference dH = a*dV/g = "
          f"{pipe.o.wave_speed * v0 / 9.81:.2f} m (instant full closure)")

    wall_start = time.perf_counter()
    coupler.run(0.1)  # open-valve steady state
    pre = coupler.history()
    print(f"\nOpen-valve steady: plenum head = {pre['plenum_head'][-1]:+.3f} m")

    coupler.run(0.4)  # closure + decay
    wall_s = time.perf_counter() - wall_start

    hist = coupler.history()
    post = hist["t"] > 0.1
    print(f"After closure: min nozzle flow = {hist['nozzle_flow'][post].min():.3e} m^3/s")
    print(f"               min plenum head  = {hist['plenum_head'][post].min():+.4f} m")
    print(f"3D incompressible: max |div| = {cfd.divergence_norm():.2e}")

    lat = hist["exchange_latency_ms"]
    print(f"\nExchange latency (boundary transfer only):")
    print(f"  median = {np.median(lat):.3f} ms, p95 = {np.percentile(lat, 95):.3f} ms")
    print(f"Macro step: {wall_s / len(hist['t']) * 1e3:.1f} ms wall "
          f"(incl. both solvers, fixed-point passes)")
    print(f"Minimum exchange period (macro_dt): {coupler.macro_dt * 1e3:.1f} ms")

    out = ROOT / "outputs" / "cfd_coupling"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "valve_closure_history.npz"
    np.savez(path, **hist)
    print(f"\nHistory saved to {path}")


if __name__ == "__main__":
    main()
