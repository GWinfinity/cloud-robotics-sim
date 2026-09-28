"""Headless demo: nozzle exhaust 1D <-> 3D bidirectional coupling.

Scenario (all parameters are PLACEHOLDERS in generic compressible-flow
ranges — substitute data from your own lawful, professionally supervised
sources before drawing any conclusions):

1. Startup transient: the chamber pressure/temperature curves ramp up over
   10 ms (curve-driven, replayable).
2. Steady choked exhaust into the open plenum.
3. Throttling event at t = 40 ms: the 3D outlet is restricted
   (``jet.set_outlet_patch``) -> the plenum pressurizes -> the backpressure
   crosses the interface -> the 1D nozzle responds (shock regime: exit
   pressure rises / shock pushed upstream; deeper restriction would drive it
   subsonic and throttle the mass flux).

Prints per-phase calibration and exchange-latency statistics, and saves the
per-macro-step history to ``outputs/nozzle_coupling/nozzle_exhaust_history.npz``.
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from plugins.solvers.nozzle_coupling.core.coupler import (
    NozzleCoupler,
    NozzleCouplingOptions,
)
from plugins.solvers.nozzle_coupling.core.jet3d import Jet3D, JetOptions
from plugins.solvers.nozzle_coupling.core.nozzle1d import Nozzle1D, NozzleOptions

P_AMB = 101325.0


def main() -> None:
    """Run the startup -> steady -> throttling-event demo."""
    # --- placeholder chamber curves (ramp over 10 ms) -------------------
    nozzle = Nozzle1D(
        NozzleOptions(
            throat_area=1.0e-3,
            exit_area=3.0e-3,
            gamma=1.25,
            molecular_weight=0.025,
            curve_t=np.array([0.0, 0.01, 0.05]),
            curve_p=np.array([P_AMB, 130000.0, 130000.0]),
            curve_T=np.array([300.0, 600.0, 600.0]),
            curve_alpha=np.array([0.0, 0.0, 0.0]),
            particle_lag_factor=0.9,
        )
    )
    jet = Jet3D(JetOptions(domain=(0.24, 0.12, 0.12), cells=(12, 6, 6)))
    coupler = NozzleCoupler(
        nozzle,
        jet,
        NozzleCouplingOptions(
            macro_dt=2.0e-5,
            n_substeps=2,
            fixed_point_iters=2,
            inlet_ramp_time=0.004,
        ),
    )

    def run_until(t_end: float) -> None:
        while nozzle.t < t_end:
            coupler.step()

    # Phase 1+2: startup and open-outlet steady state.
    run_until(0.04)
    h = coupler.history()
    mdot_steady = h["mdot"][-100:].mean()
    p_back_open = coupler.backpressure
    p_exit_open = h["exit_pressure_pa"][-1]
    print("=== phase 1/2: startup + open-outlet steady state ===")
    print(f"  t            = {nozzle.t:.4f} s")
    print(f"  mdot(steady) = {mdot_steady:.5f} kg/s   regime = {h['regime'][-1]}")
    print(f"  p_back       = {p_back_open:.1f} Pa (gauge {p_back_open - P_AMB:+.1f})")
    print(f"  p_exit (1D)  = {h['exit_pressure_pa'][-1]:.1f} Pa")
    print(
        f"  T_exit (1D)  = {nozzle.evaluate(coupler.backpressure, nozzle.t).temperature:.1f} K"
    )
    print(f"  jet T max    = {float(jet.T.max()):.1f} K")

    # Phase 3: throttling event.
    jet.set_outlet_patch(((0.4, 0.6), (0.4, 0.6)))
    run_until(0.08)
    h = coupler.history()
    print("=== phase 3: throttling event (outlet restricted at t=40 ms) ===")
    print(f"  t            = {nozzle.t:.4f} s")
    print(
        f"  mdot         = {h['mdot'][-100:].mean():.5f} kg/s   regime = {h['regime'][-1]}"
    )
    print(
        f"  p_back       = {coupler.backpressure:.1f} Pa (gauge {coupler.backpressure - P_AMB:+.1f})"
    )
    print(
        f"  p_exit (1D)  = {h['exit_pressure_pa'][-1]:.1f} Pa (shock pushed upstream)"
    )
    print(
        f"  signal: Δp_back = {coupler.backpressure - p_back_open:+.1f} Pa, "
        f"Δp_exit = {h['exit_pressure_pa'][-1] - p_exit_open:+.1f} Pa"
    )

    lat = h["exchange_latency_ms"]
    wall = h["wall_time_ms"]
    print("=== exchange statistics ===")
    print(
        f"  exchange latency ms/step: median {np.median(lat):.3f} "
        f"p95 {np.percentile(lat, 95):.3f}"
    )
    print(
        f"  macro step wall time  ms: median {np.median(wall):.2f} "
        f"p95 {np.percentile(wall, 95):.2f}"
    )

    out = Path("outputs/nozzle_coupling")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "nozzle_exhaust_history.npz", **h)
    print(f"  history saved to {out / 'nozzle_exhaust_history.npz'}")


if __name__ == "__main__":
    main()
