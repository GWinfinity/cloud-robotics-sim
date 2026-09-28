"""Genesis integration tests for the nozzle_coupling plugin.

One ``gs.init`` per process (quadrants/kernel JIT requires it); all genesis
tests live in this module. Skips automatically without genesis-world.
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

gs = pytest.importorskip("genesis")

gs.init(backend=gs.cpu)

import genesis as gs_mod  # noqa: F811  (re-export after init)

from plugins.solvers.nozzle_coupling import (
    JetOptions,
    NozzleOptions,
    install,
    install_jet,
    install_nozzle,
)
from plugins.solvers.nozzle_coupling.core import NozzleCouplingOptions
from plugins.solvers.nozzle_coupling.solver import (
    CoupledNozzleSolverOptions,
    JetSolverOptions,
    NozzleSolverOptions,
)

P_AMB = 101325.0


def _scene(substep_dt: float = 2.0e-4):
    return gs_mod.Scene(
        sim_options=gs_mod.options.SimOptions(
            dt=substep_dt, substeps=1, gravity=(0.0, 0.0, 0.0)
        )
    )


class TestCoupledNozzleSolverIntegration:
    """Coupled solver install/build/step/probe/reset lifecycle."""

    def test_install_build_step_and_probes(self):
        scene = _scene()
        solver = install(
            scene,
            CoupledNozzleSolverOptions(
                nozzle=NozzleOptions(
                    throat_area=1.0e-3,
                    exit_area=3.0e-3,
                    chamber_pressure=1.3 * P_AMB,
                    chamber_temperature=600.0,
                ),
                jet=JetOptions(
                    domain=(0.24, 0.12, 0.12),
                    cells=(12, 6, 6),
                ),
                coupling=NozzleCouplingOptions(
                    macro_dt=2.0e-4, n_substeps=2, inlet_ramp_time=0.001
                ),
            ),
        )
        assert scene.sim.nozzle_coupling_solver is solver
        scene.build()
        for _ in range(20):
            scene.step()
        h = solver.coupler.history()
        assert h["mdot"][-1] > 0.0
        assert solver.coupler.backpressure > 0.9 * P_AMB
        assert solver.nozzle.t == pytest.approx(20 * 2.0e-4, rel=1e-9)

    def test_reset_restores_coupled_state(self):
        scene = _scene()
        solver = install(
            scene,
            CoupledNozzleSolverOptions(
                nozzle=NozzleOptions(
                    throat_area=1.0e-3,
                    exit_area=3.0e-3,
                    chamber_pressure=1.3 * P_AMB,
                    chamber_temperature=600.0,
                ),
                jet=JetOptions(domain=(0.24, 0.12, 0.12), cells=(12, 6, 6)),
                coupling=NozzleCouplingOptions(
                    macro_dt=2.0e-4, n_substeps=2, inlet_ramp_time=0.001
                ),
            ),
        )
        scene.build()
        for _ in range(20):
            scene.step()
        state = scene.get_state()
        mdot_before = solver.coupler.history()["mdot"][-1]
        p_before = solver.coupler.backpressure
        t_before = solver.nozzle.t
        for _ in range(20):
            scene.step()
        scene.reset(state)
        assert solver.nozzle.t == pytest.approx(t_before, rel=1e-12)
        assert solver.coupler.backpressure == pytest.approx(p_before, rel=1e-12)
        solver.coupler.step()
        assert solver.coupler.history()["mdot"][-1] == pytest.approx(
            mdot_before, rel=1e-9
        )


class TestStandaloneSolvers:
    """Standalone 1D/3D solver lifecycle."""

    def test_install_nozzle(self):
        scene = _scene(substep_dt=2.0e-4)
        solver = install_nozzle(
            scene,
            NozzleSolverOptions(
                nozzle=NozzleOptions(
                    throat_area=1.0e-4,
                    exit_area=3.0e-4,
                    chamber_pressure=5.0e6,
                    chamber_temperature=3000.0,
                ),
                backpressure_pa=P_AMB,
            ),
        )
        scene.build()
        for _ in range(5):
            scene.step()
        assert solver.nozzle.n_steps == 5 * solver._n_sub
        assert solver.nozzle.evaluate(P_AMB, solver.nozzle.t).mdot > 0.0

    def test_install_jet(self):
        scene = _scene(substep_dt=1.0e-4)
        solver = install_jet(
            scene,
            JetSolverOptions(
                jet=JetOptions(domain=(0.24, 0.12, 0.12), cells=(12, 6, 6)),
                inlet_mass_flux=0.05,
            ),
        )
        scene.build()
        solver.jet.set_inlet(0.05, stagnation_temp=600.0, r_eff=332.6)
        for _ in range(20):
            scene.step()
        assert solver.jet.inlet_mass_flow() == pytest.approx(0.05, rel=1e-6)
        state = scene.get_state()
        for _ in range(10):
            scene.step()
        scene.reset(state)
        assert solver.jet.t == pytest.approx(20 * 1.0e-4, rel=1e-12)
