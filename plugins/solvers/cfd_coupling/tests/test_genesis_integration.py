"""Genesis-world integration tests for the cfd_coupling plugin solver.

Skipped automatically when genesis-world is not installed. The numerical
validation lives in ``test_coupling.py`` (genesis-free); these tests only
check the ``gs.Scene`` lifecycle: install -> build -> scene.step() ->
reset/state.
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

gs = pytest.importorskip("genesis")

from plugins.solvers.cfd_coupling import (  # noqa: E402
    CFDOptions,
    CouplingOptions,
    PipeOptions,
    install,
    install_cfd,
    install_pipe,
)
from plugins.solvers.cfd_coupling.solver import (  # noqa: E402
    CFDSolver,
    CFDSolverOptions,
    CoupledCFDSolver,
    CoupledSolverOptions,
    PipeSolver,
    PipeSolverOptions,
)

# Initialize Genesis once for the test module (CPU backend).
gs.init(backend=gs.cpu)

# Scene substeps: dt=0.002, substeps=2 -> substep_dt=0.001, which equals the
# 1D MOC step (n_reaches=50, wave_speed=200 -> dt=0.001) for an exact fit.
_DT = 0.002
_SUBSTEPS = 2


def _pipe_options() -> PipeOptions:
    return PipeOptions(
        length=10.0,
        diameter=0.05,
        n_reaches=50,
        wave_speed=200.0,
        friction=0.02,
        reservoir_head=30.0,
        nozzle_area=3.0e-5,
        discharge_coeff=0.8,
    )


def _cfd_options() -> CFDOptions:
    return CFDOptions(
        domain=(0.2, 0.1, 0.1),
        cells=(20, 10, 10),
        viscosity=1.0e-4,
        inlet_patch=((0.4, 0.6), (0.4, 0.6)),
        outlet_patch=((0.45, 0.55), (0.45, 0.55)),
        advect_temperature=False,
    )


def _make_scene(dt: float = _DT, substeps: int = _SUBSTEPS):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    return scene


class TestCoupledSolverIntegration:
    def test_install_registers_and_steps(self):
        scene = _make_scene()
        solver = install(
            scene,
            CoupledSolverOptions(
                pipe=_pipe_options(),
                cfd=_cfd_options(),
                coupling=CouplingOptions(
                    macro_dt=0.001, fixed_point_iters=2, inlet_ramp_time=0.02
                ),
            ),
        )
        scene.build()

        assert isinstance(solver, CoupledCFDSolver)
        assert solver in scene.sim._active_solvers
        assert scene.sim.cfd_coupling_solver is solver
        # Macro step snapped onto the 1D MOC step grid.
        assert solver.coupler.macro_dt == pytest.approx(0.001)

        for _ in range(50):  # 50 steps * 2 substeps = 0.1 s simulated
            scene.step()

        assert solver.pipe.t == pytest.approx(0.1, rel=1e-6)
        assert solver.cfd.n_steps == 100
        assert len(solver.coupler.logs) == 100
        # Bidirectional exchange actually happened: the plenum pressurised.
        assert solver.coupler.plenum_head > 0.1
        hist = solver.coupler.history()
        assert np.all(np.diff(hist["t"]) > 0)
        assert np.all(hist["exchange_latency_ms"] >= 0.0)

    def test_valve_event_during_scene_stepping(self):
        scene = _make_scene()
        solver = install(
            scene,
            CoupledSolverOptions(
                pipe=_pipe_options(),
                cfd=_cfd_options(),
                coupling=CouplingOptions(
                    macro_dt=0.001,
                    fixed_point_iters=1,
                    inlet_ramp_time=0.02,
                    valve_closure_start=0.05,
                    valve_closure_duration=0.04,
                ),
            ),
        )
        scene.build()

        flow_pre = None
        for _ in range(100):  # two substeps each: t -> 0.2 s
            scene.step()
            if solver.pipe.t <= 0.05 + 1e-9:
                flow_pre = solver.pipe.nozzle_discharge()
        flow_post = min(log.nozzle_flow for log in solver.coupler.logs if log.t > 0.06)
        assert flow_pre is not None
        assert flow_post < 0.5 * flow_pre

    def test_reset_restores_coupled_state(self):
        scene = _make_scene()
        solver = install(
            scene,
            CoupledSolverOptions(
                pipe=_pipe_options(),
                cfd=_cfd_options(),
                coupling=CouplingOptions(macro_dt=0.001, inlet_ramp_time=0.02),
            ),
        )
        scene.build()

        for _ in range(20):
            scene.step()
        state = scene.get_state()
        t_snapshot = solver.pipe.t
        steps_snapshot = solver.cfd.n_steps
        head_snapshot = solver.coupler.plenum_head
        q_snapshot = solver.pipe.Q.copy()

        for _ in range(20):
            scene.step()
        assert solver.pipe.t > t_snapshot

        scene.reset(state)
        assert solver.pipe.t == pytest.approx(t_snapshot)
        assert solver.cfd.n_steps == steps_snapshot
        assert solver.coupler.plenum_head == pytest.approx(head_snapshot)
        assert np.array_equal(solver.pipe.Q, q_snapshot)


class TestStandaloneSolvers:
    def test_install_pipe_steps_with_subcycling(self):
        scene = _make_scene()
        solver = install_pipe(scene, PipeSolverOptions(pipe=_pipe_options()))
        scene.build()

        assert isinstance(solver, PipeSolver)
        assert solver in scene.sim._active_solvers
        assert solver._n_sub == 1  # substep_dt == MOC dt

        q0 = solver.pipe.Q[-1]
        for _ in range(10):
            scene.step()
        assert solver.pipe.t == pytest.approx(0.02, rel=1e-6)
        # Open valve, steady reservoir: flow must not drift wildly.
        assert solver.pipe.Q[-1] == pytest.approx(q0, rel=0.2)

    def test_install_cfd_steps_and_reset(self):
        scene = _make_scene()
        solver = install_cfd(
            scene,
            CFDSolverOptions(cfd=_cfd_options(), inlet_velocity=0.5),
        )
        scene.build()

        assert isinstance(solver, CFDSolver)
        for _ in range(10):
            scene.step()
        assert solver.cfd.n_steps == 20
        assert solver.cfd.divergence_norm() < 1e-4

        state = scene.get_state()
        for _ in range(10):
            scene.step()
        scene.reset(state)
        assert solver.cfd.n_steps == 20
