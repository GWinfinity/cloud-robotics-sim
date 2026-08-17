"""Tests for the ThermalSolver plugin."""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import genesis as gs

from plugins.solvers.thermal import install, ThermalOptions
from plugins.solvers.thermal.core.thermal_solver import (
    ThermalSolver,
    ThermalSolverState,
)

# Initialize Genesis once for the test module.
gs.init(backend=gs.cpu)


def _make_scene(dt: float = 0.01, substeps: int = 1):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    return scene


class TestThermalOptions:
    """Validation tests for ThermalOptions."""

    def test_alpha_direct(self):
        opts = ThermalOptions(alpha=1e-4)
        assert opts.alpha == pytest.approx(1e-4)

    def test_alpha_from_k_rho_cp(self):
        opts = ThermalOptions(k=2.0, rho=4.0, cp=2.0)
        assert opts.alpha == pytest.approx(0.25)

    def test_missing_alpha_or_material_raises(self):
        with pytest.raises(ValueError):
            ThermalOptions(alpha=None, k=None, rho=1.0, cp=1.0)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            ThermalOptions(dim=1)

    def test_invalid_boundary_mode(self):
        with pytest.raises(ValueError):
            ThermalOptions(boundary_mode="periodic")


class TestThermalSolver:
    """Integration tests for ThermalSolver plugged into a gs.Scene."""

    def test_install_adds_solver_to_active_list(self):
        scene = _make_scene()
        thermal = install(scene, ThermalOptions(resolution=(16, 16), alpha=1e-4))
        scene.build()

        assert isinstance(thermal, ThermalSolver)
        assert thermal in scene.sim._active_solvers
        assert scene.sim.thermal_solver is thermal

    def test_get_set_temperature(self):
        scene = _make_scene()
        thermal = install(scene, ThermalOptions(resolution=(8, 8), alpha=1e-4))
        scene.build()

        temperature = thermal.get_temperature()
        assert temperature.shape == (8, 8)
        assert np.allclose(temperature, 0.0)

        thermal.set_temperature(np.ones((8, 8)))
        assert np.allclose(thermal.get_temperature(), 1.0)

    def test_dirichlet_boundary_remains_fixed(self):
        scene = _make_scene()
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                alpha=1e-4,
                boundary_mode="dirichlet",
                boundary_value=0.5,
                initial_temperature=1.0,
            ),
        )
        scene.build()

        for _ in range(10):
            scene.step()

        temperature = thermal.get_temperature()
        assert temperature[0, :].mean() == pytest.approx(0.5, abs=1e-6)
        assert temperature[-1, :].mean() == pytest.approx(0.5, abs=1e-6)
        assert temperature[:, 0].mean() == pytest.approx(0.5, abs=1e-6)
        assert temperature[:, -1].mean() == pytest.approx(0.5, abs=1e-6)

    def test_neumann_conserves_total_heat(self):
        """With zero-flux boundaries, explicit FTCS should conserve total heat."""
        scene = _make_scene()
        init = np.zeros((16, 16), dtype=float)
        init[6:10, 6:10] = 1.0

        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                alpha=1e-4,
                boundary_mode="dirichlet",
                boundary_value=0.0,
                initial_temperature=init,
            ),
        )
        scene.build()

        initial_sum = float(thermal.get_temperature().sum())
        for _ in range(50):
            scene.step()
        final_sum = float(thermal.get_temperature().sum())

        # Allow a small tolerance for floating-point drift.
        assert final_sum == pytest.approx(initial_sum, rel=1e-3)

    def test_cfl_raises_when_unstable(self):
        scene = _make_scene(dt=0.01)
        # alpha=1e-2, dx=0.01 -> alpha*dt/dx^2 = 1.0 > 0.25
        install(
            scene,
            ThermalOptions(resolution=(8, 8), dx=0.01, alpha=1e-2),
        )
        with pytest.raises(ValueError, match="unstable"):
            scene.build()

    def test_rigid_source_heats_grid(self):
        """A registered rigid entity should act as a heat source."""
        scene = _make_scene(dt=0.01, substeps=1)
        box = scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.08, 0.08, 0.01)),
            material=gs.materials.Rigid(),
        )
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=0.0,
            ),
        )
        thermal.add_source(
            entity=box,
            temperature=1.0,
            radius=0.03,
            rate=10.0,
            heat_capacity=1.0,
        )
        scene.build()

        for _ in range(50):
            scene.step()

        temperature = thermal.get_temperature()
        assert temperature.max() > 0.1
        # Cell nearest to the box position (0.08, 0.08) -> index (8, 8).
        assert temperature[8, 8] > 0.1

    def test_two_way_coupling_conserves_energy(self):
        """Heat exchange between body and grid conserves total thermal energy."""
        scene = _make_scene(dt=0.01, substeps=1)
        box = scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.08, 0.08, 0.01)),
            material=gs.materials.Rigid(),
        )
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                grid_rho=1.0,
                grid_cp=1.0,
                initial_temperature=0.0,
            ),
        )
        source = thermal.add_source(
            entity=box,
            temperature=1.0,
            radius=0.03,
            rate=1e6,  # near-instant equilibration per substep
            heat_capacity=0.01,
        )
        scene.build()

        cell_capacity = 1.0 * 1.0 * (0.01**2)
        initial_grid_energy = 0.0
        initial_energy = source.heat_capacity * source.temperature + initial_grid_energy

        for _ in range(20):
            scene.step()

        grid_energy = float(cell_capacity * thermal.get_temperature().sum())
        body_energy = source.heat_capacity * source.temperature
        final_energy = body_energy + grid_energy

        # Body should have cooled down by transferring heat to the grid.
        assert source.temperature < 1.0
        assert thermal.get_temperature().max() > 0.0

        # Total energy should be conserved.
        assert final_energy == pytest.approx(initial_energy, rel=1e-4)

    def test_matches_numpy_reference(self):
        """One explicit FTCS step should match a plain NumPy implementation."""
        scene = _make_scene(dt=0.01, substeps=1)
        init = np.zeros((16, 16), dtype=float)
        init[6:10, 6:10] = 1.0

        alpha = 1e-4
        dx = 0.01
        dt = 0.01
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=dx,
                alpha=alpha,
                boundary_mode="dirichlet",
                boundary_value=0.0,
                initial_temperature=init,
            ),
        )
        scene.build()
        scene.step()
        t_plugin = thermal.get_temperature()

        # NumPy reference with the same Dirichlet boundary.
        r = alpha * dt / (dx * dx)
        t_ref = init.copy()
        t_new = init.copy()
        t_new[1:-1, 1:-1] = t_ref[1:-1, 1:-1] + r * (
            t_ref[2:, 1:-1]
            + t_ref[:-2, 1:-1]
            + t_ref[1:-1, 2:]
            + t_ref[1:-1, :-2]
            - 4.0 * t_ref[1:-1, 1:-1]
        )
        t_new[0, :] = 0.0
        t_new[-1, :] = 0.0
        t_new[:, 0] = 0.0
        t_new[:, -1] = 0.0

        assert np.allclose(t_plugin, t_new, atol=1e-6)

    def test_add_source_after_build(self):
        """Sources can be registered after scene.build() up to max_sources."""
        scene = _make_scene(dt=0.01, substeps=1)
        box = scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.08, 0.08, 0.01)),
            material=gs.materials.Rigid(),
        )
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=0.0,
                max_sources=4,
            ),
        )
        scene.build()

        source = thermal.add_source(
            entity=box,
            temperature=1.0,
            radius=0.03,
            rate=10.0,
            heat_capacity=1.0,
        )

        for _ in range(50):
            scene.step()

        assert thermal.get_temperature().max() > 0.1
        assert source.temperature < 1.0

    def test_overlapping_sources_conserve_energy(self):
        """Two overlapping sources exchange heat with the grid without energy drift."""

        class _FixedSource:
            def __init__(self, pos):
                self._pos = np.asarray(pos, dtype=float)

            def get_pos(self):
                return self._pos

        scene = _make_scene(dt=0.01, substeps=1)
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                grid_rho=1.0,
                grid_cp=1.0,
                initial_temperature=0.0,
            ),
        )
        s1 = thermal.add_source(
            entity=_FixedSource((0.06, 0.06, 0.0)),
            temperature=1.0,
            radius=0.04,
            rate=1e6,
            heat_capacity=0.01,
        )
        s2 = thermal.add_source(
            entity=_FixedSource((0.10, 0.10, 0.0)),
            temperature=0.0,
            radius=0.04,
            rate=1e6,
            heat_capacity=0.01,
        )
        scene.build()

        cell_capacity = 1.0 * 1.0 * (0.01**2)
        initial_energy = (
            s1.heat_capacity * s1.temperature + s2.heat_capacity * s2.temperature
        )

        for _ in range(20):
            scene.step()

        grid_energy = float(cell_capacity * thermal.get_temperature().sum())
        final_energy = (
            s1.heat_capacity * s1.temperature
            + s2.heat_capacity * s2.temperature
            + grid_energy
        )

        assert thermal.get_temperature().max() > 0.0
        # Energy correction should make conservation exact to near machine precision.
        assert final_energy == pytest.approx(initial_energy, abs=1e-7)

    def test_add_source_after_step(self):
        """Sources can be registered after the simulation has already stepped."""
        scene = _make_scene(dt=0.01, substeps=1)
        box = scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.08, 0.08, 0.01)),
            material=gs.materials.Rigid(),
        )
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=0.0,
                max_sources=4,
            ),
        )
        scene.build()

        for _ in range(5):
            scene.step()

        source = thermal.add_source(
            entity=box,
            temperature=1.0,
            radius=0.03,
            rate=10.0,
            heat_capacity=1.0,
        )

        for _ in range(50):
            scene.step()

        assert thermal.get_temperature().max() > 0.1
        assert source.temperature < 1.0

    def test_n_envs_batching(self):
        """Different initial temperatures per environment evolve independently."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=0.0,
            ),
        )
        scene.build(n_envs=2)

        init = np.zeros((2, 16, 16), dtype=float)
        init[0, 6:10, 6:10] = 1.0
        thermal.set_temperature(init)

        for _ in range(20):
            scene.step()

        temperature = thermal.get_temperature()
        assert temperature.shape == (2, 16, 16)
        assert temperature[0].sum() > temperature[1].sum()

    def test_n_envs_with_sources(self):
        """Sources evolve independently per environment."""

        class _FixedSource:
            def __init__(self, pos):
                self._pos = np.asarray(pos, dtype=float)

            def get_pos(self):
                return self._pos

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(16, 16),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="neumann",
                grid_rho=1.0,
                grid_cp=1.0,
                initial_temperature=0.0,
            ),
        )
        thermal.add_source(
            entity=_FixedSource((0.08, 0.08, 0.0)),
            temperature=np.array([1.0, 0.5]),
            radius=0.04,
            rate=1e6,
            heat_capacity=0.01,
        )
        scene.build(n_envs=2)

        for _ in range(20):
            scene.step()

        # Env 0 started hotter, so it should remain hotter than env 1.
        temps = thermal._source_temperatures.to_numpy()[0, :]
        assert 0.0 < temps[1] < temps[0] < 1.0

    def test_gradient_flows_to_initial_temperature(self):
        """The thermal field is differentiable through the simulation rollout."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(8, 8),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="dirichlet",
                boundary_value=0.0,
                initial_temperature=0.0,
            ),
        )
        scene.build()

        init = np.zeros((8, 8), dtype=float)
        init[3:5, 3:5] = 1.0
        thermal.set_temperature(init)

        state0 = scene.get_state()
        scene.step()
        state1 = scene.get_state()

        thermal_state1 = [
            s for s in state1.solvers_state if isinstance(s, ThermalSolverState)
        ][0]
        loss = thermal_state1.T.sum()
        scene.backward(loss)

        thermal_state0 = [
            s for s in state0.solvers_state if isinstance(s, ThermalSolverState)
        ][0]
        grad = thermal_state0.T.grad
        assert grad is not None
        grad_np = grad.detach().cpu().numpy()
        assert not np.allclose(grad_np, 0.0)

    def test_gradient_flows_to_initial_temperature_with_sources(self):
        """The thermal field remains differentiable when sources are present."""

        class _FixedSource:
            def __init__(self, pos):
                self._pos = np.asarray(pos, dtype=float)

            def get_pos(self):
                return self._pos

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        thermal = install(
            scene,
            ThermalOptions(
                resolution=(8, 8),
                dx=0.01,
                alpha=1e-4,
                boundary_mode="dirichlet",
                boundary_value=0.0,
                initial_temperature=0.0,
            ),
        )
        thermal.add_source(
            entity=_FixedSource((0.04, 0.04, 0.0)),
            temperature=1.0,
            radius=0.03,
            rate=10.0,
            heat_capacity=0.1,
        )
        scene.build()

        init = np.zeros((8, 8), dtype=float)
        init[3:5, 3:5] = 1.0
        thermal.set_temperature(init)

        state0 = scene.get_state()
        scene.step()
        state1 = scene.get_state()

        thermal_state1 = [
            s for s in state1.solvers_state if isinstance(s, ThermalSolverState)
        ][0]
        loss = thermal_state1.T.sum()
        scene.backward(loss)

        thermal_state0 = [
            s for s in state0.solvers_state if isinstance(s, ThermalSolverState)
        ][0]
        grad = thermal_state0.T.grad
        assert grad is not None
        grad_np = grad.detach().cpu().numpy()
        assert not np.allclose(grad_np, 0.0)
