"""Tests for the JouleHeatingSolver plugin."""

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

from plugins.solvers.joule_heating import install, JouleHeatingOptions
from plugins.solvers.joule_heating.core.joule_heating_solver import JouleHeatingSolver

# Initialize Genesis once for the test module.
gs.init(backend=gs.cpu)


def _make_scene(dt: float = 0.01, substeps: int = 1):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    return scene


class TestJouleHeatingOptions:
    """Validation tests for JouleHeatingOptions."""

    def test_defaults(self):
        opts = JouleHeatingOptions()
        assert opts.dim == 2
        assert opts.dx == pytest.approx(0.01)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            JouleHeatingOptions(dim=1)

    def test_resolution_mismatch(self):
        with pytest.raises(ValueError):
            JouleHeatingOptions(dim=3, resolution=(16, 16))

    def test_negative_dx(self):
        with pytest.raises(ValueError):
            JouleHeatingOptions(dx=0.0)

    def test_negative_rho(self):
        with pytest.raises(ValueError):
            JouleHeatingOptions(rho=0.0)


class TestJouleHeatingSolver:
    """Integration tests for JouleHeatingSolver plugged into a gs.Scene."""

    def test_install_adds_solver_to_active_list(self):
        scene = _make_scene()
        solver = install(scene, JouleHeatingOptions(resolution=(16, 16)))
        scene.build()

        assert isinstance(solver, JouleHeatingSolver)
        assert solver in scene.sim._active_solvers
        assert scene.sim.joule_heating_solver is solver

    def test_get_set_voltage(self):
        scene = _make_scene()
        solver = install(scene, JouleHeatingOptions(resolution=(8, 8)))
        scene.build()

        voltage = solver.get_voltage()
        assert voltage.shape == (8, 8)
        assert np.allclose(voltage, 0.0)

        solver.set_voltage(np.ones((8, 8)))
        assert np.allclose(solver.get_voltage(), 1.0)

    def test_set_conductivity(self):
        scene = _make_scene()
        solver = install(scene, JouleHeatingOptions(resolution=(8, 8)))
        scene.build()

        sigma = np.full((8, 8), 1e5, dtype=float)
        solver.set_conductivity(sigma)
        # Conductivity is not directly exposed, but setting should not raise.

    def test_linear_potential_2d(self):
        """With V=1 at x=0 and V=0 at x=L, the interior should become linear."""
        scene = _make_scene()
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(32, 8),
                dx=1.0,
                sigma=1.0,
                max_iter=2000,
                tol=1e-7,
            ),
        )
        scene.build()

        solver.set_voltage_boundary("x_min", 1.0)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        voltage = solver.get_voltage()
        nx = voltage.shape[0]
        expected = np.linspace(1.0, 0.0, nx)
        # Average along y and compare with the analytic linear profile.
        profile = voltage.mean(axis=1)
        assert profile == pytest.approx(expected, abs=1e-3)

    def test_zero_source_when_no_boundary_differential(self):
        """If all boundaries have the same voltage, the field is uniform and Q=0."""
        scene = _make_scene()
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 16),
                dx=1.0,
                sigma=1.0,
                max_iter=500,
                tol=1e-7,
            ),
        )
        scene.build()

        solver.set_voltage_boundary("x_min", 1.0)
        solver.set_voltage_boundary("x_max", 1.0)
        solver.set_voltage_boundary("y_min", 1.0)
        solver.set_voltage_boundary("y_max", 1.0)

        scene.step()

        assert np.allclose(solver.get_voltage(), 1.0, atol=1e-6)
        assert np.allclose(solver.get_heat_source(), 0.0, atol=1e-6)

    def test_internal_thermal_heats_up(self):
        """With a voltage difference, the internal temperature field should rise."""
        scene = _make_scene(dt=0.001, substeps=1)
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 4),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                initial_temperature=300.0,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=False,
            ),
        )
        scene.build()

        solver.set_voltage_boundary("x_min", 10.0)
        solver.set_voltage_boundary("x_max", 0.0)

        initial = float(solver.get_temperature().mean())
        for _ in range(20):
            scene.step()
        final = float(solver.get_temperature().mean())

        assert final > initial

    def test_couple_to_thermal_increases_temperature(self):
        """When coupled, the thermal solver's temperature should rise."""
        from plugins.solvers.thermal import install as install_thermal
        from plugins.solvers.thermal import ThermalOptions

        scene = _make_scene(dt=0.001, substeps=1)
        thermal = install_thermal(
            scene,
            ThermalOptions(
                resolution=(16, 4),
                dx=1.0,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=300.0,
            ),
        )
        joule = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 4),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=1.0,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=True,
            ),
        )
        scene.build()

        joule.set_voltage_boundary("x_min", 10.0)
        joule.set_voltage_boundary("x_max", 0.0)

        initial = float(thermal.get_temperature().mean())
        for _ in range(20):
            scene.step()
        final = float(thermal.get_temperature().mean())

        assert final > initial

    def test_n_envs_batch(self):
        """Solver should work with batched environments."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, substeps=1),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 4),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                max_iter=500,
                tol=1e-7,
            ),
        )
        scene.build(n_envs=2)

        solver.set_voltage_boundary("x_min", 5.0)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        voltage = solver.get_voltage()
        assert voltage.shape == (2, 16, 4)
        assert np.allclose(voltage[0], voltage[1])

    def test_total_power_matches_i2r(self):
        """Total Joule power in a uniform conductor should match I²R."""
        scene = _make_scene()
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(32, 8),
                dx=1.0,
                sigma=1.0,
                max_iter=2000,
                tol=1e-7,
            ),
        )
        scene.build()

        v_high = 10.0
        solver.set_voltage_boundary("x_min", v_high)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        q_arr = solver.get_heat_source()
        dx = solver._dx
        # 2D: integrate over area; the analytic estimate uses unit depth.
        total_power = float(q_arr.sum()) * dx * dx
        nx, ny = q_arr.shape
        # Discrete 2D estimate: only interior cells carry a non-zero source
        # because Q is set to zero on the boundary in this implementation.
        expected_power = (
            solver._sigma_scalar
            * (v_high / ((nx - 1) * dx)) ** 2
            * (nx - 2)
            * (ny - 2)
            * dx
            * dx
        )
        assert total_power == pytest.approx(expected_power, rel=1e-2)

    def test_gradient_flows_to_conductivity(self):
        """The conductivity field is differentiable through the electric solve."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 8),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=False,
            ),
        )
        scene.build()

        # Non-uniform conductivity so a gradient has somewhere to go.
        sigma = np.ones((16, 8), dtype=float)
        sigma[4:12, 2:6] = 2.0
        solver.set_conductivity(sigma)

        solver.set_voltage_boundary("x_min", 10.0)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        # Seed the loss gradient into _Q at substep 0 and run the solver's
        # backward pass manually (JouleHeatingSolver does not yet expose a
        # SolverState wrapper for scene.backward).
        q_grad = solver._Q.grad.to_numpy()
        q_grad.fill(0.0)
        q_grad[0] = 1.0
        solver._Q.grad.from_numpy(q_grad)
        solver.substep_pre_coupling_grad(0)

        sigma_grad = solver._sigma.grad
        assert sigma_grad is not None
        grad_np = sigma_grad.to_numpy()
        assert not np.allclose(grad_np, 0.0)
        # The region with higher conductivity should carry a different gradient.
        assert not np.allclose(grad_np[0, 4:12, 2:6], grad_np[0, 0, 0])

    def test_gradient_flows_through_thermal_coupling(self):
        """Gradients propagate from thermal _T back to Joule _sigma via coupling."""
        from plugins.solvers.thermal import install as install_thermal
        from plugins.solvers.thermal import ThermalOptions

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        thermal = install_thermal(
            scene,
            ThermalOptions(
                resolution=(16, 4),
                dx=1.0,
                alpha=1e-4,
                boundary_mode="neumann",
                initial_temperature=300.0,
            ),
        )
        joule = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 4),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=1.0,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=True,
            ),
        )
        scene.build()

        # Non-uniform conductivity so gradients have somewhere to go.
        sigma = np.ones((16, 4), dtype=float)
        sigma[4:12, 1:3] = 2.0
        joule.set_conductivity(sigma)

        joule.set_voltage_boundary("x_min", 10.0)
        joule.set_voltage_boundary("x_max", 0.0)

        scene.step()

        # Seed the loss gradient into the thermal field at substep 1 and pull it
        # back through the Joule -> Thermal coupling into Joule _Q, then through
        # the electric solve into _sigma.
        thermal_t = thermal._T
        assert thermal_t is not None and thermal_t.grad is not None
        t_grad = thermal_t.grad.to_numpy()
        t_grad.fill(0.0)
        t_grad[1] = 1.0
        thermal_t.grad.from_numpy(t_grad)

        joule.reset_grad()
        joule.substep_pre_coupling_grad(0)

        sigma_grad = joule._sigma.grad
        assert sigma_grad is not None
        grad_np = sigma_grad.to_numpy()
        assert not np.allclose(grad_np, 0.0)
        assert not np.allclose(grad_np[0, 4:12, 1:3], grad_np[0, 0, 0])

    def test_gradient_flows_to_boundary_voltage(self):
        """Dirichlet boundary voltage is differentiable through the Jacobi solve."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 8),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=False,
            ),
        )
        scene.build()

        solver.set_voltage_boundary("x_min", 10.0)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        # Seed the loss gradient into _Q at substep 0 and run the solver's
        # backward pass. The gradient should reach the boundary-voltage field.
        q_grad = solver._Q.grad.to_numpy()
        q_grad.fill(0.0)
        q_grad[0] = 1.0
        solver._Q.grad.from_numpy(q_grad)
        solver.substep_pre_coupling_grad(0)

        bv_field = solver._boundary_voltage_field
        assert bv_field is not None and bv_field.grad is not None
        bv_grad = bv_field.grad.to_numpy()
        assert not np.allclose(bv_grad, 0.0)
        # x_min (high voltage) and x_max (ground) should receive different gradients.
        assert not np.allclose(bv_grad[0, 0, :], bv_grad[0, -1, :])

    def test_gradient_flows_to_scalar_material_params(self):
        """Scalar rho/cp/k are differentiable through the internal thermal step."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(16, 8),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                initial_temperature=lambda x, y: 300.0 + x,
                max_iter=500,
                tol=1e-7,
                couple_to_thermal=False,
            ),
        )
        scene.build()

        solver.set_voltage_boundary("x_min", 10.0)
        solver.set_voltage_boundary("x_max", 0.0)

        scene.step()

        # Seed a non-uniform loss gradient into the final temperature frame and
        # run the solver's backward pass. The gradient should reach rho/cp/k.
        t_grad = solver._T.grad.to_numpy()
        t_grad.fill(0.0)
        t_grad[1] = np.indices((16, 8)).sum(axis=0).astype(float)
        solver._T.grad.from_numpy(t_grad)
        solver.substep_pre_coupling_grad(0)

        for name in ("_rho_field", "_cp_field", "_k_field"):
            field = getattr(solver, name)
            assert field is not None and field.grad is not None
            grad_np = field.grad.to_numpy()
            assert not np.allclose(grad_np, 0.0), f"{name} gradient is zero"

    def test_scalar_param_setters_update_fields(self):
        """set_rho/set_cp/set_k update both the field and cached Python value."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            JouleHeatingOptions(
                resolution=(8, 4),
                dx=1.0,
                sigma=1.0,
                rho=1.0,
                cp=1.0,
                k=0.01,
                max_iter=100,
                couple_to_thermal=False,
            ),
        )
        scene.build()

        solver.set_rho(2.5)
        solver.set_cp(800.0)
        solver.set_k(0.5)

        assert solver._rho == 2.5
        assert solver._cp == 800.0
        assert solver._k == 0.5
        np.testing.assert_allclose(solver.get_rho(), 2.5)
        np.testing.assert_allclose(solver.get_cp(), 800.0)
        np.testing.assert_allclose(solver.get_k(), 0.5)
