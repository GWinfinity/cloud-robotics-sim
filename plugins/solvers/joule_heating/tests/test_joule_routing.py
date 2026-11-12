"""Behavioral tests for the per-env dynamic routing of the fused Jacobi solve.

Routing semantics under test (the production counterpart of
``outputs/steel_qd/test_route_jacobi.py``): converged + unexcited environments
freeze bit-exactly until a setter re-excites them; results match the unrouted
solver at the same fixed point; routing is off by default, in differentiable
scenes, and with the direct solver.

Run with the sibling module (it owns the one ``gs.init`` of the process):
``pytest plugins/solvers/joule_heating/tests -q``
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
import genesis as gs

from plugins.solvers.joule_heating import JouleHeatingOptions, install

if not gs._initialized:
    gs.init(backend=gs.cpu)


def _make_scene(n_envs: int = 4, routing: bool = True, **overrides):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    options = JouleHeatingOptions(
        resolution=(32, 8),
        dx=1.0,
        sigma=1.0,
        max_iter=2000,
        tol=1e-7,
        dynamic_routing=routing,
        **overrides,
    )
    solver = install(scene, options)
    scene.build(n_envs=n_envs)
    solver.set_voltage_boundary("x_min", 1.0)
    solver.set_voltage_boundary("x_max", 0.0)
    return scene, solver


def test_frozen_envs_carry_over_bit_exact():
    """Converged envs skip the solve; their field is bit-exactly carried over."""
    scene, solver = _make_scene()
    for _ in range(4):
        scene.step()
    assert np.all(solver.get_route() == 0)
    frozen = solver.get_voltage()
    for _ in range(3):
        scene.step()
    assert np.array_equal(solver.get_voltage(), frozen)
    assert np.all(solver.get_route() == 0)


def test_boundary_change_reexcites_frozen_envs():
    """A setter marks every env dirty: they solve again and re-freeze."""
    scene, solver = _make_scene()
    for _ in range(4):
        scene.step()
    assert np.all(solver.get_route() == 0)

    solver.set_voltage_boundary("x_min", 2.0)
    scene.step()
    assert np.all(solver.get_route() == 1)
    voltage = solver.get_voltage()
    assert np.allclose(voltage[:, 0, :], 2.0, atol=1e-6)

    for _ in range(4):
        scene.step()
    assert np.all(solver.get_route() == 0)
    profile = solver.get_voltage().mean(axis=2)  # (B, nx), averaged over y
    expected = np.broadcast_to(np.linspace(2.0, 0.0, 32)[None, :], profile.shape)
    np.testing.assert_allclose(profile, expected, atol=1e-2)


def test_routed_matches_unrouted_fixed_point():
    """Routing only skips work whose result the dense solve would not change."""
    scene_r, solver_r = _make_scene(routing=True)
    scene_d, solver_d = _make_scene(routing=False)
    for _ in range(6):
        scene_r.step()
        scene_d.step()
    np.testing.assert_allclose(
        solver_r.get_voltage(), solver_d.get_voltage(), atol=1e-4
    )


def test_routing_disabled_by_default():
    """Routing is opt-in; without it get_route raises."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(scene, JouleHeatingOptions(resolution=(8, 8)))
    scene.build()
    assert not solver._route_enabled
    with pytest.raises(RuntimeError):
        solver.get_route()


def test_routing_disabled_in_differentiable_scene():
    """The recorded fixed-iteration tape must match the forward exactly."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(
        scene,
        JouleHeatingOptions(resolution=(8, 8), dynamic_routing=True),
    )
    scene.build()
    assert not solver._route_enabled


def test_routing_disabled_with_direct_solver():
    """The cached-factorization direct path manages its own reuse."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(
        scene,
        JouleHeatingOptions(
            resolution=(8, 8),
            solver_type="direct",
            dynamic_routing=True,
        ),
    )
    scene.build()
    assert not solver._route_enabled
