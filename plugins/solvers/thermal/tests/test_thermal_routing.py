"""Behavioral tests for the per-env dynamic routing of the transient update.

Routing semantics under test: stationary + source-free environments freeze
bit-exactly (a uniform field at the Dirichlet value is the exact fixed point,
so it freezes after the initial dirty substep); ``set_temperature`` re-excites
and re-freezes; any active source keeps every environment routed (source
coupling can touch any env's field); routing is off by default and in
differentiable scenes.

Note: natural freeze-up of a *perturbed* field is not exercised with a large
hotspot because FTCS diffusion on a small grid has a slow low-frequency tail
(hundreds of steps to stationarity); the fixed-point and setter transitions
cover the routing mechanism itself, and ``test_routed_matches_unrouted_decay``
covers numerical equivalence with the dense update over a live transient.

Run with the sibling module (it owns the one ``gs.init`` of the process):
``pytest plugins/solvers/thermal/tests -q``
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

from plugins.solvers.thermal import ThermalOptions, install

if not gs._initialized:
    gs.init(backend=gs.cpu)

# FTCS CFL: alpha * dt / dx^2 = 2e-3 * 0.01 / 1e-4 = 0.2 < 0.25.
_ALPHA = 2e-3
_DX = 0.01
_DT = 0.01
_RES = 16


def _hotspot(peak: float = 1.0) -> np.ndarray:
    field = np.zeros((_RES, _RES), dtype=float)
    field[_RES // 2, _RES // 2] = peak
    return field


def _make_scene(n_envs: int = 4, routing: bool = True, initial=None, **overrides):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    options = ThermalOptions(
        resolution=(_RES, _RES),
        dx=_DX,
        alpha=_ALPHA,
        boundary_mode="dirichlet",
        boundary_value=0.0,
        initial_temperature=np.zeros((_RES, _RES)) if initial is None else initial,
        dynamic_routing=routing,
        **overrides,
    )
    solver = install(scene, options)
    scene.build(n_envs=n_envs)
    return scene, solver


def _set_all_temperatures(solver, field: np.ndarray) -> None:
    solver.set_temperature(np.broadcast_to(field, (solver._B, *field.shape)).copy())


def _freeze(scene, solver) -> np.ndarray:
    """Step past the initial dirty substep until every env is frozen."""
    for _ in range(4):
        scene.step()
        if np.all(solver.get_route() == 0):
            return solver.get_temperature()
    raise AssertionError("environments did not freeze")


def test_frozen_envs_carry_over_bit_exact():
    """Uniform field at the Dirichlet value: update is an exact no-op, and the
    frozen envs' fields carry over bit-exactly.
    """
    scene, solver = _make_scene()
    frozen = _freeze(scene, solver)
    for _ in range(3):
        scene.step()
    assert np.array_equal(solver.get_temperature(), frozen)
    assert np.all(solver.get_route() == 0)


def test_routed_matches_unrouted_decay():
    """Routing only skips updates whose result the dense step would not change."""
    initial = _hotspot()
    scene_r, solver_r = _make_scene(initial=initial)
    scene_d, solver_d = _make_scene(initial=initial, routing=False)
    for _ in range(20):
        scene_r.step()
        scene_d.step()
    np.testing.assert_allclose(
        solver_r.get_temperature(), solver_d.get_temperature(), atol=1e-5
    )


def test_set_temperature_reexcites_and_refreezes():
    """A host write re-excites the frozen envs; back at the fixed point they
    freeze again with the field carried over bit-exactly.
    """
    scene, solver = _make_scene()
    _freeze(scene, solver)
    _set_all_temperatures(solver, _hotspot())
    scene.step()
    assert np.all(solver.get_route() == 1)
    assert solver.get_temperature().max() == pytest.approx(1.0, abs=1e-6)
    _set_all_temperatures(solver, np.zeros((_RES, _RES)))
    refrozen = _freeze(scene, solver)
    assert np.array_equal(refrozen, np.zeros((solver._B, _RES, _RES)))


def test_active_source_keeps_every_env_routed():
    """Source coupling may touch any env's field: the predicate stays on."""
    scene, solver, box = _make_source_scene()
    solver.add_source(entity=box, temperature=1.0, radius=0.05, rate=1.0 / _DT)
    for _ in range(10):
        scene.step()
        assert np.all(solver.get_route() == 1)


def test_removed_source_lets_envs_freeze():
    """With the source gone, nothing keeps the envs routed: at the fixed point
    they freeze again (two substeps: one to consume the dirty flag).
    """
    scene, solver, box = _make_source_scene()
    source = solver.add_source(entity=box, temperature=1.0, radius=0.05, rate=1.0 / _DT)
    for _ in range(5):
        scene.step()
    solver.remove_source(source)
    # With the source gone, nothing keeps the envs routed: at the fixed point
    # they freeze again (two substeps: one to consume the dirty flag).
    _set_all_temperatures(solver, np.zeros((_RES, _RES)))
    _freeze(scene, solver)
    assert np.all(solver.get_route() == 0)


def _make_source_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.08, 0.08, 0.02)),
        material=gs.materials.Rigid(),
    )
    solver = install(
        scene,
        ThermalOptions(
            resolution=(_RES, _RES),
            dx=_DX,
            alpha=_ALPHA,
            boundary_mode="dirichlet",
            boundary_value=0.0,
            initial_temperature=np.zeros((_RES, _RES)),
            dynamic_routing=True,
            max_sources=2,
        ),
    )
    scene.build(n_envs=4)
    return scene, solver, box


def test_routing_disabled_by_default():
    """Routing is opt-in; without it get_route raises."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(scene, ThermalOptions(resolution=(8, 8), alpha=_ALPHA))
    scene.build()
    assert not solver._route_enabled
    with pytest.raises(RuntimeError):
        solver.get_route()


def test_routing_disabled_in_differentiable_scene():
    """The recorded tape must match the forward exactly."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(
        scene,
        ThermalOptions(
            resolution=(8, 8),
            alpha=_ALPHA,
            dynamic_routing=True,
        ),
    )
    scene.build()
    assert not solver._route_enabled
