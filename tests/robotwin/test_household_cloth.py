"""Tests for household cloth grids and Genesis PBD spawn wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.cloth import (
    CLOTH_SPECS,
    cloth_grid,
    particle_positions,
    spawn_pbd_cloth,
)


def test_cloth_grids_are_metric_two_dimensional_surfaces() -> None:
    """Rag and towel are open triangulated sheets with their specified area."""
    for name, spec in CLOTH_SPECS.items():
        mesh = cloth_grid(spec)
        assert len(mesh.vertices) == spec.columns * spec.rows, name
        assert len(mesh.faces) == (spec.columns - 1) * (spec.rows - 1) * 2, name
        assert mesh.area == pytest.approx(spec.width * spec.depth), name
        assert not mesh.is_watertight, name
        assert np.all(mesh.vertices[:, 2] == 0.0), name


def test_spawn_uses_native_pbd_cloth_and_zup_mesh() -> None:
    """The spawn helper passes cloth material to Scene.add_entity."""
    fake_gs = SimpleNamespace(
        materials=SimpleNamespace(PBD=SimpleNamespace(Cloth=mock.Mock())),
        morphs=SimpleNamespace(Mesh=mock.Mock()),
        surfaces=SimpleNamespace(Default=mock.Mock()),
    )
    scene = SimpleNamespace(add_entity=mock.Mock())
    with mock.patch.dict(sys.modules, {"genesis": fake_gs}):
        spawn_pbd_cloth(scene, CLOTH_SPECS["rag"], "rag.obj", (0.0, 0.0, 0.2))
    material = fake_gs.materials.PBD.Cloth.call_args.kwargs
    assert material["rho"] == CLOTH_SPECS["rag"].density
    assert material["stretch_compliance"] > 0.0
    morph = fake_gs.morphs.Mesh.call_args.kwargs
    assert morph["file_meshes_are_zup"] is True
    assert morph["pos"] == (0.0, 0.0, 0.2)
    assert fake_gs.surfaces.Default.call_args.kwargs["double_sided"] is True
    scene.add_entity.assert_called_once()


def test_particle_positions_are_host_numpy_rows() -> None:
    """Single-env particle output is normalized to an (N, 3) host array."""
    entity = SimpleNamespace(get_particles_pos=lambda: np.arange(12).reshape(1, 4, 3))
    assert particle_positions(entity).shape == (4, 3)
