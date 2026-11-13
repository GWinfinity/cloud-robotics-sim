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
    cloth_mesh,
    particle_positions,
    spawn_pbd_cloth,
)


def test_cloth_grids_are_metric_two_dimensional_surfaces() -> None:
    """Rectangular cloths are open triangulated sheets with their specified area."""
    for name, spec in CLOTH_SPECS.items():
        if spec.shape != "grid":
            continue
        mesh = cloth_grid(spec)
        assert len(mesh.vertices) == spec.columns * spec.rows, name
        assert len(mesh.faces) == (spec.columns - 1) * (spec.rows - 1) * 2, name
        assert mesh.area == pytest.approx(spec.width * spec.depth), name
        assert not mesh.is_watertight, name
        assert np.all(mesh.vertices[:, 2] == 0.0), name


def test_garment_and_bags_have_connected_open_surfaces() -> None:
    """Shaped cloths retain their silhouette, bag volume and open boundaries."""
    garment_spec = CLOTH_SPECS["garment"]
    garment = cloth_mesh(garment_spec)
    assert garment.extents[:2] == pytest.approx(
        [garment_spec.width, garment_spec.depth]
    )
    assert garment.area < garment_spec.width * garment_spec.depth
    assert np.all(garment.vertices[:, 2] == 0.0)
    assert not garment.is_watertight
    assert len(garment.split(only_watertight=False)) == 1

    for name in ("garbage_bag", "plastic_bag"):
        spec = CLOTH_SPECS[name]
        bag = cloth_mesh(spec)
        assert bag.extents == pytest.approx([spec.width, spec.depth, spec.width * 0.24])
        assert bag.area > spec.width * spec.depth * 2.0
        assert bag.is_winding_consistent
        assert not bag.is_watertight
        assert len(bag.split(only_watertight=False)) == 1


def test_pillow_is_a_closed_bulging_shell() -> None:
    """The pillow has a sealed, positive-volume shell without pressure semantics."""
    spec = CLOTH_SPECS["pillow"]
    pillow = cloth_mesh(spec)
    assert pillow.extents == pytest.approx(
        [spec.width, spec.depth, min(spec.width, spec.depth) * 0.4]
    )
    assert pillow.area > spec.width * spec.depth * 2.0
    assert pillow.volume > 0.0
    assert pillow.is_watertight
    assert pillow.is_winding_consistent
    assert len(pillow.split(only_watertight=False)) == 1


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
