"""Tests for RoboTwin spawn material resolution via robomat.

These tests mock the Genesis scene/module so they do not require a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from cloud_robotics_sim.robotwin.object_library import (
    RoboTwinObjectLibrary,
    _resolve_robotwin_material,
)


def _make_class(root: Path, name: str) -> None:
    class_dir = root / name
    (class_dir / "collision").mkdir(parents=True)
    (class_dir / "model_data0.json").write_text(
        json.dumps(
            {
                "center": [0.0, 0.0, 0.0],
                "extents": [0.1, 0.1, 0.1],
                "scale": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    (class_dir / "collision" / "base0.glb").write_bytes(b"glb")


def _fake_gs_module() -> Any:
    """Return a fake genesis module with minimal morph/surface/material types."""

    class FakeRigid:
        def __init__(self, rho: float, friction: float = 0.5):
            self.rho = rho
            self.friction = friction

    class FakeSurface:
        def __init__(self, roughness: float = 0.6):
            self.roughness = roughness

    class FakeMesh:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            FakeMesh.calls.append(kwargs)

    class FakeURDF:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            FakeURDF.calls.append(kwargs)

    FakeMesh.calls.clear()
    FakeURDF.calls.clear()

    gs = SimpleNamespace(
        surfaces=SimpleNamespace(Default=FakeSurface),
        materials=SimpleNamespace(Rigid=FakeRigid),
        morphs=SimpleNamespace(Mesh=FakeMesh, URDF=FakeURDF),
    )
    return gs


@pytest.fixture()
def fake_library(tmp_path: Path) -> RoboTwinObjectLibrary:
    """Return a RoboTwinObjectLibrary with a single fake 021_cup class."""
    _make_class(tmp_path, "021_cup")
    return RoboTwinObjectLibrary(tmp_path)


def test_resolve_robotwin_material_returns_rigid() -> None:
    """Robomat should resolve a known RoboTwin class to a Genesis Rigid material."""
    material = _resolve_robotwin_material("021_cup")
    assert material is not None
    assert hasattr(material, "rho")
    assert hasattr(material, "friction")


def test_resolve_robotwin_material_unknown_class() -> None:
    """Unknown classes fall back to None without raising."""
    assert _resolve_robotwin_material("zzz_unknown_class_999") is None


def test_spawn_mesh_passes_material(
    fake_library: RoboTwinObjectLibrary, tmp_path: Path
) -> None:
    """spawn_in_scene forwards a resolved material to gs.morphs.Mesh."""
    fake_gs = _fake_gs_module()
    scene = SimpleNamespace(add_entity=lambda morph, surface=None: morph)

    with mock.patch.dict(sys.modules, {"genesis": fake_gs, "trimesh": None}):
        fake_library.spawn_in_scene(scene, "021_cup", pos=(0.0, 0.0, 0.0))

    assert len(fake_gs.morphs.Mesh.calls) == 1
    call = fake_gs.morphs.Mesh.calls[0]
    assert "material" in call
    assert call["material"] is not None
    assert call["material"].rho > 0


def test_spawn_urdf_passes_material(tmp_path: Path) -> None:
    """spawn_in_scene forwards a resolved material to gs.morphs.URDF."""
    class_dir = tmp_path / "009_kettle"
    sub = class_dir / "102730"
    sub.mkdir(parents=True)
    (sub / "mobility.urdf").write_text("<robot name='x'/>", encoding="utf-8")
    (sub / "model_data.json").write_text(json.dumps({"scale": [0.2]}), encoding="utf-8")
    lib = RoboTwinObjectLibrary(tmp_path)

    fake_gs = _fake_gs_module()
    scene = SimpleNamespace(add_entity=lambda morph, surface=None: morph)

    with mock.patch.dict(sys.modules, {"genesis": fake_gs}):
        lib.spawn_in_scene(scene, "009_kettle", pos=(0.0, 0.0, 0.0))

    assert len(fake_gs.morphs.URDF.calls) == 1
    call = fake_gs.morphs.URDF.calls[0]
    assert "material" in call
    assert call["material"] is not None


def test_spawn_falls_back_when_robomat_unavailable(
    fake_library: RoboTwinObjectLibrary,
) -> None:
    """If robomat cannot be imported, spawn still works with material=None."""
    fake_gs = _fake_gs_module()
    scene = SimpleNamespace(add_entity=lambda morph, surface=None: morph)

    with mock.patch.dict(
        sys.modules, {"robomat": None, "genesis": fake_gs, "trimesh": None}
    ):
        fake_library.spawn_in_scene(scene, "021_cup", pos=(0.0, 0.0, 0.0))

    assert len(fake_gs.morphs.Mesh.calls) == 1
    assert fake_gs.morphs.Mesh.calls[0]["material"] is None
