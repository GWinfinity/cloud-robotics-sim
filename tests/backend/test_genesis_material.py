"""Tests for GenesisBackend material resolution via robomat."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

import cloud_robotics_sim.backends.genesis_backend as genesis_backend_module
from cloud_robotics_sim.backends.genesis_backend import GenesisBackend


def _fake_gs_module() -> Any:
    """Return a minimal fake genesis module for create_mesh tests."""

    class FakeRigid:
        def __init__(self, rho: float, friction: float = 0.5):
            self.rho = rho
            self.friction = friction

    class FakeSurface:
        def __init__(self, color: Any = None, roughness: float = 0.8):
            self.color = color
            self.roughness = roughness

    class FakeMesh:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            FakeMesh.calls.append(kwargs)

    FakeMesh.calls.clear()

    return SimpleNamespace(
        surfaces=SimpleNamespace(Default=FakeSurface),
        materials=SimpleNamespace(Rigid=FakeRigid),
        morphs=SimpleNamespace(Mesh=FakeMesh),
    )


@pytest.fixture()
def backend() -> GenesisBackend:
    """Return a GenesisBackend whose _gs_backend is a stub."""
    return GenesisBackend()


def test_create_mesh_with_default_material_uses_no_physics_material(
    backend: GenesisBackend,
) -> None:
    """When material is 'default', no physics material is injected."""
    fake_gs = _fake_gs_module()
    backend._gs_backend = SimpleNamespace()  # type: ignore[attr-defined]

    with mock.patch.object(genesis_backend_module, "gs", fake_gs):
        entity = backend.create_mesh(
            file="mesh.obj",
            pos=(0.0, 0.0, 0.0),
            material="default",
        )

    assert entity is not None
    assert len(fake_gs.morphs.Mesh.calls) == 1
    assert "material" not in fake_gs.morphs.Mesh.calls[0]


def test_create_mesh_with_robomat_material(backend: GenesisBackend) -> None:
    """A known robomat material hint is converted to gs.materials.Rigid."""
    fake_gs = _fake_gs_module()
    backend._gs_backend = SimpleNamespace()  # type: ignore[attr-defined]

    with mock.patch.object(genesis_backend_module, "gs", fake_gs):
        entity = backend.create_mesh(
            file="cup.obj",
            pos=(0.0, 0.0, 0.0),
            material="021_cup",
        )

    assert entity is not None
    assert len(fake_gs.morphs.Mesh.calls) == 1
    mesh_call = fake_gs.morphs.Mesh.calls[0]
    assert "material" in mesh_call
    assert mesh_call["material"] is not None
    assert mesh_call["material"].rho > 0


def test_create_mesh_falls_back_on_unknown_material(backend: GenesisBackend) -> None:
    """Unknown material hints do not break mesh creation."""
    fake_gs = _fake_gs_module()
    backend._gs_backend = SimpleNamespace()  # type: ignore[attr-defined]

    with mock.patch.object(genesis_backend_module, "gs", fake_gs):
        entity = backend.create_mesh(
            file="unknown.obj",
            pos=(0.0, 0.0, 0.0),
            material="zzz_qwerty_12345",
        )

    assert entity is not None
    mesh_call = fake_gs.morphs.Mesh.calls[0]
    assert mesh_call.get("material") is None


def test_create_mesh_falls_back_when_robomat_unavailable(
    backend: GenesisBackend,
) -> None:
    """If robomat is not importable, creation still succeeds."""
    fake_gs = _fake_gs_module()
    backend._gs_backend = SimpleNamespace()  # type: ignore[attr-defined]

    with (
        mock.patch.object(genesis_backend_module, "gs", fake_gs),
        mock.patch.dict("sys.modules", {"robomat": None}),
    ):
        entity = backend.create_mesh(
            file="cup.obj",
            pos=(0.0, 0.0, 0.0),
            material="021_cup",
        )

    assert entity is not None
    mesh_call = fake_gs.morphs.Mesh.calls[0]
    assert mesh_call.get("material") is None
