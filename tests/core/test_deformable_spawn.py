"""Tests for deformable object spawning through ObjectSpawn."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cloud_robotics_sim.backend.types import DeformableConfig, DeformableMaterialType
from cloud_robotics_sim.core.scene import ObjectLibrary, ObjectSpawn


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a mock simulator backend that supports create_deformable."""
    backend = MagicMock()
    backend.create_deformable.return_value = MagicMock()
    return backend


@pytest.fixture
def mock_scene_backend(mock_backend: MagicMock) -> MagicMock:
    """Return a mock scene backend attached to the mock simulator backend."""
    scene = MagicMock()
    scene.backend = mock_backend
    return scene


def test_object_spawn_deformable_requires_config(
    mock_scene_backend: MagicMock,
) -> None:
    """A deformable spawn without deformable_config should raise ValueError."""
    spawn = ObjectSpawn(
        name="bad_soft_cube",
        shape_type="deformable",
        size=(0.1, 0.1, 0.1),
        position=(0.0, 0.0, 0.05),
    )
    with pytest.raises(ValueError):
        spawn.spawn(mock_scene_backend)


def test_object_spawn_deformable_box(mock_scene_backend: MagicMock) -> None:
    """ObjectLibrary.deformable_soft_cube should forward a box deformable."""
    config = DeformableConfig(
        material=DeformableMaterialType.FEM_ELASTIC,
        resolution_level=3,
    )
    spawn = ObjectLibrary.deformable_soft_cube(
        name="soft_cube",
        position=(0.5, 0.0, 0.04),
        size=0.08,
        deformable_config=config,
    )
    entity = spawn.spawn(mock_scene_backend)

    assert entity is not None
    mock_scene_backend.backend.create_deformable.assert_called_once()
    call_kwargs = mock_scene_backend.backend.create_deformable.call_args.kwargs
    assert call_kwargs["config"] is config
    assert call_kwargs["shape"] == "box"
    assert call_kwargs["size"] == (0.08, 0.08, 0.08)


def test_object_spawn_deformable_shape_hint(mock_scene_backend: MagicMock) -> None:
    """The deformable_shape property should select a sphere primitive."""
    config = DeformableConfig()
    spawn = ObjectSpawn(
        name="soft_sphere",
        shape_type="deformable",
        size=(0.1, 0.1, 0.1),
        position=(0.0, 0.0, 0.05),
        deformable_config=config,
        properties={"deformable_shape": "sphere"},
    )
    spawn.spawn(mock_scene_backend)

    call_kwargs = mock_scene_backend.backend.create_deformable.call_args.kwargs
    assert call_kwargs["shape"] == "sphere"
    assert call_kwargs["radius"] == 0.1
