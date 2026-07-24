"""Unit tests for the backend abstraction layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.backend import (
    ArticulationBackend,
    BackendName,
    EntityBackend,
    LightDescription,
    LightType,
    RendererBackend,
    SceneBackend,
    SimulatorBackend,
    ViewerOptions,
    available_backends,
    get_backend,
)
from tests.conftest import MockArticulation, MockEntity


class MockRenderer(RendererBackend):
    """A minimal renderer backend for testing."""

    def __init__(self) -> None:
        self._cameras: dict[str, Any] = {}

    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float],
        resolution: tuple[int, int],
        fov: float = 60.0,
    ) -> Any:
        self._cameras[name] = MagicMock()
        return self._cameras[name]

    def render(
        self,
        camera_name: str | None = None,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Any:
        return MagicMock()

    def render_async(
        self,
        camera_names: list[str],
        *,
        rgb: bool = True,
        depth: bool = False,
    ) -> dict[str, Any]:
        return {name: MagicMock() for name in camera_names}


class MockScene(SceneBackend):
    """A minimal scene backend for testing."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend
        self._renderer = MockRenderer()
        self.entities: list[EntityBackend] = []
        self.articulations: list[ArticulationBackend] = []
        self.lights: list[LightDescription] = []
        self.built = False

    @property
    def backend(self) -> SimulatorBackend:
        return self._backend

    @property
    def renderer(self) -> RendererBackend | None:
        return self._renderer

    def add_entity(self, entity: EntityBackend) -> None:
        self.entities.append(entity)

    def add_articulation(self, articulation: ArticulationBackend) -> None:
        self.articulations.append(articulation)

    def add_light(self, light: LightDescription) -> None:
        self.lights.append(light)

    def build(self) -> None:
        self.built = True

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_physics_state(self) -> Any:
        return MagicMock()

    def set_physics_state(self, state: Any) -> None:
        pass


class MockBackend(SimulatorBackend):
    """A minimal simulator backend for testing."""

    def __init__(self) -> None:
        self.initialized = False
        self._scene: MockScene | None = None

    @property
    def name(self) -> BackendName:
        return BackendName.MUJOCO

    def initialize(
        self,
        *,
        headless: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.initialized = True

    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
    ) -> SceneBackend:
        self._scene = MockScene(self)
        return self._scene

    def create_box(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> EntityBackend:
        return MockEntity(name=kwargs.get("name"))

    def create_sphere(
        self,
        radius: float,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> EntityBackend:
        return MockEntity(name=kwargs.get("name"))

    def create_cylinder(
        self,
        radius: float,
        height: float,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> EntityBackend:
        return MockEntity(name=kwargs.get("name"))

    def create_mesh(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> EntityBackend:
        return MockEntity(name=kwargs.get("name"))

    def load_mjcf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        return MockArticulation(name=file)

    def load_urdf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        return MockArticulation(name=file)

    def create_light(self, description: LightDescription) -> LightDescription:
        return description


def test_cannot_instantiate_abstract_backend() -> None:
    """Verify abstract SimulatorBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SimulatorBackend()


def test_mock_backend_workflow() -> None:
    """Exercise the full backend workflow with a mock implementation."""
    backend = MockBackend()
    backend.initialize(headless=True, device="cpu")
    assert backend.initialized

    scene = backend.create_scene(dt=0.01, substeps=10)
    assert isinstance(scene, MockScene)
    assert scene.backend is backend

    box = backend.create_box(size=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0), name="test_box")
    scene.add_entity(box)
    assert len(scene.entities) == 1

    robot = backend.load_mjcf(file="test.xml", pos=(0.0, 0.0, 0.1))
    scene.add_articulation(robot)
    assert len(scene.articulations) == 1

    scene.add_light(
        LightDescription(
            light_type=LightType.DIRECTIONAL,
            pos=(1.0, 2.0, 3.0),
        )
    )
    assert len(scene.lights) == 1

    scene.build()
    assert scene.built


def test_articulation_batch_interface() -> None:
    """Verify batch state/control methods on a mock articulation."""
    robot = MockArticulation(n_dofs=7, n_qs=7)
    targets = np.ones((1, 7), dtype=np.float64)
    robot.control_position_batch(targets)
    np.testing.assert_array_equal(robot.get_qpos(), np.ones(7))

    state = robot.get_state_batch()
    assert state.qpos is not None
    assert state.qpos.shape == (7,)


@pytest.mark.parametrize(
    "backend_name",
    [BackendName.GENESIS, BackendName.MT_LAMBDA],
    ids=["genesis", "mt_lambda"],
)
def test_factory_returns_backend_by_name(backend_name: BackendName) -> None:
    """Factory should return a backend instance matching the requested name."""
    backend = get_backend(backend_name)
    assert backend.name == backend_name


def test_available_backends_contains_expected() -> None:
    """Available backends should include Genesis and MT Lambda."""
    names = available_backends()
    assert BackendName.GENESIS in names
    assert BackendName.MT_LAMBDA in names


def test_mt_lambda_backend_inherits_genesis_backend() -> None:
    """MTLambdaBackend should be a Genesis-compatible backend variant."""
    from cloud_robotics_sim.backends.genesis_backend import GenesisBackend
    from cloud_robotics_sim.backends.mt_lambda_backend import MTLambdaBackend

    backend = get_backend(BackendName.MT_LAMBDA)
    assert isinstance(backend, MTLambdaBackend)
    assert isinstance(backend, GenesisBackend)
    assert backend.name == BackendName.MT_LAMBDA
