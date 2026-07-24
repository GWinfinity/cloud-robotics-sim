"""Shared test fixtures and mock classes for cloud_robotics_sim tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from cloud_robotics_sim.backend import (
    ArticulationBackend,
    ArticulationState,
    EntityBackend,
    Pose,
    SceneBackend,
)

# ---------------------------------------------------------------------------
# Shared mock classes
# ---------------------------------------------------------------------------


class MockEntity(EntityBackend):
    """Minimal entity backend for unit tests."""

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self._pos = np.zeros(3, dtype=np.float64)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def name(self) -> str | None:
        return self._name

    def get_pos(self) -> np.ndarray:
        return self._pos.copy()

    def set_pos(self, pos: np.ndarray) -> None:
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self) -> np.ndarray:
        return self._quat.copy()

    def set_quat(self, quat: np.ndarray) -> None:
        self._quat = np.asarray(quat, dtype=np.float64)

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        pass

    def apply_force(
        self,
        force: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> None:
        pass


class MockArticulation(MockEntity, ArticulationBackend):
    """Minimal articulation backend for unit tests.

    Tracks ``set_qpos`` and ``control_dofs_position`` calls so tests can
    assert on state changes without a real physics engine.
    """

    def __init__(
        self,
        name: str | None = None,
        n_dofs: int = 7,
        n_qs: int = 7,
    ) -> None:
        super().__init__(name)
        self._n_dofs = n_dofs
        self._n_qs = n_qs
        self._qpos = np.zeros(n_qs, dtype=np.float64)
        self._qvel = np.zeros(n_qs, dtype=np.float64)
        self.set_qpos_calls: list[np.ndarray] = []
        self.control_dofs_position_calls: list[np.ndarray] = []

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def n_qs(self) -> int:
        return self._n_qs

    def get_qpos(self) -> np.ndarray:
        return self._qpos.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        self._qpos = np.asarray(qpos, dtype=np.float64)
        self.set_qpos_calls.append(self._qpos.copy())

    def get_qvel(self) -> np.ndarray:
        return self._qvel.copy()

    def set_qvel(self, qvel: np.ndarray) -> None:
        self._qvel = np.asarray(qvel, dtype=np.float64)

    def control_dofs_position(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        arr = np.asarray(targets, dtype=np.float64)
        self.control_dofs_position_calls.append(arr)
        if arr.ndim == 1:
            self._qpos[: len(arr)] = arr
        elif arr.ndim == 2:
            self._qpos[: arr.shape[1]] = arr[0]

    def control_dofs_velocity(self, targets: np.ndarray) -> None:
        pass

    def control_dofs_force(self, targets: np.ndarray) -> None:
        pass

    def get_state_batch(
        self,
        env_ids: list[int] | None = None,
    ) -> ArticulationState:
        return ArticulationState(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            pos=self.get_pos(),
            quat=self.get_quat(),
        )

    def set_state_batch(
        self,
        state: ArticulationState,
        env_ids: list[int] | None = None,
    ) -> None:
        if state.qpos is not None:
            self.set_qpos(state.qpos)
        if state.qvel is not None:
            self.set_qvel(state.qvel)
        if state.pos is not None:
            self.set_pos(state.pos)
        if state.quat is not None:
            self.set_quat(state.quat)

    def control_position_batch(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        self.control_dofs_position(targets, stiffness=stiffness, damping=damping)

    def get_end_effector_pose(self) -> Pose:
        return Pose(pos=self.get_pos(), quat=self.get_quat())


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------


def make_mock_scene_backend() -> MagicMock:
    """Create a MagicMock scene with a mock simulator backend attached."""
    backend = MagicMock()
    scene = MagicMock(spec=SceneBackend)
    scene.backend = backend
    return scene
