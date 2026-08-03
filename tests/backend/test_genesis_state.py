"""Tests for Genesis backend physics state save/restore."""

from __future__ import annotations

from unittest.mock import MagicMock

import genesis as gs
import numpy as np
import pytest

from cloud_robotics_sim.backend.types import ArticulationState, PhysicsState
from cloud_robotics_sim.backends.genesis_backend import (
    GenesisArticulationBackend,
    GenesisSceneBackend,
)
from tests.conftest import MockEntity


def _make_genesis_articulation(name: str, n_qs: int = 7) -> GenesisArticulationBackend:
    """Create a GenesisArticulationBackend bound to a mocked entity."""
    backend = GenesisArticulationBackend(morph=MagicMock(), name=name)
    mock_entity = MagicMock()
    mock_entity.get_pos.return_value = np.zeros(3)
    mock_entity.get_quat.return_value = np.array([1.0, 0.0, 0.0, 0.0])
    mock_entity.get_qpos.return_value = np.zeros(n_qs)
    mock_entity.get_qvel.return_value = np.zeros(n_qs)

    def _set_pos(p):
        mock_entity.get_pos.return_value = np.asarray(p, dtype=np.float64)

    def _set_quat(q):
        mock_entity.get_quat.return_value = np.asarray(q, dtype=np.float64)

    def _set_qpos(q, qs_idx_local=None):
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        current = mock_entity.get_qpos.return_value.copy()
        if qs_idx_local is not None:
            for i, idx in enumerate(qs_idx_local):
                current[idx] = q_arr[i]
            mock_entity.get_qpos.return_value = current
        else:
            mock_entity.get_qpos.return_value = q_arr

    def _set_qvel(v):
        mock_entity.get_qvel.return_value = np.asarray(v, dtype=np.float64).flatten()

    mock_entity.set_pos.side_effect = _set_pos
    mock_entity.set_quat.side_effect = _set_quat
    mock_entity.set_qpos.side_effect = _set_qpos
    mock_entity.set_qvel.side_effect = _set_qvel
    backend.bind(mock_entity)
    return backend


class TestGenesisSceneState:
    """Tests for GenesisSceneBackend.get/set_physics_state."""

    @pytest.fixture
    def scene_backend(self) -> GenesisSceneBackend:
        """Create a GenesisSceneBackend with a mocked gs_scene."""
        mock_backend = MagicMock()
        mock_gs_scene = MagicMock()
        mock_gs_scene.t = 0.42
        return GenesisSceneBackend(mock_backend, mock_gs_scene)

    def test_get_physics_state_tracks_entities(
        self, scene_backend: GenesisSceneBackend
    ) -> None:
        """State snapshot includes all tracked entities."""
        box = MockEntity(name="box")
        box.set_pos(np.array([1.0, 2.0, 3.0]))
        box.set_quat(np.array([0.0, 0.0, 0.0, 1.0]))

        robot = _make_genesis_articulation("franka", n_qs=7)
        robot._entity.set_qpos(np.arange(7, dtype=np.float64))
        robot._entity.set_qvel(np.ones(7, dtype=np.float64))

        scene_backend._entities = [box, robot]

        state = scene_backend.get_physics_state()

        assert state.time == pytest.approx(0.42)
        entities = state.custom["entities"]
        assert "box" in entities
        assert "franka" in entities

        np.testing.assert_array_equal(entities["box"]["pos"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(entities["box"]["quat"], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(entities["franka"]["qpos"], np.arange(7))
        np.testing.assert_array_equal(entities["franka"]["qvel"], np.ones(7))

    def test_set_physics_state_restores_entities(
        self, scene_backend: GenesisSceneBackend
    ) -> None:
        """State restore updates tracked entity poses and joint states."""
        box = MockEntity(name="box")
        robot = _make_genesis_articulation("franka", n_qs=7)
        scene_backend._entities = [box, robot]

        saved_entities = {
            "box": {
                "pos": np.array([1.0, 2.0, 3.0]),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            "franka": {
                "pos": np.array([0.5, 0.0, 0.0]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "qpos": np.arange(7, dtype=np.float64),
                "qvel": np.ones(7, dtype=np.float64),
            },
        }
        state = PhysicsState(time=0.0, custom={"entities": saved_entities})
        scene_backend.set_physics_state(state)

        np.testing.assert_array_equal(box.get_pos(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(robot.get_pos(), [0.5, 0.0, 0.0])
        robot._entity.set_qpos.assert_called_once()
        robot._entity.set_qvel.assert_called_once()
        np.testing.assert_array_equal(robot.get_qvel(), np.ones(7))

    def test_set_physics_state_missing_entities_logs_debug(
        self, scene_backend: GenesisSceneBackend, caplog
    ) -> None:
        """Restore with empty custom dict is a no-op."""
        box = MockEntity(name="box")
        scene_backend._entities = [box]

        state = PhysicsState(time=0.0, custom={})
        with caplog.at_level("DEBUG"):
            scene_backend.set_physics_state(state)

        assert "No entity states found" in caplog.text


class TestGenesisArticulationState:
    """Tests for GenesisArticulationBackend state batch helpers."""

    def test_set_state_batch_with_free_joint_skips_root(
        self,
    ) -> None:
        """For floating-base robots, qpos is applied only to actuated joints."""
        backend = GenesisArticulationBackend(morph=MagicMock(), name="floating_robot")
        mock_entity = MagicMock()

        free_joint = MagicMock()
        free_joint.type = gs.JOINT_TYPE.FREE
        free_joint.qs_idx_local = list(range(7))

        revolute_joint = MagicMock()
        revolute_joint.type = gs.JOINT_TYPE.REVOLUTE
        revolute_joint.qs_idx_local = [7]

        mock_entity.joints = [free_joint, revolute_joint]
        backend.bind(mock_entity)

        backend.set_state_batch(
            ArticulationState(qpos=np.array([0.1]), qvel=None, pos=None, quat=None)
        )

        mock_entity.set_qpos.assert_called_once()
        call_args = mock_entity.set_qpos.call_args
        assert call_args.kwargs.get("qs_idx_local") == [7]

    def test_set_state_batch_fixed_base_uses_full_qpos(
        self,
    ) -> None:
        """For fixed-base robots, qpos is applied directly."""
        backend = GenesisArticulationBackend(morph=MagicMock(), name="fixed_robot")
        mock_entity = MagicMock()

        revolute_joint = MagicMock()
        revolute_joint.type = gs.JOINT_TYPE.REVOLUTE
        revolute_joint.qs_idx_local = list(range(7))

        mock_entity.joints = [revolute_joint]
        backend.bind(mock_entity)

        backend.set_state_batch(
            ArticulationState(
                qpos=np.arange(7, dtype=np.float64), qvel=None, pos=None, quat=None
            )
        )

        mock_entity.set_qpos.assert_called_once()
        call_args = mock_entity.set_qpos.call_args
        np.testing.assert_array_equal(call_args.args[0], np.arange(7, dtype=np.float64))
        assert call_args.kwargs.get("qs_idx_local") == list(range(7))
