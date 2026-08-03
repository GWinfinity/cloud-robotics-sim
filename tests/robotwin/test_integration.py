"""Integration tests for the RoboTwin replay pipeline.

These tests exercise the interaction between the bridge, scene, robot, and task
using mock backends so they do not require real Genesis assets or a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.backend import ArticulationBackend, SceneBackend
from cloud_robotics_sim.robotwin.bridge import (
    ObjectAsset,
    RobotwinBridge,
    RobotwinFrame,
)
from cloud_robotics_sim.robotwin.dual_arm_embodiment import (
    AlohaAgileX,
    AlohaAgileXConfig,
)
from cloud_robotics_sim.robotwin.replay_scene import RobotwinReplayScene
from cloud_robotics_sim.robotwin.replay_task import RobotwinReplayTask


def _make_bridge(num_frames: int = 3) -> RobotwinBridge:
    """Create a small bridge with one mesh object and one URDF object."""
    frames = [
        RobotwinFrame(
            timestamp=i * 0.05,
            robot_command=np.zeros(14, dtype=np.float64),
            robot_achieved_qpos=np.zeros(14, dtype=np.float64),
            robot_base_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            robot_base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            object_states={
                "can": {
                    "pos": np.array([0.5, 0.0, 0.74], dtype=np.float64),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                },
                "drawer": {
                    "pos": np.array([0.3, 0.2, 0.74], dtype=np.float64),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "qpos": np.array([0.1], dtype=np.float64),
                },
            },
        )
        for i in range(num_frames)
    ]
    return RobotwinBridge(
        task_name="integration_test",
        seed=0,
        fps=20.0,
        robot_urdf="robot.urdf",
        table_height=0.74,
        object_assets={
            "can": ObjectAsset(name="can", asset_type="mesh", path="can.glb"),
            "drawer": ObjectAsset(
                name="drawer",
                asset_type="urdf",
                path="drawer.urdf",
                is_articulation=True,
            ),
        },
        frames=frames,
    )


def _make_mock_backend() -> MagicMock:
    """Create a mock backend that returns mock entities."""
    backend = MagicMock()
    backend.create_box.return_value = MagicMock(name="box_entity")
    backend.create_mesh.return_value = MagicMock(name="mesh_entity")
    backend.load_urdf.return_value = MagicMock(
        spec=ArticulationBackend, name="urdf_entity"
    )
    return backend


def _make_mock_scene_backend(backend: MagicMock) -> MagicMock:
    """Create a mock scene backend wrapping the given backend."""
    scene_backend = MagicMock(spec=SceneBackend)
    scene_backend.backend = backend
    scene_backend.add_entity = MagicMock()
    scene_backend.add_articulation = MagicMock()
    return scene_backend


class TestReplaySceneIntegration:
    """Integration tests for RobotwinReplayScene.build."""

    def test_build_spawns_table_and_mesh_objects(self) -> None:
        """The scene spawns a table and registers mesh objects through the backend."""
        bridge = _make_bridge()
        scene = RobotwinReplayScene(bridge)
        backend = _make_mock_backend()
        scene_backend = _make_mock_scene_backend(backend)

        scene.build(scene_backend)

        # Room structure creates 4 walls + floor; replay scene adds 1 table + 1 mesh object.
        table_calls = [
            c
            for c in backend.create_box.call_args_list
            if c.kwargs.get("name", "").endswith("_table")
        ]
        assert len(table_calls) == 1
        backend.create_mesh.assert_called_once()
        assert scene_backend.add_entity.call_count >= 2

    def test_build_loads_urdf_objects(self) -> None:
        """URDF object assets are loaded as articulated objects."""
        bridge = _make_bridge()
        scene = RobotwinReplayScene(bridge)
        backend = _make_mock_backend()
        scene_backend = _make_mock_scene_backend(backend)

        scene.build(scene_backend)

        backend.load_urdf.assert_called_once_with(
            file="drawer.urdf",
            pos=(0.0, 0.0, 0.79),
            fixed=False,
            scale=(1.0, 1.0, 1.0),
        )
        scene_backend.add_articulation.assert_called_once()
        assert "drawer" in scene.entities

    def test_build_entities_include_spawned_objects(self) -> None:
        """Scene.entities contains the URDF object after build."""
        bridge = _make_bridge()
        scene = RobotwinReplayScene(bridge)
        backend = _make_mock_backend()
        scene_backend = _make_mock_scene_backend(backend)

        scene.build(scene_backend)

        assert "drawer" in scene.entities
        assert scene.get_articulation("drawer") is backend.load_urdf.return_value


class TestAlohaAgileXIntegration:
    """Integration tests for AlohaAgileX spawn."""

    def test_spawn_loads_urdf_and_adds_articulation(self) -> None:
        """AlohaAgileX.spawn loads the configured URDF through the backend."""
        robot = AlohaAgileX(
            AlohaAgileXConfig(
                urdf_path="assets/embodiments/aloha-agilex/urdf/robot.urdf"
            )
        )
        backend = _make_mock_backend()
        scene_backend = _make_mock_scene_backend(backend)

        # Mock raw joints so joint mapping can be built.
        raw_entity = MagicMock()
        raw_entity.joints = []
        backend.load_urdf.return_value._entity = raw_entity
        backend.load_urdf.return_value.name = "aloha_agilex"

        robot.spawn(scene_backend, position=(0.0, 0.0, 0.0))

        backend.load_urdf.assert_called_once_with(
            file="assets/embodiments/aloha-agilex/urdf/robot.urdf",
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        )
        scene_backend.add_articulation.assert_called_once_with(robot.entity)


class TestReplayTaskIntegration:
    """Integration tests for replay task with a mock robot and scene."""

    @pytest.fixture
    def robot(self) -> MagicMock:
        """Fake robot embodiment."""
        entity = MagicMock()
        entity.set_pos = MagicMock()
        entity.set_quat = MagicMock()

        robot = MagicMock()
        robot.entity = entity
        robot.action_dim = 14
        return robot

    @pytest.fixture
    def scene(self) -> MagicMock:
        """Fake scene with a tracked object."""
        scene = MagicMock()
        can_entity = MagicMock()
        drawer_entity = MagicMock()
        drawer_entity.set_qpos = MagicMock()
        scene.entities = {"can": can_entity, "drawer": drawer_entity}
        return scene

    def test_reset_and_step_drive_robot_and_objects(
        self, scene: MagicMock, robot: MagicMock
    ) -> None:
        """The replay task applies frame states to robot and objects."""
        bridge = _make_bridge(num_frames=3)
        task = RobotwinReplayTask(bridge)

        task.reset(scene, robot, seed=0)
        robot.entity.set_pos.assert_called_once()
        robot.apply_action.assert_called_once()

        _, terminated, _, _ = task.step(scene, robot, np.zeros(14))
        assert task.current_frame is bridge.frames[1]
        assert not terminated

        task.step(scene, robot, np.zeros(14))
        _, terminated2, _, _ = task.step(scene, robot, np.zeros(14))
        assert terminated2
        assert task.is_finished
