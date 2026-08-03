"""Tests for RoboTwin replay task and scene."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.bridge import (
    ObjectAsset,
    RobotwinBridge,
    RobotwinFrame,
)
from cloud_robotics_sim.robotwin.replay_scene import RobotwinReplayScene
from cloud_robotics_sim.robotwin.replay_task import RobotwinReplayTask


def _make_bridge(num_frames: int = 3) -> RobotwinBridge:
    """Create a minimal bridge for replay tests."""
    frames = []
    for i in range(num_frames):
        frames.append(
            RobotwinFrame(
                timestamp=i * 0.05,
                robot_command=np.arange(14, dtype=np.float64),
                robot_achieved_qpos=np.arange(14, dtype=np.float64),
                robot_base_pos=np.array([float(i), 0.0, 0.0]),
                robot_base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                object_states={
                    "can": {
                        "pos": np.array([0.5, float(i) * 0.1, 0.74]),
                        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                    }
                },
            )
        )
    return RobotwinBridge(
        task_name="move_can",
        seed=42,
        fps=20.0,
        robot_urdf="robot.urdf",
        table_height=0.74,
        object_assets={"can": ObjectAsset(name="can", asset_type="mesh", path="can.glb")},
        frames=frames,
    )


def _make_fake_robot() -> MagicMock:
    """Create a fake robot embodiment with settable base pose."""
    entity = MagicMock()
    entity._pos = np.zeros(3)
    entity._quat = np.array([1.0, 0.0, 0.0, 0.0])

    def _set_pos(p):
        entity._pos = np.asarray(p, dtype=np.float64)

    def _set_quat(q):
        entity._quat = np.asarray(q, dtype=np.float64)

    entity.set_pos.side_effect = _set_pos
    entity.set_quat.side_effect = _set_quat

    robot = MagicMock()
    robot.entity = entity
    return robot


class TestRobotwinReplayTask:
    """Tests for RobotwinReplayTask."""

    @pytest.fixture
    def scene(self) -> MagicMock:
        """Create a fake scene with one object entity."""
        scene = MagicMock()
        can_entity = MagicMock()
        scene.entities = {"can": can_entity}
        return scene

    def test_reset_applies_first_frame(self, scene: MagicMock) -> None:
        """Reset applies frame 0 robot and object states."""
        bridge = _make_bridge()
        task = RobotwinReplayTask(bridge)
        robot = _make_fake_robot()

        info = task.reset(scene, robot, seed=0)

        np.testing.assert_array_equal(robot.entity.set_pos.call_args[0][0], [0.0, 0.0, 0.0])
        robot.apply_action.assert_called_once()
        np.testing.assert_array_equal(
            robot.apply_action.call_args[0][0], np.arange(14, dtype=np.float64)
        )
        can = scene.entities["can"]
        np.testing.assert_array_equal(
            can.set_pos.call_args[0][0], [0.5, 0.0, 0.74]
        )
        assert info["frame_index"] == 0

    def test_step_advances_frame(self, scene: MagicMock) -> None:
        """Step advances the frame pointer and applies the next state."""
        bridge = _make_bridge(num_frames=3)
        task = RobotwinReplayTask(bridge)
        robot = _make_fake_robot()
        task.reset(scene, robot, seed=0)

        reward, terminated, truncated, info = task.step(scene, robot, np.zeros(14))

        can = scene.entities["can"]
        np.testing.assert_array_equal(
            can.set_pos.call_args[0][0], [0.5, 0.1, 0.74]
        )
        np.testing.assert_array_equal(
            robot.entity.set_pos.call_args[0][0], [1.0, 0.0, 0.0]
        )
        assert info["frame_index"] == 1
        assert not terminated

    def test_step_terminates_at_last_frame(self, scene: MagicMock) -> None:
        """Episode terminates once the final frame is reached."""
        bridge = _make_bridge(num_frames=2)
        task = RobotwinReplayTask(bridge)
        robot = _make_fake_robot()
        task.reset(scene, robot, seed=0)

        _, terminated1, _, _ = task.step(scene, robot, np.zeros(14))
        assert terminated1

        # Further steps should clamp to the last frame.
        _, terminated2, _, info = task.step(scene, robot, np.zeros(14))
        assert terminated2
        assert info["frame_index"] == 1

    def test_empty_bridge_does_not_crash(self, scene: MagicMock) -> None:
        """An empty bridge results in no-op resets and immediate termination."""
        bridge = _make_bridge(num_frames=0)
        task = RobotwinReplayTask(bridge)
        robot = _make_fake_robot()

        info = task.reset(scene, robot, seed=0)
        assert info["frame_index"] == 0

        reward, terminated, truncated, info = task.step(scene, robot, np.zeros(14))
        assert terminated


class TestRobotwinReplayScene:
    """Tests for RobotwinReplayScene object loading."""

    def test_build_loads_table_and_objects(self) -> None:
        """Build adds a table spawn and a mesh object spawn."""
        bridge = _make_bridge()
        scene = RobotwinReplayScene(bridge)

        assert len(scene.object_spawns) == 2
        names = {spawn.name for spawn in scene.object_spawns}
        assert names == {"table", "can"}
        table = next(spawn for spawn in scene.object_spawns if spawn.name == "table")
        assert table.shape_type == "box"

    def test_build_refuses_native_genesis_scene(self) -> None:
        """Replay scene requires the backend abstraction."""
        import genesis as gs

        bridge = _make_bridge()
        scene = RobotwinReplayScene(bridge)
        scene.scene = MagicMock(spec=gs.Scene)

        with pytest.raises(RuntimeError, match="backend abstraction"):
            scene._build_custom()
