"""Tests for RoboTwin bridge data structures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.bridge import (
    CameraConfig,
    ObjectAsset,
    RobotwinBridge,
    RobotwinFrame,
)
from cloud_robotics_sim.robotwin.loader import RobotwinBridgeLoader


class TestRobotwinBridge:
    """Tests for bridge serialization and helpers."""

    @pytest.fixture
    def sample_bridge(self) -> RobotwinBridge:
        """Create a minimal bridge for testing."""
        frame = RobotwinFrame(
            timestamp=0.0,
            robot_command=np.arange(14, dtype=np.float64),
            robot_achieved_qpos=np.arange(14, dtype=np.float64) + 0.5,
            robot_base_pos=np.array([0.0, 0.0, 0.74]),
            robot_base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            object_states={
                "can": {
                    "pos": np.array([0.2, 0.1, 0.78]),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                }
            },
        )
        return RobotwinBridge(
            task_name="move_can_pot",
            seed=42,
            fps=25.0,
            robot_urdf="assets/embodiments/aloha_agilex/arx5_description_isaac.urdf",
            robot_joint_names=[f"joint_{i}" for i in range(14)],
            table_height=0.74,
            object_assets={
                "can": ObjectAsset(
                    name="can",
                    asset_type="mesh",
                    path="assets/objects/can/model.glb",
                    scale=1.0,
                )
            },
            cameras=[
                CameraConfig(
                    name="head_camera",
                    pos=(0.4, 0.0, 1.5),
                    look_at=(0.0, 0.0, 0.78),
                    resolution=(640, 480),
                )
            ],
            frames=[frame],
        )

    def test_bridge_summary(self, sample_bridge: RobotwinBridge) -> None:
        """Summary reports key metadata."""
        summary = sample_bridge.summary()
        assert summary["task_name"] == "move_can_pot"
        assert summary["seed"] == 42
        assert summary["fps"] == 25.0
        assert summary["num_frames"] == 1
        assert summary["num_objects"] == 1
        assert summary["num_cameras"] == 1

    def test_bridge_roundtrip(self, sample_bridge: RobotwinBridge) -> None:
        """Save and load preserves all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.pkl"
            sample_bridge.save(path)
            loaded = RobotwinBridge.load(path)

        assert loaded.task_name == sample_bridge.task_name
        assert loaded.seed == sample_bridge.seed
        assert loaded.fps == sample_bridge.fps
        assert loaded.robot_urdf == sample_bridge.robot_urdf
        np.testing.assert_array_equal(
            loaded.frames[0].robot_command,
            sample_bridge.frames[0].robot_command,
        )
        np.testing.assert_array_equal(
            loaded.frames[0].object_states["can"]["pos"],
            sample_bridge.frames[0].object_states["can"]["pos"],
        )

    def test_loader_iteration(self, sample_bridge: RobotwinBridge) -> None:
        """Loader iterates over frames and exposes metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.pkl"
            sample_bridge.save(path)
            loader = RobotwinBridgeLoader(path)

        assert loader.task_name == "move_can_pot"
        assert loader.num_frames == 1
        frames = list(loader)
        assert len(frames) == 1
        np.testing.assert_array_equal(frames[0].robot_command, np.arange(14))

    def test_loader_random_access(self, sample_bridge: RobotwinBridge) -> None:
        """Loader supports indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.pkl"
            sample_bridge.save(path)
            loader = RobotwinBridgeLoader(path)

        frame = loader[0]
        np.testing.assert_array_equal(
            frame.robot_achieved_qpos, np.arange(14) + 0.5
        )
