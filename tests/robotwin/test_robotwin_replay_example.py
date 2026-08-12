"""Lightweight tests for the robotwin_replay example.

These tests import the example module and exercise its pure functions; they do
not launch Genesis, so they can run in CI without real robot assets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.robotwin.replay import _load_or_make_bridge, _make_synthetic_bridge


class TestSyntheticBridge:
    """Tests for the synthetic bridge helper used by the example."""

    def test_bridge_has_expected_structure(self) -> None:
        """Synthetic bridge contains frames, robot URDF, and a mesh object."""
        bridge = _make_synthetic_bridge(num_frames=10)

        assert bridge.task_name == "synthetic_smoke_test"
        assert len(bridge.frames) == 10
        assert bridge.fps == 20.0
        assert "can" in bridge.object_assets
        assert bridge.object_assets["can"].asset_type == "mesh"
        assert Path(bridge.robot_urdf).name == "robot.urdf"

    def test_frames_contain_14d_commands(self) -> None:
        """Each frame carries a 14-D robot command."""
        bridge = _make_synthetic_bridge(num_frames=5)
        for frame in bridge.frames:
            assert frame.robot_command.shape == (14,)
            assert frame.robot_achieved_qpos.shape == (14,)
            assert frame.robot_base_pos.shape == (3,)
            assert frame.robot_base_quat.shape == (4,)

    def test_load_or_make_bridge_falls_back_when_path_missing(self) -> None:
        """When a bridge path is missing, the helper returns a synthetic bridge."""
        bridge = _load_or_make_bridge("/nonexistent/path/episode.bridge")
        assert len(bridge.frames) > 0
        assert bridge.task_name == "synthetic_smoke_test"

    def test_synthetic_motion_is_nonzero(self) -> None:
        """At least one driven joint moves over the synthetic trajectory."""
        bridge = _make_synthetic_bridge(num_frames=60)
        commands = np.stack([frame.robot_command for frame in bridge.frames])
        assert np.any(np.abs(commands) > 1e-6)
