"""Tests for the genesis_aloha_demo example and real-Genesis integration.

The lightweight tests import the example module and exercise its pure
helpers without launching Genesis. The ``slow`` integration test runs the
full IK / plan_path / DR pipeline on the real Genesis CPU backend using the
example's synthetic arm.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from examples.robotwin.aloha_demo import (
    SYNTHETIC_ARM_URDF,
    _write_synthetic_arm,
    main,
)


class TestSyntheticArmHelper:
    """Lightweight tests for the example's fallback URDF."""

    def test_urdf_is_valid_xml(self) -> None:
        root = ET.fromstring(SYNTHETIC_ARM_URDF)
        assert root.tag == "robot"
        joint_types = [j.get("type") for j in root.iter("joint")]
        assert "revolute" in joint_types
        assert "floating" not in joint_types  # fixed-base checklist (doc 4.1)

    def test_write_synthetic_arm(self, tmp_path: Path) -> None:
        urdf = _write_synthetic_arm(tmp_path)
        assert urdf.exists()
        ET.parse(str(urdf))

    def test_main_rejects_missing_urdf(self, tmp_path: Path) -> None:
        rc = main_for(
            ["--urdf", str(tmp_path / "nope.urdf"), "--out", str(tmp_path / "out")]
        )
        assert rc == 2


def main_for(argv: list[str]) -> int:
    """Run the example's main() with a custom argv."""
    old_argv = sys.argv
    try:
        sys.argv = ["aloha_demo.py", *argv]
        return main()
    finally:
        sys.argv = old_argv


@pytest.mark.slow
class TestGenesisRealIntegration:
    """End-to-end IK / plan_path / DR on the real Genesis CPU backend."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        pytest.importorskip("genesis")
        from cloud_robotics_sim.backends.genesis_backend import GenesisBackend

        urdf = _write_synthetic_arm(tmp_path)
        backend = GenesisBackend()
        backend.initialize(headless=True, device="cpu")
        scene = backend.create_scene(dt=0.01, substeps=1, headless=True)
        robot = backend.load_urdf(str(urdf), pos=(0.0, 0.0, 0.0), fixed=True)
        scene.add_articulation(robot)
        scene.build()

        assert robot.n_dofs == 2
        assert robot.is_fixed_base()

        # PD gains (doc section 2 config.yml mapping).
        robot.set_dofs_gains(np.full(2, 50.0), np.full(2, 5.0))

        # IK -> plan_path (doc section 5.2: OMPL replaces mplib RRT).
        q_goal = robot.inverse_kinematics("link2", pos=np.array([0.0, 0.0, 0.3]))
        assert q_goal.shape[-1] == robot.n_dofs
        traj = robot.plan_path(np.asarray(q_goal).reshape(-1), num_waypoints=10)
        assert traj.shape[-1] == robot.n_dofs
        assert traj.shape[0] >= 1

        # Link pose + DR (doc section 2).
        pose = robot.get_link_pose("link2")
        assert pose.pos.shape == (3,)
        assert pose.quat.shape == (4,)
        n_links = len(robot._entity.links)
        robot.set_friction_ratio(np.full(n_links, 1.1))
        robot.set_mass_shift(np.zeros(n_links))
        robot.set_com_shift(np.zeros((n_links, 3)))

    def test_example_main_end_to_end(self, tmp_path: Path) -> None:
        pytest.importorskip("genesis")
        rc = main_for(["--steps", "10", "--out", str(tmp_path / "out")])
        assert rc == 0
        assert (tmp_path / "out" / "episode_000.hdf5").exists()

    def test_example_main_auto_planner_falls_back(self, tmp_path: Path) -> None:
        """Auto mode degrades to OMPL when the cuRobo planner is missing."""
        pytest.importorskip("genesis")
        rc = main_for(
            ["--steps", "10", "--planner", "auto", "--out", str(tmp_path / "out")]
        )
        assert rc == 0

    def test_example_main_strict_curobo_fails_without_package(
        self, tmp_path: Path
    ) -> None:
        """--planner curobo exits with code 3 when the package is missing."""
        pytest.importorskip("genesis")
        rc = main_for(
            ["--steps", "10", "--planner", "curobo", "--out", str(tmp_path / "out")]
        )
        assert rc == 3
