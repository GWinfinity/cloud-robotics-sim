"""Tests for embodiment config parsing and the mimic-joint control layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from cloud_robotics_sim.backends.genesis_backend import GenesisArticulationBackend
from cloud_robotics_sim.robotwin.embodiment_config import (
    MimicJoint,
    MimicJointMapper,
    RobotwinEmbodimentConfig,
)

CONFIG_YAML = """
urdf_path: robot.urdf
joint_stiffness: [50.0, 50.0, 20.0]
joint_damping: [5.0, 5.0, 2.0]
ee_joints: [left_gripper_base, right_gripper_base]
gripper_joints: [finger_left_joint]
mimic_joints:
  finger_right_joint:
    master: finger_left_joint
    multiplier: 1.0
    offset: 0.0
"""


def _write_config(tmp_path: Path, content: str = CONFIG_YAML) -> Path:
    path = tmp_path / "bot" / "config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestFromYaml:
    """config.yml parsing (doc section 2)."""

    def test_full_config(self, tmp_path: Path) -> None:
        config = RobotwinEmbodimentConfig.from_yaml(_write_config(tmp_path))
        assert config.name == "bot"  # falls back to directory name
        assert config.urdf_path == "robot.urdf"
        assert config.joint_stiffness == [50.0, 50.0, 20.0]
        assert config.joint_damping == [5.0, 5.0, 2.0]
        assert config.ee_joints == ["left_gripper_base", "right_gripper_base"]
        assert config.gripper_joints == ["finger_left_joint"]
        assert config.mimic_map["finger_right_joint"].master == "finger_left_joint"

    def test_real_robotwin_format(self, tmp_path: Path) -> None:
        """Scalar gains, move_group, gripper_name mimic (real config.yml)."""
        config = RobotwinEmbodimentConfig.from_yaml(
            _write_config(
                tmp_path,
                """
urdf_path: ./panda.urdf
joint_stiffness: 1000
joint_damping: 200
gripper_stiffnes: 1000
gripper_damping: 200
move_group: [panda_hand, panda_hand]
gripper_name:
  - base: panda_finger_joint1
    mimic: [[panda_finger_joint2, 1., 0.]]
dual_arm: false
planner: curobo
homestate: [[0, -0.5, 0, -2.0, 0, 2.0, 0.8], [0, 0, 0, 0, 0, 0, 0]]
""",
            )
        )
        assert config.joint_stiffness == [1000.0]
        assert config.joint_damping == [200.0]
        # franka-panda ships the 'gripper_stiffnes' typo - alias must catch it.
        assert config.gripper_stiffness == [1000.0]
        assert config.gripper_damping == [200.0]
        assert config.ee_links == ["panda_hand", "panda_hand"]
        assert config.gripper_joints == ["panda_finger_joint1"]
        mimic = config.mimic_map["panda_finger_joint2"]
        assert mimic.master == "panda_finger_joint1"
        assert mimic.multiplier == 1.0
        assert mimic.offset == 0.0
        assert config.dual_arm is False
        assert config.planner == "curobo"
        assert config.homestate[0][3] == -2.0

    def test_aliases(self, tmp_path: Path) -> None:
        config = RobotwinEmbodimentConfig.from_yaml(
            _write_config(
                tmp_path,
                "urdf: a.urdf\nstiffness: [1.0]\ndamping: [0.1]\nee_links: [ee]\n",
            )
        )
        assert config.urdf_path == "a.urdf"
        assert config.joint_stiffness == [1.0]
        assert config.joint_damping == [0.1]
        assert config.ee_links == ["ee"]

    def test_empty_config(self, tmp_path: Path) -> None:
        config = RobotwinEmbodimentConfig.from_yaml(_write_config(tmp_path, ""))
        assert config.joint_stiffness == []
        assert config.mimic_map == {}

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            RobotwinEmbodimentConfig.from_yaml(_write_config(tmp_path, "- 1\n- 2\n"))

    def test_mimic_map_merged_from_conversion_report(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, "urdf_path: robot.urdf\n")
        report = {
            "assets": [
                {
                    "urdf": "embodiments/bot/robot.urdf",
                    "mimic_map": {
                        "finger_right_joint": {
                            "master": "finger_left_joint",
                            "multiplier": -1.0,
                            "offset": 0.04,
                        }
                    },
                }
            ]
        }
        report_path = tmp_path / "conversion_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        config = RobotwinEmbodimentConfig.from_yaml(config_path, report_path)

        mimic = config.mimic_map["finger_right_joint"]
        assert mimic.multiplier == -1.0
        assert mimic.offset == 0.04

    def test_report_without_matching_urdf_ignored(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, "urdf_path: other.urdf\n")
        report = {
            "assets": [
                {
                    "urdf": "embodiments/bot/robot.urdf",
                    "mimic_map": {"a": {"master": "b"}},
                }
            ]
        }
        report_path = tmp_path / "conversion_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        config = RobotwinEmbodimentConfig.from_yaml(config_path, report_path)
        assert config.mimic_map == {}


class TestApplyPdGains:
    """Stiffness/damping pushed onto the backend (doc section 2)."""

    def test_apply_pd_gains(self, tmp_path: Path) -> None:
        config = RobotwinEmbodimentConfig.from_yaml(_write_config(tmp_path))
        robot = GenesisArticulationBackend(morph=MagicMock(), name="bot")
        robot.bind(MagicMock())

        config.apply_pd_gains(robot)

        robot._entity.set_dofs_kp.assert_called_once()
        robot._entity.set_dofs_kv.assert_called_once()
        np.testing.assert_array_equal(
            robot._entity.set_dofs_kp.call_args[0][0], [50.0, 50.0, 20.0]
        )

    def test_apply_pd_gains_broadcasts_scalar(self, tmp_path: Path) -> None:
        """Scalar configs (real RoboTwin) broadcast to all DoFs."""
        config = RobotwinEmbodimentConfig.from_yaml(
            _write_config(tmp_path, "joint_stiffness: 1000\njoint_damping: 200\n")
        )
        robot = GenesisArticulationBackend(morph=MagicMock(), name="bot")
        robot.bind(MagicMock())
        with patch.object(
            GenesisArticulationBackend, "n_dofs", new_callable=PropertyMock
        ) as n_dofs:
            n_dofs.return_value = 7
            config.apply_pd_gains(robot)
        np.testing.assert_array_equal(
            robot._entity.set_dofs_kp.call_args[0][0], [1000.0] * 7
        )
        np.testing.assert_array_equal(
            robot._entity.set_dofs_kv.call_args[0][0], [200.0] * 7
        )

    def test_apply_pd_gains_noop_without_stiffness(self) -> None:
        robot = GenesisArticulationBackend(morph=MagicMock(), name="bot")
        robot.bind(MagicMock())
        RobotwinEmbodimentConfig().apply_pd_gains(robot)
        robot._entity.set_dofs_kp.assert_not_called()


class TestMimicJointMapper:
    """Master->slave target replication (doc section 4.1)."""

    JOINTS = ["shoulder", "finger_left_joint", "finger_right_joint"]

    def test_expand_replicates_master(self) -> None:
        mapper = MimicJointMapper(
            {"finger_right_joint": MimicJoint("finger_left_joint", -1.0, 0.04)},
            self.JOINTS,
        )
        out = mapper.expand(np.array([0.5, 0.02, 0.0]))
        np.testing.assert_allclose(out, [0.5, 0.02, 0.02])
        assert mapper.n_slaves == 1

    def test_expand_batched(self) -> None:
        mapper = MimicJointMapper(
            {"finger_right_joint": MimicJoint("finger_left_joint", 1.0, 0.0)},
            self.JOINTS,
        )
        targets = np.array([[0.0, 0.01, 0.0], [0.0, 0.03, 0.0]])
        out = mapper.expand(targets)
        np.testing.assert_allclose(out[:, 2], [0.01, 0.03])
        # Input array must not be mutated.
        np.testing.assert_allclose(targets[:, 2], [0.0, 0.0])

    def test_empty_map_is_identity(self) -> None:
        mapper = MimicJointMapper({}, self.JOINTS)
        targets = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(mapper.expand(targets), targets)
        assert mapper.n_slaves == 0

    def test_unknown_joints_rejected(self) -> None:
        with pytest.raises(KeyError):
            MimicJointMapper({"nope": MimicJoint("shoulder")}, self.JOINTS)
        with pytest.raises(KeyError):
            MimicJointMapper({"shoulder": MimicJoint("nope")}, self.JOINTS)

    def test_apply_issues_position_control(self) -> None:
        robot = GenesisArticulationBackend(morph=MagicMock(), name="bot")
        robot.bind(MagicMock())
        mapper = MimicJointMapper(
            {"finger_right_joint": MimicJoint("finger_left_joint", -1.0, 0.04)},
            self.JOINTS,
        )
        mapper.apply(robot, np.array([0.0, 0.02, 0.0]))
        call = robot._entity.control_dofs_position.call_args[0][0]
        np.testing.assert_allclose(np.asarray(call), [0.0, 0.02, 0.02])


REAL_EMBODIMENTS = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robotwin"
    / "embodiments"
    / "embodiments"
)


@pytest.mark.skipif(
    not REAL_EMBODIMENTS.is_dir(), reason="RoboTwin embodiment assets not downloaded"
)
class TestRealEmbodimentConfigs:
    """Parse the actual RoboTwin 2.0 embodiment configs (downloaded assets)."""

    EXPECTED = {
        "aloha-agilex": {
            "dual_arm": True,
            "ee_links": ["fl_link6", "fr_link6"],
            "mimic": {"fl_joint8": "fl_joint7", "fr_joint8": "fr_joint7"},
        },
        "ARX-X5": {"dual_arm": False, "mimic": {"joint8": "joint7"}},
        "franka-panda": {
            "dual_arm": False,
            "mimic": {"panda_finger_joint2": "panda_finger_joint1"},
        },
        "piper": {"dual_arm": False, "mimic": {"joint8": "joint7"}},
        "ur5-wsg": {
            "dual_arm": False,
            "mimic": {"base_joint_gripper_right": "base_joint_gripper_left"},
        },
    }

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_real_config_parses(self, name: str) -> None:
        config = RobotwinEmbodimentConfig.from_yaml(
            REAL_EMBODIMENTS / name / "config.yml"
        )
        expected = self.EXPECTED[name]
        assert config.name == name
        assert config.dual_arm == expected["dual_arm"]
        assert config.joint_stiffness == [1000.0]
        assert config.joint_damping == [200.0]
        assert config.gripper_stiffness == [1000.0]  # incl. franka typo alias
        assert config.planner == "curobo"
        assert len(config.homestate) == 2
        assert {s: m.master for s, m in config.mimic_map.items()} == expected["mimic"]
        if "ee_links" in expected:
            assert config.ee_links == expected["ee_links"]

    def test_urdf_paths_resolve(self) -> None:
        """Every config's urdf_path must resolve to an existing file."""
        for yml in sorted(REAL_EMBODIMENTS.glob("*/config.yml")):
            config = RobotwinEmbodimentConfig.from_yaml(yml)
            urdf = (yml.parent / config.urdf_path).resolve()
            assert urdf.is_file(), f"{config.name}: missing {urdf}"
