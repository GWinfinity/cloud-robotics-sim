"""Tests for robot embodiment definitions."""

from unittest.mock import MagicMock

import numpy as np

from cloud_robotics_sim import (
    EmbodimentConfig,
    FrankaPanda,
    SensorConfig,
    UniversalRobotUR5,
)
from cloud_robotics_sim.core.embodiment import MobileManipulator
from tests.conftest import MockArticulation, make_mock_scene_backend


class MockEntity:
    """Minimal entity that is NOT an EntityBackend, for fallback tests."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str | None:
        return "mock_entity"


def _make_mock_scene() -> MagicMock:
    """Create a mock scene backend with a mock simulator backend."""
    return make_mock_scene_backend()


class TestSensorConfig:
    """Tests for SensorConfig dataclass."""

    def test_default_values(self):
        """Test default sensor config."""
        config = SensorConfig()

        assert config.camera_names == ["head_cam"]
        assert config.use_proprioception is True
        assert config.use_imu is False
        assert config.camera_positions == {"head_cam": (0.1, 0.0, 0.05)}
        assert config.camera_resolutions == {"head_cam": (640, 480)}

    def test_custom_values(self):
        """Test custom sensor config."""
        config = SensorConfig(
            camera_names=["cam1", "cam2"],
            use_proprioception=False,
            use_imu=True,
        )
        assert config.camera_names == ["cam1", "cam2"]
        assert config.use_proprioception is False
        assert config.use_imu is True


class TestEmbodimentConfig:
    """Tests for EmbodimentConfig dataclass."""

    def test_default_values(self):
        """Test default embodiment config."""
        config = EmbodimentConfig()

        assert config.name == "unnamed_robot"
        assert config.urdf_path is None
        assert config.base_position == (0.0, 0.0, 0.0)
        assert config.base_orientation == (1.0, 0.0, 0.0, 0.0)
        assert config.joint_stiffness == 100.0
        assert config.joint_damping == 10.0
        assert config.action_scale == 1.0
        assert isinstance(config.sensor_config, SensorConfig)

    def test_custom_values(self):
        """Test custom embodiment config."""
        config = EmbodimentConfig(
            name="franka_01",
            urdf_path="/path/to/robot.urdf",
            base_position=(1.0, 0.0, 0.0),
            joint_stiffness=50.0,
            action_scale=0.5,
        )

        assert config.name == "franka_01"
        assert config.urdf_path == "/path/to/robot.urdf"
        assert config.base_position == (1.0, 0.0, 0.0)
        assert config.joint_stiffness == 50.0
        assert config.action_scale == 0.5


class TestRobotEmbodimentBase:
    """Tests for RobotEmbodiment base class properties."""

    def test_default_properties(self):
        """Test default dimensions and action space."""
        robot = FrankaPanda()
        assert robot.obs_dim == 23
        assert robot.action_dim == 8

        action_space = robot.action_space
        assert action_space["low"] == -1.0
        assert action_space["high"] == 1.0
        assert action_space["shape"] == (8,)
        assert action_space["dtype"] == "float32"

    def test_action_space_updates_with_dim(self):
        """Test action space shape follows action_dim."""
        robot = UniversalRobotUR5()
        assert robot.action_space["shape"] == (6,)


class TestFrankaPanda:
    """Tests for FrankaPanda embodiment."""

    def test_default_dimensions(self):
        """Test default Franka dimensions."""
        robot = FrankaPanda()

        assert robot.action_dim == 8  # 7 joints + gripper
        assert robot.obs_dim == 23  # 7*3 + 2

    def test_action_space(self):
        """Test action space specification."""
        robot = FrankaPanda()
        action_space = robot.action_space

        assert action_space["low"] == -1.0
        assert action_space["high"] == 1.0
        assert action_space["shape"] == (8,)

    def test_spawn_mjcf_success(self):
        """Test Franka spawn with MJCF success path."""
        scene = _make_mock_scene()
        articulation = MockArticulation()
        scene.backend.load_mjcf.return_value = articulation

        robot = FrankaPanda(EmbodimentConfig(base_position=(1.0, 0.0, 0.0)))
        result = robot.spawn(scene, position=(0.5, 0.0, 0.1))

        assert result is robot
        assert robot.scene is scene
        assert robot.entity is articulation
        scene.backend.load_mjcf.assert_called_once_with(
            file="franka_emika_panda/panda.xml",
            pos=(0.5, 0.0, 0.1),
        )
        scene.add_articulation.assert_called_once_with(articulation)

    def test_spawn_mjcf_fallback(self):
        """Test Franka fallback to procedural box when MJCF fails."""
        scene = _make_mock_scene()
        scene.backend.load_mjcf.side_effect = RuntimeError("load failed")
        fallback_entity = MockEntity()
        scene.backend.create_box.return_value = fallback_entity

        robot = FrankaPanda()
        robot.spawn(scene)

        assert scene.backend.create_box.called
        assert scene.add_entity.call_count == 1

    def test_reset_with_dofs(self):
        """Test reset sets qpos when DOFs exist."""
        robot = FrankaPanda()
        robot.entity = MockArticulation(n_dofs=9, n_qs=9)

        robot.reset()
        assert len(robot.entity.set_qpos_calls) == 1
        np.testing.assert_array_equal(robot.entity.set_qpos_calls[0], np.zeros(9))

    def test_reset_no_dofs(self):
        """Test reset skips qpos when no DOFs."""
        robot = FrankaPanda()
        robot.entity = MockArticulation(n_dofs=0, n_qs=0)

        robot.reset()
        assert len(robot.entity.set_qpos_calls) == 0

    def test_apply_action_9_dofs(self):
        """Test apply_action expands gripper for 9-DOF Franka."""
        robot = FrankaPanda(EmbodimentConfig(action_scale=1.0))
        robot.entity = MockArticulation(n_dofs=9, n_qs=9)

        action = np.arange(8)
        robot.apply_action(action)

        assert len(robot.entity.control_dofs_position_calls) == 1
        targets = robot.entity.control_dofs_position_calls[0]
        assert len(targets) == 9
        np.testing.assert_array_equal(targets[:7], action[:7])
        assert targets[7] == action[7]
        assert targets[8] == action[7]

    def test_apply_action_8_dofs(self):
        """Test apply_action with 8-DOF Franka."""
        robot = FrankaPanda(EmbodimentConfig(action_scale=1.0))
        robot.entity = MockArticulation(n_dofs=8, n_qs=8)

        action = np.arange(8)
        robot.apply_action(action)

        targets = robot.entity.control_dofs_position_calls[0]
        assert len(targets) == 8
        np.testing.assert_array_equal(targets[:7], action[:7])
        assert targets[7] == action[7]

    def test_apply_action_zero_dofs(self):
        """Test apply_action skips when no DOFs."""
        robot = FrankaPanda()
        robot.entity = MockArticulation(n_dofs=0, n_qs=0)

        robot.apply_action(np.zeros(8))
        assert len(robot.entity.control_dofs_position_calls) == 0

    def test_apply_action_no_entity(self):
        """Test apply_action when entity is None."""
        robot = FrankaPanda()
        robot.apply_action(np.zeros(8))  # should not raise

    def test_get_observation_with_entity(self):
        """Test get_observation reads entity state."""
        robot = FrankaPanda()
        robot.entity = MockArticulation(n_dofs=9, n_qs=9)
        robot.entity._qpos = np.ones(9)
        robot.entity._qvel = np.ones(9) * 2

        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["joint_position"], np.ones(7))
        np.testing.assert_array_equal(obs["joint_velocity"], np.ones(7) * 2)
        np.testing.assert_array_equal(obs["gripper_width"], np.array([0.04]))

    def test_get_observation_no_entity(self):
        """Test get_observation returns defaults without entity."""
        robot = FrankaPanda()
        obs = robot.get_observation()

        np.testing.assert_array_equal(obs["joint_position"], np.zeros(7))
        np.testing.assert_array_equal(obs["joint_velocity"], np.zeros(7))


class TestUniversalRobotUR5:
    """Tests for UniversalRobotUR5 embodiment."""

    def test_default_dimensions(self):
        """Test default UR5 dimensions."""
        robot = UniversalRobotUR5()

        assert robot.action_dim == 6
        assert robot.obs_dim == 18  # 6*3

    def test_spawn_urdf_success(self):
        """Test UR5 spawn with URDF success path."""
        scene = _make_mock_scene()
        articulation = MockArticulation(n_dofs=6, n_qs=6)
        scene.backend.load_urdf.return_value = articulation

        robot = UniversalRobotUR5()
        robot.spawn(scene, position=(2.0, 0.0, 0.0))

        scene.backend.load_urdf.assert_called_once_with(
            file="ur5/ur5.urdf",
            pos=(2.0, 0.0, 0.0),
        )
        scene.add_articulation.assert_called_once_with(articulation)

    def test_spawn_urdf_fallback(self):
        """Test UR5 fallback when URDF fails."""
        scene = _make_mock_scene()
        scene.backend.load_urdf.side_effect = RuntimeError("load failed")
        fallback_entity = MockEntity()
        scene.backend.create_box.return_value = fallback_entity

        robot = UniversalRobotUR5()
        robot.spawn(scene)

        assert scene.backend.create_box.called
        assert scene.add_entity.call_count == 1

    def test_reset_with_dofs(self):
        """Test UR5 reset with DOFs."""
        robot = UniversalRobotUR5()
        robot.entity = MockArticulation(n_dofs=6, n_qs=6)

        robot.reset()
        assert len(robot.entity.set_qpos_calls) == 1
        np.testing.assert_array_equal(robot.entity.set_qpos_calls[0], np.zeros(6))

    def test_apply_action(self):
        """Test UR5 apply_action."""
        robot = UniversalRobotUR5(EmbodimentConfig(action_scale=0.5))
        robot.entity = MockArticulation(n_dofs=6, n_qs=6)

        action = np.ones(6)
        robot.apply_action(action)

        targets = robot.entity.control_dofs_position_calls[0]
        np.testing.assert_array_equal(targets, np.ones(6) * 0.5)

    def test_get_observation(self):
        """Test UR5 get_observation."""
        robot = UniversalRobotUR5()
        robot.entity = MockArticulation(n_dofs=6, n_qs=6)
        robot.entity._qpos = np.arange(6)
        robot.entity._qvel = np.arange(6) * 2

        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["joint_position"], np.arange(6))
        np.testing.assert_array_equal(obs["joint_velocity"], np.arange(6) * 2)


class TestMobileManipulator:
    """Tests for MobileManipulator embodiment."""

    def test_default_dimensions(self):
        """Test default mobile manipulator dimensions."""
        robot = MobileManipulator()

        assert robot.action_dim == 10  # 2 base + 8 arm/gripper
        assert robot.obs_dim == 30
        assert robot.base_type == "diff_drive"
        assert robot.arm_type == "panda"

    def test_custom_types(self):
        """Test custom base and arm types."""
        robot = MobileManipulator(base_type="omni", arm_type="ur5")
        assert robot.base_type == "omni"
        assert robot.arm_type == "ur5"

    def test_spawn(self):
        """Test mobile manipulator spawn."""
        scene = _make_mock_scene()
        entity = MockEntity()
        scene.backend.create_box.return_value = entity

        robot = MobileManipulator()
        robot.spawn(scene, position=(0.0, 0.0, 0.1))

        assert robot.entity is entity
        scene.backend.create_box.assert_called_once_with(
            size=(0.6, 0.4, 0.2),
            pos=(0.0, 0.0, 0.1),
            name="mobile_base",
        )

    def test_reset_apply_action_get_observation(self):
        """Test mobile manipulator placeholder methods."""
        robot = MobileManipulator()
        robot.reset()  # no-op
        robot.apply_action(np.zeros(10))  # no-op

        obs = robot.get_observation()
        assert "base_position" in obs
        assert "base_orientation" in obs
        assert "joint_position" in obs
