"""Tests for ALOHA-AgileX dual-arm embodiment."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.backend import ArticulationBackend, Pose
from cloud_robotics_sim.robotwin.dual_arm_embodiment import (
    AlohaAgileX,
    AlohaAgileXConfig,
)


class FakeArticulation(ArticulationBackend):
    """Minimal articulation backend for AlohaAgileX tests."""

    def __init__(self, name: str | None = None, n_qs: int = 38) -> None:
        self._name = name
        self._n_dofs = n_qs
        self._n_qs = n_qs
        self._qpos = np.zeros(n_qs, dtype=np.float64)
        self._qvel = np.zeros(n_qs, dtype=np.float64)
        self._pos = np.zeros(3, dtype=np.float64)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.control_calls: list[tuple[np.ndarray, dict]] = []
        self.set_qpos_calls: list[np.ndarray] = []

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def n_qs(self) -> int:
        return self._n_qs

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

    def apply_force(self, force: np.ndarray, pos: np.ndarray | None = None) -> None:
        pass

    def get_qpos(self) -> np.ndarray:
        return self._qpos.copy()

    def set_qpos(
        self, qpos: np.ndarray, *, qs_idx_local: list[int] | None = None
    ) -> None:
        arr = np.asarray(qpos, dtype=np.float64).flatten()
        if qs_idx_local is not None:
            self._qpos = self._qpos.copy()
            for i, idx in enumerate(qs_idx_local):
                self._qpos[idx] = arr[i]
        else:
            self._qpos = arr
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
        dofs_idx_local: list[int] | None = None,
    ) -> None:
        self.control_calls.append(
            (np.asarray(targets, dtype=np.float64), {"dofs_idx_local": dofs_idx_local})
        )

    def control_dofs_velocity(self, targets: np.ndarray) -> None:
        pass

    def control_dofs_force(self, targets: np.ndarray) -> None:
        pass

    def get_state_batch(self, env_ids=None):
        from cloud_robotics_sim.backend.types import ArticulationState

        return ArticulationState(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            pos=self.get_pos(),
            quat=self.get_quat(),
        )

    def set_state_batch(self, state, env_ids=None):
        pass

    def control_position_batch(self, targets, stiffness=None, damping=None):
        pass

    def get_end_effector_pose(self) -> Pose:
        return Pose(pos=self.get_pos(), quat=self.get_quat())

    def get_joint_names(self) -> list[str]:
        if not hasattr(self, "_joint_names"):
            return []
        return list(self._joint_names)

    def get_joint_dofs_idx_local(self, joint_name: str) -> list[int]:
        if not hasattr(self, "_joint_dofs"):
            return []
        return list(self._joint_dofs.get(joint_name, []))

    def get_joint_qs_idx_local(self, joint_name: str) -> list[int]:
        if not hasattr(self, "_joint_qs"):
            return []
        return list(self._joint_qs.get(joint_name, []))

    def is_fixed_base(self) -> bool:
        return True


class TestAlohaAgileXConfig:
    """Tests for ALOHA-AgileX configuration."""

    def test_default_joint_names(self) -> None:
        """Default config contains expected left/right arm and gripper names."""
        cfg = AlohaAgileXConfig()
        assert cfg.left_arm_joints == [f"fl_joint{i}" for i in range(1, 7)]
        assert cfg.left_gripper_joints == ["fl_joint7", "fl_joint8"]
        assert cfg.right_arm_joints == [f"fr_joint{i}" for i in range(1, 7)]
        assert cfg.right_gripper_joints == ["fr_joint7", "fr_joint8"]


class TestAlohaAgileX:
    """Tests for ALOHA-AgileX embodiment logic."""

    @pytest.fixture
    def robot(self) -> AlohaAgileX:
        """Create an embodiment with a fake articulation entity."""
        robot = AlohaAgileX(
            AlohaAgileXConfig(
                urdf_path="assets/embodiments/aloha-agilex/urdf/robot.urdf"
            )
        )
        articulation = FakeArticulation(name="aloha_agilex", n_qs=38)

        # Provide joint metadata through the public backend interface.
        joint_names = [
            *AlohaAgileXConfig().left_arm_joints,
            *AlohaAgileXConfig().left_gripper_joints,
            *AlohaAgileXConfig().right_arm_joints,
            *AlohaAgileXConfig().right_gripper_joints,
        ]
        articulation._joint_names = joint_names
        articulation._joint_dofs = {name: [i] for i, name in enumerate(joint_names)}

        robot.entity = articulation
        robot._build_joint_mapping()
        return robot

    def test_action_space(self) -> None:
        """Action space matches RoboTwin 14-D format."""
        robot = AlohaAgileX()
        assert robot.action_dim == 14
        assert robot.obs_dim == 14 * 3
        assert robot.action_space["shape"] == (14,)

    def test_joint_mapping(self, robot: AlohaAgileX) -> None:
        """Driven joints are mapped to the correct action indices."""
        assert robot._joint_name_to_action_index["fl_joint1_0"] == 0
        assert robot._joint_name_to_action_index["fl_joint6_5"] == 5
        assert robot._joint_name_to_action_index["fl_joint7_6"] == 6
        assert robot._joint_name_to_action_index["fl_joint8_7"] == 6
        assert robot._joint_name_to_action_index["fr_joint1_8"] == 7
        assert robot._joint_name_to_action_index["fr_joint6_13"] == 12
        assert robot._joint_name_to_action_index["fr_joint7_14"] == 13
        assert robot._joint_name_to_action_index["fr_joint8_15"] == 13

    def test_apply_action_maps_14d_to_dofs(self, robot: AlohaAgileX) -> None:
        """A 14-D action is expanded to the driven DoF targets."""
        action = np.arange(14, dtype=np.float64)
        robot.apply_action(action)

        assert len(robot.entity.control_calls) == 1
        targets, kwargs = robot.entity.control_calls[0]
        dofs_idx_local = kwargs["dofs_idx_local"]

        assert len(targets) == 16
        assert dofs_idx_local == list(range(16))
        assert targets[0] == action[0]
        assert targets[6] == action[6]
        assert targets[7] == action[6]
        assert targets[14] == action[13]
        assert targets[15] == action[13]

    def test_observation_maps_dof_to_action_space(self, robot: AlohaAgileX) -> None:
        """Observation maps full qpos back to 14-D action space."""
        qpos = np.zeros(38)
        qpos[0:6] = np.arange(6)
        qpos[6] = 0.4
        qpos[7] = 0.6
        qpos[8:14] = np.arange(6) + 10
        qpos[14] = 0.5
        qpos[15] = 0.5
        robot.entity._qpos = qpos

        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["joint_position"][:6], np.arange(6))
        assert obs["joint_position"][6] == pytest.approx(0.5)
        np.testing.assert_array_equal(obs["joint_position"][7:13], np.arange(6) + 10)
        assert obs["joint_position"][13] == pytest.approx(0.5)

    def test_reset_zeros_state(self, robot: AlohaAgileX) -> None:
        """Reset zeros qpos and clears last action."""
        robot._last_action = np.ones(14)
        robot.reset()
        assert len(robot.entity.set_qpos_calls) == 1
        np.testing.assert_array_equal(robot._last_action, np.zeros(14))
