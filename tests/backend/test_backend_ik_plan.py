"""Tests for the extended backend interface (RoboTwin->Genesis migration).

Covers the IK / plan_path / PD-gain / domain-randomization additions to
``ArticulationBackend`` and ``CameraBackend.get_camera_params``.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.backends.genesis_backend import (
    GenesisArticulationBackend,
    GenesisCameraBackend,
)
from cloud_robotics_sim.utils.camera import intrinsics_from_fov
from tests.conftest import MockArticulation


class TestAbcDefaults:
    """Backends without native support raise NotImplementedError."""

    def test_extended_methods_raise_not_implemented(self) -> None:
        robot = MockArticulation(n_dofs=3)
        with pytest.raises(NotImplementedError):
            robot.set_dofs_gains(np.ones(3))
        with pytest.raises(NotImplementedError):
            robot.inverse_kinematics("ee", np.zeros(3))
        with pytest.raises(NotImplementedError):
            robot.inverse_kinematics_multilink(["a", "b"], [np.zeros(3)] * 2)
        with pytest.raises(NotImplementedError):
            robot.plan_path(np.zeros(3))
        with pytest.raises(NotImplementedError):
            robot.get_link_pose("ee")
        with pytest.raises(NotImplementedError):
            robot.set_friction_ratio(np.ones(3))
        with pytest.raises(NotImplementedError):
            robot.set_mass_shift(np.ones(3))
        with pytest.raises(NotImplementedError):
            robot.set_com_shift(np.ones((3, 3)))


def _make_mocked_articulation(n_dofs: int = 2) -> GenesisArticulationBackend:
    """GenesisArticulationBackend bound to a MagicMock Genesis entity."""
    backend = GenesisArticulationBackend(morph=MagicMock(), name="arm")
    backend.bind(MagicMock())
    return backend


class TestGenesisArticulationExtended:
    """Genesis implementations delegate to the native entity APIs."""

    def test_set_dofs_gains_full(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        kp, kv = np.full(2, 50.0), np.full(2, 5.0)
        robot.set_dofs_gains(
            kp,
            kv,
            force_range=(np.full(2, -10.0), np.full(2, 10.0)),
            armature=np.full(2, 0.01),
        )
        entity.set_dofs_kp.assert_called_once()
        entity.set_dofs_kv.assert_called_once()
        entity.set_dofs_force_range.assert_called_once()
        entity.set_dofs_armature.assert_called_once()
        np.testing.assert_array_equal(entity.set_dofs_kp.call_args[0][0], kp)
        np.testing.assert_array_equal(entity.set_dofs_kv.call_args[0][0], kv)

    def test_set_dofs_gains_kp_only(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        robot.set_dofs_gains(np.ones(2))
        entity.set_dofs_kp.assert_called_once()
        entity.set_dofs_kv.assert_not_called()
        entity.set_dofs_force_range.assert_not_called()
        entity.set_dofs_armature.assert_not_called()

    def test_inverse_kinematics(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        link = MagicMock()
        entity.get_link.return_value = link
        entity.inverse_kinematics.return_value = np.array([0.1, 0.2])

        q = robot.inverse_kinematics("ee_link", pos=np.array([0.1, 0.0, 0.3]))

        entity.get_link.assert_called_once_with("ee_link")
        assert entity.inverse_kinematics.call_args[0][0] is link
        assert q.dtype == np.float64
        np.testing.assert_allclose(q, [0.1, 0.2])

    def test_inverse_kinematics_multilink(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        entity.inverse_kinematics_multilink.return_value = np.array([0.1, 0.2])

        q = robot.inverse_kinematics_multilink(
            ["left_ee", "right_ee"],
            poss=[np.array([0.3, 0.15, 0.25]), np.array([0.3, -0.15, 0.25])],
            quats=[np.array([0.0, 1.0, 0.0, 0.0])] * 2,
            rot_mask=(True, True, True),
        )

        assert entity.get_link.call_count == 2
        call = entity.inverse_kinematics_multilink.call_args
        assert call.kwargs["rot_mask"] == [True, True, True]
        assert call.kwargs["poss"].shape == (2, 3)
        assert q.dtype == np.float64

    def test_plan_path(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        entity.plan_path.return_value = np.zeros((10, 2))

        traj = robot.plan_path(np.array([0.5, 0.5]), num_waypoints=10)

        call = entity.plan_path.call_args
        assert call.kwargs["num_waypoints"] == 10
        assert traj.shape == (10, 2)
        assert traj.dtype == np.float64

    def test_get_link_pose(self) -> None:
        robot = _make_mocked_articulation()
        link = MagicMock()
        link.get_pos.return_value = np.array([1.0, 2.0, 3.0])
        link.get_quat.return_value = np.array([1.0, 0.0, 0.0, 0.0])
        robot._entity.get_link.return_value = link

        pose = robot.get_link_pose("ee_link")

        np.testing.assert_allclose(pose.pos, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(pose.quat, [1.0, 0.0, 0.0, 0.0])

    def test_domain_randomization(self) -> None:
        robot = _make_mocked_articulation()
        entity = robot._entity
        robot.set_friction_ratio(np.ones((4, 3)), envs_idx=[0, 1])
        robot.set_mass_shift(np.zeros((4, 3)))
        robot.set_com_shift(np.zeros((4, 3, 3)))
        entity.set_friction_ratio.assert_called_once()
        entity.set_mass_shift.assert_called_once()
        entity.set_COM_shift.assert_called_once()
        assert entity.set_friction_ratio.call_args.kwargs["envs_idx"] == [0, 1]


class TestIntrinsicsFromFov:
    """Pinhole intrinsic derivation (migration doc section 7)."""

    def test_matches_reference_formula(self) -> None:
        width, height, fov = 640, 480, 60.0
        k = intrinsics_from_fov(width, height, fov)
        # Genesis fov is vertical: focal length derives from image height.
        f_expected = height / (2.0 * math.tan(math.radians(fov) / 2.0))
        assert k.shape == (3, 3)
        assert k.dtype == np.float64
        assert abs(k[0, 0] - f_expected) < 1e-6
        assert abs(k[1, 1] - f_expected) < 1e-6
        assert abs(k[0, 2] - width / 2.0) < 1e-6
        assert abs(k[1, 2] - height / 2.0) < 1e-6
        assert k[2, 2] == 1.0


class TestGenesisCameraParams:
    """CameraBackend.get_camera_params for built and pre-build cameras."""

    def test_built_camera_uses_native_params(self) -> None:
        cam = MagicMock()
        cam.is_built = True
        cam.intrinsics = np.eye(3)
        cam.transform = np.eye(4)
        backend = GenesisCameraBackend("head", cam)

        intrinsic, extrinsic = backend.get_camera_params()

        np.testing.assert_allclose(intrinsic, np.eye(3))
        np.testing.assert_allclose(extrinsic, np.eye(4))

    def test_prebuild_fallback_derives_params(self) -> None:
        cam = MagicMock()
        cam.is_built = False
        cam.res = (640, 480)
        cam.fov = 60.0
        cam.pos = (0.0, -0.8, 0.6)
        cam.lookat = (0.0, 0.0, 0.2)
        backend = GenesisCameraBackend("head", cam)

        intrinsic, extrinsic = backend.get_camera_params()

        np.testing.assert_allclose(intrinsic, intrinsics_from_fov(640, 480, 60.0))
        assert extrinsic.shape == (4, 4)
        np.testing.assert_allclose(extrinsic[:3, 3], [0.0, -0.8, 0.6])
        # Genesis convention: forward = +x column points at the lookat target.
        forward = extrinsic[:3, 0]
        expected = np.array([0.0, 0.8, -0.4]) / np.linalg.norm([0.0, 0.8, -0.4])
        np.testing.assert_allclose(forward, expected, atol=1e-9)
