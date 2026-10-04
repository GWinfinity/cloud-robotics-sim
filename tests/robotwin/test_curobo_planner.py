"""Tests for the hierarchical cuRobo planner adapter (doc section 5.2).

The external ``curobo_hierarchical_planner`` package and CUDA/MUSA hardware
are not present in CI; planner behavior is verified with injected stubs,
and the OMPL fallback path uses a mocked Genesis entity.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.backends.genesis_backend import GenesisArticulationBackend
from cloud_robotics_sim.robotwin.curobo_planner import (
    CuRoboPlannerConfig,
    CuRoboPlannerUnavailableError,
    HierarchicalCuRoboPlanner,
    PlannerError,
    is_curobo_planner_available,
    plan_with_fallback,
)


def _make_config(tmp_path) -> CuRoboPlannerConfig:
    return CuRoboPlannerConfig(
        urdf_path=str(tmp_path / "robot.urdf"),
        base_link="base_link",
        ee_link="link2",
    )


def _make_mocked_robot(n_dofs: int = 2) -> GenesisArticulationBackend:
    robot = GenesisArticulationBackend(morph=MagicMock(), name="arm")
    entity = MagicMock()
    entity.get_qpos.return_value = np.zeros(n_dofs)
    entity.n_dofs = n_dofs
    entity.inverse_kinematics.return_value = np.full(n_dofs, 0.5)
    # Non-zero: plan_with_fallback rejects all-zeros OMPL paths (real Genesis
    # silent-failure signature).
    entity.plan_path.return_value = np.full((10, n_dofs), 0.5)
    robot.bind(entity)
    return robot


def _stub_planner(trajectory=None, success=True, message=""):
    """Inject a fake external planner into the adapter."""
    result = SimpleNamespace(
        success=success,
        trajectory=trajectory,
        joint_path=trajectory,
        timings={"total": 0.1},
        metrics={},
        message=message,
    )
    return SimpleNamespace(plan=MagicMock(return_value=result), robot_cfg=object())


class TestConfig:
    """CuRoboPlannerConfig validation."""

    def test_defaults(self, tmp_path) -> None:
        config = _make_config(tmp_path)
        assert config.workspace_bounds.shape == (3, 2)
        assert config.voxel_size == 0.04
        assert config.obstacles == []

    def test_bad_bounds_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            CuRoboPlannerConfig(
                urdf_path="a.urdf",
                base_link="b",
                ee_link="e",
                workspace_bounds=np.zeros((2, 2)),
            )

    def test_bad_voxel_size_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            CuRoboPlannerConfig(
                urdf_path="a.urdf", base_link="b", ee_link="e", voxel_size=0.0
            )


class TestAvailability:
    """Graceful behavior when the external package is missing."""

    def test_package_missing_in_this_env(self) -> None:
        # Envs WITHOUT curobo_hierarchical_planner: availability is False.
        # When it is installed (e.g. a grasping workstation), there is
        # nothing to assert here.
        if is_curobo_planner_available():
            pytest.skip("curobo_hierarchical_planner is installed in this env")
        assert is_curobo_planner_available() is False

    def test_build_raises_unavailable(self, tmp_path) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        assert planner.is_built is False
        # Message differs: package missing entirely ("pip install...") vs
        # installed but cuRobo/CUDA backend unusable ("Failed to build...").
        with pytest.raises(CuRoboPlannerUnavailableError):
            planner._build()


class TestPlanToEePose:
    """plan_to_ee_pose with an injected stub planner."""

    def test_success_returns_trajectory(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        planner._planner = _stub_planner(trajectory=np.zeros((7, 2)))
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())

        traj = planner.plan_to_ee_pose(
            np.zeros(2), np.array([0.3, 0.0, 0.2]), np.array([1.0, 0, 0, 0])
        )

        assert traj.shape == (7, 2)
        assert traj.dtype == np.float64
        planner._planner.plan.assert_called_once()

    def test_falls_back_to_joint_path(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        stub = _stub_planner(trajectory=np.zeros((5, 2)))
        stub.plan.return_value.trajectory = None  # only joint_path available
        planner._planner = stub
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())

        traj = planner.plan_to_ee_pose(np.zeros(2), np.zeros(3))
        assert traj.shape == (5, 2)

    def test_failure_raises_planner_error(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        planner._planner = _stub_planner(success=False, message="no path")
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())

        with pytest.raises(PlannerError, match="no path"):
            planner.plan_to_ee_pose(np.zeros(2), np.zeros(3))

    def test_success_without_trajectory_raises(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        stub = _stub_planner(trajectory=None)
        stub.plan.return_value.joint_path = None
        planner._planner = stub
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())

        with pytest.raises(PlannerError, match="no trajectory"):
            planner.plan_to_ee_pose(np.zeros(2), np.zeros(3))


class TestPlanWithFallback:
    """Doc 5.2 routing: cuRobo first, OMPL fallback."""

    def test_curobo_preferred_when_available(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        planner._planner = _stub_planner(trajectory=np.ones((7, 2)))
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())
        robot = _make_mocked_robot()

        traj, used = plan_with_fallback(
            robot, np.array([0.3, 0.0, 0.2]), ee_link="link2", planner=planner
        )

        assert used == "hierarchical_curobo"
        assert traj.shape == (7, 2)
        robot._entity.inverse_kinematics.assert_not_called()

    def test_falls_back_to_ompl_on_planner_error(self, tmp_path, monkeypatch) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        planner._planner = _stub_planner(success=False, message="stuck")
        monkeypatch.setattr(planner, "_make_request", lambda *a: object())
        robot = _make_mocked_robot()

        traj, used = plan_with_fallback(
            robot, np.array([0.3, 0.0, 0.2]), ee_link="link2", planner=planner
        )

        assert used == "ompl"
        assert traj.shape == (10, 2)
        robot._entity.inverse_kinematics.assert_called_once()
        robot._entity.plan_path.assert_called_once()

    def test_falls_back_to_ompl_when_unavailable(self, tmp_path) -> None:
        planner = HierarchicalCuRoboPlanner(_make_config(tmp_path))
        robot = _make_mocked_robot()

        traj, used = plan_with_fallback(
            robot, np.array([0.3, 0.0, 0.2]), ee_link="link2", planner=planner
        )

        assert used == "ompl"
        assert traj.shape == (10, 2)

    def test_ompl_used_without_planner(self) -> None:
        robot = _make_mocked_robot()
        traj, used = plan_with_fallback(robot, np.zeros(3), ee_link="link2")
        assert used == "ompl"
        assert traj.shape == (10, 2)

    def test_ompl_fallback_requires_ee_link(self) -> None:
        robot = _make_mocked_robot()
        with pytest.raises(ValueError, match="ee_link"):
            plan_with_fallback(robot, np.zeros(3), planner=None)

    def test_all_zeros_ompl_path_raises(self) -> None:
        """Genesis OMPL returns an all-zeros path on silent RRT failure."""
        robot = _make_mocked_robot()
        robot._entity.plan_path.return_value = np.zeros((10, 2))

        with pytest.raises(PlannerError, match="all-zeros"):
            plan_with_fallback(robot, np.zeros(3), ee_link="link2")


class TestLegacyGpuMathdx:
    """Pre-Ampere GPUs (sm_75) must fall back to warp native tile kernels."""

    def _patch_device(self, monkeypatch, capability) -> types.ModuleType:
        """Inject fake torch/warp.config modules and return the fake config."""
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda idx: capability,
            )
        )
        fake_warp_config = types.ModuleType("warp.config")
        fake_warp_config.enable_mathdx_gemm = True
        fake_warp_config.enable_mathdx_solver = True
        fake_warp = types.ModuleType("warp")
        fake_warp.config = fake_warp_config
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "warp", fake_warp)
        monkeypatch.setitem(sys.modules, "warp.config", fake_warp_config)
        return fake_warp_config

    def test_disabled_on_turing(self, monkeypatch) -> None:
        """sm_75 flips the gemm/solver flags off."""
        self._patch_device(monkeypatch, (7, 5))
        from cloud_robotics_sim.robotwin import curobo_planner as cp

        cp._disable_mathdx_on_legacy_gpu("cuda:0")

        config = sys.modules["warp.config"]
        assert config.enable_mathdx_gemm is False
        assert config.enable_mathdx_solver is False

    def test_unchanged_on_ampere_and_newer(self, monkeypatch) -> None:
        """sm_80+ leaves the libmathdx fast path enabled."""
        config = self._patch_device(monkeypatch, (8, 6))
        from cloud_robotics_sim.robotwin import curobo_planner as cp

        cp._disable_mathdx_on_legacy_gpu("cuda:0")

        assert config.enable_mathdx_gemm is True
        assert config.enable_mathdx_solver is True

    def test_noop_without_cuda_device(self, monkeypatch) -> None:
        """CPU devices never touch the warp flags."""
        config = self._patch_device(monkeypatch, (7, 5))
        from cloud_robotics_sim.robotwin import curobo_planner as cp

        cp._disable_mathdx_on_legacy_gpu("cpu")

        assert config.enable_mathdx_gemm is True
        assert config.enable_mathdx_solver is True
