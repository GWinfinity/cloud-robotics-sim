"""GenesisEnvironmentBackend 协议行为测试（fake 场景，无 genesis / 无网络）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.optional_deps import uni_robot_agent_only

pytestmark = uni_robot_agent_only

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _TESTS_DIR.parent / "core"
for _dir in (str(_TESTS_DIR), str(_CORE_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from uni_robot_agent.services.environment.contracts import (  # noqa: E402
    ArmCommand,
    JointCommand,
    JointTarget,
    NativeCommand,
    Pose,
    PoseCommand,
    RequestRejected,
)
from uni_robot_genesis.backend import GenesisEnvironmentBackend  # noqa: E402
from uni_robot_genesis.judge import build_judge  # noqa: E402
from uni_robot_genesis.scene_spec import (  # noqa: E402
    ControlConfig,
    JudgeSpec,
    PoseGoal,
)

from conftest import FakeLink, make_built, make_spec  # noqa: E402


def _pose(x=0.4, y=0.0, z=0.5):
    return Pose(position=(x, y, z), quaternion=(1.0, 0.0, 0.0, 0.0))


def _joint_command(joints=(0.0, 0.0, 0.0), gripper=0.5):
    return JointCommand(arms={"arm": JointTarget(joints=joints, gripper=gripper)})


class TestLifecycle:
    """生命周期：start/read_state/description。"""

    def test_read_state_before_start_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        with pytest.raises(RequestRejected):
            backend.read_state()

    def test_start_returns_initial_state(self, built):
        backend = GenesisEnvironmentBackend(built)
        state = backend.start()
        assert state.state_version == 0
        assert state.native_steps == 0
        arm = state.arms["arm"]
        assert len(arm.joints) == 3
        assert 0.0 <= arm.gripper <= 1.0
        assert np.allclose(arm.control_pose.position, (0.4, 0.0, 0.5))
        assert "instruction" in state.metadata
        assert "task_success" not in state.metadata  # 协议禁止 privileged 字段

    def test_description_matches_spec(self, built):
        backend = GenesisEnvironmentBackend(built)
        desc = backend.description
        assert desc.arms[0].name == "arm"
        assert desc.cameras[0].name == "head"
        assert desc.cameras[0].has_calibration is True
        assert desc.control_frequency_hz == 20.0
        assert desc.metadata["instruction"] == built.spec.instruction


class TestStepJoints:
    """joint 命令路径：执行、限位、臂集合校验。"""

    def test_step_applies_targets_and_advances_version(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        receipt = backend.step_joints(_joint_command(gripper=1.0))
        assert receipt.accepted is True
        assert receipt.state.state_version == 1
        arm = built.arms["arm"]
        # control_dofs_position 收到 臂(3)+夹爪(2) 共 5 个 dof
        dofs, values = arm.entity.control_calls[-1]
        assert dofs == [0, 1, 2, 3, 4]
        # 开度 1.0 -> 主夹爪 dof = 上限 0.04
        assert values[3] == pytest.approx(0.04)
        assert values[4] == pytest.approx(0.04)
        # 物理步进 ticks_per_control 次
        assert built.scene.physics_steps == 2
        assert receipt.state.arms["arm"].gripper_command == pytest.approx(1.0)

    def test_joint_target_out_of_limits_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        with pytest.raises(RequestRejected, match="limits"):
            backend.step_joints(_joint_command(joints=(10.0, 0.0, 0.0)))

    def test_wrong_arm_set_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        command = JointCommand(
            arms={"other": JointTarget(joints=(0.0, 0.0, 0.0), gripper=0.0)}
        )
        with pytest.raises(RequestRejected, match="all arms"):
            backend.step_joints(command)


class TestStepPose:
    """pose 命令路径：IK 收敛/拒绝与无副作用。"""

    def test_ik_path_accepted(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        command = PoseCommand(arms={"arm": ArmCommand(pose=_pose(), gripper=0.0)})
        receipt = backend.step_pose(command)
        assert receipt.accepted is True
        assert receipt.state.state_version == 1
        assert "arm_ik" in receipt.diagnostics

    def test_ik_not_converged_rejected_without_side_effects(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        built.arms["arm"].entity.ik_error = np.array([1.0, 0, 0, 0, 0, 0])
        command = PoseCommand(arms={"arm": ArmCommand(pose=_pose(x=5.0), gripper=0.0)})
        receipt = backend.step_pose(command)
        assert receipt.accepted is False
        assert receipt.diagnostics["reason"] == "ik_not_converged"
        assert receipt.state.state_version == 0
        assert built.scene.physics_steps == 0

    def test_native_command_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        with pytest.raises(RequestRejected, match="native"):
            backend.step_native(NativeCommand(values=[0.0]))


class TestValidateCommands:
    """命令前缀校验：eef 跳变、joint 跳变、native 拒绝。"""

    def test_eef_jump_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        far = PoseCommand(arms={"arm": ArmCommand(pose=_pose(x=0.9), gripper=0.0)})
        with pytest.raises(RequestRejected, match="single-step bounds"):
            backend.validate_commands([far], mode="eef")

    def test_joint_jump_between_prefix_commands_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        c1 = _joint_command(joints=(0.0, 0.0, 0.0))
        c2 = _joint_command(joints=(0.0, 1.0, 0.0))  # 跳变 1.0 rad > 0.25
        with pytest.raises(RequestRejected, match="joint jump"):
            backend.validate_commands([c1, c2], mode="joints")

    def test_native_in_validation_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        with pytest.raises(RequestRejected, match="native"):
            backend.validate_commands([NativeCommand(values=[0.0])], mode="eef")


class TestPreview:
    """preview_joints 只出诊断、不改动物理状态。"""

    def test_preview_reports_limit_diagnostics_without_stepping(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        good = _joint_command(joints=(0.1, -0.1, 0.0))
        bad = _joint_command(joints=(9.9, 0.0, 0.0))
        result = backend.preview_joints([good, bad])
        previews = result["previews"]
        assert previews[0]["arms"]["arm"]["within_limits"] is True
        assert previews[1]["arms"]["arm"]["within_limits"] is False
        assert built.scene.physics_steps == 0
        assert backend.read_state().state_version == 0


class TestCapture:
    """capture 的图像/标定/坐标转换与相机校验。"""

    def test_capture_frames_and_calibration(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        observation = backend.capture()
        assert observation.episode_id == built.spec.episode_id
        frame = observation.cameras["head"]
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth_m.shape == (480, 640)
        assert frame.depth_m.dtype == np.float32
        assert frame.intrinsic_K.shape == (3, 3)
        # 恒等 transform（平移为零）：world-from-optical 旋转 = diag(1,-1,-1)
        assert np.allclose(frame.T_world_from_optical[:3, 0], (1, 0, 0))
        assert np.allclose(frame.T_world_from_optical[:3, 1], (0, -1, 0))
        assert np.allclose(frame.T_world_from_optical[:3, 2], (0, 0, -1))
        assert frame.calibration is not None

    def test_capture_extrinsic_translation_preserved(self):
        transform = np.eye(4)
        transform[:3, 3] = (1.0, 2.0, 3.0)
        from conftest import FakeCamera

        spec = make_spec()
        built = make_built(spec)
        built.cameras["head"] = FakeCamera(transform=transform)
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        frame = backend.capture().cameras["head"]
        assert np.allclose(frame.T_world_from_optical[:3, 3], (1.0, 2.0, 3.0))

    def test_unknown_camera_rejected(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        with pytest.raises(RequestRejected, match="cameras"):
            backend.capture(cameras=["nope"])

    def test_depth_disabled_camera_returns_rgb_only(self):
        spec = make_spec()
        no_depth_cam = spec.cameras[0].__class__(
            name="head",
            position=spec.cameras[0].position,
            lookat=spec.cameras[0].lookat,
            provide_depth=False,
        )
        spec = make_spec(cameras=(no_depth_cam,))
        built = make_built(spec)
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        frame = backend.capture().cameras["head"]
        assert frame.depth_m is None


class TestFinish:
    """finish 与 judge：通用模式恒 None，位姿判据生效。"""

    def test_finish_without_judge_keeps_success_none(self, built):
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        backend.step_joints(_joint_command())
        outcome = backend.finish("session_finished")
        assert outcome["task_success"] is None
        assert outcome["steps"] == 1
        # 重复 finish 返回同一结果（幂等）
        assert backend.finish("again") == outcome

    def test_finish_with_pose_goal_judge(self):
        cube = FakeLink("cube", pos=(0.55, 0.15, 0.40))
        judge = build_judge(
            JudgeSpec(
                goals=(
                    PoseGoal(
                        object_name="cube",
                        position=(0.55, 0.15, 0.40),
                        tolerance_m=0.05,
                    ),
                )
            )
        )
        built = make_built(objects={"cube": cube}, judge=judge)
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        assert backend.finish("done")["task_success"] is True

        cube.set_pose((1.0, 1.0, 1.0), (1, 0, 0, 0))
        built2 = make_built(objects={"cube": cube}, judge=judge)
        backend2 = GenesisEnvironmentBackend(built2)
        backend2.start()
        assert backend2.finish("done")["task_success"] is False

    def test_judge_missing_object_is_undetermined(self):
        judge = build_judge(
            JudgeSpec(goals=(PoseGoal(object_name="ghost", position=(0, 0, 0)),))
        )
        built = make_built(judge=judge, objects={})
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        assert backend.finish("done")["task_success"] is None


class TestStepLimit:
    """步数上限：超限拒绝且截断标记置位。"""

    def test_step_beyond_max_episode_steps_rejected(self):
        spec = make_spec(control=ControlConfig(max_episode_steps=1))
        built = make_built(spec)
        backend = GenesisEnvironmentBackend(built)
        backend.start()
        backend.step_joints(_joint_command())
        assert backend.read_state().metadata["remaining_steps"] == 0
        with pytest.raises(RequestRejected, match="step limit"):
            backend.step_joints(_joint_command())
        truncated = backend.capture().truncated
        assert truncated is True
