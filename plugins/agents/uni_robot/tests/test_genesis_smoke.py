"""真实 Genesis 冒烟测试（CPU backend，slow）：验证 builder/backend 对接
genesis-world 1.4.5 的实际 API——URDF 加载、dof 检测、IK、相机渲染与标定。

默认插件 CI（-m "not slow"）不运行；本地或完整回归时显式运行：
    pytest plugins/agents/uni_robot/tests/test_genesis_smoke.py -v -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.optional_deps import genesis_only, uni_robot_agent_only

pytestmark = [
    genesis_only,
    uni_robot_agent_only,
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _TESTS_DIR.parent / "core"
for _dir in (str(_TESTS_DIR), str(_CORE_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from uni_robot_agent.services.environment.contracts import (  # noqa: E402
    ArmCommand,
    JointCommand,
    JointTarget,
    Pose,
    PoseCommand,
)
from uni_robot_genesis.backend import GenesisEnvironmentBackend  # noqa: E402
from uni_robot_genesis.builder import build_scene  # noqa: E402
from uni_robot_genesis.scene_spec import (  # noqa: E402
    BoxSpec,
    JudgeSpec,
    PoseGoal,
)

from conftest import make_spec  # noqa: E402


def _smoke_spec():
    return make_spec(
        sim=dict(use_cuda=False),  # type: ignore[arg-type]
        ground=True,
        fixtures=(
            BoxSpec(
                name="table",
                size=(1.2, 0.8, 0.05),
                position=(0.45, 0.0, 0.35),
                color=(0.7, 0.7, 0.7, 1.0),
                fixed=True,
            ),
        ),
        objects=(
            BoxSpec(
                name="cube",
                size=(0.04, 0.04, 0.04),
                position=(0.55, -0.25, 0.395),
                color=(0.8, 0.1, 0.1, 1.0),
                fixed=False,
            ),
        ),
        judge=JudgeSpec(
            goals=(
                PoseGoal(
                    object_name="cube",
                    position=(0.55, -0.25, 0.395),
                    tolerance_m=0.15,
                ),
            )
        ),
    )


def test_genesis_franka_smoke():
    """Franka 场景端到端：构建 → start → joint/pose step → capture → finish。"""
    built = build_scene(_smoke_spec())
    try:
        arm = built.arms["arm"]
        # builder 自动检测到 7 臂 + 2 指（franka URDF 含 finger 关键字）
        assert len(arm.arm_dofs) == 7, arm.arm_dofs
        assert len(arm.gripper_dofs) == 2, arm.gripper_dofs
        assert arm.grasp_link.name in ("panda_hand", "panda_link8")

        backend = GenesisEnvironmentBackend(built)
        state = backend.start()
        assert state.state_version == 0
        assert len(state.arms["arm"].joints) == 7

        # joint step：当前位形附近的小步进（大扫掠会碰飞桌上物体，
        # 也会超出工具层允许的单步跳变；小步进是适配器的真实使用方式）
        qpos_t = arm.entity.get_qpos()
        q0 = np.asarray(
            qpos_t.detach().cpu().numpy() if hasattr(qpos_t, "detach") else qpos_t,
            dtype=np.float64,
        )[:7]
        lo = arm.joint_limits[:, 0] + 0.02
        hi = arm.joint_limits[:, 1] - 0.02
        home = tuple(np.clip(q0 + 0.05, lo, hi).tolist())
        receipt = backend.step_joints(
            JointCommand(arms={"arm": JointTarget(joints=home, gripper=0.5)})
        )
        assert receipt.accepted is True
        assert receipt.state.native_steps == 1

        # pose step：当前位姿附近的侧向小目标（远离桌面，避免碰擦）
        pose = state.arms["arm"].control_pose
        target = Pose(
            position=(
                pose.position[0] + 0.03,
                pose.position[1],
                pose.position[2],
            ),
            quaternion=tuple(pose.quaternion),
        )
        receipt2 = backend.step_pose(
            PoseCommand(arms={"arm": ArmCommand(pose=target, gripper=0.5)})
        )
        assert receipt2.accepted is True, receipt2.diagnostics
        assert receipt2.state.native_steps == 2

        # capture：真实渲染 RGB/深度 + 内参 + 光学约定外参
        observation = backend.capture()
        frame = observation.cameras["head"]
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth_m is not None
        assert np.isfinite(frame.intrinsic_K).all()
        assert np.isfinite(frame.T_world_from_optical).all()
        # 光学约定的旋转行列式：diag(1,-1,-1) 型翻转 -> det = +1
        det = float(np.linalg.det(frame.T_world_from_optical[:3, :3]))
        assert det == pytest.approx(1.0)

        outcome = backend.finish("smoke_done")
        assert outcome["task_success"] is True  # cube 未动，判据在其初始位置
    finally:
        backend.close()
