"""uni_robot 插件测试共享夹具：sys.path 注入 + Genesis fake。

fakes 只实现 backend 消费的 duck-typed 方法面，不 import genesis，
因此测试不需要 GPU、不初始化仿真、不访问网络。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from uni_robot_genesis.built import BuiltArm, BuiltScene  # noqa: E402
from uni_robot_genesis.judge import NoGoalJudge  # noqa: E402
from uni_robot_genesis.scene_spec import (  # noqa: E402
    ArmSpec,
    CameraSpec,
    SceneSpec,
)


class FakeLink:
    """grasp_link / 判据物体的最小 duck 类型。"""

    def __init__(
        self,
        name: str,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.name = name
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_name(self) -> str:
        return self.name

    def get_pos(self, *args, **kwargs) -> np.ndarray:
        del args, kwargs
        return self._pos.copy()

    def get_quat(self, *args, **kwargs) -> np.ndarray:
        del args, kwargs
        return self._quat.copy()

    def set_pose(self, pos, quat) -> None:
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)


class FakeEntity:
    """5-dof 臂（arm dofs 0..2 + gripper dofs 3..4）的最小 duck 类型。"""

    def __init__(self, n_dofs: int = 5) -> None:
        self.n_dofs = n_dofs
        self._qpos = np.zeros(n_dofs, dtype=np.float64)
        self.control_calls: list[tuple[list[int], np.ndarray]] = []
        # 测试可注入 IK 误差（6 维 [pos(3), rot(3)]）模拟不收敛
        self.ik_error = np.zeros(6, dtype=np.float64)

    def get_qpos(self) -> np.ndarray:
        return self._qpos.copy()

    def inverse_kinematics(
        self, *, link, pos, quat, dofs_idx_local, init_qpos=None, return_error=False
    ):
        """模拟 genesis：返回全量 qpos（所选 dof 用 init/当前值），error 可注入。"""
        del link, pos, quat
        dofs = [int(i) for i in dofs_idx_local]
        if init_qpos is None:
            seed = self._qpos[dofs].copy()
        else:
            seed = np.asarray(init_qpos, dtype=np.float64).reshape(-1)
        solved_full = self._qpos.copy()
        solved_full[dofs] = seed
        if return_error:
            return solved_full, self.ik_error.copy()
        return solved_full

    def control_dofs_position(self, values, dofs_idx_local=None) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        assert len(values) == len(dofs_idx_local)
        for idx, value in zip(dofs_idx_local, values):
            self._qpos[int(idx)] = value
        self.control_calls.append((list(dofs_idx_local), values.copy()))


class FakeScene:
    """gs.Scene 的最小 duck 类型：只计数物理步进。"""

    dt = 0.01

    def __init__(self) -> None:
        self.physics_steps = 0

    def step(self) -> None:
        self.physics_steps += 1


class FakeCamera:
    """固定 480p；transform 为 world-from-cam（OpenGL 约定）。"""

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        transform: np.ndarray | None = None,
    ) -> None:
        self.height, self.width = height, width
        self._transform = (
            np.eye(4, dtype=np.float64)
            if transform is None
            else np.asarray(transform, dtype=np.float64)
        )

    def render(self, rgb=True, depth=False):
        rgb_arr = (
            np.zeros((self.height, self.width, 3), dtype=np.uint8) if rgb else None
        )
        depth_arr = (
            np.ones((self.height, self.width), dtype=np.float32) if depth else None
        )
        return (rgb_arr, depth_arr)

    @property
    def intrinsics(self) -> np.ndarray:
        f = 0.5 * self.width / np.tan(np.deg2rad(30.0))
        return np.array(
            [[f, 0.0, self.width / 2], [0.0, f, self.height / 2], [0.0, 0.0, 1.0]]
        )

    @property
    def transform(self) -> np.ndarray:
        return self._transform


def make_spec(**overrides) -> SceneSpec:
    """构造一份带默认单臂单相机的 SceneSpec。"""
    kwargs = dict(
        instruction="move the cube to the target",
        arms=(ArmSpec(),),
        cameras=(CameraSpec(name="head", position=(1.0, 0.0, 1.0), lookat=(0, 0, 0)),),
    )
    kwargs.update(overrides)
    return SceneSpec(**kwargs)


def make_built(
    spec: SceneSpec | None = None,
    *,
    judge=None,
    objects: dict | None = None,
    ticks_per_control: int = 2,
) -> BuiltScene:
    """构造挂在 fake 场景上的 BuiltScene。"""
    spec = spec or make_spec()
    arm = BuiltArm(
        name="arm",
        entity=FakeEntity(),
        arm_dofs=(0, 1, 2),
        gripper_dofs=(3, 4),
        primary_gripper_dof=3,
        grasp_link=FakeLink("panda_hand", pos=(0.4, 0.0, 0.5)),
        joint_limits=np.array([(-2.9, 2.9)] * 3, dtype=np.float64),
        gripper_limits=(0.0, 0.04),
        joint_names=["j1", "j2", "j3"],
        rotation_radius_m=0.08,
        joint_tracking_tolerance_rad=0.35,
        grasp_site_description="test grasp site",
    )
    return BuiltScene(
        scene=FakeScene(),
        dt=spec.sim.dt,
        ticks_per_control=ticks_per_control,
        arms={"arm": arm},
        cameras={"head": FakeCamera()},
        camera_meta={"head": spec.cameras[0]},
        objects={} if objects is None else objects,
        judge=NoGoalJudge() if judge is None else judge,
        spec=spec,
    )


@pytest.fixture()
def built():
    """默认 BuiltScene fixture。"""
    return make_built()
