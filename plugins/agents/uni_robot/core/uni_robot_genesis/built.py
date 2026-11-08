"""构建产物数据结构：backend 与 builder 之间的交接。

纯数据 + duck-typed 句柄，不 import genesis——backend 只依赖这里约定的
方法面（get_qpos / inverse_kinematics / control_dofs_position /
render / transform / intrinsics / step），因此测试可以用 fake 注入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .judge import GoalJudge
from .scene_spec import CameraSpec, SceneSpec


@dataclass
class BuiltArm:
    """一个臂在 Genesis 里的所有原生句柄和标定常量。"""

    name: str
    entity: Any  # gs.RigidEntity
    arm_dofs: tuple[int, ...]  # 本地 dof 索引（臂关节）
    gripper_dofs: tuple[int, ...]  # 本地 dof 索引（夹爪）
    primary_gripper_dof: int  # arm-local dof 索引，开度映射的主 dof
    grasp_link: Any  # 控制点所在的 gs.RigidLink
    joint_limits: np.ndarray  # shape (n_arm_dofs, 2)，软限位 [lower, upper]
    gripper_limits: tuple[float, float]  # 主夹爪 dof 的 (lower, upper)
    joint_names: list[str]
    rotation_radius_m: float
    joint_tracking_tolerance_rad: float | None
    grasp_site_description: str


@dataclass
class BuiltScene:
    """builder.build_scene 的返回：backend 需要的全部运行时状态。"""

    scene: Any  # gs.Scene
    dt: float
    ticks_per_control: int
    arms: dict[str, BuiltArm]
    cameras: dict[str, Any]  # name -> gs.Camera
    camera_meta: dict[str, CameraSpec]
    objects: dict[str, Any]  # name -> 可动物体 entity（供 judge 读取位置）
    judge: GoalJudge
    spec: SceneSpec
