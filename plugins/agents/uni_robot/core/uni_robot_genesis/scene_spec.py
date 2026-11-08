"""通用 Genesis 场景描述（scene spec）。

SceneSpec 是适配器的唯一输入：任意机器人、任意相机布置、任意桌面/物体
配置和一段自由文本任务指令。它不做任何测评假设——成功判据（JudgeSpec）
是可选的；缺省表示"通用执行/数据采集"模式，此时 episode 的 task_success
始终为 None，由调用方自行评估结果。

与 uni-robot-agent 的协议约定保持一致：世界坐标单位为米，四元数为 wxyz。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]  # wxyz


def _normalized_quat(value: Quaternion) -> Quaternion:
    arr = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("quaternion cannot be zero")
    return tuple(float(x / norm) for x in arr)  # type: ignore[return-value]


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimConfig(_SpecModel):
    """Genesis 仿真参数。"""

    dt: float = 0.01
    substeps: int = Field(default=2, ge=1)
    use_cuda: bool | None = None  # None -> 交给 genesis_compat 自动选择
    gravity: Vector3 = (0.0, 0.0, -9.81)


class ControlConfig(_SpecModel):
    """控制与安全约束（单步 primitive 的边界，工具层负责多步规划）。"""

    frequency_hz: float = Field(default=20.0, gt=0)
    max_episode_steps: int = Field(default=600, gt=0)
    # validate_commands(mode="eef") 允许的单步位姿增量上限
    max_step_translation_m: float = Field(default=0.05, gt=0)
    max_step_rotation_rad: float = Field(default=0.35, gt=0)
    # 相邻 joint 命令之间的跳变保护
    max_joint_jump_rad: float = Field(default=0.25, gt=0)
    # IK 收敛容差（Genesis inverse_kinematics 内部为归一化误差）
    ik_pos_tolerance_m: float = Field(default=5e-3, gt=0)
    ik_rot_tolerance_rad: float = Field(default=5e-2, gt=0)
    # 每个控制周期执行的物理 tick 数；None = 按 dt/frequency 自动推算
    ticks_per_control: int | None = Field(default=None, ge=1)


class GripperSpec(_SpecModel):
    """平行夹爪。开度归一化为 [0,1]（0=闭合，1=张开）。"""

    # 夹爪关节的本地 dof 索引；None = 按名称启发式自动检测（含 finger/gripper 的 dof）
    dofs: tuple[int, ...] | None = None
    # 主驱动 dof（开度由此 dof 的 joint limit 线性映射）；None = dofs[0]
    primary_dof: int | None = None
    # 关节限位覆盖（弧度/米，按 URDF 实际单位）；None = 从实体读取
    joint_limits: tuple[float, float] | None = None


class ArmSpec(_SpecModel):
    """单臂配置。默认按 franka_panda 预设解析 URDF。"""

    name: str = "arm"
    preset: str = "franka_panda"
    urdf_path: str | None = None  # 覆盖 preset
    base_position: Vector3 = (0.0, 0.0, 0.0)
    base_quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    # 臂关节的本地 dof 索引；None = 除夹爪外的全部 dof
    arm_dofs: tuple[int, ...] | None = None
    # 控制点（grasp site）所在 link；None = 按候选名自动选择
    eef_link: str | None = None
    # 驱动增益覆盖（Nm/rad、Nms/rad），按 dof 顺序；None = 使用 URDF/引擎默认。
    # Genesis 对 URDF 的默认增益（kp=100/kv=40）偏软：大目标跟踪慢但稳定；
    # 工具层以小步进下发，通常够用。需要更快响应时按机器人显式配置。
    kp: tuple[float, ...] | None = None
    kv: tuple[float, ...] | None = None
    # 给模型看的控制点文字说明（缺省为平行夹爪描述）
    grasp_site_description: str = (
        "Center of the parallel gripper pads; the gripper value is the normalized "
        "opening (0 = closed, 1 = fully open)."
    )
    gripper: GripperSpec = Field(default_factory=GripperSpec)
    # 旋转航点预算用的保守半径（工具层防大旋转甩动）
    rotation_radius_m: float = Field(default=0.08, ge=0)
    joint_tracking_tolerance_rad: float | None = 0.35

    @model_validator(mode="after")
    def _norm_quat(self) -> "ArmSpec":
        object.__setattr__(
            self, "base_quaternion", _normalized_quat(self.base_quaternion)
        )
        return self


class CameraSpec(_SpecModel):
    """固定位姿的场景相机（pos + lookat 定义外参）。"""

    name: str = "head"
    position: Vector3
    lookat: Vector3
    resolution: tuple[int, int] = (640, 480)  # (width, height)
    fov: float = Field(default=60.0, gt=0, lt=180)
    role: str = "scene"
    provide_depth: bool = True


class BoxSpec(_SpecModel):
    """静态几何（地面/桌面/障碍）或可抓取物体。"""

    name: str
    size: Vector3
    position: Vector3
    quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    fixed: bool = False  # True = 固定（桌面/障碍），False = 可动物体

    @model_validator(mode="after")
    def _norm_quat(self) -> "BoxSpec":
        object.__setattr__(self, "quaternion", _normalized_quat(self.quaternion))
        return self


class PoseGoal(_SpecModel):
    """声明式成功判据：物体当前位置到达目标点容差内。"""

    object_name: str
    position: Vector3
    tolerance_m: float = Field(default=0.03, gt=0)


class JudgeSpec(_SpecModel):
    """可选成功判据。goals 为空表示通用模式（task_success 恒为 None）。"""

    goals: tuple[PoseGoal, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.goals)


class SceneSpec(_SpecModel):
    """一个 episode 的完整通用描述。"""

    instruction: str = Field(min_length=1)  # 自由文本任务指令
    episode_id: str = "genesis-episode"
    sim: SimConfig = Field(default_factory=SimConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    arms: tuple[ArmSpec, ...] = Field(min_length=1)
    cameras: tuple[CameraSpec, ...] = Field(min_length=1)
    ground: bool = True
    fixtures: tuple[BoxSpec, ...] = ()  # 固定几何：桌面、围栏等
    objects: tuple[BoxSpec, ...] = ()  # 可动物体（成功的判据对象）
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_names(self) -> "SceneSpec":
        arm_names = [a.name for a in self.arms]
        cam_names = [c.name for c in self.cameras]
        object_names = [o.name for o in self.objects]
        fixture_names = [f.name for f in self.fixtures]
        for names, kind in (
            (arm_names, "arm"),
            (cam_names, "camera"),
            (object_names + fixture_names, "entity"),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {kind} names in scene spec")
        return self

    # -- 序列化辅助 ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "SceneSpec":
        """从 YAML 文件加载（examples/ 下有示例）。"""
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Scene spec {path} must contain a mapping")
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        import yaml

        payload = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
