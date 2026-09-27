"""关节资产标注的数据模型。

一次"自动标注"的产物是 :class:`ArticulationResult`：从一组已分割的
3D 部件（:class:`ArticulatedPart`）估计出的关节列表
（:class:`JointSpec`，prismatic/revolute + 轴向 + 限位 + 置信度），
以及需要人工复核的低置信度部件队列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class JointType(str, Enum):
    """支持的关节类型（与 URDF joint type 对齐）。"""

    PRISMATIC = "prismatic"  # 抽屉滑轨 / 推拉门
    REVOLUTE = "revolute"  # 柜门 / 翻盖铰链


#: 语义部件提示。估计器在 ``kind`` 缺失时按几何长宽比猜测。
KNOWN_KINDS = ("drawer", "door", "lid", "knob", "handle", "base", "other")


@dataclass
class ArticulatedPart:
    """一个已分割的 3D 部件（世界坐标，静止位姿）。

    Attributes:
        name: 稳定唯一的部件名（URDF link 名）。
        vertices: ``(N, 3)`` 顶点数组。要求 >= 4 个点且非共面。
        kind: 语义提示，``"drawer"`` / ``"door"`` / ...；``None`` 表示
            由估计器按几何特征猜测。
        mesh_path: 可选的外部 mesh 文件（导出 URDF <visual> 时引用）。
    """

    name: str
    vertices: np.ndarray
    kind: str | None = None
    mesh_path: str | None = None

    def __post_init__(self) -> None:
        """校验顶点形状与 kind 合法性。"""
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(
                f"part {self.name!r}: vertices must be (N, 3), got {v.shape}"
            )
        if v.shape[0] < 4:
            raise ValueError(
                f"part {self.name!r}: need >= 4 vertices, got {v.shape[0]}"
            )
        self.vertices = v
        if self.kind is not None and self.kind not in KNOWN_KINDS:
            raise ValueError(
                f"part {self.name!r}: unknown kind {self.kind!r}, expect one of {KNOWN_KINDS}"
            )

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


@dataclass
class JointSpec:
    """一个估计出的关节（URDF 可直接消费）。"""

    name: str
    parent: str  # 父 link（通常是 base 部件）
    child: str  # 子 link（可动部件）
    joint_type: JointType
    axis: tuple[float, float, float]  # 世界坐标单位向量（静止位姿下）
    origin: tuple[float, float, float]  # 关节原点（世界坐标）
    lower: float  # 限位下界（prismatic: 米 / revolute: 弧度，均为相对位移）
    upper: float  # 限位上界
    confidence: float = 0.0  # [0, 1]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "child": self.child,
            "type": self.joint_type.value,
            "axis": [round(float(a), 6) for a in self.axis],
            "origin": [round(float(o), 6) for o in self.origin],
            "lower": round(float(self.lower), 6),
            "upper": round(float(self.upper), 6),
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
        }


@dataclass
class ReviewItem:
    """低置信度、需要人工复核的部件。"""

    part: str
    reason: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ArticulationResult:
    """一次标注的完整结果。

    Attributes:
        base: 基座部件名（体积最大的部件）。
        joints: 估计出的关节，按置信度降序。
        review: 进入人工复核队列的部件及原因。
    """

    base: str
    joints: list[JointSpec] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """全自动接受率 = 成功估计的部件 / 全部非基座部件。"""
        total = len(self.joints) + len(self.review)
        return len(self.joints) / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "joints": [j.to_dict() for j in self.joints],
            "review": [r.to_dict() for r in self.review],
        }
