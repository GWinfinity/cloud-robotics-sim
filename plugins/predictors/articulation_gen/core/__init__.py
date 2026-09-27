"""articulation_gen core — 关节资产自动标注。

子模块：
- joint_spec: 数据模型（ArticulatedPart / JointSpec / ArticulationResult）
- estimate:   纯 numpy 几何关节估计器（核心，无推理依赖）
- segment:    YOLO + SAM 2D 分割适配层（可选依赖）
- pipeline:   输入路径编排（命名节点 mesh / 图像反投影 / 已有部件）
- export_urdf: ArticulationResult → URDF
"""

from .estimate import estimate_joints
from .export_urdf import result_to_urdf, validate_urdf
from .joint_spec import (
    KNOWN_KINDS,
    ArticulatedPart,
    ArticulationResult,
    JointSpec,
    JointType,
    ReviewItem,
)
from .verify import GenesisVerifier, JointCheck, VerificationReport

__all__ = [
    "KNOWN_KINDS",
    "ArticulatedPart",
    "ArticulationResult",
    "JointSpec",
    "JointType",
    "ReviewItem",
    "estimate_joints",
    "result_to_urdf",
    "validate_urdf",
    "GenesisVerifier",
    "JointCheck",
    "VerificationReport",
]
