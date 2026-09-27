"""articulation_gen Plugin - 关节资产自动标注（抽屉开合 / 柜门铰链）

核心实现: 分割（命名节点 / YOLO+SAM）→ 几何关节估计 → URDF 导出

来源: genesis-cloud-sim 自研模块
定位: 为 ReplicaCAD 式关节家具资产提供"自动生成"路径，
      与 plugins/envs/maniskill 的 URDF 关节加载链路对接。

核心特性:
- 三条输入路径: GLB 命名节点（离线）/ YOLO26+SAM 图像反投影（需微调权重）/ 已有部件标签
- 纯 numpy 几何估计器: OBB + 外露面选择 + 碰撞扫描验证，零推理依赖
- 置信度 + 人工复核队列: 不静默放过低置信度估计，batch 落盘 review.jsonl
- URDF 导出: revolute/prismatic + axis/origin/limit，Genesis 可直接加载

可选依赖（ guard 导入，缺失时仅图像路径不可用）:
- ultralytics: YOLO 家族检测器（官方权重或家具部件微调权重）
- sam2 / ultralytics[fastsam]: mask 精修
"""

__version__ = "0.1.0"
__category__ = "predictors"

from .core import (
    KNOWN_KINDS,
    ArticulatedPart,
    ArticulationResult,
    GenesisVerifier,
    JointCheck,
    JointSpec,
    JointType,
    ReviewItem,
    VerificationReport,
    estimate_joints,
    result_to_urdf,
    validate_urdf,
)
from .core.pipeline import (
    Camera,
    annotate_from_images,
    annotate_labeled_mesh,
    annotate_parts,
    batch_annotate,
)

__all__ = [
    "KNOWN_KINDS",
    "ArticulatedPart",
    "ArticulationResult",
    "JointSpec",
    "JointType",
    "ReviewItem",
    "VerificationReport",
    "JointCheck",
    "GenesisVerifier",
    "estimate_joints",
    "result_to_urdf",
    "validate_urdf",
    "Camera",
    "annotate_from_images",
    "annotate_labeled_mesh",
    "annotate_parts",
    "batch_annotate",
]
