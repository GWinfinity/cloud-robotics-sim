"""qd_ik Plugin - frax.ik 差分 IK / 重定向管线的 Quadrants 批量重构

任务（FrameTask/JointTask）+ CBF 约束（关节限位/地板/自碰撞）+ 弹性 QP 的
批处理 ADMM 求解 + quaternion 浮动基座，clips 批量重定向。
结构对标 frax.ik 的 DiffIKSolver/Retargeter；QP 求解为同一弹性 QP 问题的
ADMM（非 ElastiQP 的逐位等价，精度由迭代预算控制）。

依赖: numpy + quadrants（核心）; genesis-world 仅 from_urdf 经 urdfpy
解析时需要（惰性导入，不触发了 gs.init）。
"""

__version__ = "0.1.0"
__category__ = "solvers"

from .constraints import (
    FloorConstraint,
    JointLimits,
    SelfCollisionConstraint,
    SphereModel,
)
from .model import Model, ensure_init
from .retargeter import Retargeter
from .solver import DiffIKSolver, integrate
from .tasks import FrameTask, JointTask

__all__ = [
    "Model",
    "ensure_init",
    "SphereModel",
    "FrameTask",
    "JointTask",
    "JointLimits",
    "FloorConstraint",
    "SelfCollisionConstraint",
    "DiffIKSolver",
    "Retargeter",
    "integrate",
]
