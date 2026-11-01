"""qd_rbd Plugin - Pinocchio 刚体动力学算法的 Quadrants 批量重构

把 Pinocchio 的核心动力学算法（RNEA / CRBA / ABA / M·x / FK）从
「C++ / 面向对象 / 单实例」重写为「Quadrants kernel / SOA / 批量并行」：
数值上与 Pinocchio f64 对齐，大 batch 吞吐是 Pinocchio 串行的数倍至数十倍。

来源: qd_rbd 八轮迭代存档的第八轮（v4.1 scratch 打包 + C 后端，
      RNEA 默认走 ``csrc/rbd.c``，``QD_RBD_C=0`` 退回 Quadrants）。
依赖: numpy + quadrants（核心）；
      可选 ``pin``（pinocchio，仅 ``Model.from_pinocchio`` / 参考模型需要，
      Windows 无官方 wheel，请在 WSL 中使用）；
      可选 C 编译器（仅 C 后端需要，缺失时自动退回 Quadrants 后端）。
"""

__version__ = "0.1.0"
__category__ = "solvers"

import importlib.util as _ilu

#: pinocchio 是否可导入（决定 from_pinocchio / models 是否可用）。
HAS_PINOCCHIO = _ilu.find_spec("pinocchio") is not None

from .api import (  # noqa: E402
    Data,
    Model,
    aba,
    crba,
    crba_apply,
    ensure_init,
    forward_kinematics,
    non_linear_effects,
    rnea,
)

__all__ = [
    "Model",
    "Data",
    "ensure_init",
    "rnea",
    "non_linear_effects",
    "crba",
    "crba_apply",
    "aba",
    "forward_kinematics",
    "HAS_PINOCCHIO",
]
