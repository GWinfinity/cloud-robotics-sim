"""
sim2real_dexterous Plugin - Sim-to-Real 灵巧双手操作

来源: genesis-sim2real-dexterous
核心实现: 基于视觉的sim-to-real强化学习用于人形机器人灵巧操作

论文: "Sim-to-Real Reinforcement Learning for Vision-Based Dexterous 
       Manipulation on Humanoids"

核心特性:
- Real-to-Sim自动调优: 从真实数据自动调整仿真参数
- 通用奖励公式: 基于接触和物体目标的奖励设计
- 分而治之策略蒸馏: 单任务专家 → 通用多任务策略
- 混合物体表示: 视觉 + 点云 + 本体感觉

支持任务:
- grasp_and_reach: 抓取并伸展
- box_lift: 箱体提升
- bimanual_handover: 双手交接
"""

__version__ = "0.1.0"
__source__ = "genesis-sim2real-dexterous"
__paper__ = "arXiv:2502.20396"

# 环境
from .core.envs.dexterous_env import DexterousManipulationEnv, TaskType

# 算法
from .core.algorithms.real2sim_tuning import Real2SimTuner

# 模型
from .core.models.policy_distillation import PolicyDistillation
from .core.models.object_representation import HybridObjectRepresentation
from .core.models.reward_function import UniversalRewardFunction

__all__ = [
    'DexterousManipulationEnv',
    'TaskType',
    'Real2SimTuner',
    'PolicyDistillation',
    'HybridObjectRepresentation',
    'UniversalRewardFunction',
]
