"""
bfm_zero Plugin - BFM-Zero: 可提示的行为基础模型

来源: genesis-bfm-zero
核心实现: 使用无监督强化学习训练的Zero-shot人形机器人控制

论文: "BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid 
       Control Using Unsupervised Reinforcement Learning"

核心特性:
- Zero-shot: 运动跟踪、目标到达、奖励优化
- Few-shot: 基于优化的适应新任务
- 统一潜在空间: 将运动、目标、奖励嵌入共享空间
- Forward-Backward模型: 结构化的共享表示学习

核心技术:
- Forward Representation (F): 从状态-动作映射到潜在空间
- Backward Representation (B): 从目标/奖励映射到潜在空间
- Successor Features: 学习目标条件策略
- 域随机化: 支持Sim-to-Real迁移
"""

__version__ = "0.1.0"
__source__ = "genesis-bfm-zero"
__paper__ = "arXiv:2511.04131"

# 模型
from .core.models.fb_model import FBModel
from .core.models.policy import BFMZeroPolicy

# 任务
from .core.tasks.motion_tracking import MotionTrackingTask
from .core.tasks.goal_reaching import GoalReachingTask
from .core.tasks.reward_opt import RewardOptimizationTask

# 环境
from .core.envs.humanoid_env import HumanoidEnv

# 算法
from .core.algorithms.fb_training import FBTrainer

# 工具
from .core.utils.domain_rand import DomainRandomizer

__all__ = [
    'FBModel',
    'BFMZeroPolicy',
    'MotionTrackingTask',
    'GoalReachingTask',
    'RewardOptimizationTask',
    'HumanoidEnv',
    'FBTrainer',
    'DomainRandomizer',
]
