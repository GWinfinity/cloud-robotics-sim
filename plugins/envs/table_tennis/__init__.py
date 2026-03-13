"""
table_tennis Plugin - 人形机器人乒乓球环境

来源: genesis-table-tennis
核心实现: 基于预测增强的统一强化学习框架

论文: "Towards Versatile Humanoid Table Tennis: 
       Unified Reinforcement Learning with Prediction Augmentation"

核心特性:
- Unitree G1 全身控制 (手臂击球 + 腿部步法)
- 双预测器架构 (学习预测器 + 物理预测器)
- 预测增强奖励设计
- 击球率 ≥96%, 成功率 ≥92%
"""

__version__ = "0.1.0"
__source__ = "genesis-table-tennis"
__paper__ = "arXiv:2509.21690"

# 环境
from .core.envs.table_tennis_env import TableTennisEnv
from .core.envs.ball_physics import TableTennisBall, BallTrajectoryPredictor
from .core.envs.table import TableTennisTable, Racket

# 模型
from .core.models.predictor import DualPredictor
from .core.models.policy import UnifiedPolicy

# 算法
from .core.algorithms.ppo import PPO

__all__ = [
    'TableTennisEnv',
    'TableTennisBall',
    'BallTrajectoryPredictor',
    'TableTennisTable',
    'Racket',
    'DualPredictor',
    'UnifiedPolicy',
    'PPO',
]
