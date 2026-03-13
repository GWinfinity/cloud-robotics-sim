"""
Badminton Environment Plugin

来源: genesis-humanoid-badminton
核心实现:
- 人形机器人羽毛球环境
- 三阶段课程学习
- 羽毛球物理模型 (空气动力学)
- EKF 轨迹预测

用途: 训练人形机器人打羽毛球
"""

__version__ = "0.1.0"
__source__ = "genesis-humanoid-badminton"
__author__ = "Genesis Cloud Sim Team"

from .core.badminton_env import BadmintonEnv
from .core.shuttlecock import Shuttlecock, BadmintonCourt
from .core.curriculum import CurriculumStage
from .core.ekf import EKFPredictor
from .core.rewards import BadmintonRewards

__all__ = [
    'BadmintonEnv',
    'Shuttlecock',
    'BadmintonCourt',
    'CurriculumStage',
    'EKFPredictor',
    'BadmintonRewards'
]
