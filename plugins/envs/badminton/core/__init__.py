"""
Genesis Humanoid Badminton

基于 Genesis 引擎复现论文:
"Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning"

主要组件:
- envs: 环境实现，包括羽毛球物理和场地
- algorithms: PPO 强化学习算法
- utils: 工具函数，包括 EKF 轨迹预测
"""

__version__ = "0.1.0"
__author__ = "Genesis RL Team"

from envs.badminton_env import BadmintonEnv
from envs.curriculum import ThreeStageCurriculum
from algorithms.ppo import PPO

__all__ = [
    'BadmintonEnv',
    'ThreeStageCurriculum',
    'PPO'
]
