"""
humanoid_falling Plugin - 人形机器人跌倒保护

来源: genesis-humanoid-falling
核心实现: 通过深度强化学习学会跌倒时的自我保护策略

论文: "Discovering Self-Protective Falling Policy for Humanoid Robot 
       via Deep Reinforcement Learning"

核心特性:
- 自我保护策略学习: 训练人形机器人学会跌倒时的保护动作
- 三角形保护结构: 奖励函数引导形成三角形结构减少冲击
- 课程学习: 逐步增加跌倒场景难度
- 冲击监测: 实时监测关节和躯干受力

关键概念:
- 冲击最小化: 减少跌倒时对关键部位的冲击
- 三角形结构: 利用双臂和躯干形成稳定支撑
- 关节保护: 避免关节超限和自碰撞
"""

__version__ = "0.1.0"
__source__ = "genesis-humanoid-falling"

# 环境
from .core.envs.humanoid_env import HumanoidFallingEnv
from .core.envs.curriculum import FallingCurriculum

# 算法
from .core.algorithms.ppo import PPO

# 工具
from .core.utils.rewards import RewardCalculator
from .core.utils.logger import ImpactMonitor

__all__ = [
    'HumanoidFallingEnv',
    'FallingCurriculum',
    'PPO',
    'RewardCalculator',
    'ImpactMonitor',
]
