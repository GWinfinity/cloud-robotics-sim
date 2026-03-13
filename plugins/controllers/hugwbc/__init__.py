"""
hugwbc Plugin - HugWBC 人形机器人全身控制器

来源: hugwbc-genesis
核心实现: 统一的多任务全身运动控制

论文: "HugWBC: A Unified and General Humanoid Whole-Body Controller 
       for Versatile Locomotion" (RSS 2025)

核心特性:
- 统一全身控制: 适用于多种运动任务
- 非对称 Actor-Critic: Critic使用特权信息
- 命令跟踪: 支持速度命令 (vx, vy, yaw_rate)
- 步态控制: 基于相位周期的步态生成
- 域随机化: 提高策略泛化能力

支持任务:
- h1_loco: 平地行走
- h1_stairs: 上下楼梯
- h1_terrain: 复杂地形导航
- h1_int: 交互任务
"""

__version__ = "0.1.0"
__source__ = "hugwbc-genesis"
__paper__ = "HugWBC (RSS 2025)"

# 环境
from .core.envs.hugwbc_env import HugWBCEnv, TaskType

# 算法
from .core.algorithms.ppo import PPO

# 工具
from .core.utils.domain_rand import DomainRandomizer
from .core.utils.rewards import RewardsCalculator

__all__ = [
    'HugWBCEnv',
    'TaskType',
    'PPO',
    'DomainRandomizer',
    'RewardsCalculator',
]
