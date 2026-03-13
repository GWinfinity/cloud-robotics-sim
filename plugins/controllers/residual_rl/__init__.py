"""
Residual RL Plugin

来源: genesis-residual-rl
核心实现:
- 残差网络 (轻量级修正网络)
- 组合策略 (BC + Residual)
- Residual SAC 训练算法
- BC 策略基线
- 视觉编码器

用途: 在行为克隆策略基础上，通过残差学习进行安全微调
"""

__version__ = "0.1.0"
__source__ = "genesis-residual-rl"
__author__ = "Genesis Cloud Sim Team"

from .core.residual_network import ResidualNetwork, CombinedPolicy, ResidualSAC
from .core.bc_policy import BCPolicy
from .core.vision_encoder import VisionEncoder

__all__ = [
    'ResidualNetwork',
    'CombinedPolicy',
    'ResidualSAC',
    'BCPolicy',
    'VisionEncoder'
]
