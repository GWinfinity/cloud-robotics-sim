"""
slac Plugin - SLAC: Simulation-Pretrained Latent Action Space

来源: genesis-slac
核心实现: 用于高自由度机器人全身真实世界强化学习的潜在动作空间

论文: "SLAC: Simulation-Pretrained Latent Action Space for 
       Whole-Body Real-World RL"

核心特性:
- 低保真仿真预训练: 在低保真仿真器中预训练任务无关的潜在动作空间
- 无监督技能发现: 通过DIAYN风格的多样性驱动学习潜在动作空间
- 真实世界下游学习: 使用潜在动作空间作为动作接口学习具体任务
- 安全探索: 在潜在空间中约束动作，避免危险动作

三个阶段:
1. 预训练潜在动作空间 (仿真)
2. 技能发现 (无监督)
3. 下游任务学习 (真实世界)
"""

__version__ = "0.1.0"
__source__ = "genesis-slac"
__paper__ = "arXiv:2506.04147"

# 模型
from .core.models.latent_action import LatentActionVAE
from .core.models.skill_discovery import SkillDiscovery
from .core.models.downstream_policy import DownstreamPolicy

# 环境
from .core.envs.mobile_manipulator import MobileManipulatorEnv

# 控制器
class SLACPretrainer:
    """SLAC预训练器"""
    def __init__(self, latent_action_model, skill_discovery, config):
        self.latent_action_model = latent_action_model
        self.skill_discovery = skill_discovery
        self.config = config
    
    def pretrain(self, env, num_iterations):
        """预训练潜在动作空间"""
        pass

class LatentActionController:
    """潜在动作控制器"""
    def __init__(self, latent_action_vae, downstream_policy):
        self.latent_action_vae = latent_action_vae
        self.downstream_policy = downstream_policy
    
    def get_action(self, obs, latent_action):
        """将潜在动作解码为原始动作"""
        return self.latent_action_vae.decode(latent_action, obs)

__all__ = [
    'SLACPretrainer',
    'LatentActionVAE',
    'SkillDiscovery',
    'DownstreamPolicy',
    'MobileManipulatorEnv',
    'LatentActionController',
]
