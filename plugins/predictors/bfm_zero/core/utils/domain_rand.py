"""
Domain Randomization Utilities
"""

import numpy as np
import torch
from typing import Dict


class DomainRandomization:
    """域随机化管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        
    def apply_to_simulation(self, scene, robot):
        """应用随机化到仿真"""
        if not self.enabled:
            return
        
        # 质量随机化
        if 'mass_scale' in self.config:
            mass_scale = np.random.uniform(*self.config['mass_scale'])
            # 修改机器人质量
        
        # 摩擦随机化
        if 'friction_scale' in self.config:
            friction_scale = np.random.uniform(*self.config['friction_scale'])
        
        # 重力扰动
        if 'gravity_perturbation' in self.config:
            gravity_noise = np.random.randn(3) * self.config['gravity_perturbation']
    
    def add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """添加观测噪声"""
        if not self.enabled:
            return obs
        
        noise_scale = self.config.get('observation_noise', 0.0)
        noise = np.random.randn(*obs.shape) * noise_scale
        return obs + noise
    
    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """添加动作噪声"""
        if not self.enabled:
            return action
        
        noise_scale = self.config.get('action_noise', 0.0)
        noise = np.random.randn(*action.shape) * noise_scale
        return np.clip(action + noise, -1, 1)
