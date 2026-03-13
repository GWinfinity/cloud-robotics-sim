"""
Reward Optimization Task

奖励优化任务，通过潜在向量指定奖励函数
"""

import numpy as np
import torch
from typing import Callable, Dict, Optional


class RewardOptimizationTask:
    """
    奖励优化任务
    
    将奖励函数编码为潜在向量，提示策略优化该奖励
    """
    
    def __init__(
        self,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        fb_model,
        state_dim: int
    ):
        self.reward_fn = reward_fn
        self.fb_model = fb_model
        self.state_dim = state_dim
        
        # 采样一些状态-动作对来编码奖励函数
        self.latent_prompt = self._encode_reward_function()
    
    def _encode_reward_function(self) -> torch.Tensor:
        """
        编码奖励函数
        
        通过在多个状态上评估奖励来编码
        """
        # 采样随机状态
        num_samples = 100
        states = torch.randn(num_samples, self.state_dim)
        actions = torch.randn(num_samples, self.fb_model.action_dim)
        
        # 计算奖励
        rewards = []
        for s, a in zip(states, actions):
            r = self.reward_fn(s.numpy(), a.numpy())
            rewards.append(r)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        # 编码到潜在空间
        with torch.no_grad():
            # 使用平均表示
            latents = []
            for s, r in zip(states, rewards):
                z = self.fb_model.backward_model(s.unsqueeze(0), r.unsqueeze(0))
                latents.append(z)
            latent = torch.stack(latents).mean(dim=0)
        
        return latent
    
    def get_latent_prompt(self) -> np.ndarray:
        """获取潜在提示"""
        return self.latent_prompt.cpu().numpy()[0]
    
    def compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """计算奖励"""
        return self.reward_fn(state, action)


class VelocityTask(RewardOptimizationTask):
    """速度最大化任务"""
    
    def __init__(self, fb_model, state_dim: int, direction: np.ndarray):
        self.direction = direction / np.linalg.norm(direction)
        
        def reward_fn(state, action):
            # 假设状态中包含速度信息
            velocity = state[:3]  # 简化为前3维
            return np.dot(velocity, self.direction)
        
        super().__init__(reward_fn, fb_model, state_dim)


class EnergyEfficiencyTask(RewardOptimizationTask):
    """能量效率任务"""
    
    def __init__(self, fb_model, state_dim: int, action_dim: int):
        def reward_fn(state, action):
            # 惩罚大动作
            return -np.sum(action ** 2)
        
        super().__init__(reward_fn, fb_model, state_dim)


class BalanceTask(RewardOptimizationTask):
    """平衡任务"""
    
    def __init__(self, fb_model, state_dim: int):
        def reward_fn(state, action):
            # 假设状态中包含高度信息
            height = state[2]  # z坐标
            return height  # 奖励保持高度
        
        super().__init__(reward_fn, fb_model, state_dim)
