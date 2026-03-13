"""
Goal Reaching Task

目标到达任务，通过潜在向量指定目标状态
"""

import numpy as np
import torch
from typing import Dict, Optional


class GoalReachingTask:
    """
    目标到达任务
    
    将目标状态编码为潜在向量，提示策略到达该状态
    """
    
    def __init__(
        self,
        goal_state: np.ndarray,
        fb_model,
        tolerance: float = 0.1
    ):
        self.goal_state = goal_state
        self.fb_model = fb_model
        self.tolerance = tolerance
        
        # 编码目标
        self.latent_prompt = self._encode_goal()
    
    def _encode_goal(self) -> torch.Tensor:
        """编码目标状态"""
        goal_t = torch.FloatTensor(self.goal_state).unsqueeze(0)
        
        with torch.no_grad():
            latent = self.fb_model.backward_model(goal_t)
        
        return latent
    
    def get_latent_prompt(self) -> np.ndarray:
        """获取潜在提示"""
        return self.latent_prompt.cpu().numpy()[0]
    
    def compute_distance(self, current_state: np.ndarray) -> float:
        """计算到目标的距离"""
        return np.linalg.norm(current_state - self.goal_state)
    
    def is_reached(self, current_state: np.ndarray) -> bool:
        """检查是否到达目标"""
        return self.compute_distance(current_state) < self.tolerance
    
    def compute_reward(self, current_state: np.ndarray) -> float:
        """计算奖励"""
        distance = self.compute_distance(current_state)
        return np.exp(-distance)


class MultiGoalTask:
    """多目标顺序到达任务"""
    
    def __init__(
        self,
        goal_states: list,
        fb_model,
        tolerance: float = 0.1
    ):
        self.goal_states = goal_states
        self.fb_model = fb_model
        self.tolerance = tolerance
        self.current_goal_idx = 0
        
        # 预编码所有目标
        self.latent_prompts = self._encode_all_goals()
    
    def _encode_all_goals(self) -> list:
        """编码所有目标"""
        latents = []
        for goal in self.goal_states:
            goal_t = torch.FloatTensor(goal).unsqueeze(0)
            with torch.no_grad():
                latent = self.fb_model.backward_model(goal_t)
            latents.append(latent.cpu().numpy()[0])
        return latents
    
    def get_current_latent(self) -> np.ndarray:
        """获取当前目标的潜在向量"""
        return self.latent_prompts[self.current_goal_idx]
    
    def update_goal(self, current_state: np.ndarray):
        """更新当前目标"""
        if self.is_current_goal_reached(current_state):
            self.current_goal_idx = min(
                self.current_goal_idx + 1,
                len(self.goal_states) - 1
            )
    
    def is_current_goal_reached(self, current_state: np.ndarray) -> bool:
        """检查是否到达当前目标"""
        goal = self.goal_states[self.current_goal_idx]
        distance = np.linalg.norm(current_state - goal)
        return distance < self.tolerance
    
    def is_complete(self) -> bool:
        """检查是否完成所有目标"""
        return self.current_goal_idx >= len(self.goal_states) - 1
