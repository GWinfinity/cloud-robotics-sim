"""
Motion Tracking Task

通过潜在向量提示策略跟踪参考运动
"""

import numpy as np
import torch
from typing import Dict, List, Optional


class MotionTrackingTask:
    """
    运动跟踪任务
    
    将参考运动编码为潜在向量，提示策略执行
    """
    
    def __init__(
        self,
        reference_motion: np.ndarray,  # [T, state_dim]
        fb_model,
        window_size: int = 10
    ):
        self.reference_motion = reference_motion
        self.fb_model = fb_model
        self.window_size = window_size
        self.current_frame = 0
        
        # 预计算潜在向量
        self.latent_prompt = self._encode_motion()
    
    def _encode_motion(self) -> torch.Tensor:
        """将运动序列编码为潜在向量"""
        # 使用参考姿势的滑动窗口
        motion_window = self.reference_motion[:self.window_size]
        
        # 转换为tensor
        motion_t = torch.FloatTensor(motion_window).unsqueeze(0)  # [1, T, state_dim]
        
        with torch.no_grad():
            # 使用Backward模型编码
            # 简化为使用最后一帧作为目标
            goal_state = torch.FloatTensor(self.reference_motion[-1]).unsqueeze(0)
            latent = self.fb_model.backward_model(goal_state)
        
        return latent
    
    def get_latent_prompt(self) -> np.ndarray:
        """获取潜在提示"""
        return self.latent_prompt.cpu().numpy()[0]
    
    def get_reference_state(self, current_step: int) -> np.ndarray:
        """获取当前参考状态"""
        frame = min(current_step, len(self.reference_motion) - 1)
        return self.reference_motion[frame]
    
    def compute_tracking_reward(
        self,
        current_state: np.ndarray,
        current_step: int
    ) -> float:
        """
        计算跟踪奖励
        
        衡量当前状态与参考状态的匹配程度
        """
        ref_state = self.get_reference_state(current_step)
        
        # 位置误差
        pos_error = np.linalg.norm(current_state[:3] - ref_state[:3])
        
        # 关节角度误差
        joint_error = np.linalg.norm(current_state[3:] - ref_state[3:])
        
        # 综合奖励
        reward = np.exp(-pos_error - 0.1 * joint_error)
        
        return reward


class MotionLibrary:
    """运动库"""
    
    def __init__(self):
        self.motions = {}
    
    def add_motion(self, name: str, motion_data: np.ndarray):
        """添加运动"""
        self.motions[name] = motion_data
    
    def get_motion(self, name: str) -> np.ndarray:
        """获取运动"""
        return self.motions.get(name)
    
    def list_motions(self) -> List[str]:
        """列出所有运动"""
        return list(self.motions.keys())


class MotionInterpolator:
    """运动插值器"""
    
    def __init__(self, motion1: np.ndarray, motion2: np.ndarray):
        self.motion1 = motion1
        self.motion2 = motion2
    
    def interpolate(self, alpha: float) -> np.ndarray:
        """
        插值两个运动
        
        Args:
            alpha: 插值系数 (0 = motion1, 1 = motion2)
        """
        return (1 - alpha) * self.motion1 + alpha * self.motion2
