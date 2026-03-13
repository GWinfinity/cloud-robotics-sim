"""
Reward Functions for HugWBC

基于 HugWBC 论文的奖励函数实现
"""

import numpy as np
import torch
from typing import Dict, Tuple


class RewardComputer:
    """
    奖励计算器
    
    计算 HugWBC 的各种奖励项
    """
    
    def __init__(self, config: Dict, num_envs: int, device: str = 'cuda'):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        
        # 奖励权重
        self.weights = {k: v['weight'] for k, v in config['rewards'].items()}
        
        # 跟踪统计
        self.episode_sums = {k: np.zeros(num_envs) for k in config['rewards'].keys()}
        self.episode_sums['total'] = np.zeros(num_envs)
    
    def compute_rewards(
        self,
        # 状态
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        # 命令
        commands: np.ndarray,
        # 动作
        actions: np.ndarray,
        last_actions: np.ndarray,
        # 接触
        feet_contact_forces: np.ndarray,
        feet_air_time: np.ndarray,
        # 其他
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        计算所有奖励
        
        Returns:
            rewards: 每个环境的总奖励
            reward_dict: 各项奖励的字典
        """
        rewards = np.zeros(self.num_envs)
        reward_dict = {}
        
        # 1. 线速度跟踪奖励
        lin_vel_error = np.linalg.norm(base_lin_vel[:, :2] - commands[:, :2], axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / self.config['rewards']['tracking_lin_vel'].get('sigma', 0.25))
        reward_dict['tracking_lin_vel'] = tracking_lin_vel * self.weights.get('tracking_lin_vel', 0.0)
        
        # 2. 角速度跟踪奖励
        ang_vel_error = np.abs(base_ang_vel[:, 2] - commands[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / self.config['rewards']['tracking_ang_vel'].get('sigma', 0.25))
        reward_dict['tracking_ang_vel'] = tracking_ang_vel * self.weights.get('tracking_ang_vel', 0.0)
        
        # 3. Z 轴速度惩罚
        lin_vel_z = base_lin_vel[:, 2] ** 2
        reward_dict['lin_vel_z'] = lin_vel_z * self.weights.get('lin_vel_z', 0.0)
        
        # 4. XY 角速度惩罚 (惩罚躯干摇摆)
        ang_vel_xy = base_ang_vel[:, 0] ** 2 + base_ang_vel[:, 1] ** 2
        reward_dict['ang_vel_xy'] = ang_vel_xy * self.weights.get('ang_vel_xy', 0.0)
        
        # 5. 姿态惩罚 (惩罚偏离直立姿态)
        # 假设 projected_gravity 在直立时为 [0, 0, -1]
        orientation = np.linalg.norm(projected_gravity[:, :2], axis=1) ** 2
        reward_dict['orientation'] = orientation * self.weights.get('orientation', 0.0)
        
        # 6. 关节加速度惩罚
        joint_acc = np.sum(joint_vel ** 2, axis=1)
        reward_dict['dof_acc'] = joint_acc * self.weights.get('dof_acc', 0.0)
        
        # 7. 动作变化率惩罚
        action_rate = np.sum((actions - last_actions) ** 2, axis=1)
        reward_dict['action_rate'] = action_rate * self.weights.get('action_rate', 0.0)
        
        # 8. 脚部离地时间奖励
        threshold = self.config['rewards']['feet_air_time'].get('threshold', 0.3)
        feet_air_time_reward = np.sum((feet_air_time - threshold) * (feet_air_time > 0), axis=1)
        reward_dict['feet_air_time'] = feet_air_time_reward * self.weights.get('feet_air_time', 0.0)
        
        # 9. 碰撞惩罚
        collision = np.sum(feet_contact_forces > 100, axis=1)  # 简化实现
        reward_dict['collision'] = collision * self.weights.get('collision', 0.0)
        
        # 汇总奖励
        for key, value in reward_dict.items():
            rewards += value
            self.episode_sums[key] += value
        
        self.episode_sums['total'] += rewards
        reward_dict['total'] = rewards
        
        return rewards, reward_dict
    
    def reset_episode_sums(self, env_ids: np.ndarray):
        """重置指定环境的 episode 统计"""
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0
    
    def get_episode_sums(self) -> Dict:
        """获取 episode 统计"""
        return {k: v.copy() for k, v in self.episode_sums.items()}


class GaitGenerator:
    """
    步态生成器
    
    基于相位周期生成步态信号
    """
    
    def __init__(self, gait_type: str = 'trot', frequency: float = 1.25):
        """
        Args:
            gait_type: 步态类型 ('trot', 'walk', 'pace', 'bound')
            frequency: 步态频率 (Hz)
        """
        self.gait_type = gait_type
        self.frequency = frequency
        
        # 不同步态的相位偏移
        self.phase_offsets = {
            'trot': [0.0, 0.5, 0.5, 0.0],      # 对角小跑
            'walk': [0.0, 0.25, 0.5, 0.75],    # 行走
            'pace': [0.0, 0.5, 0.0, 0.5],      # 侧对步
            'bound': [0.0, 0.0, 0.5, 0.5],     # 兔子跳
        }
        
        self.offsets = self.phase_offsets.get(gait_type, self.phase_offsets['trot'])
    
    def get_phase(self, global_phase: float, leg_idx: int) -> float:
        """获取指定腿的相位"""
        phase = (global_phase + self.offsets[leg_idx]) % 1.0
        return phase
    
    def get_contact_signal(self, global_phase: float, leg_idx: int, 
                          duty_factor: float = 0.5) -> float:
        """
        获取接触信号 (0-1)
        
        Args:
            global_phase: 全局相位 [0, 1)
            leg_idx: 腿索引
            duty_factor: 占空比 (接触时间比例)
        """
        phase = self.get_phase(global_phase, leg_idx)
        return 1.0 if phase < duty_factor else 0.0
    
    def get_swing_signal(self, global_phase: float, leg_idx: int,
                        duty_factor: float = 0.5) -> float:
        """获取摆动信号"""
        return 1.0 - self.get_contact_signal(global_phase, leg_idx, duty_factor)
