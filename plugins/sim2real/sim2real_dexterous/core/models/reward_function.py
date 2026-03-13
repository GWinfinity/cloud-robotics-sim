"""
Generalized Reward Function

基于接触和物体目标的通用奖励公式
适用于多种操作任务
"""

import torch
import numpy as np
from typing import Dict, Optional


class GeneralizedRewardFunction:
    """
    通用奖励函数
    
    适用于:
    - Grasp-and-Reach
    - Box Lift
    - Bimanual Handover
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 奖励权重
        self.contact_weight = config.get('contact', {}).get('weight', 2.0)
        self.object_goal_weight = config.get('object_goal', {}).get('weight', 3.0)
        self.hand_goal_weight = config.get('hand_goal', {}).get('weight', 1.5)
        
        # 正则化权重
        self.energy_penalty = config.get('energy_penalty', -0.001)
        self.action_smoothness = config.get('action_smoothness', -0.01)
        self.joint_limit_penalty = config.get('joint_limit_penalty', -0.1)
        self.self_collision_penalty = config.get('self_collision_penalty', -1.0)
    
    def compute_reward(
        self,
        state: Dict,
        action: np.ndarray,
        task_type: str = 'grasp_and_reach'
    ) -> float:
        """
        计算通用奖励
        
        Args:
            state: 环境状态
            action: 执行的动作
            task_type: 任务类型
        """
        reward = 0.0
        
        # 1. 接触奖励
        contact_reward = self._compute_contact_reward(state)
        reward += self.contact_weight * contact_reward
        
        # 2. 物体目标奖励 (任务特定)
        if task_type == 'grasp_and_reach':
            object_reward = self._compute_grasp_reach_reward(state)
        elif task_type == 'box_lift':
            object_reward = self._compute_box_lift_reward(state)
        elif task_type == 'bimanual_handover':
            object_reward = self._compute_handover_reward(state)
        else:
            object_reward = 0.0
        
        reward += self.object_goal_weight * object_reward
        
        # 3. 手部目标奖励
        hand_reward = self._compute_hand_goal_reward(state)
        reward += self.hand_goal_weight * hand_reward
        
        # 4. 正则化惩罚
        reward += self.energy_penalty * np.sum(action ** 2)
        
        # 动作平滑性 (需要存储上一帧动作)
        # reward += self.action_smoothness * action_diff
        
        return reward
    
    def _compute_contact_reward(self, state: Dict) -> float:
        """
        接触奖励
        
        鼓励手掌和手指与物体接触
        """
        reward = 0.0
        
        # 左手接触
        left_palm_contact = state.get('left_palm_contact_force', 0.0)
        left_finger_contacts = state.get('left_finger_contact_forces', [0.0] * 5)
        
        # 右手接触
        right_palm_contact = state.get('right_palm_contact_force', 0.0)
        right_finger_contacts = state.get('right_finger_contact_forces', [0.0] * 5)
        
        # 手掌接触奖励
        palm_threshold = self.config.get('contact', {}).get('palm_contact_threshold', 1.0)
        if left_palm_contact > palm_threshold:
            reward += 0.5
        if right_palm_contact > palm_threshold:
            reward += 0.5
        
        # 手指接触奖励
        finger_threshold = self.config.get('contact', {}).get('finger_contact_threshold', 0.5)
        for force in left_finger_contacts:
            if force > finger_threshold:
                reward += 0.1
        for force in right_finger_contacts:
            if force > finger_threshold:
                reward += 0.1
        
        # 稳定抓取奖励 (多手指同时接触)
        left_fingers_touching = sum(1 for f in left_finger_contacts if f > finger_threshold)
        right_fingers_touching = sum(1 for f in right_finger_contacts if f > finger_threshold)
        
        stable_grasp_threshold = 3  # 至少3个手指
        if left_fingers_touching >= stable_grasp_threshold:
            reward += self.config.get('contact', {}).get('stable_grasp_bonus', 1.0)
        if right_fingers_touching >= stable_grasp_threshold:
            reward += self.config.get('contact', {}).get('stable_grasp_bonus', 1.0)
        
        return reward
    
    def _compute_grasp_reach_reward(self, state: Dict) -> float:
        """Grasp-and-Reach任务奖励"""
        obj_pos = state.get('object_position', np.zeros(3))
        target_pos = state.get('target_position', np.zeros(3))
        
        # 物体到目标的距离
        distance = np.linalg.norm(obj_pos - target_pos)
        
        # 距离奖励 (指数衰减)
        reward = np.exp(-distance * 5)
        
        # 成功奖励
        success_threshold = self.config.get('tasks', {}).get('grasp_and_reach', {}).get('success_threshold', 0.05)
        if distance < success_threshold:
            reward += 10.0
        
        return reward
    
    def _compute_box_lift_reward(self, state: Dict) -> float:
        """Box Lift任务奖励"""
        obj_pos = state.get('object_position', np.zeros(3))
        obj_height = obj_pos[2]
        
        lift_threshold = self.config.get('tasks', {}).get('box_lift', {}).get('lift_height_threshold', 0.3)
        
        # 提升高度奖励
        reward = min(obj_height / lift_threshold, 1.0)
        
        # 成功奖励
        if obj_height > lift_threshold:
            reward += 5.0
            
            # 检查稳定性
            obj_vel = state.get('object_velocity', np.zeros(3))
            if np.linalg.norm(obj_vel) < 0.1:  # 稳定
                reward += 5.0
        
        return reward
    
    def _compute_handover_reward(self, state: Dict) -> float:
        """Bimanual Handover任务奖励"""
        reward = 0.0
        
        obj_pos = state.get('object_position', np.zeros(3))
        handover_pos = np.array(self.config.get('tasks', {}).get('bimanual_handover', {}).get('handover_position', [0.5, 0, 0.5]))
        
        # 物体接近交接点
        distance_to_handover = np.linalg.norm(obj_pos - handover_pos)
        reward += np.exp(-distance_to_handover * 3)
        
        # 左手释放 (接触力减小)
        left_contact = state.get('left_palm_contact_force', 0.0)
        if left_contact < 0.5:  # 左手已释放
            reward += 2.0
        
        # 右手抓取
        right_contact = state.get('right_palm_contact_force', 0.0)
        if right_contact > 1.0:  # 右手已抓取
            reward += 2.0
        
        # 物体稳定
        obj_vel = state.get('object_velocity', np.zeros(3))
        if np.linalg.norm(obj_vel) < 0.1 and right_contact > 1.0:
            reward += 5.0
        
        return reward
    
    def _compute_hand_goal_reward(self, state: Dict) -> float:
        """
        手部目标奖励
        
        鼓励手部到达合适位置
        """
        reward = 0.0
        
        left_hand_pos = state.get('left_hand_position', np.zeros(3))
        right_hand_pos = state.get('right_hand_position', np.zeros(3))
        obj_pos = state.get('object_position', np.zeros(3))
        
        # 手到物体的距离
        left_dist = np.linalg.norm(left_hand_pos - obj_pos)
        right_dist = np.linalg.norm(right_hand_pos - obj_pos)
        
        position_threshold = self.config.get('hand_goal', {}).get('position_threshold', 0.05)
        
        # 接近奖励
        reward += np.exp(-left_dist * 3) + np.exp(-right_dist * 3)
        
        # 到达奖励
        if left_dist < position_threshold:
            reward += 1.0
        if right_dist < position_threshold:
            reward += 1.0
        
        return reward
