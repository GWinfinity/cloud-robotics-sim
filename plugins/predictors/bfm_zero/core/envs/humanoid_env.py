"""
Unitree G1 Humanoid Environment for BFM-Zero
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional


class HumanoidEnv:
    """
    人形机器人环境 (Unitree G1)
    
    支持域随机化和非对称观测
    """
    
    def __init__(
        self,
        config: Dict,
        num_envs: int = 1,
        headless: bool = False
    ):
        self.config = config
        self.num_envs = num_envs
        self.headless = headless
        
        # 初始化Genesis
        gs.init(backend=gs.backends.CUDA)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 创建机器人
        self.robot = self._create_robot()
        
        # 状态维度
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        
        # 域随机化
        self.domain_rand = DomainRandomizer(config.get('domain_randomization', {}))
        
        # 历史观测 (用于非对称学习)
        self.history_length = config.get('asymmetric_learning', {}).get('history_length', 5)
        self.observation_history = []
        
        # 特权信息 (仅训练时使用)
        self.privileged_info = {}
        
    def _create_scene(self):
        """创建场景"""
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 3, 2),
                camera_lookat=(0, 0, 1),
                camera_fov=45,
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless,
        )
    
    def _create_robot(self):
        """创建Unitree G1机器人"""
        # 使用Genesis内置的humanoid作为近似
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
        )
        
        # 添加地面
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
        )
        
        return robot
    
    def _get_state_dim(self) -> int:
        """获取状态维度"""
        # 位置(3) + 方向(4) + 关节位置(12) + 关节速度(12) = 31
        # 简化版
        return 31
    
    def _get_action_dim(self) -> int:
        """获取动作维度"""
        return 12  # 简化为12个关节
    
    def reset(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """重置环境"""
        # 应用域随机化
        self.domain_rand.randomize(self.robot)
        
        # 重置位置
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 构建场景
        self.scene.build()
        
        # 重置历史
        self.observation_history = []
        
        # 获取观测
        obs = self._get_observation()
        privileged = self._get_privileged_info()
        
        return obs, privileged
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 应用域随机化的动作噪声
        action = self.domain_rand.add_action_noise(action)
        
        # 执行动作
        self.robot.control_dofs_position(action)
        self.scene.step()
        
        # 获取新观测
        obs = self._get_observation()
        privileged = self._get_privileged_info()
        
        # 更新历史
        self.observation_history.append(obs)
        if len(self.observation_history) > self.history_length:
            self.observation_history.pop(0)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止
        done = self._check_termination()
        
        info = {
            'privileged_info': privileged
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        # 基础观测 (所有策略都能看到的)
        base_pos = self.robot.get_pos().cpu().numpy()
        base_quat = self.robot.get_quat().cpu().numpy()
        joint_pos = self.robot.get_dofs_position().cpu().numpy()[:12]
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()[:12]
        
        obs = np.concatenate([base_pos, base_quat, joint_pos, joint_vel])
        
        return obs.astype(np.float32)
    
    def _get_privileged_info(self) -> np.ndarray:
        """获取特权信息 (仅训练时使用)"""
        # 包括: 身体速度、接触力、摩擦系数等
        base_vel = self.robot.get_vel().cpu().numpy()
        base_ang_vel = self.robot.get_ang().cpu().numpy()
        
        # 简化的特权信息
        privileged = np.concatenate([base_vel, base_ang_vel])
        
        return privileged.astype(np.float32)
    
    def get_history_observation(self) -> Optional[np.ndarray]:
        """获取历史观测"""
        if len(self.observation_history) < self.history_length:
            return None
        
        # 展平历史
        return np.concatenate(self.observation_history).astype(np.float32)
    
    def _compute_reward(self) -> float:
        """计算基础奖励"""
        # 存活奖励
        reward = 1.0
        
        # 能量惩罚
        # 可以从动作计算，但这里简化
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 检查是否倒地
        base_pos = self.robot.get_pos().cpu().numpy()
        if base_pos[2] < 0.3:  # 高度过低
            return True
        
        return False


class DomainRandomizer:
    """域随机化"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def randomize(self, robot):
        """随机化机器人参数"""
        if not self.enabled:
            return
        
        # 质量随机化
        if 'mass_scale' in self.config:
            scale = np.random.uniform(*self.config['mass_scale'])
            # 应用到机器人
        
        # 摩擦随机化
        if 'friction_scale' in self.config:
            scale = np.random.uniform(*self.config['friction_scale'])
    
    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """添加动作噪声"""
        if not self.enabled:
            return action
        
        noise_scale = self.config.get('action_noise', 0.0)
        noise = np.random.randn(*action.shape) * noise_scale
        return action + noise
