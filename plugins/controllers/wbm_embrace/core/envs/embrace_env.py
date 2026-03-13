"""
Whole-Body Embracing Environment

人形机器人全身拥抱环境
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional

from .bulky_objects import BulkyObjectGenerator


class EmbraceEnv:
    """
    全身拥抱环境
    
    核心功能:
    - 生成多样化的大型物体
    - 多接触点检测 (手臂+躯干)
    - NSDF几何感知
    - 运动自然性评估
    """
    
    def __init__(
        self,
        config: Dict,
        num_envs: int = 1,
        headless: bool = False,
        device: str = 'cuda'
    ):
        self.config = config
        self.num_envs = num_envs
        self.headless = headless
        self.device = device
        
        # 初始化Genesis
        gs.init(backend=gs.backends.CUDA)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 创建地面
        self._create_ground()
        
        # 创建机器人
        self.robot = self._create_robot()
        
        # 物体生成器
        self.object_generator = BulkyObjectGenerator(self.scene, config['env']['objects'])
        self.current_object = None
        self.current_object_info = None
        
        # 状态维度
        self.state_dim = self._get_state_dim()
        self.action_dim = 23  # 23个关节
        
        # 回合统计
        self.episode_stats = {
            'contacts': 0,
            'object_lifted': False,
            'hold_time': 0.0,
            'success': False
        }
        
        self.current_step = 0
        self.max_steps = config['env']['episode_length']
        
    def _create_scene(self):
        """创建场景"""
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2, 2, 2),
                camera_lookat=(0.5, 0, 0.5),
                camera_fov=60,
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless,
        )
    
    def _create_ground(self):
        """创建地面"""
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0))
        )
    
    def _create_robot(self):
        """创建人形机器人"""
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.8, 1.0))
        )
        
        init_pos = self.config['robot']['init_pos']
        robot.set_pos(init_pos)
        
        return robot
    
    def _get_state_dim(self) -> int:
        """获取状态维度"""
        # 关节位置(23) + 关节速度(23) + 根节点位置(3) + 根节点方向(4) + 物体位置(7) = 60
        return 60
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {
            'contacts': 0,
            'object_lifted': False,
            'hold_time': 0.0,
            'success': False
        }
        
        # 重置机器人
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 生成新物体
        if self.current_object is not None:
            # 移除旧物体
            pass  # Genesis中移除实体的方法
        
        self.current_object, self.current_object_info = \
            self.object_generator.generate_random_object()
        
        # 构建场景
        self.scene.build()
        
        # 获取观测
        obs = self.get_observation()
        info = {
            'object_info': self.current_object_info,
            'object_vertices': self.object_generator.get_object_mesh_vertices(
                self.current_object, self.current_object_info
            )
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 应用动作
        self._apply_action(action)
        
        # 仿真步进
        self.scene.step()
        
        # 获取观测
        obs = self.get_observation()
        
        # 检测接触
        contact_info = self._check_contacts()
        
        # 计算奖励
        reward = self._compute_reward(action, contact_info)
        
        # 更新统计
        self.episode_stats['contacts'] = contact_info['num_contacts']
        
        # 检查物体是否被抬起
        obj_pos = self.current_object.get_pos().cpu().numpy()
        if obj_pos[2] > self.current_object_info['position'][2] + 0.1:
            self.episode_stats['object_lifted'] = True
            self.episode_stats['hold_time'] += self.config['genesis']['dt']
        
        # 检查成功
        if self.episode_stats['object_lifted'] and \
           self.episode_stats['hold_time'] >= self.config['rewards']['success']['hold_time']:
            self.episode_stats['success'] = True
        
        # 检查终止
        self.current_step += 1
        done = self._check_termination()
        
        info = {
            'contact_info': contact_info,
            'stats': self.episode_stats.copy()
        }
        
        return obs, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作到机器人"""
        # 缩放动作
        action_scale = 0.5
        action = np.clip(action * action_scale, -1, 1)
        
        # 设置关节位置
        self.robot.control_dofs_position(action)
    
    def _check_contacts(self) -> Dict:
        """检查多接触点"""
        # 获取接触信息
        contacts = self.robot.get_contacts()
        
        # 统计与物体的接触
        contact_bodies = []
        contact_forces = []
        
        for contact in contacts:
            # 检查是否是与当前物体的接触
            # 简化处理
            contact_bodies.append(contact.get('link', 'unknown'))
            contact_forces.append(contact.get('force', 0.0))
        
        return {
            'num_contacts': len(contact_bodies),
            'contact_bodies': contact_bodies,
            'contact_forces': contact_forces
        }
    
    def get_observation(self) -> np.ndarray:
        """获取观测"""
        # 机器人状态
        joint_pos = self.robot.get_dofs_position().cpu().numpy()[:23]
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()[:23]
        
        # 填充到23维
        joint_pos = np.pad(joint_pos, (0, 23 - len(joint_pos)), 'constant')
        joint_vel = np.pad(joint_vel, (0, 23 - len(joint_vel)), 'constant')
        
        root_pos = self.robot.get_pos().cpu().numpy()
        root_quat = self.robot.get_quat().cpu().numpy()
        
        # 物体状态 (位置+四元数)
        obj_pos = self.current_object.get_pos().cpu().numpy()
        obj_quat = self.current_object.get_quat().cpu().numpy()
        
        # 组合观测
        obs = np.concatenate([
            joint_pos.flatten()[:23],
            joint_vel.flatten()[:23],
            root_pos.flatten()[:3],
            root_quat.flatten()[:4],
            obj_pos.flatten()[:3],
            obj_quat.flatten()[:4]
        ])
        
        # 填充到60维
        if len(obs) < 60:
            obs = np.pad(obs, (0, 60 - len(obs)), 'constant')
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray, contact_info: Dict) -> float:
        """计算奖励"""
        reward = 0.0
        reward_config = self.config['rewards']
        
        # 1. 接触奖励
        num_contacts = contact_info['num_contacts']
        target_contacts = reward_config['contact']['num_contacts_threshold']
        if num_contacts >= target_contacts:
            reward += reward_config['contact']['weight']
        else:
            reward += reward_config['contact']['weight'] * (num_contacts / target_contacts)
        
        # 2. 稳定性奖励
        if self.episode_stats['object_lifted']:
            # 物体速度
            obj_vel = self.current_object.get_vel().cpu().numpy()
            obj_speed = np.linalg.norm(obj_vel)
            
            if obj_speed < reward_config['stability']['object_velocity_threshold']:
                reward += reward_config['stability']['weight']
        
        # 3. 载荷奖励
        if self.episode_stats['object_lifted']:
            obj_pos = self.current_object.get_pos().cpu().numpy()
            lift_height = obj_pos[2] - self.current_object_info['position'][2]
            
            if lift_height > reward_config['payload']['lift_height_threshold']:
                reward += reward_config['payload']['weight']
        
        # 4. 成功奖励
        if self.episode_stats['success']:
            reward += reward_config['success']['weight']
        
        # 5. 能量惩罚
        energy_penalty = np.sum(action ** 2) * reward_config['energy_penalty']
        reward += energy_penalty
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 时间限制
        if self.current_step >= self.max_steps:
            return True
        
        # 成功完成
        if self.episode_stats['success']:
            return True
        
        # 机器人倒地
        root_pos = self.robot.get_pos().cpu().numpy()
        if root_pos[2] < 0.3:
            return True
        
        return False
    
    def close(self):
        """关闭环境"""
        gs.destroy()
