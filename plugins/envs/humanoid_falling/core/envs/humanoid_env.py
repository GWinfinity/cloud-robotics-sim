"""
Humanoid Falling Environment for Genesis
基于论文 "Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning"
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional, List
import yaml


class HumanoidFallingEnv:
    """
    人形机器人跌倒保护训练环境
    
    目标: 训练机器人在被推倒时学会自我保护动作，形成"三角形"结构减少冲击损伤
    """
    
    def __init__(self, config_path: str = None, num_envs: int = 1, headless: bool = False):
        """
        初始化环境
        
        Args:
            config_path: 配置文件路径
            num_envs: 并行环境数量
            headless: 是否无头模式
        """
        self.num_envs = num_envs
        self.headless = headless
        self.config = self._load_config(config_path)
        
        # 初始化 Genesis
        gs.init(backend=gs.backends.CUDA)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 添加地面
        self._add_ground()
        
        # 添加机器人
        self.robot = self._add_robot()
        
        # 环境参数
        self.episode_length = self.config['env']['episode_length']
        self.current_step = 0
        
        # 动作和观察空间
        self.num_actions = self._get_num_actions()
        self.num_obs = self._get_num_obs()
        
        # 推力参数 (由课程学习控制)
        self.push_force_range = [50, 100]
        self.push_duration = 0.1
        self.push_timer = 0
        self.is_pushing = False
        self.push_direction = np.zeros(3)
        
        # 历史动作 (用于平滑性奖励)
        self.last_actions = np.zeros((num_envs, self.num_actions))
        
        # 记录碰撞信息
        self.contact_forces = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'genesis': {'dt': 0.01, 'substeps': 10},
                'scene': {'show_viewer': True},
                'env': {'episode_length': 500},
                'rewards': {
                    'survival': {'weight': 1.0},
                    'impact': {'weight': -0.1},
                    'triangle_structure': {'weight': 2.0},
                    'joint_limit': {'weight': -0.5},
                    'orientation': {'weight': 0.5},
                    'action_smoothness': {'weight': -0.01},
                    'energy': {'weight': -0.001}
                }
            }
    
    def _create_scene(self):
        """创建 Genesis 场景"""
        scene_config = self.config.get('scene', {})
        viewer_config = scene_config.get('viewer', {})
        
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=viewer_config.get('camera_pos', (3, 3, 2)),
                camera_lookat=viewer_config.get('camera_lookat', (0, 0, 0.5)),
                camera_fov=viewer_config.get('camera_fov', 45),
                res=viewer_config.get('res', (1280, 720)),
                max_FPS=viewer_config.get('max_FPS', 60),
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless and scene_config.get('show_viewer', True),
        )
    
    def _add_ground(self):
        """添加地面"""
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                color=(0.9, 0.9, 0.9, 1.0),
                roughness=0.8,
            )
        )
    
    def _add_robot(self):
        """添加人形机器人"""
        robot_config = self.config.get('robot', {})
        
        # 使用 Genesis 内置的 humanoid
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(
                color=(0.8, 0.6, 0.4, 1.0),
            )
        )
        
        return robot
    
    def _get_num_actions(self) -> int:
        """获取动作维度"""
        # humanoid 通常有 21 个自由度
        return 21
    
    def _get_num_obs(self) -> int:
        """获取观察维度
        
        观察空间包括:
        - 关节位置和速度 (21 * 2 = 42)
        - 根节点位置和姿态 (3 + 4 = 7)
        - 根节点线速度和角速度 (3 + 3 = 6)
        - 上一帧动作 (21)
        - 推力信息 (3)
        总计: ~80
        """
        return 80
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.push_timer = 0
        self.is_pushing = False
        self.last_actions = np.zeros((self.num_envs, self.num_actions))
        
        # 重置机器人位置
        init_pos = self.config.get('robot', {}).get('init_pos', [0, 0, 1.0])
        self.robot.set_pos(init_pos)
        
        # 构建场景
        self.scene.build()
        
        return self.get_obs()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        执行一步仿真
        
        Args:
            actions: 动作数组 [num_envs, num_actions]
            
        Returns:
            obs: 观察值
            rewards: 奖励
            dones: 是否结束
            info: 额外信息
        """
        # 应用动作到机器人
        self._apply_actions(actions)
        
        # 施加推力 (模拟被推倒)
        self._apply_push_force()
        
        # 仿真步进
        self.scene.step()
        
        # 获取观察
        obs = self.get_obs()
        
        # 计算奖励
        rewards = self._compute_rewards(actions)
        
        # 检查是否结束
        self.current_step += 1
        dones = self._check_termination()
        
        # 更新历史动作
        self.last_actions = actions.copy()
        
        info = {
            'step': self.current_step,
            'contact_forces': self.contact_forces,
        }
        
        return obs, rewards, dones, info
    
    def _apply_actions(self, actions: np.ndarray):
        """应用动作到机器人"""
        # 将动作转换为关节目标位置
        # humanoid 的动作空间是关节位置
        self.robot.control_dofs_position(actions)
    
    def _apply_push_force(self):
        """施加推倒力"""
        if not self.is_pushing:
            # 随机触发推力
            push_interval = self.config['env'].get('push_interval', 100)
            if self.current_step > 0 and self.current_step % push_interval == 0:
                self.is_pushing = True
                self.push_timer = int(self.push_duration / self.config['genesis']['dt'])
                # 随机推力方向 (水平方向)
                angle = np.random.uniform(0, 2 * np.pi)
                force_mag = np.random.uniform(*self.push_force_range)
                self.push_direction = np.array([
                    force_mag * np.cos(angle),
                    force_mag * np.sin(angle),
                    0
                ])
        
        if self.is_pushing and self.push_timer > 0:
            # 施加力到躯干
            self.robot.apply_force(
                force=self.push_direction,
                pos=self.robot.get_pos()
            )
            self.push_timer -= 1
            
            if self.push_timer <= 0:
                self.is_pushing = False
    
    def get_obs(self) -> np.ndarray:
        """获取观察值"""
        obs_list = []
        
        # 关节状态
        qpos = self.robot.get_dofs_position()
        qvel = self.robot.get_dofs_velocity()
        obs_list.extend(qpos.cpu().numpy().flatten())
        obs_list.extend(qvel.cpu().numpy().flatten())
        
        # 根节点状态
        root_pos = self.robot.get_pos()
        root_quat = self.robot.get_quat()
        root_vel = self.robot.get_vel()
        root_ang_vel = self.robot.get_ang()
        
        obs_list.extend(root_pos.cpu().numpy().flatten())
        obs_list.extend(root_quat.cpu().numpy().flatten())
        obs_list.extend(root_vel.cpu().numpy().flatten())
        obs_list.extend(root_ang_vel.cpu().numpy().flatten())
        
        # 上一帧动作
        obs_list.extend(self.last_actions.flatten())
        
        # 推力信息 (方向和大小)
        push_info = self.push_direction if self.is_pushing else np.zeros(3)
        obs_list.extend(push_info)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _compute_rewards(self, actions: np.ndarray) -> float:
        """计算奖励"""
        rewards = 0.0
        reward_config = self.config['rewards']
        
        # 1. 存活奖励
        rewards += reward_config['survival']['weight']
        
        # 2. 冲击惩罚 - 关键奖励，鼓励减少冲击力
        impact_penalty = self._compute_impact_penalty()
        rewards += reward_config['impact']['weight'] * impact_penalty
        
        # 3. 三角形结构奖励 - 核心奖励
        triangle_reward = self._compute_triangle_reward()
        rewards += reward_config['triangle_structure']['weight'] * triangle_reward
        
        # 4. 关节限制惩罚
        joint_penalty = self._compute_joint_limit_penalty()
        rewards += reward_config['joint_limit']['weight'] * joint_penalty
        
        # 5. 姿态稳定性奖励
        orientation_reward = self._compute_orientation_reward()
        rewards += reward_config['orientation']['weight'] * orientation_reward
        
        # 6. 动作平滑性
        smoothness_penalty = np.mean((actions - self.last_actions) ** 2)
        rewards += reward_config['action_smoothness']['weight'] * smoothness_penalty
        
        # 7. 能量效率
        energy_penalty = np.mean(actions ** 2)
        rewards += reward_config['energy']['weight'] * energy_penalty
        
        return rewards
    
    def _compute_impact_penalty(self) -> float:
        """
        计算冲击惩罚
        
        检测身体关键部位 (头部、躯干) 与地面的接触力
        """
        penalty = 0.0
        
        # 获取接触力
        contacts = self.robot.get_contacts()
        
        for contact in contacts:
            force = np.linalg.norm(contact['force'])
            # 重点关注高冲击力
            if force > 100:  # 阈值
                penalty += force / 1000.0
        
        return min(penalty, 10.0)  # 限制最大惩罚
    
    def _compute_triangle_reward(self) -> float:
        """
        计算三角形结构奖励 - 核心奖励函数
        
        鼓励机器人在跌倒时形成"三角形"支撑结构，例如:
        - 双手 + 一脚 形成的三角形
        - 一脚 + 双手 形成的三角形
        - 手 + 手 + 脚 的稳定支撑
        
        三角形结构可以分散冲击力，减少关键部位损伤
        """
        reward = 0.0
        
        # 获取关键部位位置
        body_positions = self._get_key_body_positions()
        
        # 检测是否形成稳定的三角形支撑
        # 检查双手和双脚是否接触地面
        hands_on_ground = body_positions['hands_ground']
        feet_on_ground = body_positions['feet_ground']
        
        # 三角形支撑的形成
        if sum(hands_on_ground) >= 1 and sum(feet_on_ground) >= 2:
            # 形成至少 3 点支撑
            reward += 1.0
            
            # 检查三角形的几何稳定性
            # 更大的底面积 = 更好的稳定性
            base_area = self._compute_support_base_area(body_positions)
            reward += min(base_area * 0.5, 1.0)
        
        # 额外奖励: 头部和躯干远离地面
        torso_height = body_positions['torso_height']
        head_height = body_positions['head_height']
        
        # 如果头部和躯干保持较高位置，给予奖励
        if torso_height > 0.3 and head_height > 0.5:
            reward += 0.5
        
        return reward
    
    def _get_key_body_positions(self) -> Dict:
        """获取关键身体部位的位置和状态"""
        positions = {
            'hands_ground': [False, False],  # 左右手
            'feet_ground': [False, False],   # 左右脚
            'torso_height': 0.0,
            'head_height': 0.0,
        }
        
        # 获取各部位位置 (这里需要根据实际机器人模型调整)
        # 使用 Genesis 的链接位置
        try:
            # 假设这些是链接名称
            torso_pos = self.robot.get_link('torso').get_pos()
            head_pos = self.robot.get_link('head').get_pos()
            
            positions['torso_height'] = torso_pos[2].item()
            positions['head_height'] = head_pos[2].item()
            
            # 检查接触
            for i, link_name in enumerate(['hand_l', 'hand_r']):
                try:
                    link = self.robot.get_link(link_name)
                    if link.get_contacts():
                        positions['hands_ground'][i] = True
                except:
                    pass
            
            for i, link_name in enumerate(['foot_l', 'foot_r']):
                try:
                    link = self.robot.get_link(link_name)
                    if link.get_contacts():
                        positions['feet_ground'][i] = True
                except:
                    pass
                    
        except Exception as e:
            pass
        
        return positions
    
    def _compute_support_base_area(self, body_positions: Dict) -> float:
        """计算支撑底面积 (用于评估三角形稳定性)"""
        # 简化的底面积计算
        # 实际应根据接触点位置计算凸包面积
        return 0.5  # 简化值
    
    def _compute_joint_limit_penalty(self) -> float:
        """计算关节限制惩罚"""
        # 检查关节是否超出限制
        # 这里简化处理
        return 0.0
    
    def _compute_orientation_reward(self) -> float:
        """计算姿态奖励 - 鼓励保持躯干直立"""
        try:
            torso_quat = self.robot.get_link('torso').get_quat()
            # 计算与直立姿态的差异
            # 简化为检查 torso 的高度
            torso_pos = self.robot.get_link('torso').get_pos()
            height = torso_pos[2].item()
            return max(0, min(height, 1.0))
        except:
            return 0.0
    
    def _check_termination(self) -> bool:
        """检查是否终止回合"""
        # 时间限制
        if self.current_step >= self.episode_length:
            return True
        
        # 检查机器人是否完全倒地 (躯干接触地面)
        try:
            torso_pos = self.robot.get_link('torso').get_pos()
            if torso_pos[2].item() < 0.2:  # 躯干高度过低
                return True
        except:
            pass
        
        return False
    
    def set_push_params(self, force_range: List[float], duration: float):
        """设置推力参数 (用于课程学习)"""
        self.push_force_range = force_range
        self.push_duration = duration
    
    def close(self):
        """关闭环境"""
        gs.destroy()
