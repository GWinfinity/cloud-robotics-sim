"""
人形机器人羽毛球环境

基于 Genesis 的全身控制环境，实现:
- 下肢步法控制
- 上肢挥拍控制
- 羽毛球击打任务
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional, List
import yaml

from .shuttlecock import Shuttlecock, BadmintonCourt


class BadmintonEnv:
    """
    人形机器人羽毛球环境
    
    状态空间:
    - 机器人关节位置和速度
    - 机器人根节点状态
    - 羽毛球位置和速度
    - 羽毛球历史轨迹 (用于预测)
    - 场地信息
    
    动作空间:
    - 全身关节目标位置
    - 下肢: 6 DOF (髋、膝、踝 x 2)
    - 上肢: 6 DOF (肩、肘、腕 x 2)
    """
    
    def __init__(
        self,
        config_path: str = None,
        num_envs: int = 1,
        headless: bool = False,
        curriculum_stage: int = 1
    ):
        """
        初始化环境
        
        Args:
            config_path: 配置文件路径
            num_envs: 并行环境数量
            headless: 是否无头模式
            curriculum_stage: 当前课程阶段 (1, 2, 3)
        """
        self.num_envs = num_envs
        self.headless = headless
        self.curriculum_stage = curriculum_stage
        self.config = self._load_config(config_path)
        
        # 初始化 Genesis
        gs.init(backend=gs.backends.CUDA)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 创建场地
        self.court = BadmintonCourt(self.scene, self.config['court'])
        
        # 创建机器人
        self.robot = self._create_robot()
        
        # 创建羽毛球
        self.shuttlecock = self._create_shuttlecock()
        
        # 动作和观察空间
        self.num_actions = self._get_num_actions()
        self.num_obs = self._get_num_obs()
        
        # 环境参数
        self.episode_length = self.config['env']['episode_length']
        self.current_step = 0
        
        # 回合统计
        self.episode_stats = {
            'hits': 0,
            'consecutive_hits': 0,
            'max_consecutive_hits': 0,
            'successful_landings': 0,
            'total_reward': 0.0
        }
        
        # 上一帧动作 (用于平滑性)
        self.last_actions = np.zeros((num_envs, self.num_actions))
        
        # 球拍信息
        self.racket_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'normal': np.array([1, 0, 0]),  # 默认朝右
        }
        
        # 目标击球位置 (由课程阶段控制)
        self.target_hit_position = np.array([0, 0, 1.5])
        
        # 冻结的关节 (课程学习使用)
        self.frozen_joints = self._get_frozen_joints()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'genesis': {'dt': 0.01, 'substeps': 10},
            'scene': {'show_viewer': True},
            'court': {
                'bounds': {'x_min': -6.7, 'x_max': 6.7, 'y_min': -2.59, 'y_max': 2.59},
                'net_pos': [0, 0, 0.77]
            },
            'env': {'episode_length': 1000, 'num_envs': 1},
            'rewards': {
                'hit_shuttlecock': {'weight': 10.0},
                'landing': {'weight': 5.0},
                'ball_speed': {'weight': 1.0},
                'rally': {'weight': 2.0},
                'balance': {'weight': 1.0},
                'energy': {'weight': -0.0005}
            }
        }
    
    def _create_scene(self):
        """创建 Genesis 场景"""
        scene_config = self.config.get('scene', {})
        viewer_config = scene_config.get('viewer', {})
        
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=viewer_config.get('camera_pos', (6, 6, 4)),
                camera_lookat=viewer_config.get('camera_lookat', (0, 0, 1)),
                camera_fov=viewer_config.get('camera_fov', 60),
                res=viewer_config.get('res', (1280, 720)),
                max_FPS=viewer_config.get('max_FPS', 60),
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless and scene_config.get('show_viewer', True),
        )
    
    def _create_robot(self):
        """创建人形机器人 (带球拍)"""
        robot_config = self.config.get('robot', {})
        init_pos = robot_config.get('init_pos', [-3.0, 0.0, 1.0])
        
        # 使用 Genesis 内置的 humanoid
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(
                color=(0.6, 0.7, 0.8, 1.0),
            )
        )
        
        # TODO: 在右手添加球拍 (作为附加的链接或视觉元素)
        # 简化处理: 假设手腕就是击球点
        
        return robot
    
    def _create_shuttlecock(self):
        """创建羽毛球"""
        shuttle_config = self.config.get('shuttlecock', {})
        init_pos = np.array([3.0, 0.0, 2.0])  # 对方发球位置
        
        return Shuttlecock(
            scene=self.scene,
            init_pos=init_pos,
            **shuttle_config
        )
    
    def _get_num_actions(self) -> int:
        """获取动作维度"""
        # 全身关节控制 (简化: 12个主要关节)
        # 下肢: 6 (髋x2, 膝x2, 踝x2)
        # 上肢: 6 (肩x2, 肘x2, 腕x2)
        return 12
    
    def _get_num_obs(self) -> int:
        """获取观察维度"""
        # 机器人状态: 关节位置(12) + 速度(12) + 根节点位置(3) + 姿态(4) + 速度(6) = 37
        # 羽毛球状态: 位置(3) + 速度(3) + 速度大小(1) = 7
        # 历史轨迹: 5个点 x 3维 = 15
        # 目标信息: 3
        # 上一动作: 12
        # 总计: ~75
        return 75
    
    def _get_frozen_joints(self) -> List[str]:
        """获取当前课程阶段冻结的关节"""
        frozen = []
        
        if self.curriculum_stage == 1:
            # 第一阶段: 冻结上肢，只训练步法
            frozen = self.config.get('robot', {}).get('upper_body_joints', [])
        elif self.curriculum_stage == 2:
            # 第二阶段: 冻结下肢，只训练挥拍
            # 实际上第二阶段应该同时训练，但重点在挥拍
            frozen = []  # 不解冻
        
        return frozen
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {
            'hits': 0,
            'consecutive_hits': 0,
            'max_consecutive_hits': 0,
            'successful_landings': 0,
            'total_reward': 0.0
        }
        self.last_actions = np.zeros((self.num_envs, self.num_actions))
        
        # 重置机器人
        init_pos = self.config.get('robot', {}).get('init_pos', [-3.0, 0.0, 1.0])
        self.robot.set_pos(init_pos)
        
        # 重置羽毛球 (发球)
        self._serve_shuttlecock()
        
        # 构建场景
        self.scene.build()
        
        return self.get_obs()
    
    def _serve_shuttlecock(self):
        """发球"""
        # 从对方场地发过来
        serve_config = self.config['env']['serve']
        
        # 发球位置 (对方场地)
        serve_pos = np.array([
            3.0 + np.random.uniform(-0.5, 0.5),  # 对方半场
            np.random.uniform(-1.0, 1.0),
            2.0 + np.random.uniform(0, 0.5)
        ])
        
        # 发球速度 (朝向己方)
        speed = np.random.uniform(*serve_config['initial_velocity_range'])
        angle = np.random.uniform(*serve_config['initial_angle_range'])
        
        velocity = np.array([
            -speed * np.cos(angle),  # 向负x方向
            speed * np.sin(angle) * np.random.uniform(-0.3, 0.3),
            -speed * 0.2  # 略微向下
        ])
        
        self.shuttlecock.reset(serve_pos, velocity)
    
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
        # 应用动作 (考虑冻结的关节)
        self._apply_actions(actions)
        
        # 更新球拍状态
        self._update_racket_state()
        
        # 检查击球
        hit_info = self._check_hit()
        
        # 应用羽毛球空气动力学
        self.shuttlecock.apply_aerodynamics()
        self.shuttlecock.record_trajectory()
        
        # 仿真步进
        self.scene.step()
        
        # 获取观察
        obs = self.get_obs()
        
        # 计算奖励
        rewards = self._compute_rewards(actions, hit_info)
        
        # 检查是否结束
        self.current_step += 1
        dones = self._check_termination(hit_info)
        
        # 更新统计
        if hit_info['hit']:
            self.episode_stats['hits'] += 1
            self.episode_stats['consecutive_hits'] += 1
            self.episode_stats['max_consecutive_hits'] = max(
                self.episode_stats['max_consecutive_hits'],
                self.episode_stats['consecutive_hits']
            )
        
        self.episode_stats['total_reward'] += rewards
        
        # 更新上一帧动作
        self.last_actions = actions.copy()
        
        info = {
            'step': self.current_step,
            'hit': hit_info['hit'],
            'hit_speed': hit_info.get('speed', 0.0),
            'shuttle_state': self.shuttlecock.get_state(),
            'episode_stats': self.episode_stats.copy()
        }
        
        # 如果回合结束或球落地，重置羽毛球
        if dones or self._check_shuttle_landing():
            if not dones:
                self._serve_shuttlecock()
                self.episode_stats['consecutive_hits'] = 0  # 重置连续击球
        
        return obs, rewards, dones, info
    
    def _apply_actions(self, actions: np.ndarray):
        """应用动作到机器人"""
        # 考虑冻结的关节
        full_actions = actions.copy()
        
        # 如果有冻结的关节，将对应动作设为0或默认值
        # 这里简化处理
        
        self.robot.control_dofs_position(full_actions)
    
    def _update_racket_state(self):
        """更新球拍状态 (位置和速度)"""
        # 简化: 假设球拍在右手腕位置
        try:
            # 获取手腕位置和速度
            wrist_pos = self.robot.get_link('hand_r').get_pos()
            wrist_vel = self.robot.get_link('hand_r').get_vel()
            
            self.racket_state['position'] = wrist_pos.cpu().numpy()
            self.racket_state['velocity'] = wrist_vel.cpu().numpy()
            
            # 球拍法向量 (根据手腕姿态计算)
            # 简化: 假设球拍面朝向速度方向
            if np.linalg.norm(wrist_vel) > 0.1:
                self.racket_state['normal'] = wrist_vel.cpu().numpy()
                self.racket_state['normal'] /= np.linalg.norm(self.racket_state['normal'])
                
        except Exception as e:
            # 如果获取失败，使用默认值
            pass
    
    def _check_hit(self) -> Dict:
        """检查是否击中羽毛球"""
        hit, impact_speed = self.shuttlecock.check_racket_collision(
            self.racket_state['position'],
            self.racket_state['velocity'],
            self.racket_state['normal']
        )
        
        hit_info = {'hit': False, 'speed': 0.0}
        
        if hit:
            # 计算击球方向 (朝向对方场地)
            hit_direction = np.array([1, 0, 0.1])  # 向正x方向，略微向上
            hit_direction /= np.linalg.norm(hit_direction)
            
            # 击球速度
            hit_speed = max(10.0, impact_speed * 1.5)  # 最小10m/s
            
            # 应用击球
            self.shuttlecock.apply_hit(
                hit_direction,
                hit_speed,
                self.racket_state['position']
            )
            
            hit_info = {'hit': True, 'speed': hit_speed}
        
        return hit_info
    
    def _check_shuttle_landing(self) -> bool:
        """检查羽毛球是否落地"""
        return self.shuttlecock.position[2] <= 0.05
    
    def get_obs(self) -> np.ndarray:
        """获取观察值"""
        obs_list = []
        
        # 1. 机器人状态
        # 关节位置
        qpos = self.robot.get_dofs_position().cpu().numpy().flatten()
        obs_list.extend(qpos[:self.num_actions])  # 只取控制的关节
        
        # 关节速度
        qvel = self.robot.get_dofs_velocity().cpu().numpy().flatten()
        obs_list.extend(qvel[:self.num_actions])
        
        # 根节点状态
        root_pos = self.robot.get_pos().cpu().numpy().flatten()
        root_quat = self.robot.get_quat().cpu().numpy().flatten()
        root_vel = self.robot.get_vel().cpu().numpy().flatten()
        root_ang_vel = self.robot.get_ang().cpu().numpy().flatten()
        
        obs_list.extend(root_pos)
        obs_list.extend(root_quat)
        obs_list.extend(root_vel)
        obs_list.extend(root_ang_vel)
        
        # 2. 羽毛球状态
        shuttle_state = self.shuttlecock.get_state()
        obs_list.extend(shuttle_state)
        
        # 3. 历史轨迹
        trajectory = self.shuttlecock.get_trajectory_for_ekf()
        if len(trajectory) > 0:
            # 展平最近5个点的位置
            for i in range(min(5, len(trajectory))):
                obs_list.extend(trajectory[-(i+1)])
            # 填充
            while len(obs_list) < 37 + 7 + 15:
                obs_list.append(0.0)
        else:
            obs_list.extend([0.0] * 15)
        
        # 4. 目标信息
        obs_list.extend(self.target_hit_position)
        
        # 5. 上一帧动作
        obs_list.extend(self.last_actions.flatten())
        
        return np.array(obs_list, dtype=np.float32)
    
    def _compute_rewards(self, actions: np.ndarray, hit_info: Dict) -> float:
        """计算奖励"""
        reward = 0.0
        reward_config = self.config['rewards']
        
        # 根据课程阶段使用不同的奖励
        if self.curriculum_stage == 1:
            reward = self._compute_stage1_rewards(actions)
        elif self.curriculum_stage == 2:
            reward = self._compute_stage2_rewards(actions, hit_info)
        else:
            reward = self._compute_stage3_rewards(actions, hit_info)
        
        return reward
    
    def _compute_stage1_rewards(self, actions: np.ndarray) -> float:
        """第一阶段: 步法奖励"""
        reward = 0.0
        
        # 位置到达奖励 (移动到最优击球位置)
        robot_pos = self.robot.get_pos().cpu().numpy()
        distance_to_optimal = np.linalg.norm(robot_pos[:2] - self.target_hit_position[:2])
        position_reward = np.exp(-distance_to_optimal)
        reward += self.config['rewards'].get('position', {}).get('weight', 1.0) * position_reward
        
        # 平衡奖励
        torso_height = robot_pos[2]
        balance_reward = 1.0 if torso_height > 0.8 else 0.0
        reward += self.config['rewards'].get('balance', {}).get('weight', 0.5) * balance_reward
        
        # 能量惩罚
        energy_penalty = np.mean(actions ** 2)
        reward += self.config['rewards'].get('energy', {}).get('weight', -0.001) * energy_penalty
        
        return reward
    
    def _compute_stage2_rewards(self, actions: np.ndarray, hit_info: Dict) -> float:
        """第二阶段: 挥拍奖励"""
        reward = 0.0
        
        # 击中奖励
        if hit_info['hit']:
            reward += self.config['rewards'].get('hit_shuttlecock', {}).get('weight', 5.0)
            
            # 击球速度奖励
            hit_speed = hit_info.get('speed', 0.0)
            target_speed = self.config['rewards'].get('ball_speed', {}).get('target_speed', 15.0)
            speed_reward = min(hit_speed / target_speed, 1.0)
            reward += self.config['rewards'].get('ball_speed', {}).get('weight', 0.5) * speed_reward
        
        # 动作平滑性
        smoothness = np.mean((actions - self.last_actions) ** 2)
        reward += self.config['rewards'].get('smoothness', {}).get('weight', -0.01) * smoothness
        
        return reward
    
    def _compute_stage3_rewards(self, actions: np.ndarray, hit_info: Dict) -> float:
        """第三阶段: 全身协调奖励"""
        reward = 0.0
        
        # 击中奖励
        if hit_info['hit']:
            reward += self.config['rewards'].get('hit_shuttlecock', {}).get('weight', 10.0)
            
            # 击球速度奖励
            hit_speed = hit_info.get('speed', 0.0)
            target_speed = self.config['rewards'].get('ball_speed', {}).get('target_speed', 15.0)
            speed_reward = min(hit_speed / target_speed, 2.0)
            reward += self.config['rewards'].get('ball_speed', {}).get('weight', 1.0) * speed_reward
            
            # 落点奖励
            landing_pred = self.shuttlecock.predict_landing()
            if landing_pred is not None:
                ideal_landing = np.array(self.config['rewards'].get('landing', {}).get('ideal_landing', [3, 0, 0]))
                landing_dist = np.linalg.norm(landing_pred[:2] - ideal_landing[:2])
                landing_reward = np.exp(-landing_dist / 2.0)
                reward += self.config['rewards'].get('landing', {}).get('weight', 5.0) * landing_reward
            
            # 连续击球奖励
            consecutive_bonus = self.episode_stats['consecutive_hits'] * \
                               self.config['rewards'].get('rally', {}).get('consecutive_bonus', 0.5)
            reward += self.config['rewards'].get('rally', {}).get('weight', 2.0) + consecutive_bonus
        
        # 平衡奖励
        robot_pos = self.robot.get_pos().cpu().numpy()
        torso_height = robot_pos[2]
        balance_reward = 1.0 if torso_height > 0.8 else max(0, torso_height)
        reward += self.config['rewards'].get('balance', {}).get('weight', 1.0) * balance_reward
        
        # 能量效率
        energy_penalty = np.mean(actions ** 2)
        reward += self.config['rewards'].get('energy', {}).get('weight', -0.0005) * energy_penalty
        
        # 动作平滑性
        smoothness = np.mean((actions - self.last_actions) ** 2)
        reward += self.config['rewards'].get('smoothness', {}).get('weight', -0.01) * smoothness
        
        return reward
    
    def _check_termination(self, hit_info: Dict) -> bool:
        """检查是否结束回合"""
        # 时间限制
        if self.current_step >= self.episode_length:
            return True
        
        # 机器人倒地
        try:
            torso_pos = self.robot.get_pos().cpu().numpy()
            if torso_pos[2] < 0.5:  # 躯干高度过低
                return True
        except:
            pass
        
        # 羽毛球出界 (可选)
        # if not self.shuttlecock.is_in_court(self.config['court']['bounds']):
        #     return True
        
        return False
    
    def set_curriculum_stage(self, stage: int):
        """设置课程阶段"""
        self.curriculum_stage = stage
        self.frozen_joints = self._get_frozen_joints()
        print(f"Curriculum stage set to {stage}")
    
    def close(self):
        """关闭环境"""
        gs.destroy()
