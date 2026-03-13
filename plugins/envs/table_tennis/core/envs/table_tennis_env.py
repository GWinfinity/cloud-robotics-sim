"""
Table Tennis Environment for Unitree G1

统一全身控制环境：手臂击球 + 腿部步法
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional, List

from .ball_physics import TableTennisBall, BallTrajectoryPredictor
from .table import TableTennisTable, Racket


class TableTennisEnv:
    """
    乒乓球环境
    
    核心特性:
    - Unitree G1 全身控制 (29 DOF)
    - 双预测器集成
    - 预测增强奖励
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
        
        # 创建球桌
        self.table = TableTennisTable(self.scene, config=config.get('table', {}))
        
        # 创建机器人 (Unitree G1)
        self.robot = self._create_robot()
        
        # 创建乒乓球
        self.ball = self._create_ball()
        
        # 创建球拍
        self.racket = Racket(self.scene)
        
        # 预测器
        self.physics_predictor = BallTrajectoryPredictor(
            config['ball'],
            dt=config['genesis']['dt']
        )
        
        # 状态维度
        self.state_dim = self._get_state_dim()
        self.action_dim = 29  # Unitree G1
        
        # 回合统计
        self.episode_stats = {
            'hits': 0,
            'successful_returns': 0,
            'consecutive_hits': 0,
            'max_consecutive': 0,
            'total_reward': 0.0
        }
        
        self.current_step = 0
        self.max_steps = config['env']['episode_length']
        
    def _create_scene(self):
        """创建场景"""
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 3, 2),
                camera_lookat=(0, 0, 1),
                camera_fov=60,
                res=(1280, 720),
                max_FPS=60,
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless,
        )
    
    def _create_robot(self):
        """创建Unitree G1机器人"""
        # 使用Genesis内置humanoid作为近似
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(
                color=(0.7, 0.7, 0.8, 1.0),
            )
        )
        
        # 设置初始位置
        init_pos = self.config['robot']['init_pos']
        robot.set_pos(init_pos)
        
        return robot
    
    def _create_ball(self):
        """创建乒乓球"""
        ball_config = self.config['ball']
        return TableTennisBall(
            scene=self.scene,
            init_pos=np.array([1.0, 0.0, 1.0]),
            **ball_config
        )
    
    def _get_state_dim(self) -> int:
        """获取状态维度"""
        # 球状态(6) + 预测(60) + 机器人本体(40) = 106
        return 106
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {
            'hits': 0,
            'successful_returns': 0,
            'consecutive_hits': 0,
            'max_consecutive': 0,
            'total_reward': 0.0
        }
        
        # 重置机器人
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 发球
        self._serve_ball()
        
        # 构建场景
        self.scene.build()
        
        return self.get_observation()
    
    def _serve_ball(self):
        """发球"""
        serve_config = self.config['env']['serve']
        
        # 发球位置 (对方半场)
        serve_pos = np.array([
            1.0 + np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.5, 0.5),
            self.config['table']['height'] + 0.3 + np.random.uniform(0, 0.2)
        ])
        
        # 发球速度 (朝向己方)
        speed = np.random.uniform(*serve_config['velocity_range'])
        angle = np.random.uniform(-0.3, 0.3)
        
        velocity = np.array([
            -speed * np.cos(angle),
            speed * np.sin(angle) * 0.3,
            -speed * 0.1
        ])
        
        # 旋转
        spin = np.random.uniform(*serve_config['spin_range'])
        angular_vel = np.array([0, 0, spin])
        
        self.ball.reset(serve_pos, velocity, angular_vel)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 全身关节目标位置 [29]
            
        Returns:
            obs: 观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 应用动作
        self._apply_action(action)
        
        # 更新球拍位置
        self._update_racket()
        
        # 检查击球
        hit_info = self._check_hit()
        
        # 应用球物理
        self.ball.apply_physics(self.config['genesis']['dt'])
        
        # 仿真步进
        self.scene.step()
        
        # 获取观测
        obs = self.get_observation()
        
        # 计算奖励
        reward = self._compute_reward(action, hit_info)
        
        # 更新统计
        if hit_info['hit']:
            self.episode_stats['hits'] += 1
            self.episode_stats['consecutive_hits'] += 1
            self.episode_stats['max_consecutive'] = max(
                self.episode_stats['max_consecutive'],
                self.episode_stats['consecutive_hits']
            )
        
        self.episode_stats['total_reward'] += reward
        
        # 检查终止
        self.current_step += 1
        done = self._check_termination(hit_info)
        
        info = {
            'hit': hit_info['hit'],
            'hit_speed': hit_info.get('speed', 0.0),
            'stats': self.episode_stats.copy()
        }
        
        # 重置球如果回合结束
        if done or self._check_ball_reset():
            if not done:
                self._serve_ball()
                self.episode_stats['consecutive_hits'] = 0
        
        return obs, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作到机器人"""
        # 缩放动作
        action_scale = self.config['env']['action'].get('delta_scale', 0.1)
        action = np.clip(action * action_scale, -1, 1)
        
        # 设置关节位置
        self.robot.control_dofs_position(action)
    
    def _update_racket(self):
        """更新球拍位置和速度"""
        try:
            # 获取手腕位置和速度
            hand_pos = self.robot.get_link('hand_r').get_pos()
            hand_vel = self.robot.get_link('hand_r').get_vel()
            
            # 简化的拍面法向量
            hand_quat = self.robot.get_link('hand_r').get_quat()
            # 根据四元数计算法向量 (简化)
            normal = np.array([0, 0, 1])
            
            self.racket.update_transform(
                hand_pos.cpu().numpy(),
                hand_vel.cpu().numpy(),
                normal
            )
        except:
            pass
    
    def _check_hit(self) -> Dict:
        """检查是否击中球"""
        hit_info = {'hit': False}
        
        if self.racket.check_collision(self.ball.position, self.ball.radius):
            # 计算击球效果
            racket_speed = np.linalg.norm(self.racket.velocity)
            
            # 新速度 (简化模型)
            new_vel = self.racket.normal * racket_speed * 1.5
            new_vel[2] = abs(new_vel[2]) + 2.0  # 向上分量
            
            # 应用击球
            self.ball.velocity = new_vel
            self.ball.angular_velocity *= 0.5
            
            hit_info = {
                'hit': True,
                'speed': racket_speed,
                'position': self.ball.position.copy()
            }
        
        return hit_info
    
    def get_observation(self) -> np.ndarray:
        """获取观测"""
        # 1. 球状态 (6维)
        ball_state = self.ball.get_state()[:6]  # 位置+速度
        
        # 2. 预测状态 (60维)
        ball_history = self.ball.get_trajectory_history(n=5)
        # 这里应该使用学习预测器，但简化处理
        pred_state = np.zeros(60)
        
        # 3. 机器人本体感觉 (40维)
        joint_pos = self.robot.get_dofs_position().cpu().numpy()[:20]
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()[:20]
        
        # 填充到固定维度
        joint_pos = np.pad(joint_pos, (0, 20 - len(joint_pos)), 'constant')
        joint_vel = np.pad(joint_vel, (0, 20 - len(joint_vel)), 'constant')
        
        proprio = np.concatenate([joint_pos, joint_vel])
        
        # 组合观测
        obs = np.concatenate([ball_state, pred_state, proprio])
        
        # 填充到106维
        if len(obs) < 106:
            obs = np.pad(obs, (0, 106 - len(obs)), 'constant')
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray, hit_info: Dict) -> float:
        """
        计算奖励
        
        使用预测增强的奖励设计
        """
        reward = 0.0
        reward_config = self.config['rewards']
        
        # 1. 击球奖励
        if hit_info['hit']:
            reward += reward_config['hit']['weight']
            
            # 击球速度奖励
            hit_speed = hit_info.get('speed', 0.0)
            speed_reward = min(hit_speed / 5.0, 1.0) * reward_config['hit']['bonus_speed']
            reward += speed_reward
        
        # 2. 预测增强奖励 (核心)
        # 使用物理预测器计算未来的奖励
        robot_pos = self.robot.get_pos().cpu().numpy()
        prediction = self.physics_predictor.get_prediction_for_reward(
            self.ball.get_state()[:6],
            robot_pos
        )
        
        # 击球概率奖励
        hit_prob = prediction['hit_probability']
        reward += reward_config['predictive']['weight'] * hit_prob
        
        # 3. 步法奖励
        if prediction['contact_point'] is not None:
            contact = prediction['contact_point']
            distance = np.linalg.norm(contact[:2] - robot_pos[:2])
            footwork_reward = np.exp(-distance) * reward_config['footwork']['weight']
            reward += footwork_reward
        
        # 4. 姿态奖励
        torso_height = robot_pos[2]
        if torso_height > 0.6:
            reward += reward_config['posture']['weight'] * reward_config['posture']['upright_bonus']
        
        # 5. 能量惩罚
        energy_penalty = np.sum(action ** 2) * reward_config['energy_penalty']
        reward += energy_penalty
        
        # 6. 动作平滑性
        # 这里应该存储上一帧动作
        
        return reward
    
    def _check_termination(self, hit_info: Dict) -> bool:
        """检查是否结束回合"""
        # 时间限制
        if self.current_step >= self.max_steps:
            return True
        
        # 检查机器人是否倒地
        try:
            torso_pos = self.robot.get_pos().cpu().numpy()
            if torso_pos[2] < 0.4:
                return True
        except:
            pass
        
        return False
    
    def _check_ball_reset(self) -> bool:
        """检查是否需要重置球"""
        # 球落地
        if self.ball.position[2] <= 0:
            return True
        
        # 球出界
        if abs(self.ball.position[0]) > 3 or abs(self.ball.position[1]) > 2:
            return True
        
        return False
    
    def close(self):
        """关闭环境"""
        gs.destroy()
