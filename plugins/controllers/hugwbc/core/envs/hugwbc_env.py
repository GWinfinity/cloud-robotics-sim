"""
HugWBC Environment for Genesis

基于论文 "HugWBC: A Unified and General Humanoid Whole-Body Controller for Versatile Locomotion"
适配到 Genesis 物理引擎
"""

import os
import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional, List
import yaml
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    LOCO = "h1_loco"           # 平地行走
    STAIRS = "h1_stairs"       # 上下楼梯
    TERRAIN = "h1_terrain"     # 复杂地形
    INTERACTION = "h1_int"     # 交互任务


class HugWBCEnv:
    """
    HugWBC 人形机器人全身控制环境 (Genesis 版本)
    """
    
    def __init__(
        self,
        task: str = "h1_loco",
        config_path: Optional[str] = None,
        num_envs: int = 1,
        headless: bool = False,
        device: str = 'cuda'
    ):
        # 目前只支持单环境
        if num_envs != 1:
            print(f"Warning: num_envs={num_envs} not supported yet, using num_envs=1")
            num_envs = 1
        
        self.task = task
        self.num_envs = num_envs
        self.headless = headless
        self.device = device
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化 Genesis
        backend = gs.cuda if device == 'cuda' and torch.cuda.is_available() else gs.cpu
        gs.init(backend=backend)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 创建地形
        self.terrain = self._create_terrain()
        
        # 创建机器人
        self.robot = self._create_robot()
        
        # 构建场景
        self.scene.build()
        
        # 构建后设置初始位置
        init_pos = self.config.get('robot', {}).get('init_pos', [0, 0, 1.0])
        self.robot.set_pos(init_pos)
        
        # 环境参数
        self.episode_length = self.config['env']['episode_length']
        self.current_step = 0
        
        # 获取机器人 DOF 信息
        self.n_dofs = self.robot.n_dofs
        self.n_qs = self.robot.n_qs
        
        # 动作和观察空间
        self.num_actions = self.n_dofs
        self.num_obs = self._get_num_obs()
        self.num_privileged_obs = self._get_num_privileged_obs()
        
        # 命令目标 (vx, vy, yaw_rate)
        self.commands = np.zeros((num_envs, 3))
        self.command_ranges = self.config['env']['command_ranges']
        
        # 步态相位
        self.gait_phase = np.zeros(num_envs)
        self.gait_frequency = self.config['env'].get('gait_frequency', 1.25)
        
        # 历史动作
        self.last_actions = np.zeros((num_envs, self.num_actions))
        self.last_last_actions = np.zeros((num_envs, self.num_actions))
        
        # 默认关节位置
        self.default_joint_pos = self._get_default_joint_pos()
        
        # 域随机化参数
        self.domain_rand_params = self._init_domain_rand_params()
        
        # 统计信息
        self.episode_stats = {
            'episode_length': np.zeros(num_envs),
            'total_reward': np.zeros(num_envs),
            'command_tracking_error': np.zeros(num_envs),
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'genesis': {
                'dt': 0.005,
                'substeps': 20,
            },
            'scene': {
                'show_viewer': True,
                'camera_pos': (3.0, 3.0, 2.0),
                'camera_lookat': (0.0, 0.0, 0.5),
                'camera_fov': 45,
            },
            'env': {
                'episode_length': 1000,
                'command_ranges': {
                    'lin_vel_x': [-1.0, 2.0],
                    'lin_vel_y': [-0.5, 0.5],
                    'ang_vel_yaw': [-1.0, 1.0],
                },
                'gait_frequency': 1.25,
                'termination_height': 0.3,
            },
            'rewards': {
                'tracking_lin_vel': {'weight': 2.0},
                'tracking_ang_vel': {'weight': 0.5},
                'lin_vel_z': {'weight': -2.0},
                'ang_vel_xy': {'weight': -0.05},
                'orientation': {'weight': -1.0},
                'torques': {'weight': -0.0001},
                'dof_acc': {'weight': -2.5e-7},
                'action_rate': {'weight': -0.01},
                'feet_air_time': {'weight': 1.0},
                'collision': {'weight': -1.0},
                'dof_pos_limits': {'weight': -10.0},
            },
            'normalization': {
                'clip_observations': 100.0,
                'clip_actions': 100.0,
            },
            'control': {
                'action_scale': 0.25,
            }
        }
    
    def _create_scene(self):
        """创建 Genesis 场景"""
        scene_config = self.config.get('scene', {})
        
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=scene_config.get('camera_pos', (3.0, 3.0, 2.0)),
                camera_lookat=scene_config.get('camera_lookat', (0.0, 0.0, 0.5)),
                camera_fov=scene_config.get('camera_fov', 45),
                res=scene_config.get('res', (1280, 720)),
                max_FPS=scene_config.get('max_FPS', 60),
            ) if not self.headless else None,
            sim_options=gs.options.SimOptions(
                dt=self.config['genesis']['dt'],
                substeps=self.config['genesis']['substeps'],
            ),
            show_viewer=not self.headless and self.config['scene'].get('show_viewer', True),
        )
    
    def _create_terrain(self):
        """创建地形"""
        ground = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                color=(0.9, 0.9, 0.9, 1.0),
                roughness=0.8,
            )
        )
        
        if self.task == "h1_stairs":
            self._add_stairs()
        elif self.task == "h1_terrain":
            self._add_rough_terrain()
        
        return ground
    
    def _add_stairs(self):
        """添加楼梯地形"""
        stair_config = self.config.get('stairs', {})
        num_steps = stair_config.get('num_steps', 10)
        step_height = stair_config.get('step_height', 0.15)
        step_depth = stair_config.get('step_depth', 0.3)
        
        for i in range(num_steps):
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(step_depth, 1.0, step_height),
                    pos=(i * step_depth, 0, i * step_height + step_height / 2)
                ),
                surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7, 1.0))
            )
    
    def _add_rough_terrain(self):
        """添加复杂地形"""
        np.random.seed(42)
        for i in range(10):
            x = np.random.uniform(2, 10)
            y = np.random.uniform(-2, 2)
            height = np.random.uniform(0.05, 0.2)
            size = np.random.uniform(0.3, 0.8)
            
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(size, size, height),
                    pos=(x, y, height / 2)
                ),
                surface=gs.surfaces.Default(color=(0.6, 0.5, 0.4, 1.0))
            )
    
    def _create_robot(self):
        """创建 H1 机器人"""
        humanoid_path = os.path.join(gs.__path__[0], 'assets', 'xml', 'humanoid.xml')
        
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file=humanoid_path),
            surface=gs.surfaces.Default(
                color=(0.8, 0.6, 0.4, 1.0),
            )
        )
        
        return robot
    
    def _get_num_obs(self) -> int:
        """获取观察空间维度"""
        return self.n_dofs * 3 + 3 + 3 + 3 + 2
    
    def _get_num_privileged_obs(self) -> int:
        """获取特权观察维度"""
        return self.num_obs + 10
    
    def _get_default_joint_pos(self) -> np.ndarray:
        """获取默认关节位置"""
        return self.robot.init_qpos[:self.n_dofs]
    
    def _init_domain_rand_params(self) -> Dict:
        """初始化域随机化参数"""
        return {
            'friction': np.ones(self.num_envs),
            'mass': np.ones(self.num_envs),
            'com_offset': np.zeros((self.num_envs, 3)),
        }
    
    def reset(self, env_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """重置环境"""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        self.current_step = 0
        
        for i in env_ids:
            self.episode_stats['episode_length'][i] = 0
            self.episode_stats['total_reward'][i] = 0
            self.episode_stats['command_tracking_error'][i] = 0
        
        # 重置机器人位置
        init_pos = self.config.get('robot', {}).get('init_pos', [0, 0, 1.0])
        if hasattr(self.robot, 'set_pos'):
            self.robot.set_pos(init_pos)
        
        # 重置关节位置
        if hasattr(self.robot, 'set_dofs_position'):
            self.robot.set_dofs_position(self.default_joint_pos)
        
        # 重置动作历史
        self.last_actions[env_ids] = 0
        self.last_last_actions[env_ids] = 0
        
        # 重置步态相位
        self.gait_phase[env_ids] = 0
        
        # 采样新命令
        self._resample_commands(env_ids)
        
        # 获取观察 - 目前只支持 num_envs=1
        obs = self.get_observations()
        privileged_obs = self.get_privileged_observations()
        
        return obs, privileged_obs
    
    def _resample_commands(self, env_ids: List[int]):
        """重新采样速度命令"""
        ranges = self.command_ranges
        
        for i in env_ids:
            self.commands[i, 0] = np.random.uniform(*ranges['lin_vel_x'])
            self.commands[i, 1] = np.random.uniform(*ranges['lin_vel_y'])
            self.commands[i, 2] = np.random.uniform(*ranges['ang_vel_yaw'])
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """执行一步仿真"""
        # 裁剪动作
        actions = np.clip(actions, -1, 1)
        
        # 确保动作形状正确
        if self.num_envs == 1 and len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
        
        # 应用动作到机器人
        self._apply_actions(actions)
        
        # 仿真步进
        self.scene.step()
        
        # 更新步态相位
        self.gait_phase += self.gait_frequency * self.config['genesis']['dt']
        self.gait_phase %= 1.0
        
        # 获取观察 - 目前只支持 num_envs=1
        obs = self.get_observations()
        privileged_obs = self.get_privileged_observations()
        
        # 计算奖励
        rewards = self._compute_rewards(actions)
        
        # 检查终止
        self.current_step += 1
        dones = self._check_termination()
        
        # 更新统计
        for i in range(self.num_envs):
            self.episode_stats['episode_length'][i] += 1
            self.episode_stats['total_reward'][i] += rewards[i]
        
        # 更新历史动作
        self.last_last_actions = self.last_actions.copy()
        self.last_actions = actions.copy()
        
        # 重置完成的 episode
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            obs, privileged_obs = self.reset(done_indices.tolist())
        
        info = {
            'episode_stats': self.episode_stats.copy(),
            'commands': self.commands.copy(),
        }
        
        return obs, privileged_obs, rewards, dones, info
    
    def _apply_actions(self, actions: np.ndarray):
        """应用动作到机器人"""
        action_scale = self.config.get('control', {}).get('action_scale', 0.25)
        
        # 只使用第一个环境的动作（目前只支持单环境）
        target_positions = self.default_joint_pos + actions[0] * action_scale
        self.robot.control_dofs_position(target_positions)
    
    def get_observations(self) -> np.ndarray:
        """获取观察值 - 目前只支持 num_envs=1"""
        # 关节状态
        qpos = self.robot.get_dofs_position()
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()[:self.n_dofs]
        else:
            qpos = np.array(qpos)[:self.n_dofs]
        
        qvel = self.robot.get_dofs_velocity()
        if hasattr(qvel, 'cpu'):
            qvel = qvel.cpu().numpy()[:self.n_dofs]
        else:
            qvel = np.array(qvel)[:self.n_dofs]
        
        # 上一帧动作 (取第一个环境)
        last_action = self.last_actions[0]
        
        # IMU 数据
        base_ang_vel = self.robot.get_ang()
        if hasattr(base_ang_vel, 'cpu'):
            base_ang_vel = base_ang_vel.cpu().numpy()
        else:
            base_ang_vel = np.array(base_ang_vel)
        if base_ang_vel.shape != (3,):
            base_ang_vel = base_ang_vel[:3]
        
        # 投影重力
        base_quat = self.robot.get_quat()
        if hasattr(base_quat, 'cpu'):
            base_quat = base_quat.cpu().numpy()
        else:
            base_quat = np.array(base_quat)
        if base_quat.shape != (4,):
            base_quat = base_quat[:4]
        projected_gravity = self._quat_rotate_inverse(base_quat, np.array([0, 0, -1]))
        
        # 命令 (取第一个环境)
        command = self.commands[0]
        
        # 步态信息 (取第一个环境)
        gait_sin = np.sin(self.gait_phase[0] * 2 * np.pi)
        gait_cos = np.cos(self.gait_phase[0] * 2 * np.pi)
        
        # 组合观察
        obs = np.concatenate([
            qpos.flatten()[:self.n_dofs],
            qvel.flatten()[:self.n_dofs],
            last_action.flatten()[:self.n_dofs],
            base_ang_vel.flatten()[:3],
            projected_gravity.flatten()[:3],
            command.flatten()[:3],
            np.array([gait_sin]),
            np.array([gait_cos]),
        ])
        
        # 确保维度正确
        if len(obs) != self.num_obs:
            if len(obs) < self.num_obs:
                obs = np.pad(obs, (0, self.num_obs - len(obs)), 'constant')
            else:
                obs = obs[:self.num_obs]
        
        return obs.astype(np.float32)
    
    def get_privileged_observations(self) -> np.ndarray:
        """获取特权观察值"""
        obs = self.get_observations()
        
        if len(obs) != self.num_obs:
            if len(obs) < self.num_obs:
                obs = np.pad(obs, (0, self.num_obs - len(obs)), 'constant')
            else:
                obs = obs[:self.num_obs]
        
        privileged_info = np.zeros(10)
        return np.concatenate([obs, privileged_info]).astype(np.float32)
    
    def _quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """四元数逆旋转"""
        return v
    
    def _compute_rewards(self, actions: np.ndarray) -> np.ndarray:
        """计算奖励"""
        rewards = np.zeros(self.num_envs)
        reward_config = self.config['rewards']
        
        base_vel = self.robot.get_vel()
        if hasattr(base_vel, 'cpu'):
            base_vel = base_vel.cpu().numpy()
        else:
            base_vel = np.array(base_vel)
        
        base_ang_vel = self.robot.get_ang()
        if hasattr(base_ang_vel, 'cpu'):
            base_ang_vel = base_ang_vel.cpu().numpy()
        else:
            base_ang_vel = np.array(base_ang_vel)
        
        # 确保形状正确
        if base_vel.shape != (3,):
            base_vel = base_vel[:3]
        if base_ang_vel.shape != (3,):
            base_ang_vel = base_ang_vel[:3]
        
        if np.any(np.isnan(base_vel)):
            base_vel = np.zeros(3)
        if np.any(np.isnan(base_ang_vel)):
            base_ang_vel = np.zeros(3)
        
        # 确保 actions 是 (num_envs, num_actions) 形状
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
        
        # 只计算第一个环境的奖励（目前只支持单环境）
        i = 0
        lin_vel_error = np.abs(base_vel[0] - self.commands[i, 0])
        rewards[i] += reward_config['tracking_lin_vel']['weight'] * np.exp(-lin_vel_error)
        
        ang_vel_error = np.abs(base_ang_vel[2] - self.commands[i, 2])
        rewards[i] += reward_config['tracking_ang_vel']['weight'] * np.exp(-ang_vel_error)
        
        rewards[i] += reward_config['lin_vel_z']['weight'] * (base_vel[2] ** 2)
        rewards[i] += reward_config['ang_vel_xy']['weight'] * (base_ang_vel[0] ** 2 + base_ang_vel[1] ** 2)
        
        action_rate = np.mean((actions[i] - self.last_actions[i]) ** 2)
        rewards[i] += reward_config['action_rate']['weight'] * action_rate
        
        torque_penalty = np.mean(actions[i] ** 2) * abs(reward_config['torques']['weight'])
        rewards[i] -= torque_penalty
        
        return rewards
    
    def _check_termination(self) -> np.ndarray:
        """检查终止条件"""
        dones = np.zeros(self.num_envs, dtype=bool)
        
        if self.current_step >= self.episode_length:
            dones[:] = True
            return dones
        
        base_pos = self.robot.get_pos()
        if hasattr(base_pos, 'cpu'):
            base_pos = base_pos.cpu().numpy()
        else:
            base_pos = np.array(base_pos)
        
        # 处理不同形状的 base_pos
        if base_pos.ndim == 0:
            # 标量情况（不应该发生）
            height = float(base_pos)
        elif base_pos.ndim == 1:
            # (3,) 形状 - 单个位置
            height = float(base_pos[2])
        else:
            # 多维情况
            height = float(base_pos.flat[2])
        
        termination_height = self.config['env']['termination_height']
        dones |= height < termination_height
        
        return dones
    
    def get_command(self, env_idx: int = 0) -> np.ndarray:
        """获取当前命令"""
        return self.commands[env_idx]
    
    def set_command(self, command: np.ndarray, env_idx: Optional[int] = None):
        """设置命令"""
        if env_idx is None:
            self.commands[:] = command
        else:
            self.commands[env_idx] = command
    
    def close(self):
        """关闭环境"""
        gs.destroy()
