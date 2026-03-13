"""
Manipulation Environment

支持多种操作任务: pick-and-place, insertion, stacking等
"""

import numpy as np
import genesis as gs
from typing import Dict, Tuple, Optional


class ManipulationEnv:
    """
    操作环境
    
    支持:
    - 视觉观测 (相机图像)
    - 稀疏二元奖励
    - 多种操作任务
    """
    
    def __init__(
        self,
        config: Dict,
        task_name: str = 'pick_and_place',
        num_envs: int = 1,
        headless: bool = False,
        device: str = 'cuda'
    ):
        self.config = config
        self.task_name = task_name
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
        
        # 创建物体
        self.objects = self._create_task_objects()
        
        # 状态维度
        self.obs_dim = 100  # 根据配置调整
        self.action_dim = config['robot']['num_joints']
        
        # 回合统计
        self.episode_stats = {
            'success': False,
            'steps_to_success': 0
        }
        self.current_step = 0
        self.max_steps = 500
        
    def _create_scene(self):
        """创建场景"""
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.5),
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
        """创建机器人"""
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.8, 1.0))
        )
        
        init_pos = self.config['robot']['init_pos']
        robot.set_pos(init_pos)
        
        return robot
    
    def _create_task_objects(self):
        """创建任务相关物体"""
        objects = []
        
        if self.task_name == 'pick_and_place':
            # 创建物体
            obj = self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.05, 0.05, 0.05),
                    pos=[0.5, 0, 0.025]
                ),
                surface=gs.surfaces.Default(color=(0.8, 0.4, 0.2, 1.0))
            )
            objects.append(obj)
            
            # 目标位置标记
            self.goal_pos = np.array([0.7, 0.2, 0.025])
        
        elif self.task_name == 'insertion':
            # 创建peg和hole
            peg = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=0.02,
                    height=0.1,
                    pos=[0.5, 0, 0.05]
                ),
                surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))
            )
            objects.append(peg)
        
        return objects
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {'success': False, 'steps_to_success': 0}
        
        # 重置机器人
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 构建场景
        self.scene.build()
        
        obs = self.get_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        # 应用动作
        self._apply_action(action)
        
        # 仿真步进
        self.scene.step()
        
        # 获取观测
        obs = self.get_observation()
        
        # 检查成功 (稀疏奖励)
        success = self._check_success()
        
        # 稀疏二元奖励
        reward = 1.0 if success else 0.0
        
        # 更新统计
        if success and not self.episode_stats['success']:
            self.episode_stats['success'] = True
            self.episode_stats['steps_to_success'] = self.current_step
        
        # 检查终止
        self.current_step += 1
        done = success or self.current_step >= self.max_steps
        
        info = {
            'success': success,
            'stats': self.episode_stats.copy()
        }
        
        return obs, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作"""
        action = np.clip(action, -1, 1)
        self.robot.control_dofs_position(action)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """获取观测"""
        # 本体感觉
        joint_pos = self.robot.get_dofs_position().cpu().numpy()[:self.action_dim]
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()[:self.action_dim]
        
        # 填充
        joint_pos = np.pad(joint_pos, (0, self.action_dim - len(joint_pos)), 'constant')
        joint_vel = np.pad(joint_vel, (0, self.action_dim - len(joint_vel)), 'constant')
        
        root_pos = self.robot.get_pos().cpu().numpy()
        
        # 组合
        proprio = np.concatenate([joint_pos, joint_vel, root_pos])
        
        # 填充
        if len(proprio) < self.obs_dim:
            proprio = np.pad(proprio, (0, self.obs_dim - len(proprio)), 'constant')
        
        return {
            'proprioception': proprio.astype(np.float32),
            'image': None  # 视觉观测由外部相机处理
        }
    
    def _check_success(self) -> bool:
        """检查任务成功 (稀疏奖励)"""
        if self.task_name == 'pick_and_place':
            if len(self.objects) > 0:
                obj_pos = self.objects[0].get_pos().cpu().numpy()
                distance = np.linalg.norm(obj_pos - self.goal_pos)
                return distance < 0.05
        
        return False
    
    def close(self):
        """关闭环境"""
        gs.destroy()
