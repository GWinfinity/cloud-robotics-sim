"""
Mobile Manipulator Environment

移动操作器环境，支持全身操作任务
"""

import numpy as np
import genesis as gs
from typing import Dict, Tuple, Optional


class MobileManipulatorEnv:
    """
    移动操作器环境
    
    支持:
    - 移动底座 (3 DOF: x, y, theta)
    - 双臂操作 (14 DOF: 7 x 2)
    - 躯干 (3 DOF)
    
    总共: 20 DOF
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
        
        # 创建物体
        self.objects = self._create_objects()
        
        # 状态维度
        self.state_dim = 50  # 根据实际观测调整
        self.action_dim = 20  # 20 DOF
        
        # 回合统计
        self.episode_stats = {}
        self.current_step = 0
        self.max_steps = config['env']['episode_length']
        
    def _create_scene(self):
        """创建场景"""
        return gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 3, 2),
                camera_lookat=(0, 0, 0.5),
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
        """创建移动操作器机器人"""
        # 使用Genesis内置模型作为近似
        robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.8, 1.0))
        )
        
        # 设置初始位置
        init_pos = self.config['robot']['init_pos']
        robot.set_pos(init_pos)
        
        return robot
    
    def _create_objects(self):
        """创建环境中的物体"""
        objects = []
        
        num_objects = self.config['env']['objects']['num_objects']
        position_range = self.config['env']['objects']['position_range']
        
        for i in range(num_objects):
            # 随机位置
            pos = np.array([
                np.random.uniform(*position_range[0]),
                np.random.uniform(*position_range[1]),
                np.random.uniform(*position_range[2])
            ])
            
            # 创建物体 (简化: 使用球体)
            obj = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=0.05,
                    pos=pos
                ),
                surface=gs.surfaces.Default(
                    color=(0.8, 0.4, 0.2, 1.0)
                )
            )
            
            objects.append(obj)
        
        return objects
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {}
        
        # 重置机器人
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 构建场景
        self.scene.build()
        
        return self.get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 应用动作
        self._apply_action(action)
        
        # 仿真步进
        self.scene.step()
        
        # 获取观测
        obs = self.get_observation()
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 检查终止
        self.current_step += 1
        done = self._check_termination()
        
        info = {
            'stats': self.episode_stats.copy()
        }
        
        return obs, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作到机器人"""
        # 裁剪动作
        action = np.clip(action, -1, 1)
        
        # 设置关节位置
        self.robot.control_dofs_position(action)
    
    def get_observation(self) -> np.ndarray:
        """获取观测"""
        # 机器人状态
        joint_pos = self.robot.get_dofs_position().cpu().numpy()[:20]
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()[:20]
        
        # 填充
        joint_pos = np.pad(joint_pos, (0, 20 - len(joint_pos)), 'constant')
        joint_vel = np.pad(joint_vel, (0, 20 - len(joint_vel)), 'constant')
        
        # 根节点状态
        root_pos = self.robot.get_pos().cpu().numpy()
        root_quat = self.robot.get_quat().cpu().numpy()
        
        # 物体状态 (简化: 只取第一个物体的位置)
        if len(self.objects) > 0:
            obj_pos = self.objects[0].get_pos().cpu().numpy()
        else:
            obj_pos = np.zeros(3)
        
        # 组合观测
        obs = np.concatenate([
            joint_pos[:20],
            joint_vel[:20],
            root_pos[:3],
            root_quat[:4],
            obj_pos[:3]
        ])
        
        # 填充到50维
        if len(obs) < 50:
            obs = np.pad(obs, (0, 50 - len(obs)), 'constant')
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """计算奖励"""
        # 基础存活奖励
        reward = 1.0
        
        # 能量惩罚
        reward -= 0.001 * np.sum(action ** 2)
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def close(self):
        """关闭环境"""
        gs.destroy()
