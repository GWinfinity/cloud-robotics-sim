"""
Dexterous Manipulation Environment

灵巧操作环境，支持三种任务:
1. Grasp-and-Reach
2. Box Lift
3. Bimanual Handover
"""

import numpy as np
import genesis as gs
from typing import Dict, Tuple, Optional


class DexterousManipulationEnv:
    """
    灵巧双手操作环境
    
    人形机器人 + 灵巧双手 (各12 DOF)
    """
    
    def __init__(
        self,
        config: Dict,
        task_name: str = 'grasp_and_reach',
        num_envs: int = 1,
        headless: bool = False
    ):
        self.config = config
        self.task_name = task_name
        self.num_envs = num_envs
        self.headless = headless
        
        # 初始化Genesis
        gs.init(backend=gs.backends.CUDA)
        
        # 创建场景
        self.scene = self._create_scene()
        
        # 创建环境
        self._create_environment()
        
        # 状态维度
        self.obs_dim = 200  # 视觉 + 本体感觉 + 物体状态
        self.action_dim = config['robot']['num_joints']  # 54
        
        # 回合统计
        self.episode_stats = {}
        self.current_step = 0
        self.max_steps = 500
        
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
    
    def _create_environment(self):
        """创建环境元素"""
        # 地面
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0))
        )
        
        # 机器人
        self.robot = self.scene.add_entity(
            morph=gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.8, 1.0))
        )
        
        # 根据任务创建物体
        self.objects = self._create_task_objects()
        
        # 设置初始位置
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
    
    def _create_task_objects(self):
        """创建任务相关物体"""
        objects = []
        
        if self.task_name == 'grasp_and_reach':
            # 创建可抓取物体
            obj = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=0.04,
                    pos=[0.5, 0, 0.04]
                ),
                surface=gs.surfaces.Default(color=(0.8, 0.4, 0.2, 1.0))
            )
            objects.append(obj)
            
        elif self.task_name == 'box_lift':
            # 创建箱体
            obj = self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.15, 0.15, 0.15),
                    pos=[0.5, 0, 0.075]
                ),
                surface=gs.surfaces.Default(color=(0.4, 0.6, 0.8, 1.0))
            )
            objects.append(obj)
            
        elif self.task_name == 'bimanual_handover':
            # 创建长条物体 (需要双手)
            obj = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=0.02,
                    height=0.3,
                    pos=[0.4, 0, 0.15]
                ),
                surface=gs.surfaces.Default(color=(0.6, 0.8, 0.4, 1.0))
            )
            objects.append(obj)
        
        return objects
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self.current_step = 0
        self.episode_stats = {}
        
        # 重置机器人
        init_pos = self.config['robot']['init_pos']
        self.robot.set_pos(init_pos)
        
        # 构建场景
        self.scene.build()
        
        return self.get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        # 应用动作
        self._apply_action(action)
        
        # 仿真步进
        self.scene.step()
        
        # 获取观测
        obs = self.get_observation()
        
        # 计算奖励 (使用通用奖励函数)
        from models.reward_function import GeneralizedRewardFunction
        reward_fn = GeneralizedRewardFunction(self.config['reward_function'])
        
        state = self._get_state_dict()
        reward = reward_fn.compute_reward(state, action, self.task_name)
        
        # 检查终止
        self.current_step += 1
        success = self._check_success()
        done = success or self.current_step >= self.max_steps
        
        info = {
            'success': success,
            'task': self.task_name
        }
        
        return obs, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作"""
        action = np.clip(action, -1, 1)
        self.robot.control_dofs_position(action)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """获取观测"""
        # 本体感觉
        joint_pos = self.robot.get_dofs_position().cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()
        
        # 填充到固定维度
        joint_pos = np.pad(joint_pos, (0, 54 - len(joint_pos)), 'constant')
        joint_vel = np.pad(joint_vel, (0, 54 - len(joint_vel)), 'constant')
        
        # 物体状态
        if len(self.objects) > 0:
            obj_pos = self.objects[0].get_pos().cpu().numpy()
            obj_quat = self.objects[0].get_quat().cpu().numpy()
        else:
            obj_pos = np.zeros(3)
            obj_quat = np.array([0, 0, 0, 1])
        
        proprio = np.concatenate([joint_pos, joint_vel, obj_pos, obj_quat])
        
        # 填充
        if len(proprio) < self.obs_dim:
            proprio = np.pad(proprio, (0, self.obs_dim - len(proprio)), 'constant')
        
        return {
            'proprioception': proprio.astype(np.float32),
            'image': None,  # 视觉由外部相机处理
            'task_id': self._get_task_id()
        }
    
    def _get_state_dict(self) -> Dict:
        """获取状态字典 (用于奖励计算)"""
        return {
            'object_position': self.objects[0].get_pos().cpu().numpy() if self.objects else np.zeros(3),
            'target_position': np.array([0.7, 0.2, 0.5]),  # 示例目标
            'left_palm_contact_force': 0.0,  # 简化
            'right_palm_contact_force': 0.0,
        }
    
    def _get_task_id(self) -> int:
        """获取任务ID"""
        task_map = {
            'grasp_and_reach': 0,
            'box_lift': 1,
            'bimanual_handover': 2
        }
        return task_map.get(self.task_name, 0)
    
    def _check_success(self) -> bool:
        """检查任务成功"""
        # 简化版成功检测
        if self.task_name == 'box_lift':
            if len(self.objects) > 0:
                obj_height = self.objects[0].get_pos().cpu().numpy()[2]
                return obj_height > 0.3
        return False
    
    def close(self):
        """关闭环境"""
        gs.destroy()
