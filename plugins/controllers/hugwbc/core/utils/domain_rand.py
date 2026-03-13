"""
Domain Randomization for HugWBC

域随机化实现，提高策略泛化能力
"""

import numpy as np
from typing import Dict, Tuple


class DomainRandomizer:
    """
    域随机化器
    
    对物理参数进行随机化，提高策略鲁棒性
    """
    
    def __init__(self, config: Dict, num_envs: int):
        self.config = config
        self.num_envs = num_envs
        
        # 随机化范围
        self.friction_range = config.get('friction_range', [0.5, 1.5])
        self.mass_scale_range = config.get('mass_scale_range', [0.9, 1.1])
        self.com_displacement_range = config.get('com_displacement_range', [-0.05, 0.05])
        self.motor_strength_range = config.get('motor_strength_range', [0.9, 1.1])
        self.joint_damping_range = config.get('joint_damping_range', [0.9, 1.1])
        
        # 当前随机化参数
        self.params = self._generate_params()
    
    def _generate_params(self) -> Dict:
        """生成随机参数"""
        return {
            'friction': np.random.uniform(*self.friction_range, self.num_envs),
            'mass_scale': np.random.uniform(*self.mass_scale_range, self.num_envs),
            'com_displacement': np.random.uniform(
                self.com_displacement_range[0],
                self.com_displacement_range[1],
                (self.num_envs, 3)
            ),
            'motor_strength': np.random.uniform(*self.motor_strength_range, self.num_envs),
            'joint_damping': np.random.uniform(*self.joint_damping_range, self.num_envs),
        }
    
    def randomize(self, env_ids: np.ndarray = None):
        """
        重新随机化参数
        
        Args:
            env_ids: 需要随机化的环境索引，None 表示全部
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        self.params['friction'][env_ids] = np.random.uniform(
            *self.friction_range, len(env_ids)
        )
        self.params['mass_scale'][env_ids] = np.random.uniform(
            *self.mass_scale_range, len(env_ids)
        )
        self.params['com_displacement'][env_ids] = np.random.uniform(
            self.com_displacement_range[0],
            self.com_displacement_range[1],
            (len(env_ids), 3)
        )
        self.params['motor_strength'][env_ids] = np.random.uniform(
            *self.motor_strength_range, len(env_ids)
        )
        self.params['joint_damping'][env_ids] = np.random.uniform(
            *self.joint_damping_range, len(env_ids)
        )
    
    def get_params(self, env_idx: int) -> Dict:
        """获取指定环境的参数"""
        return {k: v[env_idx] for k, v in self.params.items()}
    
    def apply_to_sim(self, scene, robot, env_idx: int):
        """
        将随机化参数应用到仿真
        
        注意：这需要根据具体的物理引擎 API 实现
        """
        params = self.get_params(env_idx)
        
        # 这里应该调用 Genesis 的 API 来修改参数
        # 由于 Genesis API 可能不同，这里仅作为示例
        
        # 例如：修改摩擦系数
        # robot.set_friction(params['friction'])
        
        # 例如：修改质量
        # robot.set_mass_scale(params['mass_scale'])
        
        pass


class Curriculum:
    """
    课程学习管理器
    
    逐渐增加任务难度
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 难度级别
        self.levels = config.get('levels', [
            {'name': 'easy', 'max_force': 50, 'command_range': 0.5},
            {'name': 'medium', 'max_force': 100, 'command_range': 1.0},
            {'name': 'hard', 'max_force': 150, 'command_range': 1.5},
        ])
        
        self.current_level = 0
        self.success_threshold = config.get('success_threshold', 0.8)
        self.success_history = []
        self.window_size = config.get('window_size', 100)
    
    def get_current_params(self) -> Dict:
        """获取当前难度参数"""
        return self.levels[self.current_level]
    
    def update(self, success_rate: float):
        """
        更新课程进度
        
        Args:
            success_rate: 最近的成功率
        """
        self.success_history.append(success_rate)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # 检查是否升级
        if len(self.success_history) >= self.window_size:
            avg_success = np.mean(self.success_history[-self.window_size:])
            if avg_success > self.success_threshold and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                print(f"Curriculum: Level up to {self.levels[self.current_level]['name']}")
                self.success_history = []
    
    def get_command_range_scale(self) -> float:
        """获取当前命令范围缩放因子"""
        return self.levels[self.current_level]['command_range']
    
    def reset(self):
        """重置课程"""
        self.current_level = 0
        self.success_history = []
