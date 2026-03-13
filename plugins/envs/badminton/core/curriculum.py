"""
三阶段课程学习管理器

论文中的三阶段训练:
1. Footwork Acquisition - 步法获取
2. Precision-Guided Swing Generation - 精确引导的挥拍生成
3. Task-Focused Refinement - 任务聚焦的精炼
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml


class ThreeStageCurriculum:
    """
    三阶段课程学习管理器
    
    阶段1: 步法获取
    - 冻结上肢关节
    - 训练下肢移动到击球位置
    - 简单发球
    
    阶段2: 挥拍生成
    - 解冻上肢
    - 训练挥拍动作
    - 中等难度发球
    
    阶段3: 任务精炼
    - 全身协调训练
    - 真实对打场景
    - 困难发球
    """
    
    def __init__(self, config_path: str = None):
        """初始化课程管理器"""
        self.config = self._load_config(config_path)
        self.current_stage = 1
        self.stage_progress = {
            1: {'completed': False, 'steps': 0},
            2: {'completed': False, 'steps': 0},
            3: {'completed': False, 'steps': 0}
        }
        
        # 当前阶段配置
        self.current_config = self._get_stage_config(1)
        
        # 统计信息
        self.stage_stats = {
            1: {'avg_reward': 0.0, 'success_rate': 0.0},
            2: {'avg_reward': 0.0, 'hit_rate': 0.0},
            3: {'avg_reward': 0.0, 'rally_length': 0.0}
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if config_path:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('curriculum', {})
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'stage1_footwork': {
                'name': 'Footwork Acquisition',
                'duration_steps': 2000000,
                'freeze_upper_body': True,
                'serve_difficulty': 'easy',
                'target_type': 'stationary'
            },
            'stage2_swing': {
                'name': 'Precision-Guided Swing',
                'duration_steps': 3000000,
                'freeze_upper_body': False,
                'serve_difficulty': 'medium',
                'target_type': 'guided'
            },
            'stage3_refinement': {
                'name': 'Task-Focused Refinement',
                'duration_steps': 5000000,
                'freeze_upper_body': False,
                'freeze_lower_body': False,
                'serve_difficulty': 'hard',
                'target_type': 'rally'
            }
        }
    
    def _get_stage_config(self, stage: int) -> Dict:
        """获取阶段配置"""
        key_map = {
            1: 'stage1_footwork',
            2: 'stage2_swing',
            3: 'stage3_refinement'
        }
        key = key_map.get(stage, 'stage1_footwork')
        return self.config.get(key, {})
    
    def update(self, total_steps: int, metrics: Dict) -> bool:
        """
        更新课程进度
        
        Args:
            total_steps: 当前总训练步数
            metrics: 当前阶段的训练指标
            
        Returns:
            stage_changed: 是否切换了阶段
        """
        # 更新统计
        self.stage_stats[self.current_stage] = metrics
        self.stage_progress[self.current_stage]['steps'] += 1
        
        # 检查是否需要升级
        if self._should_advance_stage(total_steps, metrics):
            return self.advance_stage()
        
        return False
    
    def _should_advance_stage(self, total_steps: int, metrics: Dict) -> bool:
        """检查是否应该进入下一阶段"""
        # 获取当前阶段配置
        stage_config = self.current_config
        required_steps = stage_config.get('duration_steps', 1000000)
        
        # 检查最小训练步数
        current_steps = self.stage_progress[self.current_stage]['steps']
        if current_steps < required_steps:
            return False
        
        # 检查成功率指标
        if self.current_stage == 1:
            # 阶段1: 位置到达率
            success_rate = metrics.get('success_rate', 0.0)
            return success_rate > 0.7
            
        elif self.current_stage == 2:
            # 阶段2: 击球率
            hit_rate = metrics.get('hit_rate', 0.0)
            return hit_rate > 0.5
            
        elif self.current_stage == 3:
            # 阶段3: 不需要升级
            return False
        
        return False
    
    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.current_stage >= 3:
            return False
        
        # 标记当前阶段完成
        self.stage_progress[self.current_stage]['completed'] = True
        
        # 升级
        self.current_stage += 1
        self.current_config = self._get_stage_config(self.current_stage)
        
        print(f"\n{'='*60}")
        print(f"Curriculum Stage Advanced!")
        print(f"Now at Stage {self.current_stage}: {self.current_config.get('name', 'Unknown')}")
        print(f"Serve Difficulty: {self.current_config.get('serve_difficulty', 'normal')}")
        print(f"Target Type: {self.current_config.get('target_type', 'normal')}")
        print(f"{'='*60}\n")
        
        return True
    
    def get_current_stage(self) -> int:
        """获取当前阶段"""
        return self.current_stage
    
    def get_stage_config(self) -> Dict:
        """获取当前阶段配置"""
        return self.current_config
    
    def get_serve_params(self) -> Dict:
        """
        获取当前阶段的发球参数
        
        Returns:
            发球参数字典
        """
        difficulty = self.current_config.get('serve_difficulty', 'easy')
        
        if difficulty == 'easy':
            return {
                'velocity_range': [5, 10],
                'angle_range': [-0.2, 0.2],
                'height_range': [2.0, 2.5],
                'position_variance': 0.5
            }
        elif difficulty == 'medium':
            return {
                'velocity_range': [8, 15],
                'angle_range': [-0.3, 0.3],
                'height_range': [1.8, 2.8],
                'position_variance': 1.0
            }
        else:  # hard
            return {
                'velocity_range': [10, 20],
                'angle_range': [-0.5, 0.5],
                'height_range': [1.5, 3.0],
                'position_variance': 1.5
            }
    
    def get_target_position(self, robot_pos: np.ndarray, shuttle_pos: np.ndarray) -> np.ndarray:
        """
        获取目标击球位置
        
        Args:
            robot_pos: 机器人当前位置
            shuttle_pos: 羽毛球当前位置
            
        Returns:
            目标击球位置 [x, y, z]
        """
        target_type = self.current_config.get('target_type', 'stationary')
        
        if target_type == 'stationary':
            # 固定目标位置
            return np.array([-2.0, 0.0, 1.5])
            
        elif target_type == 'guided':
            # 引导式: 羽毛球轨迹上的某点
            # 简化: 机器人前方一定距离
            return np.array([
                robot_pos[0] + 1.0,
                shuttle_pos[1] * 0.5,  # 部分跟随
                1.5 + shuttle_pos[2] * 0.2
            ])
            
        else:  # rally
            # 对打模式: 动态计算最优击球点
            # 考虑羽毛球轨迹预测
            return np.array([
                -1.5,  # 靠近网但不过网
                np.clip(shuttle_pos[1], -1.0, 1.0),  # 横向居中
                1.2 + np.random.uniform(0, 0.5)  # 适当高度
            ])
    
    def should_freeze_upper_body(self) -> bool:
        """是否应该冻结上肢"""
        return self.current_config.get('freeze_upper_body', False)
    
    def should_freeze_lower_body(self) -> bool:
        """是否应该冻结下肢"""
        return self.current_config.get('freeze_lower_body', False)
    
    def get_frozen_joint_indices(self, joint_names: List[str]) -> List[int]:
        """
        获取应该冻结的关节索引
        
        Args:
            joint_names: 关节名称列表
            
        Returns:
            应该冻结的关节索引列表
        """
        frozen_indices = []
        
        upper_body_keywords = ['shoulder', 'elbow', 'wrist', 'arm']
        lower_body_keywords = ['hip', 'knee', 'ankle', 'leg']
        
        for i, name in enumerate(joint_names):
            if self.should_freeze_upper_body():
                if any(kw in name.lower() for kw in upper_body_keywords):
                    frozen_indices.append(i)
            
            if self.should_freeze_lower_body():
                if any(kw in name.lower() for kw in lower_body_keywords):
                    frozen_indices.append(i)
        
        return frozen_indices
    
    def get_stats(self) -> Dict:
        """获取课程学习统计"""
        return {
            'current_stage': self.current_stage,
            'stage_name': self.current_config.get('name', 'Unknown'),
            'stage_progress': self.stage_progress,
            'stage_stats': self.stage_stats
        }


class AdaptiveStageTransition:
    """
    自适应阶段过渡
    
    根据训练指标自动调整阶段过渡条件
    """
    
    def __init__(self, base_curriculum: ThreeStageCurriculum):
        self.curriculum = base_curriculum
        self.performance_buffer = {
            1: [],
            2: [],
            3: []
        }
        self.buffer_size = 100
        
    def update_metrics(self, stage: int, metric_value: float):
        """更新性能指标"""
        self.performance_buffer[stage].append(metric_value)
        if len(self.performance_buffer[stage]) > self.buffer_size:
            self.performance_buffer[stage].pop(0)
    
    def get_moving_average(self, stage: int) -> float:
        """获取滑动平均性能"""
        if len(self.performance_buffer[stage]) == 0:
            return 0.0
        return np.mean(self.performance_buffer[stage])
    
    def should_advance(self) -> bool:
        """基于滑动平均判断是否该升级"""
        current_stage = self.curriculum.get_current_stage()
        avg_performance = self.get_moving_average(current_stage)
        
        thresholds = {
            1: 0.6,  # 阶段1: 平均成功率 > 60%
            2: 0.4,  # 阶段2: 平均击球率 > 40%
        }
        
        return avg_performance > thresholds.get(current_stage, 1.0)


class DomainRandomization:
    """
    域随机化
    
    在训练后期引入随机化以提高 sim-to-real 能力
    """
    
    def __init__(self):
        self.enabled = False
        self.randomization_params = {
            'shuttle_mass': (0.0048, 0.0056),      # 羽毛球质量
            'shuttle_drag': (0.5, 0.7),             # 阻力系数
            'court_friction': (0.8, 1.2),           # 场地摩擦
            'robot_mass_scale': (0.95, 1.05),       # 机器人质量缩放
            'motor_strength': (0.9, 1.1),           # 电机强度
        }
        
    def enable(self):
        """启用域随机化"""
        self.enabled = True
        print("Domain Randomization Enabled!")
        
    def get_randomized_params(self) -> Dict:
        """获取随机化参数"""
        if not self.enabled:
            return {}
        
        return {
            param: np.random.uniform(low, high)
            for param, (low, high) in self.randomization_params.items()
        }
    
    def apply_to_shuttlecock(self, shuttlecock):
        """应用随机化到羽毛球"""
        if not self.enabled:
            return
        
        params = self.get_randomized_params()
        shuttlecock.mass = params.get('shuttle_mass', 0.0052)
        shuttlecock.drag_coefficient = params.get('shuttle_drag', 0.6)
