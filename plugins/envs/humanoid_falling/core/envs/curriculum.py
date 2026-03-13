"""
课程学习管理器 - Domain Diversification Curriculum

基于论文的课程学习策略，逐步增加任务难度:
1. 从小推力开始，逐渐增加推力和持续时间
2. 从不同方向施加推力
3. 最终在不同地面材质上训练
"""

import numpy as np
from typing import Dict, List, Tuple
import yaml


class CurriculumManager:
    """
    课程学习管理器
    
    管理训练过程中任务难度的逐步增加，包括:
    - 推力大小
    - 推力持续时间
    - 推力方向分布
    - 地面摩擦系数 (可选)
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化课程管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.levels = self.config['curriculum']['levels']
        self.current_level_idx = 0
        self.current_step = 0
        
        # 当前参数
        self.current_params = self._get_level_params(0)
        
        # 进度跟踪
        self.level_success_rates = [0.0] * len(self.levels)
        self.level_steps = [0] * len(self.levels)
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if config_path:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'curriculum': {
                    'levels': [
                        {
                            'name': 'easy',
                            'push_force_range': [50, 100],
                            'push_duration': 0.1,
                            'duration_steps': 1000000
                        },
                        {
                            'name': 'medium',
                            'push_force_range': [100, 200],
                            'push_duration': 0.2,
                            'duration_steps': 2000000
                        },
                        {
                            'name': 'hard',
                            'push_force_range': [200, 400],
                            'push_duration': 0.3,
                            'duration_steps': 3000000
                        },
                        {
                            'name': 'expert',
                            'push_force_range': [300, 600],
                            'push_duration': 0.5,
                            'duration_steps': 4000000
                        }
                    ]
                }
            }
    
    def _get_level_params(self, level_idx: int) -> Dict:
        """获取指定等级的参数"""
        if level_idx < len(self.levels):
            return self.levels[level_idx]
        return self.levels[-1]
    
    def update(self, total_steps: int, success_rate: float) -> bool:
        """
        更新课程进度
        
        Args:
            total_steps: 当前总训练步数
            success_rate: 当前成功率 (例如: 存活率)
            
        Returns:
            level_changed: 是否切换到了新等级
        """
        self.current_step = total_steps
        
        # 更新当前等级的统计
        self.level_success_rates[self.current_level_idx] = success_rate
        self.level_steps[self.current_level_idx] += 1
        
        # 检查是否升级到下一级
        level_changed = False
        
        # 升级条件:
        # 1. 达到当前等级最小训练步数
        # 2. 成功率超过阈值 (例如 70%)
        
        if self.current_level_idx < len(self.levels) - 1:
            current_level = self.levels[self.current_level_idx]
            required_steps = current_level['duration_steps']
            
            # 累计步数检查
            cumulative_steps = sum(
                self.level_steps[i] for i in range(self.current_level_idx + 1)
            )
            
            if cumulative_steps >= required_steps and success_rate > 0.7:
                self.current_level_idx += 1
                self.current_params = self._get_level_params(self.current_level_idx)
                level_changed = True
                
                print(f"\n{'='*50}")
                print(f"Curriculum Level Up! Now at Level {self.current_level_idx + 1}: {self.current_params['name']}")
                print(f"Push Force Range: {self.current_params['push_force_range']} N")
                print(f"Push Duration: {self.current_params['push_duration']} s")
                print(f"{'='*50}\n")
        
        return level_changed
    
    def get_current_params(self) -> Dict:
        """获取当前等级的参数"""
        return self.current_params
    
    def get_push_params(self) -> Tuple[List[float], float]:
        """
        获取当前推力的参数
        
        Returns:
            force_range: [min_force, max_force]
            duration: 推力持续时间
        """
        params = self.current_params
        return (
            params['push_force_range'],
            params['push_duration']
        )
    
    def get_push_direction(self) -> np.ndarray:
        """
        获取随机推力方向
        
        随着等级提升，推力方向分布更加多样化
        """
        # 基础: 随机水平方向
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 高级: 可能包含垂直分量
        if self.current_level_idx >= 2:  # hard 及以上
            # 有一定概率施加带有垂直分量的力
            if np.random.random() < 0.3:
                # 向上或向下的推力
                vertical = np.random.choice([-1, 1]) * np.random.uniform(0, 0.3)
            else:
                vertical = 0
        else:
            vertical = 0
        
        direction = np.array([
            np.cos(angle),
            np.sin(angle),
            vertical
        ])
        
        return direction / np.linalg.norm(direction)
    
    def get_stats(self) -> Dict:
        """获取课程学习统计信息"""
        return {
            'current_level': self.current_level_idx,
            'current_level_name': self.current_params['name'],
            'level_success_rates': self.level_success_rates,
            'level_steps': self.level_steps,
            'total_steps': self.current_step
        }


class AdaptiveCurriculumManager(CurriculumManager):
    """
    自适应课程学习管理器
    
    根据智能体的实时表现动态调整难度，而不仅仅是固定步数
    """
    
    def __init__(self, config_path: str = None, window_size: int = 100):
        super().__init__(config_path)
        self.window_size = window_size
        self.recent_rewards = []
        self.reward_thresholds = {
            'upgrade': 0.8,   # 平均奖励超过此值则升级
            'downgrade': 0.3  # 平均奖励低于此值则降级
        }
        
    def update(self, total_steps: int, episode_reward: float) -> bool:
        """
        基于奖励的自适应更新
        
        Args:
            total_steps: 当前总训练步数
            episode_reward: 最近回合的奖励
            
        Returns:
            level_changed: 是否切换了等级
        """
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        self.current_step = total_steps
        
        # 计算滑动窗口平均奖励
        if len(self.recent_rewards) >= self.window_size // 2:
            avg_reward = np.mean(self.recent_rewards)
            
            # 检查升级
            if (avg_reward > self.reward_thresholds['upgrade'] and 
                self.current_level_idx < len(self.levels) - 1):
                
                self.current_level_idx += 1
                self.current_params = self._get_level_params(self.current_level_idx)
                self.recent_rewards = []  # 重置历史
                
                print(f"\n{'='*50}")
                print(f"Adaptive Level Up! Level {self.current_level_idx + 1}: {self.current_params['name']}")
                print(f"Avg Reward: {avg_reward:.3f}")
                print(f"{'='*50}\n")
                return True
            
            # 检查降级 (可选，防止卡住)
            elif (avg_reward < self.reward_thresholds['downgrade'] and 
                  self.current_level_idx > 0):
                
                self.current_level_idx -= 1
                self.current_params = self._get_level_params(self.current_level_idx)
                self.recent_rewards = []
                
                print(f"\n{'='*50}")
                print(f"Level Downgrade to {self.current_level_idx + 1}: {self.current_params['name']}")
                print(f"Avg Reward: {avg_reward:.3f}")
                print(f"{'='*50}\n")
                return True
        
        return False


class DomainRandomizationCurriculum:
    """
    域随机化课程
    
    在训练后期引入更多领域随机化，提高 sim-to-real 迁移能力
    """
    
    def __init__(self):
        self.physics_params = {
            'friction': (0.5, 1.5),
            'mass_scale': (0.9, 1.1),
            'motor_strength': (0.9, 1.1),
        }
        self.enable_randomization = False
        
    def enable(self):
        """启用域随机化"""
        self.enable_randomization = True
        print("Domain Randomization Enabled!")
        
    def get_randomized_params(self) -> Dict:
        """获取随机化的物理参数"""
        if not self.enable_randomization:
            return {k: 1.0 for k in self.physics_params.keys()}
        
        return {
            param: np.random.uniform(low, high)
            for param, (low, high) in self.physics_params.items()
        }
