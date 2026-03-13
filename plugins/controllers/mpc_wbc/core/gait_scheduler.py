"""
Gait Scheduler

步态调度器: 管理摆动腿时序
"""

import numpy as np


class GaitScheduler:
    """
    步态调度器
    
    管理步态周期，决定哪条腿支撑，哪条腿摆动。
    
    步态类型:
    - trot: 对角腿同步 (0.5 duty factor)
    - walk: 四足顺序 (0.75 duty factor)
    - pace: 同侧腿同步
    - bound: 前后腿同步
    """
    
    def __init__(
        self,
        frequency: float = 1.25,
        gait_type: str = 'trot',
        duty_factor: float = 0.5
    ):
        """
        Args:
            frequency: 步态频率 (Hz)
            gait_type: 步态类型
            duty_factor: 支撑相占比
        """
        self.frequency = frequency
        self.gait_type = gait_type
        self.duty_factor = duty_factor
        
        # 状态
        self.phase = 0.0  # 当前相位 [0, 1]
        self.period = 1.0 / frequency
    
    def update(self, dt: float) -> float:
        """
        更新步态相位
        
        Args:
            dt: 时间步长
        
        Returns:
            当前相位 [0, 1]
        """
        self.phase += self.frequency * dt
        self.phase %= 1.0
        return self.phase
    
    def get_swing_leg(self) -> str:
        """
        获取当前摆动腿
        
        Returns:
            'left' 或 'right'
        """
        if self.gait_type == 'trot':
            # 对角腿同步
            if self.phase < 0.5:
                return 'left'
            else:
                return 'right'
        elif self.gait_type == 'walk':
            # 顺序步态
            if self.phase < 0.25:
                return 'left'
            elif self.phase < 0.5:
                return 'right'
            elif self.phase < 0.75:
                return 'left'
            else:
                return 'right'
        else:
            # 默认
            return 'left' if self.phase < 0.5 else 'right'
    
    def get_leg_state(self, leg: str) -> int:
        """
        获取腿的状态
        
        Args:
            leg: 'left' 或 'right'
        
        Returns:
            0: 摆动相 (swing)
            1: 早期支撑相 (early contact)
            2: 支撑相 (stance)
        """
        swing_leg = self.get_swing_leg()
        
        if leg == swing_leg:
            return 0  # 摆动
        else:
            # 简单判断早期接触
            phase_norm = self.phase % 0.5
            if phase_norm < 0.1:
                return 1  # 早期接触
            else:
                return 2  # 支撑
    
    def get_swing_height(self) -> float:
        """
        获取摆动脚目标高度
        
        Returns:
            目标高度 (m)
        """
        # 正弦轨迹
        swing_phase = self.phase % 0.5 / 0.5  # [0, 1]
        height = 0.08 * np.sin(swing_phase * np.pi)
        return max(0, height)
    
    def is_swing_start(self) -> bool:
        """检查是否是摆动相开始"""
        return self.phase < 0.02 or (0.5 <= self.phase < 0.52)
    
    def is_swing_end(self) -> bool:
        """检查是否是摆动相结束"""
        return (0.48 <= self.phase < 0.5) or self.phase >= 0.98
