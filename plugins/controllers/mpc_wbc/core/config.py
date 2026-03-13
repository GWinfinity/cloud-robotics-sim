"""
MPC-WBC Configuration

配置类: 管理 MPC + WBC 控制器的所有参数
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class MPCConfig:
    """MPC 配置"""
    dt: float = 0.005                    # 控制周期
    horizon: int = 10                    # 预测步数
    control_horizon: int = 3             # 控制时域
    mass: float = 77.35                  # 机器人质量 (kg)
    
    # 权重参数
    alpha: float = 1e-6                  # 输入正则化
    L_diag: List[float] = field(default_factory=lambda: [
        1.0, 1.0, 1.0,      # 欧拉角
        1.0, 200.0, 1.0,    # 位置 (高度权重高)
        1e-7, 1e-7, 1e-7,   # 角速度
        100.0, 10.0, 1.0    # 线速度
    ])
    K_diag: List[float] = field(default_factory=lambda: [1.0] * 13)
    
    # 约束
    max_force: Optional[List[float]] = None
    min_force: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.max_force is None:
            self.max_force = [1000.0, 1000.0, -3.0 * self.mass * (-9.81), 
                             20.0, 80.0, 100.0]
        if self.min_force is None:
            self.min_force = [-1000.0, -1000.0, 0.0, 
                             -20.0, -80.0, -100.0]


@dataclass
class WBCConfig:
    """WBC 配置"""
    num_dofs: int = 19                   # 自由度数量
    dt: float = 0.005                    # 控制周期
    
    # PD 增益
    kp_com: float = 100.0                # 质心位置
    kd_com: float = 20.0
    kp_swing: float = 400.0              # 摆动脚
    kd_swing: float = 40.0
    kp_orientation: float = 100.0        # 姿态
    kd_orientation: float = 20.0


@dataclass
class GaitConfig:
    """步态配置"""
    frequency: float = 1.25              # 步态频率 (Hz)
    gait_type: str = 'trot'              # 步态类型
    duty_factor: float = 0.5             # 支撑相占比
    swing_height: float = 0.08           # 摆动高度 (m)


@dataclass
class WBCConfig:
    """完整配置"""
    mpc: MPCConfig = field(default_factory=MPCConfig)
    wbc: WBCConfig = field(default_factory=WBCConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    
    # 启用开关
    use_mpc: bool = True
    use_wbc: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> 'WBCConfig':
        """从 YAML 文件加载配置"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # 简化实现: 直接使用默认配置
        config = cls()
        
        if 'mpc' in data:
            for key, value in data['mpc'].items():
                if hasattr(config.mpc, key):
                    setattr(config.mpc, key, value)
        
        if 'wbc' in data:
            for key, value in data['wbc'].items():
                if hasattr(config.wbc, key):
                    setattr(config.wbc, key, value)
        
        if 'gait' in data:
            for key, value in data['gait'].items():
                if hasattr(config.gait, key):
                    setattr(config.gait, key, value)
        
        return config
    
    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        import yaml
        
        data = {
            'mpc': self.mpc.__dict__,
            'wbc': self.wbc.__dict__,
            'gait': self.gait.__dict__,
            'use_mpc': self.use_mpc,
            'use_wbc': self.use_wbc
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
