"""
MPC + WBC Controller Plugin

来源: openloong-dyn-control
核心实现: 
- MPC (Model Predictive Control) 模型预测控制
- WBC (Whole-Body Control) 全身控制

用途: 人形机器人全身运动控制，支持平地行走、上下楼梯、复杂地形
"""

__version__ = "0.1.0"
__source__ = "openloong-dyn-control"
__author__ = "Genesis Cloud Sim Team"

from .core.mpc_controller import MPCController
from .core.wbc_controller import WBCController
from .core.combined_controller import MPCWBCController
from .core.gait_scheduler import GaitScheduler
from .core.config import WBCConfig

__all__ = [
    'MPCController',
    'WBCController', 
    'MPCWBCController',
    'GaitScheduler',
    'WBCConfig'
]

# Auto-register plugin
try:
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager
    pm = get_plugin_manager()
    # Plugin will be discovered via plugin.yaml
except ImportError:
    pass  # Plugin manager not available, that's OK
