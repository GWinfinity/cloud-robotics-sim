"""
maniskill Plugin - Genesis ManiSkill 机器人操作仿真平台

来源: genesis-maniskill
核心实现: 基于Genesis的机器人操作仿真，整合RoboCasa场景和ManiSkill API

核心特性:
- Genesis后端: 高性能GPU加速物理仿真
- RoboCasa场景: 完整的厨房环境、家具和物品
- ManiSkill API: 熟悉的接口设计，易于迁移
- GPU并行: 支持数千个环境并行训练
- 统一资产: 整合多个来源的机器人模型和场景

支持的场景:
- Kitchen: 完整厨房（6种布局，8种风格，2500+物品）
- TableTop: 桌面操作环境

支持的任务:
- pick_place: 拾取放置
- open_drawer: 打开抽屉
- push: 推动物体
- stack: 堆叠物体
- prepare_food: 食物准备
- cleanup: 清理
- organize_cabinet: 整理橱柜

支持的机器人:
- Franka Emika Panda
- Universal Robots UR5
- Kinova Gen3
- UFACTORY xArm
- Fetch
- Unitree G1
"""

__version__ = "0.1.0"
__source__ = "genesis-maniskill"

# 环境
try:
    from .core.genesis_maniskill.envs.kitchen_env import KitchenEnv
    from .core.genesis_maniskill.envs.tabletop_env import TableTopEnv
    
    # 任务
    from .core.genesis_maniskill.tasks.pick_place import PickPlaceTask
    from .core.genesis_maniskill.tasks.open_drawer import OpenDrawerTask
    
    # 机器人
    from .core.genesis_maniskill.agents.robot_loader import RobotLoader
    from .core.genesis_maniskill.agents.franka import FrankaRobot
    from .core.genesis_maniskill.agents.ur5 import UR5Robot
    from .core.genesis_maniskill.agents.g1 import G1Robot
    
    # 数据集
    from .core.genesis_maniskill.datasets.formats import TrajectoryDataset
    
    __all__ = [
        'KitchenEnv',
        'TableTopEnv',
        'PickPlaceTask',
        'OpenDrawerTask',
        'RobotLoader',
        'FrankaRobot',
        'UR5Robot',
        'G1Robot',
        'TrajectoryDataset',
    ]
except ImportError:
    # 开发模式，可能部分模块未完全迁移
    __all__ = []
