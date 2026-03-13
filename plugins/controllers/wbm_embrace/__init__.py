"""
wbm_embrace Plugin - WBM Embrace: 人形机器人全身操作大型物体

来源: genesis-wbm-embrace
核心实现: 使用全身操作（WBM）来拥抱大型物体

论文: "Embracing Bulky Objects with Humanoid Robots: 
       Whole-Body Manipulation with Reinforcement Learning"

核心特性:
- 人类运动先验: 预训练大规模人类运动数据，生成自然的全身运动
- NSDF: 神经符号距离场提供准确的几何感知
- 教师-学生架构: 蒸馏人类运动知识到机器人策略
- 全身拥抱策略: 手臂和躯干协调控制，稳定多接触交互

解决的问题:
- 传统末端执行器抓取的稳定性限制
- 大型物体的载荷限制
- 多接触点的协调控制
"""

__version__ = "0.1.0"
__source__ = "genesis-wbm-embrace"
__paper__ = "arXiv:2509.13534"

# 环境
from .core.envs.embrace_env import EmbraceEnv
from .core.envs.bulky_objects import BulkyObjectGenerator

# 模型
from .core.models.motion_prior import MotionPrior
from .core.models.nsdf import NSDF
from .core.models.teacher_student import TeacherStudentPolicy

__all__ = [
    'EmbraceEnv',
    'BulkyObjectGenerator',
    'MotionPrior',
    'NSDF',
    'TeacherStudentPolicy',
]
