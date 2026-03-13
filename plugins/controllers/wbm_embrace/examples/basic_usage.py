"""
WBM Embrace Plugin - 基础使用示例

演示如何使用 WBM Embrace 进行人形机器人全身操作大型物体
"""

import numpy as np
import torch
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.controllers.wbm_embrace import (
        EmbraceEnv, MotionPrior, NSDF, TeacherStudentPolicy,
        BulkyObjectGenerator
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.envs.embrace_env import EmbraceEnv
    from core.envs.bulky_objects import BulkyObjectGenerator
    from core.models.motion_prior import MotionPrior
    from core.models.nsdf import NSDF
    from core.models.teacher_student import TeacherStudentPolicy


def example_motion_prior():
    """示例1: 人类运动先验"""
    print("=" * 60)
    print("示例1: 人类运动先验 (Motion Prior)")
    print("=" * 60)
    
    print("""
Motion Prior的作用:

人形机器人控制面临的挑战:
- 高维动作空间 (29+ DOF)
- 运动不自然 (像机器人不像人)
- 物理不可行 (自碰撞、不稳定)

解决方案: 从人类运动数据学习先验

人类运动数据集 (如AMASS):
├── 全身运动捕捉
├── 多样姿态和动作
├── 自然运动模式
└── 物理可行性

VAE架构:
  编码器: 人类姿态 (72D) → 潜在空间 (32D)
  解码器: 潜在向量 (32D) → 重建姿态 (72D)

应用:
  1. 生成自然的参考运动
  2. 约束机器人动作空间
  3. 提高运动质量
""")
    
    print("\n运动先验训练:")
    steps = [
        "1. 收集人类运动数据 (AMASS)",
        "2. 训练VAE重建人类姿态",
        "3. 验证生成的运动自然性",
        "4. 保存预训练模型",
    ]
    for step in steps:
        print(f"  {step}")
    
    print("\n✓ 运动先验说明完成\n")


def example_teacher_student():
    """示例2: 教师-学生架构"""
    print("=" * 60)
    print("示例2: 教师-学生架构")
    print("=" * 60)
    
    print("""
教师-学生蒸馏:

问题: 直接训练机器人策略困难
- 高维动作空间探索困难
- 容易陷入局部最优
- 运动不自然

解决方案: 知识蒸馏

教师 (预训练 Motion Prior):
├── 冻结参数
├── 提供人类运动参考
└── 输出: 自然姿态

学生 (机器人策略):
├── 可训练参数
├── 输入: 机器人观测
├── 输出: 机器人动作
└── 目标: 模仿教师 + 完成任务

训练目标:
  L = α * ||student - teacher||² + β * L_task
      ↑ 蒸馏损失              ↑ 任务损失
""")
    
    print("\n架构对比:")
    print("  教师 (冻结):")
    print("    输入: 目标姿态描述")
    print("    输出: 人类参考姿态")
    print("    作用: 提供运动先验")
    print()
    print("  学生 (训练):")
    print("    输入: 机器人观测")
    print("    输出: 机器人动作")
    print("    目标: 自然运动 + 任务完成")
    
    print("\n✓ 教师-学生架构说明完成\n")


def example_nsdf():
    """示例3: NSDF几何感知"""
    print("=" * 60)
    print("示例3: 神经符号距离场 (NSDF)")
    print("=" * 60)
    
    print("""
NSDF: Neural Signed Distance Field

传统几何表示的问题:
- 网格表示: 不连续，查询效率低
- 点云表示: 稀疏，不完整
- 体素表示: 内存占用大

NSDF优势:
- 连续表示: 任意点距离查询
- 紧凑存储: 神经网络参数
- 梯度信息: 表面法向
- 隐式表面: 高分辨率

定义:
  SDF(x) = signed distance from point x to surface
  ├── SDF(x) < 0: x在物体内
  ├── SDF(x) = 0: x在表面上
  └── SDF(x) > 0: x在物体外

应用:
  1. 接触检测: SDF < threshold
  2. 碰撞避免: 最小化负SDF
  3. 接近引导: 跟随SDF梯度
""")
    
    print("\nNSDF网络结构:")
    print("  输入: 查询点 p (3D坐标)")
    print("  隐藏层: 64 → 64")
    print("  输出: 有符号距离 (标量)")
    print()
    print("  查询示例:")
    print("    query_point = [0.5, 0.2, 0.3]")
    print("    distance = nsdf.query(query_point)")
    print("    # distance = -0.05 (物体内部5cm)")
    
    print("\n✓ NSDF说明完成\n")


def example_embrace_strategy():
    """示例4: 全身拥抱策略"""
    print("=" * 60)
    print("示例4: 全身拥抱策略")
    print("=" * 60)
    
    print("""
全身拥抱 vs 传统抓取:

传统末端执行器抓取:
├── 接触点: 1-2个 (手指)
├── 稳定性: 低 (容易滑落)
├── 最大载荷: 5-10kg
├── 物体形状: 限制大
└── 控制: 简单

全身拥抱 (WBM):
├── 接触点: 8-12个 (双手+躯干)
├── 稳定性: 高 (多接触支撑)
├── 最大载荷: 30-50kg
├── 物体形状: 限制小
└── 控制: 复杂 (全身协调)

全身拥抱动作:
  1. 接近物体
     ├── 手臂张开准备
     └── NSDF引导接近

  2. 形成拥抱
     ├── 双臂环抱物体
     ├── 躯干前倾接触
     └── 多接触点建立

  3. 稳定保持
     ├── 力量分配优化
     ├── 防止滑落
     └── 平衡维持

  4. 搬运/放置
     ├── 协调移动
     └── 稳定释放
""")
    
    print("\n接触点分布:")
    contacts = [
        ("左手掌", "主要支撑"),
        ("右手掌", "主要支撑"),
        ("左前臂", "辅助稳定"),
        ("右前臂", "辅助稳定"),
        ("胸部", "躯干支撑"),
        ("腹部", "躯干支撑"),
        ("左肩", "侧面支撑"),
        ("右肩", "侧面支撑"),
    ]
    for part, role in contacts:
        print(f"  {part:10s}: {role}")
    
    print("\n✓ 全身拥抱策略说明完成\n")


def example_bulky_objects():
    """示例5: 大型物体类型"""
    print("=" * 60)
    print("示例5: 大型物体类型")
    print("=" * 60)
    
    print("支持的物体类型:")
    
    objects = [
        ("Box", "0.3-1.0m", "5-30kg", "⭐⭐", "标准箱子"),
        ("Cylinder", "0.3-0.6m", "5-40kg", "⭐⭐⭐", "桶状物体"),
        ("Sphere", "0.2-0.5m", "5-35kg", "⭐⭐⭐⭐", "球状物体"),
        ("Irregular", "混合", "10-50kg", "⭐⭐⭐⭐⭐", "不规则形状"),
    ]
    
    print(f"\n  {'类型':12s} {'尺寸':12s} {'质量':10s} {'难度':8s} {'描述'}")
    print("  " + "-" * 60)
    for obj_type, size, mass, diff, desc in objects:
        print(f"  {obj_type:12s} {size:12s} {mass:10s} {diff:8s} {desc}")
    
    print("\n物体属性随机化:")
    print("  - 尺寸: 宽、高、深")
    print("  - 质量: 影响惯性和稳定性")
    print("  - 摩擦: 影响抓取难度")
    print("  - 形状: 规则/不规则")
    
    print("\n✓ 大型物体说明完成\n")


def example_whole_body_control():
    """示例6: 全身控制架构"""
    print("=" * 60)
    print("示例6: 全身控制架构")
    print("=" * 60)
    
    print("""
全身控制观测:
  ├── 本体感觉 (Proprioception)
  │   ├── 关节位置 (19维)
  │   ├── 关节速度 (19维)
  │   └── 接触力 (12维)
  │
  ├── 物体状态
  │   ├── 位置 (3维)
  │   ├── 姿态 (4维四元数)
  │   └── 速度 (6维)
  │
  ├── NSDF特征
  │   ├── 距离场采样 (100点)
  │   └── 梯度信息 (100×3)
  │
  └── 任务信息
      ├── 目标位置 (3维)
      └── 目标姿态 (4维)

全身控制动作:
  ├── 左臂 (7维): 肩3 + 肘2 + 腕2
  ├── 右臂 (7维): 肩3 + 肘2 + 腕2
  ├── 躯干 (8维): 腰3 + 胸3 + 颈2
  ├── 头部 (2维): 俯仰 + 偏航
  └── 腿部 (5维): 辅助平衡

总动作维度: 29维
""")
    
    print("\n控制策略网络:")
    print("  输入: 150维 (观测)")
    print("  隐藏层: 512 → 256 → 128")
    print("  输出: 29维 (动作)")
    print("  激活: ReLU (隐藏), Tanh (输出)")
    
    print("\n奖励设计:")
    rewards = [
        ("接触建立", "成功建立接触点"),
        ("力量平衡", "力量均匀分布"),
        ("物体稳定", "物体不滑落"),
        ("运动自然", "符合运动先验"),
        ("能量效率", "最小化能量消耗"),
    ]
    for reward, desc in rewards:
        print(f"  {reward:12s}: {desc}")
    
    print("\n✓ 全身控制架构说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("WBM Embrace Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查依赖
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("警告: PyTorch 未安装")
    
    try:
        import genesis as gs
        print(f"Genesis 可用")
    except ImportError:
        print("警告: Genesis 未安装")
    
    print()
    
    # 运行示例
    examples = [
        ("运动先验", example_motion_prior),
        ("教师-学生", example_teacher_student),
        ("NSDF", example_nsdf),
        ("拥抱策略", example_embrace_strategy),
        ("大型物体", example_bulky_objects),
        ("全身控制", example_whole_body_control),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"{name} 示例失败: {e}\n")
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
