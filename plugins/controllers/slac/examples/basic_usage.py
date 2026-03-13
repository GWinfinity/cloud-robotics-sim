"""
SLAC Plugin - 基础使用示例

演示如何使用 SLAC 进行潜在动作空间预训练和下游任务学习
"""

import numpy as np
import torch
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.controllers.slac import (
        SLACPretrainer, LatentActionVAE, SkillDiscovery,
        DownstreamPolicy, LatentActionController, MobileManipulatorEnv
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.models.latent_action import LatentActionVAE
    from core.models.skill_discovery import SkillDiscovery
    from core.models.downstream_policy import DownstreamPolicy
    from core.envs.mobile_manipulator import MobileManipulatorEnv
    from __init__ import SLACPretrainer, LatentActionController


def example_vae_architecture():
    """示例1: VAE架构说明"""
    print("=" * 60)
    print("示例1: 潜在动作VAE架构")
    print("=" * 60)
    
    print("""
SLAC核心: 潜在动作变分自编码器 (VAE)

原始动作空间 → 潜在动作空间 → 原始动作空间
(50-100维)    →  (4-16维)   →  (50-100维)

VAE编码器 (动作 → 潜在空间):
  输入: 原始动作 a_t (50-100维)
  MLP: [action_dim] → [256] → [128]
  输出: 潜在动作的均值 μ 和方差 σ
  采样: z_t = μ + σ * ε,  ε ~ N(0,1)

VAE解码器 (潜在空间 → 动作):
  输入: 潜在动作 z_t (4-16维) + 观测 o_t
  MLP: [latent_dim + obs_dim] → [128] → [256]
  输出: 重建动作 â_t (50-100维)

损失函数:
  L_VAE = ||a_t - â_t||² + β * KL(q(z|a) || p(z))
        ↑ 重建损失        ↑ KL散度正则化
""")
    
    print("压缩比对比:")
    configs = [
        ("高压缩", 100, 4, 25.0),
        ("中压缩", 50, 8, 6.25),
        ("低压缩", 50, 16, 3.125),
    ]
    for name, action_dim, latent_dim, ratio in configs:
        print(f"  {name}: {action_dim}D → {latent_dim}D (压缩比 {ratio}x)")
    
    print("\n✓ VAE架构说明完成\n")


def example_skill_discovery():
    """示例2: 技能发现机制"""
    print("=" * 60)
    print("示例2: 无监督技能发现")
    print("=" * 60)
    
    print("""
DIAYN风格的技能发现:

目标: 学习多样化的潜在动作技能 (无需任务奖励)

判别器 D(s, z):
  输入: 状态 s + 潜在动作 z
  目标: 预测使用了哪个技能
  损失: 最大化互信息 I(s; z)

策略 π(z):
  目标: 生成使判别器困惑的潜在动作
        (即最大化状态多样性)
  奖励: -log D(s, z)  (判别器不确定性)

训练循环:
  1. 采样潜在动作 z ~ π(o)
  2. 执行动作, 观察下一状态 s'
  3. 判别器更新: 提高 D(s', z) 准确性
  4. 策略更新: 最大化 -log D(s', z)

结果: 每个潜在动作维度对应一个独立技能
""")
    
    print("\n技能示例:")
    skills = [
        ("z[0]", "左右移动", "-1.0 到 +1.0"),
        ("z[1]", "前后移动", "-1.0 到 +1.0"),
        ("z[2]", "手臂伸展", "0.0 到 1.0"),
        ("z[3]", "抓取动作", "0.0 到 1.0"),
    ]
    for dim, description, range_val in skills:
        print(f"  {dim}: {description} (范围: {range_val})")
    
    print("\n✓ 技能发现机制说明完成\n")


def example_three_stage_pipeline():
    """示例3: 三阶段流程"""
    print("=" * 60)
    print("示例3: SLAC三阶段流程")
    print("=" * 60)
    
    print("""
阶段1: 仿真预训练 (任务无关)
─────────────────────────────
环境: 低保真仿真器
目标: 学习通用的潜在动作空间
数据: 无监督交互 (无任务奖励)
时间: 2-4小时
输出: 训练好的VAE权重

阶段2: 技能发现 (可选增强)
─────────────────────────────
目标: 增强潜在动作的可解释性
方法: DIAYN风格多样性驱动
结果: 每个维度对应一个独立技能

阶段3: 真实世界下游学习 (任务相关)
─────────────────────────────
环境: 真实机器人
目标: 学习具体任务 (如抓取)
方法: 在潜在空间训练策略
输入: 观测 → 输出: 潜在动作
时间: < 1小时
关键: VAE参数冻结!
""")
    
    print("\n阶段对比:")
    stages = [
        ("阶段1", "仿真", "无监督", "2-4h", "VAE"),
        ("阶段2", "仿真", "无监督", "1-2h", "技能"),
        ("阶段3", "真实", "任务奖励", "<1h", "策略"),
    ]
    for stage, env, reward, time, output in stages:
        print(f"  {stage}: {env} | {reward} | {time} | 输出: {output}")
    
    print("\n✓ 三阶段流程说明完成\n")


def example_latent_action_benefits():
    """示例4: 潜在动作的优势"""
    print("=" * 60)
    print("示例4: 潜在动作的优势")
    print("=" * 60)
    
    print("""
1. 时间抽象 (Temporal Abstraction)
   原始动作: 每步都需决策 (20-50Hz)
   潜在动作: 可以持续多步 (5-10Hz)
   优势: 减少决策频率，平滑动作

2. 安全探索
   原始空间: 50维空间探索 → 危险动作
   潜在空间: 8维空间探索 → 安全约束
   优势: 避免危险关节配置

3. 样本效率
   原始空间: 需要 1000+ 真实世界样本
   潜在空间: 仅需 100-200 样本
   优势: 10倍样本效率提升

4. 迁移学习
   预训练的VAE可用于多个任务
   不同任务只需学习不同的下游策略
   优势: 知识复用

5. 可解释性
   每个潜在维度对应特定行为
   例如: z[0]=移动, z[1]=抓取
   优势: 理解策略行为
""")
    
    print("\n对比实验 (模拟):")
    comparisons = [
        ("指标", "标准RL", "SLAC"),
        ("真实世界样本", "1000+", "100-200"),
        ("学习时间", "5-10h", "<1h"),
        ("安全违规", "频繁", "极少"),
        ("策略平滑性", "抖动", "平滑"),
    ]
    for row in comparisons:
        print(f"  {row[0]:20s} | {row[1]:15s} | {row[2]}")
    
    print("\n✓ 潜在动作优势说明完成\n")


def example_downstream_training():
    """示例5: 下游任务学习"""
    print("=" * 60)
    print("示例5: 下游任务学习")
    print("=" * 60)
    
    print("""
下游学习流程:

1. 加载预训练VAE
   vae = LatentActionVAE(action_dim=50, latent_dim=8)
   vae.load_state_dict(torch.load('slac_pretrained.pt'))
   vae.eval()  # 冻结!

2. 创建下游策略
   policy = DownstreamPolicy(
       obs_dim=100,
       latent_action_dim=8  # 使用潜在动作!
   )

3. 真实世界训练循环
   for episode in range(100):
       obs = env.reset()
       for step in range(100):
           # 策略输出潜在动作
           latent_action = policy(obs)
           
           # VAE解码为原始动作
           with torch.no_grad():
               action = vae.decode(latent_action, obs)
           
           # 执行并观察
           next_obs, reward, done, _ = env.step(action)
           
           # 只更新策略 (VAE不变!)
           policy.update(obs, latent_action, reward, next_obs)
           obs = next_obs

关键: VAE参数完全冻结!
     只学习 obs → latent_action 的映射
""")
    
    print("\n下游策略网络:")
    print("  输入: 观测 (100维)")
    print("  隐藏层: [256] → [128]")
    print("  输出: 潜在动作 (8维)")
    print("  激活函数: ReLU")
    
    print("\n✓ 下游任务学习说明完成\n")


def example_safety_mechanisms():
    """示例6: 安全机制"""
    print("=" * 60)
    print("示例6: 安全机制")
    print("=" * 60)
    
    print("""
SLAC的安全保障:

1. 潜在空间约束
   - 潜在动作范围: [-1, 1]
   - 通过tanh激活函数限制
   - 避免极端原始动作

2. VAE重建验证
   - 监控重建误差
   - 误差过大时触发安全动作
   - 确保潜在→原始映射可靠

3. 动作平滑性
   - 潜在空间变化平滑
   - 原始动作自然平滑
   - 减少机械冲击

4. 关节限制保护
   - 解码后动作裁剪到关节限制
   - 避免关节超限
   - 防止硬件损坏

5. 紧急停止
   - 检测异常状态
   - 触发保护姿态
   - 安全停机
""")
    
    print("\n安全检查清单:")
    checks = [
        "☐ 潜在动作范围检查",
        "☐ VAE重建误差监控",
        "☐ 关节限制验证",
        "☐ 速度/力矩限制",
        "☐ 碰撞检测",
        "☐ 紧急停止按钮",
    ]
    for check in checks:
        print(f"  {check}")
    
    print("\n✓ 安全机制说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("SLAC Plugin - 使用示例")
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
    
    # 运行示例 (这些主要是概念说明，不依赖实际环境)
    examples = [
        ("VAE架构", example_vae_architecture),
        ("技能发现", example_skill_discovery),
        ("三阶段流程", example_three_stage_pipeline),
        ("潜在动作优势", example_latent_action_benefits),
        ("下游训练", example_downstream_training),
        ("安全机制", example_safety_mechanisms),
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
