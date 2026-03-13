"""
Basic Usage Example: Residual RL

演示如何使用残差 RL 微调 BC 策略
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from cloud_robotics_sim.plugins.controllers.residual_rl import (
    ResidualNetwork, CombinedPolicy, BCPolicy, ResidualSAC
)


class MockQNetwork(torch.nn.Module):
    """模拟 Q 网络用于示例"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


def demo_basic_usage():
    """演示基础使用"""
    print("=" * 60)
    print("Residual RL - Basic Usage Demo")
    print("=" * 60)
    
    obs_dim = 100
    action_dim = 10
    
    # 1. 创建 BC 策略
    print("\n1. Creating BC Policy...")
    bc_policy = BCPolicy(obs_dim=obs_dim, action_dim=action_dim)
    print(f"  BC Policy: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # 2. 创建残差网络
    print("\n2. Creating Residual Network...")
    residual_net = ResidualNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        residual_scale=0.2,
        use_layer_norm=True
    )
    print(f"  Residual Network: residual_scale=0.2")
    
    # 3. 组合策略
    print("\n3. Creating Combined Policy...")
    policy = CombinedPolicy(
        bc_policy=bc_policy,
        residual_net=residual_net,
        freeze_bc=True
    )
    print(f"  Combined Policy: BC (frozen) + Residual (trainable)")
    
    # 4. 前向传播
    print("\n4. Forward Pass...")
    obs = torch.randn(1, obs_dim)
    output = policy(obs)
    
    print(f"  BC Action: {output['bc_action'].detach().numpy()[0][:5]}")
    print(f"  Residual:  {output['residual'].detach().numpy()[0][:5]}")
    print(f"  Final Action: {output['action'].detach().numpy()[0][:5]}")
    print(f"  Residual Norm: {output['residual'].norm().item():.4f}")


def demo_training_loop():
    """演示训练循环"""
    print("\n" + "=" * 60)
    print("Residual RL - Training Loop Demo")
    print("=" * 60)
    
    obs_dim = 100
    action_dim = 10
    
    # 创建策略和训练器
    bc_policy = BCPolicy(obs_dim, action_dim)
    residual_net = ResidualNetwork(obs_dim, action_dim, residual_scale=0.2)
    policy = CombinedPolicy(bc_policy, residual_net)
    
    q1 = MockQNetwork(obs_dim, action_dim)
    q2 = MockQNetwork(obs_dim, action_dim)
    
    trainer = ResidualSAC(
        combined_policy=policy,
        q_network1=q1,
        q_network2=q2,
        lr_residual=3e-4,
        lr_q=3e-4,
        gamma=0.99,
        tau=0.005
    )
    
    print("\n1. Training Setup:")
    print(f"  Policy: Combined (BC + Residual)")
    print(f"  Q Networks: 2 (twin Q)")
    print(f"  Learning Rate: residual={3e-4}, q={3e-4}")
    
    # 模拟训练
    print("\n2. Simulating Training...")
    for episode in range(5):
        obs = np.random.randn(obs_dim)
        
        # 选择动作
        action = trainer.select_action(obs)
        
        # 模拟环境交互
        next_obs = np.random.randn(obs_dim)
        reward = np.random.randn()
        done = False
        
        # 存储经验 (简化)
        # replay_buffer.add(obs, action, reward, next_obs, done)
        
        # 更新 (每4步)
        if episode % 4 == 0:
            # 模拟批次
            batch = {
                'obs': torch.randn(32, obs_dim),
                'action': torch.randn(32, action_dim),
                'reward': torch.randn(32),
                'next_obs': torch.randn(32, obs_dim),
                'done': torch.zeros(32)
            }
            losses = trainer.update(batch)
            print(f"  Episode {episode}: q1_loss={losses['q1_loss']:.4f}, "
                  f"residual_loss={losses['residual_loss']:.4f}")


def demo_residual_scale_schedule():
    """演示 residual_scale 课程调度"""
    print("\n" + "=" * 60)
    print("Residual Scale Curriculum Schedule")
    print("=" * 60)
    
    def get_residual_scale(episode):
        """逐步增大 ε"""
        if episode < 100:
            return 0.1   # 初始: 小范围探索
        elif episode < 500:
            return 0.2   # 中期: 增大改进空间
        else:
            return 0.3   # 后期: 允许更大调整
    
    print("\nEpisode -> Residual Scale:")
    for episode in [0, 50, 100, 250, 500, 750, 1000]:
        scale = get_residual_scale(episode)
        print(f"  Episode {episode:4d}: ε = {scale:.1f}")
    
    print("\n说明:")
    print("  - 早期小 ε: 保证安全性，只做微调")
    print("  - 后期大 ε: 允许更大改进，优化性能")


def demo_ab_comparison():
    """演示 A/B 对比: 纯 BC vs Residual RL"""
    print("\n" + "=" * 60)
    print("A/B Comparison: BC vs Residual RL")
    print("=" * 60)
    
    obs_dim = 100
    action_dim = 10
    
    # 创建两种策略
    bc_policy = BCPolicy(obs_dim, action_dim)
    residual_net = ResidualNetwork(obs_dim, action_dim, residual_scale=0.2)
    
    # A: 纯 BC
    bc_only = lambda obs: bc_policy(obs)['action']
    
    # B: BC + Residual
    combined = CombinedPolicy(bc_policy, residual_net)
    residual_rl = lambda obs: combined(obs)['action']
    
    print("\n1. BC Only (Variant A):")
    obs = torch.randn(1, obs_dim)
    bc_action = bc_only(obs)
    print(f"   Action: {bc_action.detach().numpy()[0][:5]}")
    print(f"   Range: [{bc_action.min():.2f}, {bc_action.max():.2f}]")
    
    print("\n2. Residual RL (Variant B):")
    residual_action = residual_rl(obs)
    print(f"   Action: {residual_action.detach().numpy()[0][:5]}")
    print(f"   Range: [{residual_action.min():.2f}, {residual_action.max():.2f}]")
    
    print("\n3. Difference:")
    diff = (residual_action - bc_action).detach().numpy()[0][:5]
    print(f"   Residual: {diff}")
    print(f"   Residual Norm: {torch.norm(residual_action - bc_action).item():.4f}")
    print(f"   Max Residual (should < 0.2): {torch.abs(residual_action - bc_action).max().item():.4f}")


if __name__ == '__main__':
    demo_basic_usage()
    demo_training_loop()
    demo_residual_scale_schedule()
    demo_ab_comparison()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
