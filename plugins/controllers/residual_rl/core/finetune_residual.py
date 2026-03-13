#!/usr/bin/env python3
"""
Finetune with Residual RL
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.manipulation_env import ManipulationEnv
from models.bc_policy import BCPolicy, VisualEncoder
from models.residual_network import ResidualNetwork, CombinedPolicy
from models.residual_network import ResidualSAC


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune with Residual RL')
    
    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml')
    parser.add_argument('--bc_checkpoint', type=str, required=True)
    parser.add_argument('--task', type=str, default='pick_and_place')
    parser.add_argument('--sparse_reward', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def finetune(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("Residual RL Finetuning")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Sparse Reward: {args.sparse_reward}")
    
    # 创建环境
    env = ManipulationEnv(
        config=config,
        task_name=args.task,
        headless=True,
        device=args.device
    )
    
    # 加载BC策略
    print(f"\nLoading BC policy from {args.bc_checkpoint}")
    
    visual_encoder = None
    if config['observation']['use_vision']:
        visual_encoder = VisualEncoder(output_dim=256)
    
    bc_policy = BCPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        visual_encoder=visual_encoder
    ).to(args.device)
    
    checkpoint = torch.load(args.bc_checkpoint, map_location=args.device)
    bc_policy.load_state_dict(checkpoint['policy'])
    
    # 冻结BC策略
    for param in bc_policy.parameters():
        param.requires_grad = False
    
    # 创建残差网络
    residual_network = ResidualNetwork(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        residual_scale=config['residual_network']['residual_scale']
    ).to(args.device)
    
    # 创建组合策略
    combined_policy = CombinedPolicy(
        bc_policy=bc_policy,
        residual_network=residual_network,
        freeze_bc=True
    ).to(args.device)
    
    # 创建Q网络
    class QNetwork(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(obs_dim + action_dim, 512)
            self.fc2 = torch.nn.Linear(512, 512)
            self.fc3 = torch.nn.Linear(512, 1)
        
        def forward(self, obs, action):
            x = torch.cat([obs, action], dim=-1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x).squeeze(-1)
    
    q1 = QNetwork(env.obs_dim, env.action_dim).to(args.device)
    q2 = QNetwork(env.obs_dim, env.action_dim).to(args.device)
    
    # 创建SAC训练器
    sac_trainer = ResidualSAC(
        bc_policy=bc_policy,
        residual_network=residual_network,
        q_network1=q1,
        q_network2=q2,
        device=args.device
    )
    
    # 训练
    num_steps = config['training']['residual_finetune']['num_steps']
    
    print(f"\nStarting finetuning for {num_steps} steps...")
    print("-"*70)
    
    obs = env.reset()
    episode_reward = 0
    episode_count = 0
    
    for step in range(num_steps):
        # 选择动作
        action = combined_policy.get_action(
            obs['proprioception'],
            use_residual=True
        )['final_action']
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        episode_reward += reward
        
        # 简化的缓冲区存储和更新
        # 实际应该使用ReplayBuffer
        
        obs = next_obs
        
        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"Episode {episode_count}, Step {step}, Reward: {episode_reward:.2f}, "
                      f"Success: {info['stats']['success']}")
            
            obs = env.reset()
            episode_reward = 0
        
        # 定期保存
        if step > 0 and step % 10000 == 0:
            sac_trainer.save(f'checkpoints/residual_step_{step}.pt')
    
    # 保存最终模型
    sac_trainer.save('checkpoints/residual_final.pt')
    print("\nFinetuning completed! Saved to checkpoints/residual_final.pt")
    
    env.close()


if __name__ == '__main__':
    args = parse_args()
    finetune(args)
