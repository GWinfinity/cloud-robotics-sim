"""
FB Model Training Algorithm

无监督预训练算法，学习通用的潜在表示
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
import copy

from models.fb_model import FBModel
from models.policy import LatentConditionedPolicy


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1000000,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 预分配内存
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
        self.goals = torch.zeros((capacity, state_dim), device=device)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        goal: np.ndarray
    ):
        """添加经验"""
        self.states[self.ptr] = torch.FloatTensor(state)
        self.actions[self.ptr] = torch.FloatTensor(action)
        self.next_states[self.ptr] = torch.FloatTensor(next_state)
        self.rewards[self.ptr] = torch.FloatTensor([reward])
        self.dones[self.ptr] = torch.FloatTensor([done])
        self.goals[self.ptr] = torch.FloatTensor(goal)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'next_states': self.next_states[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'goals': self.goals[indices]
        }


class FBTrainer:
    """
    FB模型训练器
    
    执行无监督预训练，学习通用的行为表示
    """
    
    def __init__(
        self,
        fb_model: FBModel,
        policy: LatentConditionedPolicy,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        
        # 模型
        self.fb_model = fb_model.to(device)
        self.policy = policy.to(device)
        
        # 优化器
        self.fb_optimizer = optim.Adam(
            list(fb_model.forward_model.parameters()) + 
            list(fb_model.backward_model.parameters()),
            lr=config['pretraining']['learning_rate'],
            weight_decay=config['pretraining']['weight_decay']
        )
        
        self.policy_optimizer = optim.Adam(
            policy.parameters(),
            lr=config['pretraining']['learning_rate'],
            weight_decay=config['pretraining']['weight_decay']
        )
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            capacity=config['pretraining']['buffer_size'],
            device=device
        )
        
        # 训练统计
        self.total_steps = 0
        self.episode_count = 0
    
    def train_step(self, batch_size: int) -> Dict[str, float]:
        """
        执行一步训练
        
        包括:
        1. FB模型更新
        2. 策略更新
        """
        if self.replay_buffer.size < batch_size:
            return {}
        
        # 采样批次
        batch = self.replay_buffer.sample(batch_size)
        
        # ===== 更新FB模型 =====
        fb_losses = self._update_fb_model(batch)
        
        # ===== 更新策略 =====
        policy_losses = self._update_policy(batch)
        
        # ===== 软更新目标网络 =====
        if self.total_steps % self.config['pretraining']['target_update_freq'] == 0:
            tau = self.config['pretraining']['target_tau']
            self.fb_model.update_target_networks(tau)
        
        self.total_steps += 1
        
        # 合并损失
        losses = {**fb_losses, **policy_losses}
        return losses
    
    def _update_fb_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新FB模型"""
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        rewards = batch['rewards']
        dones = batch['dones']
        goals = batch['goals']
        
        # 计算FB损失
        fb_loss_dict = self.fb_model.compute_fb_loss(
            state=states,
            action=actions,
            next_state=next_states,
            goal_state=goals,
            reward=rewards,
            done=dones,
            gamma=self.config['pretraining']['gamma']
        )
        
        # 总损失
        fb_loss = (
            fb_loss_dict['fb_loss'] +
            self.config['pretraining']['ortho_loss_weight'] * fb_loss_dict['ortho_loss'] +
            self.config['pretraining'].get('diversity_loss_weight', 0.01) * fb_loss_dict['diversity_loss']
        )
        
        # 反向传播
        self.fb_optimizer.zero_grad()
        fb_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.fb_model.forward_model.parameters()) + 
            list(self.fb_model.backward_model.parameters()),
            max_norm=1.0
        )
        self.fb_optimizer.step()
        
        return {
            'fb_loss': fb_loss_dict['fb_loss'].item(),
            'fb_ortho_loss': fb_loss_dict['ortho_loss'].item(),
            'fb_diversity_loss': fb_loss_dict['diversity_loss'].item(),
            'fb_alignment': fb_loss_dict['fb_alignment'].item()
        }
    
    def _update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新策略"""
        states = batch['states']
        actions = batch['actions']
        goals = batch['goals']
        rewards = batch['rewards']
        
        batch_size = states.shape[0]
        
        # 获取目标潜在向量
        with torch.no_grad():
            z_goals = self.fb_model.backward_model(goals, rewards)
        
        # 前向传播获取动作
        policy_output = self.policy(states, z_goals)
        pred_actions = policy_output['action']
        
        # ===== 行为克隆损失 (基于FB表示) =====
        # 使用FB对齐分数作为权重
        with torch.no_grad():
            z_forward = self.fb_model.forward_model(states, actions)
            z_backward = self.fb_model.backward_model(goals, rewards)
            
            # 对齐分数: 高分数表示动作有利于到达目标
            alignment = (z_forward * z_backward).sum(dim=-1, keepdim=True)
            weights = torch.sigmoid(alignment)  # 转换为权重
        
        # 加权行为克隆
        bc_loss = F.mse_loss(pred_actions, actions, reduction='none').mean(dim=-1)
        weighted_bc_loss = (bc_loss * weights.squeeze()).mean()
        
        # ===== 熵奖励 =====
        mean = policy_output['mean']
        std = policy_output['std']
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().mean()
        
        entropy_coef = self.config['pretraining']['entropy_coef']
        
        # 总策略损失
        policy_loss = weighted_bc_loss - entropy_coef * entropy
        
        # 反向传播
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'policy_bc_loss': weighted_bc_loss.item(),
            'policy_entropy': entropy.item()
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'fb_model': self.fb_model.state_dict(),
            'policy': self.policy.state_dict(),
            'fb_optimizer': self.fb_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.fb_model.load_state_dict(checkpoint['fb_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.fb_optimizer.load_state_dict(checkpoint['fb_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.total_steps = checkpoint['total_steps']


class FBOptimizer:
    """
    基于优化的适应方法
    
    用于few-shot适应新任务
    通过优化潜在向量来适应目标任务
    """
    
    def __init__(
        self,
        fb_model: FBModel,
        policy: LatentConditionedPolicy,
        num_steps: int = 100,
        lr: float = 0.01
    ):
        self.fb_model = fb_model
        self.policy = policy
        self.num_steps = num_steps
        self.lr = lr
    
    def adapt(
        self,
        target_task_data: List[Dict],
        initial_latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        适应新任务
        
        Args:
            target_task_data: 少量目标任务数据
            initial_latent: 初始潜在向量
            
        Returns:
            优化后的潜在向量
        """
        # 初始化潜在向量
        if initial_latent is None:
            latent = torch.randn(1, self.fb_model.latent_dim, requires_grad=True)
        else:
            latent = initial_latent.clone().requires_grad_(True)
        
        optimizer = optim.Adam([latent], lr=self.lr)
        
        # 优化循环
        for step in range(self.num_steps):
            total_loss = 0
            
            for data in target_task_data:
                state = data['state']
                action = data['action']
                reward = data['reward']
                
                # 前向传播
                z_f = self.fb_model.forward_model(state, action)
                
                # 对齐损失: 使latent与F表示对齐
                alignment = (z_f * latent).sum()
                
                # 奖励预测损失
                reward_pred = alignment  # 简化为内积
                reward_loss = F.mse_loss(reward_pred, reward)
                
                total_loss += reward_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return latent.detach()
