"""
Divide-and-Conquer Policy Distillation

分而治之的策略蒸馏框架
单任务专家 → 通用多任务策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class ExpertPolicy(nn.Module):
    """
    单任务专家策略
    
    为特定任务训练的专家策略
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        task_name: str,
        hidden_dims: list = [512, 512, 256]
    ):
        super().__init__()
        
        self.task_name = task_name
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_net(obs)
        mean = torch.tanh(self.mean_head(features))
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return {
            'mean': mean,
            'std': std,
            'action': mean + torch.randn_like(mean) * std
        }
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = self.forward(obs_t)['action'].cpu().numpy()[0]
        return action


class TaskEmbedding(nn.Module):
    """任务嵌入"""
    
    def __init__(self, num_tasks: int, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embedding_dim)
    
    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_id)


class MultiTaskPolicy(nn.Module):
    """
    多任务策略
    
    通过任务条件化支持多种任务
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_tasks: int = 3,
        task_embedding_dim: int = 16,
        hidden_dims: list = [512, 512, 256],
        conditioning: str = 'film'  # 'concat', 'film', 'attention'
    ):
        super().__init__()
        
        self.conditioning = conditioning
        
        # 任务嵌入
        self.task_embedding = TaskEmbedding(num_tasks, task_embedding_dim)
        
        # 网络
        if conditioning == 'concat':
            # 简单拼接
            input_dim = obs_dim + task_embedding_dim
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.feature_net = nn.Sequential(*layers)
        
        elif conditioning == 'film':
            # FiLM conditioning
            self.feature_net = self._build_film_network(
                obs_dim, task_embedding_dim, hidden_dims
            )
        
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def _build_film_network(self, obs_dim, task_dim, hidden_dims):
        """构建FiLM网络"""
        # FiLM: Feature-wise Linear Modulation
        # 使用任务嵌入生成scale和shift参数
        
        layers = []
        prev_dim = obs_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # FiLM层: 为每个隐藏层生成scale和shift
            layers.append(FiLMLayer(hidden_dim, task_dim))
            
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        obs: torch.Tensor,
        task_id: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测
            task_id: 任务ID [batch] 或 [batch, 1]
        """
        # 任务嵌入
        task_embed = self.task_embedding(task_id)
        
        if self.conditioning == 'concat':
            x = torch.cat([obs, task_embed], dim=-1)
            features = self.feature_net(x)
        else:
            features = self.feature_net(obs)  # FiLM内部处理task_embed
        
        mean = torch.tanh(self.mean_head(features))
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return {
            'mean': mean,
            'std': std,
            'action': mean + torch.randn_like(mean) * std
        }


class FiLMLayer(nn.Module):
    """FiLM层"""
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.scale_net = nn.Linear(condition_dim, feature_dim)
        self.shift_net = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        return features * (1 + scale) + shift


class PolicyDistillation:
    """
    策略蒸馏
    
    将多个专家策略蒸馏为单一多任务策略
    """
    
    def __init__(
        self,
        expert_policies: List[ExpertPolicy],
        student_policy: MultiTaskPolicy,
        method: str = 'dagger',
        device: str = 'cuda'
    ):
        self.expert_policies = expert_policies
        self.student_policy = student_policy
        self.method = method
        self.device = device
        
        # 冻结专家策略
        for expert in expert_policies:
            for param in expert.parameters():
                param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.Adam(student_policy.parameters(), lr=3e-4)
    
    def distill_step(
        self,
        observations: torch.Tensor,
        task_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        蒸馏步骤
        
        Args:
            observations: [batch, obs_dim]
            task_ids: [batch] 每个样本对应的任务ID
        """
        observations = observations.to(self.device)
        task_ids = task_ids.to(self.device)
        
        # 学生策略输出
        student_output = self.student_policy(observations, task_ids)
        student_actions = student_output['mean']
        
        # 获取专家动作
        expert_actions = []
        for i, task_id in enumerate(task_ids):
            task_idx = task_id.item()
            expert = self.expert_policies[task_idx]
            with torch.no_grad():
                expert_out = expert(observations[i:i+1])
                expert_actions.append(expert_out['mean'])
        
        expert_actions = torch.cat(expert_actions, dim=0)
        
        # 蒸馏损失 (MSE)
        distill_loss = F.mse_loss(student_actions, expert_actions)
        
        # 熵奖励 (鼓励探索)
        entropy = student_output['std'].mean()
        
        # 总损失
        loss = distill_loss - 0.01 * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'distill_loss': distill_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, path: str):
        """保存学生策略"""
        torch.save(self.student_policy.state_dict(), path)
    
    def load(self, path: str):
        """加载学生策略"""
        self.student_policy.load_state_dict(torch.load(path))
