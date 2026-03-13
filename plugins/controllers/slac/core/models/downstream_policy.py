"""
Downstream Task Policy

使用预训练的潜在动作空间学习下游任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class DownstreamPolicy(nn.Module):
    """
    下游任务策略
    
    输入: 任务相关观测
    输出: 潜在动作
    
    潜在动作 -> 潜在动作空间 -> 原始动作序列
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 8,
        hidden_dims: list = [256, 256],
        activation: str = 'relu',
        min_std: float = 0.1,
        max_std: float = 1.0
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.min_std = min_std
        self.max_std = max_std
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 输出潜在动作
        self.mean_head = nn.Linear(prev_dim, latent_dim)
        self.log_std_head = nn.Linear(prev_dim, latent_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: [batch, obs_dim]
            deterministic: 是否确定性输出
            
        Returns:
            dict: 包含潜在动作、均值、标准差
        """
        features = self.feature_net(obs)
        
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), np.log(min_std), np.log(max_std))
        std = torch.exp(log_std)
        
        if deterministic:
            latent_action = mean
        else:
            latent_action = mean + torch.randn_like(mean) * std
        
        return {
            'latent_action': latent_action,
            'mean': mean,
            'std': std,
            'log_prob': None  # 计算需要额外步骤
        }
    
    def get_latent_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """获取潜在动作 (numpy接口)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            output = self.forward(obs_t, deterministic)
            latent_action = output['latent_action'].cpu().numpy()[0]
        
        return latent_action


class QNetwork(nn.Module):
    """Q网络 (用于SAC)"""
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 8,
        hidden_dims: list = [256, 256],
        activation: str = 'relu'
    ):
        super().__init__()
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 网络
        layers = []
        prev_dim = obs_dim + latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """Q值估计"""
        x = torch.cat([obs, latent_action], dim=-1)
        return self.network(x).squeeze(-1)


class LatentSAC:
    """
    Soft Actor-Critic with Latent Actions
    
    使用潜在动作空间的SAC算法
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 8,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # 策略网络
        self.policy = DownstreamPolicy(obs_dim, latent_dim).to(device)
        
        # Q网络 (双Q)
        self.q1 = QNetwork(obs_dim, latent_dim).to(device)
        self.q2 = QNetwork(obs_dim, latent_dim).to(device)
        
        # 目标Q网络
        self.q1_target = QNetwork(obs_dim, latent_dim).to(device)
        self.q2_target = QNetwork(obs_dim, latent_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 温度参数
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.target_entropy = -latent_dim
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=q_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            output = self.policy(obs_t, deterministic)
            latent_action = output['latent_action'].cpu().numpy()[0]
        
        return latent_action
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """更新网络"""
        obs = batch['obs'].to(self.device)
        latent_action = batch['latent_action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        
        # ===== 更新Q网络 =====
        with torch.no_grad():
            # 下一个潜在动作
            next_output = self.policy(next_obs)
            next_latent_action = next_output['latent_action']
            next_log_prob = self._compute_log_prob(next_output)
            
            # 目标Q值
            next_q1 = self.q1_target(next_obs, next_latent_action)
            next_q2 = self.q2_target(next_obs, next_latent_action)
            next_q = torch.min(next_q1, next_q2)
            
            alpha = self.log_alpha.exp()
            next_q = next_q - alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # 当前Q值
        current_q1 = self.q1(obs, latent_action)
        current_q2 = self.q2(obs, latent_action)
        
        # Q损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ===== 更新策略 =====
        output = self.policy(obs)
        new_latent_action = output['latent_action']
        log_prob = self._compute_log_prob(output)
        
        new_q1 = self.q1(obs, new_latent_action)
        new_q2 = self.q2(obs, new_latent_action)
        new_q = torch.min(new_q1, new_q2)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_prob - new_q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ===== 更新温度 =====
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
    
    def _compute_log_prob(self, policy_output: Dict) -> torch.Tensor:
        """计算对数概率 (简化版)"""
        # 这里应该使用更精确的分布计算
        # 简化为使用标准差估计
        std = policy_output['std']
        log_prob = -0.5 * torch.sum(torch.log(2 * np.pi * std ** 2) + 1, dim=-1)
        return log_prob
    
    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'log_alpha': self.log_alpha.item()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=self.device)
