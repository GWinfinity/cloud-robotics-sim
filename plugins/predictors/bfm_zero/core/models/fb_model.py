"""
Forward-Backward (FB) Model Implementation

FB模型的核心思想:
- Forward (F): 从 (s, a) -> z (潜在表示)
- Backward (B): 从 (s_g, r) -> z (目标/奖励表示)
- 两者在潜在空间对齐，实现目标条件控制

参考: "Forward-Backward Representation Learning for Online and Offline RL"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class ForwardModel(nn.Module):
    """
    Forward Representation F(s, a) -> z
    
    编码当前状态-动作对到潜在空间
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        state_hidden: list = [512, 512, 256],
        action_hidden: list = [256, 256],
        combined_hidden: list = [512, 256],
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # 状态编码器
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in state_hidden:
            state_layers.append(nn.Linear(prev_dim, hidden_dim))
            state_layers.append(self.activation)
            prev_dim = hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)
        
        # 动作编码器
        action_layers = []
        prev_dim = action_dim
        for hidden_dim in action_hidden:
            action_layers.append(nn.Linear(prev_dim, hidden_dim))
            action_layers.append(self.activation)
            prev_dim = hidden_dim
        self.action_encoder = nn.Sequential(*action_layers)
        
        # 联合编码器 (状态+动作)
        combined_input_dim = state_hidden[-1] + action_hidden[-1]
        combined_layers = []
        prev_dim = combined_input_dim
        for hidden_dim in combined_hidden:
            combined_layers.append(nn.Linear(prev_dim, hidden_dim))
            combined_layers.append(self.activation)
            prev_dim = hidden_dim
        combined_layers.append(nn.Linear(prev_dim, latent_dim))
        self.combined_encoder = nn.Sequential(*combined_layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            
        Returns:
            z: [batch, latent_dim] - 潜在表示
        """
        # 分别编码状态和动作
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        
        # 拼接
        combined = torch.cat([state_feat, action_feat], dim=-1)
        
        # 编码到潜在空间
        z = self.combined_encoder(combined)
        
        # L2归一化 (重要: FB模型通常归一化潜在向量)
        z = F.normalize(z, p=2, dim=-1)
        
        return z


class BackwardModel(nn.Module):
    """
    Backward Representation B(s_g, r) -> z
    
    编码目标状态和奖励到潜在空间
    作为任务提示(prompt)使用
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 256,
        goal_hidden: list = [512, 512, 256],
        activation: str = 'elu',
        use_reward: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.use_reward = use_reward
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # 输入维度 (状态 + 可选的奖励)
        input_dim = state_dim + (1 if use_reward else 0)
        
        # 目标编码器
        goal_layers = []
        prev_dim = input_dim
        for hidden_dim in goal_hidden:
            goal_layers.append(nn.Linear(prev_dim, hidden_dim))
            goal_layers.append(self.activation)
            prev_dim = hidden_dim
        goal_layers.append(nn.Linear(prev_dim, latent_dim))
        self.goal_encoder = nn.Sequential(*goal_layers)
        
    def forward(
        self,
        goal_state: torch.Tensor,
        reward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            goal_state: [batch, state_dim] - 目标状态
            reward: [batch, 1] - 奖励 (可选)
            
        Returns:
            z: [batch, latent_dim] - 潜在表示
        """
        # 拼接奖励 (如果使用)
        if self.use_reward and reward is not None:
            # 确保reward是二维的
            if reward.dim() == 1:
                reward = reward.unsqueeze(-1)
            input_feat = torch.cat([goal_state, reward], dim=-1)
        else:
            input_feat = goal_state
        
        # 编码到潜在空间
        z = self.goal_encoder(input_feat)
        
        # L2归一化
        z = F.normalize(z, p=2, dim=-1)
        
        return z


class FBModel(nn.Module):
    """
    完整的 Forward-Backward 模型
    
    结合 F 和 B 网络，提供统一的接口
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.latent_dim = config['latent_dim']
        
        # 创建F和B网络
        fb_config = config['fb_model']
        
        self.forward_model = ForwardModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            **fb_config['forward']
        )
        
        self.backward_model = BackwardModel(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            **fb_config['backward']
        )
        
        # 目标网络 (用于稳定训练)
        self.forward_target = ForwardModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            **fb_config['forward']
        )
        
        self.backward_target = BackwardModel(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            **fb_config['backward']
        )
        
        # 初始化目标网络
        self.forward_target.load_state_dict(self.forward_model.state_dict())
        self.backward_target.load_state_dict(self.backward_model.state_dict())
        
        # 冻结目标网络
        for param in self.forward_target.parameters():
            param.requires_grad = False
        for param in self.backward_target.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal_state: torch.Tensor,
        reward: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            dict: 包含F和B的潜在表示
        """
        # Forward表示
        z_forward = self.forward_model(state, action)
        
        # Backward表示
        z_backward = self.backward_model(goal_state, reward)
        
        return {
            'z_forward': z_forward,
            'z_backward': z_backward,
            'similarity': (z_forward * z_backward).sum(dim=-1)  # 余弦相似度
        }
    
    def compute_fb_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        goal_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """
        计算FB损失
        
        核心思想: 对齐F(s,a)和B(s_g)使得它们的内积近似Q函数
        
        Returns:
            dict: 各种损失组件
        """
        batch_size = state.shape[0]
        
        # 当前 (s, a) 的F表示
        z_f = self.forward_model(state, action)
        
        # 目标 (s_g) 的B表示
        z_b = self.backward_model(goal_state, reward)
        
        # 下一状态的F表示 (用于TD学习)
        with torch.no_grad():
            # 使用目标网络
            z_f_next = self.forward_target(next_state, action)  # 简化为相同动作
            z_b_next = self.backward_target(goal_state, reward)
        
        # ===== FB对齐损失 =====
        # F和B的内积应该近似 Successor Feature
        fb_alignment = (z_f * z_b).sum(dim=-1)
        
        # TD目标
        with torch.no_grad():
            # 下一状态的F-B对齐
            td_target = (z_f_next * z_b_next).sum(dim=-1)
            td_target = reward.squeeze(-1) + gamma * td_target * (1 - done)
        
        # FB损失 (MSE)
        fb_loss = F.mse_loss(fb_alignment, td_target)
        
        # ===== 正交性损失 =====
        # 鼓励不同的B向量相互正交 (提高表示能力)
        ortho_loss = 0.0
        if batch_size > 1:
            # B向量的内积矩阵
            b_similarity = torch.mm(z_b, z_b.t())
            # 理想情况下应该是单位矩阵
            identity = torch.eye(batch_size, device=z_b.device)
            ortho_loss = F.mse_loss(b_similarity, identity)
        
        # ===== 多样性损失 =====
        # 鼓励F向量有较大的方差
        f_mean = z_f.mean(dim=0, keepdim=True)
        f_var = ((z_f - f_mean) ** 2).mean()
        diversity_loss = -f_var  # 最大化方差
        
        return {
            'fb_loss': fb_loss,
            'ortho_loss': ortho_loss,
            'diversity_loss': diversity_loss,
            'fb_alignment': fb_alignment.mean()
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """软更新目标网络"""
        for param, target_param in zip(
            self.forward_model.parameters(),
            self.forward_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
        
        for param, target_param in zip(
            self.backward_model.parameters(),
            self.backward_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def get_task_embedding(
        self,
        goal_state: torch.Tensor,
        reward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取任务嵌入 (用于提示策略)
        
        这是BFM-Zero的核心: 将任务描述转换为潜在向量
        """
        with torch.no_grad():
            z_task = self.backward_model(goal_state, reward)
        return z_task


class SuccessorFeatureFB(nn.Module):
    """
    基于Successor Features的FB模型变体
    
    为连续动作空间优化
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        num_features: int = 32
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # 特征提取器 (phi)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, num_features)
        )
        
        # Successor Feature网络 (psi)
        self.sf_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, num_features)
        )
        
    def forward(self, state, action, next_state):
        """计算Successor Features"""
        # 当前特征
        phi = self.feature_extractor(next_state)
        
        # Successor features
        sa = torch.cat([state, action], dim=-1)
        psi = self.sf_network(sa)
        
        return phi, psi
