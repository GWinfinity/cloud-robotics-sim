"""
Human Motion Prior using Conditional VAE

从大规模人类运动数据学习先验，生成运动学自然、物理可行的全身运动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class MotionPriorVAE(nn.Module):
    """
    条件变分自编码器 (Conditional VAE)
    
    输入: 目标物体信息 (条件)
    输出: 人体姿态 (关节位置)
    """
    
    def __init__(
        self,
        input_dim: int = 69,           # 23 joints x 3
        latent_dim: int = 32,
        condition_dim: int = 64,       # 物体条件信息
        encoder_hidden: list = [512, 256, 128],
        decoder_hidden: list = [128, 256, 512],
        beta: float = 0.001
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.beta = beta
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_output = nn.Linear(prev_dim, input_dim)
        
    def encode(self, pose: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码姿态到潜在空间"""
        x = torch.cat([pose, condition], dim=-1)
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """从潜在向量解码姿态"""
        x = torch.cat([z, condition], dim=-1)
        h = self.decoder(x)
        
        # 使用tanh限制输出范围
        output = torch.tanh(self.fc_output(h))
        
        return output
    
    def forward(self, pose: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码
        mu, logvar = self.encode(pose, condition)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_pose = self.decode(z, condition)
        
        return {
            'recon_pose': recon_pose,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def loss_function(self, recon_pose: torch.Tensor, target_pose: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        VAE损失函数
        
        包含:
        1. 重构损失 (MSE)
        2. KL散度 (正则化)
        """
        # 重构损失
        recon_loss = F.mse_loss(recon_pose, target_pose, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从条件分布采样姿态
        
        Args:
            condition: [condition_dim] 或 [batch, condition_dim]
            num_samples: 采样数量
            
        Returns:
            poses: [num_samples, input_dim] 或 [batch, num_samples, input_dim]
        """
        with torch.no_grad():
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).expand(num_samples, -1)
            
            # 从标准正态分布采样潜在向量
            z = torch.randn(num_samples, self.latent_dim).to(condition.device)
            
            # 解码
            poses = self.decode(z, condition)
        
        return poses
    
    def get_motion_naturalness_reward(self, pose: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        计算运动自然性奖励
        
        衡量给定姿态与运动先验的匹配程度
        用于强化学习的奖励塑造
        """
        with torch.no_grad():
            # 编码
            mu, logvar = self.encode(pose, condition)
            z = self.reparameterize(mu, logvar)
            
            # 重构
            recon_pose = self.decode(z, condition)
            
            # 重构误差越小，自然性越高
            recon_error = F.mse_loss(recon_pose, pose, reduction='none').mean(dim=-1)
            naturalness = torch.exp(-recon_error * 10)  # 转换为奖励
        
        return naturalness
    
    def interpolate(self, pose1: torch.Tensor, pose2: torch.Tensor, 
                    condition: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        在两个姿态之间插值
        
        在潜在空间进行平滑插值
        """
        with torch.no_grad():
            # 编码到潜在空间
            mu1, _ = self.encode(pose1, condition)
            mu2, _ = self.encode(pose2, condition)
            
            # 潜在空间插值
            alphas = torch.linspace(0, 1, num_steps).to(pose1.device)
            interpolated_poses = []
            
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                pose = self.decode(z, condition)
                interpolated_poses.append(pose)
            
            return torch.stack(interpolated_poses, dim=0)


class HumanMotionDataset:
    """
    人类运动数据集
    
    用于预训练Motion Prior
    """
    
    def __init__(self, motion_data: np.ndarray, object_conditions: np.ndarray):
        """
        Args:
            motion_data: [N, num_joints, 3] 人体姿态序列
            object_conditions: [N, condition_dim] 对应的物体条件
        """
        self.motion_data = torch.FloatTensor(motion_data)
        self.object_conditions = torch.FloatTensor(object_conditions)
    
    def __len__(self):
        return len(self.motion_data)
    
    def __getitem__(self, idx):
        pose = self.motion_data[idx].flatten()  # 展平关节位置
        condition = self.object_conditions[idx]
        return pose, condition


class MotionPriorTrainer:
    """运动先验训练器"""
    
    def __init__(self, model: MotionPriorVAE, lr: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_step(self, pose_batch: torch.Tensor, condition_batch: torch.Tensor) -> Dict[str, float]:
        """训练一步"""
        # 前向传播
        output = self.model(pose_batch, condition_batch)
        
        # 计算损失
        losses = self.model.loss_function(
            output['recon_pose'],
            pose_batch,
            output['mu'],
            output['logvar']
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {
            'total_loss': losses['total_loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'kl_loss': losses['kl_loss'].item()
        }
