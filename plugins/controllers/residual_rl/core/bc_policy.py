"""
Behavior Cloning (BC) Policy

基础策略，从人类演示数据学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class VisualEncoder(nn.Module):
    """
    视觉编码器
    
    将图像观测编码为特征向量
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 256,
        image_size: int = 224
    ):
        super().__init__()
        
        # 简化的CNN编码器 (ResNet18风格的简化版)
        self.conv_layers = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 特征投影
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            image: [batch, channels, height, width]
            
        Returns:
            features: [batch, output_dim]
        """
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BCPolicy(nn.Module):
    """
    行为克隆策略
    
    从演示数据学习映射: 观测 -> 动作
    作为残差学习的基础策略 (黑盒)
    """
    
    def __init__(
        self,
        obs_dim: int = 512,           # 视觉+本体感觉特征
        action_dim: int = 50,         # 动作维度
        visual_encoder: Optional[nn.Module] = None,
        hidden_dims: list = [512, 512, 256],
        activation: str = 'relu',
        output_activation: str = 'tanh',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 视觉编码器
        self.visual_encoder = visual_encoder
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 策略网络 (MLP)
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*layers)
        
        # 输出层
        self.mean_head = nn.Linear(prev_dim, action_dim)
        
        # 输出激活
        if output_activation == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        else:
            self.output_activation = lambda x: x
    
    def forward(
        self,
        obs: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测特征 (如果无视觉编码器) 或 本体感觉
            image: 图像观测 (可选)
            
        Returns:
            dict: 动作分布
        """
        # 如果有图像，先编码
        if self.visual_encoder is not None and image is not None:
            visual_features = self.visual_encoder(image)
            
            # 合并视觉和本体感觉
            if obs is not None:
                features = torch.cat([visual_features, obs], dim=-1)
            else:
                features = visual_features
        else:
            features = obs
        
        # 策略网络
        h = self.policy_net(features)
        
        # 输出动作
        mean = self.mean_head(h)
        mean = self.output_activation(mean)
        
        # BC策略通常是确定性的，但我们可以添加小噪声用于探索
        std = torch.ones_like(mean) * 0.1  # 固定小噪声
        
        return {
            'mean': mean,
            'std': std,
            'action': mean  # 确定性输出
        }
    
    def get_action(
        self,
        obs: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """获取动作 (numpy接口)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0) if obs is not None else None
            image_t = torch.FloatTensor(image).unsqueeze(0) if image is not None else None
            
            output = self.forward(obs_t, image_t)
            action = output['action'].cpu().numpy()[0]
        
        return action
    
    def compute_loss(
        self,
        obs: torch.Tensor,
        action_target: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算BC损失 (MSE)
        
        Args:
            obs: 观测
            action_target: 目标动作 (来自演示)
            image: 图像 (可选)
            
        Returns:
            loss: MSE损失
        """
        output = self.forward(obs, image)
        pred_action = output['action']
        
        loss = F.mse_loss(pred_action, action_target)
        
        return loss


class BCTrainer:
    """BC训练器"""
    
    def __init__(
        self,
        policy: BCPolicy,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda'
    ):
        self.policy = policy.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.train_step = 0
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        self.policy.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            obs = batch['obs'].to(self.device)
            action = batch['action'].to(self.device)
            image = batch.get('image', None)
            if image is not None:
                image = image.to(self.device)
            
            # 计算损失
            loss = self.policy.compute_loss(obs, action, image)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.train_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'bc_loss': avg_loss,
            'train_step': self.train_step
        }
    
    def save(self, path: str):
        """保存策略"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """加载策略"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']


class DemonstrationDataset:
    """演示数据集"""
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        images: Optional[np.ndarray] = None,
        augment: bool = True,
        noise_scale: float = 0.01
    ):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        self.images = torch.FloatTensor(images) if images is not None else None
        
        self.augment = augment
        self.noise_scale = noise_scale
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        # 数据增强: 添加噪声
        if self.augment:
            obs = obs + torch.randn_like(obs) * self.noise_scale
            action = action + torch.randn_like(action) * self.noise_scale * 0.1
        
        item = {
            'obs': obs,
            'action': action
        }
        
        if self.images is not None:
            item['image'] = self.images[idx]
        
        return item
