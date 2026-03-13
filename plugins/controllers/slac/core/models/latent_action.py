"""
Latent Action Space for SLAC

将高维原始动作压缩到低维潜在空间
促进时间抽象和安全探索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class LatentActionSpace(nn.Module):
    """
    潜在动作空间 VAE
    
    编码器: 原始动作序列 -> 潜在动作
    解码器: 潜在动作 -> 原始动作序列 (时间抽象)
    """
    
    def __init__(
        self,
        primitive_dim: int = 20,      # 原始动作维度
        latent_dim: int = 8,           # 潜在动作维度
        temporal_abstraction: int = 10, # 每个潜在动作执行10步
        encoder_hidden: list = [256, 256, 128],
        beta: float = 0.001,
        smoothness_weight: float = 0.1,
        magnitude_weight: float = 0.01
    ):
        super().__init__()
        
        self.primitive_dim = primitive_dim
        self.latent_dim = latent_dim
        self.temporal_abstraction = temporal_abstraction
        self.beta = beta
        self.smoothness_weight = smoothness_weight
        self.magnitude_weight = magnitude_weight
        
        # 编码器: 原始动作序列 -> 潜在动作
        # 输入: 多个时间步的原始动作
        encoder_layers = []
        input_dim = primitive_dim * temporal_abstraction
        prev_dim = input_dim
        
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def encode(self, primitive_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码原始动作序列到潜在动作
        
        Args:
            primitive_sequence: [batch, temporal_abstraction, primitive_dim]
        
        Returns:
            mu, logvar: 潜在动作的分布参数
        """
        batch_size = primitive_sequence.shape[0]
        
        # 展平时间序列
        x = primitive_sequence.view(batch_size, -1)
        
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, primitive_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            dict: 包含潜在动作、分布参数等
        """
        mu, logvar = self.encode(primitive_sequence)
        z = self.reparameterize(mu, logvar)
        
        return {
            'latent_action': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def loss_function(
        self,
        primitive_sequence: torch.Tensor,
        reconstructed: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        VAE损失函数
        
        包括:
        1. 重构损失
        2. KL散度
        3. 平滑性约束
        4. 幅度约束
        """
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(reconstructed, primitive_sequence, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 平滑性约束: 相邻动作应该平滑
        if primitive_sequence.shape[1] > 1:
            action_diff = primitive_sequence[:, 1:, :] - primitive_sequence[:, :-1, :]
            smoothness_loss = torch.mean(action_diff ** 2)
        else:
            smoothness_loss = torch.tensor(0.0)
        
        # 幅度约束: 动作幅度不要太大
        magnitude_loss = torch.mean(primitive_sequence ** 2)
        
        # 总损失
        total_loss = (
            recon_loss +
            self.beta * kl_loss +
            self.smoothness_weight * smoothness_loss +
            self.magnitude_weight * magnitude_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'smoothness_loss': smoothness_loss,
            'magnitude_loss': magnitude_loss
        }


class PrimitiveActionDecoder(nn.Module):
    """
    原始动作解码器
    
    将潜在动作解码为原始动作序列 (实现时间抽象)
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        primitive_dim: int = 20,
        temporal_abstraction: int = 10,
        hidden_dims: list = [128, 256, 256],
        output_activation: str = 'tanh'
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.primitive_dim = primitive_dim
        self.temporal_abstraction = temporal_abstraction
        
        # 解码器网络
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # 输出层: 生成整个动作序列
        self.fc_out = nn.Linear(prev_dim, primitive_dim * temporal_abstraction)
        
        # 输出激活
        if output_activation == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        else:
            self.output_activation = lambda x: x
    
    def forward(self, latent_action: torch.Tensor) -> torch.Tensor:
        """
        解码潜在动作为原始动作序列
        
        Args:
            latent_action: [batch, latent_dim]
            
        Returns:
            primitive_sequence: [batch, temporal_abstraction, primitive_dim]
        """
        h = self.decoder(latent_action)
        out = self.fc_out(h)
        
        # 应用输出激活
        out = self.output_activation(out)
        
        # reshape为序列
        batch_size = latent_action.shape[0]
        primitive_sequence = out.view(batch_size, self.temporal_abstraction, self.primitive_dim)
        
        return primitive_sequence
    
    def decode_single_step(self, latent_action: torch.Tensor, step: int) -> torch.Tensor:
        """
        解码特定时间步的原始动作 (用于逐步执行)
        
        Args:
            latent_action: [batch, latent_dim]
            step: 时间步索引 (0 to temporal_abstraction-1)
            
        Returns:
            primitive_action: [batch, primitive_dim]
        """
        sequence = self.forward(latent_action)
        return sequence[:, step, :]


class LatentActionController:
    """
    潜在动作控制器
    
    管理潜在动作的执行和转换
    """
    
    def __init__(
        self,
        latent_action_space: LatentActionSpace,
        primitive_decoder: PrimitiveActionDecoder,
        temporal_abstraction: int = 10
    ):
        self.latent_space = latent_action_space
        self.decoder = primitive_decoder
        self.temporal_abstraction = temporal_abstraction
        
        self.current_primitive_sequence = None
        self.current_step = 0
    
    def set_latent_action(self, latent_action: np.ndarray):
        """设置新的潜在动作"""
        with torch.no_grad():
            latent_t = torch.FloatTensor(latent_action).unsqueeze(0)
            primitive_seq = self.decoder(latent_t).cpu().numpy()[0]
        
        self.current_primitive_sequence = primitive_seq
        self.current_step = 0
    
    def get_next_primitive_action(self) -> Optional[np.ndarray]:
        """获取下一步的原始动作"""
        if self.current_primitive_sequence is None:
            return None
        
        if self.current_step >= self.temporal_abstraction:
            return None  # 序列执行完毕
        
        action = self.current_primitive_sequence[self.current_step]
        self.current_step += 1
        
        return action
    
    def is_sequence_done(self) -> bool:
        """检查当前序列是否执行完毕"""
        return self.current_step >= self.temporal_abstraction


class SafetyConstrainedLatentSpace(nn.Module):
    """
    安全约束的潜在空间
    
    在潜在动作上添加安全约束，避免危险动作
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        safety_bounds: Optional[Dict] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 潜在动作的边界
        if safety_bounds is None:
            self.register_buffer('latent_min', torch.ones(latent_dim) * -3)
            self.register_buffer('latent_max', torch.ones(latent_dim) * 3)
        else:
            self.register_buffer('latent_min', torch.FloatTensor(safety_bounds['min']))
            self.register_buffer('latent_max', torch.FloatTensor(safety_bounds['max']))
    
    def clip_latent_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        """裁剪潜在动作到安全范围"""
        return torch.clamp(latent_action, self.latent_min, self.latent_max)
    
    def check_safety(self, latent_action: torch.Tensor) -> torch.Tensor:
        """检查潜在动作是否安全"""
        in_bounds = (latent_action >= self.latent_min) & (latent_action <= self.latent_max)
        return in_bounds.all(dim=-1)
