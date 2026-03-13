"""
Residual Network for RL Finetuning

轻量级残差网络，学习对BC策略的修正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class ResidualNetwork(nn.Module):
    """
    残差网络
    
    轻量级网络，学习每步残差修正:
    最终动作 = BC动作 + α * 残差动作
    
    特点:
    - 轻量级 (参数量少)
    - 输出范围受限 (确保安全)
    - 与BC策略输入相同
    """
    
    def __init__(
        self,
        obs_dim: int = 512,
        action_dim: int = 50,
        hidden_dims: list = [256, 128, 64],
        residual_scale: float = 0.3,
        activation: str = 'tanh',
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.residual_scale = residual_scale
        
        # 构建网络
        layers = []
        prev_dim = obs_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 输出层
        self.residual_head = nn.Linear(prev_dim, action_dim)
        
        # 输出激活 (限制残差范围)
        if activation == 'tanh':
            self.output_activation = lambda x: torch.tanh(x) * residual_scale
        elif activation == 'clamp':
            self.output_activation = lambda x: torch.clamp(x, -residual_scale, residual_scale)
        else:
            self.output_activation = lambda x: x
        
        # 初始化 (小权重确保初始残差接近0)
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重为小值，确保初始残差接近0"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self,
        obs: torch.Tensor,
        bc_action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测特征
            bc_action: BC策略输出的动作 (用于残差连接)
            
        Returns:
            dict: 包含残差和最终动作
        """
        # 特征提取
        features = self.feature_net(obs)
        
        # 计算残差
        residual = self.residual_head(features)
        residual = self.output_activation(residual)
        
        # 如果提供了BC动作，计算最终动作
        if bc_action is not None:
            final_action = bc_action + residual
            # 裁剪到有效范围
            final_action = torch.clamp(final_action, -1, 1)
        else:
            final_action = None
        
        return {
            'residual': residual,
            'final_action': final_action
        }
    
    def get_residual(self, obs: np.ndarray) -> np.ndarray:
        """获取残差 (numpy接口)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            residual = self.forward(obs_t)['residual'].cpu().numpy()[0]
        return residual


class CombinedPolicy(nn.Module):
    """
    组合策略 = BC策略 + 残差网络
    
    最终动作 = BC动作 + 残差动作
    """
    
    def __init__(
        self,
        bc_policy: nn.Module,
        residual_network: nn.Module,
        freeze_bc: bool = True
    ):
        super().__init__()
        
        self.bc_policy = bc_policy
        self.residual_network = residual_network
        
        # 冻结BC策略 (可选)
        if freeze_bc:
            for param in self.bc_policy.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        obs: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        use_residual: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测
            image: 图像 (可选)
            use_residual: 是否使用残差
            
        Returns:
            dict: 包含BC动作、残差、最终动作
        """
        # BC策略输出
        with torch.no_grad() if not self.training else torch.enable_grad():
            bc_output = self.bc_policy(obs, image)
            bc_action = bc_output['action']
        
        if use_residual:
            # 残差网络输出
            residual_output = self.residual_network(obs, bc_action)
            residual = residual_output['residual']
            final_action = residual_output['final_action']
        else:
            residual = torch.zeros_like(bc_action)
            final_action = bc_action
        
        return {
            'bc_action': bc_action,
            'residual': residual,
            'final_action': final_action
        }
    
    def get_action(
        self,
        obs: np.ndarray,
        image: Optional[np.ndarray] = None,
        use_residual: bool = True
    ) -> Dict[str, np.ndarray]:
        """获取动作 (numpy接口)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0) if obs is not None else None
            image_t = torch.FloatTensor(image).unsqueeze(0) if image is not None else None
            
            output = self.forward(obs_t, image_t, use_residual)
            
            return {
                'bc_action': output['bc_action'].cpu().numpy()[0],
                'residual': output['residual'].cpu().numpy()[0],
                'final_action': output['final_action'].cpu().numpy()[0]
            }


class ResidualSAC:
    """
    残差SAC算法
    
    只更新残差网络，BC策略保持不变
    """
    
    def __init__(
        self,
        bc_policy: nn.Module,
        residual_network: nn.Module,
        q_network1: nn.Module,
        q_network2: nn.Module,
        lr_residual: float = 3e-4,
        lr_q: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # BC策略 (冻结)
        self.bc_policy = bc_policy.to(device)
        for param in self.bc_policy.parameters():
            param.requires_grad = False
        
        # 残差网络 (可训练)
        self.residual_network = residual_network.to(device)
        
        # Q网络 (双Q)
        self.q1 = q_network1.to(device)
        self.q2 = q_network2.to(device)
        
        # 目标Q网络
        self.q1_target = type(q_network1)(
            q_network1.obs_dim if hasattr(q_network1, 'obs_dim') else 512,
            q_network1.action_dim if hasattr(q_network1, 'action_dim') else 50
        ).to(device)
        self.q2_target = type(q_network2)(
            q_network2.obs_dim if hasattr(q_network2, 'obs_dim') else 512,
            q_network2.action_dim if hasattr(q_network2, 'action_dim') else 50
        ).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 温度参数
        if target_entropy is None:
            target_entropy = -residual_network.action_dim
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 优化器 (只优化残差网络和Q网络)
        self.residual_optimizer = torch.optim.Adam(
            self.residual_network.parameters(),
            lr=lr_residual
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_q)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
    
    def select_action(self, obs: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            image_t = torch.FloatTensor(image).unsqueeze(0).to(self.device) if image is not None else None
            
            # BC策略输出
            bc_output = self.bc_policy(obs_t, image_t)
            bc_action = bc_output['action']
            
            # 残差网络输出
            residual_output = self.residual_network(obs_t, bc_action)
            final_action = residual_output['final_action']
            
            return final_action.cpu().numpy()[0]
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """更新网络"""
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        image = batch.get('image', None)
        if image is not None:
            image = image.to(self.device)
        next_image = batch.get('next_image', None)
        if next_image is not None:
            next_image = next_image.to(self.device)
        
        # ===== 更新Q网络 =====
        with torch.no_grad():
            # 下一个BC动作
            next_bc_output = self.bc_policy(next_obs, next_image)
            next_bc_action = next_bc_output['action']
            
            # 下一个残差动作
            next_residual_output = self.residual_network(next_obs, next_bc_action)
            next_action = next_residual_output['final_action']
            
            # 目标Q值
            next_q1 = self.q1_target(next_obs, next_action)
            next_q2 = self.q2_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2)
            
            alpha = self.log_alpha.exp()
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # 当前Q值
        current_q1 = self.q1(obs, action)
        current_q2 = self.q2(obs, action)
        
        # Q损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ===== 更新残差网络 =====
        bc_output = self.bc_policy(obs, image)
        bc_action = bc_output['action']
        
        residual_output = self.residual_network(obs, bc_action)
        new_action = residual_output['final_action']
        
        new_q1 = self.q1(obs, new_action)
        new_q2 = self.q2(obs, new_action)
        new_q = torch.min(new_q1, new_q2)
        
        alpha = self.log_alpha.exp()
        residual_loss = (alpha * 0 - new_q).mean()  # 简化版，实际应计算log_prob
        
        self.residual_optimizer.zero_grad()
        residual_loss.backward()
        self.residual_optimizer.step()
        
        # ===== 更新温度 =====
        # 简化的温度更新
        
        # 软更新目标网络
        self._soft_update_target()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'residual_loss': residual_loss.item(),
            'alpha': alpha.item()
        }
    
    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'residual_network': self.residual_network.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'log_alpha': self.log_alpha.item()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.residual_network.load_state_dict(checkpoint['residual_network'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=self.device)
