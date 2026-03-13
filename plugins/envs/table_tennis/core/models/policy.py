"""
Unified Policy for Whole-Body Control

统一策略：直接映射球位置观测到全身关节命令

输入:
- 当前球位置/速度
- 预测的未来球状态
- 机器人本体感觉

输出:
- 全身29个关节的目标位置 (Unitree G1)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional


class UnifiedPolicy(nn.Module):
    """
    统一全身控制策略
    
    直接输出手臂和腿部的关节命令
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,  # Unitree G1: 29 DOF
        hidden_dims: list = [512, 512, 256],
        activation: str = 'elu',
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_noise_std: float = 1.0
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # 特征提取网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 输出层
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 输出层使用较小初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
    
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
            dict: 包含动作、均值、标准差等
        """
        # 特征提取
        features = self.feature_net(obs)
        
        # 动作分布
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # 限制标准差范围
        log_std = torch.clamp(log_std, np.log(self.min_std), np.log(self.max_std))
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # 采样
            dist = Normal(mean, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            
            # 计算对数概率
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            # tanh修正
            log_prob -= torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)
        
        return {
            'action': action,
            'mean': mean,
            'std': std,
            'log_std': log_std,
            'log_prob': log_prob
        }
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        获取动作 (numpy接口)
        
        Returns:
            action: 动作数组
            log_prob: 对数概率 (如果非确定性)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            output = self.forward(obs_t, deterministic)
            action = output['action'].cpu().numpy()[0]
            log_prob = output['log_prob'].cpu().item() if output['log_prob'] is not None else None
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作 (用于PPO更新)
        
        Returns:
            log_prob: 动作对数概率
            entropy: 分布熵
            mean: 动作均值
        """
        output = self.forward(obs, deterministic=False)
        
        # 重新计算对数概率
        mean = output['mean']
        std = output['std']
        
        # 反变换到pre-tanh空间
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * torch.log((1 + actions_clamped) / (1 - actions_clamped))
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob -= torch.log(1 - actions ** 2 + 1e-6).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, mean


class ValueNetwork(nn.Module):
    """价值网络 (Critic)"""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [512, 512, 256],
        activation: str = 'elu'
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
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(obs).squeeze(-1)


class ObservationEncoder(nn.Module):
    """
    观测编码器
    
    处理不同类型的观测:
    - 球状态
    - 预测状态
    - 机器人本体感觉
    """
    
    def __init__(
        self,
        ball_dim: int = 6,
        pred_dim: int = 60,
        proprio_dim: int = 40,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 球状态编码
        self.ball_encoder = nn.Sequential(
            nn.Linear(ball_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64)
        )
        
        # 预测状态编码
        self.pred_encoder = nn.Sequential(
            nn.Linear(pred_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128)
        )
        
        # 本体感觉编码
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 128, output_dim),
            nn.ELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        ball_state: torch.Tensor,
        prediction: torch.Tensor,
        proprioception: torch.Tensor
    ) -> torch.Tensor:
        """编码观测"""
        ball_feat = self.ball_encoder(ball_state)
        pred_feat = self.pred_encoder(prediction)
        proprio_feat = self.proprio_encoder(proprioception)
        
        combined = torch.cat([ball_feat, pred_feat, proprio_feat], dim=-1)
        output = self.fusion(combined)
        
        return output
