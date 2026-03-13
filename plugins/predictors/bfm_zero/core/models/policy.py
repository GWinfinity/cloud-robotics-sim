"""
Latent-Conditioned Policy

BFM-Zero的核心: 策略以潜在向量z为条件，实现可提示的控制

z可以是:
- 运动跟踪: 参考姿势编码
- 目标到达: 目标状态编码
- 奖励优化: 奖励函数编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Optional


class LatentConditionedPolicy(nn.Module):
    """
    潜在条件策略 π(a | s, z)
    
    通过潜在向量z来提示策略执行不同任务
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        hidden_dims: list = [512, 512, 256],
        activation: str = 'elu',
        min_std: float = 0.1,
        max_std: float = 1.0,
        history_dim: int = 0  # 历史观测维度 (用于非对称学习)
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.min_std = min_std
        self.max_std = max_std
        self.history_dim = history_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # 输入维度: 状态 + 潜在向量 + 历史
        input_dim = state_dim + latent_dim + history_dim
        
        # 策略网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        self.policy_network = nn.Sequential(*layers)
        
        # 输出层
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 最后一层使用较小的初始化
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            latent: [batch, latent_dim] - 任务潜在向量
            history: [batch, history_dim] - 历史观测 (可选)
            deterministic: 是否确定性输出
            
        Returns:
            dict: 包含动作分布参数
        """
        # 拼接输入
        inputs = [state, latent]
        if history is not None and self.history_dim > 0:
            inputs.append(history)
        x = torch.cat(inputs, dim=-1)
        
        # 特征提取
        features = self.policy_network(x)
        
        # 输出动作分布
        mean = self.mean_layer(features)
        
        # 标准差 (可学习)
        log_std = self.log_std_layer(features)
        std = torch.exp(log_std)
        std = torch.clamp(std, self.min_std, self.max_std)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            # 采样
            dist = Normal(mean, std)
            sample = dist.rsample()
            action = torch.tanh(sample)
        
        return {
            'action': action,
            'mean': mean,
            'std': std,
            'log_std': log_std
        }
    
    def get_action(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        history: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """获取动作 (numpy接口)"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            latent_t = torch.FloatTensor(latent).unsqueeze(0)
            
            history_t = None
            if history is not None and self.history_dim > 0:
                history_t = torch.FloatTensor(history).unsqueeze(0)
            
            output = self.forward(state_t, latent_t, history_t, deterministic)
            action = output['action'].cpu().numpy()[0]
        
        return action
    
    def compute_log_prob(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算动作的对数概率
        
        用于策略优化
        """
        # 前向传播
        inputs = [state, latent]
        if history is not None and self.history_dim > 0:
            inputs.append(history)
        x = torch.cat(inputs, dim=-1)
        
        features = self.policy_network(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        std = torch.exp(log_std)
        std = torch.clamp(std, self.min_std, self.max_std)
        
        # 计算对数概率
        # 注意: tanh变换后的对数概率需要修正
        dist = Normal(mean, std)
        
        # 反变换动作到pre-tanh空间
        # 这是一个近似，实际应该存储pre-tanh动作
        pre_tanh_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        
        # tanh修正项
        log_prob -= torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)
        
        return log_prob


class PromptEncoder(nn.Module):
    """
    提示编码器
    
    将不同类型的任务提示编码为统一的潜在表示
    
    支持的任务类型:
    - 运动跟踪: 参考姿势序列
    - 目标到达: 目标状态
    - 奖励函数: 奖励权重向量
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 256,
        max_seq_len: int = 10
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # 运动跟踪编码器 (处理姿势序列)
        self.motion_encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        self.motion_head = nn.Linear(256, latent_dim)
        
        # 目标到达编码器
        self.goal_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )
        
        # 奖励函数编码器
        self.reward_encoder = nn.Sequential(
            nn.Linear(state_dim + 10, 256),  # 状态 + 奖励特征
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )
    
    def encode_motion(
        self,
        motion_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        编码运动序列
        
        Args:
            motion_sequence: [batch, seq_len, state_dim]
            
        Returns:
            latent: [batch, latent_dim]
        """
        # LSTM编码
        lstm_out, (hidden, cell) = self.motion_encoder(motion_sequence)
        
        # 使用最后一个隐藏状态
        last_hidden = hidden[-1]  # [batch, 256]
        
        # 映射到潜在空间
        latent = self.motion_head(last_hidden)
        latent = F.normalize(latent, p=2, dim=-1)
        
        return latent
    
    def encode_goal(
        self,
        goal_state: torch.Tensor
    ) -> torch.Tensor:
        """编码目标状态"""
        latent = self.goal_encoder(goal_state)
        latent = F.normalize(latent, p=2, dim=-1)
        return latent
    
    def encode_reward(
        self,
        state: torch.Tensor,
        reward_features: torch.Tensor
    ) -> torch.Tensor:
        """编码奖励函数"""
        x = torch.cat([state, reward_features], dim=-1)
        latent = self.reward_encoder(x)
        latent = F.normalize(latent, p=2, dim=-1)
        return latent


class MultiTaskPolicy(nn.Module):
    """
    多任务策略
    
    统一管理不同任务的策略执行
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        num_tasks: int = 3,
        **policy_kwargs
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # 共享的策略网络
        self.policy = LatentConditionedPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            **policy_kwargs
        )
        
        # 任务特定的潜在嵌入
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, latent_dim)
        )
        
        # 初始化
        nn.init.orthogonal_(self.task_embeddings)
    
    def forward(
        self,
        state: torch.Tensor,
        task_id: int,
        custom_latent: Optional[torch.Tensor] = None
    ):
        """
        执行特定任务
        
        Args:
            state: 当前状态
            task_id: 任务ID (0: motion_tracking, 1: goal_reaching, 2: reward_opt)
            custom_latent: 自定义潜在向量 (覆盖默认的任务嵌入)
        """
        if custom_latent is not None:
            latent = custom_latent
        else:
            latent = self.task_embeddings[task_id].unsqueeze(0).expand(state.shape[0], -1)
        
        return self.policy(state, latent)
    
    def get_task_latent(self, task_id: int) -> torch.Tensor:
        """获取任务潜在向量"""
        return self.task_embeddings[task_id]
