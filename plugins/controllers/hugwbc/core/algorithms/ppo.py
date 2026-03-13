"""
PPO Algorithm for HugWBC

基于 rsl_rl 的 PPO 实现，适配 Genesis 环境
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
import os


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络 (支持非对称 Actor-Critic)
    
    Actor: 根据观察输出动作分布
    Critic: 根据特权观察输出状态价值
    """
    
    def __init__(
        self,
        obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.privileged_obs_dim = privileged_obs_dim
        self.action_dim = action_dim
        
        # 激活函数
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Actor 网络
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(self.activation)
            prev_dim = hidden_dim
        self.actor_backbone = nn.Sequential(*actor_layers)
        
        # Actor 输出层 (均值和对数标准差)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_noise_std))
        
        # Critic 网络 (使用特权观察)
        critic_layers = []
        prev_dim = privileged_obs_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(self.activation)
            prev_dim = hidden_dim
        self.critic_backbone = nn.Sequential(*critic_layers)
        
        # Critic 输出层
        self.critic_head = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 输出层使用较小的初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)
    
    def forward(self):
        """前向传播 (不直接使用)"""
        raise NotImplementedError
    
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据观察选择动作
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        action_mean = self.actor_mean(self.actor_backbone(obs))
        std = torch.exp(self.actor_log_std)
        
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, obs: torch.Tensor, privileged_obs: torch.Tensor, 
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作 (用于训练)
        
        Returns:
            log_probs: 动作对数概率
            entropy: 策略熵
            value: 状态价值
        """
        # Actor
        action_mean = self.actor_mean(self.actor_backbone(obs))
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic
        value = self.critic_head(self.critic_backbone(privileged_obs)).squeeze(-1)
        
        return log_probs, entropy, value
    
    def get_value(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        """获取状态价值估计"""
        return self.critic_head(self.critic_backbone(privileged_obs)).squeeze(-1)


class RolloutBuffer:
    """
    PPO 经验回放缓冲区
    支持并行环境
    """
    
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        device: torch.device
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        
        # 缓冲区
        self.observations = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.privileged_obs = torch.zeros((num_steps, num_envs, privileged_obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def add(self, obs, privileged_obs, action, log_prob, reward, value, done):
        """添加一步经验"""
        self.observations[self.step] = torch.as_tensor(obs, device=self.device)
        self.privileged_obs[self.step] = torch.as_tensor(privileged_obs, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, device=self.device)
        self.log_probs[self.step] = torch.as_tensor(log_prob, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, device=self.device)
        self.values[self.step] = torch.as_tensor(value, device=self.device)
        self.dones[self.step] = torch.as_tensor(done, device=self.device)
        
        self.step += 1
    
    def compute_returns_and_advantages(self, next_value: torch.Tensor, 
                                        gamma: float = 0.99, gae_lambda: float = 0.95):
        """使用 GAE 计算回报和优势"""
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value_t = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD 误差
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            
            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            
            # 回报
            self.returns[t] = self.advantages[t] + self.values[t]
    
    def get_batches(self, batch_size: int):
        """生成训练批次"""
        # 展平缓冲区
        b_obs = self.observations.reshape(-1, self.observations.shape[-1])
        b_privileged_obs = self.privileged_obs.reshape(-1, self.privileged_obs.shape[-1])
        b_actions = self.actions.reshape(-1, self.actions.shape[-1])
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # 生成随机索引
        indices = torch.randperm(b_obs.shape[0], device=self.device)
        
        # 分批
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            
            yield {
                'obs': b_obs[batch_indices],
                'privileged_obs': b_privileged_obs[batch_indices],
                'actions': b_actions[batch_indices],
                'old_log_probs': b_log_probs[batch_indices],
                'advantages': b_advantages[batch_indices],
                'returns': b_returns[batch_indices],
                'old_values': b_values[batch_indices]
            }
    
    def reset(self):
        """重置缓冲区"""
        self.step = 0


class PPO:
    """
    PPO 训练器
    """
    
    def __init__(
        self,
        obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        num_envs: int = 4096,
        num_steps: int = 24,
        device: str = 'cuda',
        # 网络参数
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        # 训练参数
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.0,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        num_epochs: int = 5,
        batch_size: int = 16384,
        desired_kl: float = 0.01,
        **kwargs
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建网络
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            privileged_obs_dim=privileged_obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.learning_rate = learning_rate
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 缓冲区
        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_dim=obs_dim,
            privileged_obs_dim=privileged_obs_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.desired_kl = desired_kl
        
        # 统计
        self.total_steps = 0
        self.updates = 0
    
    def act(self, obs: np.ndarray, privileged_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据观察选择动作
        
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 价值估计
        """
        with torch.no_grad():
            # 确保输入是 2D 张量 (batch_size, obs_dim)
            if len(obs.shape) == 1:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            else:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            if len(privileged_obs.shape) == 1:
                privileged_obs_tensor = torch.FloatTensor(privileged_obs).unsqueeze(0).to(self.device)
            else:
                privileged_obs_tensor = torch.FloatTensor(privileged_obs).to(self.device)
            
            action, log_prob = self.policy.act(obs_tensor)
            value = self.policy.get_value(privileged_obs_tensor)
        
        # 确保返回值是正确的形状
        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        value = value.cpu().numpy()
        
        # 如果是单样本，保持 0 维
        return action, log_prob, value
    
    def store_transition(self, obs, privileged_obs, action, log_prob, reward, value, done):
        """存储转移"""
        self.buffer.add(obs, privileged_obs, action, log_prob, reward, value, done)
        self.total_steps += 1
    
    def update(self, next_obs: np.ndarray, next_privileged_obs: np.ndarray) -> Dict:
        """
        更新策略
        
        Returns:
            metrics: 训练指标字典
        """
        # 计算下一个状态的价值
        with torch.no_grad():
            next_value = self.policy.get_value(
                torch.FloatTensor(next_privileged_obs).to(self.device)
            )
        
        # 计算回报和优势
        self.buffer.compute_returns_and_advantages(next_value, self.gamma, self.gae_lambda)
        
        # PPO 更新
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'approx_kl': [],
            'clip_fraction': [],
        }
        
        mean_kl = 0.0
        
        for epoch in range(self.num_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # 前向传播
                new_log_probs, entropy, new_values = self.policy.evaluate(
                    batch['obs'], batch['privileged_obs'], batch['actions']
                )
                
                # 策略损失 (PPO-Clip)
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])
                
                advantages = batch['advantages']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_pred_clipped = batch['old_values'] + torch.clamp(
                    new_values - batch['old_values'],
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (new_values - batch['returns']) ** 2
                value_loss2 = (value_pred_clipped - batch['returns']) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录指标
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                    mean_kl = approx_kl.item()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy_loss'].append(entropy_loss.item())
                metrics['approx_kl'].append(approx_kl.item())
                metrics['clip_fraction'].append(clip_fraction.item())
            
            # KL 散度检查 (提前停止)
            if mean_kl > 1.5 * self.desired_kl:
                break
        
        # 重置缓冲区
        self.buffer.reset()
        
        self.updates += 1
        
        # 返回平均指标
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'updates': self.updates,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)
        print(f"Model loaded from {path}")
