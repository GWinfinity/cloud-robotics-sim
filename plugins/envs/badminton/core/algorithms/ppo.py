"""
PPO (Proximal Policy Optimization) 实现

用于训练人形机器人羽毛球策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, List, Optional


class ActorCritic(nn.Module):
    """Actor-Critic 网络"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden: List[int] = [512, 512, 256],
        critic_hidden: List[int] = [512, 512, 256],
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # Actor 网络
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in actor_hidden:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(self.activation)
            prev_dim = hidden_dim
        self.actor_hidden = nn.Sequential(*actor_layers)
        
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 网络
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in critic_hidden:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(self.activation)
            prev_dim = hidden_dim
        self.critic_hidden = nn.Sequential(*critic_layers)
        
        self.critic_head = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        actor_features = self.actor_hidden(obs)
        action_mean = self.actor_mean(actor_features)
        
        critic_features = self.critic_hidden(obs)
        value = self.critic_head(critic_features)
        
        return action_mean, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值"""
        action_mean, value = self.forward(obs)
        
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """获取价值"""
        _, value = self.forward(obs)
        return value


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        
        self.observations = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def add(self, obs, action, log_prob, reward, value, done):
        """添加经验"""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1
    
    def compute_returns_and_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        """计算回报和优势"""
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value_t = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]
    
    def get_batches(self, batch_size: int):
        """获取训练批次"""
        b_obs = self.observations.reshape(-1, self.observations.shape[-1])
        b_actions = self.actions.reshape(-1, self.actions.shape[-1])
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        indices = torch.randperm(b_obs.shape[0], device=self.device)
        
        batches = []
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            
            batches.append({
                'obs': b_obs[batch_indices],
                'actions': b_actions[batch_indices],
                'old_log_probs': b_log_probs[batch_indices],
                'advantages': b_advantages[batch_indices],
                'returns': b_returns[batch_indices],
                'old_values': b_values[batch_indices]
            })
        
        return batches
    
    def reset(self):
        """重置缓冲区"""
        self.step = 0


class PPO:
    """PPO 训练器"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_envs: int,
        num_steps: int,
        device: str = 'cuda',
        **kwargs
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 网络
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden=kwargs.get('actor_hidden', [512, 512, 256]),
            critic_hidden=kwargs.get('critic_hidden', [512, 512, 256]),
            activation=kwargs.get('activation', 'elu')
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=kwargs.get('learning_rate', 3e-4),
            eps=1e-5
        )
        
        # 缓冲区
        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        # 超参数
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.value_coef = kwargs.get('value_coef', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.batch_size = kwargs.get('batch_size', 512)
        
        self.total_steps = 0
        self.updates = 0
    
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """选择动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """存储转移"""
        self.buffer.add(
            obs=torch.FloatTensor(obs).to(self.device),
            action=torch.FloatTensor(action).to(self.device),
            log_prob=torch.FloatTensor([log_prob]).to(self.device),
            reward=torch.FloatTensor([reward]).to(self.device),
            value=torch.FloatTensor([value]).to(self.device),
            done=torch.FloatTensor([done]).to(self.device)
        )
    
    def update(self, next_obs: np.ndarray) -> Dict:
        """更新策略"""
        with torch.no_grad():
            next_value = self.policy.get_value(torch.FloatTensor(next_obs).to(self.device))
        
        self.buffer.compute_returns_and_advantages(next_value, self.gamma, self.gae_lambda)
        
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        for epoch in range(self.num_epochs):
            batches = self.buffer.get_batches(self.batch_size)
            
            for batch in batches:
                _, new_log_probs, entropy, new_values = \
                    self.policy.get_action_and_value(batch['obs'], batch['actions'])
                
                # 策略损失
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
                
                # 熵
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy_loss'].append(entropy_loss.item())
                metrics['approx_kl'].append(approx_kl.item())
                metrics['clip_fraction'].append(clip_fraction.item())
        
        self.buffer.reset()
        self.updates += 1
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'updates': self.updates
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
