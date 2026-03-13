"""
PPO with Prediction Augmentation

标准PPO + 预测器联合训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

from models.policy import UnifiedPolicy, ValueNetwork
from models.predictor import LearnedPredictor


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = 'cuda'
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # 缓冲区
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
        # 展平
        b_obs = self.observations.reshape(-1, self.observations.shape[-1])
        b_actions = self.actions.reshape(-1, self.actions.shape[-1])
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # 生成随机索引
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
        """重置"""
        self.step = 0


class PPOTrainer:
    """
    PPO训练器
    
    同时训练:
    - 策略网络
    - 价值网络
    - 预测器网络 (可选)
    """
    
    def __init__(
        self,
        policy: UnifiedPolicy,
        value_net: ValueNetwork,
        predictor: Optional[LearnedPredictor],
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        
        # 网络
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.predictor = predictor.to(device) if predictor else None
        
        # 优化器
        policy_params = list(policy.parameters())
        if predictor:
            policy_params += list(predictor.parameters())
        
        self.policy_optimizer = optim.Adam(
            policy_params,
            lr=config['training']['learning_rate']
        )
        
        self.value_optimizer = optim.Adam(
            value_net.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # 缓冲区
        self.buffer = RolloutBuffer(
            num_steps=config['env']['episode_length'],
            num_envs=config['env']['num_envs'],
            obs_dim=policy.obs_dim,
            action_dim=policy.action_dim,
            device=device
        )
        
        # 超参数
        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.clip_epsilon = config['training']['clip_epsilon']
        self.entropy_coef = config['training']['entropy_coef']
        self.value_coef = config['training']['value_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['mini_batch_size']
        
        self.total_steps = 0
    
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """选择动作"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device)
            
            # 策略
            policy_out = self.policy(obs_t)
            action = policy_out['action']
            log_prob = policy_out['log_prob']
            
            # 价值
            value = self.value_net(obs_t)
        
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy()
        )
    
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
    
    def update(self, next_obs: np.ndarray) -> Dict[str, float]:
        """更新网络"""
        # 计算下一状态价值
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            next_value = self.value_net(next_obs_t)
        
        # 计算回报和优势
        self.buffer.compute_returns_and_advantages(next_value, self.gamma, self.gae_lambda)
        
        # 记录损失
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        # PPO更新
        for epoch in range(self.num_epochs):
            batches = self.buffer.get_batches(self.batch_size)
            
            for batch in batches:
                # 评估动作
                new_log_probs, entropy, _ = self.policy.evaluate_actions(
                    batch['obs'], batch['actions']
                )
                
                new_values = self.value_net(batch['obs'])
                
                # 策略损失 (PPO)
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
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 反向传播
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # 记录
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['approx_kl'].append(approx_kl.item())
                metrics['clip_fraction'].append(clip_fraction.item())
        
        self.buffer.reset()
        self.total_steps += 1
        
        # 返回平均损失
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'policy': self.policy.state_dict(),
            'value': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'total_steps': self.total_steps
        }
        
        if self.predictor:
            checkpoint['predictor'] = self.predictor.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_net.load_state_dict(checkpoint['value'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.total_steps = checkpoint['total_steps']
        
        if self.predictor and 'predictor' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor'])
