"""
Skill Discovery for SLAC

基于DIAYN的无监督技能发现
促进多样性和状态覆盖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class DIAYNDiscriminator(nn.Module):
    """
    DIAYN判别器
    
    预测当前技能标签给定状态
    用于计算多样性奖励
    """
    
    def __init__(
        self,
        state_dim: int,
        num_skills: int = 16,
        hidden_dims: list = [256, 128],
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_skills = num_skills
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_skills))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        预测技能 logits
        
        Args:
            state: [batch, state_dim]
            
        Returns:
            logits: [batch, num_skills]
        """
        return self.network(state)
    
    def predict_skill(self, state: torch.Tensor) -> torch.Tensor:
        """预测最可能的技能"""
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)
    
    def get_log_prob(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """
        获取给定状态下特定技能的对数概率
        
        Args:
            state: [batch, state_dim]
            skill: [batch] 技能索引
            
        Returns:
            log_prob: [batch]
        """
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 选择对应技能的概率
        batch_size = state.shape[0]
        skill_log_probs = log_probs[torch.arange(batch_size), skill]
        
        return skill_log_probs


class SkillDiscovery(nn.Module):
    """
    技能发现模块
    
    实现DIAYN风格的无监督技能发现
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_skills: int = 16,
        diversity_weight: float = 1.0,
        coverage_weight: float = 0.5,
        entropy_weight: float = 0.01,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_skills = num_skills
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.entropy_weight = entropy_weight
        
        # 判别器
        self.discriminator = DIAYNDiscriminator(
            state_dim=state_dim,
            num_skills=num_skills,
            hidden_dims=hidden_dims
        )
        
        # 技能嵌入 (可学习或固定)
        self.skill_embedding = nn.Embedding(num_skills, action_dim)
        
    def sample_skill(self, batch_size: int = 1) -> torch.Tensor:
        """随机采样技能"""
        return torch.randint(0, self.num_skills, (batch_size,))
    
    def get_skill_embedding(self, skill: torch.Tensor) -> torch.Tensor:
        """获取技能嵌入向量"""
        return self.skill_embedding(skill)
    
    def compute_diversity_reward(
        self,
        state: torch.Tensor,
        skill: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多样性奖励
        
        DIAYN核心: log(q(z|s)) - log(p(z))
        鼓励状态区分不同技能
        """
        # 判别器对数概率
        log_q_z_given_s = self.discriminator.get_log_prob(state, skill)
        
        # 先验概率 (均匀分布)
        log_p_z = -np.log(self.num_skills)
        
        # 多样性奖励
        diversity_reward = log_q_z_given_s - log_p_z
        
        return diversity_reward
    
    def compute_coverage_reward(
        self,
        state: torch.Tensor,
        visited_states: list
    ) -> torch.Tensor:
        """
        计算状态覆盖奖励
        
        鼓励探索未访问的状态区域
        """
        if len(visited_states) == 0:
            return torch.ones(state.shape[0])
        
        # 计算与已访问状态的最小距离
        visited = torch.stack(visited_states)
        distances = torch.cdist(state, visited)
        min_distances = distances.min(dim=-1)[0]
        
        # 距离越远，奖励越高
        coverage_reward = min_distances
        
        return coverage_reward
    
    def compute_entropy_bonus(self) -> torch.Tensor:
        """
        计算策略熵奖励
        
        鼓励策略的探索性
        """
        # 均匀分布的熵
        uniform_entropy = np.log(self.num_skills)
        
        # 我们希望策略熵接近均匀分布
        return torch.tensor(uniform_entropy)
    
    def get_intrinsic_reward(
        self,
        state: torch.Tensor,
        skill: torch.Tensor,
        visited_states: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        获取内在奖励
        
        组合多种内在奖励信号
        """
        # 多样性奖励
        diversity = self.compute_diversity_reward(state, skill)
        
        # 覆盖奖励
        if visited_states is not None:
            coverage = self.compute_coverage_reward(state, visited_states)
        else:
            coverage = torch.zeros_like(diversity)
        
        # 熵奖励
        entropy = self.compute_entropy_bonus()
        
        # 组合
        total_intrinsic = (
            self.diversity_weight * diversity +
            self.coverage_weight * coverage +
            self.entropy_weight * entropy
        )
        
        return {
            'total_intrinsic': total_intrinsic,
            'diversity_reward': diversity,
            'coverage_reward': coverage,
            'entropy_reward': entropy
        }
    
    def update_discriminator(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        更新判别器
        
        最大化判别器预测技能的准确率
        """
        logits = self.discriminator(states)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, skills)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class SkillConditionedPolicy(nn.Module):
    """
    技能条件策略
    
    策略以技能标签为条件
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_skills: int = 16,
        hidden_dims: list = [256, 256],
        activation: str = 'relu'
    ):
        super().__init__()
        
        # 技能嵌入
        self.skill_embedding = nn.Embedding(num_skills, 16)
        
        # 网络输入: 状态 + 技能嵌入
        input_dim = state_dim + 16
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 输出
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        skill: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            skill: [batch] 技能索引
            
        Returns:
            dict: 动作分布参数
        """
        # 技能嵌入
        skill_embed = self.skill_embedding(skill)
        
        # 拼接
        x = torch.cat([state, skill_embed], dim=-1)
        
        # 特征提取
        features = self.feature_net(x)
        
        # 输出
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return {
            'mean': mean,
            'std': std,
            'action': mean + torch.randn_like(mean) * std
        }


class SkillLibrary:
    """
    技能库
    
    存储和检索学到的技能
    """
    
    def __init__(self, num_skills: int = 16):
        self.num_skills = num_skills
        self.skills = {i: [] for i in range(num_skills)}  # 技能ID -> 状态序列
        self.skill_descriptions = {}
    
    def add_trajectory(self, skill_id: int, trajectory: list):
        """添加轨迹到技能库"""
        self.skills[skill_id].append(trajectory)
    
    def get_skill_trajectories(self, skill_id: int) -> list:
        """获取特定技能的所有轨迹"""
        return self.skills[skill_id]
    
    def analyze_skills(self) -> Dict:
        """分析学到的技能"""
        analysis = {}
        
        for skill_id in range(self.num_skills):
            trajectories = self.skills[skill_id]
            
            if len(trajectories) > 0:
                # 计算平均回报
                avg_return = np.mean([sum(step['reward'] for step in traj) for traj in trajectories])
                
                # 计算状态覆盖率
                all_states = np.concatenate([np.array([step['state'] for step in traj]) for traj in trajectories])
                state_variance = np.var(all_states, axis=0).mean()
                
                analysis[skill_id] = {
                    'num_trajectories': len(trajectories),
                    'avg_return': avg_return,
                    'state_variance': state_variance
                }
        
        return analysis
