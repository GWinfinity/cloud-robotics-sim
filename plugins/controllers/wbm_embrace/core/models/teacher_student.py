"""
Teacher-Student Architecture for WBM

教师策略使用特权信息，学生策略通过蒸馏学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class TeacherPolicy(nn.Module):
    """
    教师策略
    
    使用完整的特权信息:
    - 物体完整几何 (NSDF)
    - 接触力信息
    - 质心、惯性等物理属性
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        nsdf_dim: int = 64,
        contact_dim: int = 32,
        hidden_dims: list = [512, 512, 256],
        activation: str = 'elu'
    ):
        super().__init__()
        
        # 特权信息包括:
        # - 基础观测 (机器人状态)
        # - NSDF特征 (物体几何)
        # - 接触信息
        
        total_input_dim = state_dim + nsdf_dim + contact_dim
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # 网络
        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 输出
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # NSDF编码器 (将NSDF查询结果编码为特征)
        self.nsdf_encoder = nn.Sequential(
            nn.Linear(16, 128),  # 假设查询16个点
            nn.ReLU(),
            nn.Linear(128, nsdf_dim)
        )
        
        # 接触编码器
        self.contact_encoder = nn.Sequential(
            nn.Linear(8 * 3, 64),  # 8个接触点 x 3维力
            nn.ReLU(),
            nn.Linear(64, contact_dim)
        )
    
    def forward(self, state: torch.Tensor, nsdf_features: torch.Tensor, 
                contact_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 基础观测
            nsdf_features: NSDF查询结果
            contact_info: 接触力信息
        """
        # 编码NSDF和接触信息
        nsdf_encoded = self.nsdf_encoder(nsdf_features)
        contact_encoded = self.contact_encoder(contact_info.flatten(start_dim=1))
        
        # 合并所有输入
        x = torch.cat([state, nsdf_encoded, contact_encoded], dim=-1)
        
        # 特征提取
        features = self.feature_net(x)
        
        # 输出动作分布
        mean = torch.tanh(self.mean_head(features))
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return {
            'mean': mean,
            'std': std,
            'action': mean + torch.randn_like(mean) * std
        }


class StudentPolicy(nn.Module):
    """
    学生策略
    
    仅使用可观测信息:
    - 机器人本体感觉
    - 物体位置/方向
    - 简化的几何信息
    
    通过蒸馏学习教师策略
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        use_nsdf_embedding: bool = True,
        nsdf_embedding_dim: int = 32,
        hidden_dims: list = [512, 512, 256],
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.use_nsdf_embedding = use_nsdf_embedding
        
        # 激活函数
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.Tanh()
        
        # NSDF嵌入 (将在线查询的NSDF结果嵌入低维)
        if use_nsdf_embedding:
            self.nsdf_embedding = nn.Sequential(
                nn.Linear(16, 64),  # 假设16个查询点
                nn.ReLU(),
                nn.Linear(64, nsdf_embedding_dim)
            )
            input_dim = obs_dim + nsdf_embedding_dim
        else:
            input_dim = obs_dim
        
        # 主网络
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
    
    def forward(self, obs: torch.Tensor, nsdf_query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        if self.use_nsdf_embedding and nsdf_query is not None:
            nsdf_embed = self.nsdf_embedding(nsdf_query)
            x = torch.cat([obs, nsdf_embed], dim=-1)
        else:
            x = obs
        
        features = self.feature_net(x)
        
        mean = torch.tanh(self.mean_head(features))
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return {
            'mean': mean,
            'std': std,
            'action': mean + torch.randn_like(mean) * std
        }
    
    def get_action(self, obs: np.ndarray, nsdf_query: Optional[np.ndarray] = None) -> np.ndarray:
        """获取动作 (numpy接口)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            nsdf_t = torch.FloatTensor(nsdf_query).unsqueeze(0) if nsdf_query is not None else None
            
            output = self.forward(obs_t, nsdf_t)
            action = output['mean'].cpu().numpy()[0]
        
        return action


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失
    
    将教师策略的知识传递给学生策略
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        蒸馏损失
        
        使用 softened softmax 匹配分布
        """
        # 软化分布
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL散度
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # 缩放
        loss = loss * (self.temperature ** 2)
        
        return loss


class WBMTrainer:
    """
    全身操作训练器
    
    同时训练教师和学生策略
    """
    
    def __init__(
        self,
        teacher: TeacherPolicy,
        student: StudentPolicy,
        distillation_weight: float = 0.5,
        temperature: float = 1.0,
        lr: float = 3e-4
    ):
        self.teacher = teacher
        self.student = student
        self.distillation_weight = distillation_weight
        
        # 优化器
        self.teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
        self.student_optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        
        # 蒸馏损失
        self.distillation_loss = DistillationLoss(temperature)
    
    def train_step(
        self,
        state: torch.Tensor,
        nsdf_features: torch.Tensor,
        contact_info: torch.Tensor,
        student_obs: torch.Tensor,
        nsdf_query: torch.Tensor,
        target_action: torch.Tensor
    ) -> Dict[str, float]:
        """
        训练一步
        
        包括:
        1. 教师策略更新 (使用特权信息)
        2. 学生策略更新 (蒸馏 + 任务奖励)
        """
        # ===== 教师策略更新 =====
        teacher_output = self.teacher(state, nsdf_features, contact_info)
        
        # 监督损失
        teacher_loss = F.mse_loss(teacher_output['mean'], target_action)
        
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_optimizer.step()
        
        # ===== 学生策略更新 =====
        student_output = self.student(student_obs, nsdf_query)
        
        # 任务损失
        student_task_loss = F.mse_loss(student_output['mean'], target_action)
        
        # 蒸馏损失 (与教师匹配)
        student_distill_loss = self.distillation_loss(
            student_output['mean'],
            teacher_output['mean'].detach()
        )
        
        # 总损失
        student_total_loss = student_task_loss + \
                            self.distillation_weight * student_distill_loss
        
        self.student_optimizer.zero_grad()
        student_total_loss.backward()
        self.student_optimizer.step()
        
        return {
            'teacher_loss': teacher_loss.item(),
            'student_task_loss': student_task_loss.item(),
            'student_distill_loss': student_distill_loss.item(),
            'student_total_loss': student_total_loss.item()
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'teacher': self.teacher.state_dict(),
            'student': self.student.state_dict(),
            'teacher_optimizer': self.teacher_optimizer.state_dict(),
            'student_optimizer': self.student_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.teacher.load_state_dict(checkpoint['teacher'])
        self.student.load_state_dict(checkpoint['student'])
        self.teacher_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
        self.student_optimizer.load_state_dict(checkpoint['student_optimizer'])
