"""
Dual Predictor Architecture

论文核心创新:
1. 学习预测器 (Learned Predictor): 轻量级网络，用于策略推理
2. 物理预测器 (Physics Predictor): 基于物理模型，用于训练奖励
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List


class LearnedPredictor(nn.Module):
    """
    轻量级学习预测器
    
    输入: 最近球位置历史
    输出: 未来球状态估计
    
    用于:
    - 增强策略观测
    - 实现主动决策
    """
    
    def __init__(
        self,
        input_history: int = 5,      # 历史帧数
        state_dim: int = 6,          # 每帧状态维度 (位置3 + 速度3)
        hidden_dims: list = [128, 128, 64],
        output_horizon: int = 10,    # 预测未来多少步
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.input_history = input_history
        self.state_dim = state_dim
        self.output_horizon = output_horizon
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # 输入: 历史状态序列
        input_dim = input_history * state_dim
        
        # 编码器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出头: 预测未来位置和速度
        output_dim = output_horizon * state_dim
        self.predictor_head = nn.Linear(prev_dim, output_dim)
        
    def forward(self, history_states: torch.Tensor) -> torch.Tensor:
        """
        预测未来球状态
        
        Args:
            history_states: [batch, input_history * state_dim]
            
        Returns:
            future_states: [batch, output_horizon * state_dim]
        """
        # 编码
        features = self.encoder(history_states)
        
        # 预测
        prediction = self.predictor_head(features)
        
        return prediction
    
    def predict_future(
        self,
        history_states: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        numpy接口预测
        
        Returns:
            未来状态列表，每个包含位置和速度
        """
        with torch.no_grad():
            states_t = torch.FloatTensor(history_states).unsqueeze(0)
            prediction = self.forward(states_t).cpu().numpy()[0]
        
        # 解析预测结果
        future_states = []
        for i in range(self.output_horizon):
            start_idx = i * self.state_dim
            pos = prediction[start_idx:start_idx+3]
            vel = prediction[start_idx+3:start_idx+6]
            future_states.append({
                'position': pos,
                'velocity': vel
            })
        
        return future_states
    
    def get_immediate_prediction(self, history_states: np.ndarray) -> Dict[str, np.ndarray]:
        """获取最近的预测"""
        predictions = self.predict_future(history_states)
        return predictions[0] if predictions else None


class PhysicsPredictor:
    """
    物理预测器
    
    基于物理模型进行精确轨迹预测
    
    用于:
    - 构建密集预测奖励
    - 训练时提供准确的未来信息
    """
    
    def __init__(
        self,
        ball_config: dict,
        dt: float = 0.008,
        integration_steps: int = 50
    ):
        self.ball_config = ball_config
        self.dt = dt
        self.integration_steps = integration_steps
        
        # 预计算常量
        self.air_density = ball_config.get('air_density', 1.225)
        self.radius = ball_config.get('radius', 0.02)
        self.mass = ball_config.get('mass', 0.0027)
        self.drag_coeff = ball_config.get('drag_coefficient', 0.5)
        self.restitution = ball_config.get('restitution', 0.85)
        
        self.cross_area = np.pi * self.radius ** 2
    
    def predict_trajectory(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        initial_omega: Optional[np.ndarray] = None,
        time_horizon: Optional[float] = None
    ) -> List[Dict]:
        """
        预测完整轨迹
        
        使用欧拉积分模拟物理
        """
        if time_horizon is None:
            time_horizon = self.dt * self.integration_steps
        
        num_steps = int(time_horizon / self.dt)
        
        trajectory = []
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        omega = initial_omega.copy() if initial_omega is not None else np.zeros(3)
        
        for step in range(num_steps):
            # 记录
            trajectory.append({
                'position': pos.copy(),
                'velocity': vel.copy(),
                'time': step * self.dt
            })
            
            # 计算力
            speed = np.linalg.norm(vel)
            
            # 空气阻力
            if speed > 0.01:
                drag_force = -0.5 * self.air_density * self.cross_area * \
                            self.drag_coeff * speed * vel
                drag_accel = drag_force / self.mass
            else:
                drag_accel = np.zeros(3)
            
            # 重力
            gravity = np.array([0, 0, -9.81])
            
            # 总加速度
            accel = drag_accel + gravity
            
            # 积分
            vel = vel + accel * self.dt
            pos = pos + vel * self.dt
            
            # 检查弹跳
            if pos[2] <= self.ball_config.get('table_height', 0.76) and vel[2] < 0:
                pos[2] = self.ball_config.get('table_height', 0.76)
                vel[2] = -vel[2] * self.restitution
                # 添加摩擦
                vel[0] *= 0.9
                vel[1] *= 0.9
            
            # 检查地面
            if pos[2] <= 0:
                pos[2] = 0
                break
        
        return trajectory
    
    def predict_contact_point(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        racket_height: float = 0.8,
        max_time: float = 1.0
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        预测击球点
        
        Returns:
            (击球位置, 击球时间) 或 None
        """
        trajectory = self.predict_trajectory(initial_pos, initial_vel, time_horizon=max_time)
        
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i]['position'], trajectory[i+1]['position']
            
            # 检查是否穿过击球高度
            if (p1[2] > racket_height and p2[2] <= racket_height) or \
               (p1[2] < racket_height and p2[2] >= racket_height):
                
                # 线性插值
                alpha = (racket_height - p1[2]) / (p2[2] - p1[2] + 1e-8)
                contact_pos = p1 + alpha * (p2 - p1)
                contact_time = trajectory[i]['time'] + alpha * self.dt
                
                return contact_pos, contact_time
        
        return None
    
    def predict_landing_point(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        table_height: float = 0.76
    ) -> Optional[Tuple[np.ndarray, float]]:
        """预测落点"""
        trajectory = self.predict_trajectory(initial_pos, initial_vel)
        
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i]['position'], trajectory[i+1]['position']
            
            if p1[2] > table_height and p2[2] <= table_height:
                alpha = (table_height - p1[2]) / (p2[2] - p1[2] + 1e-8)
                landing_pos = p1 + alpha * (p2 - p1)
                landing_time = trajectory[i]['time'] + alpha * self.dt
                
                return landing_pos, landing_time
        
        return None
    
    def compute_hit_probability(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        robot_position: np.ndarray,
        max_reach: float = 1.5
    ) -> float:
        """
        计算击球概率
        
        基于物理预测，判断机器人是否能到达击球点
        """
        # 预测击球点
        contact = self.predict_contact_point(initial_pos, initial_vel)
        
        if contact is None:
            return 0.0
        
        contact_pos, contact_time = contact
        
        # 检查机器人能否到达
        distance = np.linalg.norm(contact_pos[:2] - robot_position[:2])
        
        if distance > max_reach:
            return 0.0
        
        # 检查高度是否合理
        if not (0.5 <= contact_pos[2] <= 1.5):
            return 0.0
        
        # 计算概率 (基于距离和时间)
        prob = np.exp(-distance / max_reach) * np.exp(-contact_time)
        
        return min(prob, 1.0)


class DualPredictor(nn.Module):
    """
    双预测器融合
    
    结合学习预测器和物理预测器的优势
    """
    
    def __init__(
        self,
        learned_predictor: LearnedPredictor,
        physics_predictor: PhysicsPredictor,
        fusion_method: str = 'weighted'
    ):
        super().__init__()
        
        self.learned = learned_predictor
        self.physics = physics_predictor
        self.fusion_method = fusion_method
        
        # 可学习的融合权重
        if fusion_method == 'adaptive':
            self.fusion_weight = nn.Parameter(torch.tensor(0.7))
    
    def forward(
        self,
        history_states: torch.Tensor,
        use_physics: bool = False
    ) -> torch.Tensor:
        """
        预测
        
        Args:
            history_states: 历史状态
            use_physics: 是否使用物理预测器 (训练时使用)
        """
        # 学习预测器 (始终使用)
        learned_pred = self.learned(history_states)
        
        if not use_physics or self.physics is None:
            return learned_pred
        
        # 物理预测器 (仅训练时使用)
        # 将numpy结果转换为tensor
        batch_size = history_states.shape[0]
        physics_preds = []
        
        for i in range(batch_size):
            hist_np = history_states[i].cpu().numpy()
            # 解析最后的位置和速度
            pos = hist_np[-6:-3]
            vel = hist_np[-3:]
            
            traj = self.physics.predict_trajectory(pos, vel, time_horizon=0.08)
            pred = []
            for point in traj[:10]:  # 取前10步
                pred.extend(point['position'])
                pred.extend(point['velocity'])
            
            # 填充
            while len(pred) < 10 * 6:
                pred.extend([0] * 6)
            
            physics_preds.append(pred[:60])
        
        physics_pred = torch.FloatTensor(physics_preds).to(history_states.device)
        
        # 融合
        if self.fusion_method == 'weighted':
            weight = 0.7
            fused = weight * learned_pred + (1 - weight) * physics_pred
        elif self.fusion_method == 'adaptive':
            weight = torch.sigmoid(self.fusion_weight)
            fused = weight * learned_pred + (1 - weight) * physics_pred
        else:
            fused = learned_pred
        
        return fused
    
    def get_prediction_for_reward(
        self,
        current_ball_state: np.ndarray,
        robot_position: np.ndarray
    ) -> Dict:
        """
        获取用于奖励计算的预测
        
        使用物理预测器获得精确的预测信息
        """
        pos = current_ball_state[:3]
        vel = current_ball_state[3:6]
        
        # 击球点预测
        contact = self.physics.predict_contact_point(pos, vel)
        
        # 落点预测
        landing = self.physics.predict_landing_point(pos, vel)
        
        # 击球概率
        hit_prob = self.physics.compute_hit_probability(pos, vel, robot_position)
        
        return {
            'contact_point': contact[0] if contact else None,
            'contact_time': contact[1] if contact else None,
            'landing_point': landing[0] if landing else None,
            'landing_time': landing[1] if landing else None,
            'hit_probability': hit_prob
        }
