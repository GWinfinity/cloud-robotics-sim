"""
扩展卡尔曼滤波 (EKF) 羽毛球轨迹预测

用于估计和预测羽毛球的运动轨迹，实现精确的击球时机控制。
"""

import numpy as np
from typing import Optional, List, Tuple


class ShuttlecockEKF:
    """
    羽毛球轨迹 EKF 预测器
    
    状态向量: [x, y, z, vx, vy, vz, ax, ay, az]^T
    观测向量: [x, y, z]^T (位置)
    
    考虑羽毛球的非线性空气动力学
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        prediction_horizon: float = 0.5
    ):
        """
        初始化 EKF
        
        Args:
            dt: 时间步长
            process_noise: 过程噪声
            measurement_noise: 观测噪声
            prediction_horizon: 预测时间范围
        """
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        
        # 状态维度 (位置3 + 速度3 + 加速度3)
        self.state_dim = 9
        self.measurement_dim = 3
        
        # 状态: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim)
        
        # 过程噪声协方差
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[0:3, 0:3] *= 0.1  # 位置噪声较小
        self.Q[3:6, 3:6] *= 1.0  # 速度噪声中等
        self.Q[6:9, 6:9] *= 10.0  # 加速度噪声较大
        
        # 观测噪声协方差
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # 观测矩阵 (只观测位置)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0:3, 0:3] = np.eye(3)
        
        # 羽毛球物理参数
        self.mass = 0.0052
        self.drag_coefficient = 0.6
        self.air_density = 1.225
        self.cross_area = np.pi * (0.035 ** 2)
        self.terminal_velocity = 6.5
        
        # 重力
        self.gravity = np.array([0, 0, -9.81])
        
    def reset(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        """重置滤波器"""
        self.state[0:3] = initial_position
        if initial_velocity is not None:
            self.state[3:6] = initial_velocity
        else:
            self.state[3:6] = np.zeros(3)
        self.state[6:9] = self.gravity  # 初始加速度为重力
        
        self.covariance = np.eye(self.state_dim)
        self.covariance[0:3, 0:3] *= 0.1  # 位置不确定性
        self.covariance[3:6, 3:6] *= 1.0  # 速度不确定性
        self.covariance[6:9, 6:9] *= 10.0  # 加速度不确定性
        
    def predict(self):
        """预测步骤 (非线性动力学)"""
        # 提取当前状态
        pos = self.state[0:3]
        vel = self.state[3:6]
        
        # 计算空气阻力 (非线性)
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            drag_force = 0.5 * self.air_density * self.cross_area * self.drag_coefficient * speed ** 2
            drag_accel = -drag_force / self.mass * (vel / speed)
        else:
            drag_accel = np.zeros(3)
        
        # 总加速度 (重力 + 阻力)
        accel = self.gravity + drag_accel
        
        # 状态转移 (使用当前加速度)
        self.state[0:3] += vel * self.dt + 0.5 * accel * self.dt ** 2
        self.state[3:6] += accel * self.dt
        self.state[6:9] = accel
        
        # 计算雅可比矩阵 (线性化)
        F = self._compute_jacobian(vel, speed)
        
        # 更新协方差
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """更新步骤 (卡尔曼增益)"""
        # 预测观测
        predicted_measurement = self.H @ self.state
        
        # 创新 (观测残差)
        innovation = measurement - predicted_measurement
        
        # 创新协方差
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.state = self.state + K @ innovation
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ self.H) @ self.covariance
        
    def _compute_jacobian(self, vel: np.ndarray, speed: float) -> np.ndarray:
        """
        计算状态转移雅可比矩阵
        
        对非线性动力学进行线性化
        """
        F = np.eye(self.state_dim)
        
        # 位置对速度的偏导
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # 速度对加速度的偏导
        F[3:6, 6:9] = np.eye(3) * self.dt
        
        # 加速度对速度的偏导 (阻力项)
        if speed > 0.1:
            # 阻力加速度对速度的偏导
            v_norm = vel / speed
            
            # d(a_drag)/dv = -k * (I * |v| + v * v^T / |v|) / |v|
            k = 0.5 * self.air_density * self.cross_area * self.drag_coefficient / self.mass
            
            drag_jacobian = -k * (speed * np.eye(3) + np.outer(vel, vel) / speed)
            
            F[6:9, 3:6] = drag_jacobian
        
        return F
    
    def predict_trajectory(self, horizon: Optional[float] = None) -> np.ndarray:
        """
        预测未来轨迹
        
        Args:
            horizon: 预测时间范围 (秒)
            
        Returns:
            预测的轨迹点数组 [N, 3]
        """
        if horizon is None:
            horizon = self.prediction_horizon
        
        num_steps = int(horizon / self.dt)
        
        # 保存当前状态
        saved_state = self.state.copy()
        saved_covariance = self.covariance.copy()
        
        # 预测轨迹
        trajectory = np.zeros((num_steps, 3))
        
        for i in range(num_steps):
            # 提取当前位置
            trajectory[i] = self.state[0:3].copy()
            
            # 预测下一步 (不使用更新，因为未来没有观测)
            self.predict()
            
            # 检查是否落地
            if self.state[2] <= 0:
                trajectory = trajectory[:i+1]
                break
        
        # 恢复状态
        self.state = saved_state
        self.covariance = saved_covariance
        
        return trajectory
    
    def get_impact_point(self, court_height: float = 0.0) -> Optional[Tuple[np.ndarray, float]]:
        """
        预测落点位置和时间
        
        Args:
            court_height: 场地高度
            
        Returns:
            (落点位置, 落地时间) 或 None
        """
        trajectory = self.predict_trajectory(horizon=3.0)  # 预测3秒
        
        # 查找与地面的交点
        for i in range(len(trajectory) - 1):
            z1, z2 = trajectory[i, 2], trajectory[i+1, 2]
            
            if z1 > court_height and z2 <= court_height:
                # 线性插值
                t = (court_height - z1) / (z2 - z1)
                landing_pos = trajectory[i] + t * (trajectory[i+1] - trajectory[i])
                landing_time = i * self.dt + t * self.dt
                
                return landing_pos, landing_time
        
        return None
    
    def get_optimal_hit_point(
        self,
        robot_position: np.ndarray,
        max_reach: float = 2.0,
        min_height: float = 0.5,
        max_height: float = 2.5
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        计算最优击球点
        
        考虑机器人的可达范围和击球高度
        
        Args:
            robot_position: 机器人位置
            max_reach: 最大 reach 距离
            min_height: 最小击球高度
            max_height: 最大击球高度
            
        Returns:
            (最优击球点, 击球时间) 或 None
        """
        trajectory = self.predict_trajectory(horizon=2.0)
        
        best_point = None
        best_time = 0
        best_score = -float('inf')
        
        for i, point in enumerate(trajectory):
            # 检查高度
            if point[2] < min_height or point[2] > max_height:
                continue
            
            # 检查距离
            distance = np.linalg.norm(point[:2] - robot_position[:2])
            if distance > max_reach:
                continue
            
            # 评分 (考虑距离和高度)
            # 越靠近场地中心越好，高度适中越好
            court_center = np.array([0, 0])
            dist_to_center = np.linalg.norm(point[:2] - court_center)
            
            # 高度适中 (1.2m - 1.8m) 得分更高
            height_score = 1.0 - abs(point[2] - 1.5) / 1.5
            
            score = (1.0 - distance / max_reach) * 0.4 + \
                   (1.0 - dist_to_center / 5.0) * 0.3 + \
                   height_score * 0.3
            
            if score > best_score:
                best_score = score
                best_point = point
                best_time = i * self.dt
        
        if best_point is not None:
            return best_point, best_time
        
        return None
    
    def step(self, measurement: Optional[np.ndarray] = None):
        """执行一步 EKF (预测 + 可选更新)"""
        self.predict()
        
        if measurement is not None:
            self.update(measurement)


class PredictionFreeVariant:
    """
    无预测变体
    
    论文中提到的 prediction-free 方法，直接使用当前观测而不进行轨迹预测
    """
    
    def __init__(self, observation_history_size: int = 3):
        self.history_size = observation_history_size
        self.position_history = []
        self.velocity_estimate = np.zeros(3)
        
    def update(self, current_position: np.ndarray):
        """更新观测"""
        self.position_history.append(current_position.copy())
        
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # 估计速度
        if len(self.position_history) >= 2:
            dt = 0.01  # 假设固定时间步
            self.velocity_estimate = (
                self.position_history[-1] - self.position_history[-2]
            ) / dt
    
    def get_velocity(self) -> np.ndarray:
        """获取估计速度"""
        return self.velocity_estimate
    
    def get_position(self) -> np.ndarray:
        """获取当前位置"""
        if len(self.position_history) > 0:
            return self.position_history[-1]
        return np.zeros(3)
    
    def should_hit(self, robot_position: np.ndarray, threshold_distance: float = 1.5) -> bool:
        """
        判断是否应该击球 (简化决策)
        
        基于当前距离和接近速度
        """
        if len(self.position_history) == 0:
            return False
        
        current_pos = self.position_history[-1]
        distance = np.linalg.norm(current_pos[:2] - robot_position[:2])
        
        # 距离足够近且在下降
        if distance < threshold_distance and self.velocity_estimate[2] < 0:
            return True
        
        return False


class TrajectoryBuffer:
    """
    轨迹缓冲区
    
    用于存储和可视化羽毛球轨迹
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.actual_trajectory = []
        self.predicted_trajectory = []
        
    def add_actual(self, position: np.ndarray):
        """添加实际轨迹点"""
        self.actual_trajectory.append(position.copy())
        if len(self.actual_trajectory) > self.max_size:
            self.actual_trajectory.pop(0)
    
    def set_predicted(self, trajectory: np.ndarray):
        """设置预测轨迹"""
        self.predicted_trajectory = trajectory.copy()
    
    def get_actual(self) -> np.ndarray:
        """获取实际轨迹"""
        return np.array(self.actual_trajectory)
    
    def get_predicted(self) -> np.ndarray:
        """获取预测轨迹"""
        return self.predicted_trajectory
    
    def clear(self):
        """清空缓冲区"""
        self.actual_trajectory = []
        self.predicted_trajectory = []
    
    def compute_prediction_error(self) -> float:
        """计算预测误差"""
        if len(self.actual_trajectory) == 0 or len(self.predicted_trajectory) == 0:
            return 0.0
        
        # 计算实际和预测轨迹的差异
        min_len = min(len(self.actual_trajectory), len(self.predicted_trajectory))
        errors = []
        
        for i in range(min_len):
            error = np.linalg.norm(
                self.actual_trajectory[-min_len+i] - self.predicted_trajectory[i]
            )
            errors.append(error)
        
        return np.mean(errors) if errors else 0.0
