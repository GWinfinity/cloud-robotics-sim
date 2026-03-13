"""
Table Tennis Ball Physics

实现乒乓球的物理特性，包括:
- 空气阻力
- 马格努斯效应 (旋转)
- 弹跳模型
"""

import numpy as np
import genesis as gs
from typing import Optional, Tuple, List


class TableTennisBall:
    """
    乒乓球物理模型
    
    考虑:
    - 重力
    - 空气阻力
    - 旋转产生的马格努斯力
    - 与球桌/球拍的碰撞
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        init_pos: np.ndarray = np.array([1.0, 0.0, 1.0]),
        radius: float = 0.02,
        mass: float = 0.0027,
        drag_coeff: float = 0.5,
        lift_coeff: float = 0.2,
        restitution: float = 0.85
    ):
        self.scene = scene
        self.radius = radius
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.lift_coeff = lift_coeff
        self.restitution = restitution
        
        # 空气密度
        self.air_density = 1.225
        self.cross_area = np.pi * radius ** 2
        
        # 创建球实体
        self.entity = scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=radius,
                pos=init_pos,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.8, 0.0, 1.0),  # 黄色
            )
        )
        
        # 状态
        self.position = init_pos.copy()
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)  # 旋转
        
        # 轨迹历史
        self.trajectory_history = []
        
        # 碰撞信息
        self.last_bounce_pos = None
        self.bounce_count = 0
        
    def reset(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None
    ):
        """重置球状态"""
        self.position = position.copy()
        self.velocity = velocity.copy() if velocity is not None else np.zeros(3)
        self.angular_velocity = angular_velocity.copy() if angular_velocity is not None else np.zeros(3)
        self.trajectory_history = []
        self.last_bounce_pos = None
        self.bounce_count = 0
        
        self.entity.set_pos(position)
        self.entity.set_vel(self.velocity)
    
    def apply_physics(self, dt: float = 0.008):
        """
        应用物理
        
        包括:
        1. 重力
        2. 空气阻力
        3. 马格努斯力 (如果有旋转)
        """
        # 获取当前速度
        vel = self.entity.get_vel()
        speed = np.linalg.norm(vel)
        
        if speed < 0.01:
            return
        
        # 1. 空气阻力
        drag_force = -0.5 * self.air_density * self.cross_area * self.drag_coeff * speed * vel
        
        # 2. 马格努斯力 (旋转效应)
        magnus_force = np.zeros(3)
        if np.linalg.norm(self.angular_velocity) > 0.1:
            # F_magnus = 0.5 * rho * A * C_L * (omega x v) / |omega|
            cross_prod = np.cross(self.angular_velocity, vel)
            magnus_force = 0.5 * self.air_density * self.cross_area * self.lift_coeff * cross_prod
        
        # 总力
        total_force = drag_force + magnus_force
        
        # 应用力
        self.entity.apply_force(force=total_force)
        
        # 更新状态
        self.velocity = self.entity.get_vel()
        self.position = self.entity.get_pos()
        
        # 记录轨迹
        self.trajectory_history.append({
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'time': self.scene.get_time() if hasattr(self.scene, 'get_time') else 0.0
        })
    
    def check_table_collision(self, table_height: float = 0.76) -> bool:
        """检查与球桌的碰撞"""
        if self.position[2] <= table_height + self.radius and self.velocity[2] < 0:
            # 在桌面上方且向下运动
            if 0.2 < self.position[0] < 1.2 and abs(self.position[1]) < 0.76:
                # 在桌面范围内
                return True
        return False
    
    def handle_bounce(self, surface_normal: np.ndarray = np.array([0, 0, 1])):
        """处理弹跳"""
        # 反射速度
        v_normal = np.dot(self.velocity, surface_normal)
        v_tangent = self.velocity - v_normal * surface_normal
        
        # 应用弹性系数
        v_normal_new = -self.restitution * v_normal
        
        # 摩擦减速 (切向)
        friction_coeff = 0.3
        v_tangent_new = v_tangent * (1 - friction_coeff)
        
        self.velocity = v_normal_new * surface_normal + v_tangent_new
        
        # 更新旋转 (简化)
        self.angular_velocity *= 0.9
        
        # 记录弹跳
        self.last_bounce_pos = self.position.copy()
        self.bounce_count += 1
    
    def check_net_collision(self, net_pos: float = 0.0, net_height: float = 0.1525) -> bool:
        """检查与球网的碰撞"""
        if abs(self.position[0] - net_pos) < self.radius:
            if self.position[2] < net_height:
                return True
        return False
    
    def get_state(self) -> np.ndarray:
        """获取球状态"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.angular_velocity
        ])
    
    def get_trajectory_history(self, n: int = 5) -> np.ndarray:
        """获取最近n个轨迹点"""
        if len(self.trajectory_history) < n:
            # 填充
            pad = [self.trajectory_history[0] if self.trajectory_history else 
                   {'position': np.zeros(3), 'velocity': np.zeros(3)}] * (n - len(self.trajectory_history))
            history = pad + self.trajectory_history
        else:
            history = self.trajectory_history[-n:]
        
        # 展平
        result = []
        for h in history:
            result.extend(h['position'])
            result.extend(h['velocity'][:3])  # 只取线速度
        
        return np.array(result)


class BallTrajectoryPredictor:
    """
    物理预测器
    
    基于物理模型预测球的未来轨迹
    用于训练时的奖励计算
    """
    
    def __init__(
        self,
        ball_config: dict,
        time_horizon: float = 0.5,
        dt: float = 0.01
    ):
        self.ball_config = ball_config
        self.time_horizon = time_horizon
        self.dt = dt
        
    def predict(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        initial_omega: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        预测轨迹
        
        Returns:
            轨迹点列表，每个点包含位置和时间
        """
        trajectory = []
        
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        omega = initial_omega.copy() if initial_omega is not None else np.zeros(3)
        
        t = 0
        while t < self.time_horizon:
            # 记录
            trajectory.append({
                'position': pos.copy(),
                'velocity': vel.copy(),
                'time': t
            })
            
            # 物理积分 (简化欧拉法)
            speed = np.linalg.norm(vel)
            
            # 空气阻力
            drag_accel = -0.5 * self.ball_config['air_density'] * \
                        np.pi * self.ball_config['radius']**2 * \
                        self.ball_config['drag_coefficient'] * speed * vel / self.ball_config['mass']
            
            # 重力
            gravity = np.array([0, 0, -9.81])
            
            # 总加速度
            accel = drag_accel + gravity
            
            # 积分
            vel = vel + accel * self.dt
            pos = pos + vel * self.dt
            
            # 检查地面碰撞
            if pos[2] <= 0:
                pos[2] = 0
                vel[2] = -vel[2] * self.ball_config['restitution']
                break
            
            t += self.dt
        
        return trajectory
    
    def predict_contact_point(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        racket_plane_z: float = 0.8
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        预测击球点
        
        Args:
            racket_plane_z: 球拍所在平面的高度
            
        Returns:
            (击球位置, 时间) 或 None
        """
        trajectory = self.predict(initial_pos, initial_vel)
        
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i]['position'], trajectory[i+1]['position']
            
            # 检查是否穿过球拍平面
            if (p1[2] > racket_plane_z and p2[2] <= racket_plane_z) or \
               (p1[2] < racket_plane_z and p2[2] >= racket_plane_z):
                # 线性插值
                alpha = (racket_plane_z - p1[2]) / (p2[2] - p1[2] + 1e-8)
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
        """
        预测落点
        
        Returns:
            (落点位置, 时间) 或 None
        """
        trajectory = self.predict(initial_pos, initial_vel)
        
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i]['position'], trajectory[i+1]['position']
            
            # 检查是否到达桌面高度
            if p1[2] > table_height and p2[2] <= table_height:
                alpha = (table_height - p1[2]) / (p2[2] - p1[2] + 1e-8)
                landing_pos = p1 + alpha * (p2 - p1)
                landing_time = trajectory[i]['time'] + alpha * self.dt
                
                return landing_pos, landing_time
        
        return None
