"""
羽毛球物理模拟

实现了羽毛球特有的空气动力学特性:
- 高速时阻力较小
- 低速时羽毛产生的阻力使球减速
- 翻转效应 (tumble effect)
"""

import numpy as np
import genesis as gs
from typing import Optional, Tuple


class Shuttlecock:
    """
    羽毛球物理模型
    
    羽毛球的特性:
    1. 非对称阻力: 羽毛端阻力大，球头阻力小
    2. 速度衰减: 速度快速下降到终端速度 (~6.5 m/s)
    3. 翻转: 飞行中会翻转使羽毛端朝后
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        init_pos: np.ndarray = np.array([3.0, 0.0, 2.0]),
        mass: float = 0.0052,
        radius: float = 0.033,
        drag_coefficient: float = 0.6,
        terminal_velocity: float = 6.5
    ):
        """
        初始化羽毛球
        
        Args:
            scene: Genesis 场景
            init_pos: 初始位置
            mass: 质量 (kg)
            radius: 球头半径 (m)
            drag_coefficient: 阻力系数
            terminal_velocity: 终端速度 (m/s)
        """
        self.scene = scene
        self.mass = mass
        self.radius = radius
        self.drag_coefficient = drag_coefficient
        self.terminal_velocity = terminal_velocity
        
        # 空气密度
        self.air_density = 1.225
        
        # 羽毛球截面面积 (近似)
        self.cross_area = np.pi * (0.035 ** 2)  # 羽毛端面积
        
        # 创建羽毛球实体
        self.entity = self._create_entity(init_pos)
        
        # 状态记录
        self.position = init_pos.copy()
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # 飞行历史 (用于 EKF)
        self.trajectory_history = []
        self.max_history = 20
        
        # 被击中标志
        self.was_hit = False
        self.hit_by = None
        self.hit_speed = 0.0
        
    def _create_entity(self, init_pos: np.ndarray):
        """创建 Genesis 实体"""
        # 使用球体近似羽毛球
        entity = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=self.radius,
                pos=init_pos,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 0.8, 1.0),  # 白色带黄
            )
        )
        return entity
    
    def reset(self, position: np.ndarray, velocity: Optional[np.ndarray] = None):
        """重置羽毛球状态"""
        self.position = position.copy()
        self.velocity = velocity.copy() if velocity is not None else np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.trajectory_history = []
        self.was_hit = False
        self.hit_by = None
        self.hit_speed = 0.0
        
        # 更新 Genesis 实体
        self.entity.set_pos(position)
        self.entity.set_vel(self.velocity)
    
    def apply_aerodynamics(self):
        """
        应用空气动力学力
        
        这是羽毛球飞行特性的核心:
        - 高速时近似抛物线
        - 速度快速衰减到终端速度
        """
        # 获取当前速度
        vel = self.entity.get_vel()
        speed = np.linalg.norm(vel)
        
        if speed < 0.1:
            return
        
        # 速度方向
        vel_dir = vel / speed
        
        # 计算阻力 (与速度平方成正比)
        # 羽毛球的特殊: 速度越快，减速越快，直到终端速度
        drag_force = 0.5 * self.air_density * self.cross_area * self.drag_coefficient * speed ** 2
        
        # 阻力方向与速度相反
        drag_accel = -drag_force / self.mass * vel_dir
        
        # 添加翻转效应 (tumble effect)
        # 羽毛球会以羽毛端朝后的方式飞行
        # 简化为额外的阻力
        tumble_factor = 1.0 + 0.5 * (speed / self.terminal_velocity)
        drag_accel *= tumble_factor
        
        # 应用力
        self.entity.apply_force(force=drag_accel * self.mass)
        
        # 更新状态
        self.velocity = self.entity.get_vel()
        self.position = self.entity.get_pos()
        
    def record_trajectory(self):
        """记录轨迹点 (用于 EKF)"""
        self.trajectory_history.append({
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'time': self.scene.get_time() if hasattr(self.scene, 'get_time') else 0.0
        })
        
        # 限制历史长度
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)
    
    def get_state(self) -> np.ndarray:
        """
        获取羽毛球状态向量
        
        Returns:
            [x, y, z, vx, vy, vz, speed]
        """
        speed = np.linalg.norm(self.velocity)
        return np.concatenate([self.position, self.velocity, [speed]])
    
    def is_in_court(self, court_bounds: dict) -> bool:
        """检查羽毛球是否在场地内"""
        x, y, z = self.position
        return (
            court_bounds['x_min'] <= x <= court_bounds['x_max'] and
            court_bounds['y_min'] <= y <= court_bounds['y_max'] and
            z >= 0
        )
    
    def is_valid_landing(self, court_bounds: dict, net_x: float = 0.0, side: str = 'opponent') -> bool:
        """
        检查是否为有效落点
        
        Args:
            court_bounds: 场地边界
            net_x: 网的位置 (x坐标)
            side: 'opponent' 或 'self'
        """
        if not self.is_in_court(court_bounds):
            return False
        
        # 检查是否过网
        if side == 'opponent':
            # 必须落在对方场地 (x > net_x)
            return self.position[0] > net_x
        else:
            # 必须落在己方场地 (x < net_x)
            return self.position[0] < net_x
    
    def predict_landing(self, court_height: float = 0.0) -> Optional[np.ndarray]:
        """
        预测落点位置 (简化版)
        
        Args:
            court_height: 场地高度
            
        Returns:
            预测的落点位置 [x, y, 0] 或 None
        """
        if self.velocity[2] >= 0:
            # 仍在上升，无法预测
            return None
        
        # 简化预测: 假设速度方向不变，计算与地面的交点
        time_to_land = (self.position[2] - court_height) / (-self.velocity[2])
        
        if time_to_land < 0:
            return None
        
        # 考虑空气阻力减速
        # 简化: 使用平均速度
        avg_vx = self.velocity[0] * 0.7  # 考虑减速
        avg_vy = self.velocity[1] * 0.7
        
        landing_x = self.position[0] + avg_vx * time_to_land
        landing_y = self.position[1] + avg_vy * time_to_land
        
        return np.array([landing_x, landing_y, court_height])
    
    def check_racket_collision(
        self,
        racket_pos: np.ndarray,
        racket_vel: np.ndarray,
        racket_normal: np.ndarray
    ) -> Tuple[bool, float]:
        """
        检查与球拍的碰撞
        
        Args:
            racket_pos: 球拍位置
            racket_vel: 球拍速度
            racket_normal: 球拍面法向量
            
        Returns:
            (是否碰撞, 碰撞速度)
        """
        # 计算距离
        distance = np.linalg.norm(self.position - racket_pos)
        
        # 球拍面半径约 0.11m，加上球半径
        collision_threshold = 0.11 + self.radius
        
        if distance > collision_threshold:
            return False, 0.0
        
        # 检查球是否在球拍面法向量方向
        to_shuttle = self.position - racket_pos
        to_shuttle_norm = to_shuttle / (np.linalg.norm(to_shuttle) + 1e-8)
        
        # 点积判断是否在击球面方向
        alignment = np.abs(np.dot(to_shuttle_norm, racket_normal))
        
        if alignment < 0.5:  # 需要大致朝向球拍面
            return False, 0.0
        
        # 计算碰撞速度 (球拍速度 + 球速度)
        relative_vel = racket_vel - self.velocity
        impact_speed = np.linalg.norm(relative_vel)
        
        return True, impact_speed
    
    def apply_hit(
        self,
        hit_direction: np.ndarray,
        hit_speed: float,
        hit_point: np.ndarray
    ):
        """
        应用击球效果
        
        Args:
            hit_direction: 击球方向 (单位向量)
            hit_speed: 击球速度
            hit_point: 击球点位置
        """
        # 设置新速度
        self.velocity = hit_direction * hit_speed
        
        # 更新位置到击球点
        self.position = hit_point
        self.entity.set_pos(hit_point)
        self.entity.set_vel(self.velocity)
        
        # 记录击球信息
        self.was_hit = True
        self.hit_speed = hit_speed
        
    def get_trajectory_for_ekf(self) -> np.ndarray:
        """获取用于 EKF 的轨迹数据"""
        if len(self.trajectory_history) < 3:
            return np.array([])
        
        # 返回最近的位置观测
        positions = [t['position'] for t in self.trajectory_history[-5:]]
        return np.array(positions)


class BadmintonNet:
    """羽毛球网"""
    
    def __init__(self, scene: gs.Scene, pos: np.ndarray = np.array([0, 0, 0.77])):
        self.scene = scene
        self.pos = pos
        self.height = 1.55
        self.width = 6.1
        
        # 创建网 (使用一系列小圆柱体或平面)
        self.entity = self._create_net()
    
    def _create_net(self):
        """创建网实体"""
        # 简化: 使用一个平面表示网
        entity = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.05, self.width, self.height),
                pos=self.pos + np.array([0, 0, self.height/2 - 0.77])
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0, 0.8),
            )
        )
        return entity


class BadmintonCourt:
    """羽毛球场地"""
    
    def __init__(self, scene: gs.Scene, config: dict):
        self.scene = scene
        self.config = config
        self.bounds = config['bounds']
        
        # 创建场地
        self._create_court()
        self._create_net()
    
    def _create_court(self):
        """创建场地地面"""
        length = self.config['length']
        width = self.config['width']
        
        # 地面
        self.floor = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                color=(0.3, 0.5, 0.3, 1.0),  # 绿色场地
            )
        )
        
        # 场地边界线 (简化: 使用薄长方体)
        line_height = 0.01
        line_width = 0.05
        
        # 边线
        for x in [self.bounds['x_min'], self.bounds['x_max']]:
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(line_width, width, line_height),
                    pos=[x, 0, line_height/2]
                ),
                surface=gs.surfaces.Default(color=(1, 1, 1, 1))
            )
        
        for y in [self.bounds['y_min'], self.bounds['y_max']]:
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(length, line_width, line_height),
                    pos=[0, y, line_height/2]
                ),
                surface=gs.surfaces.Default(color=(1, 1, 1, 1))
            )
        
        # 中线
        self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(line_width, width, line_height),
                pos=[0, 0, line_height/2]
            ),
            surface=gs.surfaces.Default(color=(1, 1, 1, 1))
        )
    
    def _create_net(self):
        """创建球网"""
        self.net = BadmintonNet(self.scene, np.array(self.config['net_pos']))
    
    def get_side(self, position: np.ndarray) -> str:
        """
        获取位置所在场地 side
        
        Returns:
            'left' (己方) 或 'right' (对方)
        """
        return 'left' if position[0] < 0 else 'right'
    
    def is_valid_serve_position(self, position: np.ndarray, serving_side: str) -> bool:
        """检查是否为有效发球位置"""
        x, y = position[0], position[1]
        
        # 简化检查: 在己方半场底线附近
        if serving_side == 'left':
            return -6.5 <= x <= -5.5 and self.bounds['y_min'] <= y <= self.bounds['y_max']
        else:
            return 5.5 <= x <= 6.5 and self.bounds['y_min'] <= y <= self.bounds['y_max']
