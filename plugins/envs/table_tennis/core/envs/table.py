"""
Table Tennis Table Model

标准ITTF乒乓球桌规格
"""

import numpy as np
import genesis as gs


class TableTennisTable:
    """
    乒乓球桌模型
    
    标准尺寸:
    - 长度: 2.74m
    - 宽度: 1.525m
    - 高度: 0.76m
    - 网高: 0.1525m (15.25cm)
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        position: np.ndarray = np.array([0, 0, 0]),
        config: dict = None
    ):
        self.scene = scene
        self.position = position
        self.config = config or {}
        
        # 标准尺寸
        self.length = 2.74
        self.width = 1.525
        self.height = 0.76
        self.net_height = 0.1525
        
        # 创建球桌
        self._create_table()
        self._create_net()
        self._create_lines()
    
    def _create_table(self):
        """创建桌面"""
        # 桌面 (绿色)
        color = self.config.get('table_color', [0.1, 0.3, 0.1, 1.0])
        
        self.table_surface = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(self.length, self.width, 0.02),
                pos=self.position + np.array([0, 0, self.height])
            ),
            surface=gs.surfaces.Default(
                color=color,
                roughness=0.8,
            )
        )
        
        # 桌腿
        leg_positions = [
            [-self.length/2 + 0.2, -self.width/2 + 0.2, self.height/2],
            [-self.length/2 + 0.2, self.width/2 - 0.2, self.height/2],
            [self.length/2 - 0.2, -self.width/2 + 0.2, self.height/2],
            [self.length/2 - 0.2, self.width/2 - 0.2, self.height/2],
        ]
        
        for leg_pos in leg_positions:
            self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=0.03,
                    height=self.height,
                    pos=self.position + leg_pos
                ),
                surface=gs.surfaces.Default(
                    color=(0.2, 0.2, 0.2, 1.0)
                )
            )
    
    def _create_net(self):
        """创建球网"""
        # 网柱
        post_positions = [
            [0, -self.width/2 - 0.015, self.height + self.net_height/2],
            [0, self.width/2 + 0.015, self.height + self.net_height/2]
        ]
        
        for post_pos in post_positions:
            self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=0.015,
                    height=self.net_height,
                    pos=self.position + post_pos
                ),
                surface=gs.surfaces.Default(
                    color=(0.1, 0.1, 0.1, 1.0)
                )
            )
        
        # 网面
        self.net = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.02, self.width + 0.03, self.net_height),
                pos=self.position + np.array([0, 0, self.height + self.net_height/2])
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0, 0.5),
            )
        )
    
    def _create_lines(self):
        """创建边线"""
        line_width = 0.02
        line_color = (1.0, 1.0, 1.0, 1.0)
        z_pos = self.height + 0.01  # 略高于桌面
        
        # 边线
        lines = [
            # 长边
            ([self.length, line_width], [0, -self.width/2, z_pos]),
            ([self.length, line_width], [0, self.width/2, z_pos]),
            # 短边
            ([line_width, self.width], [-self.length/2, 0, z_pos]),
            ([line_width, self.width], [self.length/2, 0, z_pos]),
            # 中线
            ([line_width, self.width], [0, 0, z_pos]),
        ]
        
        for size, pos in lines:
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(size[0], size[1], 0.005),
                    pos=self.position + np.array(pos)
                ),
                surface=gs.surfaces.Default(color=line_color)
            )
    
    def is_in_bounds(self, pos: np.ndarray, side: str = 'either') -> bool:
        """
        检查位置是否在有效区域内
        
        Args:
            pos: 位置 [x, y, z]
            side: 'left', 'right', 或 'either'
        """
        x, y = pos[0], pos[1]
        
        # 基本边界检查
        if abs(y) > self.width / 2:
            return False
        
        if side == 'left':
            return -self.length/2 <= x <= 0
        elif side == 'right':
            return 0 <= x <= self.length/2
        else:
            return -self.length/2 <= x <= self.length/2
    
    def get_serve_position(self, side: str = 'left') -> np.ndarray:
        """获取发球位置"""
        if side == 'left':
            x = -self.length/2 + 0.5
        else:
            x = self.length/2 - 0.5
        
        y = np.random.uniform(-self.width/4, self.width/4)
        z = self.height + 0.2
        
        return self.position + np.array([x, y, z])


class Racket:
    """
    乒乓球拍
    
    简化为附着在机器人手上的碰撞体
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        radius: float = 0.08,
        thickness: float = 0.01
    ):
        self.scene = scene
        self.radius = radius
        self.thickness = thickness
        
        # 创建球拍模型
        self.entity = scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=radius,
                height=thickness,
                pos=[0, 0, 0]  # 位置由机器人手控制
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.2, 0.2, 1.0),  # 红色
            )
        )
        
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.normal = np.array([0, 0, 1])  # 拍面法向量
    
    def update_transform(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        normal: np.ndarray
    ):
        """更新球拍位置和朝向"""
        self.position = position
        self.velocity = velocity
        self.normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        self.entity.set_pos(position)
        # 设置朝向 (简化)
    
    def check_collision(self, ball_pos: np.ndarray, ball_radius: float = 0.02) -> bool:
        """检查与球的碰撞"""
        distance = np.linalg.norm(ball_pos - self.position)
        return distance < (self.radius + ball_radius)
