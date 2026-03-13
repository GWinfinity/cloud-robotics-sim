"""
Bulky Object Generation

生成多样化的物体用于拥抱任务
"""

import numpy as np
import genesis as gs
from typing import Dict, Tuple, List


class BulkyObjectGenerator:
    """
    大型物体生成器
    
    生成不同形状和大小的物体:
    - 长方体
    - 圆柱体
    - 球体
    - 不规则形状
    """
    
    def __init__(self, scene: gs.Scene, config: Dict):
        self.scene = scene
        self.config = config
        
    def generate_box(self, size: Tuple[float, float, float], 
                     position: np.ndarray, mass: float) -> gs.Entity:
        """生成长方体"""
        entity = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=position
            ),
            surface=gs.surfaces.Default(
                color=(0.6, 0.4, 0.3, 1.0),  # 棕色纸箱颜色
            )
        )
        # 设置质量和摩擦
        # 注意: Genesis中需要通过其他方式设置物理属性
        return entity
    
    def generate_cylinder(self, radius: float, height: float,
                          position: np.ndarray, mass: float) -> gs.Entity:
        """生成圆柱体"""
        entity = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=radius,
                height=height,
                pos=position
            ),
            surface=gs.surfaces.Default(
                color=(0.5, 0.5, 0.6, 1.0),  # 灰色
            )
        )
        return entity
    
    def generate_sphere(self, radius: float,
                        position: np.ndarray, mass: float) -> gs.Entity:
        """生成球体"""
        entity = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=radius,
                pos=position
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.7, 0.5, 1.0),  # 米黄色
            )
        )
        return entity
    
    def generate_random_object(self, object_type: str = None) -> Tuple[gs.Entity, Dict]:
        """
        生成随机物体
        
        Returns:
            entity: Genesis实体
            info: 物体信息字典
        """
        # 随机选择类型
        if object_type is None:
            object_type = np.random.choice(['box', 'cylinder', 'sphere'])
        
        # 随机尺寸
        size_range = self.config.get('size_range', [0.3, 0.8])
        mass_range = self.config.get('mass_range', [5, 50])
        
        if object_type == 'box':
            size = (
                np.random.uniform(*size_range),
                np.random.uniform(*size_range),
                np.random.uniform(*size_range)
            )
            mass = np.random.uniform(*mass_range)
            position = np.array([0.5, 0, size[2]/2 + 0.1])  # 前方地面上
            entity = self.generate_box(size, position, mass)
            info = {'type': 'box', 'size': size, 'mass': mass, 'position': position}
            
        elif object_type == 'cylinder':
            radius = np.random.uniform(0.15, 0.4)
            height = np.random.uniform(*size_range)
            mass = np.random.uniform(*mass_range)
            position = np.array([0.5, 0, height/2 + 0.1])
            entity = self.generate_cylinder(radius, height, position, mass)
            info = {'type': 'cylinder', 'radius': radius, 'height': height, 
                   'mass': mass, 'position': position}
            
        else:  # sphere
            radius = np.random.uniform(0.2, 0.4)
            mass = np.random.uniform(*mass_range)
            position = np.array([0.5, 0, radius + 0.1])
            entity = self.generate_sphere(radius, position, mass)
            info = {'type': 'sphere', 'radius': radius, 'mass': mass, 'position': position}
        
        return entity, info
    
    def get_object_mesh_vertices(self, entity: gs.Entity, info: Dict) -> np.ndarray:
        """
        获取物体网格顶点 (用于NSDF)
        
        简化为采样表面点
        """
        obj_type = info['type']
        pos = info['position']
        
        if obj_type == 'box':
            size = info['size']
            # 采样长方体表面
            vertices = self._sample_box_surface(size, pos, num_points=1000)
        elif obj_type == 'cylinder':
            radius = info['radius']
            height = info['height']
            vertices = self._sample_cylinder_surface(radius, height, pos, num_points=1000)
        else:  # sphere
            radius = info['radius']
            vertices = self._sample_sphere_surface(radius, pos, num_points=1000)
        
        return vertices
    
    def _sample_box_surface(self, size: Tuple[float, float, float], 
                           position: np.ndarray, num_points: int = 1000) -> np.ndarray:
        """采样长方体表面点"""
        points = []
        sx, sy, sz = size
        
        for _ in range(num_points):
            face = np.random.randint(0, 6)
            if face == 0:  # +x
                point = [sx/2, np.random.uniform(-sy/2, sy/2), np.random.uniform(-sz/2, sz/2)]
            elif face == 1:  # -x
                point = [-sx/2, np.random.uniform(-sy/2, sy/2), np.random.uniform(-sz/2, sz/2)]
            elif face == 2:  # +y
                point = [np.random.uniform(-sx/2, sx/2), sy/2, np.random.uniform(-sz/2, sz/2)]
            elif face == 3:  # -y
                point = [np.random.uniform(-sx/2, sx/2), -sy/2, np.random.uniform(-sz/2, sz/2)]
            elif face == 4:  # +z
                point = [np.random.uniform(-sx/2, sx/2), np.random.uniform(-sy/2, sy/2), sz/2]
            else:  # -z
                point = [np.random.uniform(-sx/2, sx/2), np.random.uniform(-sy/2, sy/2), -sz/2]
            
            points.append(point)
        
        return np.array(points) + position
    
    def _sample_cylinder_surface(self, radius: float, height: float,
                                  position: np.ndarray, num_points: int = 1000) -> np.ndarray:
        """采样圆柱体表面点"""
        points = []
        
        for _ in range(num_points):
            theta = np.random.uniform(0, 2*np.pi)
            face = np.random.choice(['side', 'top', 'bottom'])
            
            if face == 'side':
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                z = np.random.uniform(-height/2, height/2)
            elif face == 'top':
                r = np.random.uniform(0, radius)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = height/2
            else:  # bottom
                r = np.random.uniform(0, radius)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = -height/2
            
            points.append([x, y, z])
        
        return np.array(points) + position
    
    def _sample_sphere_surface(self, radius: float,
                                position: np.ndarray, num_points: int = 1000) -> np.ndarray:
        """采样球体表面点"""
        points = []
        
        for _ in range(num_points):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points) + position
