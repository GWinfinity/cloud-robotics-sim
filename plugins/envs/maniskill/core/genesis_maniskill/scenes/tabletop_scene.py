"""
Table-top scene builder for manipulation tasks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import genesis as gs


class TableTopSceneBuilder:
    """
    Builder for table-top manipulation scenes.
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        table_size: Tuple[float, float, float] = (1.0, 0.6, 0.05),
        num_objects: int = 1,
        object_types: Optional[List[str]] = None,
        num_envs: int = 1,
        config: Optional[Dict] = None
    ):
        self.scene = scene
        self.table_size = table_size
        self.num_objects = num_objects
        self.object_types = object_types or ["cube"]
        self.num_envs = num_envs
        self.config = config or {}
        
        self.table = None
        self.objects = {}
        
    def build(self):
        """Build the table-top scene."""
        # Build table
        self._build_table()
        
        # Build objects
        self._build_objects()
        
        # Build ground
        self._build_ground()
        
    def _build_table(self):
        """Build the table."""
        table_pos = self.config.get("table_pos", (0, 0, 0.4))
        
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=self.table_size,
                pos=table_pos,
            ),
            material=gs.materials.Rigid(
                color=(0.8, 0.6, 0.4, 1.0)  # Wood color
            )
        )
        
    def _build_objects(self):
        """Build objects on the table."""
        for i in range(self.num_objects):
            obj_type = self.object_types[i % len(self.object_types)]
            obj_name = f"{obj_type}_{i}"
            obj = self._create_object(obj_name, obj_type)
            if obj:
                self.objects[obj_name] = obj
    
    def _create_object(self, name: str, obj_type: str):
        """Create an object."""
        # Random position on table
        table_pos = self.table.get_pos()
        x = table_pos[0] + np.random.uniform(-0.3, 0.3)
        y = table_pos[1] + np.random.uniform(-0.2, 0.2)
        z = table_pos[2] + self.table_size[2] / 2 + 0.05
        
        if obj_type == "cube":
            size = np.random.uniform(0.04, 0.06)
            obj = self.scene.add_entity(
                gs.morphs.Box(
                    size=(size, size, size),
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.2, 0.5, 0.8, 1.0)
                )
            )
        elif obj_type == "sphere":
            radius = np.random.uniform(0.03, 0.05)
            obj = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=radius,
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.8, 0.3, 0.3, 1.0)
                )
            )
        elif obj_type == "cylinder":
            radius = np.random.uniform(0.03, 0.05)
            height = np.random.uniform(0.08, 0.12)
            obj = self.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=radius,
                    height=height,
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.3, 0.7, 0.4, 1.0)
                )
            )
        elif obj_type == "capsule":
            radius = np.random.uniform(0.02, 0.04)
            height = np.random.uniform(0.08, 0.12)
            obj = self.scene.add_entity(
                gs.morphs.Capsule(
                    radius=radius,
                    height=height,
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.7, 0.4, 0.7, 1.0)
                )
            )
        else:
            # Default to cube
            obj = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.05, 0.05, 0.05),
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.5, 0.5, 0.5, 1.0)
                )
            )
        
        return obj
    
    def _build_ground(self):
        """Build ground plane."""
        self.scene.add_entity(
            gs.morphs.Plane(
                size=(10, 10),
                pos=(0, 0, 0),
            ),
            material=gs.materials.Rigid(
                color=(0.2, 0.2, 0.2, 1.0)
            )
        )
    
    def randomize_object_placement(self):
        """Randomize object positions on the table."""
        table_pos = self.table.get_pos()
        
        for obj in self.objects.values():
            x = table_pos[0] + np.random.uniform(-0.3, 0.3)
            y = table_pos[1] + np.random.uniform(-0.2, 0.2)
            z = table_pos[2] + self.table_size[2] / 2 + 0.05
            obj.set_pos(torch.tensor([x, y, z]))
            
            # Random rotation
            if hasattr(obj, 'set_quat'):
                angle = np.random.uniform(0, 2 * np.pi)
                quat = self._euler_to_quat(0, 0, angle)
                obj.set_quat(torch.tensor(quat))
    
    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> Tuple:
        """Convert Euler angles to quaternion."""
        import math
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)
