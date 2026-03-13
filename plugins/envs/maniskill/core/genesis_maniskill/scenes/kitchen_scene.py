"""
Kitchen scene builder adapted from RoboCasa for Genesis.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import genesis as gs


class KitchenSceneBuilder:
    """
    Builder for kitchen scenes.
    
    Supports multiple layouts and styles similar to RoboCasa.
    """
    
    # Room dimensions
    ROOM_SIZE = (4.0, 3.0, 3.0)  # width, depth, height
    WALL_THICKNESS = 0.1
    
    def __init__(
        self,
        scene: gs.Scene,
        layout_id: int = 0,
        style_id: int = 0,
        num_envs: int = 1,
        config: Optional[Dict] = None
    ):
        self.scene = scene
        self.layout_id = layout_id
        self.style_id = style_id
        self.num_envs = num_envs
        self.config = config or {}
        
        self.fixtures = {}
        self.objects = {}
        self.floor = None
        self.walls = []
        
    def build(self):
        """Build the kitchen scene."""
        # Build room structure
        self._build_room()
        
        # Build fixtures based on layout
        self._build_fixtures()
        
        # Place objects
        self._place_objects()
        
    def _build_room(self):
        """Build room structure (floor and walls)."""
        # Floor
        self.floor = self.scene.add_entity(
            gs.morphs.Plane(
                size=self.ROOM_SIZE[:2],
                pos=(0, 0, 0),
            ),
            material=self._get_floor_material()
        )
        
        # Walls (4 walls)
        wall_positions = [
            # Back wall
            ((0, -self.ROOM_SIZE[1]/2, self.ROOM_SIZE[2]/2), (0, 0, 0)),
            # Front wall
            ((0, self.ROOM_SIZE[1]/2, self.ROOM_SIZE[2]/2), (0, 0, 0)),
            # Left wall
            ((-self.ROOM_SIZE[0]/2, 0, self.ROOM_SIZE[2]/2), (0, 0, np.pi/2)),
            # Right wall
            ((self.ROOM_SIZE[0]/2, 0, self.ROOM_SIZE[2]/2), (0, 0, np.pi/2)),
        ]
        
        for i, (pos, rot) in enumerate(wall_positions):
            wall = self.scene.add_entity(
                gs.morphs.Box(
                    size=(self.ROOM_SIZE[0], self.WALL_THICKNESS, self.ROOM_SIZE[2]),
                    pos=pos,
                    euler=rot,
                ),
                material=self._get_wall_material()
            )
            self.walls.append(wall)
    
    def _build_fixtures(self):
        """Build kitchen fixtures based on layout."""
        # Define fixture types and positions for each layout
        layout_configs = self._get_layout_config()
        
        for fixture_name, fixture_config in layout_configs.items():
            fixture = self._create_fixture(fixture_name, fixture_config)
            if fixture:
                self.fixtures[fixture_name] = fixture
    
    def _get_layout_config(self) -> Dict:
        """Get fixture configuration for current layout."""
        # Simplified layouts - can be extended
        layouts = {
            # G-shaped
            0: {
                "counter_main": {"type": "counter", "pos": (0, -1.0, 0.45), "size": (2.0, 0.6, 0.9)},
                "counter_left": {"type": "counter", "pos": (-1.0, 0, 0.45), "size": (0.6, 2.0, 0.9)},
                "counter_right": {"type": "counter", "pos": (1.0, -0.3, 0.45), "size": (0.6, 1.4, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.2, -1.2, 0.9), "size": (0.8, 0.7, 1.8)},
                "stove": {"type": "stove", "pos": (0, -1.0, 0.9), "size": (0.6, 0.6, 0.1)},
                "sink": {"type": "sink", "pos": (0.5, -1.0, 0.9), "size": (0.6, 0.5, 0.1)},
            },
            # U-shaped
            1: {
                "counter_back": {"type": "counter", "pos": (0, -1.0, 0.45), "size": (3.0, 0.6, 0.9)},
                "counter_left": {"type": "counter", "pos": (-1.2, 0, 0.45), "size": (0.6, 2.0, 0.9)},
                "counter_right": {"type": "counter", "pos": (1.2, 0, 0.45), "size": (0.6, 2.0, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.2, -1.2, 0.9)},
                "stove": {"type": "stove", "pos": (0, -1.0, 0.9)},
                "sink": {"type": "sink", "pos": (1.0, -1.0, 0.9)},
            },
            # L-shaped
            2: {
                "counter_main": {"type": "counter", "pos": (0, -1.0, 0.45), "size": (2.0, 0.6, 0.9)},
                "counter_side": {"type": "counter", "pos": (-1.0, 0, 0.45), "size": (0.6, 2.0, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.2, -1.2, 0.9)},
                "stove": {"type": "stove", "pos": (0.5, -1.0, 0.9)},
                "sink": {"type": "sink", "pos": (-0.5, -1.0, 0.9)},
            },
            # Single wall
            3: {
                "counter": {"type": "counter", "pos": (0, -1.0, 0.45), "size": (3.0, 0.6, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.0, -1.2, 0.9)},
                "stove": {"type": "stove", "pos": (0, -1.0, 0.9)},
                "sink": {"type": "sink", "pos": (1.0, -1.0, 0.9)},
            },
            # Island
            4: {
                "counter_back": {"type": "counter", "pos": (0, -1.0, 0.45), "size": (3.0, 0.6, 0.9)},
                "island": {"type": "counter", "pos": (0, 0.3, 0.45), "size": (1.5, 0.8, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.2, -1.2, 0.9)},
                "stove": {"type": "stove", "pos": (0, -1.0, 0.9)},
                "sink": {"type": "sink", "pos": (1.0, -1.0, 0.9)},
            },
            # Galley
            5: {
                "counter_left": {"type": "counter", "pos": (-1.0, 0, 0.45), "size": (0.6, 3.0, 0.9)},
                "counter_right": {"type": "counter", "pos": (1.0, 0, 0.45), "size": (0.6, 3.0, 0.9)},
                "fridge": {"type": "fridge", "pos": (-1.0, -1.2, 0.9)},
                "stove": {"type": "stove", "pos": (-1.0, 0, 0.9)},
                "sink": {"type": "sink", "pos": (1.0, 0, 0.9)},
            },
        }
        
        # Return layout or default to G-shaped
        return layouts.get(self.layout_id, layouts[0])
    
    def _create_fixture(self, name: str, config: Dict):
        """Create a kitchen fixture."""
        fixture_type = config["type"]
        pos = config["pos"]
        size = config.get("size", (0.6, 0.6, 0.9))
        
        if fixture_type == "counter":
            return self.scene.add_entity(
                gs.morphs.Box(
                    size=size,
                    pos=pos,
                ),
                material=self._get_counter_material()
            )
        elif fixture_type == "fridge":
            return self._create_fridge(name, pos, size)
        elif fixture_type == "stove":
            return self._create_stove(name, pos, size)
        elif fixture_type == "sink":
            return self._create_sink(name, pos, size)
        elif fixture_type == "cabinet":
            return self._create_cabinet(name, pos, size)
        
        return None
    
    def _create_fridge(self, name: str, pos: tuple, size: tuple):
        """Create fridge fixture."""
        # Main body
        body = self.scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            material=gs.materials.Rigid(
                color=(0.9, 0.9, 0.95, 1.0)  # White-ish
            )
        )
        return body
    
    def _create_stove(self, name: str, pos: tuple, size: tuple):
        """Create stove fixture."""
        stove = self.scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            material=gs.materials.Rigid(
                color=(0.2, 0.2, 0.2, 1.0)  # Black
            )
        )
        return stove
    
    def _create_sink(self, name: str, pos: tuple, size: tuple):
        """Create sink fixture."""
        sink = self.scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            material=gs.materials.Metal(
                color=(0.7, 0.7, 0.75, 1.0)  # Steel
            )
        )
        return sink
    
    def _create_cabinet(self, name: str, pos: tuple, size: tuple):
        """Create cabinet fixture."""
        cabinet = self.scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            material=self._get_cabinet_material()
        )
        return cabinet
    
    def _place_objects(self):
        """Place objects in the kitchen."""
        # Common kitchen objects
        object_types = ["apple", "orange", "banana", "bowl", "plate", "mug", "bottle"]
        
        for obj_name in object_types:
            obj = self._create_object(obj_name)
            if obj:
                self.objects[obj_name] = obj
    
    def _create_object(self, obj_type: str):
        """Create a kitchen object."""
        # Random position on counter
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-1.2, -0.6)
        z = 1.0  # On top of counter
        
        if obj_type in ["apple", "orange", "banana"]:
            # Sphere-ish fruits
            radius = 0.05
            obj = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=radius,
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=self._get_object_color(obj_type)
                )
            )
        elif obj_type in ["bowl", "plate", "mug"]:
            # Cylinder-ish containers
            obj = self.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.08,
                    height=0.1,
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.9, 0.9, 0.9, 1.0)
                )
            )
        else:
            # Box-ish objects
            obj = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.2),
                    pos=(x, y, z),
                ),
                material=gs.materials.Rigid(
                    color=(0.5, 0.5, 0.7, 1.0)
                )
            )
        
        return obj
    
    def randomize_object_placement(self):
        """Randomize object positions."""
        for obj in self.objects.values():
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-1.2, -0.6)
            z = 1.0
            obj.set_pos(torch.tensor([x, y, z]))
    
    def _get_floor_material(self):
        """Get floor material based on style."""
        style_colors = {
            0: (0.8, 0.8, 0.8),  # modern - gray
            1: (0.6, 0.6, 0.6),  # industrial - dark gray
            2: (0.9, 0.85, 0.7),  # mediterranean - beige
            3: (0.85, 0.8, 0.75), # transitional
            4: (0.7, 0.6, 0.5),   # rustic - brown
            5: (0.9, 0.9, 0.85),  # farmhouse - cream
            6: (0.8, 0.85, 0.9),  # coastal - light blue
            7: (0.75, 0.7, 0.65), # traditional
        }
        color = style_colors.get(self.style_id, style_colors[0])
        return gs.materials.Rigid(color=(*color, 1.0))
    
    def _get_wall_material(self):
        """Get wall material."""
        return gs.materials.Rigid(color=(0.95, 0.95, 0.9, 1.0))
    
    def _get_counter_material(self):
        """Get counter material."""
        return gs.materials.Rigid(color=(0.8, 0.7, 0.6, 1.0))
    
    def _get_cabinet_material(self):
        """Get cabinet material."""
        return gs.materials.Rigid(color=(0.9, 0.9, 0.9, 1.0))
    
    def _get_object_color(self, obj_type: str) -> tuple:
        """Get object color."""
        colors = {
            "apple": (0.9, 0.2, 0.2),
            "orange": (1.0, 0.6, 0.2),
            "banana": (0.9, 0.9, 0.3),
        }
        return colors.get(obj_type, (0.5, 0.5, 0.5))
