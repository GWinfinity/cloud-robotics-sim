# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""
Furniture Assets for ART Scenes

This module provides furniture creation utilities adapted from
ART's spring-festival genesis_scene.py for use with genesis-cloud-sim.

Includes:
- Sofa (L-shaped, single-seat)
- Coffee table
- TV stand with TV
- Carpet

References:
    - Original: ART/spring-festival/genesis_scene.py
"""

from typing import Optional, List, Tuple, Dict, Any, Union

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    trimesh = None

# Genesis imports
try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None


# Type aliases
ColorType = Union[Tuple[float, float, float], Tuple[float, float, float, float]]
PositionType = Union[Tuple[float, float, float], List[float]]


def _to_rgba_color(color: ColorType) -> Tuple[int, int, int, int]:
    """Convert float color to integer RGBA."""
    rgb = tuple(int(c * 255) for c in color[:3])
    if len(color) == 4:
        return rgb + (int(color[3] * 255),)
    return rgb + (255,)


class Sofa:
    """Sofa furniture creator."""
    
    # Default colors
    FABRIC_COLOR = (0.96, 0.96, 0.86)  # 米白色
    PILLOW_COLOR = (0.83, 0.69, 0.22)  # 浅金色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
        
    def create_l_shaped_sofa(
        self,
        main_position: PositionType = (0, 0, 7),
        chaise_position: PositionType = (4.5, 0, 5.5),
        single_position: PositionType = (-6, 0, 5),
        fabric_color: Optional[ColorType] = None,
        pillow_color: Optional[ColorType] = None,
        add_pillows: bool = True,
    ) -> List[Any]:
        """Create an L-shaped sofa with main section, chaise lounge, and single seat.
        
        Args:
            main_position: Position of the main (3-seat) sofa section
            chaise_position: Position of the chaise lounge section
            single_position: Position of the single-seat sofa
            fabric_color: Fabric color (default: cream white)
            pillow_color: Pillow color (default: light gold)
            add_pillows: Whether to add decorative pillows
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        fabric = fabric_color or self.FABRIC_COLOR
        pillow = pillow_color or self.PILLOW_COLOR
        
        # Main sofa base (3-seat)
        main_base_mesh = trimesh.creation.box(extents=(5, 0.9, 2.2))
        main_base_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(main_base_mesh.vertices)
        main_base = self.scene.add_entity(
            gs.morphs.Mesh(file=main_base_mesh, pos=(main_position[0], main_position[1] + 0.45, main_position[2])),
            material=gs.materials.Rigid(friction=0.6, density=100),
        )
        parts.append(main_base)
        
        # Main sofa back
        main_back_mesh = trimesh.creation.box(extents=(5, 1.8, 0.4))
        main_back_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(main_back_mesh.vertices)
        main_back = self.scene.add_entity(
            gs.morphs.Mesh(file=main_back_mesh, pos=(main_position[0], main_position[1] + 1.35, main_position[2] + 1)),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(main_back)
        
        # Left arm
        left_arm_mesh = trimesh.creation.box(extents=(0.5, 1.2, 2.2))
        left_arm_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(left_arm_mesh.vertices)
        left_arm = self.scene.add_entity(
            gs.morphs.Mesh(file=left_arm_mesh, pos=(main_position[0] - 2.75, main_position[1] + 1.05, main_position[2])),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(left_arm)
        
        # Right arm
        right_arm_mesh = trimesh.creation.box(extents=(0.5, 1.2, 2.2))
        right_arm_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(right_arm_mesh.vertices)
        right_arm = self.scene.add_entity(
            gs.morphs.Mesh(file=right_arm_mesh, pos=(main_position[0] + 2.75, main_position[1] + 1.05, main_position[2])),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(right_arm)
        
        # Chaise lounge base
        chaise_base_mesh = trimesh.creation.box(extents=(2.2, 0.9, 3))
        chaise_base_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(chaise_base_mesh.vertices)
        chaise_base = self.scene.add_entity(
            gs.morphs.Mesh(file=chaise_base_mesh, pos=(chaise_position[0], chaise_position[1] + 0.45, chaise_position[2])),
            material=gs.materials.Rigid(friction=0.6, density=100),
        )
        parts.append(chaise_base)
        
        # Chaise back
        chaise_back_mesh = trimesh.creation.box(extents=(0.4, 1.8, 3))
        chaise_back_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(chaise_back_mesh.vertices)
        chaise_back = self.scene.add_entity(
            gs.morphs.Mesh(file=chaise_back_mesh, pos=(chaise_position[0] + 1.1, chaise_position[1] + 1.35, chaise_position[2])),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(chaise_back)
        
        # Chaise arm
        chaise_arm_mesh = trimesh.creation.box(extents=(2.2, 1.2, 0.5))
        chaise_arm_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(chaise_arm_mesh.vertices)
        chaise_arm = self.scene.add_entity(
            gs.morphs.Mesh(file=chaise_arm_mesh, pos=(chaise_position[0], chaise_position[1] + 1.05, chaise_position[2] - 1.75)),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(chaise_arm)
        
        # Single sofa base (rotated)
        single_base_mesh = trimesh.creation.box(extents=(2, 0.9, 2.2))
        single_base_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(single_base_mesh.vertices)
        if HAS_NUMPY:
            quat = gs.utils.geom.quat_from_euler("xyz", (0, np.pi/6, 0))
        else:
            quat = (0.9659, 0, 0.2588, 0)  # 30-degree rotation around Y
        single_base = self.scene.add_entity(
            gs.morphs.Mesh(file=single_base_mesh, pos=(single_position[0], single_position[1] + 0.45, single_position[2]), quat=quat),
            material=gs.materials.Rigid(friction=0.6, density=100),
        )
        parts.append(single_base)
        
        # Single sofa back
        single_back_mesh = trimesh.creation.box(extents=(2, 1.8, 0.4))
        single_back_mesh.visual.vertex_colors = list(_to_rgba_color(fabric)) * len(single_back_mesh.vertices)
        if HAS_NUMPY:
            # Calculate rotated position
            offset_x = -0.5 * np.cos(np.pi/6) - 0.8 * np.sin(np.pi/6)
            offset_z = -0.5 * np.sin(np.pi/6) + 0.8 * np.cos(np.pi/6)
        else:
            offset_x, offset_z = -0.83, 0.43
        single_back = self.scene.add_entity(
            gs.morphs.Mesh(file=single_back_mesh, pos=(single_position[0] + offset_x, single_position[1] + 1.35, single_position[2] + offset_z), quat=quat),
            material=gs.materials.Rigid(friction=0.6, density=80),
        )
        parts.append(single_back)
        
        # Pillows
        if add_pillows:
            pillow_positions = [
                (main_position[0] - 1.5, main_position[1] + 1.4, main_position[2] + 0.5),
                (main_position[0], main_position[1] + 1.4, main_position[2] + 0.5),
                (main_position[0] + 1.5, main_position[1] + 1.4, main_position[2] + 0.5),
                (chaise_position[0], chaise_position[1] + 1.4, chaise_position[2] - 1),
                (chaise_position[0], chaise_position[1] + 1.4, chaise_position[2]),
                (chaise_position[0], chaise_position[1] + 1.4, chaise_position[2] + 1),
                (single_position[0], single_position[1] + 1.4, single_position[2] + 0.3),
            ]
            
            for pos in pillow_positions:
                pillow_mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.3))
                pillow_mesh.visual.vertex_colors = list(_to_rgba_color(pillow)) * len(pillow_mesh.vertices)
                pillow_entity = self.scene.add_entity(
                    gs.morphs.Mesh(file=pillow_mesh, pos=pos),
                    material=gs.materials.Rigid(friction=0.7, density=20),
                )
                parts.append(pillow_entity)
        
        return parts


class CoffeeTable:
    """Coffee table furniture creator."""
    
    # Default colors
    WOOD_COLOR = (0.55, 0.27, 0.07)  # 木质色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_table(
        self,
        position: PositionType = (0, 0, 2.5),
        wood_color: Optional[ColorType] = None,
        width: float = 3.6,
        depth: float = 2.1,
        height: float = 1.0,
        with_shelf: bool = True,
    ) -> List[Any]:
        """Create a wooden coffee table with cylindrical legs.
        
        Args:
            position: Table position
            wood_color: Wood color (default: brown)
            width: Table width
            depth: Table depth
            height: Table height
            with_shelf: Whether to add a lower shelf
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        wood = wood_color or self.WOOD_COLOR
        
        # Table top frame
        frame_mesh = trimesh.creation.box(extents=(width, 0.05, depth))
        frame_mesh.visual.vertex_colors = list(_to_rgba_color(wood)) * len(frame_mesh.vertices)
        frame = self.scene.add_entity(
            gs.morphs.Mesh(file=frame_mesh, pos=(position[0], position[1] + height, position[2])),
            material=gs.materials.Rigid(friction=0.5, density=150),
        )
        parts.append(frame)
        
        # Table legs
        leg_positions = [
            (position[0] - width/2 + 0.3, position[1] + height/2, position[2] - depth/2 + 0.3),
            (position[0] + width/2 - 0.3, position[1] + height/2, position[2] - depth/2 + 0.3),
            (position[0] - width/2 + 0.3, position[1] + height/2, position[2] + depth/2 - 0.3),
            (position[0] + width/2 - 0.3, position[1] + height/2, position[2] + depth/2 - 0.3),
        ]
        
        for leg_pos in leg_positions:
            leg_mesh = trimesh.creation.cylinder(radius=0.08, height=height)
            leg_mesh.visual.vertex_colors = list(_to_rgba_color(wood)) * len(leg_mesh.vertices)
            leg = self.scene.add_entity(
                gs.morphs.Mesh(file=leg_mesh, pos=leg_pos),
                material=gs.materials.Rigid(friction=0.5, density=100),
            )
            parts.append(leg)
        
        # Lower shelf
        if with_shelf:
            shelf_mesh = trimesh.creation.box(extents=(width - 0.6, 0.05, depth - 0.6))
            shelf_mesh.visual.vertex_colors = list(_to_rgba_color(wood)) * len(shelf_mesh.vertices)
            shelf = self.scene.add_entity(
                gs.morphs.Mesh(file=shelf_mesh, pos=(position[0], position[1] + 0.3, position[2])),
                material=gs.materials.Rigid(friction=0.5, density=100),
            )
            parts.append(shelf)
        
        return parts


class TVSet:
    """TV stand and television creator."""
    
    # Default colors
    CABINET_COLOR = (0.29, 0.22, 0.16)  # 深棕色
    BLACK_COLOR = (0.07, 0.07, 0.07)  # 黑色
    GOLD_COLOR = (1.0, 0.84, 0.0)  # 金色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_tv_set(
        self,
        position: PositionType = (0, 0, -8),
        cabinet_color: Optional[ColorType] = None,
        screen_color: Optional[ColorType] = None,
        gold_color: Optional[ColorType] = None,
    ) -> List[Any]:
        """Create a TV stand with drawers, handles, TV screen, and stand.
        
        Args:
            position: Position of the TV set
            cabinet_color: Cabinet color (default: dark brown)
            screen_color: Screen color (default: black)
            gold_color: Handle color (default: gold)
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        cabinet = cabinet_color or self.CABINET_COLOR
        black = screen_color or self.BLACK_COLOR
        gold = gold_color or self.GOLD_COLOR
        
        # TV cabinet
        cabinet_mesh = trimesh.creation.box(extents=(6, 1.2, 1.5))
        cabinet_mesh.visual.vertex_colors = list(_to_rgba_color(cabinet)) * len(cabinet_mesh.vertices)
        cabinet_entity = self.scene.add_entity(
            gs.morphs.Mesh(file=cabinet_mesh, pos=(position[0], position[1] + 0.6, position[2])),
            material=gs.materials.Rigid(friction=0.5, density=200),
        )
        parts.append(cabinet_entity)
        
        # Drawers and handles
        for i in range(3):
            # Drawer front
            drawer_color = (0.36, 0.25, 0.20)
            drawer_mesh = trimesh.creation.box(extents=(1.8, 0.4, 0.05))
            drawer_mesh.visual.vertex_colors = list(_to_rgba_color(drawer_color)) * len(drawer_mesh.vertices)
            drawer = self.scene.add_entity(
                gs.morphs.Mesh(file=drawer_mesh, pos=(position[0] - 1.8 + i * 1.8, position[1] + 0.9, position[2] + 0.76)),
                material=gs.materials.Rigid(friction=0.5, density=50),
            )
            parts.append(drawer)
            
            # Handle
            handle_mesh = trimesh.creation.icosphere(radius=0.05)
            handle_mesh.visual.vertex_colors = list(_to_rgba_color(gold)) * len(handle_mesh.vertices)
            handle = self.scene.add_entity(
                gs.morphs.Mesh(file=handle_mesh, pos=(position[0] - 1.8 + i * 1.8, position[1] + 0.9, position[2] + 0.8)),
                material=gs.materials.Rigid(friction=0.3, density=30),
            )
            parts.append(handle)
        
        # TV screen
        screen_mesh = trimesh.creation.box(extents=(5, 3, 0.2))
        screen_mesh.visual.vertex_colors = list(_to_rgba_color(black)) * len(screen_mesh.vertices)
        screen = self.scene.add_entity(
            gs.morphs.Mesh(file=screen_mesh, pos=(position[0], position[1] + 2.8, position[2])),
            material=gs.materials.Rigid(friction=0.3, density=80),
        )
        parts.append(screen)
        
        # TV bezel
        bezel_color = (0.20, 0.20, 0.20)
        bezel_mesh = trimesh.creation.box(extents=(5.2, 3.2, 0.15))
        bezel_mesh.visual.vertex_colors = list(_to_rgba_color(bezel_color)) * len(bezel_mesh.vertices)
        bezel = self.scene.add_entity(
            gs.morphs.Mesh(file=bezel_mesh, pos=(position[0], position[1] + 2.8, position[2] - 0.05)),
            material=gs.materials.Rigid(friction=0.3, density=60),
        )
        parts.append(bezel)
        
        # TV stand
        stand_mesh = trimesh.creation.box(extents=(2, 0.8, 0.5))
        stand_mesh.visual.vertex_colors = list(_to_rgba_color(bezel_color)) * len(stand_mesh.vertices)
        stand = self.scene.add_entity(
            gs.morphs.Mesh(file=stand_mesh, pos=(position[0], position[1] + 1.4, position[2])),
            material=gs.materials.Rigid(friction=0.3, density=100),
        )
        parts.append(stand)
        
        return parts


class Carpet:
    """Carpet/rug creator."""
    
    # Default colors
    CARPET_COLOR = (0.82, 0.71, 0.55)  # 浅棕色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_carpet(
        self,
        position: PositionType = (0, 0, 4),
        color: Optional[ColorType] = None,
        width: float = 8,
        depth: float = 6,
        thickness: float = 0.02,
    ) -> Optional[Any]:
        """Create a carpet/rug on the floor.
        
        Args:
            position: Carpet position
            color: Carpet color (default: light brown)
            width: Carpet width
            depth: Carpet depth
            thickness: Carpet thickness
            
        Returns:
            Created entity or None
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return None
        
        carpet_color = color or self.CARPET_COLOR
        
        carpet_mesh = trimesh.creation.box(extents=(width, thickness, depth))
        carpet_mesh.visual.vertex_colors = list(_to_rgba_color(carpet_color)) * len(carpet_mesh.vertices)
        
        carpet = self.scene.add_entity(
            gs.morphs.Mesh(file=carpet_mesh, pos=(position[0], position[1] + thickness/2, position[2])),
            material=gs.materials.Rigid(friction=0.9, density=10),
        )
        
        return carpet


def create_furniture_set(
    scene: Any,
    sofa_position: PositionType = (0, 0, 7),
    table_position: PositionType = (0, 0, 2.5),
    tv_position: PositionType = (0, 0, -8),
    carpet_position: PositionType = (0, 0, 4),
) -> Dict[str, List[Any]]:
    """Create a complete furniture set (sofa, table, TV, carpet).
    
    Args:
        scene: Genesis scene object
        sofa_position: Sofa position
        table_position: Coffee table position
        tv_position: TV set position
        carpet_position: Carpet position
        
    Returns:
        Dictionary with keys 'sofa', 'table', 'tv', 'carpet'
    """
    furniture = {}
    
    sofa = Sofa(scene)
    furniture['sofa'] = sofa.create_l_shaped_sofa(
        main_position=sofa_position,
        chaise_position=(sofa_position[0] + 4.5, sofa_position[1], sofa_position[2] - 1.5),
        single_position=(sofa_position[0] - 6, sofa_position[1], sofa_position[2] - 2),
    )
    
    table = CoffeeTable(scene)
    furniture['table'] = table.create_table(position=table_position)
    
    tv = TVSet(scene)
    furniture['tv'] = tv.create_tv_set(position=tv_position)
    
    carpet = Carpet(scene)
    carpet_entity = carpet.create_carpet(position=carpet_position)
    furniture['carpet'] = [carpet_entity] if carpet_entity else []
    
    return furniture


__all__ = [
    # Classes
    "Sofa",
    "CoffeeTable",
    "TVSet",
    "Carpet",
    # Functions
    "create_furniture_set",
]
