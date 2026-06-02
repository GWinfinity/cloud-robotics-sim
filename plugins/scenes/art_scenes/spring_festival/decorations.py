# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""
Chinese New Year (Spring Festival) Decorations

This module provides Chinese New Year decoration creators adapted from
ART's spring-festival genesis_scene.py for use with genesis-cloud-sim.

Includes:
- Lanterns (灯笼)
- Fu character (福字 - "Fu" means good fortune)
- Chinese knot (中国结)

References:
    - Original: ART/spring-festival/genesis_scene.py
"""

from typing import Any, Dict, Optional, List, Tuple, Union

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


class Lantern:
    """Chinese lantern (灯笼) creator."""
    
    # Traditional colors
    RED = (1.0, 0.0, 0.0)  # 红色
    GOLD = (1.0, 0.84, 0.0)  # 金色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_lantern(
        self,
        position: PositionType,
        scale: float = 1.0,
        red_color: Optional[ColorType] = None,
        gold_color: Optional[ColorType] = None,
    ) -> List[Any]:
        """Create a traditional Chinese lantern.
        
        The lantern consists of:
        - Main body (red sphere, scaled to oval)
        - Top cap (gold cylinder)
        - Bottom cap (gold cylinder)
        - Tassel (red cylinder, tapered)
        
        Args:
            position: Position (x, y, z) - center of the lantern body
            scale: Scale factor for the lantern size
            red_color: Red color (default: bright red)
            gold_color: Gold color (default: gold)
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        red = red_color or self.RED
        gold = gold_color or self.GOLD
        
        x, y, z = position
        
        # Lantern body (sphere scaled to oval)
        sphere_mesh = trimesh.creation.icosphere(radius=1.0 * scale, subdivisions=2)
        sphere_mesh.apply_scale([1, 1.3, 1])  # Scale to make it oval
        sphere_mesh.visual.vertex_colors = list(_to_rgba_color(red)) * len(sphere_mesh.vertices)
        
        lantern_body = self.scene.add_entity(
            gs.morphs.Mesh(file=sphere_mesh, pos=(x, y, z)),
            material=gs.materials.Rigid(friction=0.4, density=5),
        )
        parts.append(lantern_body)
        
        # Top cap (gold cylinder)
        top_cap_mesh = trimesh.creation.cylinder(radius=0.3 * scale, height=0.3 * scale)
        top_cap_mesh.visual.vertex_colors = list(_to_rgba_color(gold)) * len(top_cap_mesh.vertices)
        top_cap = self.scene.add_entity(
            gs.morphs.Mesh(file=top_cap_mesh, pos=(x, y + 1.2 * scale, z)),
            material=gs.materials.Rigid(friction=0.3, density=10),
        )
        parts.append(top_cap)
        
        # Bottom cap (gold cylinder)
        bottom_cap_mesh = trimesh.creation.cylinder(radius=0.3 * scale, height=0.3 * scale)
        bottom_cap_mesh.visual.vertex_colors = list(_to_rgba_color(gold)) * len(bottom_cap_mesh.vertices)
        bottom_cap = self.scene.add_entity(
            gs.morphs.Mesh(file=bottom_cap_mesh, pos=(x, y - 1.2 * scale, z)),
            material=gs.materials.Rigid(friction=0.3, density=10),
        )
        parts.append(bottom_cap)
        
        # Tassel (red tapered cylinder)
        tassel_mesh = trimesh.creation.cylinder(
            radius=0.05 * scale,
            height=1.5 * scale,
        )
        # Apply tapering by scaling bottom vertices
        if HAS_NUMPY:
            vertices = tassel_mesh.vertices.copy()
            bottom_mask = vertices[:, 1] < 0  # Y-axis is up in Genesis
            vertices[bottom_mask, 0] *= 3  # Scale X
            vertices[bottom_mask, 2] *= 3  # Scale Z
            tassel_mesh.vertices = vertices
        
        tassel_mesh.visual.vertex_colors = list(_to_rgba_color(red)) * len(tassel_mesh.vertices)
        tassel = self.scene.add_entity(
            gs.morphs.Mesh(file=tassel_mesh, pos=(x, y - 2.2 * scale, z)),
            material=gs.materials.Rigid(friction=0.5, density=2),
        )
        parts.append(tassel)
        
        return parts
    
    def create_lantern_string(
        self,
        start_position: PositionType,
        count: int = 4,
        spacing: float = 4.0,
        scale: float = 1.0,
    ) -> List[Any]:
        """Create a string of lanterns in a row.
        
        Args:
            start_position: Starting position of the first lantern
            count: Number of lanterns
            spacing: Spacing between lanterns
            scale: Scale factor for all lanterns
            
        Returns:
            List of all created entities
        """
        all_parts = []
        x, y, z = start_position
        
        for i in range(count):
            lantern_parts = self.create_lantern(
                position=(x + i * spacing, y, z),
                scale=scale,
            )
            all_parts.extend(lantern_parts)
        
        return all_parts


class FuCharacter:
    """Fu character (福字) creator - "Fu" means good fortune/blessing.
    
    Traditionally hung upside down (倒福) as a pun:
    "Fu dao" (福倒) sounds like "Fu dao" (福到) meaning "Fortune arrives".
    """
    
    RED = (1.0, 0.0, 0.0)  # 红色
    GOLD = (1.0, 0.84, 0.0)  # 金色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_fu(
        self,
        position: PositionType,
        size: float = 1.5,
        thickness: float = 0.05,
        upside_down: bool = True,
        red_color: Optional[ColorType] = None,
    ) -> List[Any]:
        """Create a Fu character decoration.
        
        Args:
            position: Position of the Fu character
            size: Size of the diamond shape
            thickness: Thickness of the plaque
            upside_down: Whether to hang upside down (traditional)
            red_color: Red color (default: bright red)
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        red = red_color or self.RED
        
        # Create diamond shape (box rotated 45 degrees)
        diamond_mesh = trimesh.creation.box(extents=(size, size, thickness))
        
        # Rotate 45 degrees around Z axis
        if HAS_NUMPY:
            angle = np.pi / 4
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            diamond_mesh.apply_transform(rotation_matrix)
        
        # Apply upside down rotation if requested
        if upside_down:
            if HAS_NUMPY:
                # Rotate 180 degrees around X (or Z) to flip upside down
                flip_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                diamond_mesh.apply_transform(flip_matrix)
        
        diamond_mesh.visual.vertex_colors = list(_to_rgba_color(red)) * len(diamond_mesh.vertices)
        
        # Calculate quaternion for Genesis
        if upside_down and HAS_NUMPY and gs is not None:
            quat = gs.utils.geom.quat_from_euler("xyz", (0, np.pi, np.pi))
        else:
            quat = None
        
        diamond = self.scene.add_entity(
            gs.morphs.Mesh(
                file=diamond_mesh,
                pos=position,
                quat=quat,
            ),
            material=gs.materials.Rigid(friction=0.3, density=1),
        )
        parts.append(diamond)
        
        return parts


class ChineseKnot:
    """Chinese knot (中国结) creator.
    
    A traditional decorative knot symbolizing good luck and prosperity.
    """
    
    RED = (1.0, 0.0, 0.0)  # 红色
    GOLD = (1.0, 0.84, 0.0)  # 金色
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
    
    def create_knot(
        self,
        position: PositionType,
        scale: float = 1.0,
        red_color: Optional[ColorType] = None,
        gold_color: Optional[ColorType] = None,
    ) -> List[Any]:
        """Create a Chinese knot decoration.
        
        The knot consists of:
        - Knot body (red torus)
        - Tassel (red tapered cylinder)
        - Gold ball (gold icosphere)
        
        Args:
            position: Position of the knot center
            scale: Scale factor
            red_color: Red color (default: bright red)
            gold_color: Gold color (default: gold)
            
        Returns:
            List of created entities
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return []
        
        parts = []
        red = red_color or self.RED
        gold = gold_color or self.GOLD
        
        x, y, z = position
        
        # Knot body (torus)
        knot_mesh = trimesh.creation.torus(
            r_major=0.5 * scale,
            r_minor=0.15 * scale,
            sections_major=64,
            sections_minor=16
        )
        knot_mesh.visual.vertex_colors = list(_to_rgba_color(red)) * len(knot_mesh.vertices)
        
        knot = self.scene.add_entity(
            gs.morphs.Mesh(file=knot_mesh, pos=(x, y, z)),
            material=gs.materials.Rigid(friction=0.4, density=3),
        )
        parts.append(knot)
        
        # Tassel (tapered cylinder)
        tassel_mesh = trimesh.creation.cylinder(
            radius=0.1 * scale,
            height=1.5 * scale,
        )
        # Apply tapering
        if HAS_NUMPY:
            vertices = tassel_mesh.vertices.copy()
            bottom_mask = vertices[:, 1] < (y - 0.75 * scale)
            # Actually for tassel we want it wider at bottom
            # So scale the bottom vertices outward
            relative_y = vertices[:, 1] - (y - 0.75 * scale)
            scale_factor = 1 + 2 * (relative_y / (1.5 * scale))  # Linear taper
            vertices[:, 0] *= scale_factor
            vertices[:, 2] *= scale_factor
            tassel_mesh.vertices = vertices
        
        tassel_mesh.visual.vertex_colors = list(_to_rgba_color(red)) * len(tassel_mesh.vertices)
        tassel = self.scene.add_entity(
            gs.morphs.Mesh(file=tassel_mesh, pos=(x, y - 1.5 * scale, z)),
            material=gs.materials.Rigid(friction=0.5, density=2),
        )
        parts.append(tassel)
        
        # Gold ball decoration
        gold_ball_mesh = trimesh.creation.icosphere(radius=0.2 * scale, subdivisions=2)
        gold_ball_mesh.visual.vertex_colors = list(_to_rgba_color(gold)) * len(gold_ball_mesh.vertices)
        gold_ball = self.scene.add_entity(
            gs.morphs.Mesh(file=gold_ball_mesh, pos=(x, y - 0.8 * scale, z)),
            material=gs.materials.Rigid(friction=0.3, density=15),
        )
        parts.append(gold_ball)
        
        return parts


class SpringFestivalDecorations:
    """Complete Spring Festival decoration set."""
    
    def __init__(self, scene: Any):
        """
        Args:
            scene: Genesis scene object.
        """
        self.scene = scene
        self.lantern = Lantern(scene)
        self.fu = FuCharacter(scene)
        self.knot = ChineseKnot(scene)
    
    def create_full_decorations(
        self,
        room_width: float = 20.0,
        room_depth: float = 20.0,
        room_height: float = 10.0,
    ) -> Dict[str, List[Any]]:
        """Create a full set of Spring Festival decorations for a room.
        
        Args:
            room_width: Room width
            room_depth: Room depth
            room_height: Room height
            
        Returns:
            Dictionary with keys 'lanterns', 'fu', 'knots'
        """
        decorations = {}
        
        # Lanterns - hang from ceiling in a row
        lantern_y = room_height - 2.0  # Slightly below ceiling
        lanterns = self.lantern.create_lantern_string(
            start_position=(-room_width/2 + 4, lantern_y, 0),
            count=4,
            spacing=4.0,
            scale=1.0,
        )
        decorations['lanterns'] = lanterns
        
        # Fu character - on the back wall, upside down
        fu_position = (0, room_height/2, -room_depth/2 + 0.2)
        fu_parts = self.fu.create_fu(
            position=fu_position,
            size=1.5,
            upside_down=True,  # Traditional: upside down means "Fortune arrives"
        )
        decorations['fu'] = fu_parts
        
        # Chinese knot - also on the back wall, below the Fu
        knot_position = (0, room_height/2 - 2, -room_depth/2 + 0.3)
        knot_parts = self.knot.create_knot(
            position=knot_position,
            scale=1.0,
        )
        decorations['knots'] = knot_parts
        
        return decorations


__all__ = [
    # Classes
    "Lantern",
    "FuCharacter",
    "ChineseKnot",
    "SpringFestivalDecorations",
]
