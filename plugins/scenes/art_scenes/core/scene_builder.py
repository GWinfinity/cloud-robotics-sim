# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""
Scene Building Utilities for ART Scenes

This module provides generic scene building utilities adapted from
ART's spring-festival genesis_scene.py for use with genesis-cloud-sim.

Includes:
- Room structure creation (walls, floors, ceilings)
- Lighting setup
- Material utilities
- Scene composition helpers

References:
    - Original: ART/spring-festival/genesis_scene.py
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

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
    from genesis.engine.scene import Scene
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None
    Scene = None


# Type aliases
ColorType = Union[Tuple[float, float, float], Tuple[float, float, float, float]]
PositionType = Union[Tuple[float, float, float], List[float]]
SizeType = Union[Tuple[float, float, float], List[float]]


@dataclass
class RoomConfig:
    """Configuration for room creation."""
    width: float = 20.0
    depth: float = 20.0
    height: float = 10.0
    wall_color: ColorType = (0.55, 0.27, 0.07)  # 暖棕色
    floor_color: ColorType = (0.9, 0.9, 0.9)  # 浅灰色
    ceiling_color: ColorType = (1.0, 1.0, 1.0)  # 白色
    wall_thickness: float = 0.2
    with_baseboard: bool = True
    baseboard_color: ColorType = (1.0, 1.0, 1.0)  # 白色
    baseboard_height: float = 0.3


@dataclass
class LightingConfig:
    """Configuration for scene lighting."""
    ambient_intensity: float = 0.4
    ambient_color: ColorType = (1.0, 1.0, 1.0)
    
    directional_intensity: float = 1.0
    directional_color: ColorType = (1.0, 0.84, 0.0)  # 暖黄色
    directional_pos: PositionType = (10, 20, 10)
    directional_dir: PositionType = (0, -1, 0)
    
    point_lights: Optional[List[Dict[str, Any]]] = None


class SceneBuilder:
    """
    Generic scene builder for creating room structures and environments.
    
    Provides methods to create walls, floors, lighting, and other scene elements
    using Genesis physics simulator.
    """
    
    def __init__(self, scene: Optional[Any] = None):
        """
        Args:
            scene: Genesis scene object. If None, creates a new scene.
        """
        self.scene = scene
        self.created_entities = []
        
    def set_scene(self, scene: Any):
        """Set the Genesis scene to build into."""
        self.scene = scene
        
    def create_box_from_trimesh(
        self,
        pos: PositionType,
        size: SizeType,
        color: ColorType,
        fixed: bool = True,
        material_props: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """Create a box entity using trimesh.
        
        Args:
            pos: Position (x, y, z)
            size: Size (width, height, depth)
            color: RGB or RGBA color tuple
            fixed: Whether the entity is fixed in place
            material_props: Additional material properties
            **kwargs: Additional arguments for gs.morphs.Mesh
            
        Returns:
            Created entity or None if dependencies not available
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return None
            
        # Create box mesh
        mesh = trimesh.creation.box(extents=size)
        
        # Add visual colors
        rgb_color = tuple(int(c * 255) for c in color[:3])
        if len(color) == 4:
            rgb_color = rgb_color + (int(color[3] * 255),)
        else:
            rgb_color = rgb_color + (255,)
        mesh.visual.vertex_colors = list(rgb_color) * len(mesh.vertices)
        
        # Create material
        material_kwargs = {"friction": 0.5}
        if material_props:
            material_kwargs.update(material_props)
        
        # Add entity to scene
        entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file=mesh,
                pos=pos,
                fixed=fixed,
                **kwargs
            ),
            material=gs.materials.Rigid(**material_kwargs),
        )
        
        self.created_entities.append(entity)
        return entity
    
    def create_cylinder_from_trimesh(
        self,
        pos: PositionType,
        radius: float,
        height: float,
        color: ColorType,
        radius_top: Optional[float] = None,
        fixed: bool = True,
        material_props: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """Create a cylinder entity using trimesh.
        
        Args:
            pos: Position (x, y, z)
            radius: Cylinder radius (or bottom radius if tapered)
            height: Cylinder height
            color: RGB or RGBA color tuple
            radius_top: Top radius for tapered cylinders (None for constant radius)
            fixed: Whether the entity is fixed in place
            material_props: Additional material properties
            **kwargs: Additional arguments for gs.morphs.Mesh
            
        Returns:
            Created entity or None
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return None
            
        # Create cylinder mesh
        if radius_top is not None and radius_top != radius:
            # Tapered cylinder (cone-like)
            mesh = trimesh.creation.cylinder(
                radius=radius,
                height=height,
                sections=32
            )
            # Apply tapering by scaling top vertices
            vertices = mesh.vertices.copy()
            top_mask = vertices[:, 2] > 0
            scale = radius_top / radius if radius > 0 else 1.0
            vertices[top_mask, 0] *= scale
            vertices[top_mask, 1] *= scale
            mesh.vertices = vertices
        else:
            mesh = trimesh.creation.cylinder(
                radius=radius,
                height=height,
                sections=32
            )
        
        # Add visual colors
        rgb_color = tuple(int(c * 255) for c in color[:3])
        if len(color) == 4:
            rgb_color = rgb_color + (int(color[3] * 255),)
        else:
            rgb_color = rgb_color + (255,)
        mesh.visual.vertex_colors = list(rgb_color) * len(mesh.vertices)
        
        # Create material
        material_kwargs = {"friction": 0.5}
        if material_props:
            material_kwargs.update(material_props)
        
        # Add entity to scene
        entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file=mesh,
                pos=pos,
                fixed=fixed,
                **kwargs
            ),
            material=gs.materials.Rigid(**material_kwargs),
        )
        
        self.created_entities.append(entity)
        return entity
    
    def create_sphere_from_trimesh(
        self,
        pos: PositionType,
        radius: float,
        color: ColorType,
        subdivisions: int = 2,
        fixed: bool = True,
        scale: Optional[Tuple[float, float, float]] = None,
        material_props: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """Create a sphere entity using trimesh.
        
        Args:
            pos: Position (x, y, z)
            radius: Sphere radius
            color: RGB or RGBA color tuple
            subdivisions: Icosphere subdivisions (higher = smoother)
            fixed: Whether the entity is fixed in place
            scale: Optional scale factors [sx, sy, sz] to deform sphere
            material_props: Additional material properties
            **kwargs: Additional arguments for gs.morphs.Mesh
            
        Returns:
            Created entity or None
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return None
            
        # Create sphere mesh
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
        
        # Apply scale if provided
        if scale is not None:
            mesh.apply_scale(scale)
        
        # Add visual colors
        rgb_color = tuple(int(c * 255) for c in color[:3])
        if len(color) == 4:
            rgb_color = rgb_color + (int(color[3] * 255),)
        else:
            rgb_color = rgb_color + (255,)
        mesh.visual.vertex_colors = list(rgb_color) * len(mesh.vertices)
        
        # Create material
        material_kwargs = {"friction": 0.5}
        if material_props:
            material_kwargs.update(material_props)
        
        # Add entity to scene
        entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file=mesh,
                pos=pos,
                fixed=fixed,
                **kwargs
            ),
            material=gs.materials.Rigid(**material_kwargs),
        )
        
        self.created_entities.append(entity)
        return entity
    
    def create_torus_from_trimesh(
        self,
        pos: PositionType,
        r_major: float,
        r_minor: float,
        color: ColorType,
        sections_major: int = 64,
        sections_minor: int = 16,
        fixed: bool = True,
        material_props: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """Create a torus (ring) entity using trimesh.
        
        Args:
            pos: Position (x, y, z)
            r_major: Major radius (distance from center to tube center)
            r_minor: Minor radius (tube radius)
            color: RGB or RGBA color tuple
            sections_major: Number of sections around the ring
            sections_minor: Number of sections around the tube
            fixed: Whether the entity is fixed in place
            material_props: Additional material properties
            **kwargs: Additional arguments for gs.morphs.Mesh
            
        Returns:
            Created entity or None
        """
        if not HAS_TRIMESH or not HAS_GENESIS or self.scene is None:
            return None
            
        # Create torus mesh
        mesh = trimesh.creation.torus(
            r_major=r_major,
            r_minor=r_minor,
            sections_major=sections_major,
            sections_minor=sections_minor
        )
        
        # Add visual colors
        rgb_color = tuple(int(c * 255) for c in color[:3])
        if len(color) == 4:
            rgb_color = rgb_color + (int(color[3] * 255),)
        else:
            rgb_color = rgb_color + (255,)
        mesh.visual.vertex_colors = list(rgb_color) * len(mesh.vertices)
        
        # Create material
        material_kwargs = {"friction": 0.5}
        if material_props:
            material_kwargs.update(material_props)
        
        # Add entity to scene
        entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file=mesh,
                pos=pos,
                fixed=fixed,
                **kwargs
            ),
            material=gs.materials.Rigid(**material_kwargs),
        )
        
        self.created_entities.append(entity)
        return entity
    
    def create_room(self, config: Optional[RoomConfig] = None) -> List[Any]:
        """Create a complete room structure.
        
        Args:
            config: Room configuration. Uses defaults if None.
            
        Returns:
            List of created entities
        """
        if config is None:
            config = RoomConfig()
            
        entities = []
        
        # Floor
        floor = self.create_box_from_trimesh(
            pos=(0, 0, 0),
            size=(config.width, 0.1, config.depth),
            color=config.floor_color,
            fixed=True,
            material_props={"friction": 0.8},
        )
        if floor:
            entities.append(floor)
        
        # Back wall
        back_wall = self.create_box_from_trimesh(
            pos=(0, config.height/2, -config.depth/2),
            size=(config.width, config.height, config.wall_thickness),
            color=config.wall_color,
            fixed=True,
        )
        if back_wall:
            entities.append(back_wall)
        
        # Left wall
        left_wall = self.create_box_from_trimesh(
            pos=(-config.width/2, config.height/2, 0),
            size=(config.wall_thickness, config.height, config.depth),
            color=config.wall_color,
            fixed=True,
        )
        if left_wall:
            entities.append(left_wall)
        
        # Right wall
        right_wall = self.create_box_from_trimesh(
            pos=(config.width/2, config.height/2, 0),
            size=(config.wall_thickness, config.height, config.depth),
            color=config.wall_color,
            fixed=True,
        )
        if right_wall:
            entities.append(right_wall)
        
        # Ceiling
        ceiling = self.create_box_from_trimesh(
            pos=(0, config.height, 0),
            size=(config.width, config.wall_thickness, config.depth),
            color=config.ceiling_color,
            fixed=True,
        )
        if ceiling:
            entities.append(ceiling)
        
        # Baseboards
        if config.with_baseboard:
            # Back wall baseboard
            back_bb = self.create_box_from_trimesh(
                pos=(0, config.baseboard_height/2, -config.depth/2 + config.wall_thickness/2),
                size=(config.width, config.baseboard_height, 0.05),
                color=config.baseboard_color,
                fixed=True,
            )
            if back_bb:
                entities.append(back_bb)
            
            # Left wall baseboard
            left_bb = self.create_box_from_trimesh(
                pos=(-config.width/2 + config.wall_thickness/2, config.baseboard_height/2, 0),
                size=(0.05, config.baseboard_height, config.depth),
                color=config.baseboard_color,
                fixed=True,
            )
            if left_bb:
                entities.append(left_bb)
            
            # Right wall baseboard
            right_bb = self.create_box_from_trimesh(
                pos=(config.width/2 - config.wall_thickness/2, config.baseboard_height/2, 0),
                size=(0.05, config.baseboard_height, config.depth),
                color=config.baseboard_color,
                fixed=True,
            )
            if right_bb:
                entities.append(right_bb)
        
        return entities
    
    def setup_lighting(self, config: Optional[LightingConfig] = None):
        """Setup scene lighting.
        
        Args:
            config: Lighting configuration. Uses defaults if None.
        """
        if not HAS_GENESIS or self.scene is None:
            return
            
        if config is None:
            config = LightingConfig()
        
        # Ambient light
        self.scene.add_light(
            gs.lights.AmbientLight(
                color=config.ambient_color,
                intensity=config.ambient_intensity,
            )
        )
        
        # Directional light
        self.scene.add_light(
            gs.lights.DirectionalLight(
                color=config.directional_color,
                intensity=config.directional_intensity,
                pos=config.directional_pos,
                dir=config.directional_dir,
            )
        )
        
        # Point lights
        if config.point_lights:
            for light_config in config.point_lights:
                self.scene.add_light(
                    gs.lights.PointLight(
                        color=light_config.get("color", (1.0, 1.0, 1.0)),
                        intensity=light_config.get("intensity", 0.8),
                        pos=light_config.get("pos", (0, 5, 0)),
                    )
                )


def create_scene_with_room(
    room_config: Optional[RoomConfig] = None,
    lighting_config: Optional[LightingConfig] = None,
    viewer_options: Optional[Dict[str, Any]] = None,
    sim_options: Optional[Dict[str, Any]] = None,
    show_viewer: bool = True,
) -> Optional[Tuple[Any, SceneBuilder]]:
    """Convenience function to create a scene with a complete room.
    
    Args:
        room_config: Room configuration
        lighting_config: Lighting configuration
        viewer_options: Genesis viewer options
        sim_options: Genesis simulation options
        show_viewer: Whether to show the viewer
        
    Returns:
        Tuple of (scene, builder) or None if Genesis not available
    """
    if not HAS_GENESIS:
        return None
    
    # Default options
    if viewer_options is None:
        viewer_options = {
            "camera_pos": (0, 12, 15),
            "camera_lookat": (0, 0, 0),
            "camera_up": (0, 1, 0),
        }
    
    if sim_options is None:
        sim_options = {
            "dt": 0.01,
            "substeps": 10,
        }
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(**viewer_options),
        sim_options=gs.options.SimOptions(**sim_options),
        show_viewer=show_viewer,
    )
    
    # Create builder
    builder = SceneBuilder(scene)
    
    # Add floor plane
    scene.add_entity(
        gs.morphs.Plane(
            pos=(0, 0, 0),
            normal=(0, 1, 0),
        ),
        material=gs.materials.Rigid(friction=0.8),
    )
    
    # Create room structure
    builder.create_room(room_config)
    
    # Setup lighting
    builder.setup_lighting(lighting_config)
    
    return scene, builder


__all__ = [
    # Configuration
    "RoomConfig",
    "LightingConfig",
    # Builder
    "SceneBuilder",
    "create_scene_with_room",
]
