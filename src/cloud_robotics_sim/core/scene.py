"""Scene definition and management.

Scenes define the environment layout independent of specific furniture.
Objects are dynamically injected via ObjectSpawn configurations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import genesis as gs
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectSpawn:
    """Configuration for spawning an object in the scene.
    
    ObjectSpawn provides a reusable, data-driven way to define objects
    that can be instantiated across different scenes.
    
    Attributes:
        name: Unique identifier for the object.
        shape_type: Geometry type ('box', 'sphere', 'cylinder', 'mesh').
        size: Dimensions (interpretation depends on shape_type).
        mesh_path: Path to mesh file (required if shape_type='mesh').
        position: Initial position (x, y, z).
        orientation: Initial orientation as quaternion (w, x, y, z).
        mass: Mass in kilograms (for dynamic objects).
        static: If True, object is immovable (furniture).
        friction: Surface friction coefficient.
        material: Material identifier for rendering.
        color: RGBA color tuple.
        tags: Categorical tags for querying.
        properties: Additional custom properties.
        
    Example:
        >>> table = ObjectSpawn(
        ...     name="coffee_table",
        ...     shape_type="box",
        ...     size=(1.2, 0.6, 0.5),
        ...     position=(2.0, 1.0, 0.25),
        ...     static=True,
        ...     tags=["furniture", "table"]
        ... )
    """
    name: str
    shape_type: str = "box"
    size: tuple[float, ...] = (1.0, 1.0, 1.0)
    mesh_path: str | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 1.0
    static: bool = True
    friction: float = 0.5
    material: str = "default"
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    tags: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def spawn(self, scene: Any, prefix: str = "") -> Any:
        """Instantiate this object in the given scene.
        
        Args:
            scene: The Genesis scene to spawn into.
            prefix: Optional prefix for entity naming.
            
        Returns:
            The created Genesis entity.
            
        Raises:
            ValueError: If shape_type is not supported.
        """
        entity_name = f"{prefix}_{self.name}" if prefix else self.name
        
        # Create appropriate morph based on shape type
        match self.shape_type:
            case "box":
                morph = gs.morphs.Box(
                    size=self.size,
                    pos=self.position,
                    quat=self.orientation,
                )
            case "sphere":
                morph = gs.morphs.Sphere(
                    radius=self.size[0],
                    pos=self.position,
                )
            case "cylinder":
                morph = gs.morphs.Cylinder(
                    radius=self.size[0],
                    height=self.size[1],
                    pos=self.position,
                    quat=self.orientation,
                )
            case "mesh" if self.mesh_path:
                morph = gs.morphs.Mesh(
                    file=self.mesh_path,
                    pos=self.position,
                    quat=self.orientation,
                    scale=self.size,
                )
            case _:
                raise ValueError(f"Unsupported shape type: {self.shape_type}")
        
        # Create surface material
        surface = gs.surfaces.Default(
            color=self.color,
            roughness=0.8,
        )
        
        # Spawn entity
        entity = scene.add_entity(morph=morph, surface=surface)
        logger.debug(f"Spawned '{entity_name}' at {self.position}")
        
        return entity


@dataclass
class SceneConfig:
    """Configuration for scene geometry and appearance.
    
    Attributes:
        name: Scene identifier.
        size: Room dimensions (width, depth, height) in meters.
        wall_thickness: Thickness of wall geometry.
        floor_material: Floor material identifier.
        wall_material: Wall material identifier.
        ambient_light: Ambient light intensity (RGB).
        main_light: Configuration for primary directional light.
        default_camera_pos: Default camera position.
        default_camera_lookat: Default camera look-at point.
    """
    name: str = "unnamed_scene"
    size: tuple[float, float, float] = (10.0, 10.0, 3.0)
    wall_thickness: float = 0.2
    floor_material: str = "wood"
    wall_material: str = "paint_white"
    ambient_light: tuple[float, float, float] = (0.3, 0.3, 0.3)
    main_light: dict = field(default_factory=lambda: {
        'pos': (5.0, -5.0, 8.0),
        'color': (1.0, 0.95, 0.9),
        'intensity': 1.0,
    })
    default_camera_pos: tuple[float, float, float] = (5.0, 5.0, 5.0)
    default_camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Scene(ABC):
    """Abstract base class for simulation scenes.
    
    Scenes define the environment structure (room geometry, lighting)
    and contain dynamically spawned objects. They do not embed specific
    furniture, allowing for flexible object configuration.
    
    To create a custom scene:
        1. Subclass Scene
        2. Implement _build_custom() for scene-specific setup
        3. Optionally override get_spawn_positions()
    
    Attributes:
        config: Scene configuration.
        object_spawns: List of objects to spawn.
        entities: Dictionary of spawned entities by name.
        room_entities: Dictionary of room structure entities.
    """

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()
        self.scene: Any = None
        
        self.object_spawns: list[ObjectSpawn] = []
        self.entities: dict[str, Any] = {}
        self.room_entities: dict[str, gs.Entity] = {}
        
        # Tag-based object indexing
        self._tag_index: dict[str, list[str]] = {}

    def add_object(self, spawn: ObjectSpawn) -> Scene:
        """Add an object to the scene configuration.
        
        Args:
            spawn: ObjectSpawn configuration.
            
        Returns:
            Self for method chaining.
        """
        self.object_spawns.append(spawn)
        
        for tag in spawn.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(spawn.name)
        
        return self

    def add_objects(self, spawns: list[ObjectSpawn]) -> Scene:
        """Add multiple objects to the scene."""
        for spawn in spawns:
            self.add_object(spawn)
        return self

    def get_objects_by_tag(self, tag: str) -> list[ObjectSpawn]:
        """Retrieve objects by their tag.
        
        Args:
            tag: The tag to search for.
            
        Returns:
            List of matching ObjectSpawn configurations.
        """
        names = self._tag_index.get(tag, [])
        return [s for s in self.object_spawns if s.name in names]

    def build(self, gs_scene: gs.Scene) -> Scene:
        """Build the scene in Genesis.
        
        Args:
            gs_scene: The Genesis scene to build into.
            
        Returns:
            Self for method chaining.
        """
        self.scene = gs_scene
        logger.info(f"Building scene: {self.config.name}")
        
        self._build_room_structure()
        self._setup_lighting()
        self._spawn_objects()
        self._build_custom()
        
        logger.info(f"Scene built with {len(self.entities)} objects")
        return self

    def _build_room_structure(self) -> None:
        """Create the room shell (floor and walls)."""
        width, depth, height = self.config.size
        thickness = self.config.wall_thickness
        
        # Floor
        floor = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(width, depth, thickness),
                pos=(0.0, 0.0, -thickness / 2),
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
        )
        self.room_entities['floor'] = floor
        
        # Walls
        wall_configs = [
            ('north', (0.0, depth/2 + thickness/2, height/2), (width, thickness, height)),
            ('south', (0.0, -depth/2 - thickness/2, height/2), (width, thickness, height)),
            ('east', (width/2 + thickness/2, 0.0, height/2), (thickness, depth, height)),
            ('west', (-width/2 - thickness/2, 0.0, height/2), (thickness, depth, height)),
        ]
        
        for name, pos, size in wall_configs:
            wall = self.scene.add_entity(
                morph=gs.morphs.Box(size=size, pos=pos),
                surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0)),
            )
            self.room_entities[f'wall_{name}'] = wall

    def _setup_lighting(self) -> None:
        """Configure scene lighting."""
        # Ambient light
        self.scene.add_light(
            gs.lights.Ambient(
                color=(1.0, 1.0, 1.0),
                intensity=self.config.ambient_light[0],
            )
        )
        
        # Main directional light
        main = self.config.main_light
        self.scene.add_light(
            gs.lights.Directional(
                pos=main['pos'],
                direction=(0.0, 0.3, -1.0),
                color=main['color'],
                intensity=main['intensity'],
                cast_shadow=True,
            )
        )

    def _spawn_objects(self) -> None:
        """Instantiate all configured objects."""
        for spawn in self.object_spawns:
            try:
                entity = spawn.spawn(self.scene, prefix=self.config.name)
                self.entities[spawn.name] = entity
            except Exception as e:
                logger.error(f"Failed to spawn '{spawn.name}': {e}")

    @abstractmethod
    def _build_custom(self) -> None:
        """Override for scene-specific setup."""
        pass

    def get_spawn_positions(self) -> list[tuple[float, float, float]]:
        """Get valid robot spawn positions.
        
        Returns:
            List of (x, y, z) positions near the scene center.
        """
        return [
            (0.0, 0.0, 0.1),
            (1.0, 0.0, 0.1),
            (-1.0, 0.0, 0.1),
            (0.0, 1.0, 0.1),
            (0.0, -1.0, 0.1),
        ]

    def get_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Get scene bounding box.
        
        Returns:
            (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        w, d, h = self.config.size
        return (-w/2, -d/2, 0.0, w/2, d/2, h)

    def reset(self) -> None:
        """Reset the scene state (e.g., dynamic object positions)."""
        for spawn in self.object_spawns:
            if not spawn.static and spawn.name in self.entities:
                # Reset dynamic objects
                pass  # Implementation depends on Genesis API


class ObjectLibrary:
    """Library of pre-defined, reusable objects.
    
    This class provides factory methods for common furniture and
    interactive objects used in robotics simulation.
    
    Example:
        >>> scene.add_object(ObjectLibrary.coffee_table(position=(2, 1, 0)))
        >>> scene.add_object(ObjectLibrary.graspable_cube(
        ...     name="red_block",
        ...     position=(1.5, 0.5, 0.5),
        ...     color=(0.9, 0.2, 0.2, 1.0)
        ... ))
    """

    @staticmethod
    def sofa_three_seat(
        position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> ObjectSpawn:
        """A standard three-seat sofa."""
        return ObjectSpawn(
            name="sofa_three_seat",
            shape_type="box",
            size=(2.2, 0.9, 0.8),
            position=position,
            static=True,
            material="fabric",
            color=(0.6, 0.5, 0.4, 1.0),
            tags=["furniture", "seating", "living_room"],
        )

    @staticmethod
    def coffee_table(
        position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> ObjectSpawn:
        """A rectangular coffee table."""
        return ObjectSpawn(
            name="coffee_table",
            shape_type="box",
            size=(1.2, 0.6, 0.5),
            position=position,
            static=True,
            material="wood",
            color=(0.7, 0.5, 0.3, 1.0),
            tags=["furniture", "table", "living_room"],
        )

    @staticmethod
    def graspable_cube(
        name: str = "cube",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: float = 0.05,
        color: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        mass: float = 0.1,
    ) -> ObjectSpawn:
        """A graspable cube (dynamic object).
        
        Args:
            name: Object identifier.
            position: Initial position.
            size: Cube side length in meters.
            color: RGBA color tuple.
            mass: Mass in kilograms.
        """
        return ObjectSpawn(
            name=name,
            shape_type="box",
            size=(size, size, size),
            position=position,
            static=False,
            mass=mass,
            color=color,
            tags=["graspable", "cube", "manipulable"],
        )

    @staticmethod
    def refrigerator(
        position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> ObjectSpawn:
        """A standard refrigerator."""
        return ObjectSpawn(
            name="refrigerator",
            shape_type="box",
            size=(0.8, 0.8, 1.8),
            position=position,
            static=True,
            color=(0.9, 0.9, 0.95, 1.0),
            tags=["furniture", "appliance", "kitchen", "articulated"],
        )

    @staticmethod
    def bed_double(
        position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> ObjectSpawn:
        """A double-size bed."""
        return ObjectSpawn(
            name="bed_double",
            shape_type="box",
            size=(2.0, 1.5, 0.5),
            position=position,
            static=True,
            color=(0.8, 0.8, 0.9, 1.0),
            tags=["furniture", "bedroom"],
        )

    @staticmethod
    def obstacle_box(
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> ObjectSpawn:
        """A static obstacle box."""
        return ObjectSpawn(
            name="obstacle",
            shape_type="box",
            size=size,
            position=position,
            static=True,
            color=(0.5, 0.5, 0.5, 1.0),
            tags=["obstacle", "static"],
        )
