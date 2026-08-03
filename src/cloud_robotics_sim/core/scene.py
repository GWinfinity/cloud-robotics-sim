"""Scene definition and management.

Scenes define the environment layout independent of specific furniture.
Objects are dynamically injected via ObjectSpawn configurations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cloud_robotics_sim.backend import SceneBackend
from cloud_robotics_sim.backend.types import (
    DeformableConfig,
    DeformableMaterialType,
    LightDescription,
    LightType,
)

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
    scale: tuple[float, float, float] | float = (1.0, 1.0, 1.0)
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
    deformable_config: DeformableConfig | None = None

    def spawn(self, scene: SceneBackend, prefix: str = "") -> Any:
        """Instantiate this object in the given scene backend.

        Args:
            scene: The backend scene to spawn into.
            prefix: Optional prefix for entity naming.

        Returns:
            The created backend entity.

        Raises:
            ValueError: If shape_type is not supported.
        """
        entity_name = f"{prefix}_{self.name}" if prefix else self.name

        backend = scene.backend if hasattr(scene, "backend") else None
        if backend is None:
            raise RuntimeError("Scene backend is not available for spawning objects.")

        # Normalize size to a 3-tuple for backend APIs.
        size: tuple[float, float, float] = tuple(
            float(v) for v in self.size[:3]
        )  # type: ignore[assignment]

        # Normalize scale to a 3-tuple for mesh / deformable assets.
        scale = _normalize_scale(self.scale)

        # Create entity through the backend factory.
        match self.shape_type:
            case "box":
                entity = backend.create_box(
                    size=size,
                    pos=self.position,
                    quat=self.orientation,
                    color=self.color,
                    static=self.static,
                    friction=self.friction,
                    name=entity_name,
                )
            case "sphere":
                entity = backend.create_sphere(
                    radius=self.size[0],
                    pos=self.position,
                    quat=self.orientation,
                    color=self.color,
                    static=self.static,
                    friction=self.friction,
                    name=entity_name,
                )
            case "cylinder":
                entity = backend.create_cylinder(
                    radius=self.size[0],
                    height=self.size[1],
                    pos=self.position,
                    quat=self.orientation,
                    color=self.color,
                    static=self.static,
                    friction=self.friction,
                    name=entity_name,
                )
            case "mesh" if self.mesh_path:
                entity = backend.create_mesh(
                    file=self.mesh_path,
                    pos=self.position,
                    quat=self.orientation,
                    scale=scale,
                    color=self.color,
                    static=self.static,
                    friction=self.friction,
                    name=entity_name,
                )
            case "deformable":
                if self.deformable_config is None:
                    raise ValueError(
                        f"ObjectSpawn '{entity_name}' has shape_type='deformable' "
                        "but no deformable_config"
                    )
                entity = backend.create_deformable(
                    config=self.deformable_config,
                    shape=self._deformable_shape(),
                    size=size,
                    radius=self.size[0] if self.size else None,
                    file=self.mesh_path,
                    scale=scale,
                    pos=self.position,
                    quat=self.orientation,
                    color=self.color,
                    name=entity_name,
                )
            case _:
                raise ValueError(f"Unsupported shape type: {self.shape_type}")

        # Add entity to scene
        scene.add_entity(entity)
        logger.debug(f"Spawned '{entity_name}' at {self.position}")

        return entity

    def _deformable_shape(self) -> str:
        """Infer the underlying primitive shape for a deformable entity."""
        if self.mesh_path:
            return "mesh"
        shape_hint = str(self.properties.get("deformable_shape", "box"))
        if shape_hint in {"box", "sphere", "mesh"}:
            return shape_hint
        return "box"


def _normalize_scale(scale: tuple[float, float, float] | float) -> tuple[float, float, float]:
    """Normalize a uniform or per-axis scale to a 3-tuple."""
    if isinstance(scale, (int, float)):
        return (float(scale), float(scale), float(scale))
    values = [float(v) for v in scale]
    while len(values) < 3:
        values.append(values[-1] if values else 1.0)
    return (values[0], values[1], values[2])


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
    main_light: dict = field(
        default_factory=lambda: {
            "pos": (5.0, -5.0, 8.0),
            "color": (1.0, 0.95, 0.9),
            "intensity": 1.0,
        }
    )
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
        self.room_entities: dict[str, Any] = {}

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

    def build(self, scene: SceneBackend) -> Scene:
        """Build the scene using the provided backend scene.

        Args:
            scene: The backend scene to build into.

        Returns:
            Self for method chaining.
        """
        self.scene = scene
        logger.info(f"Building scene: {self.config.name}")

        self._build_room_structure()
        self._setup_lighting()
        self._build_custom()
        self._spawn_objects()

        logger.info(f"Scene built with {len(self.entities)} objects")
        return self

    def _build_room_structure(self) -> None:
        """Create the room shell (floor and walls)."""
        width, depth, height = self.config.size
        thickness = self.config.wall_thickness

        backend = self.scene.backend if hasattr(self.scene, "backend") else None
        if backend is None:
            raise RuntimeError(
                "Scene backend is not available for building room structure"
            )

        # Floor
        floor = backend.create_box(
            size=(width, depth, thickness),
            pos=(0.0, 0.0, -thickness / 2),
            color=(0.9, 0.9, 0.9, 1.0),
            static=True,
            name="floor",
        )
        self.scene.add_entity(floor)
        self.room_entities["floor"] = floor

        # Walls
        wall_configs = [
            (
                "north",
                (0.0, depth / 2 + thickness / 2, height / 2),
                (width, thickness, height),
            ),
            (
                "south",
                (0.0, -depth / 2 - thickness / 2, height / 2),
                (width, thickness, height),
            ),
            (
                "east",
                (width / 2 + thickness / 2, 0.0, height / 2),
                (thickness, depth, height),
            ),
            (
                "west",
                (-width / 2 - thickness / 2, 0.0, height / 2),
                (thickness, depth, height),
            ),
        ]

        for name, pos, size in wall_configs:
            wall = backend.create_box(
                size=size,
                pos=pos,
                color=(0.95, 0.95, 0.95, 1.0),
                static=True,
                name=f"wall_{name}",
            )
            self.scene.add_entity(wall)
            self.room_entities[f"wall_{name}"] = wall

    def _setup_lighting(self) -> None:
        """Configure scene lighting through the backend."""
        # Ambient light
        self.scene.add_light(
            LightDescription(
                light_type=LightType.AMBIENT,
                color=(1.0, 1.0, 1.0),
                intensity=self.config.ambient_light[0],
            )
        )

        # Main directional light
        main = self.config.main_light
        self.scene.add_light(
            LightDescription(
                light_type=LightType.DIRECTIONAL,
                pos=main["pos"],
                direction=(0.0, 0.3, -1.0),
                color=main["color"],
                intensity=main["intensity"],
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
        return (-w / 2, -d / 2, 0.0, w / 2, d / 2, h)

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
    def deformable_soft_cube(
        name: str = "soft_cube",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: float = 0.08,
        color: tuple[float, float, float, float] = (0.2, 0.7, 0.3, 1.0),
        deformable_config: DeformableConfig | None = None,
    ) -> ObjectSpawn:
        """A deformable soft cube for grasping experiments.

        Args:
            name: Object identifier.
            position: Initial position.
            size: Cube side length in meters.
            color: RGBA color tuple.
            deformable_config: Optional deformable material configuration.
                Defaults to a soft FEM elastic material.
        """
        config = deformable_config or DeformableConfig(
            material=DeformableMaterialType.FEM_ELASTIC,
            youngs_modulus=1.0e4,
            poisson_ratio=0.45,
            density=1000.0,
            resolution_level=2,
        )
        return ObjectSpawn(
            name=name,
            shape_type="deformable",
            size=(size, size, size),
            position=position,
            static=False,
            mass=0.1,
            color=color,
            tags=["graspable", "soft_body", "manipulable"],
            deformable_config=config,
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
