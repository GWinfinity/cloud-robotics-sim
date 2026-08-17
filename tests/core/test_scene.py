"""Tests for scene definitions and management."""

from unittest.mock import MagicMock

import pytest

from cloud_robotics_sim import ObjectLibrary, ObjectSpawn, SceneConfig
from cloud_robotics_sim.backend import LightDescription, LightType
from cloud_robotics_sim.core.scenes import EmptyRoom, Kitchen, LivingRoom, Office
from tests.conftest import make_mock_scene_backend


def _make_mock_scene_backend() -> MagicMock:
    """Create a mock SceneBackend with a mock simulator backend."""
    return make_mock_scene_backend()


class TestSceneConfig:
    """Tests for SceneConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SceneConfig()

        assert config.name == "unnamed_scene"
        assert config.size == (10.0, 10.0, 3.0)
        assert config.wall_thickness == 0.2
        assert config.ambient_light == (0.3, 0.3, 0.3)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SceneConfig(
            name="test_scene",
            size=(5.0, 5.0, 3.0),
            ambient_light=(0.5, 0.5, 0.5),
        )

        assert config.name == "test_scene"
        assert config.size == (5.0, 5.0, 3.0)
        assert config.ambient_light == (0.5, 0.5, 0.5)


class TestObjectSpawn:
    """Tests for ObjectSpawn dataclass and spawn method."""

    def test_default_values(self):
        """Test default spawn values."""
        spawn = ObjectSpawn(name="test_obj")

        assert spawn.shape_type == "box"
        assert spawn.position == (0.0, 0.0, 0.0)
        assert spawn.static is True

    def test_spawn_box(self):
        """Test creating a box spawn."""
        spawn = ObjectLibrary.graspable_cube(
            name="red_cube",
            position=(1.0, 0.0, 0.5),
            color=(0.9, 0.2, 0.2, 1.0),
        )

        assert spawn.name == "red_cube"
        assert spawn.shape_type == "box"
        assert spawn.position == (1.0, 0.0, 0.5)
        assert spawn.static is False

    def test_spawn_box_mocked(self):
        """Test ObjectSpawn.spawn for box shape."""
        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_box.return_value = entity

        spawn = ObjectSpawn(name="box", shape_type="box", size=(1.0, 2.0, 3.0))
        result = spawn.spawn(scene_backend, prefix="room")

        assert result is entity
        scene_backend.backend.create_box.assert_called_once_with(
            size=(1.0, 2.0, 3.0),
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            color=(0.8, 0.8, 0.8, 1.0),
            static=True,
            friction=0.5,
            name="room_box",
        )
        scene_backend.add_entity.assert_called_once_with(entity)

    def test_spawn_sphere(self):
        """Test ObjectSpawn.spawn for sphere shape."""
        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_sphere.return_value = entity

        spawn = ObjectSpawn(name="sphere", shape_type="sphere", size=(0.5,))
        spawn.spawn(scene_backend)

        scene_backend.backend.create_sphere.assert_called_once_with(
            radius=0.5,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            color=(0.8, 0.8, 0.8, 1.0),
            static=True,
            friction=0.5,
            name="sphere",
        )

    def test_spawn_cylinder(self):
        """Test ObjectSpawn.spawn for cylinder shape."""
        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_cylinder.return_value = entity

        spawn = ObjectSpawn(
            name="cylinder",
            shape_type="cylinder",
            size=(0.3, 1.0),
            position=(1.0, 1.0, 0.5),
        )
        spawn.spawn(scene_backend)

        scene_backend.backend.create_cylinder.assert_called_once_with(
            radius=0.3,
            height=1.0,
            pos=(1.0, 1.0, 0.5),
            quat=(1.0, 0.0, 0.0, 0.0),
            color=(0.8, 0.8, 0.8, 1.0),
            static=True,
            friction=0.5,
            name="cylinder",
        )

    def test_spawn_mesh(self):
        """Test ObjectSpawn.spawn for mesh shape."""
        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_mesh.return_value = entity

        spawn = ObjectSpawn(
            name="mesh_obj",
            shape_type="mesh",
            scale=(2.0, 2.0, 2.0),
            mesh_path="path/to/mesh.obj",
        )
        spawn.spawn(scene_backend)

        scene_backend.backend.create_mesh.assert_called_once_with(
            file="path/to/mesh.obj",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            scale=(2.0, 2.0, 2.0),
            color=(0.8, 0.8, 0.8, 1.0),
            static=True,
            friction=0.5,
            material="default",
            name="mesh_obj",
        )

    def test_spawn_mesh_with_material_hint(self):
        """Test ObjectSpawn.spawn forwards material hint for mesh shape."""
        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_mesh.return_value = entity

        spawn = ObjectSpawn(
            name="wood_chair",
            shape_type="mesh",
            mesh_path="path/to/chair.obj",
            material="wood",
        )
        spawn.spawn(scene_backend)

        scene_backend.backend.create_mesh.assert_called_once_with(
            file="path/to/chair.obj",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.8, 0.8, 0.8, 1.0),
            static=True,
            friction=0.5,
            material="wood",
            name="wood_chair",
        )

    def test_spawn_unsupported_shape(self):
        """Test ObjectSpawn.spawn raises ValueError for unsupported shapes."""
        spawn = ObjectSpawn(name="weird", shape_type="torus")
        with pytest.raises(ValueError, match="Unsupported shape type"):
            spawn.spawn(_make_mock_scene_backend())


class TestSceneManagement:
    """Tests for Scene object management."""

    def test_add_object(self):
        """Test adding a single object."""
        scene = EmptyRoom()
        obj = ObjectLibrary.coffee_table()
        scene.add_object(obj)

        assert obj in scene.object_spawns
        assert scene.get_objects_by_tag("table") == [obj]

    def test_add_objects(self):
        """Test adding multiple objects."""
        scene = EmptyRoom()
        sofa = ObjectLibrary.sofa_three_seat()
        table = ObjectLibrary.coffee_table()
        scene.add_objects([sofa, table])

        assert len(scene.object_spawns) == 2
        assert len(scene.get_objects_by_tag("furniture")) == 2

    def test_get_objects_by_tag_no_match(self):
        """Test querying tags with no matches."""
        scene = EmptyRoom()
        assert scene.get_objects_by_tag("nonexistent") == []

    def test_build(self):
        """Test Scene.build orchestrates construction."""
        scene = EmptyRoom()
        scene._build_room_structure = MagicMock()
        scene._setup_lighting = MagicMock()
        scene._build_custom = MagicMock()
        scene._spawn_objects = MagicMock()

        scene_backend = _make_mock_scene_backend()
        result = scene.build(scene_backend)

        assert result is scene
        assert scene.scene is scene_backend
        scene._build_room_structure.assert_called_once()
        scene._setup_lighting.assert_called_once()
        scene._build_custom.assert_called_once()
        scene._spawn_objects.assert_called_once()

    def test_build_room_structure(self):
        """Test room structure creation."""
        scene = EmptyRoom(size=(4.0, 6.0, 3.0))
        scene_backend = _make_mock_scene_backend()
        scene.scene = scene_backend

        scene._build_room_structure()

        assert "floor" in scene.room_entities
        assert "wall_north" in scene.room_entities
        assert "wall_south" in scene.room_entities
        assert "wall_east" in scene.room_entities
        assert "wall_west" in scene.room_entities
        assert scene_backend.backend.create_box.call_count == 5
        assert scene_backend.add_entity.call_count == 5

    def test_setup_lighting(self):
        """Test lighting setup dispatches LightDescription objects."""
        scene = EmptyRoom()
        scene_backend = _make_mock_scene_backend()
        scene.scene = scene_backend

        scene._setup_lighting()

        assert scene_backend.add_light.call_count == 2
        calls = scene_backend.add_light.call_args_list
        assert isinstance(calls[0].args[0], LightDescription)
        assert calls[0].args[0].light_type == LightType.AMBIENT
        assert isinstance(calls[1].args[0], LightDescription)
        assert calls[1].args[0].light_type == LightType.DIRECTIONAL

    def test_spawn_objects(self):
        """Test spawning configured objects."""
        scene = EmptyRoom()
        cube = ObjectLibrary.graspable_cube(name="cube")
        scene.add_object(cube)

        scene_backend = _make_mock_scene_backend()
        entity = MagicMock()
        scene_backend.backend.create_box.return_value = entity
        scene.scene = scene_backend
        scene._spawn_objects()

        assert "cube" in scene.entities
        scene_backend.add_entity.assert_called_once_with(entity)

    def test_spawn_objects_failure(self):
        """Test failed object spawn is logged without raising."""
        scene = EmptyRoom()
        bad = ObjectSpawn(name="bad", shape_type="torus")
        scene.add_object(bad)

        scene.scene = _make_mock_scene_backend()
        scene._spawn_objects()  # should not raise

        assert "bad" not in scene.entities

    def test_get_spawn_positions(self):
        """Test default spawn positions."""
        scene = EmptyRoom()
        positions = scene.get_spawn_positions()
        assert len(positions) > 0
        assert all(len(p) == 3 for p in positions)

    def test_get_bounds(self):
        """Test scene bounding box."""
        scene = EmptyRoom(size=(4.0, 6.0, 3.0))
        bounds = scene.get_bounds()
        assert bounds == (-2.0, -3.0, 0.0, 2.0, 3.0, 3.0)

    def test_reset(self):
        """Test scene reset handles dynamic objects."""
        scene = EmptyRoom()
        cube = ObjectLibrary.graspable_cube(name="cube")
        scene.add_object(cube)
        scene.entities["cube"] = MagicMock()

        scene.reset()  # no-op placeholder


class TestObjectLibrary:
    """Tests for ObjectLibrary factory methods."""

    def test_sofa_three_seat(self):
        """Test sofa factory."""
        sofa = ObjectLibrary.sofa_three_seat(position=(2.0, 1.0, 0.0))
        assert sofa.name == "sofa_three_seat"
        assert sofa.shape_type == "box"
        assert sofa.static is True
        assert "furniture" in sofa.tags

    def test_coffee_table(self):
        """Test coffee table factory."""
        table = ObjectLibrary.coffee_table(position=(1.0, 0.5, 0.0))
        assert table.name == "coffee_table"
        assert table.static is True
        assert "table" in table.tags

    def test_graspable_cube(self):
        """Test graspable cube factory."""
        cube = ObjectLibrary.graspable_cube(
            name="red_block",
            position=(1.5, 0.5, 0.5),
            color=(0.9, 0.2, 0.2, 1.0),
            mass=0.2,
        )
        assert cube.name == "red_block"
        assert cube.static is False
        assert cube.mass == 0.2
        assert "graspable" in cube.tags

    def test_refrigerator(self):
        """Test refrigerator factory."""
        fridge = ObjectLibrary.refrigerator(position=(0.0, 2.0, 0.0))
        assert fridge.name == "refrigerator"
        assert fridge.shape_type == "box"
        assert "appliance" in fridge.tags

    def test_bed_double(self):
        """Test bed factory."""
        bed = ObjectLibrary.bed_double(position=(0.0, 0.0, 0.0))
        assert bed.name == "bed_double"
        assert bed.static is True
        assert "bedroom" in bed.tags

    def test_obstacle_box(self):
        """Test obstacle box factory."""
        obstacle = ObjectLibrary.obstacle_box(
            position=(1.0, 1.0, 0.0),
            size=(0.6, 0.6, 0.6),
        )
        assert obstacle.name == "obstacle"
        assert obstacle.static is True
        assert obstacle.size == (0.6, 0.6, 0.6)
        assert "obstacle" in obstacle.tags


class TestPredefinedScenes:
    """Tests for predefined scene subclasses."""

    def test_empty_room(self):
        """Test EmptyRoom creation."""
        scene = EmptyRoom(size=(4.0, 4.0, 2.5))
        assert scene.config.name == "empty_room"
        assert scene.config.size == (4.0, 4.0, 2.5)

    def test_living_room(self):
        """Test LivingRoom creation."""
        scene = LivingRoom()
        scene._build_custom()
        assert scene.config.name == "living_room"
        assert len(scene.object_spawns) == 2

    def test_kitchen(self):
        """Test Kitchen creation."""
        scene = Kitchen()
        scene._build_custom()
        assert scene.config.name == "kitchen"
        assert len(scene.object_spawns) == 1

    def test_office(self):
        """Test Office creation."""
        scene = Office()
        assert scene.config.name == "office"
        assert len(scene.object_spawns) == 0
