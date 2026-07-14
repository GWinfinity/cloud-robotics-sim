"""Tests for scene definitions and management."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cloud_robotics_sim import ObjectLibrary, ObjectSpawn, SceneConfig
from cloud_robotics_sim.core.scenes import EmptyRoom, Kitchen, LivingRoom, Office


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

    def test_spawn_box_mocked(self, monkeypatch):
        """Test ObjectSpawn.spawn for box shape."""
        scene = MagicMock()
        entity = MagicMock()
        scene.add_entity.return_value = entity

        gs = SimpleNamespace(
            morphs=SimpleNamespace(
                Box=MagicMock(),
                Sphere=MagicMock(),
                Cylinder=MagicMock(),
                Mesh=MagicMock(),
            ),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        spawn = ObjectSpawn(name="box", shape_type="box", size=(1.0, 2.0, 3.0))
        result = spawn.spawn(scene, prefix="room")

        assert result is entity
        gs.morphs.Box.assert_called_once_with(
            size=(1.0, 2.0, 3.0),
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        gs.surfaces.Default.assert_called_once_with(
            color=(0.8, 0.8, 0.8, 1.0),
            roughness=0.8,
        )

    def test_spawn_sphere(self, monkeypatch):
        """Test ObjectSpawn.spawn for sphere shape."""
        scene = MagicMock()
        scene.add_entity.return_value = MagicMock()

        gs = SimpleNamespace(
            morphs=SimpleNamespace(
                Box=MagicMock(),
                Sphere=MagicMock(),
                Cylinder=MagicMock(),
                Mesh=MagicMock(),
            ),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        spawn = ObjectSpawn(name="sphere", shape_type="sphere", size=(0.5,))
        spawn.spawn(scene)

        gs.morphs.Sphere.assert_called_once_with(
            radius=0.5,
            pos=(0.0, 0.0, 0.0),
        )

    def test_spawn_cylinder(self, monkeypatch):
        """Test ObjectSpawn.spawn for cylinder shape."""
        scene = MagicMock()
        scene.add_entity.return_value = MagicMock()

        gs = SimpleNamespace(
            morphs=SimpleNamespace(
                Box=MagicMock(),
                Sphere=MagicMock(),
                Cylinder=MagicMock(),
                Mesh=MagicMock(),
            ),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        spawn = ObjectSpawn(
            name="cylinder",
            shape_type="cylinder",
            size=(0.3, 1.0),
            position=(1.0, 1.0, 0.5),
        )
        spawn.spawn(scene)

        gs.morphs.Cylinder.assert_called_once_with(
            radius=0.3,
            height=1.0,
            pos=(1.0, 1.0, 0.5),
            quat=(1.0, 0.0, 0.0, 0.0),
        )

    def test_spawn_mesh(self, monkeypatch):
        """Test ObjectSpawn.spawn for mesh shape."""
        scene = MagicMock()
        scene.add_entity.return_value = MagicMock()

        gs = SimpleNamespace(
            morphs=SimpleNamespace(
                Box=MagicMock(),
                Sphere=MagicMock(),
                Cylinder=MagicMock(),
                Mesh=MagicMock(),
            ),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        spawn = ObjectSpawn(
            name="mesh_obj",
            shape_type="mesh",
            size=(1.0, 1.0, 1.0),
            mesh_path="path/to/mesh.obj",
        )
        spawn.spawn(scene)

        gs.morphs.Mesh.assert_called_once_with(
            file="path/to/mesh.obj",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )

    def test_spawn_unsupported_shape(self):
        """Test ObjectSpawn.spawn raises ValueError for unsupported shapes."""
        spawn = ObjectSpawn(name="weird", shape_type="torus")
        with pytest.raises(ValueError, match="Unsupported shape type"):
            spawn.spawn(MagicMock())


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

    def test_build(self, monkeypatch):
        """Test Scene.build orchestrates construction."""
        scene = EmptyRoom()
        scene._build_room_structure = MagicMock()
        scene._setup_lighting = MagicMock()
        scene._build_custom = MagicMock()
        scene._spawn_objects = MagicMock()

        gs_scene = MagicMock()
        result = scene.build(gs_scene)

        assert result is scene
        assert scene.scene is gs_scene
        scene._build_room_structure.assert_called_once()
        scene._setup_lighting.assert_called_once()
        scene._build_custom.assert_called_once()
        scene._spawn_objects.assert_called_once()

    def test_build_room_structure(self, monkeypatch):
        """Test room structure creation."""
        scene = EmptyRoom(size=(4.0, 6.0, 3.0))
        gs_scene = MagicMock()
        scene.scene = gs_scene

        gs = SimpleNamespace(
            morphs=SimpleNamespace(Box=MagicMock()),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        scene._build_room_structure()

        assert "floor" in scene.room_entities
        assert "wall_north" in scene.room_entities
        assert "wall_south" in scene.room_entities
        assert "wall_east" in scene.room_entities
        assert "wall_west" in scene.room_entities
        assert gs_scene.add_entity.call_count == 5

    def test_setup_lighting_with_gs_lights(self, monkeypatch):
        """Test lighting setup when gs.lights is available."""
        scene = EmptyRoom()
        gs_scene = MagicMock()
        scene.scene = gs_scene

        lights = SimpleNamespace(
            Ambient=MagicMock(),
            Directional=MagicMock(),
        )
        monkeypatch.setattr(
            "cloud_robotics_sim.core.scene.get_genesis_lights",
            MagicMock(return_value=lights),
        )

        scene._setup_lighting()

        assert gs_scene.add_light.call_count == 2

    def test_setup_lighting_without_gs_lights(self, monkeypatch):
        """Test lighting setup skips when gs.lights unavailable."""
        scene = EmptyRoom()
        gs_scene = MagicMock()
        scene.scene = gs_scene

        monkeypatch.setattr(
            "cloud_robotics_sim.core.scene.get_genesis_lights",
            MagicMock(return_value=None),
        )

        scene._setup_lighting()

        gs_scene.add_light.assert_not_called()

    def test_setup_lighting_attribute_error(self, monkeypatch):
        """Test lighting setup handles AttributeError."""
        scene = EmptyRoom()
        gs_scene = MagicMock()
        scene.scene = gs_scene

        lights = SimpleNamespace(
            Ambient=MagicMock(side_effect=AttributeError("no ambient")),
        )
        monkeypatch.setattr(
            "cloud_robotics_sim.core.scene.get_genesis_lights",
            MagicMock(return_value=lights),
        )

        scene._setup_lighting()  # should not raise

    def test_spawn_objects(self, monkeypatch):
        """Test spawning configured objects."""
        scene = EmptyRoom()
        cube = ObjectLibrary.graspable_cube(name="cube")
        scene.add_object(cube)

        gs = SimpleNamespace(
            morphs=SimpleNamespace(
                Box=MagicMock(),
                Sphere=MagicMock(),
                Cylinder=MagicMock(),
                Mesh=MagicMock(),
            ),
            surfaces=SimpleNamespace(Default=MagicMock()),
        )
        monkeypatch.setattr("cloud_robotics_sim.core.scene.gs", gs)

        scene.scene = MagicMock()
        scene._spawn_objects()

        assert "cube" in scene.entities

    def test_spawn_objects_failure(self, monkeypatch):
        """Test failed object spawn is logged without raising."""
        scene = EmptyRoom()
        bad = ObjectSpawn(name="bad", shape_type="torus")
        scene.add_object(bad)

        scene.scene = MagicMock()
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
