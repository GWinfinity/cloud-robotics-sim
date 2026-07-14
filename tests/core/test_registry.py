"""Tests for the component registry system."""

import pytest

from cloud_robotics_sim.core.registry import (
    AssetRegistry,
    Registry,
    default_registry,
    register_robot,
    register_scene,
    register_task,
)


class TestRegistry:
    """Tests for the generic Registry class."""

    def test_register_and_create(self):
        """Test registering a factory and creating a component."""
        registry = Registry[int]("test")

        @registry.register("answer")
        def make_answer() -> int:
            return 42

        assert registry.create("answer") == 42

    def test_create_unknown_raises(self):
        """Test creating an unknown component raises KeyError."""
        registry = Registry[str]("test")

        with pytest.raises(KeyError):
            registry.create("missing")

    def test_list_components(self):
        """Test listing registered components."""
        registry = Registry[str]("test")

        @registry.register("a")
        def make_a() -> str:
            return "a"

        @registry.register("b")
        def make_b() -> str:
            return "b"

        assert sorted(registry.list_components()) == ["a", "b"]

    def test_get_metadata(self):
        """Test retrieving component metadata."""
        registry = Registry[str]("test")

        @registry.register("meta", metadata={"key": "value"})
        def make_meta() -> str:
            return "meta"

        assert registry.get_metadata("meta") == {"key": "value"}


class TestAssetRegistry:
    """Tests for the unified AssetRegistry."""

    def test_create_scene(self):
        """Test creating a scene through the asset registry."""
        assets = AssetRegistry()

        @assets.scenes.register("test_scene")
        def make_scene(scene_name: str = "test") -> dict:
            return {"name": scene_name}

        assert assets.create_scene("test_scene", scene_name="custom") == {
            "name": "custom"
        }

    def test_default_registry_singleton(self):
        """Test default registry is a singleton."""
        reg1 = default_registry()
        reg2 = default_registry()
        assert reg1 is reg2


class TestDecorators:
    """Tests for module-level registration decorators."""

    def test_register_scene_decorator(self):
        """Test register_scene decorator."""

        @register_scene("test_scene_decorator")
        def make_scene() -> dict:
            return {"scene": True}

        reg = default_registry()
        assert "test_scene_decorator" in reg.scenes.list_components()

    def test_register_robot_decorator(self):
        """Test register_robot decorator."""

        @register_robot("test_robot_decorator")
        def make_robot() -> dict:
            return {"robot": True}

        reg = default_registry()
        assert "test_robot_decorator" in reg.robots.list_components()

    def test_register_task_decorator(self):
        """Test register_task decorator."""

        @register_task("test_task_decorator")
        def make_task() -> dict:
            return {"task": True}

        reg = default_registry()
        assert "test_task_decorator" in reg.tasks.list_components()
