"""Tests for ART Scenes Plugin."""

import numpy as np
import pytest

from plugins.scenes.art_scenes.core.scene_builder import SceneBuilder, RoomConfig, LightingConfig
from plugins.scenes.art_scenes.assets.furniture import Sofa, CoffeeTable, create_furniture_set
from plugins.scenes.art_scenes.spring_festival.scene import SpringFestivalScene
from plugins.scenes.art_scenes.spring_festival.decorations import Lantern, FuCharacter


class TestRoomConfig:
    """Room configuration tests."""

    def test_init(self):
        """Config initializes."""
        cfg = RoomConfig(width=10.0, height=3.0, depth=8.0)
        assert cfg.width == 10.0


class TestLightingConfig:
    """Lighting config tests."""

    def test_init(self):
        """Config initializes."""
        cfg = LightingConfig()
        assert cfg is not None


class TestSceneBuilder:
    """Scene builder tests."""

    def test_init(self):
        """Builder initializes."""
        builder = SceneBuilder()
        assert builder is not None

    def test_set_scene(self):
        """Can set scene."""
        builder = SceneBuilder()
        builder.set_scene("dummy")
        assert builder.scene == "dummy"


class TestFurniture:
    """Furniture asset tests."""

    def test_sofa(self):
        """Sofa initializes."""
        sofa = Sofa(scene="dummy")
        assert sofa is not None

    def test_coffee_table(self):
        """Coffee table initializes."""
        table = CoffeeTable(scene="dummy")
        assert table is not None

    def test_furniture_set(self):
        """Furniture set creation."""
        try:
            items = create_furniture_set(scene="dummy")
            assert len(items) >= 2
        except AttributeError:
            pytest.skip("Requires Genesis scene object")


class TestSpringFestivalScene:
    """Spring festival scene tests."""

    def test_init(self):
        """Scene initializes."""
        scene = SpringFestivalScene()
        assert scene is not None

    def test_has_methods(self):
        """Scene has expected methods."""
        scene = SpringFestivalScene()
        assert hasattr(scene, 'add_decorations')


class TestDecorations:
    """Decoration tests."""

    def test_lantern(self):
        """Lantern initializes."""
        lantern = Lantern(scene="dummy")
        assert lantern is not None

    def test_fu_character(self):
        """Fu character initializes."""
        fu = FuCharacter(scene="dummy")
        assert fu is not None
