# ART Scenes Plugin for genesis-cloud-sim
#
# This plugin provides artistic and culturally-themed scene presets
# for Genesis physics simulations.
#
# Based on ART/spring-festival project.

__version__ = "0.1.0"
__plugin_name__ = "art_scenes"
__plugin_type__ = "scene"

# Core exports
from .core.scene_builder import (
    SceneBuilder,
    RoomConfig,
    LightingConfig,
    create_scene_with_room,
)

# Furniture exports
from .assets.furniture import (
    Sofa,
    CoffeeTable,
    TVSet,
    Carpet,
    create_furniture_set,
)

# Spring Festival exports
from .spring_festival.scene import (
    SpringFestivalScene,
    create_spring_festival_scene,
)
from .spring_festival.decorations import (
    Lantern,
    FuCharacter,
    ChineseKnot,
    SpringFestivalDecorations,
)

__all__ = [
    # Core
    "SceneBuilder",
    "RoomConfig",
    "LightingConfig",
    "create_scene_with_room",
    # Furniture
    "Sofa",
    "CoffeeTable",
    "TVSet",
    "Carpet",
    "create_furniture_set",
    # Spring Festival
    "SpringFestivalScene",
    "create_spring_festival_scene",
    "Lantern",
    "FuCharacter",
    "ChineseKnot",
    "SpringFestivalDecorations",
]
