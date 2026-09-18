# WFC Chinese Home Scenes Plugin for genesis-cloud-sim
#
# 波函数坍缩（WFC）驱动的中式家居场景自动生成：
# 整理好的中国生活方式规则 + 程序化布局 + 家具填充。

__version__ = "0.1.0"
__plugin_name__ = "wfc_scenes"
__plugin_type__ = "scene"

from .assets.furniture_cn import (
    Prim,
    PrimBuilder,
    TrimeshBackend,
    build_tile_prims,
)
from .core.rules_cn import (
    TILES,
    Layout,
    TileSpec,
    build_variants,
    compatible,
    exterior_ok,
    validate_layout,
)
from .core.wfc import (
    TileVariant,
    WFCContradictionError,
    collapse,
    expand_variants,
    rotated_sockets,
)
from .scene import (
    ChineseHomeScene,
    build_structure_prims,
    load_layout,
    populate_genesis_scene,
)

__all__ = [
    # Rules
    "TILES",
    "TileSpec",
    "Layout",
    "build_variants",
    "compatible",
    "exterior_ok",
    "validate_layout",
    # WFC
    "TileVariant",
    "WFCContradictionError",
    "collapse",
    "expand_variants",
    "rotated_sockets",
    # Furniture
    "Prim",
    "PrimBuilder",
    "TrimeshBackend",
    "build_tile_prims",
    # Scene
    "ChineseHomeScene",
    "build_structure_prims",
    "load_layout",
    "populate_genesis_scene",
]
