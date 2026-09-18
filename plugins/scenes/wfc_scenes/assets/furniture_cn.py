# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Chinese-home furniture as data-driven primitive specs.

Every piece of furniture is described as a list of :class:`Prim` (boxes,
cylinders, spheres) in tile-local coordinates (N = +y). Builders write into a
:class:`PrimBuilder` that applies the tile's position and WFC rotation and
carries the cell's **actual half extents** ``(hx, hy)`` — wall-anchored
furniture is placed relative to the walls (inner face at ``±(h - 0.06)``) and
a few pieces (kitchen counters, carpets, corridor runner) adapt their length
to the span, so variable row/column sizes (方案 2) work out of the box.

Two backends consume prims:

- :class:`TrimeshBackend` — pure trimesh, no Genesis required. Powers GLB
  export and all offline tests.
- :class:`GenesisBackend` — spawns one rigid entity per prim into a live
  ``gs.Scene`` (genesis-world >= 1.4 primitive morphs with surfaces).

The style follows the everyday mainland-Chinese apartment: 米白布艺沙发 +
实木茶几 + 电视墙 living room, 圆餐桌 dining, U 型台面 + 灶台 + 冰箱 kitchen,
双人床 + 衣柜 bedroom, 鞋柜 + 换鞋凳 entryway, 洗衣机 + 晾衣架 balcony, 茶台 +
坐墩 tea room, plus 字画/绿植/鱼缸 accents. Spring Festival decorations
(灯笼/福字) are opt-in via ``festival=True``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

try:
    import trimesh
    from trimesh.transformations import euler_matrix, quaternion_from_euler

    HAS_TRIMESH = True
except ImportError:  # pragma: no cover - trimesh is a hard dep of genesis
    trimesh = None
    euler_matrix = None
    quaternion_from_euler = None
    HAS_TRIMESH = False

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    gs = None
    HAS_GENESIS = False


# ---------------------------------------------------------------------------
# Palette (中国家常配色)
# ---------------------------------------------------------------------------

WOOD_DARK = (0.29, 0.22, 0.16)  # 深胡桃木
WOOD = (0.55, 0.35, 0.18)  # 榆木/橡木
WOOD_LIGHT = (0.72, 0.55, 0.35)  # 浅色木地板家具
FABRIC = (0.93, 0.91, 0.86)  # 米白布艺
FABRIC_GREY = (0.62, 0.63, 0.65)  # 灰布艺
RED = (0.75, 0.12, 0.10)  # 中国红
GOLD = (0.85, 0.68, 0.25)  # 金色五金/描边
BLACK = (0.08, 0.08, 0.08)
WHITE = (0.92, 0.92, 0.92)
STEEL = (0.75, 0.77, 0.80)
MARBLE = (0.86, 0.86, 0.84)  # 石材台面
MATTRESS = (0.90, 0.89, 0.86)
QUILT = (0.63, 0.68, 0.74)  # 灰蓝被面
GLASS = (0.65, 0.80, 0.85, 0.45)  # 玻璃（鱼缸/镜面）
WATER = (0.25, 0.45, 0.60, 0.70)
GREEN = (0.22, 0.46, 0.19)  # 绿植
POT = (0.58, 0.32, 0.22)  # 红陶花盆
PORCELAIN = (0.94, 0.94, 0.96)  # 白瓷
CARPET_BEIGE = (0.80, 0.70, 0.55)
CARPET_RED = (0.52, 0.24, 0.18)  # 玄关地垫/红地毯

ColorType = Sequence[float]

# 墙内面到墙中心线的距离（隔墙厚 0.12，中心线在格子边界上）。
WALL_INSET = 0.06


# ---------------------------------------------------------------------------
# Primitive spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prim:
    """One primitive mesh in world coordinates (already tile-rotated)."""

    kind: str  # "box" | "cylinder" | "sphere"
    size: tuple[
        float, ...
    ]  # box: (x,y,z); cylinder: (radius, height); sphere: (radius,)
    pos: tuple[float, float, float]  # world center
    color: ColorType
    name: str = "prim"
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # XYZ degrees
    rho: float = 300.0  # kg/m^3 (Genesis 1.4 material API)
    friction: float = 0.5
    fixed: bool = False


class PrimBuilder:
    """Accumulates prims for one tile, applying the tile pose to local coords.

    Local frame: origin at the tile center, +y = tile north (the direction of
    the tile's local N socket), +x = tile east. ``rot`` is the clockwise WFC
    rotation in degrees. ``half`` is the cell's actual half extent
    ``(hx, hy)`` — furniture anchors to the walls via ``pb.hx``/``pb.hy``.
    """

    def __init__(
        self,
        origin: tuple[float, float] = (0.0, 0.0),
        rot: float = 0.0,
        half: tuple[float, float] = (1.6, 1.6),
    ):
        self.origin = origin
        self.rot = rot
        self.hx = half[0]
        self.hy = half[1]
        self.primljs: list[Prim] = []
        self._counter = 0

    # -- placement ----------------------------------------------------------

    def _to_world(self, lx: float, ly: float) -> tuple[float, float]:
        # 顺时针旋转（与 rotated_sockets 的插座约定一致：rot=90 时局部 +y/N
        # 转到世界 +x/E）。
        rad = math.radians(self.rot)
        c, s = math.cos(rad), math.sin(rad)
        return (
            self.origin[0] + lx * c + ly * s,
            self.origin[1] - lx * s + ly * c,
        )

    def _next_name(self, name: str) -> str:
        self._counter += 1
        return f"{name}_{self._counter}" if name else f"prim_{self._counter}"

    # -- primitive adders ----------------------------------------------------

    def box(
        self,
        size: tuple[float, float, float],
        lpos: tuple[float, float, float],
        color: ColorType,
        name: str = "",
        rho: float = 300.0,
        friction: float = 0.5,
        fixed: bool = False,
        spin: float = 0.0,
    ) -> Prim:
        """Add a box; ``lpos`` is tile-local, ``spin`` an extra Z rotation."""
        wx, wy = self._to_world(lpos[0], lpos[1])
        prim = Prim(
            kind="box",
            size=tuple(size),
            pos=(wx, wy, lpos[2]),
            color=tuple(color),
            name=self._next_name(name or "box"),
            euler=(0.0, 0.0, self.rot + spin),
            rho=rho,
            friction=friction,
            fixed=fixed,
        )
        self.primljs.append(prim)
        return prim

    def cyl(
        self,
        radius: float,
        height: float,
        lpos: tuple[float, float, float],
        color: ColorType,
        name: str = "",
        rho: float = 300.0,
        friction: float = 0.5,
        fixed: bool = False,
        axis: str = "z",
    ) -> Prim:
        """Add a cylinder; ``axis`` is the cylinder axis in tile-local frame."""
        wx, wy = self._to_world(lpos[0], lpos[1])
        euler = {
            "z": (0.0, 0.0, self.rot),
            "x": (0.0, 90.0, self.rot),
            "y": (90.0, 0.0, self.rot),
        }[axis]
        prim = Prim(
            kind="cylinder",
            size=(radius, height),
            pos=(wx, wy, lpos[2]),
            color=tuple(color),
            name=self._next_name(name or "cyl"),
            euler=euler,
            rho=rho,
            friction=friction,
            fixed=fixed,
        )
        self.primljs.append(prim)
        return prim

    def sphere(
        self,
        radius: float,
        lpos: tuple[float, float, float],
        color: ColorType,
        name: str = "",
        rho: float = 300.0,
        friction: float = 0.5,
        fixed: bool = False,
    ) -> Prim:
        wx, wy = self._to_world(lpos[0], lpos[1])
        prim = Prim(
            kind="sphere",
            size=(radius,),
            pos=(wx, wy, lpos[2]),
            color=tuple(color),
            name=self._next_name(name or "sph"),
            euler=(0.0, 0.0, self.rot),
            rho=rho,
            friction=friction,
            fixed=fixed,
        )
        self.primljs.append(prim)
        return prim

    def extend(self, prims: Sequence[Prim]) -> None:
        self.primljs.extend(prims)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _rgba(color: ColorType) -> tuple[int, int, int, int]:
    rgb = tuple(int(c * 255) for c in color[:3])
    alpha = int(color[3] * 255) if len(color) > 3 else 255
    return rgb + (alpha,)


def prim_mesh(prim: Prim) -> Any:
    """Build a colored trimesh for a prim (rotation applied to the mesh)."""
    if prim.kind == "box":
        mesh = trimesh.creation.box(extents=prim.size)
    elif prim.kind == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=prim.size[0], height=prim.size[1], sections=24
        )
    elif prim.kind == "sphere":
        mesh = trimesh.creation.icosphere(radius=prim.size[0], subdivisions=2)
    else:
        raise ValueError(f"unknown prim kind: {prim.kind}")
    if any(prim.euler):
        mesh.apply_transform(
            euler_matrix(*[math.radians(a) for a in prim.euler], axes="sxyz")
        )
    rgba = _rgba(prim.color)
    mesh.visual.vertex_colors = [rgba] * len(mesh.vertices)
    return mesh


class TrimeshBackend:
    """Offline backend: collects prims into a ``trimesh.Scene`` (GLB-ready)."""

    def __init__(self) -> None:
        self.scene = trimesh.Scene()
        self._names: dict[str, int] = {}

    def add(self, prim: Prim) -> None:
        n = self._names.get(prim.name, 0)
        self._names[prim.name] = n + 1
        node = prim.name if n == 0 else f"{prim.name}_{n}"
        self.scene.add_geometry(prim_mesh(prim), node_name=node, geom_name=node)


class GenesisBackend:
    """Live backend: one rigid entity per prim in a Genesis scene.

    Uses genesis-world >= 1.4 primitive morphs (``Box``/``Cylinder``/
    ``Sphere``) with ``gs.surfaces.Default`` colors — ``gs.morphs.Mesh`` no
    longer accepts in-memory trimesh objects in 1.4.
    """

    def __init__(self, gs_scene: Any):
        self.scene = gs_scene

    def add(self, prim: Prim) -> Any:
        rgba = tuple(prim.color[:3]) + (
            (prim.color[3],) if len(prim.color) > 3 else (1.0,)
        )
        surface = gs.surfaces.Default(color=rgba, roughness=0.8)
        kwargs: dict[str, Any] = {"pos": prim.pos, "fixed": prim.fixed}
        if any(prim.euler) and quaternion_from_euler is not None:
            # trimesh/Gohlke quaternions are w-first, same as Genesis.
            kwargs["quat"] = tuple(
                quaternion_from_euler(
                    *[math.radians(a) for a in prim.euler], axes="sxyz"
                )
            )
        if prim.kind == "box":
            morph = gs.morphs.Box(size=tuple(prim.size), **kwargs)
        elif prim.kind == "cylinder":
            morph = gs.morphs.Cylinder(
                radius=prim.size[0], height=prim.size[1], **kwargs
            )
        elif prim.kind == "sphere":
            morph = gs.morphs.Sphere(radius=prim.size[0], **kwargs)
        else:
            raise ValueError(f"unknown prim kind: {prim.kind}")
        return self.scene.add_entity(
            morph,
            surface=surface,
            material=gs.materials.Rigid(rho=prim.rho, friction=prim.friction),
        )


# ---------------------------------------------------------------------------
# Furniture pieces (tile-local, N = +y, anchored to pb.hx / pb.hy walls)
# ---------------------------------------------------------------------------


def add_sofa(
    pb: PrimBuilder,
    lx: float = 0.0,
    ly: Optional[float] = None,
    width: Optional[float] = None,
    color: ColorType = FABRIC,
    with_pillows: bool = True,
) -> float:
    """三人布艺沙发，靠背贴 N 墙，朝 -y。返回沙发中心的局部 y（供茶几定位）。"""
    ly = pb.hy - 0.49 if ly is None else ly
    width = min(width or 2.3, 2 * pb.hx - 0.7)
    pb.box((width, 0.85, 0.42), (lx, ly, 0.21), color, "sofa_base", rho=120)
    pb.box((width, 0.22, 0.60), (lx, ly + 0.32, 0.72), color, "sofa_back", rho=100)
    pb.box(
        (0.22, 0.85, 0.28),
        (lx - width / 2 + 0.11, ly, 0.56),
        color,
        "sofa_arm",
        rho=100,
    )
    pb.box(
        (0.22, 0.85, 0.28),
        (lx + width / 2 - 0.11, ly, 0.56),
        color,
        "sofa_arm",
        rho=100,
    )
    seats = max(int(width // 0.75), 1)
    for k in range(seats):
        cx = lx - width / 2 + 0.11 + (width - 0.22) / seats * (k + 0.5)
        pb.box(
            ((width - 0.22) / seats - 0.06, 0.62, 0.13),
            (cx, ly - 0.05, 0.49),
            FABRIC_GREY,
            "sofa_cushion",
            rho=40,
        )
    if with_pillows:
        for sx in (-0.35, 0.35):
            pb.box(
                (0.42, 0.16, 0.42),
                (lx + sx * width / 2, ly + 0.18, 0.62),
                GOLD,
                "sofa_pillow",
                rho=20,
            )
    return ly


def add_tea_table(
    pb: PrimBuilder,
    lx: float = 0.0,
    ly: Optional[float] = None,
    width: float = 1.2,
    depth: float = 0.6,
    height: float = 0.42,
) -> None:
    """实木茶几 + 茶具（默认贴在沙发前方）。"""
    ly = pb.hy - 1.66 if ly is None else ly
    width = min(width, 2 * pb.hx - 0.6)
    pb.box((width, depth, 0.04), (lx, ly, height), WOOD, "tea_table_top", rho=280)
    pb.box(
        (width - 0.25, depth - 0.2, 0.03),
        (lx, ly, 0.14),
        WOOD,
        "tea_table_shelf",
        rho=180,
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            pb.cyl(
                0.03,
                height - 0.03,
                (
                    lx + sx * (width / 2 - 0.08),
                    ly + sy * (depth / 2 - 0.08),
                    (height - 0.03) / 2,
                ),
                WOOD_DARK,
                "tea_table_leg",
            )
    # 茶壶 + 茶杯（中国家庭的茶几标配）
    pb.cyl(0.055, 0.05, (lx + 0.12, ly, height + 0.045), PORCELAIN, "teapot", rho=500)
    pb.sphere(0.05, (lx + 0.12, ly, height + 0.10), PORCELAIN, "teapot_body", rho=500)
    pb.cyl(
        0.032,
        0.025,
        (lx - 0.18, ly + 0.08, height + 0.032),
        PORCELAIN,
        "teacup",
        rho=500,
    )
    pb.cyl(
        0.032,
        0.025,
        (lx - 0.05, ly - 0.12, height + 0.032),
        PORCELAIN,
        "teacup",
        rho=500,
    )


def add_tv_set(pb: PrimBuilder, lx: float = 0.0, ly: Optional[float] = None) -> None:
    """电视墙：电视柜贴 S 墙 + 壁挂电视（固定在隔墙/外墙上）。"""
    ly = 0.27 - pb.hy if ly is None else ly
    cabinet_w = min(2.0, 2 * pb.hx - 0.6)
    pb.box((cabinet_w, 0.42, 0.45), (lx, ly, 0.225), WOOD_DARK, "tv_cabinet", rho=260)
    pb.box(
        (min(1.58, cabinet_w - 0.3), 0.05, 0.92),
        (lx, ly - 0.12, 1.38),
        BLACK,
        "tv_bezel",
        rho=350,
        fixed=True,
    )
    pb.box(
        (min(1.48, cabinet_w - 0.4), 0.04, 0.82),
        (lx, ly - 0.16, 1.38),
        (0.05, 0.07, 0.12),
        "tv_screen",
        rho=300,
        fixed=True,
    )
    pb.box(
        (0.5, 0.3, 0.22),
        (lx - 0.55, ly + 0.02, 0.56),
        (0.2, 0.2, 0.22),
        "router",
        rho=60,
    )
    # 机顶盒的日常感
    pb.box((0.3, 0.2, 0.05), (lx + 0.5, ly + 0.02, 0.475), BLACK, "settop", rho=60)


def add_tv_set_corner(pb: PrimBuilder, ly: float = -0.2) -> None:
    """转角客厅的电视墙：贴 E 墙，屏幕朝 W（面向沙发）。"""
    lx = pb.hx - 0.27
    span_y = min(2.0, 2 * pb.hy - 0.6)
    pb.box((0.42, span_y, 0.45), (lx, ly, 0.225), WOOD_DARK, "tv_cabinet", rho=260)
    pb.box(
        (0.05, min(1.58, span_y - 0.3), 0.92),
        (lx - 0.12, ly, 1.38),
        BLACK,
        "tv_bezel",
        rho=350,
        fixed=True,
    )
    pb.box(
        (0.04, min(1.48, span_y - 0.4), 0.82),
        (lx - 0.16, ly, 1.38),
        (0.05, 0.07, 0.12),
        "tv_screen",
        rho=300,
        fixed=True,
    )
    pb.box(
        (0.3, 0.5, 0.22),
        (lx + 0.02, ly + 0.55, 0.56),
        (0.2, 0.2, 0.22),
        "router",
        rho=60,
    )
    pb.box((0.2, 0.3, 0.05), (lx + 0.02, ly - 0.5, 0.475), BLACK, "settop", rho=60)


def add_carpet(
    pb: PrimBuilder,
    lx: float = 0.0,
    ly: float = -0.1,
    width: float = 2.6,
    depth: float = 1.9,
    color: ColorType = CARPET_BEIGE,
) -> None:
    """地毯（自动裁剪到格子内）。"""
    width = min(width, 2 * pb.hx - 0.5)
    depth = min(depth, 2 * pb.hy - 0.5)
    pb.box((width, depth, 0.02), (lx, ly, 0.012), color, "carpet", rho=60, fixed=True)


def add_plant(pb: PrimBuilder, lx: float, ly: float, scale: float = 1.0) -> None:
    """盆栽绿植（客厅/阳台常见的绿萝、发财树）。"""
    pb.cyl(
        0.17 * scale, 0.30 * scale, (lx, ly, 0.15 * scale), POT, "plant_pot", rho=600
    )
    pb.sphere(0.30 * scale, (lx, ly, 0.52 * scale), GREEN, "plant_leaves", rho=60)
    pb.sphere(
        0.22 * scale,
        (lx + 0.10 * scale, ly - 0.06 * scale, 0.74 * scale),
        GREEN,
        "plant_leaves",
        rho=60,
    )


def add_fish_tank(
    pb: PrimBuilder, lx: Optional[float] = None, ly: Optional[float] = None
) -> None:
    """鱼缸 + 木柜底座（客厅角落常见），默认 NE 角。"""
    lx = pb.hx - 0.42 if lx is None else lx
    ly = pb.hy - 0.44 if ly is None else ly
    pb.box((0.62, 0.48, 0.55), (lx, ly, 0.275), WOOD_DARK, "tank_stand", rho=350)
    pb.box((0.58, 0.44, 0.42), (lx, ly, 0.76), GLASS, "tank_glass", rho=250, fixed=True)
    pb.box((0.52, 0.38, 0.34), (lx, ly, 0.73), WATER, "tank_water", rho=400, fixed=True)
    pb.box((0.60, 0.46, 0.03), (lx, ly, 0.985), GOLD, "tank_lid", rho=200, fixed=True)


def add_scroll(
    pb: PrimBuilder, lx: float, wall_y: Optional[float] = None, facing: int = -1
) -> None:
    """墙面字画（装裱卷轴），``wall_y`` 默认贴 N 墙内面，``facing`` 朝向。"""
    wall_y = pb.hy - WALL_INSET if wall_y is None else wall_y
    pb.box(
        (1.1, 0.03, 0.55),
        (lx, wall_y, 1.80),
        WOOD_DARK,
        "scroll_frame",
        rho=150,
        fixed=True,
    )
    pb.box(
        (1.0, 0.02, 0.45),
        (lx, wall_y + 0.025 * facing, 1.80),
        PORCELAIN,
        "scroll_paper",
        rho=60,
        fixed=True,
    )
    pb.box(
        (0.08, 0.022, 0.08),
        (lx - 0.36, wall_y + 0.03 * facing, 1.66),
        RED,
        "scroll_seal",
        rho=60,
        fixed=True,
    )


def add_lantern(pb: PrimBuilder, lx: float, ly: float, z: float = 2.3) -> None:
    """红灯笼（节日装饰，吸顶吊挂，固定）。"""
    pb.cyl(0.012, 0.16, (lx, ly, z + 0.17), GOLD, "lantern_cord", fixed=True)
    pb.cyl(0.05, 0.03, (lx, ly, z + 0.10), GOLD, "lantern_cap", fixed=True)
    pb.sphere(0.16, (lx, ly, z - 0.03), RED, "lantern_body", rho=40, fixed=True)
    pb.cyl(0.05, 0.03, (lx, ly, z - 0.16), GOLD, "lantern_base", fixed=True)
    pb.box((0.02, 0.14, 0.14), (lx, ly, z - 0.03), GOLD, "lantern_tassel", fixed=True)


def add_dining_set(pb: PrimBuilder, lx: float = 0.0, ly: float = 0.0) -> None:
    """圆餐桌 + 转盘 + 四把椅子（中式餐厅核心；椅子半径随格子收缩）。"""
    r = min(0.92, min(pb.hx, pb.hy) - 0.55)
    pb.cyl(0.34, 0.05, (lx, ly, 0.025), WOOD_DARK, "dining_base")
    pb.cyl(0.08, 0.68, (lx, ly, 0.39), WOOD_DARK, "dining_column")
    pb.cyl(0.62, 0.05, (lx, ly, 0.745), WOOD, "dining_top", rho=280)
    pb.cyl(0.32, 0.02, (lx, ly, 0.78), GLASS, "dining_turntable", rho=200)
    # 桌上的碗筷茶杯
    for k in range(4):
        ang = math.radians(45 + 90 * k)
        bx, by = lx + 0.36 * math.cos(ang), ly + 0.36 * math.sin(ang)
        pb.cyl(0.055, 0.03, (bx, by, 0.795), PORCELAIN, "bowl", rho=500)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = lx + sx * r, ly + sy * r
            face = math.degrees(math.atan2(-sy, -sx))  # 面向餐桌
            pb.box(
                (0.42, 0.42, 0.06),
                (cx, cy, 0.45),
                WOOD,
                "chair_seat",
                rho=150,
                spin=face,
            )
            pb.box(
                (0.42, 0.05, 0.50),
                (cx - sx * 0.185, cy - sy * 0.185, 0.73),
                WOOD,
                "chair_back",
                rho=150,
                spin=face,
            )
            pb.box(
                (0.36, 0.36, 0.42), (cx, cy, 0.21), WOOD_DARK, "chair_pedestal", rho=150
            )


def add_kitchen(pb: PrimBuilder) -> None:
    """U 型橱柜 + 灶台 + 抽油烟机 + 水槽 + 冰箱（rot=0: 台面沿 N/W/S，开口朝 E）。

    台面长度随格子跨度自适应（最小边长 2.8m 保证灶台/水槽放得下）。
    """
    cab = (0.84, 0.80, 0.74)
    # N 台面（灶台位）
    ln = 2 * pb.hx - 0.3
    pb.box((ln, 0.6, 0.82), (0.0, pb.hy - 0.36, 0.41), cab, "kitchen_cabinet", rho=300)
    pb.box(
        (ln + 0.04, 0.64, 0.04),
        (0.0, pb.hy - 0.36, 0.845),
        MARBLE,
        "kitchen_counter",
        rho=500,
    )
    # W 台面（水槽位）
    lw = 2 * pb.hy - 0.3
    pb.box(
        (0.6, lw, 0.82), (-(pb.hx - 0.36), 0.0, 0.41), cab, "kitchen_cabinet", rho=300
    )
    pb.box(
        (0.64, lw + 0.04, 0.04),
        (-(pb.hx - 0.36), 0.0, 0.845),
        MARBLE,
        "kitchen_counter",
        rho=500,
    )
    # S 台面（半段，贴 S 墙，给冰箱让位）
    ls = min(1.5, 2 * pb.hx - 1.45)
    pb.box(
        (ls, 0.6, 0.82),
        (-(pb.hx - 0.3) + ls / 2, -(pb.hy - 0.36), 0.41),
        cab,
        "kitchen_cabinet",
        rho=300,
    )
    pb.box(
        (ls + 0.04, 0.64, 0.04),
        (-(pb.hx - 0.3) + ls / 2, -(pb.hy - 0.36), 0.845),
        MARBLE,
        "kitchen_counter",
        rho=500,
    )
    # 灶台（双灶）+ 抽油烟机（N 台面西段）
    stove_x = -(pb.hx / 2 - 0.15)
    stove_y = pb.hy - 0.36
    pb.box((0.68, 0.44, 0.03), (stove_x, stove_y, 0.875), BLACK, "stove_top", rho=350)
    for sx in (-0.17, 0.17):
        pb.cyl(
            0.085,
            0.02,
            (stove_x + sx, stove_y, 0.90),
            (0.25, 0.12, 0.10),
            "stove_burner",
            rho=400,
        )
    pb.box(
        (0.70, 0.50, 0.22),
        (stove_x, stove_y + 0.06, 1.72),
        STEEL,
        "range_hood",
        rho=250,
        fixed=True,
    )
    pb.box(
        (0.30, 0.30, 0.55),
        (stove_x, stove_y + 0.14, 2.10),
        STEEL,
        "range_duct",
        rho=250,
        fixed=True,
    )
    # 水槽 + 龙头（W 台面）
    sink_x = -(pb.hx - 0.36)
    pb.box((0.40, 0.50, 0.03), (sink_x, 0.15 * pb.hy, 0.865), STEEL, "sink", rho=500)
    pb.cyl(0.02, 0.28, (sink_x - 0.18, 0.15 * pb.hy, 0.99), STEEL, "faucet", rho=500)
    pb.cyl(
        0.015,
        0.18,
        (sink_x - 0.10, 0.15 * pb.hy, 1.11),
        STEEL,
        "faucet_arm",
        rho=500,
        axis="x",
    )
    # 冰箱（SE 角，不挡 E 开口）
    pb.box(
        (0.64, 0.62, 1.72),
        (pb.hx - 0.42, -(pb.hy - 0.42), 0.86),
        STEEL,
        "fridge",
        rho=180,
    )
    pb.box(
        (0.04, 0.05, 0.5),
        (pb.hx - 0.77, -(pb.hy - 0.24), 1.05),
        (0.55, 0.57, 0.60),
        "fridge_handle",
        rho=200,
    )


def add_bed_group(pb: PrimBuilder, lx: float = 0.0, dx: float = 0.0) -> None:
    """双人床组（床头贴 N 墙）；``dx`` 为整组沿 x 的偏移。"""
    hy = pb.hy
    pb.box(
        (1.62, 0.06, 1.00),
        (lx + dx, hy - 0.09, 0.50),
        WOOD,
        "bed_headboard",
        rho=250,
        fixed=True,
    )
    pb.box((1.60, 2.10, 0.22), (lx + dx, hy - 1.16, 0.11), WOOD, "bed_frame", rho=220)
    pb.box(
        (1.50, 1.95, 0.22), (lx + dx, hy - 1.13, 0.33), MATTRESS, "bed_mattress", rho=60
    )
    pb.box((1.48, 1.15, 0.09), (lx + dx, hy - 1.52, 0.485), QUILT, "bed_quilt", rho=40)
    for sx in (-1, 1):
        pb.box(
            (0.52, 0.34, 0.12),
            (lx + dx + sx * 0.37, hy - 0.42, 0.50),
            WHITE,
            "bed_pillow",
            rho=30,
        )


def add_nightstand(pb: PrimBuilder, lx: float) -> None:
    """床头柜 + 台灯。"""
    pb.box(
        (0.45, 0.40, 0.50), (lx, pb.hy - 0.28, 0.25), WOOD_DARK, "nightstand", rho=250
    )
    pb.cyl(0.06, 0.18, (lx, pb.hy - 0.28, 0.59), (0.9, 0.85, 0.7), "lamp_base", rho=200)
    pb.sphere(0.09, (lx, pb.hy - 0.28, 0.74), (0.95, 0.90, 0.72), "lamp_shade", rho=60)


def add_wardrobe(pb: PrimBuilder, lx: float, ly: float, along_x: bool = True) -> None:
    """衣柜（靠墙放置，``along_x`` 决定柜体沿 x 还是 y 方向）。"""
    size = (1.80, 0.58, 2.20) if along_x else (0.58, 1.80, 2.20)
    pb.box(size, (lx, ly, 1.10), WOOD_LIGHT, "wardrobe", rho=280)
    if along_x:
        for sx in (-0.45, 0.45):
            pb.box(
                (0.03, 0.02, 0.30),
                (lx + sx, ly + 0.30, 1.15),
                GOLD,
                "wardrobe_handle",
                rho=200,
            )
    else:
        for sy in (-0.45, 0.45):
            pb.box(
                (0.02, 0.03, 0.30),
                (lx - 0.30, ly + sy, 1.15),
                GOLD,
                "wardrobe_handle",
                rho=200,
            )


def add_bedroom(pb: PrimBuilder) -> None:
    """双人床（床头贴 N 墙）+ 床头柜×2 + 衣柜（贴 S 墙）。"""
    add_bed_group(pb)
    for sx in (-1, 1):
        add_nightstand(pb, sx * min(1.06, pb.hx - 0.34))
    add_wardrobe(pb, 0.0, -(pb.hy - 0.35), along_x=True)


def add_desk(pb: PrimBuilder) -> None:
    """书桌 + 座椅 + 显示器（贴 N 墙）。"""
    ly = pb.hy - 0.40
    pb.box((1.50, 0.68, 0.04), (0.0, ly, 0.74), WOOD, "desk_top", rho=280)
    for sx in (-1, 1):
        pb.box(
            (0.05, 0.62, 0.72), (sx * 0.70, ly, 0.36), WOOD_DARK, "desk_leg", rho=250
        )
    pb.box(
        (0.55, 0.04, 0.34),
        (0.0, ly + 0.16, 0.94),
        BLACK,
        "monitor",
        rho=180,
        fixed=True,
    )
    pb.box((0.18, 0.12, 0.02), (0.0, ly - 0.16, 0.77), BLACK, "keyboard", rho=60)
    pb.box((0.45, 0.45, 0.06), (0.0, ly - 0.73, 0.46), WOOD, "chair_seat", rho=150)
    pb.box((0.45, 0.05, 0.55), (0.0, ly - 0.94, 0.76), WOOD, "chair_back", rho=150)
    pb.cyl(0.04, 0.44, (0.0, ly - 0.73, 0.22), STEEL, "chair_post")
    pb.cyl(0.24, 0.04, (0.0, ly - 0.73, 0.02), STEEL, "chair_base")


def add_bookshelf(
    pb: PrimBuilder,
    lx: float = 0.0,
    ly: Optional[float] = None,
    book_seed: int = 0,
    along_x: bool = True,
) -> None:
    """书架 + 一排排书（默认贴 S 墙）。"""
    ly = -(pb.hy - 0.22) if ly is None else ly
    if along_x:
        pb.box((1.80, 0.32, 2.05), (lx, ly, 1.025), WOOD_DARK, "bookshelf", rho=280)
        for level in range(4):
            z = 0.35 + level * 0.48
            pb.box((1.66, 0.26, 0.03), (lx, ly, z), WOOD, "bookshelf_board", rho=200)
            n_books = 7 + int(random.Random(book_seed + level).randint(0, 3))
            for b in range(n_books):
                col = [
                    (0.6, 0.2, 0.15),
                    (0.2, 0.35, 0.55),
                    (0.75, 0.6, 0.25),
                    (0.3, 0.5, 0.3),
                ][b % 4]
                bx = lx - 0.72 + b * (1.44 / n_books)
                pb.box((0.05, 0.2, 0.26), (bx, ly, z + 0.15), col, "book", rho=350)
    else:
        pb.box((0.32, 1.80, 2.05), (lx, ly, 1.025), WOOD_DARK, "bookshelf", rho=280)
        for level in range(4):
            z = 0.35 + level * 0.48
            pb.box((0.26, 1.66, 0.03), (lx, ly, z), WOOD, "bookshelf_board", rho=200)
            n_books = 7 + int(random.Random(book_seed + level).randint(0, 3))
            for b in range(n_books):
                col = [
                    (0.6, 0.2, 0.15),
                    (0.2, 0.35, 0.55),
                    (0.75, 0.6, 0.25),
                    (0.3, 0.5, 0.3),
                ][b % 4]
                by = ly - 0.72 + b * (1.44 / n_books)
                pb.box((0.2, 0.05, 0.26), (lx, by, z + 0.15), col, "book", rho=350)


def add_study(pb: PrimBuilder) -> None:
    """书房：书桌靠 N 墙，书架贴 S 墙。"""
    add_desk(pb)
    add_bookshelf(pb)


def add_study_corner(pb: PrimBuilder) -> None:
    """转角书房：书桌靠 N 墙，书架贴 E 墙。"""
    add_desk(pb)
    add_bookshelf(pb, lx=pb.hx - 0.22, ly=0.0, book_seed=11, along_x=False)


def add_entry(pb: PrimBuilder, festival: bool = False) -> None:
    """鞋柜 + 换鞋凳 + 地垫 + 挂衣架（玄关；rot=0 时门在 W 外墙）。"""
    pb.box(
        (0.35, 1.20, 0.90), (pb.hx - 0.275, 0.0, 0.45), WOOD, "shoe_cabinet", rho=280
    )
    for sy in (-0.3, 0.3):
        pb.box(
            (0.02, 0.03, 0.14),
            (pb.hx - 0.465, sy, 0.62),
            GOLD,
            "shoe_handle",
            rho=200,
        )
    pb.box(
        (0.40, 0.80, 0.42),
        (-pb.hx * 0.35, pb.hy * 0.4, 0.21),
        WOOD_DARK,
        "entry_bench",
        rho=220,
    )
    pb.box(
        (0.36, 0.72, 0.06),
        (-pb.hx * 0.35, pb.hy * 0.4, 0.45),
        FABRIC_GREY,
        "entry_bench_pad",
        rho=40,
    )
    pb.box(
        (0.90, 0.60, 0.02),
        (-(pb.hx - 0.6), 0.0, 0.012),
        CARPET_RED,
        "doormat",
        rho=60,
        fixed=True,
    )
    pb.cyl(
        0.025, 1.70, (pb.hx * 0.55, pb.hy * 0.62, 0.85), WOOD_DARK, "coat_rack", rho=200
    )
    for ang in (30, 150, 270):
        hx_pt = pb.hx * 0.55 + 0.14 * math.cos(math.radians(ang))
        hy_pt = pb.hy * 0.62 + 0.14 * math.sin(math.radians(ang))
        pb.sphere(0.03, (hx_pt, hy_pt, 1.62), WOOD_DARK, "coat_hook", rho=150)
    pb.box(
        (0.03, 0.50, 1.50),
        (pb.hx - 0.095, pb.hy * 0.45, 1.10),
        GLASS,
        "entry_mirror",
        rho=200,
        fixed=True,
    )
    if festival:
        pb.box(
            (0.30, 0.02, 0.30),
            (pb.hx - 0.275, -pb.hy * 0.45, 1.60),
            RED,
            "fu_character",
            rho=60,
            fixed=True,
        )


def add_balcony(pb: PrimBuilder) -> None:
    """洗衣机 + 晾衣架 + 绿植（生活阳台；rot=0 时栏杆在 W 外墙）。"""
    wx = -(pb.hx - 0.4)
    pb.box((0.60, 0.60, 0.85), (wx, pb.hy * 0.55, 0.425), WHITE, "washer", rho=220)
    pb.cyl(
        0.21,
        0.03,
        (wx + 0.32, pb.hy * 0.55, 0.55),
        GLASS,
        "washer_door",
        rho=150,
        axis="x",
    )
    pb.cyl(0.03, 0.10, (wx, pb.hy * 0.55, 0.93), STEEL, "washer_knob", rho=200)
    pb.box(
        (0.50, 0.45, 0.60),
        (-(pb.hx - 0.37), -(pb.hy * 0.55), 0.30),
        (0.82, 0.80, 0.76),
        "balcony_cabinet",
        rho=260,
    )
    px1, px2 = pb.hx * 0.2, pb.hx * 0.55
    pb.cyl(0.03, 1.9, (px1, 0.0, 0.95), STEEL, "dry_rack_pole", rho=180)
    pb.cyl(0.03, 1.9, (px2, 0.0, 0.95), STEEL, "dry_rack_pole", rho=180)
    pb.cyl(
        0.02,
        px2 - px1,
        ((px1 + px2) / 2, 0.0, 1.88),
        STEEL,
        "dry_rack_bar",
        rho=180,
        axis="x",
    )
    add_plant(pb, pb.hx - 0.35, pb.hy * 0.55, scale=1.0)
    add_plant(pb, pb.hx - 0.45, -(pb.hy * 0.2), scale=0.8)


def add_tea_table_set(pb: PrimBuilder, festival: bool = False) -> float:
    """茶台 + 茶具（贴 N 墙）。返回台面局部 y。"""
    ly = pb.hy - 0.41
    pb.box((1.60, 0.70, 0.06), (0.0, ly, 0.52), WOOD, "tea_table_top", rho=300)
    for sx in (-1, 1):
        pb.box(
            (0.08, 0.60, 0.50),
            (sx * 0.68, ly, 0.25),
            WOOD_DARK,
            "tea_table_leg",
            rho=250,
        )
    pb.cyl(0.06, 0.07, (0.0, ly, 0.585), PORCELAIN, "teapot_big", rho=500)
    pb.sphere(0.08, (0.0, ly, 0.66), PORCELAIN, "teapot_body", rho=500)
    for k in range(4):
        ang = math.radians(45 + 90 * k)
        pb.cyl(
            0.035,
            0.03,
            (0.28 * math.cos(ang), ly + 0.22 * math.sin(ang), 0.575),
            PORCELAIN,
            "tea_cup",
            rho=500,
        )
    return ly


def add_stools(pb: PrimBuilder) -> None:
    """坐墩×4（茶台南侧，两排）。"""
    row1, row2 = -(pb.hy - 1.45), -(pb.hy - 0.7)
    for sx in (-0.75, 0.75):
        pb.cyl(0.22, 0.38, (sx, row1, 0.19), WOOD_LIGHT, "tea_stool", rho=220)
        pb.cyl(0.22, 0.38, (sx, row2, 0.19), WOOD_LIGHT, "tea_stool", rho=220)


def add_curio_shelf(
    pb: PrimBuilder,
    lx: Optional[float] = None,
    ly: Optional[float] = None,
    along_x: bool = False,
) -> None:
    """博古架 + 瓷器（默认贴 E 墙）。"""
    lx = pb.hx - 0.21 if lx is None else lx
    ly = 0.2 if ly is None else ly
    size = (0.80, 0.30, 1.80) if along_x else (0.30, 0.80, 1.80)
    pb.box(size, (lx, ly, 0.90), WOOD_DARK, "curio_shelf", rho=260)
    for level in range(3):
        z = 0.45 + level * 0.55
        if along_x:
            pb.box((0.70, 0.24, 0.03), (lx, ly, z), WOOD, "curio_board", rho=200)
            pb.sphere(0.07, (lx - 0.15, ly, z + 0.13), PORCELAIN, "curio_vase", rho=400)
        else:
            pb.box((0.24, 0.70, 0.03), (lx, ly, z), WOOD, "curio_board", rho=200)
            pb.sphere(0.07, (lx, ly - 0.15, z + 0.13), PORCELAIN, "curio_vase", rho=400)


def add_tea_room(pb: PrimBuilder, festival: bool = False) -> None:
    """茶台 + 坐墩 + 博古架 + 字画（茶室）。"""
    add_tea_table_set(pb)
    add_stools(pb)
    add_curio_shelf(pb)
    add_scroll(pb, -0.85, wall_y=pb.hy - WALL_INSET, facing=-1)
    if festival:
        add_lantern(pb, -pb.hx * 0.75, pb.hy * 0.75, z=2.25)


def add_bathroom(pb: PrimBuilder) -> None:
    """马桶 + 洗手台 + 淋浴（卫生间；rot=0 时门洞在 W）。"""
    tx = pb.hx - 0.30
    pb.box((0.38, 0.16, 0.52), (tx, pb.hy - 0.30, 0.52), WHITE, "toilet_tank", rho=250)
    pb.box((0.36, 0.44, 0.38), (tx, pb.hy - 0.58, 0.19), WHITE, "toilet_base", rho=250)
    pb.cyl(0.20, 0.05, (tx, pb.hy - 0.62, 0.41), WHITE, "toilet_seat", rho=250)
    sx = -pb.hx * 0.2
    pb.box(
        (0.90, 0.50, 0.80), (sx, pb.hy - 0.31, 0.40), MARBLE, "sink_counter", rho=300
    )
    pb.cyl(0.18, 0.12, (sx, pb.hy - 0.31, 0.86), PORCELAIN, "sink_basin", rho=400)
    pb.cyl(0.018, 0.25, (sx, pb.hy - 0.12, 0.98), STEEL, "sink_faucet", rho=400)
    pb.box(
        (0.60, 0.02, 0.80),
        (sx, pb.hy - WALL_INSET - 0.01, 1.55),
        GLASS,
        "bathroom_mirror",
        rho=200,
        fixed=True,
    )
    pb.cyl(
        0.02, 1.60, (pb.hx - 0.35, -(pb.hy - 0.35), 1.20), STEEL, "shower_pipe", rho=300
    )
    pb.box(
        (0.20, 0.20, 0.04),
        (pb.hx - 0.50, -(pb.hy - 0.50), 1.95),
        STEEL,
        "shower_head",
        rho=300,
        fixed=True,
    )
    pb.cyl(
        0.30,
        0.015,
        (pb.hx - 0.55, -(pb.hy - 0.55), 0.01),
        (0.7, 0.72, 0.75),
        "shower_drain",
        rho=400,
    )


def add_corridor(pb: PrimBuilder) -> None:
    """过道地毯 + 绿植。"""
    pb.box(
        (0.80, 2 * pb.hy - 0.9, 0.02),
        (0.0, 0.0, 0.012),
        (0.55, 0.26, 0.20),
        "corridor_runner",
        rho=60,
        fixed=True,
    )
    add_plant(pb, -(pb.hx - 0.35), -(pb.hy - 0.35), scale=0.8)


def add_living_corner(pb: PrimBuilder, rng: random.Random, festival: bool) -> None:
    """转角客厅：沙发靠 N 墙，电视墙贴 E 墙（墙角相邻）。"""
    add_carpet(pb)
    sofa_ly = add_sofa(pb)  # 靠 N 墙
    add_tea_table(pb, ly=sofa_ly - 1.17)
    add_tv_set_corner(pb)  # 电视墙贴 E 墙，面向沙发
    if rng.random() < 0.55:
        add_fish_tank(pb, lx=-(pb.hx - 0.42), ly=-(pb.hy - 0.44))
    add_scroll(pb, lx=0.0, wall_y=pb.hy - WALL_INSET, facing=-1)
    if rng.random() < 0.5:
        add_plant(pb, -(pb.hx - 0.32), pb.hy * 0.15, scale=1.0)
    if festival:
        add_lantern(pb, -pb.hx * 0.6, -(pb.hy - 0.3), z=2.30)
        add_lantern(pb, pb.hx * 0.6, pb.hy - 0.3, z=2.30)


def add_bedroom_corner(pb: PrimBuilder, rng: random.Random) -> None:
    """转角卧室：床头靠 N 墙，衣柜贴 E 墙。"""
    add_bed_group(pb, dx=-0.15)
    add_nightstand(pb, -(pb.hx - 0.39))
    add_wardrobe(pb, pb.hx - 0.35, 0.0, along_x=False)
    if rng.random() < 0.35:
        add_carpet(pb, lx=-0.4, ly=-(pb.hy - 0.55), width=1.6, depth=0.9)


def add_tea_corner(pb: PrimBuilder, festival: bool) -> None:
    """转角茶室：茶台靠 N 墙，博古架贴 E 墙。"""
    add_tea_table_set(pb)
    add_stools(pb)
    add_curio_shelf(pb)
    add_scroll(pb, -0.85, wall_y=pb.hy - WALL_INSET, facing=-1)
    if festival:
        add_lantern(pb, -pb.hx * 0.75, pb.hy * 0.75, z=2.25)


# ---------------------------------------------------------------------------
# Tile assembly entry points
# ---------------------------------------------------------------------------


def build_tile_prims(
    tile: str,
    center: tuple[float, float],
    rot: float,
    rng: random.Random,
    festival: bool = False,
    pattern: int = 0,
    half: tuple[float, float] = (1.6, 1.6),
) -> list[Prim]:
    """Build all furniture prims for one collapsed tile.

    Args:
        tile: Tile name (see ``rules_cn.TILES``).
        center: World XY of the tile center.
        rot: WFC rotation (degrees, clockwise).
        rng: Layout RNG for stochastic accents (plants, fish tank, books).
        festival: Add Spring Festival decorations.
        pattern: Wall pattern index (1 = corner arrangement where offered).
        half: Cell half extents (hx, hy) — furniture anchors to these walls.

    Returns:
        List of :class:`Prim` (deterministic given ``rng``).
    """
    pb = PrimBuilder(origin=center, rot=rot, half=half)

    if tile == "living":
        if pattern == 1:
            add_living_corner(pb, rng, festival)
        else:
            add_carpet(pb, ly=-(pb.hy - 1.5))
            sofa_ly = add_sofa(pb)  # 靠 N 墙
            add_tea_table(pb, ly=sofa_ly - 1.17)
            add_tv_set(pb)  # 电视墙贴 S 墙（与沙发相对）
            if rng.random() < 0.55:
                add_fish_tank(pb)
            add_scroll(pb, lx=0.0)  # 沙发墙上挂字画
            if rng.random() < 0.5:
                add_plant(pb, -(pb.hx - 0.32), -(pb.hy - 0.32), scale=1.0)
            if festival:
                add_lantern(pb, -pb.hx * 0.6, -(pb.hy - 0.3), z=2.30)
                add_lantern(pb, pb.hx * 0.6, -(pb.hy - 0.3), z=2.30)
    elif tile == "dining":
        add_dining_set(pb)
        if rng.random() < 0.4:
            add_plant(pb, -(pb.hx - 0.32), pb.hy - 0.32, scale=0.9)
        if festival:
            add_lantern(pb, 0.0, pb.hy - 0.3, z=2.30)
    elif tile == "kitchen":
        add_kitchen(pb)
    elif tile == "bedroom":
        if pattern == 1:
            add_bedroom_corner(pb, rng)
        else:
            add_bedroom(pb)
            if rng.random() < 0.35:
                add_carpet(pb, ly=-(pb.hy - 0.55), width=1.6, depth=0.9)
    elif tile == "study":
        if pattern == 1:
            add_study_corner(pb)
        else:
            add_study(pb)
    elif tile == "entry":
        add_entry(pb, festival=festival)
    elif tile == "balcony":
        add_balcony(pb)
    elif tile == "tea":
        if pattern == 1:
            add_tea_corner(pb, festival)
        else:
            add_tea_room(pb, festival=festival)
    elif tile == "bathroom":
        add_bathroom(pb)
    elif tile == "corridor":
        add_corridor(pb)
    else:
        raise ValueError(f"unknown tile: {tile}")

    return pb.primljs


__all__ = [
    "BLACK",
    "CARPET_BEIGE",
    "CARPET_RED",
    "FABRIC",
    "FABRIC_GREY",
    "GOLD",
    "GREEN",
    "POT",
    "PORCELAIN",
    "Prim",
    "PrimBuilder",
    "QUILT",
    "RED",
    "STEEL",
    "TrimeshBackend",
    "WHITE",
    "WOOD",
    "WOOD_DARK",
    "WOOD_LIGHT",
    "build_tile_prims",
    "prim_mesh",
]
