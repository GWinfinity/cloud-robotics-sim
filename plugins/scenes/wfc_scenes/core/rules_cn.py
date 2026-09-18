# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Chinese-home layout rules for the WFC scene generator.

This module is the "整理好的规则" (organized rule set) that encodes how a
mainstream Chinese apartment is laid out, as WFC constraints:

- **模块目录 (tile catalog)**: everyday Chinese home modules — 客厅/餐厅/厨房/
  卧室/书房/玄关/阳台/茶室/卫生间/过道 — each with edge sockets and a prior
  weight reflecting how common the module is.
- **邻接规则 (adjacency)**: socket compatibility, e.g. 台面延续/顶墙
  (counter runs continue across cells or end against walls), 开放边只能接
  开放边 (open edges only meet open edges). 厨房开口面向开放空间，全局校验
  再保证“厨房必须邻餐厅/客厅”（现代户型客餐厨一体）。
- **边界规则 (boundary)**: only walls, counters, 入户门 and 阳台栏杆 may face
  the exterior; open passages may not — homes are closed shells, and this is
  what forces 玄关 doors and 阳台 railings onto exterior walls.
- **全局校验 (global checks)**: rules that single edges cannot express
  (exactly one 玄关, 进门动线, 阳台必须连居室...) are validated on the final
  layout; the scene generator retries with fresh seeds until they pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .wfc import DIRS, expand_variants, rotated_sockets

# ---------------------------------------------------------------------------
# Sockets (边插座)
# ---------------------------------------------------------------------------

SOCKET_WALL = "wall"  # 实墙 / 家具靠墙（隔墙、电视墙、床头墙…）
SOCKET_OPEN = "open"  # 开放通行（动线、门洞、厨房开口）
SOCKET_COUNTER = "counter"  # 厨房台面沿
SOCKET_RAIL = "rail"  # 阳台栏杆 / 外窗
SOCKET_DOOR = "door"  # 入户门

ALL_SOCKETS = frozenset(
    {
        SOCKET_WALL,
        SOCKET_OPEN,
        SOCKET_COUNTER,
        SOCKET_RAIL,
        SOCKET_DOOR,
    }
)

# Interior adjacency rules (对称). 整理规则时按“两个模块在这条边上如何相遇”描述。
INTERIOR_COMPAT: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({SOCKET_WALL, SOCKET_WALL}),  # 背靠背共用隔墙
        frozenset({SOCKET_OPEN, SOCKET_OPEN}),  # 开放动线连通
        frozenset({SOCKET_COUNTER, SOCKET_COUNTER}),  # 台面跨格延续
        frozenset({SOCKET_COUNTER, SOCKET_WALL}),  # 台面顶到邻居隔墙
    }
)

# Sockets allowed to face the grid exterior. open/pass_* 不允许朝外，
# 因此住宅总是封闭的；门和栏杆只能出现在外墙上。
EXTERIOR_OK: frozenset[str] = frozenset(
    {SOCKET_WALL, SOCKET_COUNTER, SOCKET_DOOR, SOCKET_RAIL}
)


def compatible(socket_a: str, socket_b: str) -> bool:
    """Whether two sockets may meet on an interior edge."""
    if socket_a == socket_b:
        return socket_a in {SOCKET_WALL, SOCKET_OPEN, SOCKET_COUNTER}
    return frozenset({socket_a, socket_b}) in INTERIOR_COMPAT


def exterior_ok(socket: str) -> bool:
    """Whether a socket may face the outside of the apartment."""
    return socket in EXTERIOR_OK


# ---------------------------------------------------------------------------
# 模块目录 (tile catalog)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """One Chinese-home module.

    Attributes:
        name: Machine name used in layouts and builders.
        zh: Chinese display name.
        sockets: Default local-frame N/E/S/W wall pattern (pattern 0).
        weight: Prior sampling weight (how common the module is).
        floor_color: RGB of the per-module floor finish.
        ascii_char: Single character used in ASCII layout maps.
        patterns: Extra local-frame wall patterns (e.g. corner rooms with two
            adjacent walls). Pattern 0 is ``sockets`` itself; furniture
            builders implement each pattern's arrangement.
        min_edge: Minimum cell edge length (m) for this module — both the
            column width and the row height hosting it must be at least this
            (厅大卧小卫小 comes from this field). Furniture builders adapt to
            the actual cell span.
    """

    name: str
    zh: str
    sockets: tuple[str, str, str, str]
    weight: float
    floor_color: tuple[float, float, float]
    ascii_char: str
    patterns: tuple[tuple[str, str, str, str], ...] = ()
    min_edge: float = 3.0

    def all_patterns(self) -> tuple[tuple[str, str, str, str], ...]:
        """All wall patterns of this tile, pattern 0 first."""
        return (self.sockets,) + self.patterns


TILES: dict[str, TileSpec] = {
    spec.name: spec
    for spec in (
        # 客厅：沙发靠一面墙、电视墙在对面（pattern 0）；转角客厅（pattern 1）
        # 电视墙与沙发墙相邻。
        TileSpec(
            "living",
            "客厅",
            ("wall", "open", "wall", "open"),
            1.0,
            (0.72, 0.56, 0.36),
            "L",
            patterns=(("wall", "wall", "open", "open"),),
            min_edge=3.2,  # 与统一格子默认一致；可变模式下客厅列/行会自动放大
        ),
        # 餐厅：四边开放（现代户型餐客一体，餐厅位置由全局校验保证）。
        TileSpec(
            "dining",
            "餐厅",
            ("open", "open", "open", "open"),
            0.75,
            (0.74, 0.58, 0.38),
            "D",
            min_edge=3.0,
        ),
        # 厨房：三面台面靠墙/外墙，一面开口通向餐/客区（中式封闭厨房）。
        TileSpec(
            "kitchen",
            "厨房",
            ("counter", "open", "counter", "counter"),
            0.35,
            (0.72, 0.73, 0.75),
            "K",
            min_edge=2.8,
        ),
        # 卧室：床头墙 + 衣柜墙（对面或转角）。
        TileSpec(
            "bedroom",
            "卧室",
            ("wall", "open", "wall", "open"),
            0.55,
            (0.78, 0.63, 0.43),
            "B",
            patterns=(("wall", "wall", "open", "open"),),
            min_edge=3.0,
        ),
        # 书房：书桌墙 + 书架墙（对面或转角）。
        TileSpec(
            "study",
            "书房",
            ("wall", "open", "wall", "open"),
            0.35,
            (0.75, 0.60, 0.40),
            "S",
            patterns=(("wall", "wall", "open", "open"),),
            min_edge=2.8,
        ),
        # 玄关：入户门外墙 + 鞋柜靠对墙（数量全局限制为 1）。
        TileSpec(
            "entry",
            "玄关",
            ("open", "wall", "open", "door"),
            1.0,
            (0.70, 0.54, 0.34),
            "E",
            min_edge=2.4,
        ),
        # 阳台：一面外栏杆（洗衣机/绿植/晾衣），其余开放。
        TileSpec(
            "balcony",
            "阳台",
            ("open", "open", "open", "rail"),
            0.5,
            (0.62, 0.63, 0.65),
            "Y",
            min_edge=2.4,
        ),
        # 茶室：茶台靠一面墙或占据转角。
        TileSpec(
            "tea",
            "茶室",
            ("wall", "open", "open", "open"),
            0.25,
            (0.71, 0.57, 0.37),
            "T",
            patterns=(("wall", "wall", "open", "open"),),
            min_edge=2.8,
        ),
        # 卫生间：三面墙、一面开放作门（干湿分离的封闭小间）。
        TileSpec(
            "bathroom",
            "卫生间",
            ("wall", "wall", "wall", "open"),
            0.22,
            (0.68, 0.74, 0.78),
            "W",
            min_edge=2.2,
        ),
        # 过道：四边开放的动线。
        TileSpec(
            "corridor",
            "过道",
            ("open", "open", "open", "open"),
            0.15,
            (0.69, 0.55, 0.35),
            "C",
            min_edge=2.0,
        ),
    )
}

# 每类模块的数量上限（一套中国住宅的合理构成），由全局校验强制。
MAX_TILE_COUNTS: dict[str, int] = {
    "living": 2,
    "entry": 1,
    "kitchen": 1,  # 一套住宅一个厨房（预设格保证其餐厨相邻）
    "bathroom": 2,
    "corridor": 2,
    "tea": 1,
    "study": 1,
    "dining": 2,
    "balcony": 2,
    "bedroom": 4,
}

# ASCII rotation arrows (clockwise): rot 0/90/180/270.
ROT_ARROWS = {0: "^", 90: ">", 180: "v", 270: "<"}

# 大户型弹性放大顺序（先放开起居类，厨房/卫生间最后）。
_CAP_FLEX_ORDER = (
    "living",
    "bedroom",
    "dining",
    "corridor",
    "balcony",
    "study",
    "tea",
    "bathroom",
    "kitchen",
)


def caps_for_area(area: int, headroom: int = 6) -> dict[str, int]:
    """Per-tile count caps scaled to the grid area.

    MAX_TILE_COUNTS describes a ~12-cell home; larger grids must be allowed
    proportionally more rooms, otherwise the caps sum below the cell count
    and WFC has no feasible solution. ``headroom`` keeps slack so the caps
    are upper bounds, not mandatory targets.
    """
    caps = dict(MAX_TILE_COUNTS)
    k = 0
    while sum(caps.values()) < area + headroom:
        caps[_CAP_FLEX_ORDER[k % len(_CAP_FLEX_ORDER)]] += 1
        k += 1
    return caps


def build_variants() -> tuple[list, list[float]]:
    """Expand the catalog into (pattern, rotation) variants + aligned weights."""
    variants = expand_variants(
        {name: spec.all_patterns() for name, spec in TILES.items()}
    )
    weights = [TILES[v.tile].weight for v in variants]
    return variants, weights


# ---------------------------------------------------------------------------
# Layout representation + global validation (全局规则)
# ---------------------------------------------------------------------------

LAYOUT_VERSION = 2

# 结构体尺寸常量（隔墙厚 / 墙高 / 阳台栏杆高）。
WALL_THICKNESS = 0.12
WALL_HEIGHT = 2.8
RAIL_HEIGHT = 1.1

# 进门动线：玄关的非门边只能通向这些模块。
ENTRY_FLOW_OK = frozenset({"living", "corridor", "dining"})
# 阳台动线：阳台必须与这些居室模块相邻。
BALCONY_HOSTS = frozenset({"living", "bedroom", "dining", "tea"})


@dataclass
class Layout:
    """A collapsed grid: tile + wall pattern + rotation per cell.

    Cell sizes: uniform ``tile_size`` squares by default. When
    ``col_widths``/``row_heights`` are given, column i is ``col_widths[i]``
    wide (x axis) and row j is ``row_heights[j]`` deep (y axis) — 行列不等宽，
    每格仍是矩形且整行/整列对齐。The whole plan is centered on the origin.
    """

    width: int
    height: int
    tile_size: float = 3.2
    seed: Optional[int] = None
    cells: list[dict] = field(default_factory=list)  # {"i","j","tile","pattern","rot"}
    col_widths: Optional[list[float]] = None
    row_heights: Optional[list[float]] = None

    # -- cell access ---------------------------------------------------------

    def cell_at(self, i: int, j: int) -> Optional[dict]:
        if 0 <= i < self.width and 0 <= j < self.height:
            return self.cells[j * self.width + i]
        return None

    def tile_at(self, i: int, j: int) -> Optional[str]:
        cell = self.cell_at(i, j)
        return cell["tile"] if cell else None

    def sockets_at(self, i: int, j: int) -> tuple[str, str, str, str]:
        """World-frame N/E/S/W sockets of cell (i, j)."""
        cell = self.cells[j * self.width + i]
        spec = TILES[cell["tile"]]
        pattern = spec.all_patterns()[cell.get("pattern", 0)]
        return rotated_sockets(pattern, cell["rot"])

    # -- variable-size geometry (可变格子几何) ---------------------------------

    def col_w(self, i: int) -> float:
        """Width of column i (x direction)."""
        return self.col_widths[i] if self.col_widths is not None else self.tile_size

    def row_h(self, j: int) -> float:
        """Depth of row j (y direction)."""
        return self.row_heights[j] if self.row_heights is not None else self.tile_size

    @property
    def total_width(self) -> float:
        return (
            sum(self.col_widths)
            if self.col_widths is not None
            else self.width * self.tile_size
        )

    @property
    def total_depth(self) -> float:
        return (
            sum(self.row_heights)
            if self.row_heights is not None
            else self.height * self.tile_size
        )

    def col_x0(self, i: int) -> float:
        """World x of the left edge of column i."""
        if self.col_widths is None:
            return (i - self.width / 2) * self.tile_size
        return -self.total_width / 2 + sum(self.col_widths[:i])

    def row_y0(self, j: int) -> float:
        """World y of the south edge of row j."""
        if self.row_heights is None:
            return (j - self.height / 2) * self.tile_size
        return -self.total_depth / 2 + sum(self.row_heights[:j])

    def cell_span(
        self, i: int, j: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x0, x1), (y0, y1)) world extents of cell (i, j)."""
        return (
            (self.col_x0(i), self.col_x0(i) + self.col_w(i)),
            (self.row_y0(j), self.row_y0(j) + self.row_h(j)),
        )

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        """World XY of a cell center; the whole grid is centered on the origin."""
        (x0, x1), (y0, y1) = self.cell_span(i, j)
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def cell_half(self, i: int, j: int) -> tuple[float, float]:
        """Half extents (hx, hy) of cell (i, j)."""
        return self.col_w(i) / 2, self.row_h(j) / 2

    def edge_segment(
        self, i: int, j: int, d: int, thickness: float = WALL_THICKNESS
    ) -> tuple[tuple[float, float, float], tuple[float, float]]:
        """Center position and (x, y) extents of the wall segment on edge d.

        The segment sits on the boundary between cell (i, j) and its neighbour
        in direction d (or on the exterior facade), spanning the cell edge and
        overlapping ``thickness`` at both ends so corners stay closed.
        """
        (x0, x1), (y0, y1) = self.cell_span(i, j)
        if d == 0:  # N
            return ((x0 + x1) / 2, y1, 0.0), (x1 - x0 + thickness, thickness)
        if d == 1:  # E
            return (x1, (y0 + y1) / 2, 0.0), (thickness, y1 - y0 + thickness)
        if d == 2:  # S
            return ((x0 + x1) / 2, y0, 0.0), (x1 - x0 + thickness, thickness)
        return (x0, (y0 + y1) / 2, 0.0), (thickness, y1 - y0 + thickness)  # W

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        data = {
            "version": LAYOUT_VERSION,
            "generator": "wfc_scenes",
            "grid": [self.width, self.height],
            "tile_size": self.tile_size,
            "seed": self.seed,
            "cells": list(self.cells),
        }
        if self.col_widths is not None:
            data["col_widths"] = list(self.col_widths)
        if self.row_heights is not None:
            data["row_heights"] = list(self.row_heights)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        if data.get("generator") != "wfc_scenes":
            raise ValueError("not a wfc_scenes layout file")
        width, height = data["grid"]
        col_widths = data.get("col_widths")
        row_heights = data.get("row_heights")
        for name, values, expect in (
            ("col_widths", col_widths, width),
            ("row_heights", row_heights, height),
        ):
            if values is not None and len(values) != expect:
                raise ValueError(f"{name} has {len(values)} entries, expected {expect}")
        layout = cls(
            width=width,
            height=height,
            tile_size=data.get("tile_size", 3.2),
            seed=data.get("seed"),
            col_widths=list(col_widths) if col_widths is not None else None,
            row_heights=list(row_heights) if row_heights is not None else None,
        )
        layout.cells = [
            {
                "i": c["i"],
                "j": c["j"],
                "tile": c["tile"],
                "pattern": c.get("pattern", 0),
                "rot": c["rot"],
            }
            for c in data["cells"]
        ]
        expected = width * height
        if len(layout.cells) != expected:
            raise ValueError(f"expected {expected} cells, got {len(layout.cells)}")
        return layout

    def ascii_map(self) -> str:
        """Top-down ASCII map (row j = 0 at the top = north)."""
        rows = []
        for j in range(self.height - 1, -1, -1):
            row = []
            for i in range(self.width):
                cell = self.cells[j * self.width + i]
                spec = TILES[cell["tile"]]
                row.append(f"{spec.ascii_char}{ROT_ARROWS[cell['rot']]}")
            rows.append(" ".join(row))
        return "\n".join(rows)


def count_tiles(layout: Layout) -> dict[str, int]:
    """Count placements per tile name ({"tile": count})."""
    counts: dict[str, int] = {}
    for cell in layout.cells:
        counts[cell["tile"]] = counts.get(cell["tile"], 0) + 1
    return counts


def validate_layout(
    layout: Layout,
    require_entry: bool = True,
    require_living: bool = True,
    require_balcony: bool = True,
    min_bedrooms: int = 1,
    max_counts: Optional[dict[str, int]] = None,
) -> list[str]:
    """Check the global Chinese-home rules that edge sockets cannot express.

    Returns a list of human-readable violations (empty list = valid).
    """
    issues: list[str] = []
    counts = count_tiles(layout)
    area = layout.width * layout.height
    limits = caps_for_area(area) if max_counts is None else max_counts

    if require_entry and counts.get("entry", 0) != 1:
        issues.append(f"玄关必须恰好 1 个，实际 {counts.get('entry', 0)} 个")
    if require_living and counts.get("living", 0) < 1:
        issues.append("缺少客厅")
    if counts.get("kitchen", 0) < 1:
        issues.append("缺少厨房")
    if require_balcony and area >= 8 and counts.get("balcony", 0) < 1:
        issues.append("缺少阳台")
    if counts.get("bedroom", 0) < min_bedrooms:
        issues.append(f"卧室至少 {min_bedrooms} 间，实际 {counts.get('bedroom', 0)} 间")
    for tile, cap in limits.items():
        if counts.get(tile, 0) > cap:
            issues.append(
                f"{TILES[tile].zh}最多 {cap} 个，实际 {counts.get(tile, 0)} 个"
            )

    for cell in layout.cells:
        i, j, tile = cell["i"], cell["j"], cell["tile"]
        min_edge = TILES[tile].min_edge
        if min(layout.col_w(i), layout.row_h(j)) + 1e-6 < min_edge:
            issues.append(
                f"{TILES[tile].zh}({i},{j})所在格子边长不足"
                f"{min_edge}m（实际 "
                f"{min(layout.col_w(i), layout.row_h(j)):.2f}m）"
            )
        sockets = layout.sockets_at(i, j)
        for d, (dx, dy) in enumerate(DIRS):
            ni, nj = i + dx, j + dy
            inside = 0 <= ni < layout.width and 0 <= nj < layout.height
            neigh = layout.tile_at(ni, nj) if inside else None
            neigh_sockets = layout.sockets_at(ni, nj) if inside else None

            if not inside:
                if not exterior_ok(sockets[d]):
                    issues.append(f"{TILES[tile].zh}({i},{j})的开放边朝向了室外")
                continue

            if not compatible(sockets[d], neigh_sockets[(d + 2) % 4]):
                issues.append(
                    f"{TILES[tile].zh}({i},{j})与"
                    f"{TILES[neigh].zh}({ni},{nj})邻接不合法"
                )

        if tile == "entry":
            flow = _neighbor_tiles(layout, i, j, skip_socket=SOCKET_DOOR)
            if flow and not flow & ENTRY_FLOW_OK:
                issues.append(f"玄关({i},{j})的进门动线不通向客厅/过道/餐厅")
        if tile == "balcony":
            hosts = _neighbor_tiles(layout, i, j, skip_socket=SOCKET_RAIL)
            if hosts and not hosts & BALCONY_HOSTS:
                issues.append(f"阳台({i},{j})未与客厅/卧室/餐厅/茶室相邻")
        if tile == "kitchen":
            # 餐厨相邻（现代户型客餐厨一体也可）：开口侧必须邻餐厅或客厅。
            hosts = _neighbor_tiles(layout, i, j, skip_socket=SOCKET_COUNTER)
            if not hosts & {"dining", "living"}:
                issues.append(f"厨房({i},{j})未与餐厅/客厅相邻")

    return issues


def _neighbor_tiles(
    layout: Layout, i: int, j: int, skip_socket: Optional[str] = None
) -> set[str]:
    """Tiles of interior neighbours, optionally skipping one socket side."""
    result: set[str] = set()
    sockets = layout.sockets_at(i, j)
    for d, (dx, dy) in enumerate(DIRS):
        if skip_socket is not None and sockets[d] == skip_socket:
            continue
        ni, nj = i + dx, j + dy
        tile = layout.tile_at(ni, nj)
        if tile is not None:
            result.add(tile)
    return result


# ---------------------------------------------------------------------------
# Partition rules shared by the scene builder
# ---------------------------------------------------------------------------


def needs_partition(socket_a: str, socket_b: str) -> bool:
    """Whether the shared edge should render as an interior partition wall.

    Rules: (wall, wall) 两个模块背靠背 — 隔墙；(counter, wall) 台面顶到隔墙。
    (counter, counter) 是同一间厨房的连续台面 — 不砌墙。
    """
    pair = frozenset({socket_a, socket_b})
    return pair in (
        frozenset({SOCKET_WALL, SOCKET_WALL}),
        frozenset({SOCKET_COUNTER, SOCKET_WALL}),
    )


__all__ = [
    "ALL_SOCKETS",
    "caps_for_area",
    "BALCONY_HOSTS",
    "DIRS",
    "ENTRY_FLOW_OK",
    "EXTERIOR_OK",
    "INTERIOR_COMPAT",
    "LAYOUT_VERSION",
    "MAX_TILE_COUNTS",
    "RAIL_HEIGHT",
    "SOCKET_COUNTER",
    "SOCKET_DOOR",
    "SOCKET_OPEN",
    "SOCKET_RAIL",
    "SOCKET_WALL",
    "TILES",
    "WALL_HEIGHT",
    "WALL_THICKNESS",
    "Layout",
    "TileSpec",
    "ROT_ARROWS",
    "build_variants",
    "compatible",
    "count_tiles",
    "exterior_ok",
    "needs_partition",
    "rotated_sockets",
    "validate_layout",
]
