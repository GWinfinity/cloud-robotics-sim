# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""WFC-driven Chinese home scene generation.

``ChineseHomeScene`` collapses a floor-plan grid with Wave Function Collapse
under the Chinese-home rules (:mod:`core.rules_cn`), then turns the collapsed
layout into:

- a JSON layout file (replayable, seed-stamped),
- an ASCII top-down map (quick rule inspection),
- a top-down SVG plan (visual rule check, no Genesis required),
- a GLB mesh of the whole home (portable preview / static stage),
- a live Genesis scene (walls, floors, furniture as rigid entities).

Typical use::

    from plugins.scenes.wfc_scenes.scene import ChineseHomeScene

    home = ChineseHomeScene(grid=(4, 3), seed=42)
    print(home.ascii_map())          # 动线一眼可查
    home.export_layout("home.json")
    home.export_glb("home.glb")
    scene = home.build(headless=True, device="cpu")

Integration hook for other plugins (DreamDojo / maniskill): call
:func:`populate_genesis_scene` after creating a ``gs.Scene`` and before
``scene.build()``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Union

try:  # Relative import inside the installed plugin package.
    from .assets.furniture_cn import (
        GenesisBackend,
        Prim,
        PrimBuilder,
        TrimeshBackend,
        build_tile_prims,
    )
    from .core.rules_cn import (
        RAIL_HEIGHT,
        ROT_ARROWS,
        SOCKET_DOOR,
        SOCKET_RAIL,
        TILES,
        WALL_HEIGHT,
        Layout,
        build_variants,
        caps_for_area,
        compatible,
        exterior_ok,
        needs_partition,
        validate_layout,
    )
    from .core.wfc import WFCContradictionError, collapse
except ImportError:  # Fallback when the plugin dir itself is on sys.path.
    from assets.furniture_cn import (  # type: ignore[no-redef]
        GenesisBackend,
        Prim,
        PrimBuilder,
        TrimeshBackend,
        build_tile_prims,
    )
    from core.rules_cn import (  # type: ignore[no-redef]
        RAIL_HEIGHT,
        ROT_ARROWS,
        SOCKET_DOOR,
        SOCKET_RAIL,
        TILES,
        WALL_HEIGHT,
        Layout,
        build_variants,
        caps_for_area,
        compatible,
        exterior_ok,
        needs_partition,
        validate_layout,
    )
    from core.wfc import WFCContradictionError, collapse  # type: ignore[no-redef]

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    gs = None
    HAS_GENESIS = False


EXTERIOR_WALL_COLOR = (0.90, 0.88, 0.84)  # 外墙乳白
PARTITION_COLOR = (0.93, 0.91, 0.87)  # 内隔墙
FLOOR_SLAB_COLOR = (0.72, 0.71, 0.70)  # 楼板
DOOR_HEIGHT = 2.05
DOOR_WIDTH = 1.0
RAIL_SILL_HEIGHT = 0.9

LayoutLike = Union["Layout", str, Path, dict]


# ---------------------------------------------------------------------------
# Structure prims (楼板 / 地板 / 外墙 / 门洞 / 栏杆 / 隔墙)
# ---------------------------------------------------------------------------


def build_structure_prims(
    layout: Layout,
    wall_height: float = WALL_HEIGHT,
) -> list[Prim]:
    """Build the shell of the apartment as prims (all fixed)."""
    pb = PrimBuilder()
    w, h = layout.width, layout.height

    # 楼板 + 各房间地板
    pb.box(
        (layout.total_width + 0.4, layout.total_depth + 0.4, 0.12),
        (0.0, 0.0, -0.06),
        FLOOR_SLAB_COLOR,
        "floor_slab",
        rho=800,
        friction=0.8,
        fixed=True,
    )
    for cell in layout.cells:
        i, j = cell["i"], cell["j"]
        cx, cy = layout.cell_center(i, j)
        color = TILES[cell["tile"]].floor_color
        pb.box(
            (layout.col_w(i) - 0.08, layout.row_h(j) - 0.08, 0.014),
            (cx, cy, 0.007),
            color,
            f"floor_{cell['tile']}",
            rho=400,
            friction=0.8,
            fixed=True,
        )

    # 外墙：整墙 / 入户门洞 / 阳台栏杆
    for cell in layout.cells:
        i, j = cell["i"], cell["j"]
        sockets = layout.sockets_at(i, j)
        for d in range(4):
            ni, nj = i + (0, 1, 0, -1)[d], j + (1, 0, -1, 0)[d]
            if 0 <= ni < w and 0 <= nj < h:
                continue  # 内部边在外墙循环里跳过
            (sx, sy, _), (ex_len, ey_len) = layout.edge_segment(i, j, d)
            name = f"ext_wall_{i}_{j}_{d}"
            if sockets[d] == SOCKET_DOOR:
                _add_wall_with_door(pb, sx, sy, ex_len, ey_len, wall_height, name)
            elif sockets[d] == SOCKET_RAIL:
                _add_railing(pb, sx, sy, ex_len, ey_len, name)
            else:
                pb.box(
                    (ex_len, ey_len, wall_height),
                    (sx, sy, wall_height / 2),
                    EXTERIOR_WALL_COLOR,
                    name,
                    rho=600,
                    friction=0.6,
                    fixed=True,
                )

    # 内隔墙：只需遍历每格的 N/E 两条边，即可无重复覆盖全部内部边
    for cell in layout.cells:
        i, j = cell["i"], cell["j"]
        sockets = layout.sockets_at(i, j)
        for d in (0, 1):  # N, E
            ni, nj = i + (0, 1)[d], j + (1, 0)[d]
            if not (0 <= ni < w and 0 <= nj < h):
                continue
            neigh_sockets = layout.sockets_at(ni, nj)
            if needs_partition(sockets[d], neigh_sockets[(d + 2) % 4]):
                (sx, sy, _), (ex_len, ey_len) = layout.edge_segment(i, j, d)
                pb.box(
                    (ex_len, ey_len, wall_height),
                    (sx, sy, wall_height / 2),
                    PARTITION_COLOR,
                    f"partition_{i}_{j}_{d}",
                    rho=600,
                    friction=0.6,
                    fixed=True,
                )
    return pb.primljs


def _add_wall_with_door(
    pb: PrimBuilder,
    sx: float,
    sy: float,
    ex_len: float,
    ey_len: float,
    wall_height: float,
    name: str,
) -> None:
    """Exterior wall segment with a DOOR_WIDTH doorway and lintel."""
    along_x = ex_len > ey_len  # N/S 边：墙沿 x 方向
    if along_x:
        post_len = (ex_len - DOOR_WIDTH) / 2
        sizes = [(post_len, ey_len, DOOR_HEIGHT), (post_len, ey_len, DOOR_HEIGHT)]
        posts = [
            (sx - DOOR_WIDTH / 2 - post_len / 2, sy),
            (sx + DOOR_WIDTH / 2 + post_len / 2, sy),
        ]
        lintel_size = (DOOR_WIDTH, ey_len, wall_height - DOOR_HEIGHT)
        lintel_pos = (sx, sy, DOOR_HEIGHT + (wall_height - DOOR_HEIGHT) / 2)
    else:
        post_len = (ey_len - DOOR_WIDTH) / 2
        sizes = [(ex_len, post_len, DOOR_HEIGHT), (ex_len, post_len, DOOR_HEIGHT)]
        posts = [
            (sx, sy - DOOR_WIDTH / 2 - post_len / 2),
            (sx, sy + DOOR_WIDTH / 2 + post_len / 2),
        ]
        lintel_size = (ex_len, DOOR_WIDTH, wall_height - DOOR_HEIGHT)
        lintel_pos = (sx, sy, DOOR_HEIGHT + (wall_height - DOOR_HEIGHT) / 2)
    for k, (pos, size) in enumerate(zip(posts, sizes)):
        pb.box(
            size,
            (pos[0], pos[1], DOOR_HEIGHT / 2),
            EXTERIOR_WALL_COLOR,
            f"{name}_post{k}",
            rho=600,
            friction=0.6,
            fixed=True,
        )
    pb.box(
        lintel_size,
        lintel_pos,
        EXTERIOR_WALL_COLOR,
        f"{name}_lintel",
        rho=600,
        friction=0.6,
        fixed=True,
    )


def _add_railing(
    pb: PrimBuilder,
    sx: float,
    sy: float,
    ex_len: float,
    ey_len: float,
    name: str,
) -> None:
    """Balcony railing: concrete sill + steel top bar."""
    pb.box(
        (ex_len, ey_len, RAIL_SILL_HEIGHT),
        (sx, sy, RAIL_SILL_HEIGHT / 2),
        (0.80, 0.79, 0.77),
        f"{name}_sill",
        rho=600,
        friction=0.6,
        fixed=True,
    )
    bar_height = RAIL_HEIGHT - RAIL_SILL_HEIGHT
    pb.box(
        (
            ex_len * 0.995 if ex_len > ey_len else 0.06,
            ey_len * 0.995 if ey_len > ex_len else 0.06,
            bar_height,
        ),
        (sx, sy, RAIL_SILL_HEIGHT + bar_height / 2),
        (0.65, 0.67, 0.70),
        f"{name}_glass",
        rho=250,
        friction=0.4,
        fixed=True,
    )
    pb.box(
        (
            ex_len if ex_len > ey_len else 0.08,
            ey_len if ey_len > ex_len else 0.08,
            0.05,
        ),
        (sx, sy, RAIL_HEIGHT + 0.025),
        (0.68, 0.70, 0.72),
        f"{name}_bar",
        rho=300,
        fixed=True,
    )


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------


class ChineseHomeScene:
    """A procedurally generated Chinese apartment (WFC floor plan + furniture).

    Args:
        grid: ``(width, height)`` in tiles.
        tile_size: Tile side length in meters.
        seed: Base seed; ``None`` picks a random one (recorded in the layout).
        festival: Add Spring Festival decorations (灯笼/福字).
        wall_height: Wall height in meters.
        require_balcony / require_living / require_entry: Global rule switches.
        min_bedrooms: Minimum bedrooms; ``None`` = 1 for grids of 8+ cells.
        layout: Optional preset layout (Layout, dict, or JSON path). When
            given, WFC is skipped and the layout is replayed as-is.
        max_attempts: WFC restarts allowed until global rules pass.
        variable_size: 行列不等宽（方案 2）——每列宽度/每行深度独立随机抽取，
            并按入住模块的最小边长（``TileSpec.min_edge``）抬升。
        min_cell / max_cell: 可变模式下列宽/行高的随机抽取范围（米）。
    """

    def __init__(
        self,
        grid: tuple[int, int] = (4, 3),
        tile_size: float = 3.2,
        seed: Optional[int] = None,
        festival: bool = False,
        wall_height: float = WALL_HEIGHT,
        require_balcony: bool = True,
        require_living: bool = True,
        require_entry: bool = True,
        min_bedrooms: Optional[int] = None,
        layout: Optional[LayoutLike] = None,
        max_attempts: Optional[int] = None,
        variable_size: bool = False,
        min_cell: float = 2.4,
        max_cell: float = 4.2,
    ):
        self.grid = tuple(grid)
        self.tile_size = tile_size
        self.seed = seed
        self.festival = festival
        self.variable_size = variable_size
        self.min_cell = min_cell
        self.max_cell = max_cell
        self.wall_height = wall_height
        self.require_balcony = require_balcony
        self.require_living = require_living
        self.require_entry = require_entry
        self.min_bedrooms = min_bedrooms
        self.max_attempts = max_attempts
        self._preset_layout: Optional[Layout] = None
        if layout is not None:
            self._preset_layout = load_layout(layout)
        self._layout: Optional[Layout] = None
        self.scene: Optional[Any] = None
        self.camera: Optional[Any] = None

    # -- layout --------------------------------------------------------------

    def generate_layout(self) -> "Layout":
        """Run WFC (with global-rule retries) and cache the layout."""
        if self._preset_layout is not None:
            self._layout = self._preset_layout
            return self._layout

        width, height = self.grid
        area = width * height
        min_bedrooms = (
            self.min_bedrooms
            if self.min_bedrooms is not None
            else (1 if area >= 8 else 0)
        )
        base_seed = (
            self.seed
            if self.seed is not None
            else random.SystemRandom().randrange(1 << 30)
        )
        variants, weights = build_variants()
        by_tile: dict[str, set[int]] = {}
        for idx, variant in enumerate(variants):
            by_tile.setdefault(variant.tile, set()).add(idx)

        # 数量上限随面积弹性放大（与校验器默认一致），否则大户型无解。
        caps = caps_for_area(area)

        def cell_ij(cell: int) -> tuple[int, int]:
            return cell % width, cell // width

        def neighbors_inside(cell: int) -> list[tuple[int, int]]:
            """(direction, neighbour cell) pairs for the inside neighbours."""
            i, j = cell_ij(cell)
            out = []
            for d, (dx, dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
                ni, nj = i + dx, j + dy
                if 0 <= ni < width and 0 <= nj < height:
                    out.append((d, nj * width + ni))
            return out

        def exterior_dirs(cell: int) -> list[int]:
            i, j = cell_ij(cell)
            return [
                d
                for d, (dx, dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0)))
                if not (0 <= i + dx < width and 0 <= j + dy < height)
            ]

        def exterior_compatible(cell: int, var_ids: set[int]) -> list[int]:
            return [
                v
                for v in sorted(var_ids)
                if all(exterior_ok(variants[v].sockets[d]) for d in exterior_dirs(cell))
            ]

        # 非角落的边缘格：玄关门/阳台栏杆只能在这些格子的外墙上。
        edge_cells = [
            j * width + i
            for j in range(height)
            for i in range(width)
            if (i in (0, width - 1) or j in (0, height - 1))
            and not (i in (0, width - 1) and j in (0, height - 1))
        ]
        all_cells = list(range(width * height))
        need_balcony = self.require_balcony and area >= 8

        def build_presets(rng: random.Random) -> dict[int, set[int]]:
            """Seed cells so the global 动线/存在 rules hold by construction.

            Rules share cells through set intersection: a cell constrained by
            two rules must satisfy both, which keeps small grids feasible.
            """
            presets: dict[int, set[int]] = {}
            state = {"conflict": False}

            def constrain(cell: int, var_ids: set[int]) -> None:
                merged = presets[cell] & var_ids if cell in presets else set(var_ids)
                if not merged:
                    state["conflict"] = True
                presets[cell] = merged

            # 玄关：随机边缘格 + 随机朝向；其开口侧邻格强制为客/餐/过道。
            if edge_cells:
                entry_cell = rng.choice(edge_cells)
                entry_ok = exterior_compatible(entry_cell, by_tile["entry"])
                if not entry_ok:
                    return {}
                entry_variant = rng.choice(entry_ok)
                constrain(entry_cell, {entry_variant})
                open_neighbors = [
                    n
                    for d, n in neighbors_inside(entry_cell)
                    if variants[entry_variant].sockets[d] == "open"
                ]
                if open_neighbors:
                    constrain(
                        rng.choice(open_neighbors),
                        by_tile["living"] | by_tile["corridor"] | by_tile["dining"],
                    )
                if state["conflict"]:
                    return {}

                # 阳台：另一边缘格贴外墙，至少一个内侧邻格为居室。
                if need_balcony:
                    balcony_free = [c for c in edge_cells if c not in presets]
                    rng.shuffle(balcony_free)
                    for balcony_cell in balcony_free:
                        balcony_ok = exterior_compatible(
                            balcony_cell, by_tile["balcony"]
                        )
                        if not balcony_ok:
                            continue
                        hosts = [
                            n
                            for _, n in neighbors_inside(balcony_cell)
                            if n != entry_cell
                        ]
                        if not hosts:
                            continue
                        snapshot = {c: set(vs) for c, vs in presets.items()}
                        constrain(balcony_cell, {rng.choice(balcony_ok)})
                        constrain(
                            rng.choice(hosts),
                            by_tile["living"]
                            | by_tile["bedroom"]
                            | by_tile["dining"]
                            | by_tile["tea"],
                        )
                        if not state["conflict"]:
                            break
                        # 该阳台格走不通，回滚后尝试下一个候选格。
                        presets.clear()
                        presets.update(snapshot)
                        state["conflict"] = False

            # 厨房：随机空闲格，开口正对一格强制为餐/客（餐厨相邻）；
            # 台面的另外三面必须顶墙/台面，避开预设了开放边的邻格。
            kitchen_free = [c for c in all_cells if c not in presets]
            rng.shuffle(kitchen_free)
            for kitchen_cell in kitchen_free:
                nbrs = neighbors_inside(kitchen_cell)

                def counters_ok(variant: int, dining_cell: int) -> bool:
                    """厨房的台面边邻格必须能提供墙/台面（开口对侧除外）。"""
                    for d_c, m in nbrs:
                        if variants[variant].sockets[d_c] != "counter":
                            continue
                        if m == dining_cell or m not in presets:
                            continue
                        opposite = (d_c + 2) % 4
                        if not any(
                            variants[pv].sockets[opposite] in ("wall", "counter")
                            for pv in presets[m]
                        ):
                            return False
                    return True

                pair = [
                    (v, n)
                    for v in sorted(by_tile["kitchen"])
                    for d, n in nbrs
                    if variants[v].sockets[d] == "open" and counters_ok(v, n)
                ]
                if pair:
                    kitchen_variant, dining_cell = rng.choice(pair)
                    constrain(kitchen_cell, {kitchen_variant})
                    constrain(dining_cell, by_tile["living"] | by_tile["dining"])
                break

            # 卧室：随机空闲格保证至少一间（小户型不强制）。
            if min_bedrooms >= 1:
                bedroom_free = [c for c in all_cells if c not in presets]
                if bedroom_free:
                    constrain(rng.choice(bedroom_free), by_tile["bedroom"])
            if state["conflict"]:
                return {}
            return presets

        max_attempts = self.max_attempts or max(64, area * 4)
        for attempt in range(max_attempts):
            run_seed = base_seed * 1000 + attempt
            rng = random.Random(run_seed)
            presets = build_presets(rng)

            # 方案 2：行列不等宽——每轮随机抽取列宽/行高，坍缩后再按入住模块
            # 的最小边长抬升（保证厅大卧小卫小的同时不浪费尝试次数）。
            col_widths = row_heights = None
            if self.variable_size:
                col_widths = [
                    round(rng.uniform(self.min_cell, self.max_cell), 3)
                    for _ in range(width)
                ]
                row_heights = [
                    round(rng.uniform(self.min_cell, self.max_cell), 3)
                    for _ in range(height)
                ]

            try:
                idx_grid = collapse(
                    width,
                    height,
                    variants,
                    weights,
                    compatible,
                    exterior_ok,
                    rng,
                    presets=presets,
                    max_counts=caps,
                )
            except WFCContradictionError:
                continue
            cells = []
            for j in range(height):
                for i in range(width):
                    variant = variants[idx_grid[j][i]]
                    cells.append(
                        {
                            "i": i,
                            "j": j,
                            "tile": variant.tile,
                            "pattern": variant.pattern,
                            "rot": variant.rot,
                        }
                    )
            if col_widths is not None:
                # 列宽 ≥ 该列所有格子的模块最小边长；行高同理。
                for i in range(width):
                    need = max(
                        TILES[variants[idx_grid[j][i]].tile].min_edge
                        for j in range(height)
                    )
                    col_widths[i] = round(max(col_widths[i], need), 3)
                for j in range(height):
                    need = max(
                        TILES[variants[idx_grid[j][i]].tile].min_edge
                        for i in range(width)
                    )
                    row_heights[j] = round(max(row_heights[j], need), 3)
            layout = Layout(
                width=width,
                height=height,
                tile_size=self.tile_size,
                seed=run_seed,
                cells=cells,
                col_widths=col_widths,
                row_heights=row_heights,
            )
            issues = validate_layout(
                layout,
                require_entry=self.require_entry,
                require_living=self.require_living,
                require_balcony=self.require_balcony,
                min_bedrooms=min_bedrooms,
                max_counts=caps,
            )
            if not issues:
                self._layout = layout
                return layout

        raise RuntimeError(
            f"WFC 在 {max_attempts} 次尝试内未能生成满足中式家居规则的布局"
            f"(grid={self.grid}, seed={self.seed})"
        )

    @property
    def layout(self) -> "Layout":
        """The collapsed layout (generated on first access)."""
        if self._layout is None:
            self.generate_layout()
        return self._layout

    def ascii_map(self) -> str:
        """Top-down ASCII map (top row = north)."""
        return self.layout.ascii_map()

    # -- prims ----------------------------------------------------------------

    def structure_prims(self) -> list[Prim]:
        return build_structure_prims(self.layout, self.wall_height)

    def furniture_prims(self) -> list[Prim]:
        """Furniture prims for every tile (deterministic per layout seed)."""
        layout = self.layout
        prims: list[Prim] = []
        for k, cell in enumerate(layout.cells):
            i, j = cell["i"], cell["j"]
            center = layout.cell_center(i, j)
            rng = random.Random((layout.seed or 0) * 7919 + k)
            prims.extend(
                build_tile_prims(
                    cell["tile"],
                    center,
                    cell["rot"],
                    rng,
                    self.festival,
                    pattern=cell.get("pattern", 0),
                    half=layout.cell_half(i, j),
                )
            )
        return prims

    # -- offline artifacts ------------------------------------------------------

    def build_trimesh(self) -> Any:
        """Whole home as a ``trimesh.Scene`` (no Genesis needed)."""
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh is required for offline scene building")
        backend = TrimeshBackend()
        for prim in self.structure_prims() + self.furniture_prims():
            backend.add(prim)
        return backend.scene

    def export_glb(self, path: Union[str, Path]) -> Path:
        """Export the whole home as GLB (portable preview / static stage)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.build_trimesh().export(str(path))
        return path

    def export_layout(self, path: Union[str, Path]) -> Path:
        """Export the collapsed layout as replayable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.layout.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def topdown_svg(self, path: Union[str, Path]) -> Path:
        """Top-down floor plan as SVG (visual rule check without Genesis)."""
        layout = self.layout
        margin = 0.6
        total_w = layout.total_width + 2 * margin
        total_h = layout.total_depth + 2 * margin
        scale = 100.0  # px per meter

        def px(x: float) -> float:
            return (x + total_w / 2) * scale

        def py(y: float) -> float:
            return (total_h / 2 - y) * scale  # SVG y 轴向下，北在上方

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w * scale:.0f}" height="{total_h * scale:.0f}" '
            f'viewBox="0 0 {total_w * scale:.0f} {total_h * scale:.0f}">',
            '<rect width="100%" height="100%" fill="#f5f2ec"/>',
        ]
        # 房间地板
        for cell in layout.cells:
            (x0, x1), (y0, y1) = layout.cell_span(cell["i"], cell["j"])
            color = TILES[cell["tile"]].floor_color
            rgb = tuple(int(c * 255) for c in color[:3])
            parts.append(
                f'<rect x="{px(x0):.1f}" y="{py(y1):.1f}" '
                f'width="{(x1 - x0) * scale:.1f}" height="{(y1 - y0) * scale:.1f}" '
                f'fill="rgb{rgb}" stroke="#ffffff" stroke-width="2"/>'
            )
        # 家具投影（box→旋转矩形，cylinder/sphere→圆）
        for prim in self.furniture_prims():
            r, g, b = (int(c * 255) for c in prim.color[:3])
            fill = f"rgb({r},{g},{b})"
            if prim.kind == "box":
                hx, hy = prim.size[0] / 2, prim.size[1] / 2
                rad = math.radians(prim.euler[2])
                cos_, sin_ = math.cos(rad), math.sin(rad)
                pts = []
                for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                    wx = prim.pos[0] + dx * cos_ + dy * sin_
                    wy = prim.pos[1] - dx * sin_ + dy * cos_
                    pts.append(f"{px(wx):.1f},{py(wy):.1f}")
                parts.append(
                    f'<polygon points="{" ".join(pts)}" fill="{fill}" '
                    f'fill-opacity="0.92"/>'
                )
            else:
                radius = prim.size[0]
                parts.append(
                    f'<circle cx="{px(prim.pos[0]):.1f}" cy="{py(prim.pos[1]):.1f}" '
                    f'r="{radius * scale:.1f}" fill="{fill}" fill-opacity="0.92"/>'
                )
        # 模块标签
        for cell in layout.cells:
            i, j = cell["i"], cell["j"]
            cx, cy = layout.cell_center(i, j)
            hy = layout.row_h(j) / 2
            spec = TILES[cell["tile"]]
            arrow = ROT_ARROWS[cell["rot"]]
            parts.append(
                f'<text x="{px(cx):.1f}" y="{py(cy - hy * 0.32):.1f}" '
                f'font-size="{0.26 * scale:.0f}" text-anchor="middle" '
                f'fill="#333" font-family="sans-serif">{spec.zh}</text>'
            )
            parts.append(
                f'<text x="{px(cx):.1f}" y="{py(cy - hy * 0.46):.1f}" '
                f'font-size="{0.18 * scale:.0f}" text-anchor="middle" '
                f'fill="#777" font-family="sans-serif">{spec.ascii_char}{arrow}</text>'
            )
        parts.append("</svg>")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    # -- Genesis -----------------------------------------------------------------

    def build(self, headless: bool = True, device: str = "cpu") -> Optional[Any]:
        """Build the live Genesis scene (walls + furniture as rigid entities).

        Returns the ``gs.Scene`` or ``None`` when Genesis is unavailable.
        """
        if not HAS_GENESIS:
            print("Warning: Genesis not available; use build_trimesh()/export_glb().")
            return None
        layout = self.layout
        try:
            from cloud_robotics_sim.utils.genesis_compat import (
                genesis_init,
                get_genesis_lights,
            )
        except ImportError:
            genesis_init = None
            get_genesis_lights = None

        if genesis_init is not None:
            genesis_init(device=device)
        else:
            gs.init(backend=gs.cpu if device == "cpu" else gs.gpu)

        total_w, total_h = layout.total_width, layout.total_depth
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0.0, -total_h * 0.65, total_h * 1.1),
                camera_lookat=(0.0, 0.0, 0.6),
                camera_up=(0.0, 0.0, 1.0),
            ),
            sim_options=gs.options.SimOptions(dt=0.01, substeps=8),
            show_viewer=not headless,
        )
        backend = GenesisBackend(scene)
        for prim in self.structure_prims() + self.furniture_prims():
            backend.add(prim)

        lights = (
            get_genesis_lights()
            if get_genesis_lights is not None
            else getattr(gs, "lights", None)
        )
        if lights is not None:
            scene.add_light(lights.AmbientLight(color=(1.0, 1.0, 1.0), intensity=0.55))
            scene.add_light(
                lights.DirectionalLight(
                    color=(1.0, 0.95, 0.85),
                    intensity=1.2,
                    pos=(total_w, total_h, 8.0),
                    dir=(-0.4, -0.5, -1.0),
                )
            )

        # 离线渲染用的俯视相机（必须在 build 之前加入）
        self.camera = scene.add_camera(
            res=(1280, 720),
            pos=(0.0, -total_h * 0.55, total_h * 0.95),
            lookat=(0.0, 0.2, 0.4),
            fov=55,
            GUI=False,
        )
        scene.build()
        self.scene = scene
        return scene

    def capture(self, path: Union[str, Path]) -> Optional[Path]:
        """Render one frame from the top-down camera into a PNG (needs PIL)."""
        if self.scene is None or self.camera is None:
            raise RuntimeError("call build() before capture()")
        rgb, _, _, _ = self.camera.render(
            rgb=True, depth=False, seg=False, normal=False
        )
        try:
            from PIL import Image
        except ImportError:
            print("Pillow not installed; cannot save PNG.")
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def load_layout(source: LayoutLike) -> Layout:
    """Accept a Layout, a dict, or a path to a layout JSON."""
    if isinstance(source, Layout):
        return source
    if isinstance(source, dict):
        return Layout.from_dict(source)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    return Layout.from_dict(data)


def populate_genesis_scene(
    gs_scene: Any,
    layout: LayoutLike,
    festival: bool = False,
) -> list[Prim]:
    """Populate an existing ``gs.Scene`` with a WFC Chinese home.

    Integration hook for DreamDojo / maniskill environments: call after
    creating the scene (and adding robots/cameras) but **before**
    ``scene.build()``.
    """
    home = ChineseHomeScene(layout=layout, festival=festival)
    backend = GenesisBackend(gs_scene)
    for prim in home.structure_prims() + home.furniture_prims():
        backend.add(prim)
    return home.furniture_prims()


__all__ = [
    "ChineseHomeScene",
    "build_structure_prims",
    "load_layout",
    "populate_genesis_scene",
]
