"""Tests for the WFC Chinese-home scene generator (offline, network-free).

These tests exercise the WFC solver, the Chinese-home rule set, furniture
primitive builders, and all offline scene artifacts (ASCII / SVG / JSON /
GLB). A Genesis CPU build smoke test runs only when genesis-world is
importable.
"""

import json
import math
import random

import pytest

from plugins.scenes.wfc_scenes.assets.furniture_cn import (
    Prim,
    TrimeshBackend,
    build_tile_prims,
    prim_mesh,
)
from plugins.scenes.wfc_scenes.core.rules_cn import (
    ALL_SOCKETS,
    DIRS,
    SOCKET_DOOR,
    SOCKET_OPEN,
    SOCKET_RAIL,
    SOCKET_WALL,
    TILES,
    Layout,
    build_variants,
    compatible,
    count_tiles,
    exterior_ok,
    needs_partition,
    rotated_sockets,
    validate_layout,
)
from plugins.scenes.wfc_scenes.core.wfc import (
    WFCContradictionError,
    collapse,
    expand_variants,
)
from plugins.scenes.wfc_scenes.scene import (
    ChineseHomeScene,
    load_layout,
)

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def make_layout(grid=(4, 3), seed=1, **kwargs):
    """Generate a validated layout, raising a helpful error if it fails."""
    home = ChineseHomeScene(grid=grid, seed=seed, **kwargs)
    return home, home.generate_layout()


class TestWFCSolver:
    """Generic solver behaviour on the Chinese-home rule set."""

    def test_collapse_completes(self):
        variants, weights = build_variants()
        rng = random.Random(0)
        result = collapse(4, 3, variants, weights, compatible, exterior_ok, rng)
        assert len(result) == 3 and all(len(row) == 4 for row in result)
        assert all(0 <= v < len(variants) for row in result for v in row)

    def test_seed_reproducibility(self):
        variants, weights = build_variants()
        a = collapse(4, 3, variants, weights, compatible, exterior_ok, random.Random(7))
        b = collapse(4, 3, variants, weights, compatible, exterior_ok, random.Random(7))
        assert a == b

    def test_different_seeds_give_variety(self):
        variants, weights = build_variants()
        layouts = {
            tuple(
                variants[v].tile
                for row in collapse(
                    4, 3, variants, weights, compatible, exterior_ok, random.Random(s)
                )
                for v in row
            )
            for s in range(12)
        }
        assert len(layouts) >= 3, "不同种子应当产生不同户型"

    def test_contradiction_raises(self):
        # A tile whose every edge is "open" can never sit in a 1x1 grid:
        # all four edges face the exterior, which forbids open sockets.
        variants = expand_variants({"room": ("open", "open", "open", "open")})
        with pytest.raises(WFCContradictionError):
            collapse(
                1,
                1,
                variants,
                [1.0] * len(variants),
                compatible,
                exterior_ok,
                random.Random(0),
            )

    def test_invalid_grid_rejected(self):
        variants, weights = build_variants()
        with pytest.raises(ValueError):
            collapse(0, 3, variants, weights, compatible, exterior_ok, random.Random(0))
        with pytest.raises(ValueError):
            collapse(
                2, 2, variants, weights[:-1], compatible, exterior_ok, random.Random(0)
            )


class TestRulesCN:
    """Tile catalog and socket rules."""

    def test_catalog_sanity(self):
        assert len(TILES) == 10
        for spec in TILES.values():
            assert len(spec.sockets) == 4
            assert set(spec.sockets) <= ALL_SOCKETS
            assert spec.weight > 0
            assert len(spec.ascii_char) == 1
            assert len(spec.zh) >= 2

    def test_compat_symmetric(self):
        names = sorted(ALL_SOCKETS)
        for i, a in enumerate(names):
            for b in names[i:]:
                assert compatible(a, b) == compatible(b, a)

    def test_core_adjacency_rules(self):
        # 背靠背墙、开放动线、台面规则
        assert compatible(SOCKET_WALL, SOCKET_WALL)
        assert compatible(SOCKET_OPEN, SOCKET_OPEN)
        # 开放边不能接墙（门洞不能被墙堵死）
        assert not compatible(SOCKET_OPEN, SOCKET_WALL)
        # 门和栏杆只出现在外墙
        assert not compatible(SOCKET_DOOR, SOCKET_OPEN)
        assert not compatible(SOCKET_RAIL, SOCKET_OPEN)
        assert not compatible(SOCKET_DOOR, SOCKET_WALL)
        assert not compatible(SOCKET_RAIL, SOCKET_WALL)

    def test_boundary_rules(self):
        assert exterior_ok(SOCKET_WALL)
        assert exterior_ok(SOCKET_DOOR)
        assert exterior_ok(SOCKET_RAIL)
        assert not exterior_ok(SOCKET_OPEN)
        # 餐厅/过道四边开放、永远位于室内；其余模块都能贴外墙/外窗。
        for name, spec in TILES.items():
            can_touch_exterior = any(exterior_ok(s) for s in spec.all_patterns()[0])
            assert can_touch_exterior == (name not in ("dining", "corridor")), name

    def test_rotated_sockets(self):
        seq = ("wall", "open", "open", "door")
        assert rotated_sockets(seq, 0) == seq
        # 顺时针 90°：旧 W 边转到新 N
        assert rotated_sockets(seq, 90) == ("door", "wall", "open", "open")
        assert rotated_sockets(seq, 180) == ("open", "door", "wall", "open")
        assert rotated_sockets(seq, 270) == ("open", "open", "door", "wall")
        # 转四次回到原位
        r = seq
        for rot in (90, 90, 90, 90):
            r = rotated_sockets(r, rot)
        assert r == seq

    def test_needs_partition(self):
        assert needs_partition("wall", "wall")
        assert needs_partition("counter", "wall")
        assert not needs_partition("counter", "counter")
        assert not needs_partition("open", "open")


class TestGeneratedLayouts:
    """End-to-end WFC + 全局中式规则校验."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_layouts_pass_cn_rules(self, seed):
        home, layout = make_layout(grid=(4, 3), seed=seed)
        assert validate_layout(layout) == []
        counts = count_tiles(layout)
        assert counts.get("entry", 0) == 1
        assert counts.get("living", 0) >= 1
        assert counts.get("kitchen", 0) >= 1
        assert counts.get("balcony", 0) >= 1
        assert counts.get("bedroom", 0) >= 1

    def test_layout_adjacency_and_boundary(self):
        _, layout = make_layout(grid=(4, 3), seed=3)
        for cell in layout.cells:
            i, j, sockets = (
                cell["i"],
                cell["j"],
                layout.sockets_at(cell["i"], cell["j"]),
            )
            for d, (dx, dy) in enumerate(DIRS):
                ni, nj = i + dx, j + dy
                if 0 <= ni < layout.width and 0 <= nj < layout.height:
                    assert compatible(
                        sockets[d], layout.sockets_at(ni, nj)[(d + 2) % 4]
                    )
                else:
                    assert exterior_ok(sockets[d])

    def test_small_grid_rejects_balcony_requirement(self):
        # 6 格小家不强制阳台（面积 < 8），应可生成
        home = ChineseHomeScene(grid=(3, 2), seed=1)
        layout = home.generate_layout()
        assert validate_layout(layout, require_balcony=False, min_bedrooms=0) == []

    def test_min_bedrooms_rule(self):
        home = ChineseHomeScene(grid=(4, 3), seed=1, min_bedrooms=2)
        layout = home.generate_layout()
        assert count_tiles(layout).get("bedroom", 0) >= 2

    def test_unsatisfiable_bedrooms_exhausts_attempts(self):
        home = ChineseHomeScene(grid=(2, 2), seed=1, min_bedrooms=5, max_attempts=4)
        with pytest.raises(RuntimeError, match="中式家居规则"):
            home.generate_layout()


class TestValidatorRejections:
    """Global rules must fire on mutated layouts."""

    def _mutate(self, layout, from_tile, to_tile):
        cells = [dict(c) for c in layout.cells]
        for cell in cells:
            if cell["tile"] == from_tile:
                cell["tile"] = to_tile
                cell["pattern"] = 0  # 墙模式是每个模块自己的属性
        return Layout(layout.width, layout.height, layout.tile_size, layout.seed, cells)

    def test_missing_entry_rejected(self):
        _, layout = make_layout()
        broken = self._mutate(layout, "entry", "corridor")
        assert any("玄关" in s for s in validate_layout(broken))

    def test_missing_living_rejected(self):
        _, layout = make_layout()
        broken = self._mutate(layout, "living", "corridor")
        assert any("客厅" in s for s in validate_layout(broken))

    def test_missing_kitchen_rejected(self):
        _, layout = make_layout()
        broken = self._mutate(layout, "kitchen", "corridor")
        assert any("厨房" in s for s in validate_layout(broken))

    def test_missing_balcony_rejected(self):
        _, layout = make_layout()
        broken = self._mutate(layout, "balcony", "corridor")
        assert any("阳台" in s for s in validate_layout(broken))

    def test_missing_bedroom_rejected(self):
        _, layout = make_layout()
        broken = self._mutate(layout, "bedroom", "study")
        assert any("卧室" in s for s in validate_layout(broken, min_bedrooms=1))


class TestFurniturePrims:
    """Primitive builders and the trimesh backend."""

    def _check_prims(self, prims):
        assert len(prims) > 0
        for prim in prims:
            assert isinstance(prim, Prim)
            assert prim.kind in ("box", "cylinder", "sphere")
            assert all(math.isfinite(v) for v in prim.pos)
            assert abs(prim.pos[0]) <= 2.6 and abs(prim.pos[1]) <= 2.6
            assert -0.2 <= prim.pos[2] <= 3.2
            assert all(0 <= c <= 1 for c in prim.color[:3])

    def test_all_tiles_produce_furniture(self):
        rng = random.Random(0)
        for tile in TILES:
            self._check_prims(build_tile_prims(tile, (0.0, 0.0), 0, rng))

    def test_unknown_tile_rejected(self):
        with pytest.raises(ValueError):
            build_tile_prims("villa", (0.0, 0.0), 0, random.Random(0))

    def test_rotation_moves_furniture(self):
        rot0 = build_tile_prims("living", (0.0, 0.0), 0, random.Random(0))
        rot180 = build_tile_prims("living", (0.0, 0.0), 180, random.Random(0))
        assert len(rot0) == len(rot180)
        sum_y0 = sum(p.pos[1] for p in rot0)
        sum_y180 = sum(p.pos[1] for p in rot180)
        assert math.isclose(sum_y0, -sum_y180, abs_tol=1e-6)

    def test_festival_adds_decorations(self):
        rng = random.Random(0)
        plain = build_tile_prims("living", (0.0, 0.0), 0, rng, festival=False)
        festive = build_tile_prims("living", (0.0, 0.0), 0, rng, festival=True)
        assert len(festive) > len(plain)

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_prim_mesh_and_backend(self):
        mesh = prim_mesh(Prim("box", (1, 1, 1), (0, 0, 0.5), (1, 0, 0), "test"))
        assert isinstance(mesh, trimesh.Trimesh)
        backend = TrimeshBackend()
        for prim in build_tile_prims("living", (0.0, 0.0), 0, random.Random(0)):
            backend.add(prim)
        assert len(backend.scene.geometry) > 10

    def test_deterministic_prims(self):
        a = build_tile_prims("bedroom", (1.0, 2.0), 90, random.Random(5))
        b = build_tile_prims("bedroom", (1.0, 2.0), 90, random.Random(5))
        assert [(p.kind, p.size, p.pos) for p in a] == [
            (p.kind, p.size, p.pos) for p in b
        ]


class TestChineseHomeScene:
    """Scene-level artifacts: ASCII / JSON / SVG / GLB."""

    def test_ascii_map_shape(self):
        home = ChineseHomeScene(grid=(4, 3), seed=2)
        lines = home.ascii_map().splitlines()
        assert len(lines) == 3
        assert all(len(line.split()) == 4 for line in lines)

    def test_layout_json_roundtrip(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=4)
        path = home.export_layout(tmp_path / "home.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["generator"] == "wfc_scenes"
        loaded = load_layout(path)
        assert loaded.width == 4 and loaded.height == 3
        assert [c["tile"] for c in loaded.cells] == [
            c["tile"] for c in home.layout.cells
        ]

    def test_from_layout_replay(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=5)
        path = home.export_layout(tmp_path / "home.json")
        replay = ChineseHomeScene(layout=path)
        assert replay.ascii_map() == home.ascii_map()
        assert replay.furniture_prims() == home.furniture_prims()

    def test_bad_layout_file_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"generator": "other"}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_layout(bad)

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_glb_export(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=6)
        path = home.export_glb(tmp_path / "home.glb")
        assert path.stat().st_size > 10_000
        scene = trimesh.load(path)
        assert len(scene.geometry) > 100

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_trimesh_build(self):
        home = ChineseHomeScene(grid=(4, 3), seed=6)
        scene = home.build_trimesh()
        assert len(scene.geometry) > 100

    def test_svg_export(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=7)
        path = home.topdown_svg(tmp_path / "plan.svg")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg") and "客厅" in text
        assert text.count("<rect") >= 12

    def test_structure_prims_present(self):
        home = ChineseHomeScene(grid=(4, 3), seed=8)
        prims = home.structure_prims()
        names = [p.name for p in prims]
        assert any(n.startswith("floor_slab") for n in names)
        assert any("ext_wall" in n for n in names)
        # 玄关门外墙必须有门洞（post/lintel 而非整墙）
        entry_cell = next(c for c in home.layout.cells if c["tile"] == "entry")
        door_edges = [
            n
            for n in names
            if n.startswith(f"ext_wall_{entry_cell['i']}_{entry_cell['j']}_")
            and "_post" in n
        ]
        assert len(door_edges) == 2

    def test_furniture_prims_deterministic(self):
        home_a = ChineseHomeScene(grid=(4, 3), seed=9)
        home_b = ChineseHomeScene(grid=(4, 3), seed=9)
        assert home_a.furniture_prims() == home_b.furniture_prims()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world not installed")
class TestGenesisBuild:
    """CPU smoke test: a full home must build and step in Genesis."""

    def test_cpu_build_and_step(self):
        home = ChineseHomeScene(grid=(3, 2), seed=11)
        scene = home.build(headless=True, device="cpu")
        assert scene is not None
        for _ in range(5):
            scene.step()
        assert len(scene.entities) > 50
        gs.destroy()


class TestVariableSize:
    """方案 2：行列不等宽 + 模块最小边长 + 家具自适应。"""

    def test_variable_layout_valid_and_within_bounds(self):
        home = ChineseHomeScene(grid=(4, 3), seed=21, variable_size=True)
        layout = home.generate_layout()
        assert validate_layout(layout) == []
        for i in range(layout.width):
            assert 2.4 - 1e-6 <= layout.col_w(i) <= 4.2 + 1e-6
        for j in range(layout.height):
            assert 2.4 - 1e-6 <= layout.row_h(j) <= 4.2 + 1e-6
        for cell in layout.cells:
            i, j, spec = cell["i"], cell["j"], TILES[cell["tile"]]
            assert min(layout.col_w(i), layout.row_h(j)) + 1e-6 >= spec.min_edge

    def test_variable_sizes_differ_across_columns(self):
        home = ChineseHomeScene(grid=(5, 3), seed=22, variable_size=True)
        layout = home.generate_layout()
        assert len(set(layout.col_widths)) >= 3, "随机列宽应产生不等宽的列"
        assert len(set(layout.row_heights)) >= 2

    def test_variable_json_roundtrip(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=23, variable_size=True)
        path = home.export_layout(tmp_path / "var.json")
        loaded = load_layout(path)
        assert loaded.col_widths == home.layout.col_widths
        assert loaded.row_heights == home.layout.row_heights
        replay = ChineseHomeScene(layout=path)
        assert replay.ascii_map() == home.ascii_map()
        assert replay.furniture_prims() == home.furniture_prims()

    def test_uniform_layout_v1_compat(self):
        """v1 布局（只有 tile_size、无 col_widths）应加载为统一格子。"""
        _, layout = make_layout(seed=24)
        data = layout.to_dict()
        data.pop("col_widths", None)
        data.pop("row_heights", None)
        data["version"] = 1
        loaded = Layout.from_dict(data)
        assert loaded.col_widths is None and loaded.row_heights is None
        assert loaded.col_w(0) == layout.tile_size == loaded.tile_size
        assert validate_layout(loaded) == []

    def test_min_edge_violation_detected(self):
        _, layout = make_layout(seed=25)
        layout.col_widths = [1.5] * layout.width  # 远小于任何模块最小边长
        issues = validate_layout(layout)
        assert any("边长不足" in s for s in issues)

    @pytest.mark.parametrize(
        "tile,span",
        [
            ("kitchen", (1.4, 1.3)),
            ("bedroom", (1.5, 1.5)),
            ("living", (1.7, 1.6)),
            ("bathroom", (1.1, 1.1)),
            ("entry", (1.2, 1.2)),
            ("balcony", (1.2, 1.3)),
            ("tea", (1.4, 1.4)),
            ("study", (1.4, 1.4)),
            ("dining", (1.5, 1.5)),
            ("corridor", (1.0, 1.6)),
        ],
    )
    def test_furniture_stays_inside_small_cells(self, tile, span):
        """最小合法格子里，家具不得越出格子边界。"""
        prims = build_tile_prims(tile, (0.0, 0.0), 0, random.Random(3), half=span)
        assert prims
        for prim in prims:
            if prim.kind == "box":
                ex, ey = prim.size[0] / 2, prim.size[1] / 2
            else:
                ex = ey = prim.size[0]
            assert prim.pos[0] - ex >= -span[0] - 0.03, (prim.name, prim.pos)
            assert prim.pos[0] + ex <= span[0] + 0.03, (prim.name, prim.pos)
            assert prim.pos[1] - ey >= -span[1] - 0.03, (prim.name, prim.pos)
            assert prim.pos[1] + ey <= span[1] + 0.03, (prim.name, prim.pos)

    def test_furniture_adapts_to_span(self):
        """同一家具在大小格子里位置应贴各自的墙（而不是固定坐标）。"""
        small = build_tile_prims(
            "bedroom", (0, 0), 0, random.Random(1), half=(1.5, 1.5)
        )
        big = build_tile_prims("bedroom", (0, 0), 0, random.Random(1), half=(1.8, 2.0))
        head_s = next(p for p in small if p.name.startswith("bed_headboard"))
        head_b = next(p for p in big if p.name.startswith("bed_headboard"))
        assert head_b.pos[1] > head_s.pos[1]  # 床头贴各自的 N 墙

    def test_structure_covers_variable_span(self):
        home = ChineseHomeScene(grid=(4, 3), seed=26, variable_size=True)
        layout = home.generate_layout()
        prims = home.structure_prims()
        slab = next(p for p in prims if p.name.startswith("floor_slab"))
        assert abs(slab.size[0] - layout.total_width - 0.4) < 1e-6
        assert abs(slab.size[1] - layout.total_depth - 0.4) < 1e-6

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_variable_svg_and_glb(self, tmp_path):
        home = ChineseHomeScene(grid=(4, 3), seed=27, variable_size=True)
        svg = home.topdown_svg(tmp_path / "var.svg")
        text = svg.read_text(encoding="utf-8")
        assert text.count("<rect") >= 12
        glb = home.export_glb(tmp_path / "var.glb")
        assert glb.stat().st_size > 10_000

    @pytest.mark.parametrize("seed", [31, 32, 33, 34, 35])
    def test_variable_generation_stable(self, seed):
        for grid in [(3, 2), (4, 3), (5, 4)]:
            home = ChineseHomeScene(grid=grid, seed=seed, variable_size=True)
            layout = home.generate_layout()
            min_bedrooms = 1 if grid[0] * grid[1] >= 8 else 0
            assert validate_layout(layout, min_bedrooms=min_bedrooms) == []
