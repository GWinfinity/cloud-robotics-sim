# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""CLI for the WFC Chinese-home scene generator.

Examples (run from the repo root)::

    # 看一眼布局（ASCII 户型图）
    python -m plugins.scenes.wfc_scenes --grid 4x3 --seed 42 --ascii

    # 生成一套场景：布局 JSON + 俯视 SVG + 整体 GLB
    python -m plugins.scenes.wfc_scenes --grid 5x4 --seed 7 \
        --out-dir outputs/wfc_scenes --name home07

    # 批量生成 N 套不同的户型
    python -m plugins.scenes.wfc_scenes --batch 12 --grid 4x3 \
        --out-dir outputs/wfc_scenes

    # 用 Genesis 交互查看（需要显示器）
    python -m plugins.scenes.wfc_scenes --grid 4x3 --seed 42 --viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plugins.scenes.wfc_scenes",
        description="波函数坍缩（WFC）中式家居场景生成器",
    )
    parser.add_argument(
        "--grid",
        default="4x3",
        type=str,
        help="户型网格，格式 宽x高（格数），默认 4x3",
    )
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument(
        "--tile-size",
        default=3.2,
        type=float,
        help="统一每格边长（米）；与 --variable 互斥",
    )
    parser.add_argument(
        "--variable",
        action="store_true",
        help="行列不等宽（方案 2）：每列宽度/每行深度独立随机，并满足模块最小边长",
    )
    parser.add_argument(
        "--min-cell",
        default=2.4,
        type=float,
        help="可变模式下列宽/行高随机下限（米）",
    )
    parser.add_argument(
        "--max-cell",
        default=4.2,
        type=float,
        help="可变模式下列宽/行高随机上限（米）",
    )
    parser.add_argument(
        "--festival",
        action="store_true",
        help="加入春节装饰（灯笼/福字），默认只保留日常生活场景",
    )
    parser.add_argument("--ascii", action="store_true", help="打印 ASCII 户型图")
    parser.add_argument("--save-json", metavar="PATH", help="导出布局 JSON")
    parser.add_argument("--save-glb", metavar="PATH", help="导出整体 GLB")
    parser.add_argument("--save-svg", metavar="PATH", help="导出俯视 SVG 户型图")
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        help="便捷导出目录：<name>.json/.svg/.glb 一次生成",
    )
    parser.add_argument(
        "--name",
        default="chinese_home",
        help="out-dir 模式下的文件名前缀",
    )
    parser.add_argument(
        "--batch",
        default=0,
        type=int,
        help="批量生成 N 套不同户型的场景到 out-dir（seed 起 +1 递增）",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="构建 Genesis 场景并打开交互 viewer（需要 genesis-world）",
    )
    return parser


def _parse_grid(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        width, height = int(w), int(h)
    except ValueError as exc:
        raise SystemExit(f"--grid 格式应为 宽x高，例如 4x3，收到: {text}") from exc
    if width < 1 or height < 1 or width * height > 64:
        raise SystemExit(f"--grid 尺寸非法: {text}")
    return width, height


def _generate_one(args, grid, seed, out_dir=None, name=None) -> int:
    from .scene import ChineseHomeScene

    home = ChineseHomeScene(
        grid=grid,
        tile_size=args.tile_size,
        seed=seed,
        festival=args.festival,
        variable_size=args.variable,
        min_cell=args.min_cell,
        max_cell=args.max_cell,
    )
    print(f"[seed={home.layout.seed}] 户型:")
    print(home.ascii_map())
    if args.variable:
        print(
            "列宽(m):",
            " ".join(f"{w:.2f}" for w in home.layout.col_widths),
        )
        print(
            "行深(m):",
            " ".join(f"{h:.2f}" for h in home.layout.row_heights),
        )

    if out_dir is not None and name is not None:
        json_path = home.export_layout(Path(out_dir) / f"{name}.json")
        svg_path = home.topdown_svg(Path(out_dir) / f"{name}.svg")
        glb_path = home.export_glb(Path(out_dir) / f"{name}.glb")
        print(f"  布局 -> {json_path}")
        print(f"  俯视图 -> {svg_path}")
        print(f"  模型 -> {glb_path}")
    else:
        if args.save_json:
            print(f"  布局 -> {home.export_layout(args.save_json)}")
        if args.save_svg:
            print(f"  俯视图 -> {home.topdown_svg(args.save_svg)}")
        if args.save_glb:
            print(f"  模型 -> {home.export_glb(args.save_glb)}")

    if args.viewer:
        scene = home.build(headless=False, device="cpu")
        if scene is None:
            print("genesis-world 未安装，跳过 viewer。")
            return 1
        for _ in range(2000):
            scene.step()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see module docstring for examples)."""
    args = _build_parser().parse_args(argv)
    grid = _parse_grid(args.grid)

    if args.batch > 0:
        if not args.out_dir:
            raise SystemExit("--batch 需要 --out-dir")
        for k in range(args.batch):
            print(f"=== 场景 {k + 1}/{args.batch} ===")
            rc = _generate_one(
                args,
                grid,
                args.seed + k,
                out_dir=Path(args.out_dir),
                name=f"{args.name}_{args.seed + k}",
            )
            if rc:
                return rc
        return 0

    if args.out_dir:
        return _generate_one(
            args, grid, args.seed, out_dir=Path(args.out_dir), name=args.name
        )
    return _generate_one(args, grid, args.seed)


if __name__ == "__main__":
    sys.exit(main())
