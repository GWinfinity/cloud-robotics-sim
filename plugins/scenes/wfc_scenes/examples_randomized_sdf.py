# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Room-content domain randomization demo (offline; no Genesis required).

Bakes a WFC home shell plus several room-content variants into packed SDF
grids, then samples and casts against two different "room contents" in one
batch — what a per-env randomized vec env does at every step (each env
holds its own volume id; the Genesis scene itself is never rebuilt).

Usage::

    python plugins/scenes/wfc_scenes/examples_randomized_sdf.py \
        --grid 3x2 --seed 42 --voxel-size 0.1 --variants kitchen bedroom study
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default="3x2", help="WFC grid WxH (default: 3x2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["kitchen", "bedroom", "study"],
        help="room types to dress the target cell as",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional .xuvdb export path for the packed grids",
    )
    args = parser.parse_args()

    from randomized_sdf import RandomizedRoomSDF
    from scene import ChineseHomeScene

    w, h = (int(x) for x in args.grid.lower().split("x"))
    home = ChineseHomeScene(grid=(w, h), seed=args.seed)
    cells = home.layout.cells
    target = next(
        (k for k, c in enumerate(cells) if c["tile"] == "living"),
        0,  # fall back to cell 0 when the seed produced no living room
    )
    print(f"layout: {home.ascii_map()}")
    print(
        f"randomizing cell {target} (native tile {cells[target]['tile']!r}) "
        f"over variants {args.variants}"
    )

    t0 = time.perf_counter()
    rsdf = RandomizedRoomSDF.build(
        home, {target: args.variants}, voxel_size=args.voxel_size
    )
    wall = time.perf_counter() - t0
    report = rsdf.memory_report()
    print(
        "packed {n} grids in {wall:.1f}s: {leaves} leaves, "
        "{active} active voxels, {mb:.1f} MB leaf buffers".format(
            n=report["n_grids"],
            wall=wall,
            leaves=report["n_leaves"],
            active=report["active_voxels"],
            mb=report["leaf_bytes"] / 1e6,
        )
    )

    # -- sampling: same world points, two different room contents ----------
    layout = home.layout
    i, j = cells[target]["i"], cells[target]["j"]
    cx, cy = layout.cell_center(i, j)
    hx, hy = layout.cell_half(i, j)
    rng = np.random.default_rng(0)
    pts = rng.uniform(
        (cx - hx, cy - hy, 0.3), (cx + hx, cy + hy, 1.8), size=(4096, 3)
    ).astype(np.float32)
    id_a = rsdf.id_for(target, args.variants[0])
    id_b = rsdf.id_for(target, args.variants[1])
    ids = np.array([id_a, id_b], dtype=np.int32)
    pts2 = np.concatenate([pts, pts])
    vals = rsdf.sample(pts2, np.repeat(ids, len(pts)))
    d_a, d_b = vals[: len(pts)], vals[len(pts) :]
    differ = int((np.abs(d_a - d_b) > 1e-6).sum())
    print(
        f"sampled {len(pts)} pts x 2 variants: {differ} pts differ "
        f"({100.0 * differ / len(pts):.1f}%) — per-point domain randomization"
    )

    # -- ray casting: downward rays, one batch, both variants --------------
    nx, ny = 12, 12
    xs = np.linspace(cx - hx * 0.8, cx + hx * 0.8, nx)
    ys = np.linspace(cy - hy * 0.8, cy + hy * 0.8, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)], axis=1)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(origins), 1))
    for tile in args.variants[:2]:
        rid = np.full(len(origins), rsdf.id_for(target, tile))
        t, hit, _ = rsdf.cast(origins, dirs, rid)
        hits = hit[np.isfinite(t)]
        print(
            f"cast as {tile:8s}: {len(hits)}/{len(origins)} hits, "
            f"mean hit z = {hits[:, 2].mean() if len(hits) else float('nan'):.3f}"
        )

    if args.out is not None:
        from cloud_robotics_sim.vdb.core.vdb_xuvdb import save_xuvdb

        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_xuvdb(args.out, rsdf._grids, compress=True)
        print(f"exported {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
