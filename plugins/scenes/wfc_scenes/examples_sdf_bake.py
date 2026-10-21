"""WFC layout -> static SDF bake demo (offline; no Genesis required).

Bakes a Chinese-home layout into a query-only level set (walls/floors/
furniture) and runs the same self-check as the ReplicaCAD example. The result
is cached on disk; delete the cache file to re-measure a cold bake.

Usage::

    python plugins/scenes/wfc_scenes/examples_sdf_bake.py --grid 3x2 --seed 42 \
        --voxel-size 0.1 --out outputs/wfc_sdf/home42.xuvdb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
        "--out",
        type=Path,
        default=None,
        help="optional .xuvdb export path (the disk cache is always written)",
    )
    args = parser.parse_args()

    from scene import ChineseHomeScene
    from sdf_bake import bake_layout  # plugin dir on sys.path (see above)

    w, h = (int(x) for x in args.grid.lower().split("x"))
    home = ChineseHomeScene(grid=(w, h), seed=args.seed)

    t0 = time.perf_counter()
    vol = bake_layout(home, voxel_size=args.voxel_size, name=f"wfc-{args.seed}")
    wall = time.perf_counter() - t0

    report = vol.memory_report()
    n_in = sum(int((leaf.values < 0).sum()) for leaf in vol.grid.leaves())
    n_out = sum(int((leaf.values > 0).sum()) for leaf in vol.grid.leaves())
    print(f"bake_wall_seconds={wall:.1f} (cache hit included)")
    print(
        "SDF memory: {n_leaves} leaves, {active} active voxels, "
        "{mb:.1f} MB leaf buffers".format(
            n_leaves=report["n_leaves"],
            active=report["active_voxels"],
            mb=report["leaf_bytes"] / 1e6,
        )
    )
    print(f"SDF census: {n_in} interior / {n_out} exterior voxels")
    assert n_in > 0 and n_out > 0, "bake looks degenerate"

    if args.out is not None:
        from cloud_robotics_sim.vdb.core.vdb_xuvdb import save_xuvdb

        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_xuvdb(args.out, [vol], compress=True)
        print(f"exported {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
