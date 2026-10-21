"""ReplicaCAD static three-way classification demo: mesh mode vs sdf mode.

Builds the same ReplicaCAD scene twice (in separate processes; run this
script once per ``--mode``) and reports:

- total wall time of env construction (which includes ``scene.build()`` --
  the convex decomposition that dominates mesh mode), and
- in sdf mode: the baked static SDF's memory report plus a query self-check.

The bake result is cached on disk (``CRS_SIM_CACHE_DIR``/``outputs/sim_cache``
under ``vdb/``), so repeat sdf-mode runs measure load time, not bake time.
Delete the cache file to re-measure a cold bake.

Usage::

    python plugins/envs/maniskill/examples/replica_cad_scene_sdf.py --mode mesh
    python plugins/envs/maniskill/examples/replica_cad_scene_sdf.py --mode sdf [--voxel-size 0.05]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the plugin importable both from the repo root and standalone.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "plugins" / "envs" / "maniskill" / "core"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _self_check(static_sdf) -> None:
    """Conservative level-set sanity checks (no apartment-specific poses)."""
    report = static_sdf.memory_report()
    # Sign census over the dense leaf buffers: random *linear* samples would
    # interpolate thin wall interiors away, so assert on voxel values.
    n_in = sum(int((leaf.values < 0).sum()) for leaf in static_sdf.grid.leaves())
    n_out = sum(int((leaf.values > 0).sum()) for leaf in static_sdf.grid.leaves())
    assert n_in > 0, "no interior voxels: bake looks empty"
    assert n_out > 0, "no exterior voxels: bake looks solid"
    print(
        f"SDF self-check: leaf-buffer census {n_in} interior / {n_out} exterior voxels"
    )

    # Trilinear sampling over the bake bbox stays finite.
    lo, hi = report["bbox_ijk"]
    voxel = np.float32(report["voxel_size"])
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(1024, 3)).astype(np.float32) * voxel
    vals = np.atleast_1d(static_sdf.sample_linear(pts))
    assert np.all(np.isfinite(vals)), "non-finite SDF samples"
    print("SDF self-check: 1024 bbox samples finite")
    print(
        "SDF memory: {n_leaves} leaves, {active} active voxels, "
        "{mb:.1f} MB leaf buffers".format(
            n_leaves=report["n_leaves"],
            active=report["active_voxels"],
            mb=report["leaf_bytes"] / 1e6,
        )
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument(
        "--mode",
        choices=["mesh", "sdf"],
        default="sdf",
        help="static handling mode (default: sdf)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="static SDF voxel size in meters, sdf mode only (default: 0.05)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="random-action steps after build (default: 0 = build only)",
    )
    args = parser.parse_args()

    from genesis_maniskill.envs.replica_cad_env import ReplicaCADEnv

    scene_config = {"add_lights": False}
    if args.mode == "sdf":
        scene_config.update(
            {
                "static_mode": "sdf",
                "sdf_voxel_size": args.voxel_size,
                "sdf_ground": "box",
            }
        )

    t0 = time.perf_counter()
    env = ReplicaCADEnv(
        scene_name=args.scene,
        robot_uid="franka",
        task_type="pick_place",
        obs_mode="state",
        render_mode=None,
        scene_config=scene_config,
    )
    try:
        env.reset()
        wall = time.perf_counter() - t0
        print(f"MODE={args.mode} scene={args.scene} build_wall_seconds={wall:.1f}")
        print(
            f"objects={len(env.objects)} movable={len(env.movable_objects)} "
            f"articulated={len(env.articulations)}"
        )

        static_sdf = getattr(env.scene_builder, "static_sdf", None)
        if args.mode == "sdf":
            if static_sdf is None:
                print("ERROR: static_sdf is None in sdf mode")
                return 1
            _self_check(static_sdf)

        for _ in range(args.steps):
            env.step(env.action_space.sample())
        if args.steps:
            print(f"stepped {args.steps} random actions")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
