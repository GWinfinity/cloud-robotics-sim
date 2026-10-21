"""ReplicaCAD contact-parity evaluation: mesh mode vs sdf mode.

The plugin's PickPlaceTask targets ``env.objects[0]`` -- a STATIC furniture
entity in both modes -- so its success flag is not a meaningful contact-fidelity
metric on ReplicaCAD. This harness instead measures what the static
three-way classification actually changed: the *physics of the movable
objects*.

Per mode (two runs of this script, one per ``--mode``):

- Phase A (settle): N zero-action episodes; tracks every movable object's z.
  In mesh mode objects rest on furniture; in sdf mode they fall to the floor
  slab (supporting furniture is visual-only). Fall-through through the slab
  (z < -0.05) or NaN would be a physics regression.
- Phase B (disturb): N random-action episodes; the flailing arm nudges
  objects. Same metrics, plus |xy| runaway detection.

Report the printed JSON block per mode; compare across runs. A true grasping
success-rate comparison needs a pick policy (OMPL/RL) -- out of scope here.

Usage::

    python plugins/envs/maniskill/examples/replica_cad_contact_eval.py --mode mesh
    python plugins/envs/maniskill/examples/replica_cad_contact_eval.py --mode sdf [--voxel-size 0.05]
"""

from __future__ import annotations

import argparse
import json
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


def _xyz(entity) -> np.ndarray:
    """Entity world position as a squeezed (3,) float array."""
    pos = entity.get_pos()
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    return np.asarray(pos, dtype=np.float64).reshape(-1)[:3]


def main() -> int:
    """CLI entry point: run settle+disturb evaluation for one mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--mode", choices=["mesh", "sdf"], required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--settle-steps", type=int, default=60, help="control steps (~3 s)"
    )
    parser.add_argument("--disturb-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="SimOptions.substeps; >1 rescues thin objects that tunnel at "
        "impact speed within one dt step (control dt unchanged)",
    )
    args = parser.parse_args()

    # Pre-initialize Genesis on the CPU: the full-collision mesh mode trips a
    # quadrants kernel over this card's 64 KB dynamic-smem ceiling on CUDA
    # (66048 B requested); CPU is also the deterministic apples-to-apples
    # backend for both modes.
    import genesis as gs

    from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend

    if not gs._initialized:
        gs.init(backend=get_genesis_backend("cpu"), precision="32")

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
    # ~120-entity apartment: the default broad-phase pair budget (x8) and
    # narrow-phase candidate cap (150) overflow under contact-rich episodes
    # in BOTH modes; keep the solver settings identical for parity.
    rigid_options = gs.options.RigidOptions(
        multiplier_collision_broad_phase=64,
        max_collision_pairs=1024,
    )
    env = ReplicaCADEnv(
        scene_name=args.scene,
        robot_uid="franka",
        task_type="pick_place",
        obs_mode="state",
        render_mode=None,
        scene_config=scene_config,
        rigid_options=rigid_options,
        sim_substeps=args.substeps,
    )
    env.reset(seed=args.seed)
    build_wall = time.perf_counter() - t0

    movable = dict(env.movable_objects)
    names = sorted(movable)

    def run_episode(n_steps: int, rng: np.random.Generator, disturb: bool):
        zs = {n: [] for n in names}
        xy_max = 0.0
        for _ in range(n_steps):
            if disturb:
                # Nudge-scale actions: full-strength random flailing floods
                # Newton's broad-phase pair buffer in the full apartment
                # (max_collision_pairs_broad) -- not the signal we want.
                action = (
                    rng.uniform(-1.0, 1.0, size=env.action_space.shape) * 0.12
                ).astype(np.float32)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            env.step(action)
            for n in names:
                p = _xyz(movable[n])
                zs[n].append(float(p[2]))
                xy_max = max(xy_max, float(np.abs(p[:2]).max()))
        return zs, xy_max

    rng = np.random.default_rng(args.seed)
    report = {
        "mode": args.mode,
        "scene": args.scene,
        "build_wall_seconds": round(build_wall, 1),
        "n_movable": len(names),
        "substeps": args.substeps,
    }
    static_sdf = getattr(env.scene_builder, "static_sdf", None)
    if args.mode == "sdf":
        report["static_sdf_active_voxels"] = (
            static_sdf.active_voxel_count if static_sdf is not None else None
        )

    for phase, steps, disturb in (
        ("settle", args.settle_steps, False),
        ("disturb", args.disturb_steps, True),
    ):
        finals, mins, nan_eps, fallthrough, runaways = [], [], 0, 0, 0
        for _ in range(args.episodes):
            env.reset(seed=int(rng.integers(0, 2**31)))
            zs, xy_max = run_episode(steps, rng, disturb)
            ep_mins = [min(v) for v in zs.values()]
            ep_finals = [v[-1] for v in zs.values()]
            if any(not np.isfinite(v) for v in ep_mins + ep_finals):
                nan_eps += 1
                continue
            mins.append(min(ep_mins))
            finals.extend(ep_finals)
            fallthrough += sum(1 for v in ep_mins if v < -0.05)
            if xy_max > 10.0:
                runaways += 1
        report[phase] = {
            "episodes": args.episodes,
            "min_z_worst": round(min(mins), 4) if mins else None,
            "final_z_mean": round(float(np.mean(finals)), 4) if finals else None,
            "final_z_min": round(min(finals), 4) if finals else None,
            "final_z_max": round(max(finals), 4) if finals else None,
            "fall_throughs": fallthrough,
            "nan_episodes": nan_eps,
            "xy_runaways": runaways,
        }

    print("CONTACT_EVAL " + json.dumps(report, sort_keys=True))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
