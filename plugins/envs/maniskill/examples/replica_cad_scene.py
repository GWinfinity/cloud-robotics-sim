"""ReplicaCAD scene demo: build ManiSkill's ReplicaCAD apartment scene
(ReplicaCAD_SceneManipulation-v1 equivalent) on the Genesis backend.

The ReplicaCAD scene dataset is downloaded automatically from ModelScope
(``jessy888/ManiSkill_replica_cad_dataset``, ~289 MB) on first use; prefetch
with::

    python -m genesis_maniskill.datasets.replicacad_assets

Usage::

    python plugins/envs/maniskill/examples/replica_cad_scene.py --scene apt_0 --save-img
    python plugins/envs/maniskill/examples/replica_cad_scene.py --list-scenes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the plugin importable both from the repo root and standalone.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "plugins" / "envs" / "maniskill" / "core"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    """CLI entry point: list scenes or run a ReplicaCAD demo episode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene",
        default="apt_0",
        help="ReplicaCAD scene name (default: apt_0)",
    )
    parser.add_argument(
        "--robot",
        default="franka",
        help="Robot uid, e.g. franka, ur5, mobile_manipulator (default: franka)",
    )
    parser.add_argument(
        "--task",
        default="pick_place",
        help="Task type (default: pick_place)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of random-action steps (default: 50)",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available ReplicaCAD scene names and exit",
    )
    parser.add_argument(
        "--include-doors",
        action="store_true",
        help="Also spawn articulated door templates",
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        help="Save an offscreen render to outputs/replica_cad_<scene>.png",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Show the interactive Genesis viewer",
    )
    args = parser.parse_args()

    from genesis_maniskill.datasets.replicacad_assets import list_scenes
    from genesis_maniskill.envs.replica_cad_env import ReplicaCADEnv

    if args.list_scenes:
        for name in list_scenes():
            print(name)
        return 0

    render_mode = "human" if args.human else None
    # RGB observations add the base camera before scene.build(); required so
    # --save-img can grab an offscreen render afterwards.
    obs_mode = "rgb" if args.save_img else "state"
    env = ReplicaCADEnv(
        scene_name=args.scene,
        robot_uid=args.robot,
        task_type=args.task,
        obs_mode=obs_mode,
        render_mode=render_mode,
        scene_config={"include_doors": args.include_doors},
    )
    try:
        obs, info = env.reset()
        print(
            f"Built ReplicaCAD scene {args.scene!r}: "
            f"{len(env.objects)} objects "
            f"({len(env.movable_objects)} movable, "
            f"{len(env.articulations)} articulated), "
            f"obs_dim={obs.shape[-1] if hasattr(obs, 'shape') else '?'}"
        )

        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "human":
                env.render()

        if args.save_img:
            import numpy as np

            cam = env.cameras["base_camera"]
            rgb, *_ = cam.render(rgb=True)
            img = np.asarray(rgb)
            if img.ndim == 4:  # (n_envs, H, W, 3) -> (H, W, 3)
                img = img[0]
            out_dir = _REPO_ROOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"replica_cad_{args.scene}.png"
            import imageio.v2 as imageio

            imageio.imwrite(str(out_path), (img * 255).astype(np.uint8))
            print(f"Saved render to {out_path}")

        print("Demo finished.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
