"""Augment recorded grasp trajectories (offline, kinematically consistent).

Reads RoboTwin-format HDF5 episodes (produced by
``scripts/grasp_all_objects.py --record``) and writes N augmented copies per
episode (observation noise + time warp + subsampling).

For spatial diversity (new spawn/target poses) use the runner's online
augmentation instead: ``--episodes-per-class N --jitter-xy 0.05``.

Usage::

    uv run python scripts/augment_trajectories.py \
        --episodes-dir outputs/grasp_all_objects/episodes \
        --out outputs/grasp_all_objects/episodes_augmented \
        --copies 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cloud_robotics_sim.robotwin.trajectory_augment import augment_file  # noqa: E402

logger = logging.getLogger("augment_trajectories")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--pos-sigma", type=float, default=0.002)
    parser.add_argument("--joint-sigma", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(args.episodes_dir.glob("*.hdf5"))
    if not files:
        logger.error("no .hdf5 episodes in %s", args.episodes_dir)
        raise SystemExit(1)
    total = 0
    for i, path in enumerate(files):
        written = augment_file(
            path,
            args.out,
            n_copies=args.copies,
            pos_sigma=args.pos_sigma,
            joint_sigma=args.joint_sigma,
            seed=args.seed + i,
        )
        total += len(written)
        logger.info("%s -> %d augmented copies", path.name, len(written))
    logger.info("DONE: %d episodes -> %d augmented files in %s", len(files), total, args.out)


if __name__ == "__main__":
    main()
