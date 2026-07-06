"""Load a retargeted trajectory and replay it in Genesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from do_as_i_do.core.data import DemoSequence, RobotTrajectory
from do_as_i_do.core.env import DoAsIDoEnv


def main():
    """Replay a saved robot trajectory in Genesis."""
    parser = argparse.ArgumentParser(description="Replay a retargeted robot trajectory.")
    parser.add_argument("--trajectory", type=str, default="outputs/do_as_i_do/robot_trajectory")
    parser.add_argument("--demo", type=str, default="outputs/do_as_i_do/demo_sequence")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    traj = RobotTrajectory.load(args.trajectory)
    demo = DemoSequence.load(args.demo) if Path(args.demo + ".json").exists() else None

    env = DoAsIDoEnv(headless=args.headless)
    env.reset()
    metrics = env.replay(
        traj,
        object_trajectory=demo.object_trajectory if demo else None,
        close_on_finish=True,
    )
    print(f"Replayed {len(metrics)} steps")


if __name__ == "__main__":
    main()
