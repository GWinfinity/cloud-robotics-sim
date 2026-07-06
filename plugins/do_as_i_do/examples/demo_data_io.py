"""Demonstrate DemoSequence / RobotTrajectory serialization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from do_as_i_do.core.data import DemoSequence, RobotTrajectory
from do_as_i_do.core.reconstruction_stub import SyntheticReconstructionStage


def main():
    """Demonstrate saving and loading demo/trajectory data."""
    stage = SyntheticReconstructionStage(num_frames=30)
    demo = stage.run("synthetic_video")

    demo_path = Path("outputs/do_as_i_do/demo_sequence")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo.save(demo_path)
    print(f"Saved demo to {demo_path}")

    loaded = DemoSequence.load(demo_path)
    print(f"Loaded demo frames: {len(loaded)}")
    print(f"Object position first frame: {loaded.object_trajectory.positions[0]}")

    # Dummy robot trajectory for I/O demo.
    traj = RobotTrajectory(
        left_arm_q=loaded.left_hand.joints[:, :6],
        right_arm_q=loaded.right_hand.joints[:, :6],
        left_hand_q=loaded.left_hand.joints,
        right_hand_q=loaded.right_hand.joints,
    )
    traj_path = Path("outputs/do_as_i_do/robot_trajectory")
    traj.save(traj_path)
    print(f"Saved trajectory to {traj_path}")

    loaded_traj = RobotTrajectory.load(traj_path)
    print(f"Loaded trajectory frames: {len(loaded_traj)}")


if __name__ == "__main__":
    main()
