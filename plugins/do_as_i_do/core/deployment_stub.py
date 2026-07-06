"""Deployment stage: robot trajectory -> real-robot executable format.

This is a placeholder for the original do-as-i-do deployment, which streams
joint trajectories to dual UR3e arms + Sharpa Wave hands.  The scaffold keeps
the same interface so a real driver (e.g. UR_RTDE) can be plugged in later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import RobotTrajectory
from .retargeting import _hand_joint_names


class DeploymentStub:
    """Convert a retargeted trajectory to a real-robot deployment format."""

    def __init__(self, frequency_hz: float = 50.0, hand_type: str = "allegro") -> None:
        self.frequency_hz = frequency_hz
        self.hand_joint_names = _hand_joint_names(hand_type)

    def export_trajectory(
        self,
        trajectory: RobotTrajectory,
        output_path: str | Path,
        arm_joint_names: list[str] | None = None,
        hand_joint_names: list[str] | None = None,
    ) -> Path:
        """Export the trajectory as a JSON file that a real robot driver can play.

        Args:
            trajectory: retargeted joint-space trajectory.
            output_path: destination JSON path.
            arm_joint_names: names for the 6 UR arm joints per side.
            hand_joint_names: names for the Allegro/Sharpa joints per side.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if arm_joint_names is None:
            arm_joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        if hand_joint_names is None:
            hand_joint_names = self.hand_joint_names

        n = len(trajectory)
        dt = 1.0 / self.frequency_hz
        timestamps = trajectory.timestamps if trajectory.timestamps is not None else np.arange(n) * dt

        frames = []
        for i in range(n):
            frames.append(
                {
                    "time": float(timestamps[i]),
                    "left_arm": {
                        name: float(trajectory.left_arm_q[i, j]) for j, name in enumerate(arm_joint_names)
                    },
                    "right_arm": {
                        name: float(trajectory.right_arm_q[i, j]) for j, name in enumerate(arm_joint_names)
                    },
                    "left_hand": {
                        name: float(trajectory.left_hand_q[i, j]) for j, name in enumerate(hand_joint_names)
                    },
                    "right_hand": {
                        name: float(trajectory.right_hand_q[i, j])
                        for j, name in enumerate(hand_joint_names)
                    },
                }
            )

        payload = {
            "frequency_hz": self.frequency_hz,
            "num_frames": n,
            "frames": frames,
            "metadata": {
                "note": "This is a scaffold deployment file. "
                "Replace DeploymentStub with a real UR_RTDE / Sharpa driver to execute on hardware.",
            },
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path

    def export_to_urscript(
        self,
        trajectory: RobotTrajectory,
        output_path: str | Path,
        arm_velocity: float = 0.5,
        arm_acceleration: float = 0.5,
    ) -> Path:
        """Generate a placeholder URScript program.

        The returned script is not complete; it demonstrates how to structure
        arm waypoints for a UR controller.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "def do_as_i_do_demo():",
            f"  global velocity = {arm_velocity}",
            f"  global acceleration = {arm_acceleration}",
        ]
        for i in range(min(len(trajectory), 20)):  # downsample for readability
            q = trajectory.left_arm_q[i]
            joints = ", ".join(f"{v:.4f}" for v in q)
            lines.append(f"  movel({joints}, a=acceleration, v=velocity)")
        lines.append("end")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


def build_deployment_stage(config: dict[str, Any]) -> DeploymentStub:
    """Create a deployment stage from the pipeline configuration."""
    freq = config.get("deployment", {}).get("frequency_hz", 50.0)
    hand_type = config.get("robot", {}).get("hand_type", "allegro")
    return DeploymentStub(frequency_hz=freq, hand_type=hand_type)
