"""Reconstruction stage for the do-as-i-do reproduction.

The original reconstruction pipeline uses SAM3, SAM-3D-Objects, MoGe, HaWoR,
TAPIR and guided diffusion.  Those modules are kept behind an abstract interface
so that this scaffold can run on CPU with synthetic data, while still exposing
the same API as the original pipeline.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .data import DemoSequence, HandTrajectory, ObjectTrajectory


class ReconstructionStage(ABC):
    """Abstract reconstruction stage: video -> DemoSequence."""

    @abstractmethod
    def run(
        self,
        video_path: str | Path,
        object_name: Optional[str] = None,
        anchor_hand: str = "right",
    ) -> DemoSequence:
        """Run reconstruction and return a ``DemoSequence``."""
        ...


class SyntheticReconstructionStage(ReconstructionStage):
    """Generate a synthetic demonstration without external vision modules.

    The synthetic trajectory moves an object along the table while both hands
    approach, grasp, and release it.  It is intended for testing the
    retargeting and simulation stages on CPU-only machines.
    """

    def __init__(
        self,
        num_frames: int = 150,
        fps: float = 30.0,
        table_height: float = 0.75,
        object_size: float = 0.05,
        hand_dof: int = 16,
    ) -> None:
        self.num_frames = num_frames
        self.fps = fps
        self.table_height = table_height
        self.object_size = object_size
        self.hand_dof = hand_dof

    def run(
        self,
        video_path: str | Path,
        object_name: Optional[str] = None,
        anchor_hand: str = "right",
    ) -> DemoSequence:
        n = self.num_frames
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        timestamps = t * (n / self.fps)

        # Object starts left, moves right along the table.
        obj_pos = np.zeros((n, 3), dtype=np.float32)
        obj_pos[:, 0] = -0.15 + 0.30 * t
        obj_pos[:, 2] = self.table_height + self.object_size / 2
        obj_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))

        # Hands start at rest, move in to grasp, then release.
        left_wrist = np.zeros((n, 3), dtype=np.float32)
        right_wrist = np.zeros((n, 3), dtype=np.float32)

        # Approach phase [0, 0.3]
        approach = np.clip(t / 0.3, 0.0, 1.0)
        left_wrist[:, 0] = -0.15 - 0.10 * (1.0 - approach)
        left_wrist[:, 1] = 0.25 - 0.15 * approach
        left_wrist[:, 2] = self.table_height + 0.05

        right_wrist[:, 0] = -0.15 - 0.10 * (1.0 - approach)
        right_wrist[:, 1] = -0.25 + 0.15 * approach
        right_wrist[:, 2] = self.table_height + 0.05

        # Grasp phase [0.3, 0.7] - hands lift object slightly.
        lift = 0.05 * np.sin(np.clip((t - 0.3) / 0.4, 0.0, 1.0) * np.pi)
        left_wrist[:, 2] += lift
        right_wrist[:, 2] += lift

        # Release phase [0.7, 1.0]
        release = np.where(t > 0.7, (t - 0.7) / 0.3, 0.0)
        left_wrist[:, 1] += 0.15 * release
        right_wrist[:, 1] -= 0.15 * release

        wrist_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))

        # Hand joint motion: curl fingers during grasp.
        left_joints = self._generate_hand_joints(n, t)
        right_joints = self._generate_hand_joints(n, t)

        return DemoSequence(
            video_path=str(video_path),
            object_mesh_path=None,
            object_trajectory=ObjectTrajectory(
                positions=obj_pos,
                orientations=obj_quat,
                timestamps=timestamps,
            ),
            left_hand=HandTrajectory(
                wrist_positions=left_wrist,
                wrist_orientations=wrist_quat,
                joints=left_joints,
                timestamps=timestamps,
            ),
            right_hand=HandTrajectory(
                wrist_positions=right_wrist,
                wrist_orientations=wrist_quat,
                joints=right_joints,
                timestamps=timestamps,
            ),
            fps=self.fps,
            metadata={
                "stage": "synthetic",
                "object_name": object_name,
                "anchor_hand": anchor_hand,
            },
        )

    def _generate_hand_joints(self, n: int, t: np.ndarray) -> np.ndarray:
        """Generate a simple curling motion for the configured hand DOF count."""
        joints = np.zeros((n, self.hand_dof), dtype=np.float32)
        curl = 0.6 * np.sin(np.clip((t - 0.25) / 0.5, 0.0, 1.0) * np.pi)

        if self.hand_dof == 16:
            # 4 fingers x 4 joints (Allegro style).
            for finger in range(4):
                base = finger * 4
                joints[:, base + 1] = curl
                joints[:, base + 2] = curl
                joints[:, base + 3] = curl * 0.5
        elif self.hand_dof == 22:
            # 5 fingers: thumb 5, index/middle/ring 4, pinky 5 (Sharpa style).
            finger_dofs = [5, 4, 4, 4, 5]
            idx = 0
            for fd in finger_dofs:
                if fd >= 2:
                    joints[:, idx + 1] = curl
                if fd >= 3:
                    joints[:, idx + 2] = curl
                if fd >= 4:
                    joints[:, idx + 3] = curl * 0.5
                if fd >= 5:
                    joints[:, idx + 4] = curl * 0.25
                idx += fd
        else:
            # Generic fallback: curl all joints.
            joints[:, 1:] = curl[:, None]
        return joints


class OriginalReconstructionStage(ReconstructionStage):
    """Placeholder that would call the original do-as-i-do reconstruction.

    It is not functional in this environment because the required modules
    (SAM3, HaWoR, TAPIR, etc.) and model weights cannot be downloaded without
    GPU/network access.  Instantiate ``SyntheticReconstructionStage`` instead,
    or implement the real calls here on a machine that has the original repo
    cloned and configured.
    """

    def __init__(self, original_repo_path: Optional[str | Path] = None) -> None:
        self.original_repo_path = original_repo_path

    def run(
        self,
        video_path: str | Path,
        object_name: Optional[str] = None,
        anchor_hand: str = "right",
    ) -> DemoSequence:
        warnings.warn(
            "OriginalReconstructionStage is a placeholder. "
            "The original SAM3/HaWoR/TAPIR modules are not installed or accessible. "
            "Use SyntheticReconstructionStage to run the pipeline on CPU."
        )
        raise NotImplementedError(
            "Original do-as-i-do reconstruction pipeline is not available. "
            "Set reconstruction.backend=synthetic or provide a fully configured "
            "original repository at original_repo_path."
        )


def build_reconstruction_stage(config: dict[str, Any]) -> ReconstructionStage:
    """Factory for reconstruction stages based on configuration."""
    backend = config.get("reconstruction", {}).get("backend", "synthetic")
    hand_type = config.get("robot", {}).get("hand_type", "allegro")
    hand_dof = 22 if hand_type == "sharpa" else 16
    if backend == "synthetic":
        return SyntheticReconstructionStage(
            num_frames=config.get("reconstruction", {}).get("num_frames", 150),
            fps=config.get("reconstruction", {}).get("fps", 30.0),
            hand_dof=hand_dof,
        )
    if backend == "original":
        return OriginalReconstructionStage(
            original_repo_path=config.get("reconstruction", {}).get("original_repo_path")
        )
    raise ValueError(f"Unknown reconstruction backend: {backend}")
