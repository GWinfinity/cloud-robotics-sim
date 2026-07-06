"""Data structures and I/O for the do-as-i-do reproduction pipeline."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ObjectTrajectory:
    """Per-frame 6-DoF object trajectory.

    Attributes:
        positions: (N, 3) array of object center positions.
        orientations: (N, 4) array of quaternions [x, y, z, w].
        timestamps: Optional (N,) array of timestamps in seconds.
    """

    positions: np.ndarray
    orientations: np.ndarray
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and coerce trajectory arrays."""
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.orientations = np.asarray(self.orientations, dtype=np.float32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.orientations.ndim != 2 or self.orientations.shape[1] != 4:
            raise ValueError("orientations must have shape (N, 4)")
        if len(self.positions) != len(self.orientations):
            raise ValueError("positions and orientations must have the same length")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
            if len(self.timestamps) != len(self.positions):
                raise ValueError("timestamps length must match positions length")

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.positions)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "positions": self.positions.tolist(),
            "orientations": self.orientations.tolist(),
        }
        if self.timestamps is not None:
            data["timestamps"] = self.timestamps.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectTrajectory:
        return cls(
            positions=np.array(data["positions"], dtype=np.float32),
            orientations=np.array(data["orientations"], dtype=np.float32),
            timestamps=np.array(data["timestamps"], dtype=np.float32)
            if "timestamps" in data
            else None,
        )


@dataclass
class HandTrajectory:
    """Per-frame wrist pose and optional finger joints/keypoints.

    Attributes:
        wrist_positions: (N, 3) array.
        wrist_orientations: (N, 4) quaternion array [x, y, z, w].
        joints: Optional (N, J) joint angle array.
        keypoints: Optional (N, K, 3) hand keypoint array.
        timestamps: Optional (N,) timestamps.
    """

    wrist_positions: np.ndarray
    wrist_orientations: np.ndarray
    joints: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and coerce hand trajectory arrays."""
        self.wrist_positions = np.asarray(self.wrist_positions, dtype=np.float32)
        self.wrist_orientations = np.asarray(self.wrist_orientations, dtype=np.float32)
        if self.wrist_positions.ndim != 2 or self.wrist_positions.shape[1] != 3:
            raise ValueError("wrist_positions must have shape (N, 3)")
        if self.wrist_orientations.ndim != 2 or self.wrist_orientations.shape[1] != 4:
            raise ValueError("wrist_orientations must have shape (N, 4)")
        if len(self.wrist_positions) != len(self.wrist_orientations):
            raise ValueError("wrist positions and orientations must have the same length")
        if self.joints is not None:
            self.joints = np.asarray(self.joints, dtype=np.float32)
            if len(self.joints) != len(self.wrist_positions):
                raise ValueError("joints length must match wrist positions length")
        if self.keypoints is not None:
            self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
            if len(self.keypoints) != len(self.wrist_positions):
                raise ValueError("keypoints length must match wrist positions length")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
            if len(self.timestamps) != len(self.wrist_positions):
                raise ValueError("timestamps length must match wrist positions length")

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.wrist_positions)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "wrist_positions": self.wrist_positions.tolist(),
            "wrist_orientations": self.wrist_orientations.tolist(),
        }
        if self.joints is not None:
            data["joints"] = self.joints.tolist()
        if self.keypoints is not None:
            data["keypoints"] = self.keypoints.tolist()
        if self.timestamps is not None:
            data["timestamps"] = self.timestamps.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandTrajectory:
        return cls(
            wrist_positions=np.array(data["wrist_positions"], dtype=np.float32),
            wrist_orientations=np.array(data["wrist_orientations"], dtype=np.float32),
            joints=np.array(data["joints"], dtype=np.float32) if "joints" in data else None,
            keypoints=np.array(data["keypoints"], dtype=np.float32)
            if "keypoints" in data
            else None,
            timestamps=np.array(data["timestamps"], dtype=np.float32)
            if "timestamps" in data
            else None,
        )


@dataclass
class DemoSequence:
    """Output of the reconstruction stage and input to retargeting.

    Compatible with the layout.json produced by the original do-as-i-do
    reconstruction pipeline.
    """

    video_path: Optional[str]
    object_mesh_path: Optional[str]
    object_trajectory: ObjectTrajectory
    left_hand: HandTrajectory
    right_hand: HandTrajectory
    fps: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that all trajectories have the same length."""
        n = len(self.object_trajectory)
        if len(self.left_hand) != n or len(self.right_hand) != n:
            raise ValueError(
                "object, left_hand, and right_hand trajectories must have the same length"
            )

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.object_trajectory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "object_mesh_path": self.object_mesh_path,
            "object_trajectory": self.object_trajectory.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "fps": self.fps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DemoSequence:
        return cls(
            video_path=data.get("video_path"),
            object_mesh_path=data.get("object_mesh_path"),
            object_trajectory=ObjectTrajectory.from_dict(data["object_trajectory"]),
            left_hand=HandTrajectory.from_dict(data["left_hand"]),
            right_hand=HandTrajectory.from_dict(data["right_hand"]),
            fps=float(data.get("fps", 30.0)),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save as a compressed NPZ file plus JSON sidecar for metadata."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        np.savez(
            npz_path,
            obj_positions=self.object_trajectory.positions,
            obj_orientations=self.object_trajectory.orientations,
            obj_timestamps=self.object_trajectory.timestamps
            if self.object_trajectory.timestamps is not None
            else np.array([]),
            left_wrist_positions=self.left_hand.wrist_positions,
            left_wrist_orientations=self.left_hand.wrist_orientations,
            left_joints=self.left_hand.joints if self.left_hand.joints is not None else np.array([]),
            right_wrist_positions=self.right_hand.wrist_positions,
            right_wrist_orientations=self.right_hand.wrist_orientations,
            right_joints=self.right_hand.joints
            if self.right_hand.joints is not None
            else np.array([]),
            fps=self.fps,
        )
        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_path": self.video_path,
                    "object_mesh_path": self.object_mesh_path,
                    "fps": self.fps,
                    "metadata": self.metadata,
                    "npz_file": npz_path.name,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> DemoSequence:
        """Load from NPZ + JSON sidecar."""
        path = Path(path)
        json_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")
        if not json_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"Expected {json_path} and {npz_path}")

        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        data = np.load(npz_path)

        def _get(arr: np.ndarray) -> Optional[np.ndarray]:
            return arr if arr.size > 0 else None

        return cls(
            video_path=meta.get("video_path"),
            object_mesh_path=meta.get("object_mesh_path"),
            object_trajectory=ObjectTrajectory(
                positions=data["obj_positions"],
                orientations=data["obj_orientations"],
                timestamps=_get(data["obj_timestamps"]),
            ),
            left_hand=HandTrajectory(
                wrist_positions=data["left_wrist_positions"],
                wrist_orientations=data["left_wrist_orientations"],
                joints=_get(data["left_joints"]),
            ),
            right_hand=HandTrajectory(
                wrist_positions=data["right_wrist_positions"],
                wrist_orientations=data["right_wrist_orientations"],
                joints=_get(data["right_joints"]),
            ),
            fps=float(meta.get("fps", 30.0)),
            metadata=meta.get("metadata", {}),
        )

    @classmethod
    def from_layout_json(
        cls,
        layout_path: str | Path,
        video_path: Optional[str] = None,
        object_mesh_path: Optional[str] = None,
        anchor_hand: str = "right",
    ) -> DemoSequence:
        """Parse the original do-as-i-do ``layout_camera_frame_optimized.json``.

        The original file stores per-frame object poses. Hand meshes are stored
        separately in ``all_hand_meshes.npz``; this loader therefore creates
        placeholder hand trajectories.  Override or extend this method once
        hand poses are available.
        """
        layout_path = Path(layout_path)
        with layout_path.open("r", encoding="utf-8") as f:
            frames = json.load(f)

        if isinstance(frames, dict) and "frames" in frames:
            frames = frames["frames"]

        positions = []
        orientations = []
        timestamps = []
        for frame in frames:
            pos = frame.get("position")
            if pos is None:
                pos = frame.get("obj_pos")
            quat = frame.get("orientation")
            if quat is None:
                quat = frame.get("obj_quat")
            if pos is None or quat is None:
                warnings.warn(f"Skipping frame without position/orientation: {frame.keys()}")
                continue
            positions.append(pos)
            orientations.append(quat)
            if "time" in frame:
                timestamps.append(frame["time"])

        n = len(positions)
        object_trajectory = ObjectTrajectory(
            positions=np.array(positions, dtype=np.float32),
            orientations=np.array(orientations, dtype=np.float32),
            timestamps=np.array(timestamps, dtype=np.float32) if timestamps else None,
        )

        # Placeholder hand trajectories: will be overwritten by real retargeting.
        placeholder_hand = HandTrajectory(
            wrist_positions=np.zeros((n, 3), dtype=np.float32),
            wrist_orientations=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
        )
        left_hand = placeholder_hand
        right_hand = placeholder_hand
        if anchor_hand == "left":
            right_hand = placeholder_hand
        else:
            left_hand = placeholder_hand

        return cls(
            video_path=video_path,
            object_mesh_path=object_mesh_path,
            object_trajectory=object_trajectory,
            left_hand=left_hand,
            right_hand=right_hand,
            fps=30.0,
            metadata={"source": "layout.json", "layout_path": str(layout_path)},
        )


@dataclass
class RobotTrajectory:
    """Output of the retargeting stage.

    Attributes:
        left_arm_q: (N, 6) UR3 joint positions in radians.
        right_arm_q: (N, 6) UR3 joint positions in radians.
        left_hand_q: (N, J) left hand joint positions in radians.
        right_hand_q: (N, J) right hand joint positions in radians.
        timestamps: Optional (N,) timestamps.
    """

    left_arm_q: np.ndarray
    right_arm_q: np.ndarray
    left_hand_q: np.ndarray
    right_hand_q: np.ndarray
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and coerce robot trajectory arrays."""
        self.left_arm_q = np.asarray(self.left_arm_q, dtype=np.float32)
        self.right_arm_q = np.asarray(self.right_arm_q, dtype=np.float32)
        self.left_hand_q = np.asarray(self.left_hand_q, dtype=np.float32)
        self.right_hand_q = np.asarray(self.right_hand_q, dtype=np.float32)
        n = len(self.left_arm_q)
        for arr, name in [
            (self.right_arm_q, "right_arm_q"),
            (self.left_hand_q, "left_hand_q"),
            (self.right_hand_q, "right_hand_q"),
        ]:
            if len(arr) != n:
                raise ValueError(f"{name} length ({len(arr)}) does not match left_arm_q ({n})")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
            if len(self.timestamps) != n:
                raise ValueError("timestamps length must match trajectory length")

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.left_arm_q)

    def full_q(self) -> np.ndarray:
        """Return concatenated (N, 44) joint positions: arms + hands."""
        return np.concatenate(
            [self.left_arm_q, self.right_arm_q, self.left_hand_q, self.right_hand_q], axis=1
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "left_arm_q": self.left_arm_q.tolist(),
            "right_arm_q": self.right_arm_q.tolist(),
            "left_hand_q": self.left_hand_q.tolist(),
            "right_hand_q": self.right_hand_q.tolist(),
        }
        if self.timestamps is not None:
            data["timestamps"] = self.timestamps.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotTrajectory:
        return cls(
            left_arm_q=np.array(data["left_arm_q"], dtype=np.float32),
            right_arm_q=np.array(data["right_arm_q"], dtype=np.float32),
            left_hand_q=np.array(data["left_hand_q"], dtype=np.float32),
            right_hand_q=np.array(data["right_hand_q"], dtype=np.float32),
            timestamps=np.array(data["timestamps"], dtype=np.float32)
            if "timestamps" in data
            else None,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        np.savez(
            npz_path,
            left_arm_q=self.left_arm_q,
            right_arm_q=self.right_arm_q,
            left_hand_q=self.left_hand_q,
            right_hand_q=self.right_hand_q,
            timestamps=self.timestamps if self.timestamps is not None else np.array([]),
        )
        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({"npz_file": npz_path.name}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> RobotTrajectory:
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        data = np.load(npz_path)
        timestamps = data["timestamps"]
        return cls(
            left_arm_q=data["left_arm_q"],
            right_arm_q=data["right_arm_q"],
            left_hand_q=data["left_hand_q"],
            right_hand_q=data["right_hand_q"],
            timestamps=timestamps if timestamps.size > 0 else None,
        )
