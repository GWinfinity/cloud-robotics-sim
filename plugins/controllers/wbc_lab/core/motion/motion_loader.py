"""Motion data loading for WBC training.

Ported from wbc_lab/motion/  -loads NPZ motion libraries and provides
per-frame reference features for the WBC MDP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionClip:
    """Single motion clip loaded from NPZ."""

    name: str
    fps: float
    num_frames: int
    # Body kinematics (T, num_bodies, ...)
    body_pos_w: np.ndarray      # (T, B, 3) world-frame body positions
    body_quat_w: np.ndarray     # (T, B, 4) world-frame body quaternions (wxyz)
    body_lin_vel_w: np.ndarray  # (T, B, 3)
    body_ang_vel_w: np.ndarray  # (T, B, 3)
    # Joint state (T, num_joints)
    joint_pos: np.ndarray       # (T, J)
    joint_vel: np.ndarray       # (T, J)
    # Root state
    base_pos_w: np.ndarray      # (T, 3)
    base_quat_w: np.ndarray     # (T, 4)
    base_lin_vel_w: np.ndarray  # (T, 3)
    base_ang_vel_w: np.ndarray  # (T, 3)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


class MotionLoader:
    """Loads and manages a library of motion clips from NPZ files.

    Supports both single-bundle NPZ (stacked clips) and per-clip NPZ directories.
    """

    def __init__(
        self,
        motion_path: str | Path,
        clip_names: list[str] | None = None,
    ):
        self.motion_path = Path(motion_path)
        self.clips: list[MotionClip] = []
        self._clip_names = clip_names
        self._load()

    def _load(self) -> None:
        """Load motion clips from path."""
        if self.motion_path.is_file() and self.motion_path.suffix == ".npz":
            self._load_bundle(self.motion_path)
        elif self.motion_path.is_dir():
            self._load_directory(self.motion_path)
        else:
            raise FileNotFoundError(f"Motion path not found: {self.motion_path}")

    def _load_bundle(self, path: Path) -> None:
        """Load a stacked bundle NPZ."""
        data = np.load(path, allow_pickle=False)
        # Bundle format: stacked arrays with clip boundaries
        if "clip_names" in data:
            names = data["clip_names"]
            for i, name in enumerate(names):
                self._load_clip_from_bundle(data, i, str(name))
        else:
            # Single clip bundle
            self._load_clip_from_bundle(data, 0, path.stem)

    def _load_directory(self, path: Path) -> None:
        """Load per-clip NPZ files from a directory."""
        npz_dir = path / "npz" if (path / "npz").is_dir() else path
        files = sorted(npz_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

        for f in files:
            if self._clip_names is not None and f.stem not in self._clip_names:
                continue
            try:
                clip = self._load_single_clip(f)
                self.clips.append(clip)
            except Exception as e:
                logger.warning("Failed to load clip %s: %s", f.name, e)

        if not self.clips:
            raise RuntimeError("No valid clips loaded")

    def _load_single_clip(self, path: Path) -> MotionClip:
        """Load a single clip NPZ file."""
        data = np.load(path, allow_pickle=False)
        fps = float(data.get("fps", 30.0))
        body_pos = data["body_pos_w"]
        num_frames = body_pos.shape[0]

        return MotionClip(
            name=path.stem,
            fps=fps,
            num_frames=num_frames,
            body_pos_w=body_pos,
            body_quat_w=data.get("body_quat_w", np.zeros((num_frames, body_pos.shape[1], 4))),
            body_lin_vel_w=data.get("body_lin_vel_w", np.zeros_like(body_pos)),
            body_ang_vel_w=data.get("body_ang_vel_w", np.zeros((num_frames, body_pos.shape[1], 3))),
            joint_pos=data.get("joint_pos", np.zeros((num_frames, 29))),
            joint_vel=data.get("joint_vel", np.zeros((num_frames, 29))),
            base_pos_w=data.get("base_pos_w", body_pos[:, 0, :]),
            base_quat_w=data.get("base_quat_w", np.zeros((num_frames, 4))),
            base_lin_vel_w=data.get("base_lin_vel_w", np.zeros((num_frames, 3))),
            base_ang_vel_w=data.get("base_ang_vel_w", np.zeros((num_frames, 3))),
        )

    def _load_clip_from_bundle(self, data: Any, index: int, name: str) -> None:
        """Load a clip from a stacked bundle."""
        # Stacked bundles store arrays as (total_frames, ...) with clip_starts
        clip_starts = data.get("clip_starts")
        if clip_starts is not None:
            start = int(clip_starts[index])
            end = int(clip_starts[index + 1]) if index + 1 < len(clip_starts) else None
        else:
            start = None
            end = None

        def _slice(arr: np.ndarray) -> np.ndarray:
            return arr[start:end] if start is not None else arr

        fps = float(data.get("fps", 30.0))
        body_pos = _slice(data["body_pos_w"])
        num_frames = body_pos.shape[0]

        clip = MotionClip(
            name=name,
            fps=fps,
            num_frames=num_frames,
            body_pos_w=body_pos,
            body_quat_w=_slice(data.get("body_quat_w", np.zeros((num_frames, body_pos.shape[1], 4)))),
            body_lin_vel_w=_slice(data.get("body_lin_vel_w", np.zeros_like(body_pos))),
            body_ang_vel_w=_slice(data.get("body_ang_vel_w", np.zeros((num_frames, body_pos.shape[1], 3)))),
            joint_pos=_slice(data.get("joint_pos", np.zeros((num_frames, 29)))),
            joint_vel=_slice(data.get("joint_vel", np.zeros((num_frames, 29)))),
            base_pos_w=_slice(data.get("base_pos_w", body_pos[:, 0, :])),
            base_quat_w=_slice(data.get("base_quat_w", np.zeros((num_frames, 4)))),
            base_lin_vel_w=_slice(data.get("base_lin_vel_w", np.zeros((num_frames, 3)))),
            base_ang_vel_w=_slice(data.get("base_ang_vel_w", np.zeros((num_frames, 3)))),
        )
        self.clips.append(clip)

    def num_clips(self) -> int:
        return len(self.clips)

    def total_frames(self) -> int:
        return sum(c.num_frames for c in self.clips)

    def total_duration(self) -> float:
        return sum(c.duration for c in self.clips)

    def get_clip(self, index: int) -> MotionClip:
        return self.clips[index]

    def get_clip_by_name(self, name: str) -> MotionClip:
        for clip in self.clips:
            if clip.name == name:
                return clip
        raise KeyError(f"Clip not found: {name}")

    def build_stacked_arrays(self) -> dict[str, np.ndarray]:
        """Stack all clips into contiguous arrays for GPU training."""
        all_body_pos = np.concatenate([c.body_pos_w for c in self.clips], axis=0)
        all_joint_pos = np.concatenate([c.joint_pos for c in self.clips], axis=0)
        all_joint_vel = np.concatenate([c.joint_vel for c in self.clips], axis=0)
        clip_starts = np.cumsum([0] + [c.num_frames for c in self.clips])

        return {
            "body_pos_w": all_body_pos,
            "joint_pos": all_joint_pos,
            "joint_vel": all_joint_vel,
            "clip_starts": clip_starts,
            "fps": self.clips[0].fps if self.clips else 30.0,
        }
