"""Motion command for WBC multi-clip playback with RSI.

Ported from wbc_lab/env/mdp/commands.py  -manages motion clip playback,
reference feature computation, and RSI (Reference State Initialization).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..motion.motion_loader import MotionClip, MotionLoader

logger = logging.getLogger(__name__)


@dataclass
class MotionCommandCfg:
    """Configuration for motion command tracking."""

    motion_path: str = ""
    """Path to motion NPZ bundle or directory."""
    clip_names: list[str] | None = None
    """Specific clips to load. None = all clips."""
    anchor_body_name: str = "torso_link"
    """Body used as the tracking anchor."""
    motion_body_names: tuple[str, ...] = ()
    """Body names for keybody tracking."""
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Min/max resampling interval (seconds). Large = no mid-episode resample."""
    pose_noise_std: dict[str, float] = field(default_factory=lambda: {
        "x": 0.05, "y": 0.05, "z": 0.01,
        "roll": 0.1, "pitch": 0.1, "yaw": 0.2,
    })
    joint_noise_range: tuple[float, float] = (-0.1, 0.1)


class MotionCommand:
    """Multi-clip motion playback engine.

    Manages current clip state, frame advancement, and reference feature
    computation for the WBC observation/reward system.
    """

    def __init__(
        self,
        cfg: MotionCommandCfg,
        num_envs: int = 1,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Load motion library
        self.loader = MotionLoader(cfg.motion_path, cfg.clip_names)
        self.clips = self.loader.clips

        if not self.clips:
            raise RuntimeError("No motion clips loaded")

        # Build stacked arrays
        stacked = self.loader.build_stacked_arrays()
        self._body_pos_all = stacked["body_pos_w"]
        self._joint_pos_all = stacked["joint_pos"]
        self._joint_vel_all = stacked["joint_vel"]
        self._clip_starts = stacked["clip_starts"]
        self._fps = stacked["fps"]

        # Per-env state
        self._current_clip_idx = np.zeros(num_envs, dtype=np.int64)
        self._current_frame = np.zeros(num_envs, dtype=np.float64)
        self._global_frame = np.zeros(num_envs, dtype=np.int64)
        self._clip_duration = np.array([c.duration for c in self.clips])

        # Initialize with clip 0
        self._assign_random_clips(np.arange(num_envs))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    def _assign_random_clips(self, env_ids: np.ndarray) -> None:
        """Assign random clips to specified environments."""
        rng = np.random.default_rng()
        self._current_clip_idx[env_ids] = rng.integers(0, len(self.clips), size=len(env_ids))
        self._current_frame[env_ids] = rng.uniform(
            0,
            self._clip_duration[self._current_clip_idx[env_ids]] * self._fps,
        )
        self._update_global_frames(env_ids)

    def _update_global_frames(self, env_ids: np.ndarray) -> None:
        """Convert per-clip frame to global stacked index."""
        for i in env_ids:
            clip_idx = int(self._current_clip_idx[i])
            clip_start = int(self._clip_starts[clip_idx])
            local_frame = int(self._current_frame[i]) % self.clips[clip_idx].num_frames
            self._global_frame[i] = clip_start + local_frame

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset command state for specified environments."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self._assign_random_clips(env_ids)

    def advance(self, dt: float) -> None:
        """Advance motion playback by dt seconds."""
        self._current_frame += dt * self._fps
        # Wrap around within clip
        for i in range(self.num_envs):
            clip_idx = int(self._current_clip_idx[i])
            clip_frames = self.clips[clip_idx].num_frames
            if self._current_frame[i] >= clip_frames:
                self._current_frame[i] %= clip_frames
        self._update_global_frames(np.arange(self.num_envs))

    def get_global_frame(self) -> np.ndarray:
        """Get global stacked frame index for each env."""
        return self._global_frame.copy()

    def get_clip_idx(self) -> np.ndarray:
        """Get current clip index for each env."""
        return self._current_clip_idx.copy()

    # ─── Reference features (for observations) ───

    def ref_joint_pos(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reference joint positions (num_envs, num_joints)."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        frames = self._global_frame[env_ids]
        return self._joint_pos_all[frames].astype(np.float32)

    def ref_joint_vel(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reference joint velocities (num_envs, num_joints)."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        frames = self._global_frame[env_ids]
        return self._joint_vel_all[frames].astype(np.float32)

    def ref_body_pos(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reference body positions (num_envs, num_bodies, 3)."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        frames = self._global_frame[env_ids]
        return self._body_pos_all[frames].astype(np.float32)

    def ref_base_pos(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reference base (anchor) position (num_envs, 3)."""
        return self.ref_body_pos(env_ids)[:, 0, :]

    def ref_base_height(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reference base height (num_envs, 1)."""
        return self.ref_base_pos(env_ids)[:, 2:3]

    def get_body_tracking_error(
        self,
        current_body_pos: np.ndarray,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-body position error (num_envs, num_bodies, 3)."""
        ref = self.ref_body_pos(env_ids)
        return current_body_pos - ref

    def get_joint_tracking_error(
        self,
        current_joint_pos: np.ndarray,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-joint position error (num_envs, num_joints)."""
        ref = self.ref_joint_pos(env_ids)
        return current_joint_pos - ref
