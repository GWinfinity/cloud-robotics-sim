"""WBC termination conditions for Genesis.

Ported from wbc_lab/env/mdp/terminations.py  -position/orientation limits,
contact force checks, and episode truncation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TerminationConfig:
    """Configuration for WBC termination conditions."""

    # Anchor body position limits
    anchor_pos_xy_threshold: float = 3.0
    """Max horizontal distance from reference anchor (meters)."""
    anchor_pos_z_threshold: float = 0.8
    """Max vertical distance from reference anchor (meters)."""

    # Anchor body orientation limits
    anchor_ori_threshold: float = 1.5
    """Max orientation error (radians, ~86 deg)."""

    # Body position limits (keybody)
    body_pos_z_threshold: float = 0.8
    """Max vertical body position error (meters)."""

    # Contact force limits
    max_contact_force: float = 500.0
    """Max allowed contact force (N)."""

    # Episode length
    max_episode_steps: int = 1000
    """Maximum episode length before truncation."""

    # Height termination
    min_base_height: float = 0.3
    """Minimum base height before termination."""


class TerminationChecker:
    """Computes termination signals for WBC environments."""

    def __init__(self, cfg: TerminationConfig | None = None, num_envs: int = 1):
        self.cfg = cfg or TerminationConfig()
        self.num_envs = num_envs
        self.step_count = np.zeros(num_envs, dtype=np.int64)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset step counter for specified environments."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self.step_count[env_ids] = 0

    def check(
        self,
        *,
        base_pos: np.ndarray,
        ref_base_pos: np.ndarray | None = None,
        base_height: np.ndarray | None = None,
        anchor_ori_error: np.ndarray | None = None,
        body_pos_error_z: np.ndarray | None = None,
        contact_forces: np.ndarray | None = None,
        num_bodies: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Check all termination conditions.

        Args:
            base_pos: Current base position (num_envs, 3).
            ref_base_pos: Reference base position for distance check (num_envs, 3).
            base_height: Current base z-height (num_envs,).
            anchor_ori_error: Orientation error magnitude (num_envs,).
            body_pos_error_z: Vertical body position error (num_envs, num_bodies).
            contact_forces: Contact forces (num_envs, num_bodies).
            num_bodies: Number of tracked bodies.

        Returns:
            (terminated, truncated, info_dict) where:
              - terminated: (num_envs,) bool  -bad state detected
              - truncated: (num_envs,) bool  -episode length limit
              - info_dict: per-term termination signals
        """
        terms: dict[str, np.ndarray] = {}
        terminated = np.zeros(self.num_envs, dtype=bool)

        # Anchor XY distance
        if ref_base_pos is not None:
            xy_error = np.linalg.norm(base_pos[:, :2] - ref_base_pos[:, :2], axis=-1)
            anchor_xy_bad = xy_error > self.cfg.anchor_pos_xy_threshold
            terms["bad_anchor_pos_xy"] = anchor_xy_bad.astype(np.float64)
            terminated |= anchor_xy_bad

            # Anchor Z distance
            z_error = np.abs(base_pos[:, 2] - ref_base_pos[:, 2])
            anchor_z_bad = z_error > self.cfg.anchor_pos_z_threshold
            terms["bad_anchor_pos_z"] = anchor_z_bad.astype(np.float64)
            terminated |= anchor_z_bad

        # Base height
        if base_height is not None:
            height_bad = base_height < self.cfg.min_base_height
            terms["bad_base_height"] = height_bad.astype(np.float64)
            terminated |= height_bad

        # Anchor orientation
        if anchor_ori_error is not None:
            ori_bad = anchor_ori_error > self.cfg.anchor_ori_threshold
            terms["bad_anchor_ori"] = ori_bad.astype(np.float64)
            terminated |= ori_bad

        # Body position Z
        if body_pos_error_z is not None:
            z_bad = np.any(np.abs(body_pos_error_z) > self.cfg.body_pos_z_threshold, axis=-1)
            terms["bad_body_pos_z"] = z_bad.astype(np.float64)
            terminated |= z_bad

        # Contact force
        if contact_forces is not None:
            force_bad = np.any(contact_forces > self.cfg.max_contact_force, axis=-1)
            terms["excessive_contact"] = force_bad.astype(np.float64)
            terminated |= force_bad

        # Episode length truncation
        self.step_count += 1
        truncated = self.step_count >= self.cfg.max_episode_steps
        terms["time_out"] = truncated.astype(np.float64)

        return terminated, truncated, terms
