"""WBC reward functions for Genesis.

Ported from wbc_lab/env/mdp/rewards.py  -tracking rewards (joint, body,
velocity) and regularization rewards (action rate, torque, collision, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardTermCfg:
    """Configuration for a single reward term."""

    weight: float = 1.0
    sigma: float = 0.25
    """Kernel width for exponential tracking rewards."""


@dataclass
class RewardConfig:
    """Full reward configuration."""

    # Tracking rewards
    tracking_joint_pos: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=1.0, sigma=0.1))
    tracking_joint_vel: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=0.2, sigma=0.5))
    tracking_anchor_pos: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=2.0, sigma=0.05))
    tracking_anchor_ori: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=1.0, sigma=0.2))
    tracking_body_pos: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=1.0, sigma=0.05))
    tracking_anchor_lin_vel: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=0.5, sigma=0.25))
    tracking_anchor_ang_vel: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=0.2, sigma=0.25))
    # Regularization rewards
    action_rate: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-0.01))
    joint_acc: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-2.5e-7))
    torque_limit: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-0.0001))
    self_collision: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-1.0))
    feet_slip: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-0.5))
    angular_momentum: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-0.01))
    # Termination
    termination: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(weight=-10.0))


def _tracking_exp(error: np.ndarray, sigma: float) -> np.ndarray:
    """Exponential tracking kernel: exp(-||error||^2 / (2 * sigma^2))."""
    sq_norm = np.sum(error ** 2, axis=-1)
    return np.exp(-sq_norm / (2.0 * sigma ** 2 + 1e-8))


def _dim_scaled_exp(error: np.ndarray, sigma: float, dim: int) -> np.ndarray:
    """Dimension-scaled exponential: keeps reward scale independent of dimensionality."""
    per_dim = error ** 2 / (2.0 * sigma ** 2 + 1e-8)
    return np.exp(-np.sum(per_dim, axis=-1) / max(dim, 1))


class RewardComputer:
    """Computes all WBC reward terms.

    Each compute_* method returns (num_envs,) reward values.
    The compute_all() method returns total reward and per-term breakdown.
    """

    def __init__(self, cfg: RewardConfig | None = None, num_envs: int = 1):
        self.cfg = cfg or RewardConfig()
        self.num_envs = num_envs
        self.episode_sums: dict[str, np.ndarray] = {}

    def reset_episode(self, env_ids: np.ndarray | None = None) -> None:
        """Reset episode reward accumulators."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        for key in list(self.episode_sums.keys()):
            self.episode_sums[key][env_ids] = 0.0

    # ─── Tracking rewards ───

    def tracking_joint_pos(
        self,
        joint_pos_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Joint position tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_joint_pos.sigma
        reward = _dim_scaled_exp(joint_pos_error, s, joint_pos_error.shape[-1])
        return reward * self.cfg.tracking_joint_pos.weight

    def tracking_joint_vel(
        self,
        joint_vel_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Joint velocity tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_joint_vel.sigma
        reward = _dim_scaled_exp(joint_vel_error, s, joint_vel_error.shape[-1])
        return reward * self.cfg.tracking_joint_vel.weight

    def tracking_anchor_pos(
        self,
        anchor_pos_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Anchor body position tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_anchor_pos.sigma
        reward = _tracking_exp(anchor_pos_error, s)
        return reward * self.cfg.tracking_anchor_pos.weight

    def tracking_anchor_ori(
        self,
        anchor_ori_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Anchor body orientation tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_anchor_ori.sigma
        reward = _tracking_exp(anchor_ori_error, s)
        return reward * self.cfg.tracking_anchor_ori.weight

    def tracking_body_pos(
        self,
        body_pos_error: np.ndarray,
        num_bodies: int = 1,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Multi-body position tracking reward (keybody)."""
        s = sigma if sigma is not None else self.cfg.tracking_body_pos.sigma
        reward = _dim_scaled_exp(body_pos_error, s, body_pos_error.shape[-1] * num_bodies)
        return reward * self.cfg.tracking_body_pos.weight

    def tracking_anchor_lin_vel(
        self,
        lin_vel_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Anchor linear velocity tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_anchor_lin_vel.sigma
        reward = _tracking_exp(lin_vel_error, s)
        return reward * self.cfg.tracking_anchor_lin_vel.weight

    def tracking_anchor_ang_vel(
        self,
        ang_vel_error: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Anchor angular velocity tracking reward."""
        s = sigma if sigma is not None else self.cfg.tracking_anchor_ang_vel.sigma
        reward = _tracking_exp(ang_vel_error, s)
        return reward * self.cfg.tracking_anchor_ang_vel.weight

    # ─── Regularization rewards ───

    def action_rate(self, actions: np.ndarray, last_actions: np.ndarray) -> np.ndarray:
        """Action rate penalty: ||a_t - a_{t-1}||^2."""
        penalty = np.sum((actions - last_actions) ** 2, axis=-1)
        return penalty * self.cfg.action_rate.weight

    def joint_acc(self, joint_vel: np.ndarray) -> np.ndarray:
        """Joint acceleration penalty (proxy: ||qvel||^2)."""
        penalty = np.sum(joint_vel ** 2, axis=-1)
        return penalty * self.cfg.joint_acc.weight

    def torque_limit(self, torques: np.ndarray, limits: np.ndarray) -> np.ndarray:
        """Soft torque limit penalty."""
        ratio = np.clip(np.abs(torques) / (limits + 1e-6), 0, 1)
        penalty = np.sum(ratio ** 2, axis=-1)
        return penalty * self.cfg.torque_limit.weight

    def self_collision(self, collision_forces: np.ndarray, threshold: float = 100.0) -> np.ndarray:
        """Self-collision penalty."""
        violation = np.sum(collision_forces > threshold, axis=-1).astype(np.float64)
        return violation * self.cfg.self_collision.weight

    def feet_slip(
        self,
        foot_vel: np.ndarray,
        foot_contact: np.ndarray,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """Foot slip penalty: penalize foot velocity when in contact."""
        slip = np.sum(
            (np.linalg.norm(foot_vel, axis=-1) ** 2) * (foot_contact > 0),
            axis=-1,
        )
        return slip * self.cfg.feet_slip.weight

    def termination(self, terminated: np.ndarray) -> np.ndarray:
        """Termination penalty."""
        return terminated.astype(np.float64) * self.cfg.termination.weight

    def compute_all(
        self,
        *,
        joint_pos_error: np.ndarray,
        joint_vel_error: np.ndarray,
        anchor_pos_error: np.ndarray,
        anchor_ori_error: np.ndarray,
        anchor_lin_vel_error: np.ndarray,
        anchor_ang_vel_error: np.ndarray,
        body_pos_error: np.ndarray | None = None,
        num_bodies: int = 1,
        actions: np.ndarray | None = None,
        last_actions: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        torques: np.ndarray | None = None,
        torque_limits: np.ndarray | None = None,
        collision_forces: np.ndarray | None = None,
        terminated: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute total reward and per-term breakdown.

        Returns:
            (total_rewards, reward_dict) where total_rewards is (num_envs,)
            and reward_dict maps term names to (num_envs,) arrays.
        """
        rewards = np.zeros(self.num_envs)
        terms: dict[str, np.ndarray] = {}

        # Tracking rewards
        terms["tracking_joint_pos"] = self.tracking_joint_pos(joint_pos_error)
        terms["tracking_joint_vel"] = self.tracking_joint_vel(joint_vel_error)
        terms["tracking_anchor_pos"] = self.tracking_anchor_pos(anchor_pos_error)
        terms["tracking_anchor_ori"] = self.tracking_anchor_ori(anchor_ori_error)
        terms["tracking_anchor_lin_vel"] = self.tracking_anchor_lin_vel(anchor_lin_vel_error)
        terms["tracking_anchor_ang_vel"] = self.tracking_anchor_ang_vel(anchor_ang_vel_error)

        if body_pos_error is not None:
            terms["tracking_body_pos"] = self.tracking_body_pos(body_pos_error, num_bodies)

        # Regularization
        if actions is not None and last_actions is not None:
            terms["action_rate"] = self.action_rate(actions, last_actions)
        if joint_vel is not None:
            terms["joint_acc"] = self.joint_acc(joint_vel)
        if torques is not None and torque_limits is not None:
            terms["torque_limit"] = self.torque_limit(torques, torque_limits)
        if collision_forces is not None:
            terms["self_collision"] = self.self_collision(collision_forces)
        if terminated is not None:
            terms["termination"] = self.termination(terminated)

        # Sum
        for name, val in terms.items():
            rewards += val
            if name not in self.episode_sums:
                self.episode_sums[name] = np.zeros(self.num_envs)
            self.episode_sums[name] += val

        return rewards, terms
