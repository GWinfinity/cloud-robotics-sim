"""Minimal reference task implementing the SuperMimic MDP protocol.

A two-joint arm on a fixed base tracks a periodic sinusoid stream. The
reference is a pure function of the frame index — reward, goal frames, and
resets all derive from it — while per-episode ``context = [damping, gain]``
modulates the (real) joint dynamics, so the world model must model
context-dependent physics. This mirrors the humanoid setup where the frame
index selects reference poses and the context carries episode parameters.

Task contract highlights:
- root is constant (upright at z=1); only joint dynamics are learned, which
  keeps CPU smoke tests fast;
- ``goal_q_offset = 0``: with ``pi_ref_skip`` the policy output is a residual
  on the next-frame reference joint position stored at the head of each goal
  frame (``default_q = 0``, ``action_scale = 1``);
- no left/right symmetry, so ``mirror`` is ``None`` (``wm_mirror`` stays 0).
"""

from __future__ import annotations

import torch

from ..contracts import State
from ..geometry import mat_to_6d, yaw_mat

CONTROL_DT = 0.02
NUM_JOINTS = 2
PERIOD = 50  # reference loop length in frames

# Joint frequency multipliers (joint 1 oscillates twice as fast).
_FREQ = torch.tensor([1.0, 2.0])
_REF_AMP = 0.6


def reference(
    frames: torch.Tensor, num_frames: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference joint pos/vel for frame index tensor ``frames`` of any shape.

    Returns ``(q_ref, qd_ref)`` with a trailing ``num_frames`` axis of size 1
    squeezed — shapes match ``frames.unsqueeze(-1)`` broadcast to (..., J).
    """
    phase = frames.unsqueeze(-1) * (2.0 * torch.pi / PERIOD)
    angle = phase * _FREQ
    q_ref = _REF_AMP * torch.sin(angle)
    qd_ref = (
        _REF_AMP * torch.cos(angle) * _FREQ * (2.0 * torch.pi / PERIOD) / CONTROL_DT
    )
    return q_ref, qd_ref


class ToyMdp:
    """Two-joint sinusoid-tracking MDP for the SuperMimic learner."""

    num_joints = NUM_JOINTS
    context_dim = 2
    goal_frame_dim = 2 * NUM_JOINTS  # [q_ref, qd_ref]
    max_step_reward = 1.5 * CONTROL_DT

    def __init__(
        self,
        clip_length: int = 48,
        w_pos: float = 1.0,
        w_vel: float = 0.5,
        w_action_rate: float = 0.1,
    ):
        if clip_length > PERIOD:
            raise ValueError(
                f"clip_length {clip_length} exceeds reference period {PERIOD}"
            )
        self.clip_length = clip_length
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_action_rate = w_action_rate
        # ref-skip support: policy residual on the next-frame reference joint
        # position stored at offset 0 of each goal frame.
        self.goal_q_offset = 0
        self.default_q = torch.zeros(NUM_JOINTS)
        self.action_scale = torch.ones(NUM_JOINTS)
        self.mirror = None

    # -- SuperMimicMdpProtocol -------------------------------------------
    @property
    def obs_dim(self) -> int:
        return 2 * NUM_JOINTS + 2  # [q, qd, root_z, damping]

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        del previous_action, frames
        position, _, _, _, joint_pos, joint_vel = state
        return torch.cat(
            (joint_pos, joint_vel, position[..., 2:3], context[..., :1]), -1
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Stack the next ``num_frames`` reference frames, farthest last.

        Matches the upstream motion-library convention: frames are strictly
        in the future (f+1 .. f+num_frames), so the block at offset 0 is the
        next frame — exactly what ``pi_ref_skip`` slices for the action
        residual.
        """
        parts = []
        for step in range(1, num_frames + 1):
            q_ref, qd_ref = reference(frames + step)
            parts.append(torch.cat((q_ref, qd_ref), -1))
        return torch.cat(parts, -1)

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, _, joint_pos, joint_vel = state
        q_ref, qd_ref = reference(frames)
        pos = torch.exp(-(joint_pos - q_ref).square().mean(-1) / 0.1)
        vel = torch.exp(-(joint_vel - qd_ref).square().mean(-1) / 2.0)
        rate = (action - previous_action).square().mean(-1)
        return (
            self.w_pos * pos + self.w_vel * vel - self.w_action_rate * rate
        ) * CONTROL_DT

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del state
        return torch.zeros(frames.shape, dtype=torch.float32, device=frames.device)

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        _, rotation, _, _, joint_pos, joint_vel = state
        q_ref, qd_ref = reference(frames)
        heading = yaw_mat(rotation)
        return torch.cat(
            (
                joint_pos,
                joint_vel,
                mat_to_6d(heading),
                normalized_context,
                q_ref,
                qd_ref,
            ),
            -1,
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        """Map raw ``[damping, gain]`` to roughly unit-scale features."""
        damping = (context[..., 0:1] - 5.0) / 3.0
        gain = (context[..., 1:2] - 30.0) / 10.0
        return torch.cat((damping, gain), -1)

    def valid_imagination_start(
        self, frames: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        # Episodes always start at frame 0 (see ToyEnv), so the frame index
        # is the episode-local step and this bound is exact.
        return frames + horizon <= self.clip_length

    def to(self, device: str | torch.device) -> "ToyMdp":
        self.default_q = self.default_q.to(device)
        self.action_scale = self.action_scale.to(device)
        return self


__all__ = ["CONTROL_DT", "NUM_JOINTS", "PERIOD", "ToyMdp", "reference"]
