"""World model, goal-conditioned policy, and critic for SuperMimic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .contracts import State
from .geometry import invariant_features, so3_exp, so3_log, yaw_mat


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden_dim in hidden:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean and variance, frozen once ``max_count`` is reached."""

    no_sync: bool = False

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        mean = self.mean + delta * batch_count / total
        var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        ) / total
        if RunningNorm.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, total, self.count))
            return
        if bool(self.count >= self.max_count):
            return
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an MLP under optional autocast, returning fp32 for state arithmetic."""
    if AMP_DTYPE is None:
        return net(x)
    device_type = x.device.type
    if device_type not in ("cuda", "mps"):
        return net(x)
    with torch.autocast(device_type, dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Predict frame-local state increments and integrate them in fp32."""

    DRIFT_WEIGHT = 0.625

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        num_joints: int = 29,
        context_dim: int = 33,
        structured: bool = False,
        dt: float = 0.02,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.context_dim = context_dim
        self.structured = structured
        self.dt = dt
        self.in_dim = 13 + 3 * num_joints + context_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(
        self, state: State, action: torch.Tensor, normalized_context: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((invariant_features(state), action, normalized_context), -1)

    def _drift(
        self,
        state: State,
        next_lin_vel: torch.Tensor,
        next_ang_vel: torch.Tensor,
        next_joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position, rotation, lin_vel, ang_vel, _, joint_vel = state
        heading = yaw_mat(rotation)
        old_weight = self.dt * (1.0 - self.DRIFT_WEIGHT)
        new_weight = self.dt * self.DRIFT_WEIGHT
        position_drift = (
            heading.transpose(-1, -2)
            @ (old_weight * lin_vel + new_weight * next_lin_vel).unsqueeze(-1)
        ).squeeze(-1)
        rotation_drift = (
            rotation.transpose(-1, -2)
            @ (old_weight * ang_vel + new_weight * next_ang_vel).unsqueeze(-1)
        ).squeeze(-1)
        joint_drift = old_weight * joint_vel + new_weight * next_joint_vel
        return position_drift, rotation_drift, joint_drift

    def targets(self, state: State, next_state: State) -> torch.Tensor:
        """Return frame-local increments from ``state`` to ``next_state``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        (
            next_position,
            next_rotation,
            next_lin_vel,
            next_ang_vel,
            next_joint_pos,
            next_joint_vel,
        ) = next_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        increment = torch.cat(
            (
                (
                    heading.transpose(-1, -2) @ (next_position - position).unsqueeze(-1)
                ).squeeze(-1),
                so3_log(rotation_t @ next_rotation),
                (rotation_t @ (next_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (next_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )
        if not self.structured:
            return increment
        position_drift, rotation_drift, joint_drift = self._drift(
            state, next_lin_vel, next_ang_vel, next_joint_vel
        )
        sizes = (3, 3, 3, 3, self.num_joints, self.num_joints)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = (
            increment.split(sizes, -1)
        )
        return torch.cat(
            (
                d_position - position_drift,
                d_rotation - rotation_drift,
                d_lin_vel,
                d_ang_vel,
                d_joint_pos - joint_drift,
                d_joint_vel,
            ),
            -1,
        )

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        heading = yaw_mat(rotation)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = (
            increment.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        )
        next_lin_vel = lin_vel + (rotation @ d_lin_vel.unsqueeze(-1)).squeeze(-1)
        next_ang_vel = ang_vel + (rotation @ d_ang_vel.unsqueeze(-1)).squeeze(-1)
        next_joint_vel = joint_vel + d_joint_vel
        if self.structured:
            position_drift, rotation_drift, joint_drift = self._drift(
                state, next_lin_vel, next_ang_vel, next_joint_vel
            )
            d_position = d_position + position_drift
            d_rotation = d_rotation + rotation_drift
            d_joint_pos = d_joint_pos + joint_drift
        next_position = position + (heading @ d_position.unsqueeze(-1)).squeeze(-1)
        next_rotation = rotation @ so3_exp(d_rotation)
        return (
            next_position,
            next_rotation,
            next_lin_vel,
            next_ang_vel,
            joint_pos + d_joint_pos,
            next_joint_vel,
        )

    def step(
        self,
        state: State,
        action: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> State:
        prediction = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context)),
        )
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        return self.integrate(state, increment)

    @staticmethod
    def state_error(predicted_state: State, state: State) -> torch.Tensor:
        """Return frame-local errors in the same layout as ``targets``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        (
            predicted_position,
            predicted_rotation,
            predicted_lin_vel,
            predicted_ang_vel,
            predicted_joint_pos,
            predicted_joint_vel,
        ) = predicted_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (
                    heading.transpose(-1, -2)
                    @ (predicted_position - position).unsqueeze(-1)
                ).squeeze(-1),
                so3_log(rotation_t @ predicted_rotation),
                (rotation_t @ (predicted_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (predicted_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                predicted_joint_pos - joint_pos,
                predicted_joint_vel - joint_vel,
            ),
            -1,
        )


class Policy(nn.Module):
    """Encode a goal and decode it together with an observation history."""

    ref_q0: torch.Tensor
    ref_scale: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        num_joints: int = 29,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
        ref_skip: tuple[int, torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.hist = hist
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer("ref_q0", ref_skip[1].detach().clone().float())
            self.register_buffer("ref_scale", ref_skip[2].detach().clone().float())
        self.enc = mlp(goal_dim, hidden, z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, hidden, num_joints)
        output = self.dec[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Consume ``(batch, hist, obs_dim)`` observations, newest last."""
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        obs = self.obs_norm(obs_hist).flatten(-2)
        action = run_mlp(self.dec, torch.cat((latent, obs), -1))
        if self.ref_off is not None:
            reference_q = goal[..., self.ref_off : self.ref_off + action.shape[-1]]
            action = action + (reference_q - self.ref_q0) / self.ref_scale
        return action


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


__all__ = ["AMP_DTYPE", "Critic", "Policy", "RunningNorm", "WorldModel", "run_mlp"]
