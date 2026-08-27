"""Rectified-flow helpers shared by both generation stages.

Both flow models are trained with velocity-prediction flow matching
under the linear interpolation schedule of Rectified Flow:

    x_t = (1 - t) x0 + t x1,   v* = x1 - x0,   t ~ logit-normal

and sampled with an Euler integrator from ``t = 1`` (noise) to
``t = 0`` (clean).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def sample_t(batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Draw logit-normal timesteps in ``(0, 1)`` (Esser et al. 2024)."""
    return torch.sigmoid(torch.randn(batch, device=device))


def interpolate(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``x_t`` and the velocity target for clean/noise endpoints.

    Args:
        x0: Clean endpoint (data).
        x1: Noise endpoint (Gaussian).
        t: ``(B,)`` timesteps.

    Returns:
        ``(x_t, v_target)`` with the same shape as ``x0``.
    """
    t_ = t.view(-1, *([1] * (x0.ndim - 1)))
    x_t = (1 - t_) * x0 + t_ * x1
    return x_t, x1 - x0


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Integrate the learned velocity field from noise to data.

    Args:
        velocity_fn: ``(x_t, t_batch) -> v`` callable.
        x1: Initial Gaussian noise.
        steps: Number of Euler steps.

    Returns:
        The generated clean endpoint estimate ``x0``.
    """
    x = x1
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device)
        x = x - velocity_fn(x, t) * dt
    return x


def cfg_velocity(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cond_kwargs: dict,
    uncond_kwargs: dict,
    guidance: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a classifier-free-guided velocity callable.

    Args:
        velocity_fn: ``(x, t, **kwargs) -> v`` model call.
        cond_kwargs: Conditional forward kwargs.
        uncond_kwargs: Unconditional forward kwargs.
        guidance: CFG scale (``1.0`` = no guidance).

    Returns:
        A ``(x, t) -> v`` callable for :func:`euler_sample`.
    """

    def call(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_cond = velocity_fn(x, t, **cond_kwargs)
        if guidance == 1.0:
            return v_cond
        v_uncond = velocity_fn(x, t, **uncond_kwargs)
        return v_uncond + guidance * (v_cond - v_uncond)

    return call
