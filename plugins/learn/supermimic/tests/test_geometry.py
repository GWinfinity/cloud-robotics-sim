"""Numerical and derivative tests for batched SO(3) geometry."""

from __future__ import annotations

import math

import pytest
import torch
from supermimic.geometry import (
    mat_to_quat,
    quat_to_mat,
    so3_exp,
    so3_log,
)
from supermimic.networks import WorldModel


def _rotation_vectors(dtype: torch.dtype) -> torch.Tensor:
    near_pi = math.pi - 1e-5
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1e-7, 0.0, 0.0],
            [0.3, -0.4, 0.5],
            [2.0, -1.0, 0.5],
            [near_pi, 0.0, 0.0],
            [0.0, near_pi, 0.0],
            [0.0, 0.0, near_pi],
            [math.pi, 0.0, 0.0],
            [0.0, math.pi, 0.0],
            [0.0, 0.0, math.pi],
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-6), (torch.float64, 1e-10, 1e-10)],
)
def test_so3_exp_returns_batched_rotation_matrices(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    rotations = so3_exp(_rotation_vectors(dtype))

    assert rotations.shape == (10, 3, 3)
    identity = torch.eye(3, dtype=dtype).expand_as(rotations)
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations, identity, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(10, dtype=dtype), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-6), (torch.float64, 1e-10, 1e-10)],
)
def test_matrix_quaternion_round_trip_preserves_rotation(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    rotations = so3_exp(_rotation_vectors(dtype))

    quaternions = mat_to_quat(rotations)
    recovered = quat_to_mat(quaternions)

    assert quaternions.shape == (10, 4)
    torch.testing.assert_close(
        quaternions.norm(dim=-1), torch.ones(10, dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(recovered, rotations, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-6), (torch.float64, 1e-9, 1e-9)],
)
def test_so3_log_exp_recovers_non_singular_rotation_vectors(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    vectors = _rotation_vectors(dtype)[:7]

    recovered = so3_log(so3_exp(vectors))

    torch.testing.assert_close(recovered, vectors, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-6), (torch.float64, 1e-9, 1e-9)],
)
def test_so3_log_round_trip_at_pi_uses_rotation_equivalence(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    rotations = so3_exp(_rotation_vectors(dtype))

    recovered = so3_exp(so3_log(rotations))

    torch.testing.assert_close(recovered, rotations, rtol=rtol, atol=atol)


def test_so3_exp_double_gradcheck() -> None:
    vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], [0.2, -0.1, 0.3], [1.0, -1.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(so3_exp, (vectors,))


def test_so3_log_double_gradcheck() -> None:
    vectors = torch.tensor([[0.2, -0.1, 0.3], [1.0, -0.5, 0.7]], dtype=torch.float64)
    rotations = so3_exp(vectors).requires_grad_()

    assert torch.autograd.gradcheck(so3_log, (rotations,))


def test_world_model_step_rotation_increment_double_gradcheck() -> None:
    model = WorldModel(hidden=(), num_joints=1, context_dim=1).double()
    output = model.net[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.bias[3:6] = torch.tensor([0.15, -0.1, 0.2], dtype=torch.float64)
    rotation = so3_exp(torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64))
    state = (
        torch.zeros(1, 3, dtype=torch.float64),
        rotation,
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
    )
    action = torch.zeros(1, 1, dtype=torch.float64)
    context = torch.zeros(1, 1, dtype=torch.float64)

    def next_rotation(increment: torch.Tensor) -> torch.Tensor:
        return model.step(state, action, context)[1]

    assert torch.autograd.gradcheck(next_rotation, (output.bias,))
