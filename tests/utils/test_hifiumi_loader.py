"""Tests for the HiFi-UMI loader math utilities (no Genesis / no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest

from examples.hifiumi.hifiumi_loader import (
    HandTrajectory,
    quat_angle_deg,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    recenter_to_workspace,
    resample_trajectory,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
    slerp,
)


def _rotz_matrix(deg: float) -> np.ndarray:
    th = np.radians(deg)
    return np.array(
        [[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]]
    )


class TestRot6D:
    """rot6d -> matrix/quaternion conversions."""

    def test_identity(self) -> None:
        rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        rot = rot6d_to_matrix(rot6d)
        assert np.allclose(rot, np.eye(3), atol=1e-12)

    def test_rotz90(self) -> None:
        expected = _rotz_matrix(90.0)
        rot6d = np.concatenate([expected[0], expected[1]])
        rot = rot6d_to_matrix(rot6d)
        assert np.allclose(rot, expected, atol=1e-10)

    def test_batched_and_orthonormal(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(16, 6))
        rot = rot6d_to_matrix(raw)
        eye = np.eye(3)
        for r in rot:
            assert np.allclose(r @ r.T, eye, atol=1e-10)
            assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10)

    def test_quat_roundtrip(self) -> None:
        # rot6d -> matrix -> quat -> rotation angle should match the source rotation.
        expected = _rotz_matrix(37.0)
        rot6d = np.concatenate([expected[0], expected[1]])
        quat = rot6d_to_quat_wxyz(rot6d)
        assert np.isclose(quat_angle_deg(quat), 37.0, atol=1e-6)


class TestQuatOps:
    """Quaternion product/conjugate helpers."""

    def test_mul_identity(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        eye = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(quat_mul_wxyz(q, eye), q, atol=1e-12)

    def test_conjugate_inverse(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        prod = quat_mul_wxyz(q, quat_conjugate_wxyz(q))
        assert np.allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


class TestSlerp:
    """Slerp interpolation behavior."""

    def test_endpoints(self) -> None:
        q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q1 = np.array([[0.0, 0.0, 0.0, 1.0]])
        out = slerp(q0, q1, np.array([0.0]))
        assert np.allclose(out[0], q0[0], atol=1e-10)
        out = slerp(q0, q1, np.array([1.0]))
        assert np.allclose(out[0], q1[0], atol=1e-10)

    def test_midpoint_angle(self) -> None:
        q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
        # 90 deg about z.
        q1 = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])
        mid = slerp(q0, q1, np.array([0.5]))[0]
        assert np.isclose(quat_angle_deg(mid), 45.0, atol=1e-6)

    def test_sign_flip_shortest_path(self) -> None:
        q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q1 = -q0.copy()  # same rotation, opposite sign
        mid = slerp(q0, q1, np.array([0.5]))[0]
        assert np.isclose(quat_angle_deg(mid), 0.0, atol=1e-6)


class TestResample:
    """Trajectory resampling to a uniform rate."""

    def _hand(self, n: int = 101) -> HandTrajectory:
        t = np.linspace(0.0, 4.0, n)
        pos = np.stack([t, np.zeros(n), np.zeros(n)], axis=1)
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        gripper = t.copy()
        return HandTrajectory(
            pos=pos, quat=quat, gripper=gripper, valid=np.ones(n, dtype=bool)
        )

    def test_output_length_and_rate(self) -> None:
        hand = self._hand()
        t = np.linspace(0.0, 4.0, 101)
        t_out, pos, quat, gripper = resample_trajectory(hand, t, 125.0)
        assert len(t_out) == 501  # 4 s * 125 Hz + 1
        assert np.allclose(np.diff(t_out), 1.0 / 125.0, atol=1e-12)
        assert (
            pos.shape == (501, 3) and quat.shape == (501, 4) and gripper.shape == (501,)
        )

    def test_linear_interp_values(self) -> None:
        hand = self._hand()
        t = np.linspace(0.0, 4.0, 101)
        _, pos, _, gripper = resample_trajectory(hand, t, 250.0)
        # pos x == t along the whole trajectory.
        t_out = np.linspace(0.0, 4.0, len(pos))
        assert np.allclose(pos[:, 0], t_out, atol=1e-10)
        assert np.allclose(gripper, t_out, atol=1e-10)


class TestRecenter:
    """Workspace recentering."""

    def test_anchor(self) -> None:
        pos = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]])
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        new_pos, new_quat = recenter_to_workspace(pos, quat, np.array([0.4, 0.0, 0.3]))
        assert np.allclose(new_pos[0], [0.4, 0.0, 0.3])
        assert np.allclose(new_pos[1] - new_pos[0], pos[1] - pos[0])
        assert np.allclose(new_quat, quat)


@pytest.mark.slow
class TestRealShard:
    """Requires the downloaded shard under data/hifiumi (skipped if absent)."""

    def test_load_first_episode(self) -> None:
        from pathlib import Path

        root = Path("data/hifiumi/chunk-0000/part-0000")
        if not (root / "meta/info.json").exists():
            pytest.skip("HiFi-UMI shard not downloaded")
        from examples.hifiumi.hifiumi_loader import load_episode

        ep = load_episode(root, 0)
        assert ep.fps == 25.0
        assert ep.right.pos.shape[1] == 3
        assert ep.right.quat.shape[1] == 4
        norms = np.linalg.norm(ep.right.quat, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
