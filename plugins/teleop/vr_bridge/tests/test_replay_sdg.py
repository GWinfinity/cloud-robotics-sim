"""Offline tests for replay_sdg helpers (no Genesis required).

Covers the dreamdojo-layout append path used to accumulate replayed
variants, the source-action loader, and the pose-jitter helpers, plus
cross-plugin loadability of a replay_sdg-produced file through
``dreamdojo.core.dataset.GenesisDataset`` (same contract as
``test_e2e_recording`` for teleop-recorded files).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from replay_sdg import (  # noqa: E402
    append_dreamdojo_episode,
    jitter_camera,
    jitter_xy_yaw,
    load_source_actions,
    subsample_actions,
    yaw_to_quat,
)


def _fake_episode(rng: np.random.Generator, n_frames: int, n_dofs: int = 9):
    """Synthetic (observations, actions) pair mimicking a replayed variant."""
    observations = rng.integers(0, 255, size=(n_frames, 8, 16, 3), dtype=np.uint8)
    actions = rng.standard_normal((n_frames, n_dofs)).astype(np.float64)
    return observations, actions


def test_load_source_actions_reads_dreamdojo_layout(tmp_path):
    """Round-trip through TeleopRecorder.save_hdf5's layout."""
    src = tmp_path / "teleop.h5"
    qpos = np.arange(60, dtype=np.float64).reshape(20, 3)
    rgb = np.zeros((20, 8, 16, 3), dtype=np.uint8)
    with h5py.File(src, "w") as h5:
        group = h5.create_group("episode_0")
        group.create_dataset("observations", data=rgb)
        group.create_dataset("actions", data=qpos.astype(np.float32))

    actions = load_source_actions(src, episode=0)
    assert actions.shape == (20, 3)
    np.testing.assert_allclose(actions, qpos, atol=1e-6)  # f32 -> f64 loss

    with pytest.raises(KeyError, match="episode_1"):
        load_source_actions(src, episode=1)


def test_append_dreamdojo_episode_accumulates_and_loads(tmp_path):
    """Two appends land as episode_0/episode_1 and GenesisDataset loads them."""
    from dreamdojo.core.dataset import GenesisDataset

    out = tmp_path / "sdg.h5"
    rng = np.random.default_rng(0)
    for variant in range(2):
        observations, actions = _fake_episode(rng, n_frames=12)
        path = append_dreamdojo_episode(out, observations, actions)
        assert path == out

    with h5py.File(out, "r") as h5:
        assert sorted(h5.keys()) == ["episode_0", "episode_1"]
        assert h5["episode_0"]["observations"].shape == (12, 8, 16, 3)
        assert h5["episode_0"]["observations"].dtype == np.uint8
        assert h5["episode_0"]["actions"].dtype == np.float32
        assert h5["episode_1"]["actions"].shape == (12, 9)

    ds = GenesisDataset(
        pre_generated_path=str(out),
        num_frames=4,
        robot_type="franka",
        device="cpu",
    )
    # GenesisDataset keeps its configured num_episodes and cycles the
    # episodes actually loaded from the file (same contract as
    # test_e2e_recording for teleop-recorded files).
    assert len(ds.pre_generated_data) == 2
    assert len(ds) == ds.num_episodes
    sample = ds[1]
    assert sample["video"].shape == (4, 3, 8, 16)
    assert sample["action"].shape == (4, 9)


def test_append_dreamdojo_episode_validates_shapes(tmp_path):
    """Bad observation/action shapes are rejected before touching the file."""
    out = tmp_path / "sdg.h5"
    with pytest.raises(ValueError, match="observations must be"):
        append_dreamdojo_episode(
            out, np.zeros((4, 8, 16), dtype=np.uint8), np.zeros((4, 9))
        )
    with pytest.raises(ValueError, match="actions must be"):
        append_dreamdojo_episode(
            out,
            np.zeros((4, 8, 16, 3), dtype=np.uint8),
            np.zeros((3, 9)),  # T mismatch
        )


def test_jitter_xy_yaw_bounds_and_determinism():
    """Jitter respects its amplitude bounds; zero jitter is the identity."""
    rng = np.random.default_rng(42)
    xy, yaw = jitter_xy_yaw(rng, (0.35, -0.15), xy_jitter=0.05)
    assert np.all(np.abs(xy - (0.35, -0.15)) <= 0.05 + 1e-9)
    assert -np.pi <= yaw <= np.pi

    # zero jitter on both axes is the identity
    rng = np.random.default_rng(0)
    xy, yaw = jitter_xy_yaw(rng, (0.35, -0.15), xy_jitter=0.0, yaw_jitter=0.0)
    np.testing.assert_allclose(xy, (0.35, -0.15))
    assert yaw == pytest.approx(0.0)


def test_jitter_camera_zero_is_identity():
    """Zero camera jitter returns the nominal pose unchanged."""
    rng = np.random.default_rng(0)
    pos, lookat = jitter_camera(rng, (1.5, 0.0, 1.2), (0.1, 0.0, 0.85), jitter=0.0)
    np.testing.assert_allclose(pos, (1.5, 0.0, 1.2))
    np.testing.assert_allclose(lookat, (0.1, 0.0, 0.85))


def test_yaw_to_quat_and_subsample():
    """Yaw quaternion is unit-norm; stride/max_frames thin the action track."""
    quat = yaw_to_quat(np.pi / 2)
    assert quat[0] == pytest.approx(np.cos(np.pi / 4))
    assert quat[3] == pytest.approx(np.sin(np.pi / 4))
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0)

    actions = np.arange(40, dtype=np.float64).reshape(20, 2)
    out = subsample_actions(actions, stride=5, max_frames=2)
    np.testing.assert_allclose(out, actions[[0, 5]])
    assert subsample_actions(actions, stride=1, max_frames=0).shape == (20, 2)
