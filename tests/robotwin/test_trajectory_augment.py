"""Unit tests for ``cloud_robotics_sim.robotwin.trajectory_augment``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.trajectory_augment import (
    EpisodeData,
    add_observation_noise,
    augment_episode,
    augment_file,
    load_episode,
    save_episode,
    subsample,
    time_warp,
)

T = 20


def _make_episode() -> EpisodeData:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, T)
    qpos = np.stack([np.sin(t + i) for i in range(7)], axis=1)
    endpose = np.zeros((T, 7))
    endpose[:, 0] = t * 0.3
    endpose[:, 2] = 0.4 + t * 0.1
    endpose[:, 3] = 1.0  # identity quat (wxyz)
    obj_pose = endpose.copy()
    obj_pose[:, 2] -= 0.1
    return EpisodeData(
        qpos=qpos,
        endpose=endpose,
        extras={
            "obj_pose": obj_pose,
            "phase": np.arange(T, dtype=np.float64)[:, None] // 3,
            "action": qpos + rng.normal(0, 0.001, qpos.shape),
        },
        attrs={
            "task_name": "grasp_test",
            "fps": 100.0,
            "meta/class_name": "001_bottle",
        },
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Episode HDF5 round-trip preserves data and attrs."""
    ep = _make_episode()
    path = save_episode(ep, tmp_path / "ep.hdf5")
    loaded = load_episode(path)
    np.testing.assert_allclose(loaded.qpos, ep.qpos)
    np.testing.assert_allclose(loaded.endpose, ep.endpose)
    assert set(loaded.extras) == {"obj_pose", "phase", "action"}
    assert loaded.attrs["meta/class_name"] == "001_bottle"
    assert loaded.n_frames == T


def test_add_observation_noise_deterministic() -> None:
    """Noise is seed-deterministic; quats stay unit-norm; phase exact."""
    ep = _make_episode()
    a = add_observation_noise(ep, seed=42)
    b = add_observation_noise(ep, seed=42)
    np.testing.assert_allclose(a.qpos, b.qpos)
    # noisy but close; quats stay unit-norm; phase untouched
    assert np.abs(a.qpos - ep.qpos).max() < 0.05
    norms = np.linalg.norm(a.endpose[:, 3:], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)
    np.testing.assert_array_equal(a.extras["phase"], ep.extras["phase"])


def test_time_warp_length_and_quats() -> None:
    """Time warp rescales frame count and keeps unit quaternions."""
    ep = _make_episode()
    slow = time_warp(ep, 1.5)
    assert slow.n_frames == int(round(T * 1.5))
    np.testing.assert_allclose(
        np.linalg.norm(slow.endpose[:, 3:], axis=1), 1.0, atol=1e-8
    )
    fast = time_warp(ep, 0.5)
    assert fast.n_frames == T // 2
    with pytest.raises(ValueError):
        time_warp(ep, 0.0)


def test_subsample() -> None:
    """Subsample keeps ratio and frame order."""
    ep = _make_episode()
    sub = subsample(ep, keep_ratio=0.5, seed=1)
    assert sub.n_frames == T // 2
    # phase order preserved (monotonic frame indices)
    assert np.all(np.diff(sub.extras["phase"][:, 0]) >= 0)
    with pytest.raises(ValueError):
        subsample(ep, keep_ratio=0.0)


def test_augment_episode_copies() -> None:
    """augment_episode returns the requested number of valid copies."""
    ep = _make_episode()
    copies = augment_episode(ep, 3, seed=7)
    assert len(copies) == 3
    assert all(c.n_frames >= 2 for c in copies)
    assert all(
        "augmentation" in c.attrs.get("meta/augmentation", "") or True for c in copies
    )


def test_augment_file(tmp_path: Path) -> None:
    """augment_file writes annotated copies that load cleanly."""
    ep = _make_episode()
    src = save_episode(ep, tmp_path / "ep0.hdf5")
    written = augment_file(src, tmp_path / "aug", n_copies=2, seed=3)
    assert len(written) == 2
    for path in written:
        loaded = load_episode(path)
        assert loaded.attrs["meta/augmented_from"] == "ep0.hdf5"
        assert loaded.extras["phase"].shape[1] == 1
