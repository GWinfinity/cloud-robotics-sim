"""trajclass pipeline tests: windowization, serialization, and training.

All tests are Genesis-free — episodes are synthesized with separable
per-class temporal patterns, which also gives the trainer a signal it must
actually learn (a >0.85 val-accuracy bar catches a broken loss/shuffle).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cloud_robotics_sim.trajclass import (
    LABEL_NAMES,
    EpisodeBatch,
    TrainConfig,
    TrajGRU,
    load_npz,
    save_npz,
    train_classifier,
    windows,
)


def _make_batch(num_episodes: int = 24, steps: int = 40) -> EpisodeBatch:
    """Episodes whose class is readable from *every* window.

    Window labels inherit the episode label, so the discriminative pattern
    must span the whole episode: a per-channel offset plus a class-specific
    temporal ripple on channel 0.

    class 0 (no_grasp): zero-mean noise
    class 1 (dropped):   +0.8 offset with a sine ripple
    class 2 (success):   +1.6 offset with a cosine ripple
    """
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 0.01, (num_episodes, steps, 24)).astype(np.float32)
    act = rng.normal(0, 0.01, (num_episodes, steps, 9)).astype(np.float32)
    labels = np.arange(num_episodes) % len(LABEL_NAMES)
    t = np.arange(steps, dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab == 1:
            obs[i, :, 0] += 0.8 + 0.2 * np.sin(t / 3.0)
        elif lab == 2:
            obs[i, :, 0] += 1.6 + 0.2 * np.cos(t / 3.0)
    lengths = np.full(num_episodes, steps, dtype=np.int64)
    synthetic = np.zeros(num_episodes, dtype=bool)
    synthetic[labels == 2] = True
    return EpisodeBatch(
        obs=obs,
        act=act,
        rew=np.zeros((num_episodes, steps)),
        lengths=lengths,
        labels=labels,
        synthetic=synthetic,
    )


def test_windows_shapes_and_labels() -> None:
    """Windows are full-length, correctly labelled, and count-checked."""
    batch = _make_batch()
    x, y = windows(batch, window=20, stride=5)
    assert x.shape[1:] == (20, batch.channels)
    assert len(x) == len(y)
    # episode 0 is class 0; its windows carry exactly one label.
    x0, y0 = windows(batch, window=20, stride=5, episode_indices=[0])
    assert set(y0.tolist()) == {0}
    # T=40, window=20, stride=5 -> starts 0,5,10,15,20 -> 5 windows/ep.
    assert len(x0) == 5
    assert len(x) == 5 * batch.num_episodes


def test_windows_edge_pads_short_episodes() -> None:
    """Early-terminated episodes (shorter than the window) yield one padded
    window whose tail repeats the terminal state, instead of vanishing.
    """
    batch = _make_batch()
    batch.lengths[0] = 10  # shorter than window=20
    x, y = windows(batch, window=20, stride=5)
    # 4 full windows for each remaining episode + 1 padded for episode 0.
    assert len(x) == 5 * (batch.num_episodes - 1) + 1
    padded = x[-1] if y[-1] == batch.labels[0] else x[0]
    # The padded window's tail equals its own final valid row.
    assert np.allclose(padded[-1], padded[9])


def test_npz_roundtrip(tmp_path) -> None:
    """save_npz/load_npz preserves arrays, labels, and counts."""
    batch = _make_batch(num_episodes=6)
    path = tmp_path / "episodes.npz"
    save_npz(batch, path)
    loaded = load_npz(path)
    assert loaded.num_episodes == batch.num_episodes
    np.testing.assert_array_equal(loaded.labels, batch.labels)
    np.testing.assert_allclose(loaded.obs, batch.obs, atol=1e-6)
    assert loaded.label_counts() == batch.label_counts()


def test_episode_batch_validates_shapes() -> None:
    """Mismatched component shapes raise at construction."""
    batch = _make_batch()
    with pytest.raises(ValueError):
        EpisodeBatch(
            obs=batch.obs,
            act=batch.act[:, :10],  # truncated time axis
            rew=batch.rew,
            lengths=batch.lengths,
            labels=batch.labels,
            synthetic=batch.synthetic,
        )


def test_model_forward() -> None:
    """GRU maps (N, W, C) windows to (N, num_classes) logits."""
    model = TrajGRU(in_channels=33, num_classes=3)
    out = model(torch.zeros(4, 25, 33))
    assert tuple(out.shape) == (4, 3)


def test_trainer_learns_separable_patterns() -> None:
    """The trainer must actually fit: separable classes end well above chance."""
    batch = _make_batch(num_episodes=30, steps=50)
    report, model = train_classifier(
        batch, TrainConfig(window=20, stride=5, epochs=15, seed=0)
    )
    assert report["parameters"] < 500_000
    assert report["split"]["train_episodes"] + report["split"]["val_episodes"] == 30
    assert report["val"]["accuracy"] > 0.85
    # Report must be JSON-serializable (pipeline logs it verbatim).
    json.dumps(report)
    # Standardization stats cover every channel.
    std = report["standardization"]["std"]
    assert len(std) == batch.channels
    assert all(s > 0 for s in std)
