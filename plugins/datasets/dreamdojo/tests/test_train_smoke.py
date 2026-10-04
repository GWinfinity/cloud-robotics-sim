"""Offline tests for the training smoke path (no Genesis required).

Builds a synthetic dreamdojo-layout HDF5, serves it through GenesisDataset,
and runs a few optimizer steps of the tiny one-frame predictor, asserting
finite losses, correct shapes, and actual parameter updates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
_DATASETS = str(Path(__file__).resolve().parents[2])  # plugins/datasets
for _p in (_EXAMPLES, _DATASETS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_smoke import (  # noqa: E402
    OneStepPredictor,
    make_sample_pairs,
    train_steps,
)

from dreamdojo.core.dataset import GenesisDataset  # noqa: E402


@pytest.fixture()
def tiny_dataset(tmp_path):
    """A two-episode dreamdojo-layout HDF5 with 16-frame episodes."""
    path = tmp_path / "tiny.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        for ep in range(2):
            group = h5.create_group(f"episode_{ep}")
            group.create_dataset(
                "observations",
                data=rng.integers(0, 255, size=(16, 32, 32, 3), dtype=np.uint8),
            )
            group.create_dataset(
                "actions", data=rng.standard_normal((16, 9)).astype(np.float32)
            )
    return GenesisDataset(
        pre_generated_path=str(path),
        num_frames=8,
        robot_type="franka",
        device="cpu",
    )


def test_make_sample_pairs_shapes_and_range(tiny_dataset):
    """Pairs are (T-1) windows, frames in [0,1], downscale honored."""
    sample = tiny_dataset[0]
    frames_t, actions_t, frames_t1 = make_sample_pairs(sample, downscale=2)
    assert frames_t.shape == (7, 3, 16, 16)
    assert frames_t1.shape == frames_t.shape
    assert actions_t.shape == (7, 9)
    assert 0.0 <= float(frames_t.min()) and float(frames_t.max()) <= 1.0
    assert not torch.equal(frames_t, frames_t1)


def test_train_steps_finite_and_updates_parameters(tiny_dataset):
    """A few Adam steps produce finite losses and change every parameter."""
    model = OneStepPredictor(action_dim=9, base_channels=4)
    before = [p.detach().clone() for p in model.parameters()]

    losses = train_steps(tiny_dataset, model, steps=5, batch=4, downscale=2, seed=0)
    assert len(losses) == 5
    assert np.all(np.isfinite(losses)), f"non-finite losses: {losses}"

    unchanged = sum(
        1 for p, b in zip(model.parameters(), before) if torch.equal(p.detach(), b)
    )
    assert unchanged == 0, f"{unchanged} parameters were not updated"


def test_train_steps_rejects_empty_batch_window(tiny_dataset):
    """num_frames=2 leaves no (t, t+1) pair -> clear error, not a crash."""
    model = OneStepPredictor(action_dim=9, base_channels=4)
    tiny_dataset.num_frames = 1  # video comes back (1, C, H, W): no pairs
    with pytest.raises((ValueError, IndexError)):
        train_steps(tiny_dataset, model, steps=1, batch=1, seed=0)
