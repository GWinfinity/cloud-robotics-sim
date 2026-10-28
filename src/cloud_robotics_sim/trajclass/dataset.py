"""Episode-batch container, npz (de)serialization, and windowization.

The on-disk format is the hand-off between the Genesis data generator
(``tools/gen_traj_classifier_data.py``) and the trainer: padded per-episode
arrays plus lengths, integer labels, and a synthetic mask marking episodes
whose success path was kinematically scripted (the obs signature is real,
the physics path is not — keep them out of real-grasp evaluations).

Everything here is numpy-only so the dataset side stays Genesis-free and
unit-testable without a simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

#: Episode outcome classes, ordered by label index.
LABEL_NAMES: tuple[str, ...] = ("no_grasp", "dropped", "success")


@dataclass
class EpisodeBatch:
    """Padded episode arrays with per-episode labels.

    Attributes:
        obs: ``(E, T, obs_dim)`` per-step observations (zero-padded).
        act: ``(E, T, act_dim)`` per-step applied actions (zero-padded).
        rew: ``(E, T)`` per-step rewards.
        lengths: ``(E,)`` valid steps per episode.
        labels: ``(E,)`` integer indices into :data:`LABEL_NAMES`.
        synthetic: ``(E,)`` whether the episode's outcome was scripted.
    """

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    synthetic: np.ndarray

    def __post_init__(self) -> None:
        """Reject shape-mismatched arrays at construction time."""
        if self.obs.ndim != 3:
            raise ValueError(f"obs must be (E, T, C), got {self.obs.shape}")
        e, t = self.obs.shape[0], self.obs.shape[1]
        if self.act.ndim != 3 or self.act.shape[:2] != (e, t):
            raise ValueError(
                f"act shape {self.act.shape} does not align with obs (E={e}, T={t})"
            )
        if self.rew.shape != (e, t):
            raise ValueError(f"rew shape {self.rew.shape} != {(e, t)}")
        for name, arr in (
            ("lengths", self.lengths),
            ("labels", self.labels),
            ("synthetic", self.synthetic),
        ):
            if arr.shape != (e,):
                raise ValueError(f"{name} shape {arr.shape} != {(e,)}")

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def channels(self) -> int:
        """Window feature width: observations concatenated with actions."""
        return int(self.obs.shape[-1] + self.act.shape[-1])

    def label_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in LABEL_NAMES}
        for lab in self.labels.tolist():
            counts[LABEL_NAMES[int(lab)]] += 1
        return counts


def save_npz(batch: EpisodeBatch, path: Union[str, Path]) -> None:
    """Write the batch to ``.npz`` (padded arrays; compact, tool-agnostic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=batch.obs.astype(np.float32),
        act=batch.act.astype(np.float32),
        rew=batch.rew.astype(np.float32),
        lengths=batch.lengths.astype(np.int64),
        labels=batch.labels.astype(np.int64),
        synthetic=batch.synthetic.astype(bool),
        label_names=np.array(LABEL_NAMES),
    )


def load_npz(path: Union[str, Path]) -> EpisodeBatch:
    """Read a batch written by :func:`save_npz`."""
    data = np.load(Path(path), allow_pickle=False)
    required = ("obs", "act", "rew", "lengths", "labels", "synthetic")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} missing arrays: {missing}")
    stored_names = tuple(str(n) for n in data.get("label_names", np.array(LABEL_NAMES)))
    if stored_names != LABEL_NAMES:
        raise ValueError(f"label names mismatch: {stored_names} != {LABEL_NAMES}")
    return EpisodeBatch(
        obs=data["obs"],
        act=data["act"],
        rew=data["rew"],
        lengths=data["lengths"],
        labels=data["labels"],
        synthetic=data["synthetic"],
    )


def windows(
    batch: EpisodeBatch,
    window: int,
    stride: int = 1,
    episode_indices: Union[np.ndarray, list[int], None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice episodes into fixed-length windows of ``obs‖act`` channels.

    Windows normally start every ``stride`` steps fully inside an episode.
    Early-terminated episodes (e.g. success ending long before the step
    cap) are shorter than ``window`` — silently dropping them would delete
    exactly the class the windows exist to find, so each short episode
    contributes ONE edge-padded window instead (its final state repeated
    to ``window`` length; the terminal state is what decides the outcome).

    Args:
        batch: Padded episode arrays.
        window: Window length in steps (25 steps @ 20 ms control = 500 ms).
        stride: Start-index stride between consecutive windows.
        episode_indices: Optional subset of episode indices to slice (used to
            keep train/val splits episode-disjoint — windows from one episode
            must never straddle the split).

    Returns:
        ``X`` of shape ``(N, window, channels)`` and ``y`` of ``(N,)`` integer
        labels (the source episode's label, broadcast over its windows).
    """
    if window < 1 or stride < 1:
        raise ValueError(f"window/stride must be >= 1, got {window}/{stride}")
    idx = (
        range(batch.num_episodes)
        if episode_indices is None
        else [int(i) for i in episode_indices]
    )
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in idx:
        length = int(batch.lengths[i])
        if length < 1:
            continue
        episode = np.concatenate(
            [batch.obs[i, :length], batch.act[i, :length]], axis=-1
        )
        if length < window:
            xs.append(
                np.concatenate(
                    [episode, np.repeat(episode[-1:], window - length, axis=0)],
                    axis=0,
                )
            )
            ys.append(int(batch.labels[i]))
            continue
        n_before = len(xs)
        for start in range(0, length - window + 1, stride):
            xs.append(episode[start : start + window])
        ys.extend([int(batch.labels[i])] * (len(xs) - n_before))
    if not xs:
        return (
            np.zeros((0, window, batch.channels), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def batch_to_json(batch: EpisodeBatch) -> str:
    """Compact provenance string for reports."""
    return json.dumps(
        {
            "episodes": batch.num_episodes,
            "labels": batch.label_counts(),
            "synthetic": int(batch.synthetic.sum()),
        },
        sort_keys=True,
    )
