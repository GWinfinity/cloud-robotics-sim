"""Offline trajectory augmentation for RoboTwin-format grasp episodes.

Operates on the HDF5 episodes written by :class:`EpisodeRecorder` (with the
``/qpos``, ``/endpose`` and ``/extra/*`` datasets produced by the grasp
benchmark). All transforms here are **kinematically consistent**:

- :func:`add_observation_noise` — Gaussian jitter on positions / joint angles
  (quaternions are perturbed and re-normalized);
- :func:`time_warp` — resample the episode in time (linear positions,
  nlerp'd quaternions), simulating faster/slower executions;
- :func:`subsample` — keep a random fraction of frames (phase order kept).

**Spatial diversity (different spawn / target poses) cannot be generated
offline** without re-solving IK for every frame — use the runner's online
augmentation instead (``--episodes-per-class N --jitter-xy 0.05``), which
re-simulates with perturbed poses and stays physically consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "EpisodeData",
    "add_observation_noise",
    "augment_episode",
    "augment_file",
    "load_episode",
    "save_episode",
    "subsample",
    "time_warp",
]


@dataclass
class EpisodeData:
    """In-memory view of one recorded episode."""

    qpos: np.ndarray  # (T, D) joint positions
    endpose: np.ndarray  # (T, 7) [x,y,z, qw,qx,qy,qz]
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.qpos)


def load_episode(path: str | Path) -> EpisodeData:
    """Load a RoboTwin-format HDF5 episode."""
    with h5py.File(path, "r") as f:
        extras: dict[str, np.ndarray] = {}
        if "extra" in f:
            for name in f["extra"]:
                extras[name] = np.asarray(f["extra"][name])
        attrs = {k: v for k, v in f.attrs.items()}
        return EpisodeData(
            qpos=np.asarray(f["qpos"]),
            endpose=np.asarray(f["endpose"]),
            extras=extras,
            attrs=attrs,
        )


def save_episode(ep: EpisodeData, path: str | Path) -> Path:
    """Write an episode back to RoboTwin-format HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in ep.attrs.items():
            f.attrs[key] = value
        f.attrs["n_frames"] = ep.n_frames
        f.create_dataset("qpos", data=ep.qpos)
        f.create_dataset("endpose", data=ep.endpose)
        if ep.extras:
            group = f.create_group("extra")
            for name, data in ep.extras.items():
                group.create_dataset(name, data=data)
    return path


def _renorm(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return quats / norms


def add_observation_noise(
    ep: EpisodeData,
    pos_sigma: float = 0.002,
    joint_sigma: float = 0.005,
    seed: int | None = None,
) -> EpisodeData:
    """Add Gaussian noise to joint angles and pose positions."""
    rng = np.random.default_rng(seed)
    qpos = ep.qpos + rng.normal(0.0, joint_sigma, ep.qpos.shape)
    endpose = ep.endpose.copy()
    endpose[:, :3] += rng.normal(0.0, pos_sigma, (ep.n_frames, 3))
    endpose[:, 3:] = _renorm(
        endpose[:, 3:] + rng.normal(0.0, pos_sigma, (ep.n_frames, 4))
    )
    extras: dict[str, np.ndarray] = {}
    for name, data in ep.extras.items():
        noisy = data.astype(np.float64)
        if name.endswith("pose") and data.shape[-1] == 7:
            noisy[:, :3] += rng.normal(0.0, pos_sigma, (len(data), 3))
            noisy[:, 3:] = _renorm(noisy[:, 3:])
        elif name != "phase":  # keep labels exact
            noisy += rng.normal(0.0, joint_sigma, data.shape)
        extras[name] = noisy
    attrs = dict(ep.attrs)
    attrs["meta/augmentation"] = f"noise(pos={pos_sigma},joint={joint_sigma})"
    return EpisodeData(qpos=qpos, endpose=endpose, extras=extras, attrs=attrs)


def _resample(series: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(series))
    out = np.stack(
        [np.interp(t_new, t_old, series[:, d]) for d in range(series.shape[1])],
        axis=1,
    )
    return out


def time_warp(ep: EpisodeData, factor: float) -> EpisodeData:
    """Resample the episode to ``factor * T`` frames (nlerp'd quaternions)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    n_new = max(2, int(round(ep.n_frames * factor)))
    t_new = np.linspace(0.0, 1.0, n_new)
    qpos = _resample(ep.qpos, t_new)
    endpose = _resample(ep.endpose, t_new)
    endpose[:, 3:] = _renorm(endpose[:, 3:])
    extras: dict[str, np.ndarray] = {}
    for name, data in ep.extras.items():
        if name == "phase":  # nearest-neighbor for integer labels
            idx = np.clip(
                (t_new * (len(data) - 1)).round().astype(int), 0, len(data) - 1
            )
            extras[name] = data[idx]
        else:
            resampled = _resample(data, t_new)
            if name.endswith("pose") and data.shape[-1] == 7:
                resampled[:, 3:] = _renorm(resampled[:, 3:])
            extras[name] = resampled
    attrs = dict(ep.attrs)
    attrs["meta/augmentation"] = f"time_warp({factor:.3f})"
    return EpisodeData(qpos=qpos, endpose=endpose, extras=extras, attrs=attrs)


def subsample(
    ep: EpisodeData, keep_ratio: float = 0.8, seed: int | None = None
) -> EpisodeData:
    """Keep a random ``keep_ratio`` fraction of frames (order preserved)."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    n_keep = max(2, int(round(ep.n_frames * keep_ratio)))
    idx = np.sort(rng.choice(ep.n_frames, size=n_keep, replace=False))
    attrs = dict(ep.attrs)
    attrs["meta/augmentation"] = f"subsample({keep_ratio:.2f})"
    return EpisodeData(
        qpos=ep.qpos[idx],
        endpose=ep.endpose[idx],
        extras={name: data[idx] for name, data in ep.extras.items()},
        attrs=attrs,
    )


def augment_episode(
    ep: EpisodeData,
    n_copies: int,
    pos_sigma: float = 0.002,
    joint_sigma: float = 0.005,
    warp_range: tuple[float, float] = (0.85, 1.2),
    keep_ratio_range: tuple[float, float] = (0.85, 1.0),
    seed: int | None = None,
) -> list[EpisodeData]:
    """Generate ``n_copies`` augmented variants (noise + time-warp + subsample)."""
    rng = np.random.default_rng(seed)
    out: list[EpisodeData] = []
    for _ in range(n_copies):
        copy_seed = int(rng.integers(0, 2**31 - 1))
        aug = add_observation_noise(
            ep, pos_sigma=pos_sigma, joint_sigma=joint_sigma, seed=copy_seed
        )
        factor = float(rng.uniform(*warp_range))
        aug = time_warp(aug, factor)
        keep = float(rng.uniform(*keep_ratio_range))
        if keep < 1.0:
            aug = subsample(aug, keep_ratio=keep, seed=copy_seed)
        aug.attrs["meta/augmentation"] = (
            f"combo(noise={pos_sigma}/{joint_sigma},warp={factor:.3f},keep={keep:.2f})"
        )
        out.append(aug)
    return out


def augment_file(
    in_path: str | Path,
    out_dir: str | Path,
    n_copies: int = 3,
    seed: int | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Augment one HDF5 episode file; writes ``<stem>_aug<i>.hdf5`` copies."""
    in_path = Path(in_path)
    ep = load_episode(in_path)
    written: list[Path] = []
    for i, aug in enumerate(augment_episode(ep, n_copies, seed=seed, **kwargs)):
        src = aug.attrs.get("meta/class_name", in_path.stem)
        aug.attrs["meta/augmented_from"] = in_path.name
        aug.attrs["meta/augmentation_idx"] = i
        aug.attrs["task_name"] = f"{aug.attrs.get('task_name', 'episode')}_aug{i}"
        del src
        written.append(save_episode(aug, Path(out_dir) / f"{in_path.stem}_aug{i}.hdf5"))
    return written
