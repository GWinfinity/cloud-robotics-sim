"""Episode data recorder aligned with the RoboTwin data format.

Implements the data layer of the RoboTwin->Genesis migration skeleton
(document sections 7 and 10): per-step frame collection, HDF5 episode
export, MP4 video export, and per-env episode splitting for batched
``n_envs`` collection.

Conventions
-----------
- All captured tensors use the env dim as the first dimension. Single-env
  inputs without a leading env dim are promoted automatically.
- Camera extrinsics are stored as 4x4 camera-to-world transforms.
- Camera intrinsics follow the Genesis pinhole derivation
  (:func:`cloud_robotics_sim.utils.camera.intrinsics_from_fov`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a core dependency
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

__all__ = ["EpisodeRecorder"]


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors / array-likes to numpy arrays."""
    if HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_env_dim(array: np.ndarray, *, sample_ndim: int) -> np.ndarray:
    """Add a leading env dim to a per-sample array if it is missing."""
    if array.ndim == sample_ndim:
        return array[np.newaxis, ...]
    return array


class EpisodeRecorder:
    """Collect per-step frames and export RoboTwin-format HDF5 + MP4.

    Attributes:
        task_name: RoboTwin task identifier stored in the HDF5 attrs.
        fps: Playback frequency used for MP4 export and HDF5 metadata.
        n_envs: Number of parallel environments the captured tensors carry.
    """

    def __init__(
        self,
        task_name: str = "",
        fps: float = 30.0,
        n_envs: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.task_name = task_name
        self.fps = float(fps)
        self.n_envs = int(n_envs)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._frames: list[dict[str, Any]] = []
        self._camera_params: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of captured frames."""
        return len(self._frames)

    def set_camera_params(
        self,
        camera: str,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> None:
        """Register ``(intrinsic 3x3, extrinsic 4x4 camera-to-world)``."""
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be 3x3, got {intrinsic.shape}")
        if extrinsic.shape != (4, 4):
            raise ValueError(f"extrinsic must be 4x4, got {extrinsic.shape}")
        self._camera_params[camera] = (intrinsic, extrinsic)

    def capture(
        self,
        step: int,
        *,
        rgb: dict[str, Any] | None = None,
        depth: dict[str, Any] | None = None,
        segmentation: dict[str, Any] | None = None,
        qpos: Any | None = None,
        endpose: Any | None = None,
    ) -> None:
        """Capture one simulation step.

        Args:
            step: Simulation step index (metadata only).
            rgb: Mapping camera name -> RGB array ``(n_envs, H, W, 3)``.
            depth: Mapping camera name -> depth array ``(n_envs, H, W)``.
            segmentation: Mapping camera name -> seg array ``(n_envs, H, W)``.
            qpos: Joint positions ``(n_envs, D)`` or ``(D,)``.
            endpose: End-effector pose(s) ``(n_envs, 7)`` or ``(7,)`` as
                ``[x, y, z, qw, qx, qy, qz]``.
        """
        frame: dict[str, Any] = {"step": int(step)}
        if rgb:
            frame["rgb"] = {
                cam: _ensure_env_dim(_to_numpy(img), sample_ndim=3)
                for cam, img in rgb.items()
            }
        if depth:
            frame["depth"] = {
                cam: _ensure_env_dim(_to_numpy(img), sample_ndim=2)
                for cam, img in depth.items()
            }
        if segmentation:
            frame["segmentation"] = {
                cam: _ensure_env_dim(_to_numpy(img), sample_ndim=2)
                for cam, img in segmentation.items()
            }
        if qpos is not None:
            frame["qpos"] = _ensure_env_dim(_to_numpy(qpos), sample_ndim=1)
        if endpose is not None:
            frame["endpose"] = _ensure_env_dim(_to_numpy(endpose), sample_ndim=1)
        self._frames.append(frame)

    def clear(self) -> None:
        """Drop all captured frames (camera params are kept)."""
        self._frames.clear()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _stack(
        self, key: str, env_idx: int
    ) -> dict[str, np.ndarray] | np.ndarray | None:
        """Stack a modality across time for a single env."""
        series = [f[key] for f in self._frames if key in f]
        if not series:
            return None
        first = series[0]
        if isinstance(first, dict):
            return {
                cam: np.stack([entry[cam][env_idx] for entry in series])
                for cam in first
            }
        return np.stack([entry[env_idx] for entry in series])

    def save_hdf5(self, path: str | Path, env_idx: int = 0) -> Path:
        """Export one env's episode as a RoboTwin-format HDF5 file.

        Layout::

            /obs/rgb/<cam>          (T, H, W, 3) uint8
            /obs/depth/<cam>        (T, H, W)    float32
            /obs/segmentation/<cam> (T, H, W)    int32
            /qpos                   (T, D)       float64
            /endpose                (T, 7)       float64
            /cameras/<cam>          attrs: intrinsic, extrinsic (cam-to-world)
            attrs: task_name, fps, n_frames, env_idx, metadata
        """
        if not self._frames:
            raise RuntimeError("No frames captured; nothing to save")
        if not 0 <= env_idx < self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for n_envs={self.n_envs}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["task_name"] = self.task_name
            f.attrs["fps"] = self.fps
            f.attrs["n_frames"] = self.n_frames
            f.attrs["env_idx"] = env_idx
            for key, value in self.metadata.items():
                f.attrs[f"meta/{key}"] = value

            obs = f.create_group("obs")
            for modality in ("rgb", "depth", "segmentation"):
                stacked = self._stack(modality, env_idx)
                if isinstance(stacked, dict):
                    group = obs.create_group(modality)
                    for cam, data in stacked.items():
                        group.create_dataset(cam, data=data)

            for key in ("qpos", "endpose"):
                stacked = self._stack(key, env_idx)
                if isinstance(stacked, np.ndarray):
                    f.create_dataset(key, data=stacked)

            cameras = f.create_group("cameras")
            for cam, (intrinsic, extrinsic) in self._camera_params.items():
                node = cameras.create_group(cam)
                node.attrs["intrinsic"] = intrinsic
                node.attrs["extrinsic"] = extrinsic
        return path

    def save_zarr(self, path: str | Path, env_idx: int = 0) -> Path:
        """Export one env's episode as a Zarr store (DP/DP3 consumption).

        Mirrors the HDF5 layout (``obs/<modality>/<cam>``, ``qpos``,
        ``endpose``, plus attrs) so policy training code can switch stores
        without schema changes (migration doc section 7).
        """
        import zarr

        if not self._frames:
            raise RuntimeError("No frames captured; nothing to save")
        if not 0 <= env_idx < self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for n_envs={self.n_envs}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(path), mode="w")
        root.attrs.update(
            {
                "task_name": self.task_name,
                "fps": self.fps,
                "n_frames": self.n_frames,
                "env_idx": env_idx,
                **{f"meta/{k}": v for k, v in self.metadata.items()},
            }
        )
        obs = root.require_group("obs")
        for modality in ("rgb", "depth", "segmentation"):
            stacked = self._stack(modality, env_idx)
            if isinstance(stacked, dict):
                group = obs.require_group(modality)
                for cam, data in stacked.items():
                    group.create_array(cam, data=data)
        for key in ("qpos", "endpose"):
            stacked = self._stack(key, env_idx)
            if isinstance(stacked, np.ndarray):
                root.create_array(key, data=stacked)
        cameras = root.require_group("cameras")
        for cam, (intrinsic, extrinsic) in self._camera_params.items():
            node = cameras.require_group(cam)
            node.attrs["intrinsic"] = intrinsic.tolist()
            node.attrs["extrinsic"] = extrinsic.tolist()
        return path

    def save_mp4(
        self,
        path: str | Path,
        camera: str,
        env_idx: int = 0,
        fps: float | None = None,
    ) -> Path:
        """Export one camera's RGB stream as an MP4 video."""
        import imageio.v2 as imageio

        stacked = self._stack("rgb", env_idx)
        if not isinstance(stacked, dict) or camera not in stacked:
            raise KeyError(f"No RGB frames captured for camera '{camera}'")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = np.asarray(stacked[camera])
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        with imageio.get_writer(path, fps=fps or self.fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return path

    def save_all_per_env(
        self,
        out_dir: str | Path,
        prefix: str = "episode",
        *,
        camera: str | None = None,
    ) -> list[Path]:
        """Split a batched capture into one HDF5 (+ optional MP4) per env.

        Returns:
            List of written HDF5 paths, ordered by env index.
        """
        out_dir = Path(out_dir)
        written: list[Path] = []
        for env_idx in range(self.n_envs):
            hdf5_path = out_dir / f"{prefix}_{env_idx:04d}.hdf5"
            self.save_hdf5(hdf5_path, env_idx=env_idx)
            written.append(hdf5_path)
            if camera is not None:
                self.save_mp4(
                    out_dir / f"{prefix}_{env_idx:04d}.mp4",
                    camera=camera,
                    env_idx=env_idx,
                )
        return written
