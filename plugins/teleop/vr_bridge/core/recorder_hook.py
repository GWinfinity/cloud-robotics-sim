"""Recording hook for teleoperation data collection.

Two export formats from the same per-tick buffer:

- ``save_npz``: lightweight self-contained episode dump for debugging.
- ``save_hdf5``: dreamdojo-compatible layout (``episode_N/observations``
  uint8 video + ``episode_N/actions`` float32 joint targets), appendable so
  many teleop episodes accumulate into one file for
  ``plugins/datasets/dreamdojo`` training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .mapping import SemanticAction

_META_SKIP_KEYS = ("qpos", "t", "rgb")


class TeleopRecorder:
    """Buffers per-tick teleop frames while recording is toggled on."""

    def __init__(self) -> None:
        self._recording = False
        self._frames: list[dict] = []
        self._started_at: float | None = None

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def toggle(self) -> bool:
        """Flip the recording state; returns the new state."""
        if self._recording:
            self.stop()
            return False
        self.start()
        return True

    def start(self) -> None:
        self._frames = []
        self._recording = True
        self._started_at = time.monotonic()

    def stop(self) -> None:
        self._recording = False

    def clear(self) -> None:
        """Drop buffered frames; the recording state itself is unchanged."""
        self._frames = []

    def capture(
        self,
        action: SemanticAction,
        qpos: np.ndarray,
        rgb: np.ndarray | None = None,
    ) -> None:
        """Record one control tick (no-op when not recording).

        Args:
            action: Semantic action of this tick (intents + gripper cmds).
            qpos: Joint position target commanded this tick.
            rgb: Optional (H, W, 3) uint8 camera frame; required for
                ``save_hdf5`` (dreamdojo layout stores video).
        """
        if not self._recording:
            return
        frame = {
            "t": time.monotonic() - (self._started_at or time.monotonic()),
            "qpos": np.asarray(qpos, dtype=np.float64).copy(),
            "rgb": None if rgb is None else np.asarray(rgb, dtype=np.uint8).copy(),
            "grippers": dict(action.gripper_cmds),
            "ee_targets": {
                name: {
                    "pos": intent.pose.pos.tolist(),
                    "quat": intent.pose.quat.tolist(),
                    "engaged": intent.engaged,
                }
                for name, intent in action.arm_intents.items()
            },
        }
        self._frames.append(frame)

    def _meta_json(self) -> str:
        """Per-frame ee_targets/grippers/engaged as JSON (numpy-free)."""
        return json.dumps(
            [
                {k: v for k, v in f.items() if k not in _META_SKIP_KEYS}
                for f in self._frames
            ]
        )

    def save_npz(self, path: str | Path) -> Path | None:
        """Export the buffered episode. Returns None when empty."""
        if not self._frames:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        qpos = np.stack([f["qpos"] for f in self._frames])
        times = np.array([f["t"] for f in self._frames], dtype=np.float64)
        np.savez_compressed(path, qpos=qpos, t=times, meta=self._meta_json())
        return path

    def save_hdf5(self, path: str | Path, task_name: str = "teleop") -> Path | None:
        """Append the buffered episode to an HDF5 file in dreamdojo layout.

        The episode lands in ``episode_{n}`` (n = number of existing
        ``episode_*`` groups) with ``observations`` (T, H, W, 3) uint8 and
        ``actions`` (T, D) float32, so repeated recordings accumulate into
        one file. Per-frame teleop metadata is stored as JSON in the group
        attribute ``teleop_meta`` (ignored by dreamdojo).

        Returns None when the buffer is empty.

        Raises:
            ValueError: When rgb frames are missing — the dreamdojo layout
                stores video in ``observations`` and gaps are not allowed.
        """
        if not self._frames:
            return None
        missing = sum(1 for f in self._frames if f["rgb"] is None)
        if missing == len(self._frames):
            raise ValueError(
                "save_hdf5 requires per-frame rgb observations (the dreamdojo "
                "layout stores video in 'observations'), but none were "
                "captured. Pass capture(..., rgb=...) or use save_npz()."
            )
        if missing:
            raise ValueError(
                f"{missing} of {len(self._frames)} frames are missing rgb "
                "observations; gaps are not allowed in the dreamdojo layout."
            )

        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        qpos = np.stack([f["qpos"] for f in self._frames]).astype(np.float32)
        rgb = np.stack([f["rgb"] for f in self._frames])
        with h5py.File(path, "a") as h5:
            n = sum(1 for key in h5.keys() if key.startswith("episode_"))
            group = h5.create_group(f"episode_{n}")
            group.create_dataset("observations", data=rgb)
            group.create_dataset("actions", data=qpos)
            group.attrs["task_name"] = task_name
            group.attrs["teleop_meta"] = self._meta_json()
        return path
