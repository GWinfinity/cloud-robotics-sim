"""Resume-safe dataset writer for rendered episodes.

Layout (a HALO-shaped subset, filesystem-only)::

    <root>/
    ├── data/<episode_id>/meta.json          # spec + DR + per-camera frame counts
    ├── data/<episode_id>/<camera>/frame_000000.png
    └── meta/episodes.jsonl                  # one row per completed episode

Re-running ``write_episode`` for an existing episode is a no-op so shards can
be restarted safely, mirroring the reference pipeline's resume semantics.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .render import EpisodeResult

logger = logging.getLogger(__name__)

_EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_episode_id(episode_id: str) -> None:
    if not _EPISODE_ID_RE.match(episode_id):
        raise ValueError(
            f"episode_id must match {_EPISODE_ID_RE.pattern}, got {episode_id!r}"
        )


class DatasetWriter:
    """Write episode frames and metadata under a dataset root."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._data_dir = self._root / "data"
        self._meta_dir = self._root / "meta"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def episode_dir(self, episode_id: str) -> Path:
        _validate_episode_id(episode_id)
        return self._data_dir / episode_id

    def is_done(self, episode_id: str) -> bool:
        return (self.episode_dir(episode_id) / "meta.json").is_file()

    def written_episode_ids(self) -> list[str]:
        """Episode ids recorded in ``meta/episodes.jsonl`` (in file order)."""
        jsonl = self._meta_dir / "episodes.jsonl"
        if not jsonl.is_file():
            return []
        ids = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.append(json.loads(line)["episode_id"])
        return ids

    def write_episode(self, result: EpisodeResult) -> Path:
        """Write frames + metadata; skip when the episode already exists."""
        episode_dir = self.episode_dir(result.episode_id)
        meta_path = episode_dir / "meta.json"
        if meta_path.is_file():
            logger.info("skip existing episode %s", result.episode_id)
            return episode_dir

        episode_dir.mkdir(parents=True, exist_ok=True)
        camera_meta: dict[str, Any] = {}
        for camera_name, frames in result.frames.items():
            _validate_episode_id(camera_name)  # also used as a directory name
            camera_dir = episode_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            self._write_frames(camera_dir, frames)
            height, width = (
                (frames.shape[1], frames.shape[2]) if frames.ndim == 4 else (0, 0)
            )
            camera_meta[camera_name] = {
                "frames": int(frames.shape[0] if frames.ndim == 4 else 0),
                "res": [int(width), int(height)],
            }

        metadata = {
            **result.metadata,
            "episode_id": result.episode_id,
            "cameras_frames": camera_meta,
        }
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._append_jsonl(
            {
                "episode_id": result.episode_id,
                "steps": result.steps,
                "path": f"data/{result.episode_id}",
            }
        )
        return episode_dir

    def _append_jsonl(self, row: dict[str, Any]) -> None:
        jsonl = self._meta_dir / "episodes.jsonl"
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_frames(camera_dir: Path, frames: Any) -> None:
        if frames.ndim != 4 or frames.shape[0] == 0:
            return
        from PIL import Image

        for index, frame in enumerate(frames):
            path = camera_dir / f"frame_{index:06d}.png"
            Image.fromarray(np_asuint8(frame)).save(path)


def np_asuint8(frame: Any) -> Any:
    """Convert one frame to a contiguous uint8 array for PIL."""
    import numpy as np

    return np.ascontiguousarray(frame, dtype=np.uint8)


__all__ = ["DatasetWriter"]
