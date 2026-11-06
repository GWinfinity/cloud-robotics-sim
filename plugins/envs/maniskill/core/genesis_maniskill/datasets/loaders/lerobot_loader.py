"""LeRobot dataset loader (v3 storage format, no ``lerobot`` package).

Reads the LeRobot v3.0 dataset layout directly with pyarrow:

- ``data/chunk-XXX/file-XXX.parquet`` — frame-level tables holding whole
  episodes, identified by the ``episode_index`` / ``frame_index`` columns
  (older flat ``data/episode_*.parquet`` layouts are found too);
- ``meta/info.json`` — fps, feature schema, totals;
- ``meta/tasks.parquet`` (or ``tasks.jsonl``) — task strings by task_index;
- ``meta/episodes.jsonl`` (optional) — per-episode metadata.

Why not the ``lerobot`` package: it is a heavy, fast-moving dependency
(HF hub stack) whose Python API broke between 2.x and 3.x
(``lerobot.common.datasets`` -> ``lerobot.datasets``), while the v3
storage layout itself is stable parquet + JSON. Parsing the format
directly keeps this plugin usable on Windows CI without optional deps —
same approach as ``examples/hifiumi/hifiumi_loader.py``.

Reference: https://github.com/huggingface/lerobot (dataset format v3.0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from genesis_maniskill.datasets.formats.trajectory import Step, Trajectory

# Columns carrying per-step labels in v3 frame tables (newest first).
_REWARD_COLUMNS = ("next.reward", "reward")
_DONE_COLUMNS = ("next.done", "done")


class LeRobotLoader:
    """Loader for LeRobot v3 datasets on local disk.

    Args:
        dataset_path: Root of the dataset (the directory holding ``data/``
            and ``meta/``), or a path to the ``data`` directory itself.
        delta_timestamps: Relative offsets (seconds) used to sample history /
            future frames around each step, mirroring ``LeRobotDataset``'s
            semantics: every per-step array gains a leading axis of
            ``len(delta_timestamps)`` (nearest-timestamp lookup within the
            episode). ``None`` (default) loads one sample per step with no
            extra axis.

    Note:
        Episodes are assumed to be wholly contained in one parquet file, as
        guaranteed by the v3 writer. Files are processed independently, so
        chunking does not affect results.
    """

    # Observation key mapping from LeRobot to unified format
    OBS_KEY_MAP = {
        # Images
        "observation.image": "rgb_agentview",
        "observation.wrist_image": "rgb_hand",
        "observation.left_image": "rgb_left",
        "observation.right_image": "rgb_right",
        # State
        "observation.state": "state",
        "observation.position": "joint_pos",
        "observation.velocity": "joint_vel",
        "observation.effort": "joint_effort",
        # End-effector
        "observation.ee_pos": "eef_pos",
        "observation.ee_quat": "eef_quat",
        # Gripper
        "observation.gripper": "gripper",
        "observation.gripper_position": "gripper",
    }

    def __init__(
        self,
        dataset_path: Union[str, Path],
        delta_timestamps: list[float] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.delta_timestamps = (
            None if delta_timestamps is None else list(delta_timestamps)
        )

    # ------------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------------

    def _root(self) -> Path:
        """Dataset root, accepting either the root or its ``data`` dir."""
        return (
            self.dataset_path.parent
            if self.dataset_path.name == "data"
            else self.dataset_path
        )

    def _data_files(self) -> list[Path]:
        """All parquet files under ``data/`` (chunked v3 or flat legacy)."""
        data_dir = self._root() / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Not a LeRobot dataset (no 'data' dir): {self._root()}"
            )
        files = sorted(data_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files under {data_dir}. Expected LeRobot v3 "
                "layout: data/chunk-XXX/file-XXX.parquet"
            )
        return files

    def _info(self) -> dict[str, Any]:
        info_path = self._root() / "meta" / "info.json"
        if info_path.is_file():
            return json.loads(info_path.read_text(encoding="utf-8"))
        return {}

    def _task_names(self) -> dict[int, str]:
        """task_index -> task string, from meta/tasks.parquet or tasks.jsonl."""
        meta_dir = self._root() / "meta"
        parquet_path = meta_dir / "tasks.parquet"
        jsonl_path = meta_dir / "tasks.jsonl"
        if parquet_path.is_file():
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_path)
            names = table.column("task").to_pylist()
            idx = table.column("task_index").to_pylist()
            return {int(i): str(n) for n, i in zip(names, idx)}
        if jsonl_path.is_file():
            tasks: dict[int, str] = {}
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    tasks[int(entry["task_index"])] = str(entry["task"])
            return tasks
        return {}

    def _episode_meta(self) -> dict[int, dict[str, Any]]:
        """episode_index -> metadata entry from meta/episodes.jsonl, if present."""
        path = self._root() / "meta" / "episodes.jsonl"
        if not path.is_file():
            return {}
        episodes: dict[int, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                episodes[int(entry["episode_index"])] = entry
        return episodes

    # ------------------------------------------------------------------
    # Episode parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _column_array(values: list[Any]) -> np.ndarray:
        """Stack fixed-size list values along a new axis; scalars stay 1-D."""
        if len(values) == 0:
            return np.asarray(values)
        if isinstance(values[0], (list, tuple, np.ndarray)):
            return np.stack([np.asarray(v) for v in values])
        return np.asarray(values)

    def _episode_to_trajectory(
        self,
        table_columns: dict[str, list[Any]],
        episode_meta: dict[int, dict[str, Any]],
        task_names: dict[int, str],
        source_file: Path,
        episode_index: int,
    ) -> Trajectory:
        """Build one Trajectory from the rows of a single episode."""
        n = len(table_columns["episode_index"])

        obs_keys = [c for c in table_columns if c.startswith("observation.")]
        arrays = {
            self.OBS_KEY_MAP.get(c, c): self._column_array(table_columns[c])
            for c in obs_keys
        }
        action_col = next(
            (c for c in ("action", "action.action") if c in table_columns), None
        )
        actions = (
            self._column_array(table_columns[action_col])
            if action_col
            else np.zeros((n, 0))
        )
        rewards = (
            self._column_array(table_columns[c]).astype(float)
            if (c := next((c for c in _REWARD_COLUMNS if c in table_columns), None))
            else np.zeros(n)
        )
        dones = (
            self._column_array(table_columns[c]).astype(bool)
            if (c := next((c for c in _DONE_COLUMNS if c in table_columns), None))
            else np.zeros(n, dtype=bool)
        )
        timestamps = np.asarray(
            table_columns.get("timestamp", np.arange(n, dtype=float)), dtype=float
        )

        # Optional per-step index remap for delta_timestamps sampling.
        indices = (
            None if self.delta_timestamps is None else self._delta_indices(timestamps)
        )

        metadata: dict[str, Any] = {
            "source": "lerobot",
            "repo_id": str(self._root()),
            "episode_index": int(episode_index),
            "file": source_file.name,
        }
        if ep := episode_meta.get(episode_index):
            metadata.update(ep)
        task_idx = table_columns.get("task_index", [None])[0]
        if task_idx is not None:
            metadata["task"] = task_names.get(int(task_idx), f"task_{task_idx}")

        trajectory = Trajectory(metadata=metadata)
        for i in range(n):
            take: Any = i if indices is None else indices[i]
            step = Step(
                obs={k: v[take] for k, v in arrays.items()},
                action=actions[take],
                # reward/done describe the transition out of the *current*
                # step, so history sampling never remaps them.
                reward=float(rewards[i]),
                terminated=bool(dones[i]),
                truncated=False,
            )
            trajectory.add_step(step)
        return trajectory

    def _delta_indices(self, timestamps: np.ndarray) -> np.ndarray:
        """Map each step to nearest-frame indices for every delta offset."""
        n = len(timestamps)
        idx = np.empty((n, len(self.delta_timestamps or [])), dtype=int)
        for j, delta in enumerate(self.delta_timestamps or []):
            target = timestamps + delta
            pos = np.searchsorted(timestamps, target)
            pos_hi = np.clip(pos, 0, n - 1)
            pos_lo = np.clip(pos - 1, 0, n - 1)
            closer_lo = np.abs(timestamps[pos_lo] - target) <= np.abs(
                timestamps[pos_hi] - target
            )
            idx[:, j] = np.where(closer_lo, pos_lo, pos_hi)
        return idx

    def load_all(self, max_episodes: int | None = None) -> list[Trajectory]:
        """Load all episodes as trajectories.

        Args:
            max_episodes: Maximum number of episodes to load (in dataset
                order), useful for smoke-testing large datasets.

        Returns:
            List of Trajectory objects, one per episode.
        """
        import pyarrow.parquet as pq

        task_names = self._task_names()
        episode_meta = self._episode_meta()

        trajectories: list[Trajectory] = []
        for parquet_file in self._data_files():
            table = pq.read_table(parquet_file)
            columns = {
                name: table.column(name).to_pylist() for name in table.column_names
            }

            # Order rows by (episode_index, frame_index) and split episodes.
            ep = np.asarray(columns["episode_index"], dtype=int)
            fr = np.asarray(columns.get("frame_index", np.zeros(len(ep))), dtype=int)
            order = np.lexsort((fr, ep))
            ep_sorted = ep[order]
            boundaries = np.flatnonzero(np.diff(ep_sorted)) + 1
            for rows in np.split(order, boundaries):
                ep_idx = int(ep_sorted[rows[0]])
                ep_columns = {k: [v[i] for i in rows] for k, v in columns.items()}
                trajectories.append(
                    self._episode_to_trajectory(
                        ep_columns,
                        episode_meta,
                        task_names,
                        parquet_file,
                        ep_idx,
                    )
                )

        if max_episodes is not None:
            trajectories = trajectories[:max_episodes]
        return trajectories

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Dataset statistics from meta/info.json, with a parquet fallback."""
        info = self._info()
        if info:
            total_episodes = info.get("total_episodes")
            total_frames = info.get("total_frames")
            if total_episodes and total_frames:
                return {
                    "num_episodes": int(total_episodes),
                    "num_frames": int(total_frames),
                    "avg_frames_per_episode": total_frames / total_episodes,
                    "fps": float(info.get("fps", 10)),
                }

        import pyarrow.parquet as pq

        num_episodes = 0
        num_frames = 0
        for parquet_file in self._data_files():
            table = pq.read_table(parquet_file, columns=["episode_index"])
            episodes = table.column("episode_index").to_pylist()
            num_episodes += len(set(episodes))
            num_frames += len(episodes)

        return {
            "num_episodes": num_episodes,
            "num_frames": num_frames,
            "avg_frames_per_episode": num_frames / num_episodes if num_episodes else 0,
            "fps": float(info.get("fps", 10)),
        }


def load_lerobot_dataset(
    dataset_path: Union[str, Path],
    max_episodes: int | None = None,
    delta_timestamps: list[float] | None = None,
) -> list[Trajectory]:
    """Convenience function to load a LeRobot v3 dataset.

    Args:
        dataset_path: Root of the LeRobot dataset (or its ``data`` dir)
        max_episodes: Maximum episodes to load
        delta_timestamps: Optional relative frame sampling offsets

    Returns:
        List of trajectories
    """
    loader = LeRobotLoader(dataset_path, delta_timestamps=delta_timestamps)
    return loader.load_all(max_episodes=max_episodes)
