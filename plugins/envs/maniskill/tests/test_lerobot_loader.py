"""Tests for the LeRobot v3 dataset loader (pyarrow parsing, no lerobot pkg)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The plugin's datasets subpackage uses absolute imports, so it must be
# imported under its top-level name with ``core/`` on sys.path (same
# convention as test_replicacad.py).
_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from genesis_maniskill.datasets.loaders.lerobot_loader import (  # noqa: E402
    LeRobotLoader,
    load_lerobot_dataset,
)

FPS = 10.0

# Per-episode frame data: (state, action, reward, done)
_EP0 = (
    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]],
    [0.0, 0.0, 1.0],
    [False, False, True],
)
_EP1 = (
    [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
    [[1.5, 1.5], [1.6, 1.6], [1.7, 1.7]],
    [0.0, 0.0, 1.0],
    [False, False, True],
)


def _write_episode_tables(root: Path, episodes: list[tuple]) -> Path:
    """Write a minimal LeRobot v3 layout: one chunked parquet + meta."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_dir = root / "meta"
    meta_dir.mkdir()

    columns: dict[str, list] = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "task_index": [],
        "observation.state": [],
        "observation.gripper": [],
        "action": [],
        "next.reward": [],
        "next.done": [],
    }
    frame_index = 0
    for ep_idx, (states, actions, rewards, dones) in enumerate(episodes):
        for i in range(len(states)):
            columns["timestamp"].append(frame_index / FPS)
            columns["frame_index"].append(frame_index)
            columns["episode_index"].append(ep_idx)
            columns["task_index"].append(0)
            columns["observation.state"].append(states[i])
            columns["observation.gripper"].append(states[i][0] * 0.01)
            columns["action"].append(actions[i])
            columns["next.reward"].append(rewards[i])
            columns["next.done"].append(dones[i])
            frame_index += 1

    state_type = pa.list_(pa.float32(), 2)
    table = pa.table(
        {
            "timestamp": pa.array(columns["timestamp"], type=pa.float64()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
            "observation.state": pa.array(
                columns["observation.state"], type=state_type
            ),
            "observation.gripper": pa.array(
                columns["observation.gripper"], type=pa.float32()
            ),
            "action": pa.array(columns["action"], type=state_type),
            "next.reward": pa.array(columns["next.reward"], type=pa.float32()),
            "next.done": pa.array(columns["next.done"], type=pa.bool_()),
        }
    )
    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(table, parquet_path)

    total_frames = frame_index
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": FPS,
                "total_episodes": len(episodes),
                "total_frames": total_frames,
                "features": {},
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick_cube"]),
            }
        ),
        meta_dir / "tasks.parquet",
    )
    return root


@pytest.fixture()
def v3_dataset(tmp_path: Path) -> Path:
    """Synthetic two-episode LeRobot v3 dataset on disk."""
    return _write_episode_tables(tmp_path, [_EP0, _EP1])


class TestV3Layout:
    """Loading the chunked v3 layout end to end."""

    def test_load_all(self, v3_dataset: Path):
        trajectories = load_lerobot_dataset(v3_dataset)

        assert len(trajectories) == 2
        assert [len(t) for t in trajectories] == [3, 3]

        traj0 = trajectories[0]
        assert traj0.metadata["episode_index"] == 0
        assert traj0.metadata["task"] == "pick_cube"

        step0 = traj0.steps[0]
        assert set(step0.obs) == {"state", "gripper"}
        np.testing.assert_allclose(step0.obs["state"], [0.0, 0.0])
        np.testing.assert_allclose(step0.action, [0.5, 0.5])
        assert step0.reward == 0.0
        assert step0.terminated is False

        last = traj0.steps[-1]
        assert last.reward == 1.0
        assert last.terminated is True

        # Second episode's frames must not leak into the first.
        np.testing.assert_allclose(trajectories[1].steps[0].obs["state"], [10.0, 10.0])

    def test_max_episodes(self, v3_dataset: Path):
        trajectories = load_lerobot_dataset(v3_dataset, max_episodes=1)
        assert len(trajectories) == 1
        assert trajectories[0].metadata["episode_index"] == 0

    def test_statistics_from_info(self, v3_dataset: Path):
        stats = LeRobotLoader(v3_dataset).get_dataset_statistics()
        assert stats["num_episodes"] == 2
        assert stats["num_frames"] == 6
        assert stats["avg_frames_per_episode"] == 3
        assert stats["fps"] == FPS

    def test_data_dir_path_accepted(self, v3_dataset: Path):
        """Passing the data/ dir itself works the same as the root."""
        trajectories = LeRobotLoader(v3_dataset / "data").load_all()
        assert len(trajectories) == 2


class TestDeltaTimestamps:
    """Relative frame sampling mirrors LeRobotDataset semantics."""

    def test_history_sampling(self, v3_dataset: Path):
        loader = LeRobotLoader(v3_dataset, delta_timestamps=[0.0, -0.1])
        trajectories = loader.load_all()

        traj = trajectories[0]
        # Leading axis of len(delta_timestamps) on every per-step array.
        assert traj.steps[0].obs["state"].shape == (2, 2)
        assert traj.steps[0].action.shape == (2, 2)

        # fps=10 -> -0.1 s is exactly one frame back.
        np.testing.assert_allclose(traj.steps[1].obs["state"][0], [1.0, 1.0])
        np.testing.assert_allclose(traj.steps[1].obs["state"][1], [0.0, 0.0])

        # First frame has no history: clamps to itself.
        np.testing.assert_allclose(traj.steps[0].obs["state"][1], [0.0, 0.0])

    def test_future_sampling(self, v3_dataset: Path):
        loader = LeRobotLoader(v3_dataset, delta_timestamps=[0.2])
        traj = loader.load_all()[0]
        # +0.2 s is two frames ahead.
        np.testing.assert_allclose(traj.steps[0].obs["state"][0], [2.0, 2.0])
        # Last frame clamps to itself.
        np.testing.assert_allclose(traj.steps[2].obs["state"][0], [2.0, 2.0])

    def test_default_has_no_extra_axis(self, v3_dataset: Path):
        traj = LeRobotLoader(v3_dataset).load_all()[0]
        assert traj.steps[0].obs["state"].shape == (2,)


class TestLayoutVariants:
    """Robustness: legacy flat layout and informative errors."""

    def test_flat_data_dir(self, tmp_path: Path):
        """Legacy flat layout (data/*.parquet without chunk dirs) loads too."""
        root = _write_episode_tables(tmp_path, [_EP0])
        flat = tmp_path / "flat"
        (flat / "data").mkdir(parents=True)
        src = root / "data" / "chunk-000" / "file-000.parquet"
        src.replace(flat / "data" / "file-000.parquet")
        for meta_file in (root / "meta").iterdir():
            (flat / "meta").mkdir(exist_ok=True)
            meta_file.replace(flat / "meta" / meta_file.name)

        trajectories = load_lerobot_dataset(flat)
        assert len(trajectories) == 1
        assert len(trajectories[0]) == 3

    def test_missing_data_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no 'data' dir"):
            LeRobotLoader(tmp_path).load_all()

    def test_empty_data_dir(self, tmp_path: Path):
        (tmp_path / "data").mkdir()
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            LeRobotLoader(tmp_path).load_all()
