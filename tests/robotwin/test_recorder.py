"""Tests for the RoboTwin-format EpisodeRecorder (HDF5 + MP4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cloud_robotics_sim.robotwin.recorder import EpisodeRecorder

h5py = pytest.importorskip("h5py")


def _make_recorder(n_envs: int = 1, n_frames: int = 5) -> EpisodeRecorder:
    """Recorder populated with synthetic frames."""
    rec = EpisodeRecorder(task_name="test_task", fps=20.0, n_envs=n_envs)
    rec.set_camera_params(
        "head",
        intrinsic=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
        extrinsic=np.eye(4),
    )
    for i in range(n_frames):
        rec.capture(
            i,
            rgb={"head": np.full((n_envs, 8, 8, 3), i, dtype=np.uint8)},
            depth={"head": np.full((n_envs, 8, 8), float(i), dtype=np.float32)},
            segmentation={"head": np.zeros((n_envs, 8, 8), dtype=np.int32)},
            qpos=np.full((n_envs, 14), 0.1 * i),
            endpose=np.tile(np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0]), (n_envs, 1)),
        )
    return rec


class TestCapture:
    """Frame collection and normalization behavior."""

    def test_single_env_arrays_get_env_dim(self) -> None:
        rec = EpisodeRecorder(n_envs=1)
        rec.capture(
            0,
            rgb={"head": np.zeros((8, 8, 3), dtype=np.uint8)},
            qpos=np.zeros(14),
        )
        assert rec._frames[0]["rgb"]["head"].shape == (1, 8, 8, 3)
        assert rec._frames[0]["qpos"].shape == (1, 14)

    def test_torch_tensors_converted(self) -> None:
        rec = EpisodeRecorder(n_envs=2)
        rec.capture(0, qpos=torch.zeros(2, 14))
        frame = rec._frames[0]["qpos"]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (2, 14)

    def test_n_frames_and_clear(self) -> None:
        rec = _make_recorder(n_frames=3)
        assert rec.n_frames == 3
        rec.clear()
        assert rec.n_frames == 0

    def test_invalid_camera_params_rejected(self) -> None:
        rec = EpisodeRecorder()
        with pytest.raises(ValueError):
            rec.set_camera_params("head", np.eye(4), np.eye(4))
        with pytest.raises(ValueError):
            rec.set_camera_params("head", np.eye(3), np.eye(3))

    def test_invalid_n_envs_rejected(self) -> None:
        with pytest.raises(ValueError):
            EpisodeRecorder(n_envs=0)


class TestSaveHdf5:
    """HDF5 export layout and metadata."""

    def test_round_trip_fields(self, tmp_path) -> None:
        rec = _make_recorder(n_envs=1, n_frames=5)
        path = rec.save_hdf5(tmp_path / "episode_000.hdf5")

        with h5py.File(path) as f:
            assert f.attrs["task_name"] == "test_task"
            assert f.attrs["fps"] == 20.0
            assert f.attrs["n_frames"] == 5
            assert f["obs/rgb/head"].shape == (5, 8, 8, 3)
            assert f["obs/rgb/head"].dtype == np.uint8
            assert f["obs/depth/head"].shape == (5, 8, 8)
            assert f["obs/segmentation/head"].shape == (5, 8, 8)
            assert f["qpos"].shape == (5, 14)
            assert f["endpose"].shape == (5, 7)
            np.testing.assert_allclose(
                f["cameras/head"].attrs["intrinsic"],
                [[500.0, 0, 320], [0, 500, 240], [0, 0, 1]],
            )
            np.testing.assert_allclose(f["cameras/head"].attrs["extrinsic"], np.eye(4))

    def test_per_env_split(self, tmp_path) -> None:
        rec = _make_recorder(n_envs=3, n_frames=4)
        paths = rec.save_all_per_env(tmp_path, prefix="ep", camera="head")
        assert len(paths) == 3
        for env_idx, path in enumerate(paths):
            assert path.exists()
            assert (tmp_path / f"ep_{env_idx:04d}.mp4").exists()
            with h5py.File(path) as f:
                assert f.attrs["env_idx"] == env_idx
                assert f["qpos"].shape == (4, 14)

    def test_env_idx_out_of_range(self, tmp_path) -> None:
        rec = _make_recorder(n_envs=2)
        with pytest.raises(IndexError):
            rec.save_hdf5(tmp_path / "x.hdf5", env_idx=2)

    def test_save_without_frames_raises(self, tmp_path) -> None:
        rec = EpisodeRecorder()
        with pytest.raises(RuntimeError):
            rec.save_hdf5(tmp_path / "x.hdf5")


class TestSaveZarr:
    """Zarr export for DP/DP3 consumption (doc section 7)."""

    def test_round_trip_fields(self, tmp_path) -> None:
        zarr = pytest.importorskip("zarr")
        rec = _make_recorder(n_envs=1, n_frames=5)
        path = rec.save_zarr(tmp_path / "episode_000.zarr")

        root = zarr.open_group(str(path), mode="r")
        assert root.attrs["task_name"] == "test_task"
        assert root.attrs["n_frames"] == 5
        assert root["obs/rgb/head"].shape == (5, 8, 8, 3)
        assert root["obs/rgb/head"].dtype == np.uint8
        assert root["obs/depth/head"].shape == (5, 8, 8)
        assert root["qpos"].shape == (5, 14)
        assert root["endpose"].shape == (5, 7)
        np.testing.assert_allclose(
            np.asarray(root["cameras/head"].attrs["extrinsic"]), np.eye(4)
        )

    def test_per_env_selection(self, tmp_path) -> None:
        pytest.importorskip("zarr")
        rec = _make_recorder(n_envs=3, n_frames=4)
        path = rec.save_zarr(tmp_path / "ep.zarr", env_idx=2)
        import zarr

        root = zarr.open_group(str(path), mode="r")
        assert root.attrs["env_idx"] == 2
        assert root["qpos"].shape == (4, 14)

    def test_save_without_frames_raises(self, tmp_path) -> None:
        pytest.importorskip("zarr")
        rec = EpisodeRecorder()
        with pytest.raises(RuntimeError):
            rec.save_zarr(tmp_path / "x.zarr")


class TestSaveMp4:
    """MP4 video export."""

    def test_mp4_written(self, tmp_path) -> None:
        pytest.importorskip("imageio")
        rec = _make_recorder(n_envs=1, n_frames=4)
        path = rec.save_mp4(tmp_path / "video.mp4", camera="head")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_missing_camera_raises(self, tmp_path) -> None:
        rec = _make_recorder(n_envs=1)
        with pytest.raises(KeyError):
            rec.save_mp4(tmp_path / "video.mp4", camera="wrist_l")

    def test_float_frames_clipped_to_uint8(self, tmp_path) -> None:
        pytest.importorskip("imageio")
        rec = EpisodeRecorder(n_envs=1)
        rec.capture(0, rgb={"head": np.full((1, 8, 8, 3), 300.0)})
        path = rec.save_mp4(tmp_path / "v.mp4", camera="head")
        assert path.exists()
