"""TeleopRecorder tests: buffering, npz export, dreamdojo HDF5 export."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from vr_bridge.core.mapping import ArmIntent, ArmMapping, SemanticAction
from vr_bridge.core.messages import PoseMsg
from vr_bridge.core.recorder_hook import TeleopRecorder


def _action() -> SemanticAction:
    """One fixed semantic action (right arm intent + half-closed gripper)."""
    mapping = ArmMapping(
        name="right_arm", source="right", ee_link="hand", dofs=list(range(7))
    )
    pose = PoseMsg(pos=np.array([0.3, 0.0, 0.5]), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    return SemanticAction(
        arm_intents={"right_arm": ArmIntent(mapping=mapping, pose=pose, engaged=True)},
        gripper_cmds={"gripper": 0.5},
    )


def _rgb(seed: int = 0) -> np.ndarray:
    """One synthetic (8, 16, 3) uint8 camera frame."""
    return np.random.default_rng(seed).integers(0, 255, size=(8, 16, 3), dtype=np.uint8)


def _record_frames(recorder: TeleopRecorder, n: int, with_rgb: bool = True) -> None:
    """Capture n frames of synthetic teleop data."""
    recorder.start()
    for i in range(n):
        recorder.capture(
            _action(),
            np.full(9, 0.1 * i, dtype=np.float64),
            rgb=_rgb(i) if with_rgb else None,
        )
    recorder.stop()


def test_toggle_state_transitions():
    """toggle() flips recording on/off and reports the new state."""
    recorder = TeleopRecorder()
    assert recorder.recording is False
    assert recorder.toggle() is True
    assert recorder.recording is True
    assert recorder.toggle() is False
    assert recorder.recording is False


def test_capture_noop_when_not_recording():
    """Frames captured outside a recording are dropped."""
    recorder = TeleopRecorder()
    recorder.capture(_action(), np.zeros(9), rgb=_rgb())
    assert recorder.n_frames == 0


def test_start_resets_buffer():
    """start() drops frames from a previous episode."""
    recorder = TeleopRecorder()
    _record_frames(recorder, 3)
    recorder.start()
    assert recorder.n_frames == 0
    recorder.stop()


def test_clear_keeps_recording_state():
    """clear() empties the buffer without stopping the recording."""
    recorder = TeleopRecorder()
    recorder.start()
    recorder.capture(_action(), np.zeros(9), rgb=_rgb())
    recorder.clear()
    assert recorder.n_frames == 0
    assert recorder.recording is True
    recorder.capture(_action(), np.zeros(9), rgb=_rgb(1))
    assert recorder.n_frames == 1
    recorder.stop()


def test_save_npz_roundtrip(tmp_path):
    """Npz export preserves qpos/t/meta shapes and content."""
    recorder = TeleopRecorder()
    _record_frames(recorder, 3)
    path = recorder.save_npz(tmp_path / "ep.npz")
    assert path is not None
    data = np.load(path)
    assert data["qpos"].shape == (3, 9)
    assert data["qpos"][1] == pytest.approx(np.full(9, 0.1))
    assert data["t"].shape == (3,)
    meta = json.loads(str(data["meta"]))
    assert len(meta) == 3
    assert meta[0]["grippers"] == {"gripper": 0.5}
    assert meta[0]["ee_targets"]["right_arm"]["engaged"] is True


def test_save_npz_empty_returns_none(tmp_path):
    """An empty buffer exports nothing (npz)."""
    assert TeleopRecorder().save_npz(tmp_path / "ep.npz") is None


def test_save_hdf5_layout(tmp_path):
    """HDF5 export uses the dreamdojo episode_0/observations+actions layout."""
    recorder = TeleopRecorder()
    _record_frames(recorder, 5)
    path = recorder.save_hdf5(tmp_path / "teleop.h5", task_name="pick_place")
    assert path is not None
    with h5py.File(path, "r") as h5:
        assert list(h5.keys()) == ["episode_0"]
        episode = h5["episode_0"]
        assert episode["observations"].shape == (5, 8, 16, 3)
        assert episode["observations"].dtype == np.uint8
        assert episode["actions"].shape == (5, 9)
        assert episode["actions"].dtype == np.float32
        assert episode["actions"][2] == pytest.approx(np.full(9, 0.2))
        assert episode.attrs["task_name"] == "pick_place"
        meta = json.loads(episode.attrs["teleop_meta"])
        assert len(meta) == 5
        assert meta[0]["grippers"] == {"gripper": 0.5}


def test_save_hdf5_appends_episode_numbers(tmp_path):
    """Repeated saves accumulate as episode_0, episode_1, ... in one file."""
    out = tmp_path / "teleop.h5"
    recorder = TeleopRecorder()
    _record_frames(recorder, 4)
    recorder.save_hdf5(out)

    recorder.clear()
    _record_frames(recorder, 2)
    recorder.save_hdf5(out)

    with h5py.File(out, "r") as h5:
        assert sorted(h5.keys()) == ["episode_0", "episode_1"]
        assert h5["episode_0"]["actions"].shape == (4, 9)
        assert h5["episode_1"]["actions"].shape == (2, 9)


def test_save_hdf5_empty_returns_none(tmp_path):
    """An empty buffer exports nothing (hdf5)."""
    assert TeleopRecorder().save_hdf5(tmp_path / "teleop.h5") is None


def test_save_hdf5_requires_rgb(tmp_path):
    """Frames without any rgb cannot fill the dreamdojo video dataset."""
    recorder = TeleopRecorder()
    _record_frames(recorder, 3, with_rgb=False)
    with pytest.raises(ValueError, match="rgb"):
        recorder.save_hdf5(tmp_path / "teleop.h5")


def test_save_hdf5_rejects_partial_rgb(tmp_path):
    """Gaps in the rgb stream are rejected with the missing-frame count."""
    recorder = TeleopRecorder()
    recorder.start()
    recorder.capture(_action(), np.zeros(9), rgb=_rgb(0))
    recorder.capture(_action(), np.zeros(9), rgb=None)
    recorder.capture(_action(), np.zeros(9), rgb=_rgb(2))
    recorder.stop()
    with pytest.raises(ValueError, match="1 of 3"):
        recorder.save_hdf5(tmp_path / "teleop.h5")
