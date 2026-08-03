"""End-to-end recording test: B-key toggle -> auto-save -> dreamdojo load.

Drives a real VRBridge over loopback sockets, toggles recording with the
mapped ``b`` button (record_toggle), supplies synthetic rgb frames to
``step()``, then verifies the auto-saved HDF5 loads through
``dreamdojo.core.dataset.GenesisDataset``. No headset, no Genesis, no CUDA.
"""

from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pytest
from conftest import MAPPINGS_DIR, StubRobot
from dreamdojo.core.dataset import GenesisDataset
from loopback import HOST, LoopbackClient
from vr_bridge.core.bridge import BridgeConfig, VRBridge
from vr_bridge.core.safety import SafetyConfig


@pytest.fixture()
def rec_bridge(tmp_path):
    """VRBridge whose recording_output points at a tmp HDF5 file."""
    out = tmp_path / "teleop.h5"
    config = BridgeConfig(
        host=HOST,
        state_port=0,
        control_port=0,
        mapping_path=str(MAPPINGS_DIR / "franka_single.yaml"),
        recording_output=str(out),
        safety=SafetyConfig(
            workspace_center=(0.0, 0.0, 0.35),
            workspace_radius=0.8,
            max_ee_speed=1.0,
            freeze_timeout_ms=100.0,
            disconnect_timeout_ms=1000.0,
        ),
    )
    robot = StubRobot(n_dofs=9)
    br = VRBridge(config, robot=robot)
    br.start()
    yield br, robot, out
    br.stop()


def _step_until(predicate, br: VRBridge, rgb: np.ndarray, timeout: float = 3.0):
    """Step (always feeding rgb) until predicate() or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        br.step(0.01, rgb=rgb)
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition never became true")


def test_record_toggle_auto_saves_dreamdojo_episode(rec_bridge):
    """B -> record ticks -> b: HDF5 episode_0 appears and dreamdojo loads it."""
    br, robot, out = rec_bridge
    client = LoopbackClient(br.transport.state_port, br.transport.control_port)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(8, 16, 3), dtype=np.uint8)
    try:
        reply = client.handshake()
        assert reply["type"] == "welcome"

        client.set_hand(pos=(0.3, 0.0, 0.5), grip=1.0, trigger=0.5)
        client.start_stream()

        # B button toggles recording on (mapped to record_toggle).
        client.send_event("b", True)
        _step_until(lambda: br.recorder.recording, br, rgb)

        # Record ~30 ticks of teleop with rgb frames.
        for _ in range(30):
            br.step(0.01, rgb=rgb)
        n_recorded = br.recorder.n_frames
        assert n_recorded > 0

        # B again toggles off -> auto-save to recording_output + clear.
        client.send_event("b", True)
        _step_until(lambda: not br.recorder.recording, br, rgb)

        assert out.exists()
        assert br.recorder.n_frames == 0

        with h5py.File(out, "r") as h5:
            assert list(h5.keys()) == ["episode_0"]
            episode = h5["episode_0"]
            n_frames = episode["observations"].shape[0]
            # >= because ticks keep recording while the toggle-off event
            # travels over TCP; the buffer is cleared right after saving.
            assert n_frames >= n_recorded
            assert episode["observations"].shape[1:] == (8, 16, 3)
            assert episode["observations"].dtype == np.uint8
            assert episode["actions"].shape == (n_frames, 9)
            assert episode["actions"].dtype == np.float32
            meta = json.loads(episode.attrs["teleop_meta"])
            assert len(meta) == n_frames
            assert meta[0]["ee_targets"]["right_arm"]["engaged"] is True

        # Cross-plugin: dreamdojo's GenesisDataset loads the teleop file.
        ds = GenesisDataset(
            pre_generated_path=str(out),
            num_frames=4,
            robot_type="franka",
            device="cpu",
        )
        assert len(ds) == ds.num_episodes
        sample = ds[0]
        assert isinstance(sample, dict)
        assert sample["video"].shape == (4, 3, 8, 16)  # T,C,H,W
        assert sample["action"].shape == (4, 9)
    finally:
        client.close()
