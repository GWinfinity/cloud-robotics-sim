"""End-to-end test against a real Genesis Franka (headless, no headset).

Builds a headless Genesis scene with the bundled Franka MJCF, wires it to
VRBridge through the example adapter, and drives it with the loopback
client: clutch-engaged motion must move the real end-effector via IK, and
a camera-fed recording must land in a dreamdojo-loadable HDF5 episode.

Skipped automatically when genesis-world is not installed. This is the
M1-prep rehearsal: same control path the real PICO client will drive.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

genesis = pytest.importorskip("genesis")  # noqa: F401 - guard only

from loopback import LoopbackClient  # noqa: E402
from vr_bridge.core.bridge import BridgeConfig, VRBridge  # noqa: E402

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from run_teleop import _GenesisRobotAdapter, build_genesis_robot  # noqa: E402

MAPPING = str(
    Path(__file__).resolve().parents[1] / "configs" / "mappings" / "franka_single.yaml"
)


def _run_ticks(bridge, scene, seconds, dt=0.01, camera=None):
    """Step bridge + scene together; feed camera frames when given."""
    info = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rgb = None
        if camera is not None:
            rgb, *_ = camera.render(rgb=True)
        info = bridge.step(dt, rgb=rgb)
        scene.step()
        time.sleep(dt)
    return info


def _run_until(bridge, scene, condition, timeout, camera=None, dt=0.01):
    """Step bridge + scene until ``condition()`` or timeout; return condition result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rgb = None
        if camera is not None:
            rgb, *_ = camera.render(rgb=True)
        bridge.step(dt, rgb=rgb)
        scene.step()
        time.sleep(dt)
        if condition():
            return True
    return condition()


@pytest.fixture(scope="module")
def genesis_setup():
    """Headless Genesis Franka + offscreen camera (built once per module)."""
    scene, entity, camera = build_genesis_robot(with_camera=True)
    yield scene, _GenesisRobotAdapter(entity), camera


def test_genesis_franka_teleop_and_recording(tmp_path, genesis_setup):
    """Loopback client drives the real arm via IK; episode loads in dreamdojo."""
    scene, robot, camera = genesis_setup
    hand_pos = lambda: np.asarray(robot.get_link_pose("hand").pos).ravel()  # noqa: E731

    bridge = VRBridge(
        BridgeConfig(
            host="127.0.0.1",
            state_port=0,
            control_port=0,
            mapping_path=MAPPING,
            recording_output=str(tmp_path / "episode.h5"),
        ),
        robot=robot,
    )
    bridge.start()
    client = LoopbackClient(bridge.transport.state_port, bridge.transport.control_port)
    try:
        assert client.handshake()["type"] == "welcome"
        client.start_stream()

        # Engage the clutch and let the arm settle at the anchor pose.
        client.set_hand(pos=(0.3, 0.0, 0.5), grip=1.0)
        _run_ticks(bridge, scene, 1.0)

        # Drive the controller on a continuous circular path (like the mock
        # client's script mode). Re-anchoring after any transient stale frame
        # must not stop a continuously moving hand: the EE path range is what
        # matters, not a single point-to-point displacement.
        ee_track = []
        t0 = time.monotonic()
        deadline = t0 + 4.0
        while time.monotonic() < deadline:
            t = time.monotonic() - t0
            client.set_hand(
                pos=(0.3 + 0.06 * np.cos(2 * t), 0.06 * np.sin(2 * t), 0.5),
                grip=1.0,
            )
            bridge.step(0.01)
            scene.step()
            ee_track.append(hand_pos().copy())
            time.sleep(0.01)
        track = np.asarray(ee_track)
        xy_range = np.ptp(track[:, :2], axis=0).max()
        assert xy_range > 0.02, (
            f"real EE did not track the controller: xy_range={xy_range:.4f}, "
            f"ik_failures={bridge.ik_failures}"
        )

        # Record a short episode through the B-button toggle; auto-save on off.
        # The camera feeds every tick so no captured frame lacks an rgb
        # observation (gaps are rejected by the dreamdojo layout).
        client.send_event("b", True)
        assert _run_until(
            bridge,
            scene,
            lambda: bridge.recorder.n_frames >= 10,
            timeout=10.0,
            camera=camera,
        ), "recording never accumulated 10 frames"
        client.send_event("b", True)
        assert _run_until(
            bridge,
            scene,
            lambda: not bridge.recorder.recording,
            timeout=5.0,
            camera=camera,
        ), "record_toggle-off never landed"

        h5 = tmp_path / "episode.h5"
        assert h5.exists(), "auto-save on record_toggle-off did not write the HDF5"
        assert bridge.recorder.n_frames == 0  # buffer cleared after save

        # Cross-plugin: the episode must load into the dreamdojo dataset.
        from dreamdojo.core.dataset import GenesisDataset

        ds = GenesisDataset(
            pre_generated_path=str(h5),
            num_frames=4,
            robot_type="franka",
            device="cpu",
        )
        sample = ds[0]
        assert sample["video"].shape[0] == 4
        assert sample["video"].shape[1] == 3  # T,C,H,W
        assert sample["action"].shape[0] == 4
    finally:
        client.close()
        bridge.stop()
