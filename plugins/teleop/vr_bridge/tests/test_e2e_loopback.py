"""End-to-end loopback test: real sockets on 127.0.0.1, stub robot.

Covers the full pipeline: handshake -> UDP state stream -> filtering /
mapping / retargeting / safety -> IK -> position control, plus estop and
the freshness watchdog. No headset, no Genesis.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest
from conftest import MAPPINGS_DIR, StubRobot, make_state_dict
from loopback import HOST, LoopbackClient, run_steps, wait_state
from vr_bridge.core.bridge import BridgeConfig, VRBridge
from vr_bridge.core.messages import PROTOCOL_VERSION
from vr_bridge.core.safety import SafetyConfig


@pytest.fixture()
def bridge():
    """A running VRBridge on ephemeral ports with a stub robot attached."""
    config = BridgeConfig(
        host=HOST,
        state_port=0,  # ephemeral ports: no clashes between test runs
        control_port=0,
        mapping_path=str(MAPPINGS_DIR / "franka_single.yaml"),
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
    yield br, robot
    br.stop()


def test_full_teleop_pipeline(bridge):
    """Handshake -> state stream -> control output -> estop -> watchdog."""
    br, robot = bridge
    client = LoopbackClient(br.transport.state_port, br.transport.control_port)
    try:
        # State datagrams before the handshake must be ignored.
        pre = make_state_dict(seq=0, client_time_ms=0)
        client.udp.sendto(json.dumps(pre).encode(), (HOST, client.state_port))
        time.sleep(0.15)
        assert br.mailbox.get() is None

        # Handshake.
        reply = client.handshake()
        assert reply["type"] == "welcome"
        assert reply["protocol_version"] == PROTOCOL_VERSION

        # A second client is rejected while the session is active.
        other = socket.create_connection((HOST, client.control_port), timeout=5.0)
        other.settimeout(5.0)
        busy = LoopbackClient.read_line(other)
        assert busy["type"] == "error"
        assert busy["code"] == "SESSION_BUSY"
        other.close()

        # Engage the clutch at the neutral pose, then move +0.1m in x.
        client.set_hand(pos=(0.3, 0.0, 0.5), grip=1.0, trigger=1.0)
        client.start_stream()
        run_steps(br, 0.3)  # let the clutch anchor
        client.set_hand(pos=(0.4, 0.0, 0.5), grip=1.0, trigger=1.0)
        info = run_steps(br, 0.6)
        assert info["safety_state"] == "ok"
        assert info["session_active"] is True

        # The robot was commanded every tick with a 9-dof target.
        assert len(robot.control_calls) > 10
        q_cmd = robot.control_calls[-1]
        assert q_cmd.shape == (9,)
        # IK was reached and the EE target tracked the +0.1m motion
        # (StubRobot encodes the target pos into q[:3]).
        assert len(robot.ik_calls) > 0
        assert q_cmd[0] == pytest.approx(0.4, abs=0.03)
        assert q_cmd[1:3] == pytest.approx([0.0, 0.5], abs=0.05)
        # Trigger fully pulled -> gripper fully closed (close_value = 0.0).
        assert q_cmd[7] == pytest.approx(0.0, abs=1e-9)
        assert q_cmd[8] == pytest.approx(0.0, abs=1e-9)
        # The commanded target stays inside the workspace sphere.
        centre = np.array([0.0, 0.0, 0.35])
        for _, pos, _ in robot.ik_calls:
            assert np.linalg.norm(pos - centre) <= 0.8 + 1e-6

        # Emergency stop latches and freezes the robot.
        client.send_event("menu", True)
        info = wait_state(br, "estop")
        assert br.safety.estopped is True
        frozen = robot.control_calls[-1].copy()
        run_steps(br, 0.1)
        assert robot.control_calls[-1] == pytest.approx(frozen)

        # A menu release event is ignored (tap-safe toggle semantics).
        client.send_event("menu", False)
        run_steps(br, 0.1)
        assert br.safety.estopped is True

        # Pressing menu again releases the estop and resumes control.
        client.send_event("menu", True)
        info = wait_state(br, "ok")
        assert info["safety_state"] == "ok"

        # Watchdog: >100ms stale freezes, >1s reports disconnected.
        client.stop_stream()
        time.sleep(0.25)
        info = br.step(0.01)
        assert info["safety_state"] == "stale"
        time.sleep(1.0)
        info = br.step(0.01)
        assert info["safety_state"] == "disconnected"
    finally:
        client.close()


def test_protocol_mismatch_over_socket(bridge):
    """A hello with the wrong protocol version gets PROTOCOL_MISMATCH."""
    br, _ = bridge
    client = LoopbackClient(br.transport.state_port, br.transport.control_port)
    try:
        reply = client.handshake(version=PROTOCOL_VERSION + 1)
        assert reply["type"] == "error"
        assert reply["code"] == "PROTOCOL_MISMATCH"
        assert not br.session.is_active()
    finally:
        client.close()
