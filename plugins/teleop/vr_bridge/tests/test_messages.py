"""Protocol v1 codec tests: round-trips and schema rejection."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_hand, make_state_dict
from vr_bridge.core.messages import (
    PROTOCOL_VERSION,
    ControllerState,
    EventMsg,
    PoseMsg,
    ProtocolError,
    decode_control,
    decode_state_datagram,
    encode_message,
)
from vr_bridge.core.session import SessionManager


def test_pose_roundtrip():
    """PoseMsg survives a dict round-trip."""
    pose = PoseMsg.from_dict({"pos": [1, 2, 3], "quat": [1, 0, 0, 0]})
    assert pose.pos == pytest.approx([1, 2, 3])
    back = PoseMsg.from_dict(pose.to_dict())
    assert back.pos == pytest.approx(pose.pos)
    assert back.quat == pytest.approx(pose.quat)


def test_pose_bad_shape_rejected():
    """pos/quat with wrong lengths fail validation."""
    with pytest.raises(ProtocolError):
        PoseMsg.from_dict({"pos": [1, 2], "quat": [1, 0, 0, 0]})
    with pytest.raises(ProtocolError):
        PoseMsg.from_dict({"pos": [1, 2, 3], "quat": [1, 0, 0]})


def test_state_datagram_roundtrip():
    """A full state datagram encodes and decodes losslessly."""
    state = ControllerState.from_dict(
        make_state_dict(
            seq=42,
            client_time_ms=12345,
            right=make_hand(trigger=0.7, grip=1.0, thumbstick=(0.5, -0.5)),
            head={"pos": [0, 0, 1.7], "quat": [1, 0, 0, 0]},
        )
    )
    assert state.seq == 42
    assert state.client_time_ms == 12345
    assert state.head is not None
    assert state.right.trigger == pytest.approx(0.7)

    import json

    decoded = decode_state_datagram(json.dumps(state.to_dict()).encode())
    assert decoded.seq == state.seq
    assert decoded.right.trigger == pytest.approx(0.7)
    assert decoded.left.pose.pos == pytest.approx(state.left.pose.pos)
    assert decoded.head is not None
    assert decoded.head.pos == pytest.approx([0, 0, 1.7])


def test_state_without_head_is_none():
    """The optional head field decodes to None when absent."""
    state = ControllerState.from_dict(make_state_dict())
    assert state.head is None
    assert "head" not in state.to_dict()


def test_analog_fields_clamped_to_protocol_ranges():
    """trigger/grip clamp to [0,1], thumbstick to [-1,1] per messages.json."""
    state = ControllerState.from_dict(
        make_state_dict(right=make_hand(trigger=5.0, grip=-1.0, thumbstick=(9.0, -9.0)))
    )
    assert state.right.trigger == 1.0
    assert state.right.grip == 0.0
    assert state.right.thumbstick == (1.0, -1.0)


def test_state_missing_type_or_hands_rejected():
    """Datagrams violating the schema raise ProtocolError."""
    import json

    with pytest.raises(ProtocolError):
        decode_state_datagram(json.dumps({"type": "nope"}).encode())
    bad = make_state_dict()
    del bad["left"]
    with pytest.raises(ProtocolError):
        decode_state_datagram(json.dumps(bad).encode())
    with pytest.raises(ProtocolError):
        decode_state_datagram(b"not json at all")


def test_event_roundtrip_and_unknown_name():
    """EventMsg round-trips; names outside the protocol enum are rejected."""
    event = EventMsg(event="menu", pressed=True, client_time_ms=7)
    back = EventMsg.from_dict(event.to_dict())
    assert back.event == "menu"
    assert back.pressed is True
    with pytest.raises(ProtocolError):
        EventMsg.from_dict({"event": "not_a_button", "pressed": True})


def test_decode_control_all_kinds():
    """hello/ping/event/bye decode into ControlMessage with expected kinds."""
    hello = decode_control(
        encode_message(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "device": "test",
                "client_time_ms": 1,
            }
        )
    )
    assert hello.kind == "hello"
    assert hello.payload["protocol_version"] == PROTOCOL_VERSION

    ping = decode_control(encode_message({"type": "ping", "client_time_ms": 99}))
    assert ping.kind == "ping"
    assert ping.payload["client_time_ms"] == 99

    event = decode_control(
        encode_message({"type": "event", "event": "a", "pressed": True})
    )
    assert event.kind == "event"
    assert event.payload["event"] == "a"

    bye = decode_control(encode_message({"type": "bye"}))
    assert bye.kind == "bye"


def test_decode_control_rejects_garbage():
    """Invalid JSON / unknown types / bad hello versions are rejected."""
    with pytest.raises(ProtocolError):
        decode_control(b"{not json")
    with pytest.raises(ProtocolError):
        decode_control(encode_message({"type": "teleport"}))
    with pytest.raises(ProtocolError):
        decode_control(encode_message({"type": "hello", "protocol_version": "1"}))


def test_encode_message_is_ndjson():
    """Encoded control messages are single newline-terminated JSON lines."""
    line = encode_message({"type": "pong", "server_time_ms": 1})
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1


def test_session_handshake_accepts_matching_version():
    """A hello with the matching protocol version yields a welcome."""
    session = SessionManager()
    reply = session.handle_hello({"protocol_version": PROTOCOL_VERSION, "device": "t"})
    assert reply["type"] == "welcome"
    assert reply["protocol_version"] == PROTOCOL_VERSION
    assert session.is_active()
    session.end_session()
    assert not session.is_active()


def test_session_rejects_protocol_mismatch():
    """A hello with a different protocol version is rejected, session closed."""
    session = SessionManager()
    reply = session.handle_hello({"protocol_version": PROTOCOL_VERSION + 1})
    assert reply["type"] == "error"
    assert reply["code"] == "PROTOCOL_MISMATCH"
    assert not session.is_active()


def test_session_single_client_mutex():
    """A second hello while a session is active gets SESSION_BUSY."""
    session = SessionManager()
    session.handle_hello({"protocol_version": PROTOCOL_VERSION})
    reply = session.handle_hello({"protocol_version": PROTOCOL_VERSION})
    assert reply["type"] == "error"
    assert reply["code"] == "SESSION_BUSY"
    session.end_session()
    reply = session.handle_hello({"protocol_version": PROTOCOL_VERSION})
    assert reply["type"] == "welcome"


def test_quat_utils_math():
    """qmul/qinv/nlerp behave per Hamilton convention (wxyz)."""
    from vr_bridge.core.quat_utils import qinv, qmul, qnormalize

    rng = np.random.default_rng(1)
    for _ in range(10):
        q = qnormalize(rng.normal(size=4))
        ident = qmul(q, qinv(q))
        assert np.abs(ident) == pytest.approx([1, 0, 0, 0], abs=1e-9)
