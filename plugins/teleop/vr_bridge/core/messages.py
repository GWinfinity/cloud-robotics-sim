"""Protocol v1 message types and JSON codecs.

Authoritative spec: ``protocol/v1/README.md`` + ``messages.json``.
All poses are in the Genesis world frame (right-handed, Z-up, metres),
quaternions in wxyz order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

EVENT_NAMES = ("a", "b", "x", "y", "menu", "grip_left", "grip_right")


class ProtocolError(ValueError):
    """Raised when an incoming message fails schema validation."""


@dataclass
class PoseMsg:
    """A pose: position (3,) + quaternion (4,) in wxyz order."""

    pos: np.ndarray  # (3,)
    quat: np.ndarray  # (4,) wxyz

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str = "pose") -> "PoseMsg":
        try:
            pos = np.asarray(data["pos"], dtype=np.float64)
            quat = np.asarray(data["quat"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"{name}: bad pos/quat: {exc}") from exc
        if pos.shape != (3,) or quat.shape != (4,):
            raise ProtocolError(f"{name}: pos must be (3,), quat (4,)")
        return cls(pos=pos, quat=quat)

    def to_dict(self) -> dict[str, Any]:
        return {"pos": self.pos.tolist(), "quat": self.quat.tolist()}


@dataclass
class HandState:
    """One VR controller: pose plus analog trigger/grip/thumbstick."""

    pose: PoseMsg
    trigger: float = 0.0
    grip: float = 0.0
    thumbstick: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str) -> "HandState":
        if not isinstance(data, dict):
            raise ProtocolError(f"{name}: expected object")
        pose = PoseMsg.from_dict(data, name)
        try:
            trigger = float(data.get("trigger", 0.0))
            grip = float(data.get("grip", 0.0))
            stick = data.get("thumbstick", [0.0, 0.0])
            sx, sy = float(stick[0]), float(stick[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ProtocolError(f"{name}: bad analog fields: {exc}") from exc
        return cls(
            pose=pose,
            trigger=min(max(trigger, 0.0), 1.0),
            grip=min(max(grip, 0.0), 1.0),
            thumbstick=(
                min(max(sx, -1.0), 1.0),
                min(max(sy, -1.0), 1.0),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.pose.to_dict()
        out.update(
            {
                "trigger": self.trigger,
                "grip": self.grip,
                "thumbstick": list(self.thumbstick),
            }
        )
        return out


@dataclass
class ControllerState:
    """One state-stream datagram."""

    seq: int
    client_time_ms: int
    left: HandState
    right: HandState
    head: PoseMsg | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerState":
        if not isinstance(data, dict) or data.get("type") != "state":
            raise ProtocolError("state: missing or wrong 'type'")
        try:
            seq = int(data["seq"])
            client_time_ms = int(data["client_time_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"state: bad seq/client_time_ms: {exc}") from exc
        left = HandState.from_dict(data.get("left"), "left")
        right = HandState.from_dict(data.get("right"), "right")
        head = None
        if data.get("head") is not None:
            head = PoseMsg.from_dict(data["head"], "head")
        return cls(
            seq=seq,
            client_time_ms=client_time_ms,
            left=left,
            right=right,
            head=head,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "state",
            "seq": self.seq,
            "client_time_ms": self.client_time_ms,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }
        if self.head is not None:
            out["head"] = self.head.to_dict()
        return out


@dataclass
class EventMsg:
    """A button event on the reliable control channel."""

    event: str
    pressed: bool
    client_time_ms: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMsg":
        event = data.get("event")
        if event not in EVENT_NAMES:
            raise ProtocolError(f"event: unknown name {event!r}")
        return cls(
            event=event,
            pressed=bool(data.get("pressed", False)),
            client_time_ms=int(data.get("client_time_ms", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "event",
            "event": self.event,
            "pressed": self.pressed,
            "client_time_ms": self.client_time_ms,
        }


@dataclass
class ControlMessage:
    """Any decoded control-channel message."""

    kind: str  # hello | ping | event | bye
    payload: dict[str, Any] = field(default_factory=dict)


def decode_control(line: str | bytes) -> ControlMessage:
    """Decode one NDJSON control-channel line."""
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError(f"control: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("control: expected object")
    kind = data.get("type")
    if kind == "hello":
        version = data.get("protocol_version")
        if not isinstance(version, int):
            raise ProtocolError("hello: protocol_version must be int")
        return ControlMessage(kind="hello", payload=data)
    if kind == "ping":
        return ControlMessage(
            kind="ping", payload={"client_time_ms": int(data.get("client_time_ms", 0))}
        )
    if kind == "event":
        return ControlMessage(kind="event", payload=EventMsg.from_dict(data).__dict__)
    if kind == "bye":
        return ControlMessage(kind="bye")
    raise ProtocolError(f"control: unknown type {kind!r}")


def encode_message(msg: dict[str, Any]) -> bytes:
    """Encode a control-channel message as an NDJSON line."""
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode_state_datagram(data: bytes) -> ControllerState:
    """Decode one UDP state datagram."""
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"state datagram: invalid JSON: {exc}") from exc
    return ControllerState.from_dict(parsed)
