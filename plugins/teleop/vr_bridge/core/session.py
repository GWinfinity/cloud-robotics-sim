"""Session lifecycle: handshake, protocol negotiation, single-client mutex."""

from __future__ import annotations

import itertools
import threading

from .messages import PROTOCOL_VERSION
from .sync_buffer import now_ms

_session_counter = itertools.count(1)


class SessionManager:
    """Track the single active control session.

    The bridge intentionally allows only one client at a time: two
    operators fighting over one robot is a safety hazard, not a feature.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session_id: str | None = None
        self.device: str = ""
        self.active: bool = False

    def handle_hello(self, payload: dict) -> dict:
        """Process a ``hello`` message; return ``welcome`` or ``error``."""
        version = payload.get("protocol_version")
        if version != PROTOCOL_VERSION:
            return {
                "type": "error",
                "code": "PROTOCOL_MISMATCH",
                "message": (
                    f"server speaks protocol v{PROTOCOL_VERSION}, "
                    f"client sent v{version}"
                ),
            }
        with self._lock:
            if self.active:
                return {
                    "type": "error",
                    "code": "SESSION_BUSY",
                    "message": "another client already controls the robot",
                }
            self.session_id = f"s{next(_session_counter):04d}"
            self.device = str(payload.get("device", "unknown"))
            self.active = True
            return {
                "type": "welcome",
                "protocol_version": PROTOCOL_VERSION,
                "session_id": self.session_id,
                "server_time_ms": int(now_ms()),
            }

    def end_session(self) -> None:
        with self._lock:
            self.session_id = None
            self.device = ""
            self.active = False

    def is_active(self) -> bool:
        with self._lock:
            return self.active
