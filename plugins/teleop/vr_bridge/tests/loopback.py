"""Shared loopback helpers for vr_bridge end-to-end tests.

A minimal protocol-v1 client (UDP state stream + TCP control channel) plus
stepping utilities, used by ``test_e2e_loopback`` and ``test_e2e_recording``.
"""

from __future__ import annotations

import json
import socket
import threading
import time

from conftest import make_hand, make_state_dict
from vr_bridge.core.bridge import VRBridge
from vr_bridge.core.messages import PROTOCOL_VERSION, encode_message

HOST = "127.0.0.1"


class LoopbackClient:
    """Minimal protocol-v1 client speaking to a running VRBridge."""

    def __init__(self, state_port: int, control_port: int) -> None:
        self.state_port = state_port
        self.control_port = control_port
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tcp: socket.socket | None = None
        self.seq = 0
        self._stop_stream = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._hand = make_hand()

    # ------------------------------------------------------------------
    def handshake(self, version: int = PROTOCOL_VERSION) -> dict:
        """Connect on TCP and perform the hello/welcome handshake."""
        self.tcp = socket.create_connection((HOST, self.control_port), timeout=5.0)
        self.tcp.settimeout(5.0)
        self.tcp.sendall(
            encode_message(
                {
                    "type": "hello",
                    "protocol_version": version,
                    "device": "loopback_test",
                    "client_time_ms": 0,
                }
            )
        )
        return self.read_line(self.tcp)

    @staticmethod
    def read_line(conn: socket.socket) -> dict:
        """Read one NDJSON line from the control channel."""
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                raise ConnectionError("server closed the control channel")
            data += chunk
        return json.loads(data.split(b"\n", 1)[0])

    # ------------------------------------------------------------------
    def set_hand(self, **kwargs) -> None:
        """Replace the right-hand state sent by the streaming thread."""
        with self._lock:
            self._hand = make_hand(**kwargs)

    def start_stream(self, rate_hz: float = 50.0) -> None:
        """Stream state datagrams on a background thread."""
        self._stop_stream.clear()

        def _loop() -> None:
            while not self._stop_stream.is_set():
                with self._lock:
                    hand = dict(self._hand)
                state = make_state_dict(
                    seq=self.seq,
                    client_time_ms=int(time.monotonic() * 1000),
                    right=hand,
                )
                self.seq += 1
                try:
                    self.udp.sendto(json.dumps(state).encode(), (HOST, self.state_port))
                except OSError:
                    return
                time.sleep(1.0 / rate_hz)

        self._stream_thread = threading.Thread(target=_loop, daemon=True)
        self._stream_thread.start()

    def stop_stream(self) -> None:
        """Stop the streaming thread."""
        self._stop_stream.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)

    def send_event(self, event: str, pressed: bool) -> None:
        """Send one button event on the reliable control channel."""
        assert self.tcp is not None
        self.tcp.sendall(
            encode_message({"type": "event", "event": event, "pressed": pressed})
        )

    def close(self) -> None:
        """Say bye and release both sockets."""
        self.stop_stream()
        if self.tcp is not None:
            try:
                self.tcp.sendall(encode_message({"type": "bye"}))
                self.tcp.close()
            except OSError:
                pass
        self.udp.close()


def run_steps(bridge: VRBridge, duration: float, dt: float = 0.01) -> dict:
    """Step the bridge for ``duration`` seconds; return the last info dict."""
    info: dict = {}
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        info = bridge.step(dt)
        time.sleep(dt)
    return info


def wait_state(bridge: VRBridge, want: str, timeout: float = 3.0) -> dict:
    """Step until safety_state == want or timeout; return last info."""
    info: dict = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = bridge.step(0.01)
        if info["safety_state"] == want:
            return info
        time.sleep(0.01)
    raise AssertionError(f"safety_state never became {want!r}, last={info}")
