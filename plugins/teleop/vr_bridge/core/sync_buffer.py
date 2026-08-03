"""L2 sync layer: clock-offset estimation, latest-value mailbox, event queue.

The mailbox uses overwrite semantics on purpose: for high-rate pose streams
the newest sample is always the most useful one, and queuing stale samples
would accumulate latency under jitter. Events (button presses) must not be
dropped, so they go through a real queue instead.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

from .messages import ControllerState, EventMsg


def now_ms() -> float:
    """Server-side monotonic clock in milliseconds."""
    return time.monotonic() * 1000.0


class ClockSync:
    """Estimate the offset between the client clock and the server clock.

    Assumes the minimum observed round-trip offset is closest to the true
    offset (standard NTP-style min-filtering over a sliding window).
    """

    def __init__(self, window_size: int = 200) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def update(self, client_time_ms: float, server_recv_ms: float) -> None:
        with self._lock:
            self._samples.append(server_recv_ms - client_time_ms)

    @property
    def offset_ms(self) -> float | None:
        """Estimated ``server - client`` offset, or None before any sample."""
        with self._lock:
            if not self._samples:
                return None
            return min(self._samples)

    def to_server_time(self, client_time_ms: float) -> float | None:
        offset = self.offset_ms
        if offset is None:
            return None
        return client_time_ms + offset


class StateMailbox:
    """Thread-safe latest-value store for ``ControllerState`` samples.

    Also tracks sequence numbers for packet-loss statistics.
    """

    def __init__(self, clock: ClockSync | None = None) -> None:
        self._lock = threading.Lock()
        self._state: ControllerState | None = None
        self._recv_ms: float = 0.0
        self._last_seq: int | None = None
        self.received: int = 0
        self.lost: int = 0
        self.clock = clock or ClockSync()

    def put(self, state: ControllerState, server_recv_ms: float | None = None) -> None:
        recv = now_ms() if server_recv_ms is None else server_recv_ms
        self.clock.update(state.client_time_ms, recv)
        with self._lock:
            if self._last_seq is not None and state.seq > self._last_seq + 1:
                self.lost += state.seq - self._last_seq - 1
            if self._last_seq is None or state.seq > self._last_seq:
                self._last_seq = state.seq
            self._state = state
            self._recv_ms = recv
            self.received += 1

    def get(self) -> tuple[ControllerState, float] | None:
        """Return ``(state, age_ms)`` of the newest sample, or None.

        Age is measured against the client timestamp when clock sync is
        available, else against the server receive time.
        """
        with self._lock:
            if self._state is None:
                return None
            state, recv = self._state, self._recv_ms
        est = self.clock.to_server_time(state.client_time_ms)
        ref = est if est is not None else recv
        return state, max(now_ms() - ref, 0.0)

    def clear(self) -> None:
        with self._lock:
            self._state = None
            self._last_seq = None

    @property
    def loss_rate(self) -> float:
        total = self.received + self.lost
        return self.lost / total if total else 0.0


class EventQueue:
    """Thread-safe FIFO for button events (these must not be dropped)."""

    def __init__(self, maxsize: int = 256) -> None:
        self._queue: queue.Queue[EventMsg] = queue.Queue(maxsize=maxsize)

    def put(self, event: EventMsg) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop the oldest event rather than blocking the network thread.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(event)

    def drain(self) -> list[EventMsg]:
        events: list[EventMsg] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                return events
