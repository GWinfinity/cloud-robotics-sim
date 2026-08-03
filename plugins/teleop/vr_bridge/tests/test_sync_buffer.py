"""Sync layer tests: mailbox overwrite semantics, loss stats, clock sync."""

from __future__ import annotations

import pytest
from conftest import make_state_dict
from vr_bridge.core.messages import ControllerState, EventMsg
from vr_bridge.core.sync_buffer import ClockSync, EventQueue, StateMailbox


def _state(seq: int, client_time_ms: int = 0) -> ControllerState:
    return ControllerState.from_dict(
        make_state_dict(seq=seq, client_time_ms=client_time_ms)
    )


def test_mailbox_empty_returns_none():
    """An empty mailbox yields no sample."""
    assert StateMailbox().get() is None


def test_mailbox_keeps_only_latest():
    """Overwrite semantics: the newest put wins, older samples are gone."""
    mb = StateMailbox()
    mb.put(_state(seq=1, client_time_ms=1000), server_recv_ms=1000.0)
    mb.put(_state(seq=2, client_time_ms=2000), server_recv_ms=2000.0)
    sample = mb.get()
    assert sample is not None
    state, _ = sample
    assert state.seq == 2
    assert mb.received == 2


def test_mailbox_age_is_non_negative():
    """Reported age is clamped to >= 0 even if clocks disagree."""
    mb = StateMailbox()
    mb.put(_state(seq=1, client_time_ms=10**9))  # client clock far ahead
    _, age_ms = mb.get()
    assert age_ms >= 0.0


def test_out_of_order_packet_does_not_count_as_loss():
    """A late (lower seq) packet is kept by the mailbox but not 'lost'."""
    mb = StateMailbox()
    mb.put(_state(seq=5), server_recv_ms=1000.0)
    mb.put(_state(seq=3), server_recv_ms=1001.0)
    assert mb.lost == 0
    assert mb.received == 2
    state, _ = mb.get()
    assert state.seq == 3  # last write wins (documented overwrite semantics)


def test_packet_loss_counting_and_rate():
    """Sequence gaps are counted as lost packets."""
    mb = StateMailbox()
    mb.put(_state(seq=1), server_recv_ms=1000.0)
    mb.put(_state(seq=5), server_recv_ms=1001.0)  # 2,3,4 lost
    assert mb.lost == 3
    assert mb.loss_rate == pytest.approx(3 / 5)


def test_mailbox_clear():
    """clear() drops the sample and resets sequence tracking."""
    mb = StateMailbox()
    mb.put(_state(seq=10), server_recv_ms=1000.0)
    mb.clear()
    assert mb.get() is None
    mb.put(_state(seq=1), server_recv_ms=1002.0)
    assert mb.lost == 0  # seq tracking restarted


def test_clock_sync_offset_none_before_samples():
    """No offset estimate exists before the first sample."""
    clock = ClockSync()
    assert clock.offset_ms is None
    assert clock.to_server_time(123.0) is None


def test_clock_sync_converges_to_min_offset():
    """Min-filtering rejects jitter: estimate converges to the true offset."""
    clock = ClockSync()
    true_offset = 50.0
    for jitter in (12.0, 3.0, 30.0, 7.0, 20.0):
        client_t = 1000.0
        clock.update(client_t, server_recv_ms=client_t + true_offset + jitter)
    assert clock.offset_ms == pytest.approx(50.0 + 3.0)
    assert clock.to_server_time(1000.0) == pytest.approx(1053.0)


def test_event_queue_fifo_order():
    """Events drain in FIFO order (they must never be reordered)."""
    q = EventQueue()
    for i, name in enumerate(("a", "b", "menu")):
        q.put(EventMsg(event=name, pressed=True, client_time_ms=i))
    drained = q.drain()
    assert [e.event for e in drained] == ["a", "b", "menu"]
    assert q.drain() == []  # drained queue is empty


def test_event_queue_overflow_drops_oldest():
    """On overflow the oldest event is dropped, never the newest."""
    q = EventQueue(maxsize=4)
    for i in range(6):
        q.put(EventMsg(event="a", pressed=True, client_time_ms=i))
    drained = q.drain()
    assert len(drained) == 4
    assert [e.client_time_ms for e in drained] == [2, 3, 4, 5]
