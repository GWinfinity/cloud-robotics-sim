"""Safety supervisor tests: workspace/speed limits, watchdog, estop."""

from __future__ import annotations

import math

import numpy as np
import pytest
from vr_bridge.core.messages import PoseMsg
from vr_bridge.core.safety import SafetyConfig, SafetyState, SafetySupervisor


def _pose(pos, quat=None) -> PoseMsg:
    return PoseMsg(
        pos=np.asarray(pos, dtype=np.float64),
        quat=np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else np.asarray(quat),
    )


@pytest.fixture()
def supervisor() -> SafetySupervisor:
    """Supervisor matching configs/vr_bridge.yaml."""
    return SafetySupervisor(
        SafetyConfig(
            workspace_center=(0.0, 0.0, 0.35),
            workspace_radius=0.8,
            max_ee_speed=1.0,
            freeze_timeout_ms=100.0,
            disconnect_timeout_ms=1000.0,
        )
    )


def test_freshness_thresholds(supervisor):
    """Age <= 100ms OK, 100-1000ms STALE, > 1000ms DISCONNECTED."""
    assert supervisor.freshness(0.0) is SafetyState.OK
    assert supervisor.freshness(100.0) is SafetyState.OK
    assert supervisor.freshness(101.0) is SafetyState.STALE
    assert supervisor.freshness(1000.0) is SafetyState.STALE
    assert supervisor.freshness(1001.0) is SafetyState.DISCONNECTED
    assert supervisor.freshness(math.inf) is SafetyState.DISCONNECTED


def test_estop_latches_and_overrides_freshness(supervisor):
    """E-stop is latched, dominates the watchdog, and releases cleanly."""
    supervisor.engage_estop()
    assert supervisor.estopped is True
    assert supervisor.freshness(0.0) is SafetyState.ESTOP
    supervisor.release_estop()
    assert supervisor.estopped is False
    assert supervisor.freshness(0.0) is SafetyState.OK


def test_workspace_projection_not_dropping(supervisor):
    """Out-of-workspace targets are projected onto the sphere boundary."""
    # 1.6m along +x from the centre: 2x the radius.
    clamped = supervisor.clamp_target(_pose([1.6, 0.0, 0.35]), dt=0.01, reference=None)
    offset = clamped.pos - np.array([0.0, 0.0, 0.35])
    assert np.linalg.norm(offset) == pytest.approx(0.8)
    # Direction preserved (projected, not dropped or zeroed).
    assert clamped.pos[0] > 0.0
    assert clamped.pos[1] == pytest.approx(0.0)
    assert clamped.pos[2] == pytest.approx(0.35)


def test_workspace_inside_passes_through(supervisor):
    """Targets already inside the workspace are untouched."""
    pos = [0.3, 0.1, 0.4]
    clamped = supervisor.clamp_target(_pose(pos), dt=0.01, reference=None)
    assert clamped.pos == pytest.approx(pos)


def test_speed_limit_caps_per_tick_step(supervisor):
    """Per-tick translation is capped at max_ee_speed * dt from reference."""
    reference = _pose([0.0, 0.0, 0.35])
    target = _pose([0.1, 0.0, 0.35])  # 0.1m away; limit is 1.0 * 0.01 = 0.01m
    clamped = supervisor.clamp_target(target, dt=0.01, reference=reference)
    step = clamped.pos - reference.pos
    assert np.linalg.norm(step) == pytest.approx(0.01)
    assert clamped.pos[0] > 0.0  # direction preserved


def test_speed_limit_unclamped_without_reference(supervisor):
    """Without a previous target there is no speed reference to clamp to."""
    target = _pose([0.1, 0.0, 0.35])
    clamped = supervisor.clamp_target(target, dt=0.01, reference=None)
    assert clamped.pos == pytest.approx(target.pos)


def test_unlimited_config_disables_clamps():
    """Default config (inf radius/speed) leaves every target untouched."""
    supervisor = SafetySupervisor(SafetyConfig())
    target = _pose([100.0, -50.0, 25.0])
    clamped = supervisor.clamp_target(target, dt=0.01, reference=_pose([0, 0, 0]))
    assert clamped.pos == pytest.approx(target.pos)


def test_quaternion_passes_through_clamp(supervisor):
    """Orientation is not altered by position clamping."""
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    clamped = supervisor.clamp_target(_pose([9.0, 9.0, 9.0], quat), 0.01, None)
    assert clamped.quat == pytest.approx(quat)


def test_safety_config_from_dict_roundtrip():
    """from_dict parses the vr_bridge.yaml safety block."""
    cfg = SafetyConfig.from_dict(
        {
            "workspace_center": [0.0, 0.0, 0.35],
            "workspace_radius": 0.8,
            "max_ee_speed": 1.0,
            "freeze_timeout_ms": 100,
            "disconnect_timeout_ms": 1000,
        }
    )
    assert cfg.workspace_center == (0.0, 0.0, 0.35)
    assert cfg.workspace_radius == pytest.approx(0.8)
    assert cfg.max_ee_speed == pytest.approx(1.0)
    assert cfg.freeze_timeout_ms == pytest.approx(100.0)
    assert cfg.disconnect_timeout_ms == pytest.approx(1000.0)
