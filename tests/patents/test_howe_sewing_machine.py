"""Tests for the Howe Sewing Machine simulation (US 4,750)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.howe_sewing_machine import (
    SewingMachineSimulation,
)

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> SewingMachineSimulation:
    """Create an un-built simulation for pure-kinematics tests."""
    sim = SewingMachineSimulation(PatentSimConfig(headless=True, device="cpu"))
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _run(sim: SewingMachineSimulation, seconds: float) -> None:
    """Advance the machine kinematics without Genesis."""
    steps = int(round(seconds / sim.config.dt))
    for _ in range(steps):
        sim._advance(sim.config.dt)
        sim._time += sim.config.dt


class TestNeedleKinematics:
    """Slider-crank needle drive and shuttle timing."""

    def test_needle_zero_at_top_dead_center(self) -> None:
        sim = _bare_sim()
        assert sim.needle_position(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_needle_stroke_at_bottom_dead_center(self) -> None:
        # x(pi) = 2r: the crank term dominates, the rod harmonic vanishes.
        sim = _bare_sim()
        assert sim.needle_position(np.pi) == pytest.approx(
            2.0 * sim.CRANK_RADIUS, rel=1e-9
        )

    def test_connecting_rod_harmonic(self) -> None:
        # At theta = pi/2 the finite rod adds r^2/(2L) to the crank term.
        sim = _bare_sim()
        r, rod = sim.CRANK_RADIUS, sim.ROD_LENGTH
        expected = r + r * r / (2.0 * rod)
        assert sim.needle_position(np.pi / 2) == pytest.approx(expected, rel=1e-9)

    def test_angular_velocity_from_rpm(self) -> None:
        sim = _bare_sim(wheel_rpm=300.0)
        assert sim.angular_velocity() == pytest.approx(10.0 * np.pi)

    def test_shuttle_is_phase_lagged_sine(self) -> None:
        sim = _bare_sim()
        theta = 1.23
        assert sim.shuttle_position(theta) == pytest.approx(
            sim.SHUTTLE_AMPLITUDE * np.sin(theta - sim.SHUTTLE_PHASE)
        )
        # Peak excursion occurs after BDC (theta = pi), while the needle
        # rises and the thread loop is open for the shuttle point.
        peak_theta = sim.SHUTTLE_PHASE + np.pi / 2
        assert np.pi < peak_theta < 2.0 * np.pi

    def test_needle_phase_wraps(self) -> None:
        sim = _bare_sim()
        sim._wheel_angle = 2.5 * np.pi
        assert sim.needle_phase() == pytest.approx(0.5 * np.pi)


class TestStitchFormation:
    """One stitch per wheel revolution and the cloth feed."""

    def test_one_stitch_per_revolution(self) -> None:
        # 300 rpm = 5 rev/s; 2 s -> exactly 10 stitches.
        sim = _bare_sim(wheel_rpm=300.0, thread_tension=0.5)
        _run(sim, 2.0)
        assert sim._stitches == 10

    def test_stitches_grow_linearly_with_revolutions(self) -> None:
        sim = _bare_sim(wheel_rpm=600.0, thread_tension=0.5)
        _run(sim, 0.5)  # 5 revolutions
        first = sim._stitches
        _run(sim, 0.5)
        assert first == 5
        assert sim._stitches == 2 * first

    def test_cloth_feed_equals_stitches_times_length(self) -> None:
        sim = _bare_sim(wheel_rpm=300.0, stitch_length_mm=3.0, thread_tension=0.5)
        _run(sim, 1.0)  # 5 stitches
        assert sim._cloth_position_mm == pytest.approx(sim._stitches * 3.0)

    def test_rpm_parameter_clipped(self) -> None:
        sim = _bare_sim(wheel_rpm=10_000.0)
        assert sim.wheel_rpm() == pytest.approx(600.0)


class TestThreadTension:
    """Thread-tension window: missed stitches below, breakage above."""

    def test_low_tension_misses_stitches(self) -> None:
        # Loop too small for the shuttle point: every cycle is skipped.
        sim = _bare_sim(wheel_rpm=300.0, thread_tension=0.1)
        _run(sim, 2.0)
        assert sim._missed_stitches == 10
        assert sim._stitches == 0
        assert sim._cloth_position_mm == pytest.approx(0.0)
        assert not sim._thread_broken

    def test_high_tension_breaks_thread_and_halts(self) -> None:
        sim = _bare_sim(wheel_rpm=300.0, thread_tension=0.95)
        _run(sim, 0.5)  # first completed revolution snaps the thread
        assert sim._thread_broken
        assert sim._stitches == 0
        # The machine is halted: wheel, counters, and feed all freeze.
        angle = sim._wheel_angle
        missed = sim._missed_stitches
        _run(sim, 1.0)
        assert sim._wheel_angle == pytest.approx(angle)
        assert sim._missed_stitches == missed
        assert sim._stitches == 0

    def test_normal_tension_window_sews(self) -> None:
        for tension in (0.3, 0.5, 0.7):
            sim = _bare_sim(wheel_rpm=300.0, thread_tension=tension)
            _run(sim, 1.0)
            assert sim._stitches == 5
            assert sim._missed_stitches == 0
            assert not sim._thread_broken

    def test_tension_boundary_values(self) -> None:
        sim = _bare_sim(thread_tension=0.69)
        assert sim._compute_metrics()["tension_ok"] == 1.0
        sim = _bare_sim(thread_tension=0.71)
        assert sim._compute_metrics()["tension_ok"] == 0.0


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = SewingMachineSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["stitches"] == 0.0
        for _ in range(5):
            state = sim.step()
        # 5 steps * 10 substeps * 0.01 s = 0.5 s at 300 rpm -> 2.5 wheel
        # revolutions -> 2 completed stitches feeding 2 mm each.
        assert state.metrics["stitches"] == pytest.approx(2.0)
        assert state.metrics["cloth_position_mm"] == pytest.approx(4.0)
        assert "needle_phase" in state.metrics
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = SewingMachineSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "wheel_rpm" in sim.list_parameters()
        sim.set_parameter("wheel_rpm", 120.0)
        assert sim.get_parameter("wheel_rpm") == pytest.approx(120.0)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
