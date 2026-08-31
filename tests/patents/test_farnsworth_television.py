"""Tests for the Farnsworth Television System simulation (US 1,773,980)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.farnsworth_television import (
    TelevisionSystemSimulation,
)

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> TelevisionSystemSimulation:
    """Create an un-built simulation for pure-physics tests."""
    sim = TelevisionSystemSimulation(PatentSimConfig(headless=True, device="cpu"))
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _scan(sim: TelevisionSystemSimulation, seconds: float) -> None:
    """Advance the raster scan without Genesis."""
    steps = int(round(seconds / sim.config.dt))
    for _ in range(steps):
        sim._advance_scan(sim.config.dt)
        sim._time += sim.config.dt


class TestElectronGun:
    """Beam acceleration physics of the electron gun."""

    def test_beam_velocity_at_1kv(self) -> None:
        # v = sqrt(2 e V / m_e) = 1.876e7 m/s at 1 kV (~0.063c).
        sim = _bare_sim(accel_voltage_kv=1.0)
        assert sim.beam_velocity() == pytest.approx(1.876e7, rel=1e-2)
        metrics = sim._compute_metrics()
        assert metrics["beam_velocity_c"] == pytest.approx(0.0626, rel=1e-2)

    def test_velocity_scales_with_sqrt_voltage(self) -> None:
        sim = _bare_sim()
        assert sim.beam_velocity(4.0) == pytest.approx(
            2.0 * sim.beam_velocity(1.0), rel=1e-9
        )

    def test_nonrelativistic_regime(self) -> None:
        # Even at the 10 kV ceiling the beam stays below 0.21c, so the
        # classical kinetic-energy model is adequate.
        sim = _bare_sim(accel_voltage_kv=10.0)
        assert 0.19 < sim.beam_velocity() / 299792458.0 < 0.21

    def test_voltage_parameter_clipped(self) -> None:
        sim = _bare_sim(accel_voltage_kv=500.0)
        assert sim.accel_voltage() == pytest.approx(10_000.0)


class TestMagneticDeflection:
    """Magnetic stiffness of the deflected beam."""

    def test_quadrupling_voltage_halves_deflection(self) -> None:
        # Displacement ~ 1/v ~ 1/sqrt(V): 1 kV -> 4 kV halves the throw.
        sim = _bare_sim(deflection_drive=0.8)
        half = sim.SCREEN_WIDTH / 2
        low = sim.max_deflection(half, voltage_kv=1.0)
        high = sim.max_deflection(half, voltage_kv=4.0)
        assert high == pytest.approx(low / 2.0, rel=1e-9)

    def test_doubling_voltage_shrinks_deflection(self) -> None:
        sim = _bare_sim(deflection_drive=0.8)
        half = sim.SCREEN_WIDTH / 2
        low = sim.max_deflection(half, voltage_kv=2.0)
        high = sim.max_deflection(half, voltage_kv=4.0)
        assert high == pytest.approx(low / np.sqrt(2.0), rel=1e-9)

    def test_deflection_scales_with_drive(self) -> None:
        sim = _bare_sim()
        half = sim.SCREEN_WIDTH / 2
        assert sim.max_deflection(half, drive=0.5) == pytest.approx(
            sim.max_deflection(half, drive=1.0) / 2.0, rel=1e-9
        )
        assert sim.max_deflection(half, drive=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_full_drive_scans_full_screen_at_reference_voltage(self) -> None:
        sim = _bare_sim(accel_voltage_kv=5.0, deflection_drive=1.0)
        assert sim.max_deflection(sim.SCREEN_WIDTH / 2) == pytest.approx(
            sim.SCREEN_WIDTH / 2, rel=1e-9
        )
        assert sim.max_deflection(sim.SCREEN_HEIGHT / 2) == pytest.approx(
            sim.SCREEN_HEIGHT / 2, rel=1e-9
        )


class TestRasterScan:
    """Sawtooth scan timing, coverage, and frame bookkeeping."""

    def test_line_frequency_tracks_num_lines(self) -> None:
        sim = _bare_sim(num_lines=60.0)
        assert sim.num_lines() == 60
        assert sim.line_frequency() == pytest.approx(30.0 * 60.0)
        assert sim.frame_frequency() == pytest.approx(30.0)

    def test_spot_covers_full_screen(self) -> None:
        # Sample the spot densely over one frame: at full drive and 5 kV
        # the sawteeth sweep it across the entire screen.
        sim = _bare_sim(accel_voltage_kv=5.0, deflection_drive=1.0)
        frame = 1.0 / sim.frame_frequency()
        ts = np.linspace(0.0, frame, 50_000, endpoint=False)
        xs = np.array([sim.spot_position(float(t))[0] for t in ts])
        ys = np.array([sim.spot_position(float(t))[1] for t in ts])
        assert xs.min() == pytest.approx(-sim.SCREEN_WIDTH / 2, rel=1e-3)
        assert xs.max() == pytest.approx(sim.SCREEN_WIDTH / 2, rel=1e-3)
        assert ys.min() == pytest.approx(-sim.SCREEN_HEIGHT / 2, rel=1e-3)
        assert ys.max() == pytest.approx(sim.SCREEN_HEIGHT / 2, rel=1e-3)

    def test_spot_returns_to_origin_after_frame(self) -> None:
        # At the frame wrap the spot jumps from the bottom-right corner
        # back to the top-left corner to start the next frame.
        sim = _bare_sim()
        frame = 1.0 / sim.frame_frequency()
        x_end, y_end = sim.spot_position(frame - 1e-6)
        x_new, y_new = sim.spot_position(frame + 1e-6)
        assert y_end < 0.0 < y_new
        assert x_end > 0.0 > x_new
        assert y_new == pytest.approx(
            sim.max_deflection(sim.SCREEN_HEIGHT / 2), rel=1e-2
        )

    def test_frames_drawn_increase_with_time(self) -> None:
        sim = _bare_sim()
        assert sim.frames_drawn() == 0
        _scan(sim, 0.5)  # 0.5 s at 30 Hz -> 15 frames
        assert sim.frames_drawn() == 15
        _scan(sim, 0.5)
        assert sim.frames_drawn() == 30

    def test_line_number_advances_within_frame(self) -> None:
        sim = _bare_sim(num_lines=30.0)
        frame = 1.0 / sim.frame_frequency()
        sim._scan_time = 0.25 * frame
        quarter = sim.line_number()
        sim._scan_time = 0.75 * frame
        three_quarter = sim.line_number()
        assert quarter == 7
        assert three_quarter == 22
        assert 0 <= three_quarter < sim.num_lines()

    def test_reduced_drive_shrinks_scan(self) -> None:
        sim = _bare_sim(deflection_drive=0.5)
        half = sim.SCREEN_WIDTH / 2
        x, _ = sim.spot_position(0.5 / sim.line_frequency())
        assert abs(x) <= half * 0.5 + 1e-9


class TestTestPattern:
    """Crosshair/grid test-pattern modulation of the beam."""

    def test_crosshair_is_bright(self) -> None:
        sim = _bare_sim()
        assert sim.test_pattern(0.0, 0.0) == pytest.approx(1.0)
        assert sim.test_pattern(0.05, 0.0) == pytest.approx(1.0)

    def test_background_is_dim(self) -> None:
        sim = _bare_sim()
        assert sim.test_pattern(0.03, 0.03) == pytest.approx(0.1)


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = TelevisionSystemSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["frames_drawn"] == 0.0
        for _ in range(5):
            state = sim.step()
        assert "spot_x" in state.metrics
        assert "beam_velocity_c" in state.metrics
        # 5 steps * 10 substeps * 0.01 s = 0.5 s -> 15 frames at 30 Hz.
        assert state.metrics["frames_drawn"] == pytest.approx(15.0)
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = TelevisionSystemSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "accel_voltage_kv" in sim.list_parameters()
        sim.set_parameter("accel_voltage_kv", 8.0)
        assert sim.get_parameter("accel_voltage_kv") == pytest.approx(8.0)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
