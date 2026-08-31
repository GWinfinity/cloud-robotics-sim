"""Tests for the Goddard Rocket Apparatus simulation (US 1,155,986)."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.goddard_rocket import RocketApparatusSimulation

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> RocketApparatusSimulation:
    """Create an un-built simulation for pure-physics tests."""
    sim = RocketApparatusSimulation(PatentSimConfig(headless=True, device="cpu"))
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _fly(sim: RocketApparatusSimulation, seconds: float) -> None:
    """Integrate the flight model without Genesis."""
    steps = int(seconds / sim.config.dt)
    for _ in range(steps):
        sim._integrate_flight(sim.config.dt)
        sim._time += sim.config.dt


class TestMassAndThrust:
    """Mass bookkeeping and thrust behavior."""

    def test_liftoff_mass_includes_both_stages_and_payload(self) -> None:
        sim = _bare_sim(payload_kg=10.0)
        expected = 40.0 + 120.0 + 15.0 + 45.0 + 10.0
        assert sim.total_mass() == pytest.approx(expected)

    def test_thrust_scales_with_throttle(self) -> None:
        full = _bare_sim(throttle=1.0)
        half = _bare_sim(throttle=0.5)
        assert full.thrust() == pytest.approx(half.thrust() * 2.0)

    def test_thrust_zero_when_propellant_spent(self) -> None:
        sim = _bare_sim(throttle=1.0, auto_stage=0.0)
        sim._prop1 = 0.0
        assert sim.thrust() == 0.0

    def test_tsiolkovsky_delta_v(self) -> None:
        sim = _bare_sim(payload_kg=0.0)
        # Stage 2 alone: v_e * ln(60 / 15) = 2000 * ln(4).
        assert sim.ideal_delta_v(2) == pytest.approx(2000.0 * 1.3862944, rel=1e-3)
        assert sim.ideal_delta_v(1) > 0.0


class TestFlightDynamics:
    """Ascent, hold-down and gimbal behavior."""

    def test_full_throttle_lifts_off(self) -> None:
        sim = _bare_sim(throttle=1.0)
        _fly(sim, 2.0)
        assert sim._altitude > 0.0
        assert not sim._on_pad

    def test_low_throttle_stays_on_pad(self) -> None:
        # T/W < 1: hold-down clamps keep the vehicle on the pad.
        sim = _bare_sim(throttle=0.2)
        _fly(sim, 2.0)
        assert sim._altitude == 0.0
        assert sim._on_pad

    def test_mass_decreases_during_burn(self) -> None:
        sim = _bare_sim(throttle=1.0)
        m0 = sim.total_mass()
        _fly(sim, 5.0)
        assert sim.total_mass() < m0
        assert sim._prop1 < sim.STAGE1_PROP

    def test_gimbal_produces_downrange_drift(self) -> None:
        straight = _bare_sim(throttle=1.0, gimbal=0.0)
        tilted = _bare_sim(throttle=1.0, gimbal=1.0)
        _fly(straight, 5.0)
        _fly(tilted, 5.0)
        assert straight._downrange == pytest.approx(0.0, abs=1e-9)
        assert tilted._downrange > 1.0


class TestStaging:
    """Stage separation and second-stage ignition."""

    def test_auto_stage_separates_at_burnout(self) -> None:
        sim = _bare_sim(throttle=1.0, auto_stage=1.0)
        _fly(sim, 35.0)  # stage 1 burns 120 kg / 4 kg/s = 30 s
        assert sim._separated
        assert sim._prop1 == 0.0
        # Mass dropped by the empty stage-1 structure.
        assert sim.total_mass() == pytest.approx(15.0 + sim._prop2 + 10.0)

    def test_no_staging_when_disabled(self) -> None:
        sim = _bare_sim(throttle=1.0, auto_stage=0.0)
        _fly(sim, 35.0)
        assert not sim._separated
        # Dead weight stays attached: stage-2 propellant untouched.
        assert sim._prop2 == pytest.approx(sim.STAGE2_PROP)

    def test_second_stage_accelerates_after_staging(self) -> None:
        sim = _bare_sim(throttle=1.0, auto_stage=1.0)
        _fly(sim, 31.0)
        v_at_staging = sim._v_vertical
        _fly(sim, 3.0)
        assert sim._v_vertical > v_at_staging
        assert sim._prop2 < sim.STAGE2_PROP

    def test_booster_falls_back_after_separation(self) -> None:
        sim = _bare_sim(throttle=1.0, auto_stage=1.0)
        _fly(sim, 31.0)
        _fly(sim, 2.0)
        # The unpowered booster decelerates (still rising or falling).
        assert sim._booster_vv < sim._v_vertical
        assert sim._booster_altitude >= 0.0


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = RocketApparatusSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["on_pad"] == 1.0
        for _ in range(5):
            state = sim.step()
        assert "altitude_m" in state.metrics
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = RocketApparatusSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "throttle" in sim.list_parameters()
        sim.set_parameter("throttle", 0.5)
        assert sim.get_parameter("throttle") == pytest.approx(0.5)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
