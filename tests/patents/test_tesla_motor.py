"""Tests for the Tesla Electro-Magnetic Motor simulation (US 381,968)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation
from cloud_robotics_sim.patents.sims.tesla_motor import ElectromagneticMotorSimulation

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except Exception:
    HAS_GENESIS = False


@pytest.fixture
def sim_config() -> PatentSimConfig:
    """Return a lightweight motor config for tests."""
    return PatentSimConfig(
        patent_id="US381968",
        headless=True,
        dt=0.01,
        substeps=5,
        resolution=(320, 240),
        device="cpu",
    )


def _bare_sim() -> ElectromagneticMotorSimulation:
    """A simulation instance without a Genesis scene (physics only)."""
    config = PatentSimConfig(patent_id="US381968")
    sim = create_simulation("US381968", config=config)
    assert isinstance(sim, ElectromagneticMotorSimulation)
    return sim


# ---------------------------------------------------------------------------
# Pure physics (no Genesis required)
# ---------------------------------------------------------------------------


def test_quarter_phase_field_is_purely_forward() -> None:
    """At 90 deg phase offset the field only rotates forward."""
    sim = _bare_sim()
    fwd, bwd = sim.field_amplitudes(1.0, 90.0)
    assert fwd == pytest.approx(1.0)
    assert bwd == pytest.approx(0.0)


def test_zero_phase_offset_has_no_starting_torque() -> None:
    """A pure pulsating field (phi=0) cannot start the rotor — the
    counter-rotating field components cancel at standstill.
    """
    sim = _bare_sim()
    sim.set_parameter("phase_offset", 0.0)
    fwd, bwd = sim.field_amplitudes(1.0, 0.0)
    assert fwd == pytest.approx(bwd)
    assert sim.electromagnetic_torque(rotor_speed=0.0) == pytest.approx(0.0)


def test_reversed_phase_offset_reverses_torque() -> None:
    """At 270 deg the field rotates backwards and the torque flips sign."""
    sim = _bare_sim()
    sim.set_parameter("phase_offset", 270.0)
    assert sim.electromagnetic_torque(rotor_speed=0.0) < 0.0


def test_torque_zero_at_synchronism_and_peak_near_pullout() -> None:
    """The Kloss curve vanishes at synchronism and peaks near s_m."""
    sim = _bare_sim()
    assert sim.kloss_torque(1.0, 0.0) == pytest.approx(0.0)
    slips = np.linspace(0.01, 1.0, 100)
    torques = [sim.kloss_torque(1.0, s) for s in slips]
    peak_slip = slips[int(np.argmax(torques))]
    assert abs(peak_slip - sim.PULLOUT_SLIP) < 0.05


def test_motor_runs_up_below_synchronous_speed() -> None:
    """The rotor accelerates and settles below the synchronous speed."""
    sim = _bare_sim()
    sim.set_parameter("load", 0.2)
    for _ in range(500):  # 5 s
        sim._integrate_mechanics(0.01)
    w_sync = sim.synchronous_speed(sim.RATED_FREQUENCY)
    assert 0.0 < sim._rotor_speed < w_sync
    rpm = sim._rotor_speed * 60.0 / (2.0 * np.pi)
    assert rpm > 0.5 * w_sync * 60.0 / (2.0 * np.pi)


def test_higher_load_increases_slip() -> None:
    """More brake torque -> more slip -> lower steady speed."""
    speeds = {}
    for load in (0.1, 0.5):
        sim = _bare_sim()
        sim.set_parameter("load", load)
        for _ in range(500):
            sim._integrate_mechanics(0.01)
        speeds[load] = sim._rotor_speed
    assert speeds[0.1] > speeds[0.5] > 0.0


def test_frequency_sets_synchronous_speed() -> None:
    """Doubling the supply frequency doubles the synchronous speed."""
    sim = _bare_sim()
    assert sim.synchronous_speed(120.0) == pytest.approx(
        2.0 * sim.synchronous_speed(60.0)
    )


def test_zero_current_produces_no_torque() -> None:
    """With the stator unpowered the rotor stays at rest."""
    sim = _bare_sim()
    sim.set_parameter("current", 0.0)
    for _ in range(50):
        sim._integrate_mechanics(0.01)
    assert sim._rotor_speed == pytest.approx(0.0)


def test_motor_parameter_validation() -> None:
    """Setting an unknown parameter raises KeyError."""
    sim = _bare_sim()
    with pytest.raises(KeyError):
        sim.set_parameter("unknown_param", 1.0)
    assert "frequency" in sim.list_parameters()
    assert "phase_offset" in sim.list_parameters()


# ---------------------------------------------------------------------------
# Genesis integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_tesla_motor_build_reset_step(sim_config: PatentSimConfig) -> None:
    """The motor can be built, reset, and stepped."""
    sim = create_simulation("US381968", config=sim_config)
    sim.build()
    initial_state = sim.reset()
    assert initial_state.time == 0.0
    assert "rotor" in initial_state.bodies
    assert "stator" in initial_state.bodies
    assert initial_state.metrics["rotor_speed_rpm"] == pytest.approx(0.0)

    final_state = sim.step()
    assert final_state.time > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_tesla_motor_spins_up(sim_config: PatentSimConfig) -> None:
    """With a quarter-phase supply the rotor spins up and the angle moves."""
    sim = create_simulation("US381968", config=sim_config)
    sim.build()
    sim.reset()
    for _ in range(50):  # 2.5 simulated seconds
        state = sim.step()
    assert state.metrics["rotor_speed_rpm"] > 100.0
    assert state.metrics["torque_nm"] != 0.0
    assert state.metrics["slip"] > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_tesla_motor_render(sim_config: PatentSimConfig) -> None:
    """The simulation can render an RGB frame."""
    sim = create_simulation("US381968", config=sim_config)
    sim.build()
    sim.reset()
    frame = sim.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] == sim_config.resolution[1]
    assert frame.shape[1] == sim_config.resolution[0]
    sim.close()
