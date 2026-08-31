"""Tests for the Edison Electric-Lamp simulation (US 223,898)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation
from cloud_robotics_sim.patents.sims.edison_lamp import ElectriclampSimulation

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except Exception:
    HAS_GENESIS = False


@pytest.fixture
def sim_config() -> PatentSimConfig:
    """Return a lightweight lamp config for tests."""
    return PatentSimConfig(
        patent_id="US223898",
        headless=True,
        dt=0.01,
        substeps=5,
        resolution=(320, 240),
        device="cpu",
    )


def _bare_sim() -> ElectriclampSimulation:
    """A simulation instance without a Genesis scene (physics only)."""
    config = PatentSimConfig(patent_id="US223898")
    sim = create_simulation("US223898", config=config)
    assert isinstance(sim, ElectriclampSimulation)
    return sim


# ---------------------------------------------------------------------------
# Pure physics (no Genesis required)
# ---------------------------------------------------------------------------


def test_carbon_filament_negative_tcr() -> None:
    """Carbon resistance falls as temperature rises (unlike tungsten)."""
    sim = _bare_sim()
    assert sim.resistance(300.0) > sim.resistance(2400.0)
    assert sim.resistance(2400.0) == pytest.approx(
        sim.R_COLD * (1.0 + sim.TCR_ALPHA * 2100.0)
    )


def test_steady_state_temperature_increases_with_voltage() -> None:
    """Higher supply voltage settles at a hotter steady state."""
    temps = []
    for voltage in (40.0, 80.0, 100.0):
        sim = _bare_sim()
        sim.set_parameter("voltage", voltage)
        for _ in range(200):  # 2 s of thermal integration
            sim._integrate_thermal(0.01)
        temps.append(sim._temperature)
    assert temps[0] < temps[1] < temps[2]
    # At rated voltage the filament should reach incandescence (~2400 K).
    assert 2000.0 < temps[2] < 3000.0


def test_air_leak_cools_filament() -> None:
    """A leaky bulb (vacuum=0) runs cooler than a hard vacuum at same V."""
    temps = {}
    for vacuum in (1.0, 0.0):
        sim = _bare_sim()
        sim.set_parameter("voltage", 100.0)
        sim.set_parameter("vacuum", vacuum)
        for _ in range(200):
            sim._integrate_thermal(0.01)
        temps[vacuum] = sim._temperature
    assert temps[1.0] > temps[0.0]


def test_wear_rate_arrhenius() -> None:
    """Carbon sublimation is strongly temperature dependent."""
    sim = _bare_sim()
    cold = sim.wear_rate(2200.0, vacuum=1.0)
    hot = sim.wear_rate(2600.0, vacuum=1.0)
    assert hot > 5.0 * cold
    # Oxidation in a leaky bulb multiplies the wear.
    assert sim.wear_rate(2400.0, vacuum=0.0) > sim.wear_rate(2400.0, vacuum=1.0)
    # Rated lifetime at 2400 K in vacuum is ~100 h.
    rate = sim.wear_rate(sim.T_REF, vacuum=1.0)
    assert 1.0 / rate / 3600.0 == pytest.approx(100.0, rel=0.01)


def test_luminous_efficacy_rises_with_temperature() -> None:
    """Blackbody efficacy is near zero when dull red and rises with T."""
    sim = _bare_sim()
    low = sim.luminous_flux(1000.0)
    mid = sim.luminous_flux(2000.0)
    high = sim.luminous_flux(3000.0)
    assert low < 1e-3 * high
    assert 0.0 < mid < high


def test_burnout_opens_circuit() -> None:
    """At wear = 1 the circuit opens and the filament cools to ambient."""
    sim = _bare_sim()
    sim.set_parameter("voltage", 120.0)
    sim._wear = 0.999
    sim._temperature = 2500.0
    for _ in range(300):
        sim._integrate_thermal(0.01)
        if sim._burned_out:
            break
    assert sim._burned_out
    assert sim.input_power(120.0, 2500.0) == 0.0
    for _ in range(300):
        sim._integrate_thermal(0.01)
    assert sim._temperature == pytest.approx(sim.T_AMBIENT, rel=1e-3)


def test_zero_voltage_keeps_ambient() -> None:
    """With the supply off, the filament stays at ambient temperature."""
    sim = _bare_sim()
    sim.set_parameter("voltage", 0.0)
    for _ in range(50):
        sim._integrate_thermal(0.01)
    assert sim._temperature == pytest.approx(sim.T_AMBIENT)


def test_lamp_parameter_validation() -> None:
    """Setting an unknown parameter raises KeyError."""
    sim = _bare_sim()
    with pytest.raises(KeyError):
        sim.set_parameter("unknown_param", 1.0)
    assert "voltage" in sim.list_parameters()
    assert "vacuum" in sim.list_parameters()


# ---------------------------------------------------------------------------
# Genesis integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_edison_lamp_build_reset_step(sim_config: PatentSimConfig) -> None:
    """The lamp can be built, reset, and stepped."""
    sim = create_simulation("US223898", config=sim_config)
    sim.build()
    initial_state = sim.reset()
    assert initial_state.time == 0.0
    assert "filament" in initial_state.bodies
    assert "bulb" in initial_state.bodies
    assert initial_state.metrics["filament_temp_k"] == pytest.approx(300.0)

    final_state = sim.step()
    assert final_state.time > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_edison_lamp_heats_up_and_glows(sim_config: PatentSimConfig) -> None:
    """At rated voltage the filament heats, glows, and accumulates wear."""
    sim = create_simulation("US223898", config=sim_config)
    sim.build()
    sim.reset()
    sim.set_parameter("voltage", 100.0)
    state = sim.get_state()
    assert state.metrics["wear"] == 0.0
    for _ in range(40):  # 2 simulated seconds
        state = sim.step()
    assert state.metrics["filament_temp_k"] > 1500.0
    assert state.metrics["luminous_flux_lm"] > 0.0
    assert state.metrics["input_power_w"] > 0.0
    assert 0.0 < state.metrics["wear"] < 1.0
    assert state.metrics["lifetime_hours_remaining"] > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_edison_lamp_render(sim_config: PatentSimConfig) -> None:
    """The simulation can render an RGB frame."""
    sim = create_simulation("US223898", config=sim_config)
    sim.build()
    sim.reset()
    frame = sim.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] == sim_config.resolution[1]
    assert frame.shape[1] == sim_config.resolution[0]
    sim.close()
