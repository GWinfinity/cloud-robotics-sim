"""Tests for the Morse Electro-Magnetic Telegraph simulation (US 1,647)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation
from cloud_robotics_sim.patents.sims.morse_telegraph import (
    ElectromagneticTelegraphSimulation,
)

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except Exception:
    HAS_GENESIS = False


@pytest.fixture
def sim_config() -> PatentSimConfig:
    """Return a lightweight telegraph config for tests."""
    return PatentSimConfig(
        patent_id="US1647",
        headless=True,
        dt=0.01,
        substeps=5,
        resolution=(320, 240),
        device="cpu",
    )


def _bare_sim() -> ElectromagneticTelegraphSimulation:
    """A simulation instance without a Genesis scene (physics only)."""
    config = PatentSimConfig(patent_id="US1647")
    sim = create_simulation("US1647", config=config)
    assert isinstance(sim, ElectromagneticTelegraphSimulation)
    return sim


def _key_message(sim: ElectromagneticTelegraphSimulation, message: str) -> None:
    """Key a Morse message through the bare physics loop.

    Uses the sim's timing thresholds: dot 0.1 s, dash 0.3 s, intra-letter
    gap 0.1 s, letter gap 0.4 s, word gap 1.0 s.
    """
    table = {v: k for k, v in _morse_table().items()}
    for word in message.split(" "):
        for letter in word:
            for symbol in table[letter]:
                sim.set_parameter("key", 1.0)
                for _ in range(int(round((0.3 if symbol == "-" else 0.1) / 0.01))):
                    sim._integrate(0.01)
                sim.set_parameter("key", 0.0)
                for _ in range(int(round(0.1 / 0.01))):
                    sim._integrate(0.01)
            for _ in range(int(round(0.3 / 0.01))):
                sim._integrate(0.01)
        for _ in range(int(round(0.6 / 0.01))):
            sim._integrate(0.01)
    sim.flush_decoder()


def _morse_table() -> dict[str, str]:
    from cloud_robotics_sim.patents.sims.morse_telegraph import MORSE_TABLE

    return MORSE_TABLE


# ---------------------------------------------------------------------------
# Pure physics (no Genesis required)
# ---------------------------------------------------------------------------


def test_current_rises_with_line_time_constant() -> None:
    """Closing the key charges the RL circuit toward V / R."""
    sim = _bare_sim()
    sim.set_parameter("key", 1.0)
    sim.set_parameter("line_length_km", 0.0)
    i_ss = sim.steady_current()
    assert i_ss == pytest.approx(12.0 / sim.R_COIL)
    sim._integrate(0.005)
    assert 0.0 < sim._current < i_ss
    for _ in range(500):
        sim._integrate(0.01)
    assert sim._current == pytest.approx(i_ss, rel=1e-3)


def test_armature_pulls_in_and_holds_with_hysteresis() -> None:
    """Above the pull-in current the armature closes; it only releases
    below the much lower drop-out current.
    """
    sim = _bare_sim()
    assert sim.drop_out_current() < sim.pull_in_current()
    sim.set_parameter("key", 1.0)
    for _ in range(300):
        sim._integrate(0.01)
    assert sim._sounder_down
    assert sim._clicks >= 1
    assert sim._travel == pytest.approx(sim.ARMATURE_GAP)
    # Current between drop-out and pull-in keeps a closed armature down.
    sim._current = 0.5 * (sim.pull_in_current() + sim.drop_out_current())
    held = sim.magnetic_force(sim._current, sim.ARMATURE_GAP)
    resting = sim.magnetic_force(sim._current, 0.0)
    assert held > sim.SPRING_PRELOAD + sim.SPRING_RATE * sim.ARMATURE_GAP
    assert resting < sim.SPRING_PRELOAD


def test_long_line_prevents_pull_in() -> None:
    """A very long line attenuates the current below pull-in — the
    historical motivation for relay stations.
    """
    sim = _bare_sim()
    sim.set_parameter("key", 1.0)
    sim.set_parameter("line_length_km", 200.0)
    for _ in range(1000):
        sim._integrate(0.01)
    assert sim._current < sim.pull_in_current()
    assert not sim._sounder_down


def test_long_line_delays_pull_in() -> None:
    """Line inductance delays the current rise and the first click."""
    delays = {}
    for km in (0.0, 100.0):
        sim = _bare_sim()
        sim.set_parameter("key", 1.0)
        sim.set_parameter("line_length_km", km)
        sim.set_parameter("battery_voltage", 100.0)  # keep current ample
        t = 0.0
        while not sim._sounder_down and t < 20.0:
            sim._integrate(0.01)
            t += 0.01
        delays[km] = t
    assert delays[100.0] > delays[0.0] > 0.0


def test_key_open_releases_armature() -> None:
    """Opening the key drops the current and the armature releases."""
    sim = _bare_sim()
    sim.set_parameter("key", 1.0)
    for _ in range(300):
        sim._integrate(0.01)
    assert sim._sounder_down
    sim.set_parameter("key", 0.0)
    for _ in range(500):
        sim._integrate(0.01)
    assert not sim._sounder_down
    assert sim._travel == pytest.approx(0.0)


def test_morse_decodes_sos() -> None:
    """Keying ... --- ... decodes to SOS."""
    sim = _bare_sim()
    _key_message(sim, "SOS")
    assert sim.decoded_text == "SOS"


def test_morse_decodes_words_with_spaces() -> None:
    """Word gaps insert spaces into the decoded text."""
    sim = _bare_sim()
    _key_message(sim, "HI A")
    assert sim.decoded_text == "HI A"


def test_telegraph_parameter_validation() -> None:
    """Setting an unknown parameter raises KeyError."""
    sim = _bare_sim()
    with pytest.raises(KeyError):
        sim.set_parameter("unknown_param", 1.0)
    assert "key" in sim.list_parameters()
    assert "line_length_km" in sim.list_parameters()


# ---------------------------------------------------------------------------
# Genesis integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_telegraph_build_reset_step(sim_config: PatentSimConfig) -> None:
    """The telegraph can be built, reset, and stepped."""
    sim = create_simulation("US1647", config=sim_config)
    sim.build()
    initial_state = sim.reset()
    assert initial_state.time == 0.0
    assert "armature" in initial_state.bodies
    assert "coil" in initial_state.bodies
    assert initial_state.metrics["coil_current_a"] == pytest.approx(0.0)

    final_state = sim.step()
    assert final_state.time > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_telegraph_clicks_on_key_press(sim_config: PatentSimConfig) -> None:
    """Holding the key energizes the coil and clicks the sounder."""
    sim = create_simulation("US1647", config=sim_config)
    sim.build()
    sim.reset()
    sim.set_parameter("key", 1.0)
    for _ in range(30):  # 1.5 simulated seconds
        state = sim.step()
    assert state.metrics["coil_current_a"] > sim.pull_in_current()  # type: ignore[attr-defined]
    assert state.metrics["sounder_down"] == 1.0
    assert state.metrics["clicks"] >= 1.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_telegraph_render(sim_config: PatentSimConfig) -> None:
    """The simulation can render an RGB frame."""
    sim = create_simulation("US1647", config=sim_config)
    sim.build()
    sim.reset()
    frame = sim.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] == sim_config.resolution[1]
    assert frame.shape[1] == sim_config.resolution[0]
    sim.close()
