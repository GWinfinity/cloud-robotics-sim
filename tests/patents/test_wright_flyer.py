"""Tests for the Wright Flyer pilot simulation."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except Exception:
    HAS_GENESIS = False


@pytest.fixture
def sim_config() -> PatentSimConfig:
    """Return a lightweight Wright Flyer config for tests."""
    return PatentSimConfig(
        patent_id="US821393",
        headless=True,
        dt=0.01,
        substeps=5,
        resolution=(320, 240),
        device="cpu",
    )


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_wright_flyer_build_reset_step(sim_config: PatentSimConfig) -> None:
    """The Wright Flyer can be built, reset, and stepped."""
    sim = create_simulation("US821393", config=sim_config)
    sim.build()
    initial_state = sim.reset()
    assert initial_state.time == 0.0
    assert "aircraft" in initial_state.bodies

    final_state = sim.step()
    assert final_state.time > 0.0
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_wright_flyer_parameters(sim_config: PatentSimConfig) -> None:
    """Interactive parameters can be read and written."""
    sim = create_simulation("US821393", config=sim_config)
    params = sim.list_parameters()
    assert "wing_warp" in params
    assert "rudder" in params
    assert "elevator" in params
    assert "thrust" in params
    assert "wind_speed" in params

    sim.set_parameter("thrust", 0.8)
    assert sim.get_parameter("thrust") == pytest.approx(0.8)
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_wright_flyer_metrics(sim_config: PatentSimConfig) -> None:
    """Metrics include flight quantities after stepping."""
    sim = create_simulation("US821393", config=sim_config)
    sim.build()
    sim.reset()
    state = sim.step()
    assert "airspeed" in state.metrics
    assert "altitude" in state.metrics
    assert "roll_deg" in state.metrics
    assert "pitch_deg" in state.metrics
    assert "yaw_deg" in state.metrics
    sim.close()


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
def test_wright_flyer_render(sim_config: PatentSimConfig) -> None:
    """The simulation can render an RGB frame."""
    sim = create_simulation("US821393", config=sim_config)
    sim.build()
    sim.reset()
    frame = sim.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] == sim_config.resolution[1]
    assert frame.shape[1] == sim_config.resolution[0]
    sim.close()


def test_wright_flyer_parameter_validation() -> None:
    """Setting an unknown parameter raises KeyError."""
    config = PatentSimConfig(patent_id="US821393")
    sim = create_simulation("US821393", config=config)
    with pytest.raises(KeyError):
        sim.set_parameter("unknown_param", 1.0)


def test_wright_flyer_claim18_coupling() -> None:
    """Claim 18: the rudder is chained to the wing warp when coupled."""
    config = PatentSimConfig(patent_id="US821393")
    sim = create_simulation("US821393", config=config)

    # Default: linkage engaged. Full warp commands 0.27 rudder (the
    # historical 0.45 deg/deg ratio over the +/-15 deg warp and +/-25 deg
    # rudder ranges), regardless of the independent rudder input.
    sim.set_parameter("wing_warp", 1.0)
    assert sim._effective_rudder(1.0) == pytest.approx(0.27)
    sim.set_parameter("rudder", -0.9)
    assert sim._effective_rudder(1.0) == pytest.approx(0.27)
    assert sim._effective_rudder(-1.0) == pytest.approx(-0.27)

    # Linkage disengaged: the pilot's rudder input passes through.
    sim.set_parameter("coupled", 0.0)
    assert sim._effective_rudder(1.0) == pytest.approx(-0.9)
