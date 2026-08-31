"""Integration tests for stub patent simulations."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation, list_patents

try:
    import genesis as gs  # noqa: F401

    HAS_GENESIS = True
except Exception:
    HAS_GENESIS = False


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
@pytest.mark.parametrize(
    "patent_id", [pid for pid in list_patents() if pid != "US821393"]
)
def test_stub_simulation_builds_and_steps(patent_id: str) -> None:
    """Every stub simulation can be built, reset, and stepped briefly."""
    config = PatentSimConfig(
        patent_id=patent_id,
        headless=True,
        dt=0.01,
        substeps=2,
        resolution=(160, 120),
        device="cpu",
    )
    sim = create_simulation(patent_id, config=config)
    sim.build()
    state = sim.reset()
    assert state.time == 0.0
    state = sim.step()
    assert state.time > 0.0
    sim.close()
