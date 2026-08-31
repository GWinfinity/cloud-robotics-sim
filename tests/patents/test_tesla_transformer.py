"""Tests for the Tesla Electrical Transformer simulation (US 593,138)."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.tesla_transformer import (
    ElectricalTransformerSimulation,
)

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> ElectricalTransformerSimulation:
    """Create an un-built simulation for pure-physics tests."""
    sim = ElectricalTransformerSimulation(PatentSimConfig(headless=True, device="cpu"))
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _run(sim: ElectricalTransformerSimulation, seconds: float) -> None:
    """Integrate the circuit model without Genesis."""
    steps = int(seconds / sim.config.dt)
    for _ in range(steps):
        sim._integrate_circuits(sim.config.dt)
        sim._time += sim.config.dt


class TestCircuitConstants:
    """Resonance, coupling and ideal-gain bookkeeping."""

    def test_resonance_frequency(self) -> None:
        sim = _bare_sim()
        # L1 = 30 mH, C1 = 211 nF -> ~2 kHz.
        assert sim.resonance_hz() == pytest.approx(2000.0, rel=0.05)

    def test_mutual_inductance_scales_with_coupling(self) -> None:
        low = _bare_sim(coupling=0.1)
        high = _bare_sim(coupling=0.4)
        assert high.mutual_inductance() == pytest.approx(
            low.mutual_inductance() * 4.0, rel=1e-6
        )

    def test_ideal_gain(self) -> None:
        sim = _bare_sim(detune=1.0)
        # sqrt(C1 / C2) = sqrt(211e-9 / 5.28e-9) ~= 6.3.
        assert sim.ideal_gain() == pytest.approx(6.32, rel=0.02)


class TestSparkGap:
    """Spark-gap firing and recharge behavior."""

    def test_gap_fires_and_recharges(self) -> None:
        sim = _bare_sim(supply_voltage=8000.0, gap_threshold=8000.0)
        _run(sim, 0.05)
        assert sim._gap_firings > 0

    def test_gap_never_fires_below_threshold(self) -> None:
        sim = _bare_sim(supply_voltage=2000.0, gap_threshold=8000.0)
        _run(sim, 0.1)
        assert sim._gap_firings == 0
        assert sim._secondary_peak == 0.0

    def test_higher_supply_fires_more_often(self) -> None:
        weak = _bare_sim(supply_voltage=8500.0, gap_threshold=8000.0)
        strong = _bare_sim(supply_voltage=20000.0, gap_threshold=8000.0)
        _run(weak, 0.05)
        _run(strong, 0.05)
        assert strong._gap_firings > weak._gap_firings


class TestResonantGain:
    """High-potential transformation and tuning sensitivity."""

    def test_secondary_voltage_exceeds_primary(self) -> None:
        sim = _bare_sim(supply_voltage=8000.0, gap_threshold=8000.0, detune=1.0)
        _run(sim, 0.05)
        # High-potential transformation: secondary peak >> primary 8 kV.
        assert sim._secondary_peak > 2.0 * 8000.0

    def test_detuning_collapses_the_gain(self) -> None:
        tuned = _bare_sim(detune=1.0)
        detuned = _bare_sim(detune=1.5)
        _run(tuned, 0.05)
        _run(detuned, 0.05)
        assert tuned._secondary_peak > detuned._secondary_peak * 1.5

    def test_higher_supply_raises_secondary_peak(self) -> None:
        weak = _bare_sim(supply_voltage=6000.0, gap_threshold=6000.0)
        strong = _bare_sim(supply_voltage=12000.0, gap_threshold=12000.0)
        _run(weak, 0.05)
        _run(strong, 0.05)
        assert strong._secondary_peak > weak._secondary_peak

    def test_single_firing_respects_energy_bound(self) -> None:
        # Keep the supply far below breakdown so the manually triggered
        # firing stays the only one.
        sim = _bare_sim(detune=1.0, supply_voltage=1000.0)
        sim._v1 = 8000.0
        sim._gap_conducting = True
        _run(sim, 0.02)
        # 0.5 C2 V2^2 cannot exceed the energy stored at the firing
        # instant, 0.5 C1 V_gap^2.
        ratio = (sim._secondary_peak / 8000.0) ** 2
        assert ratio <= sim.C_PRIMARY / sim.C_SECONDARY * 1.01


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = ElectricalTransformerSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["gap_firings"] == 0.0
        for _ in range(5):
            state = sim.step()
        assert "secondary_peak_v" in state.metrics
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = ElectricalTransformerSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "coupling" in sim.list_parameters()
        sim.set_parameter("coupling", 0.3)
        assert sim.get_parameter("coupling") == pytest.approx(0.3)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
