"""Tests for the Fermi Neutronic Reactor simulation (US 2,708,656)."""

from __future__ import annotations

import math

import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.fermi_reactor import NeutronicReactorSimulation

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> NeutronicReactorSimulation:
    """Create an un-built simulation for pure-physics tests."""
    sim = NeutronicReactorSimulation(PatentSimConfig(headless=True, device="cpu"))
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _run(sim: NeutronicReactorSimulation, seconds: float) -> None:
    """Integrate the kinetics model without Genesis."""
    steps = int(seconds / sim.config.dt)
    for _ in range(steps):
        sim._integrate_kinetics(sim.config.dt)
        sim._time += sim.config.dt


class TestReactivity:
    """Control-rod worth and temperature feedback."""

    def test_beta_groups_sum_to_total(self) -> None:
        total = sum(NeutronicReactorSimulation.BETA_I)
        assert total == pytest.approx(NeutronicReactorSimulation.BETA_TOTAL, rel=2e-3)

    def test_rod_mapping(self) -> None:
        sim = _bare_sim()
        beta = sim.BETA_TOTAL
        sim.set_parameter("rod_position", 0.0)
        assert sim.reactivity() == pytest.approx(-3.0 * beta)
        sim.set_parameter("rod_position", 1.0)
        # Full withdrawal: +0.5 beta, supercritical but below prompt critical.
        assert sim.reactivity() == pytest.approx(0.5 * beta)
        assert sim.reactivity() < beta

    def test_negative_temperature_feedback(self) -> None:
        sim = _bare_sim(rod_position=1.0)
        cold = sim.reactivity(fuel_temp=sim.T_REFERENCE)
        hot = sim.reactivity(fuel_temp=sim.T_REFERENCE + 100.0)
        assert hot < cold
        assert (hot - cold) == pytest.approx(sim.ALPHA_T * 100.0)


class TestPointKinetics:
    """Power excursion, self-regulation and scram behavior."""

    def test_source_builds_subcritical_power(self) -> None:
        sim = _bare_sim(rod_position=0.3)
        _run(sim, 5.0)
        metrics = sim._compute_metrics()
        assert metrics["power_w"] > 0.0
        assert metrics["reactivity_pcm"] < 0.0
        assert metrics["precursor_total"] > 0.0

    def test_rod_withdrawal_power_rises_then_stabilizes(self) -> None:
        sim = _bare_sim(rod_position=1.0)
        _run(sim, 1.0)
        power_early = sim._compute_metrics()["power_w"]
        _run(sim, 19.0)
        metrics = sim._compute_metrics()
        power_peak = metrics["power_w"]
        # Exponential rise after withdrawal.
        assert power_peak > power_early * 2.0
        assert metrics["fuel_temp_k"] > sim.T_REFERENCE + 50.0
        # Temperature feedback arrests and reverses the excursion: the
        # pile self-regulates instead of diverging exponentially.
        _run(sim, 20.0)
        settled = sim._compute_metrics()
        assert settled["power_w"] < power_peak
        assert settled["reactivity_pcm"] < 0.0

    def test_positive_period_when_supercritical(self) -> None:
        sim = _bare_sim(rod_position=1.0)
        _run(sim, 1.0)
        metrics = sim._compute_metrics()
        assert metrics["reactivity_pcm"] > 0.0
        assert 0.0 < metrics["period_s"] < float("inf")

    def test_scram_drops_power_but_delayed_neutrons_remain(self) -> None:
        sim = _bare_sim(rod_position=1.0)
        _run(sim, 25.0)
        power_before = sim._compute_metrics()["power_w"]
        sim.set_parameter("rod_position", 0.0)
        _run(sim, 0.5)
        power_fast = sim._compute_metrics()["power_w"]
        # Prompt drop: power collapses but does not vanish instantly.
        assert 1e-3 * power_before < power_fast < 0.5 * power_before
        # Delayed-neutron tail keeps decaying on precursor timescales.
        _run(sim, 4.5)
        power_late = sim._compute_metrics()["power_w"]
        assert 0.0 < power_late < power_fast

    def test_more_cooling_lowers_fuel_temperature(self) -> None:
        low = _bare_sim(rod_position=1.0, cooling=0.5)
        high = _bare_sim(rod_position=1.0, cooling=2.0)
        _run(low, 30.0)
        _run(high, 30.0)
        t_low = low._compute_metrics()["fuel_temp_k"]
        t_high = high._compute_metrics()["fuel_temp_k"]
        assert t_high < t_low
        assert math.isfinite(t_low) and math.isfinite(t_high)


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = NeutronicReactorSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["power_w"] == 0.0
        for _ in range(5):
            state = sim.step()
        assert "period_s" in state.metrics
        assert state.metrics["power_w"] > 0.0  # start-up source builds power
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = NeutronicReactorSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "rod_position" in sim.list_parameters()
        sim.set_parameter("rod_position", 0.8)
        assert sim.get_parameter("rod_position") == pytest.approx(0.8)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
