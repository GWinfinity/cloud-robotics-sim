"""Tests for the Spencer microwave oven simulation (US 2,495,429)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.spencer_microwave import (
    MethodOfTreatingFoodstuffsSimulation,
)

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(**params: float) -> MethodOfTreatingFoodstuffsSimulation:
    """Create an un-built simulation for pure-physics tests."""
    # Overrides go through the config so the initial state (e.g. water
    # content, derived from food_mass_g) matches the requested parameters.
    return MethodOfTreatingFoodstuffsSimulation(
        PatentSimConfig(headless=True, device="cpu", parameters=dict(params))
    )


def _cook(sim: MethodOfTreatingFoodstuffsSimulation, seconds: float) -> None:
    """Integrate the heating model without Genesis."""
    steps = int(seconds / sim.config.dt)
    for _ in range(steps):
        sim._integrate_heating(sim.config.dt)
        sim._time += sim.config.dt


class TestStandingWave:
    """Absorption pattern of the cavity standing wave."""

    def test_antinode_absorbs_twice_the_corner_nodes(self) -> None:
        weights = MethodOfTreatingFoodstuffsSimulation.standing_wave_weights()
        assert weights[4] == pytest.approx(3.0)  # central antinode
        assert weights[0] == pytest.approx(1.5)  # corner node
        assert weights[4] / weights[0] == pytest.approx(2.0)

    def test_absorbed_powers_sum_to_efficiency_times_input(self) -> None:
        sim = _bare_sim(power_w=1000.0, turntable_on=0.0)
        total = float(np.sum(sim.absorbed_powers()))
        assert total == pytest.approx(sim.ETA * 1000.0)

    def test_turntable_equalizes_effective_weights(self) -> None:
        rotating = _bare_sim(turntable_on=1.0)
        fixed = _bare_sim(turntable_on=0.0)
        w_rot = rotating.effective_weights()
        w_fix = fixed.effective_weights()
        assert float(np.max(w_rot) - np.min(w_rot)) == pytest.approx(0.0)
        assert float(np.max(w_fix) - np.min(w_fix)) > 1.0


class TestHeating:
    """Basic heating behavior of the lumped cell model."""

    def test_zero_power_keeps_food_at_ambient(self) -> None:
        sim = _bare_sim(power_w=0.0, turntable_on=0.0)
        _cook(sim, 60.0)
        assert float(np.max(sim._temperatures)) == pytest.approx(sim.T_AMBIENT_C)
        assert sim._energy_j == pytest.approx(0.0)

    def test_higher_power_heats_faster(self) -> None:
        low = _bare_sim(power_w=400.0)
        high = _bare_sim(power_w=1200.0)
        _cook(low, 30.0)
        _cook(high, 30.0)
        assert float(np.mean(high._temperatures)) > float(np.mean(low._temperatures))

    def test_energy_accumulates_with_absorbed_power(self) -> None:
        sim = _bare_sim(power_w=1000.0, turntable_on=1.0)
        _cook(sim, 10.0)
        assert sim._energy_j == pytest.approx(sim.ETA * 1000.0 * 10.0, rel=1e-6)


class TestBoilingPlateau:
    """Evaporation pins wet cells at 100 C until they dry out."""

    def test_wet_cells_plateau_at_boiling_point(self) -> None:
        sim = _bare_sim(power_w=1200.0, food_mass_g=100.0, turntable_on=0.0)
        _cook(sim, 60.0)
        # The hottest (center) cell reached the plateau and stays pinned
        # while it still holds water.
        assert float(np.max(sim._temperatures)) == pytest.approx(100.0, abs=0.5)
        initial_water = sim._initial_water_per_cell_g() * 9.0
        assert 0.0 < float(np.sum(sim._water_g)) < initial_water

    def test_dry_cells_heat_past_100c(self) -> None:
        sim = _bare_sim(power_w=1200.0, food_mass_g=100.0, turntable_on=0.0)
        _cook(sim, 60.0)
        water_after_plateau = float(np.sum(sim._water_g))
        _cook(sim, 200.0)
        # The center cell has boiled dry and its temperature climbs on.
        assert sim._water_g[4] == pytest.approx(0.0)
        assert float(np.max(sim._temperatures)) > 110.0
        assert float(np.sum(sim._water_g)) < water_after_plateau


class TestTurntable:
    """Turntable rotation equalizes the absorbed power over time."""

    def test_fixed_food_develops_hot_and_cold_spots(self) -> None:
        sim = _bare_sim(power_w=800.0, food_mass_g=300.0, turntable_on=0.0)
        _cook(sim, 120.0)
        spread = float(np.max(sim._temperatures) - np.min(sim._temperatures))
        assert spread > 15.0

    def test_turntable_improves_uniformity(self) -> None:
        fixed = _bare_sim(power_w=800.0, food_mass_g=300.0, turntable_on=0.0)
        rotating = _bare_sim(power_w=800.0, food_mass_g=300.0, turntable_on=1.0)
        _cook(fixed, 120.0)
        _cook(rotating, 120.0)
        spread_fixed = float(np.max(fixed._temperatures) - np.min(fixed._temperatures))
        spread_rot = float(
            np.max(rotating._temperatures) - np.min(rotating._temperatures)
        )
        assert spread_rot < spread_fixed * 0.2
        m_fixed = fixed._compute_metrics()
        m_rot = rotating._compute_metrics()
        assert m_rot["uniformity"] > m_fixed["uniformity"]

    def test_turntable_angle_advances_only_when_on(self) -> None:
        rotating = _bare_sim(turntable_on=1.0)
        fixed = _bare_sim(turntable_on=0.0)
        _cook(rotating, 5.0)
        _cook(fixed, 5.0)
        assert rotating._turntable_angle > 0.0
        assert fixed._turntable_angle == pytest.approx(0.0)


class TestParameters:
    """Interactive parameter plumbing."""

    def test_defaults_and_overrides(self) -> None:
        sim = _bare_sim()
        assert sim.list_parameters() == ["power_w", "turntable_on", "food_mass_g"]
        assert sim.get_parameter("power_w") == pytest.approx(800.0)
        sim.set_parameter("power_w", 500.0)
        assert sim.get_parameter("power_w") == pytest.approx(500.0)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)

    def test_metrics_keys(self) -> None:
        sim = _bare_sim()
        metrics = sim._compute_metrics()
        for key in (
            "t_mean_c",
            "t_max_c",
            "t_min_c",
            "uniformity",
            "water_remaining_g",
            "energy_kj",
        ):
            assert key in metrics


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = MethodOfTreatingFoodstuffsSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["t_mean_c"] == pytest.approx(sim.T_AMBIENT_C)
        for _ in range(5):
            state = sim.step()
        assert state.metrics["t_max_c"] >= sim.T_AMBIENT_C
        assert "water_remaining_g" in state.metrics
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = MethodOfTreatingFoodstuffsSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "power_w" in sim.list_parameters()
        sim.set_parameter("power_w", 500.0)
        assert sim.get_parameter("power_w") == pytest.approx(500.0)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
