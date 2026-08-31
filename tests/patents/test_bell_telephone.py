"""Tests for the Bell Telephone simulation (US 174,465)."""

from __future__ import annotations

import math

import pytest

from cloud_robotics_sim.patents.base import PatentSimConfig
from cloud_robotics_sim.patents.sims.bell_telephone import (
    ImprovementInTelegraphyTelephoneSimulation,
)

try:
    import genesis  # noqa: F401

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False


def _bare_sim(
    **params: float,
) -> ImprovementInTelegraphyTelephoneSimulation:
    """Create an un-built simulation for pure-physics tests."""
    sim = ImprovementInTelegraphyTelephoneSimulation(
        PatentSimConfig(headless=True, device="cpu")
    )
    for key, value in params.items():
        sim.set_parameter(key, value)
    return sim


def _run(sim: ImprovementInTelegraphyTelephoneSimulation, seconds: float) -> None:
    """Integrate the electro-mechanical model without Genesis."""
    steps = int(seconds / sim.config.dt)
    for _ in range(steps):
        sim._integrate_circuit(sim.config.dt)
        sim._time += sim.config.dt


class TestInstrumentConstants:
    """Diaphragm tuning and magnetic coupling."""

    def test_diaphragm_resonance_tuning(self) -> None:
        sim = _bare_sim()
        omega = math.sqrt(sim.stiffness / sim.DIAPHRAGM_MASS)
        assert omega / (2.0 * math.pi) == pytest.approx(sim.RESONANCE_HZ)
        quality = math.sqrt(sim.stiffness * sim.DIAPHRAGM_MASS) / sim.damping
        assert quality == pytest.approx(sim.QUALITY_FACTOR)

    def test_coupling_grows_as_gap_closes(self) -> None:
        sim = _bare_sim()
        rest = sim.coupling(0.0)
        closed = sim.coupling(0.5 * sim.GAP_ZERO)
        # Inverse-square gap law: half the gap gives four times dPhi/dg.
        assert closed == pytest.approx(4.0 * rest)

    def test_line_resistance_scales_with_length(self) -> None:
        short = _bare_sim(line_length_km=0.0)
        long = _bare_sim(line_length_km=50.0)
        assert short.line_resistance() == pytest.approx(2.0 * short.R_COIL)
        assert long.line_resistance() == pytest.approx(8.0 * 50.0 + 2.0 * long.R_COIL)


class TestTransmission:
    """Voice excitation, resonance peak and line attenuation."""

    def test_no_excitation_means_no_line_current(self) -> None:
        sim = _bare_sim(voice_pressure_pa=0.0)
        _run(sim, 0.05)
        metrics = sim._compute_metrics()
        assert metrics["line_current_ma"] == 0.0
        assert metrics["tx_disp_um"] == 0.0
        assert metrics["rx_disp_um"] == 0.0

    def test_resonance_peak_at_diaphragm_tuning(self) -> None:
        resonant = _bare_sim(voice_freq_hz=1000.0, line_length_km=0.0)
        low = _bare_sim(voice_freq_hz=200.0, line_length_km=0.0)
        high = _bare_sim(voice_freq_hz=3000.0, line_length_km=0.0)
        _run(resonant, 0.1)
        _run(low, 0.1)
        _run(high, 0.1)
        m_res = resonant._compute_metrics()
        assert m_res["tx_disp_um"] > 5.0 * low._compute_metrics()["tx_disp_um"]
        assert m_res["tx_disp_um"] > 5.0 * high._compute_metrics()["tx_disp_um"]
        assert m_res["tx_emf_rms_v"] > 0.0

    def test_longer_line_attenuates_receiver(self) -> None:
        short = _bare_sim(line_length_km=0.0)
        long = _bare_sim(line_length_km=80.0)
        _run(short, 0.1)
        _run(long, 0.1)
        m_short = short._compute_metrics()
        m_long = long._compute_metrics()
        assert m_long["line_current_ma"] < 0.3 * m_short["line_current_ma"]
        assert m_long["rx_disp_um"] < 0.3 * m_short["rx_disp_um"]
        assert m_long["attenuation"] < m_short["attenuation"]

    def test_receiver_follows_transmitter(self) -> None:
        sim = _bare_sim(line_length_km=0.0)
        _run(sim, 0.1)
        metrics = sim._compute_metrics()
        assert metrics["tx_disp_um"] > 0.0
        assert metrics["rx_disp_um"] > 0.0
        assert 0.0 < metrics["attenuation"] < 1.0
        assert math.isfinite(metrics["line_current_ma"])


@pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world is not installed")
class TestGenesisIntegration:
    """Scene build / step smoke tests through Genesis."""

    def test_build_reset_step_render(self) -> None:
        sim = ImprovementInTelegraphyTelephoneSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        state = sim.reset()
        assert state.metrics["line_current_ma"] == 0.0
        for _ in range(5):
            state = sim.step()
        assert state.metrics["tx_disp_um"] > 0.0
        assert "attenuation" in state.metrics
        frame = sim.render()
        assert frame is not None
        sim.close()

    def test_parameters(self) -> None:
        sim = ImprovementInTelegraphyTelephoneSimulation(
            PatentSimConfig(headless=True, device="cpu", resolution=(64, 64))
        )
        sim.build()
        assert "voice_freq_hz" in sim.list_parameters()
        sim.set_parameter("voice_freq_hz", 440.0)
        assert sim.get_parameter("voice_freq_hz") == pytest.approx(440.0)
        with pytest.raises(KeyError):
            sim.set_parameter("nope", 1.0)
        sim.close()
