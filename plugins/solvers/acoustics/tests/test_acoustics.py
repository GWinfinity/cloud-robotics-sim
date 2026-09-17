"""Tests for the AcousticsSolver plugin."""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import genesis as gs

from plugins.solvers.acoustics import (
    AcousticProbe,
    AcousticsOptions,
    install,
)
from plugins.solvers.acoustics.core.acoustics_solver import (
    AcousticsSolver,
)

# Initialize Genesis once for the test module.
gs.init(backend=gs.cpu)

# Grid design shared by the physics tests: 0.5 m x 0.5 m domain, 2.5 mm cells.
DX = 0.0025
N = 200
DT = 4e-6  # c*dt/dx = 0.55 < 1/sqrt(2)
C = 343.0
STEPS = 400  # 1.6 ms of simulated time


def _make_scene(dt: float = DT, substeps: int = 1):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    return scene


def _gaussian_pulse(t: float) -> float:
    return float(np.exp(-(((t - 2e-5) / 8e-6) ** 2)))


def _peak_freq(freqs: np.ndarray, spl: np.ndarray) -> float:
    """Parabolic-interpolated peak frequency (sub-bin accuracy)."""
    i = int(np.argmax(spl))
    if 0 < i < spl.size - 1:
        y0, y1, y2 = spl[i - 1], spl[i], spl[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            return float(freqs[i] + offset * (freqs[1] - freqs[0]))
    return float(freqs[i])


def _install(scene, **overrides):
    opts = {
        "resolution": (N, N),
        "dx": DX,
        "c": C,
        "dt": DT,
    }
    opts.update(overrides)
    return install(scene, AcousticsOptions(**opts))


class TestAcousticsOptions:
    """Validation tests for AcousticsOptions."""

    def test_defaults(self):
        opts = AcousticsOptions()
        assert opts.c == pytest.approx(343.0)
        assert opts.boundary_mode == "absorbing"

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            AcousticsOptions(dim=4)

    def test_resolution_length_mismatch(self):
        with pytest.raises(ValueError):
            AcousticsOptions(dim=3, resolution=(16, 16))

    def test_invalid_boundary_mode(self):
        with pytest.raises(ValueError):
            AcousticsOptions(boundary_mode="periodic")

    def test_non_positive_c(self):
        with pytest.raises(ValueError):
            AcousticsOptions(c=0.0)

    def test_sponge_too_thick(self):
        with pytest.raises(ValueError):
            AcousticsOptions(resolution=(16, 16), sponge_layers=8)


class TestAcousticsSolver:
    """Integration tests for AcousticsSolver plugged into a gs.Scene."""

    def test_install_adds_solver_to_active_list(self):
        scene = _make_scene()
        solver = _install(scene)
        scene.build()
        assert isinstance(solver, AcousticsSolver)
        assert solver in scene.sim._active_solvers
        assert scene.sim.acoustics_solver is solver

    def test_get_set_pressure(self):
        scene = _make_scene()
        solver = _install(scene)
        scene.build()
        p = solver.get_pressure()
        assert p.shape == (N, N)
        assert np.allclose(p, 0.0)
        field = np.zeros((N, N))
        field[10, 10] = 5.0
        solver.set_pressure(field)
        assert solver.get_pressure()[10, 10] == pytest.approx(5.0)

    def test_cfl_violation_raises(self):
        scene = _make_scene()
        _install(scene, dt=2e-5)  # c*dt/dx = 2.7 >> 0.707
        with pytest.raises(ValueError, match="CFL|unstable"):
            scene.build()

    def test_propagation_delay(self):
        """A pulse reaches a probe at distance d after roughly t = d / c."""
        scene = _make_scene()
        solver = _install(scene)
        src_pos = (0.25, 0.25, 0.0)
        probe_pos = (0.40, 0.25, 0.0)  # 0.15 m away
        distance = 0.15
        solver.add_source(
            position=src_pos,
            signal=_gaussian_pulse,
            radius=0.005,
        )
        probe = solver.add_probe(probe_pos)
        scene.build()
        for _ in range(STEPS):
            scene.step()

        sig = solver.get_signal(probe)
        assert sig.size == STEPS
        # Peak arrival (group delay); discrete dispersion makes the numerical
        # wave a few percent slower than the continuum c.
        arrival = int(np.argmax(np.abs(sig))) * DT
        expected = distance / C
        assert arrival == pytest.approx(expected, rel=0.15)

    def test_neumann_boundary_reflects(self):
        """Rigid (Neumann) walls reflect: late-time energy exceeds the absorbing case."""
        energies = {}
        for mode in ("absorbing", "neumann"):
            scene = _make_scene()
            solver = _install(scene, boundary_mode=mode)
            solver.add_source(
                position=(0.25, 0.25, 0.0),
                signal=lambda t: 10.0 * np.sin(2 * np.pi * 2000 * t),
                radius=0.01,
            )
            probe = solver.add_probe((0.42, 0.25, 0.0))
            scene.build()
            for _ in range(STEPS):
                scene.step()
            sig = solver.get_signal(probe)
            late = sig[sig.size // 2 :]
            energies[mode] = float(np.mean(late**2))
        assert energies["neumann"] > 5.0 * energies["absorbing"]

    def test_sine_source_spl_and_spectrum(self):
        """A sinusoidal source peaks at its own frequency with the expected SPL."""
        amp = 2.0
        f0 = 2000.0
        scene = _make_scene()
        solver = _install(scene)
        solver.add_source(
            position=(0.2, 0.2, 0.0),
            signal=lambda t: amp * np.sin(2 * np.pi * f0 * t),
            radius=0.01,
        )
        probe = solver.add_probe((0.3, 0.2, 0.0))
        scene.build()
        for _ in range(STEPS):
            scene.step()

        freqs, spl = probe.spectrum(dt=DT)
        peak_freq = _peak_freq(freqs, spl)
        assert peak_freq == pytest.approx(f0, rel=0.02)
        # RMS of a sine with peak amp is amp / sqrt(2).
        expected_spl = 20 * np.log10(amp / np.sqrt(2) / 2e-5)
        measured_spl = probe.spl(dt=DT)
        # Near-field, small-grid errors are large; check the order of magnitude.
        assert measured_spl == pytest.approx(expected_spl, abs=25.0)

    def test_body_velocity_coupling(self):
        """A rigid body oscillating vertically radiates at the drive frequency."""
        f0 = 1500.0
        scene = _make_scene()
        box = scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.25, 0.25, 0.05))
        )
        solver = _install(scene)
        solver.add_body(box, radius=0.01, amplitude=50.0)
        probe = solver.add_probe((0.34, 0.25, 0.0))
        scene.build()

        t = 0.0
        for _ in range(STEPS):
            box.set_dofs_velocity(
                np.array([0.5 * np.sin(2 * np.pi * f0 * t)]), dofs_idx_local=[2]
            )
            scene.step()
            t += DT

        sig = solver.get_signal(probe)
        assert np.max(np.abs(sig)) > 1e-9
        freqs, spl = probe.spectrum(dt=DT)
        peak_freq = _peak_freq(freqs, spl)
        assert peak_freq == pytest.approx(f0, rel=0.05)

    def test_3d_propagation(self):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=1),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        solver = install(
            scene,
            AcousticsOptions(
                dim=3, resolution=(32, 32, 32), dx=0.01, c=C, dt=DT
            ),
        )
        solver.add_source(
            position=(0.16, 0.16, 0.16),
            signal=_gaussian_pulse,
            radius=0.005,
        )
        probe = solver.add_probe((0.24, 0.16, 0.16))  # 0.08 m away
        scene.build()
        for _ in range(160):
            scene.step()
        sig = solver.get_signal(probe)
        arrival = int(np.argmax(np.abs(sig))) * DT
        assert arrival == pytest.approx(0.08 / C, rel=0.2)

    def test_too_many_sources_raises(self):
        scene = _make_scene()
        solver = _install(scene, max_sources=2)
        solver.add_source(position=(0.1, 0.1, 0.0), signal=lambda t: 0.0)
        solver.add_source(position=(0.2, 0.2, 0.0), signal=lambda t: 0.0)
        with pytest.raises(RuntimeError):
            solver.add_source(position=(0.3, 0.3, 0.0), signal=lambda t: 0.0)

    def test_probe_is_dataclass_with_helpers(self):
        scene = _make_scene()
        solver = _install(scene)
        probe = solver.add_probe((0.1, 0.1, 0.0))
        scene.build()
        assert isinstance(probe, AcousticProbe)
        assert isinstance(solver._sources, list)
        sig = probe.signal()
        assert sig.size <= 1  # at most a build-time warmup sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
