"""Basic usage of the AcousticsSolver plugin.

A 200 Hz "whistle" monopole source radiates in a 2D room with rigid
(Neumann) walls; one microphone records the direct path and another sits
near a wall to catch reflections. The demo then re-runs with absorbing
(sponge) boundaries for comparison and saves waveforms + SPL spectra to
``outputs/acoustics_demo/``.

Run from the repo root:

    python plugins/solvers/acoustics/examples/basic_usage.py
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs
import numpy as np

from plugins.solvers.acoustics import AcousticsOptions, install

DX = 0.0025
DT = 4e-6
STEPS = 1200  # 4.8 ms
F0 = 200.0


def run(mode: str, out_dir: Path) -> dict[str, np.ndarray]:
    """Run the whistle demo with the given boundary mode; return signals."""
    gs.init(backend=gs.cpu) if not gs._initialized else None
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(
        scene,
        AcousticsOptions(
            resolution=(200, 200),
            dx=DX,
            c=343.0,
            dt=DT,
            boundary_mode=mode,
        ),
    )
    solver.add_source(
        position=(0.5, 0.5, 0.0),
        signal=lambda t: 5.0 * np.sin(2 * np.pi * F0 * t),
        radius=0.005,
    )
    near = solver.add_probe((0.7, 0.5, 0.0))  # direct path
    wall = solver.add_probe((0.95, 0.5, 0.0))  # next to the right wall
    scene.build()
    for _ in range(STEPS):
        scene.step()

    signals = {
        "near": solver.get_signal(near),
        "wall": solver.get_signal(wall),
    }
    t = np.arange(signals["near"].size) * DT
    for name, sig in signals.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_dir / f"{mode}_{name}.csv",
            np.column_stack([t, sig]),
            delimiter=",",
            header="t_s,pressure_pa",
            comments="",
        )
        freqs, spl = near.spectrum(dt=DT) if name == "near" else wall.spectrum(dt=DT)
        np.savetxt(
            out_dir / f"{mode}_{name}_spectrum.csv",
            np.column_stack([freqs, spl]),
            delimiter=",",
            header="freq_hz,spl_db",
            comments="",
        )
    print(f"[{mode}] near SPL = {near.spl(dt=DT):.1f} dB, wall SPL = {wall.spl(dt=DT):.1f} dB")
    return signals


def main() -> int:
    """Run the whistle demo with rigid and absorbing boundaries."""
    out_dir = ROOT / "outputs" / "acoustics_demo"
    rigid = run("neumann", out_dir)
    open_out = run("absorbing", out_dir)
    print(f"saved waveforms and spectra to {out_dir}")
    print(
        "rigid late energy near:  {:.3e} | absorbing: {:.3e}".format(
            float(np.mean(rigid["near"][-400:] ** 2)),
            float(np.mean(open_out["near"][-400:] ** 2)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
