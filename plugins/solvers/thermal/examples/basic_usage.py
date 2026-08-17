"""Basic usage example for the ThermalSolver plugin with rigid-body coupling."""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import genesis as gs

from plugins.solvers.thermal import install, ThermalOptions


def main() -> None:
    """Run a 2D heat-diffusion example with a hot rigid box as a heat source."""
    gs.init(backend=gs.cpu)

    dt = 0.01
    resolution = (64, 64)
    dx = 0.01
    alpha = 1e-4  # diffusivity: alpha*dt/dx^2 = 0.1 < 0.25

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # Hot rigid box placed near the center of the grid.
    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.32, 0.32, 0.02)),
        material=gs.materials.Rigid(),
    )

    # Cold initial field with Dirichlet boundaries.
    init = np.zeros(resolution, dtype=float)

    thermal = install(
        scene,
        ThermalOptions(
            resolution=resolution,
            dx=dx,
            alpha=alpha,
            boundary_mode="dirichlet",
            boundary_value=0.0,
            initial_temperature=init,
        ),
    )

    # Couple the rigid box to the thermal field.
    thermal.add_source(
        entity=box,
        temperature=1.0,
        radius=0.05,
        rate=10.0,
        heat_capacity=0.1,
    )

    scene.build()

    print("Initial max temperature:", float(thermal.get_temperature().max()))
    for step in range(200):
        scene.step()
        if step % 50 == 49:
            temperature = thermal.get_temperature()
            print(
                f"Step {step + 1:3d}: max T = {float(temperature.max()):.4f}, "
                f"sum T = {float(temperature.sum()):.4f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
