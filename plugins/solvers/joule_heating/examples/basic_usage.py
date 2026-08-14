"""Basic JouleHeatingSolver usage example."""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import genesis as gs
from plugins.solvers.joule_heating import JouleHeatingOptions, install


def main() -> None:
    """Run a simple Joule-heating simulation and print diagnostics."""
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    solver = install(
        scene,
        JouleHeatingOptions(
            resolution=(32, 8),
            dx=1.0,
            sigma=1.0,
            rho=1.0,
            cp=1.0,
            k=0.01,
            initial_temperature=300.0,
            max_iter=1000,
            tol=1e-7,
            couple_to_thermal=False,
        ),
    )

    scene.build()

    # Apply 10 V on the left face and ground the right face.
    solver.set_voltage_boundary("x_min", 10.0)
    solver.set_voltage_boundary("x_max", 0.0)

    # Set a small insulating inclusion in the center.
    sigma = np.ones((32, 8), dtype=float)
    sigma[14:18, 3:5] = 1e-3
    solver.set_conductivity(sigma)

    for _ in range(50):
        scene.step()

    voltage = solver.get_voltage()
    source = solver.get_heat_source()
    temperature = solver.get_temperature()

    print("Voltage range:", voltage.min(), voltage.max())
    print("Heat-source max:", source.max())
    print("Temperature mean / max:", temperature.mean(), temperature.max())


if __name__ == "__main__":
    main()
