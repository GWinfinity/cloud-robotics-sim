"""Minimal reproduction: Quadrants autodiff fails on thermal source coupling.

This script demonstrates the P0 blocker for making the thermal solver's
source-coupling fully differentiable. The diffusion step itself is
differentiable, but the energy-conserving relaxation between coupled bodies
and the grid uses patterns that Quadrants reverse-mode AD does not yet
support (atomic_add / in-place field writes / break in loops).

Run with the project venv:
    uv run python scripts/repro_thermal_source_autodiff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402, I001
import genesis as gs  # noqa: E402

from plugins.solvers.thermal import install, ThermalOptions  # noqa: E402

gs.init(backend=gs.cpu)


class _FixedSource:
    """Minimal thermal source with a fixed world position."""

    def __init__(self, pos: tuple[float, float, float]) -> None:
        self._pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self._pos


def main() -> int:
    """Run the diffusion/source-coupling autodiff reproduction."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    thermal = install(
        scene,
        ThermalOptions(
            resolution=(8, 8),
            dx=0.01,
            alpha=1e-4,
            boundary_mode="dirichlet",
            boundary_value=0.0,
            initial_temperature=0.0,
        ),
    )
    thermal.add_source(
        entity=_FixedSource((0.04, 0.04, 0.0)),
        temperature=1.0,
        radius=0.03,
        rate=10.0,
        heat_capacity=0.1,
    )
    scene.build()

    scene.step()

    # Diffusion backward works.
    t_field = thermal._T
    assert t_field is not None and t_field.grad is not None
    thermal.reset_grad()
    t_grad = t_field.grad.to_numpy()
    t_grad[1] = 1.0
    t_field.grad.from_numpy(t_grad)
    thermal._step_transient_2d_interior.grad(0)
    thermal._step_transient_2d_boundary_dirichlet.grad(0)
    print(
        "Diffusion grad OK: T[0] grad max =",
        np.abs(t_field.grad.to_numpy()[0]).max(),
    )

    # Source-coupling backward fails in Quadrants.
    t_field = thermal._T
    assert t_field is not None and t_field.grad is not None
    thermal.reset_grad()
    t_grad = t_field.grad.to_numpy()
    t_grad[1] = 1.0
    t_field.grad.from_numpy(t_grad)
    try:
        thermal._compute_source_equilibrium_2d.grad(0)
        print("Source equilibrium grad OK")
        return 0
    except RuntimeError as exc:
        print("Source equilibrium grad FAILED:")
        print(" ", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
