r"""Benchmark JouleHeatingSolver forward-pass throughput.

Usage:
    cd D:\\githbi\\genesis-cloud-sim
    .venv/Scripts/python.exe scripts/benchmark_joule_solver.py

The solver now uses fused forward kernels for the Jacobi solve and for J/Q
computation, so the dominant cost is the kernel work itself rather than
Python dispatch per iteration.
"""

from __future__ import annotations

import time

import genesis as gs

from plugins.solvers.joule_heating import JouleHeatingOptions, install


def benchmark(
    resolution: tuple[int, int] = (64, 32),
    max_iter: int = 100,
    substeps: int = 10,
    n_steps: int = 10,
) -> float:
    """Return average seconds per simulation step."""
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=substeps),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    solver = install(
        scene,
        JouleHeatingOptions(
            resolution=resolution,
            dx=1.0,
            sigma=1.0,
            rho=1.0,
            cp=1.0,
            k=0.01,
            max_iter=max_iter,
            couple_to_thermal=False,
        ),
    )
    scene.build()

    solver.set_voltage_boundary("x_min", 10.0)
    solver.set_voltage_boundary("x_max", 0.0)

    # Warmup.
    scene.step()

    start = time.perf_counter()
    for _ in range(n_steps):
        scene.step()
    elapsed = time.perf_counter() - start
    return elapsed / n_steps


if __name__ == "__main__":
    for max_iter in (10, 50, 100, 200):
        t_per_step = benchmark(max_iter=max_iter)
        print(f"max_iter={max_iter:4d}: {t_per_step:.4f}s/step")
