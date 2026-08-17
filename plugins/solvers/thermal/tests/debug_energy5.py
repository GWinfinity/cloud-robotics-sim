import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402, I001
import genesis as gs  # noqa: E402
from plugins.solvers.thermal import install, ThermalOptions  # noqa: E402

gs.init(backend=gs.cpu)


class Fixed:
    """Fixed-position body for thermal coupling tests."""

    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self._pos


scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
thermal = install(
    scene,
    ThermalOptions(
        resolution=(16, 16),
        dx=0.01,
        alpha=1e-8,
        boundary_mode="neumann",
        grid_rho=1.0,
        grid_cp=1.0,
        initial_temperature=0.0,
    ),
)
s1 = thermal.add_source(
    Fixed((0.06, 0.06, 0.0)), temperature=1.0, radius=0.04, rate=1e6, heat_capacity=0.01
)
s2 = thermal.add_source(
    Fixed((0.10, 0.10, 0.0)), temperature=0.0, radius=0.04, rate=1e6, heat_capacity=0.01
)
scene.build()

scene.step()
print("s1 temp", s1.temperature, "s2 temp", s2.temperature)
print("counts", thermal._source_cell_count.to_numpy())
print("sums", thermal._source_cell_sum.to_numpy())
print("T max", thermal.get_temperature().max())
