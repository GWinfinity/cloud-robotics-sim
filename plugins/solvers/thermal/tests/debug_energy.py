import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402, I001
from plugins.solvers.thermal import install, ThermalOptions  # noqa: E402

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, 0.0)),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
box1 = scene.add_entity(
    gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.06, 0.06, 0.01)),
    material=gs.materials.Rigid(),
)
box2 = scene.add_entity(
    gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.10, 0.10, 0.01)),
    material=gs.materials.Rigid(),
)
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
    box1, temperature=1.0, radius=0.04, rate=1e6, heat_capacity=0.01
)
s2 = thermal.add_source(
    box2, temperature=0.0, radius=0.04, rate=1e6, heat_capacity=0.01
)
scene.build()

cell_capacity = 1.0 * 1.0 * (0.01**2)


def total_energy():
    """Return the total thermal energy of coupled bodies and grid."""
    grid = float(cell_capacity * thermal.get_temperature().sum())
    return s1.heat_capacity * s1.temperature + s2.heat_capacity * s2.temperature + grid


E0 = total_energy()
print("E0", E0)
for i in range(20):
    scene.step()
    E = total_energy()
    print(i + 1, E, (E - E0) / E0)
