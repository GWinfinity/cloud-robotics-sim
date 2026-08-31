"""Minimal reproduction: LEAP hand holds a sphere with position control.

In Genesis 1.3.2 the sphere slowly slips through the fingertips and falls,
while the same MJCF pose is stable in MuJoCo.
"""
from __future__ import annotations

from pathlib import Path

import genesis as gs
import numpy as np


def main() -> None:
    gs.init(backend=gs.cpu)

    dt = 0.002
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -9.81),
            enable_collision=True,
            enable_mujoco_compatibility=True,
            friction_cone=gs.friction_cone.elliptic,
        ),
        show_viewer=False,
    )

    plugin_root = Path(__file__).parent.parent.parent
    hand_path = plugin_root / "assets" / "mujoco_menagerie" / "leap_hand" / "right_hand.xml"

    hand = scene.add_entity(
        gs.morphs.MJCF(
            file=str(hand_path),
            pos=(0.0, 0.0, 0.28),
            quat=(0.0, 1.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(friction=0.5),
    )

    sphere_radius = 0.05
    sphere_mass = 0.05
    sphere_volume = 4.0 / 3.0 * np.pi * sphere_radius**3
    sphere_density = sphere_mass / sphere_volume

    scene.add_entity(
        gs.morphs.Sphere(
            radius=sphere_radius,
            pos=(0.0, 0.0, 0.08),
            quat=(1.0, 0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(rho=sphere_density, friction=0.8),
    )

    scene.build()

    # Target grasp configuration used in the LIFT benchmark.
    hand_target = np.array(
        [
            0.5, 0.0, 1.2, 1.0,  # index
            0.5, 0.0, 1.2, 1.0,  # middle
            0.5, 0.0, 1.2, 1.0,  # ring
            0.5, 0.0, 0.8, 0.6,  # thumb
        ],
        dtype=float,
    )
    hand.set_qpos(hand_target)
    hand.set_dofs_velocity(np.zeros(hand.n_dofs, dtype=float))

    for step in range(50):
        # Get object pose from the second entity.
        obj = scene.entities[1]
        obj_pos = np.asarray(obj.get_pos())
        print(f"step={step} obj_z={obj_pos[2]:.4f}")
        hand.control_dofs_position(hand_target)
        scene.step()


if __name__ == "__main__":
    main()
