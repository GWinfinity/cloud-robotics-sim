"""Genesis scene construction for the narrow-slot insertion demo."""

from __future__ import annotations

import genesis as gs
import numpy as np

from .sim_robot import FrankaSim


class InsertionScene:
    """A simple scene: floor + Franka + two walls forming a narrow slot."""

    def __init__(
        self,
        headless: bool = True,
        dt: float = 0.005,
        substeps: int = 4,
    ) -> None:
        self.headless = headless
        self.dt = dt
        self.substeps = substeps
        self.scene: gs.Scene | None = None
        self.robot: FrankaSim | None = None
        self.peg: gs.engine.entities.RigidEntity | None = None
        self.hole_center = np.array([0.40, 0.0, 0.85])
        self.hole_quat = np.array([0.0, 1.0, 0.0, 0.0])  # z down

    def build(self) -> "InsertionScene":
        gs.init(backend=gs.cpu, precision="32")
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.substeps,
            ),
        )
        # Ground plane.
        self.scene.add_entity(gs.morphs.Plane())

        # Robot.
        self.robot = FrankaSim(self.scene, base_pos=(0.0, 0.0, 0.0))

        # Slot walls: two boxes with a 12 mm gap in between.
        gap = 0.012
        wall_half_thickness = 0.020
        wall_y = gap / 2.0 + wall_half_thickness
        wall_size = (0.10, wall_half_thickness * 2.0, 0.05)
        wall_z = 0.825
        self.scene.add_entity(
            gs.morphs.Box(size=wall_size, pos=(0.40, wall_y, wall_z))
        )
        self.scene.add_entity(
            gs.morphs.Box(size=wall_size, pos=(0.40, -wall_y, wall_z))
        )

        # Kinematic tool (cylinder).
        self.peg = self.scene.add_entity(
            gs.morphs.Cylinder(radius=0.005, height=0.08, pos=(0.40, 0.0, 0.92))
        )

        self.scene.build()
        self.robot.build()
        return self

    def get_hole_pose(self) -> np.ndarray:
        from .math_utils import pos_quat_to_matrix

        return pos_quat_to_matrix(self.hole_center, self.hole_quat)

    def destroy(self) -> None:
        if self.scene is not None:
            gs.destroy()
            self.scene = None
