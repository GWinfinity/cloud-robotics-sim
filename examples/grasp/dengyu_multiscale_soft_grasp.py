"""Deng-Yu inspired multiscale soft-body grasping example.

This example demonstrates how to combine a Franka Panda robot with a
soft deformable object inside the ``cloud_robotics_sim`` backend abstraction.
A multiscale error indicator allocates fine FEM resolution near the expected
gripper contact zone and coarse resolution farther away, echoing Deng Yu's
"local expensive computation, global compression" principle.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import genesis as gs
import numpy as np

from cloud_robotics_sim import (
    ComposerConfig,
    EmbodimentConfig,
    EnvironmentComposer,
    FrankaPanda,
    ObjectLibrary,
    PickPlaceTask,
    SceneConfig,
    TaskConfig,
)
from cloud_robotics_sim.backend.types import DeformableConfig, DeformableMaterialType
from cloud_robotics_sim.core.scene import ObjectSpawn, Scene
from cloud_robotics_sim.utils.multiscale_indicator import (
    IndicatorConfig,
    MultiscaleErrorIndicator,
)


class SoftGraspScene(Scene):
    """A simple table-top scene with a multiscale soft cube."""

    def __init__(
        self,
        gripper_center: tuple[float, float, float] | None = None,
        cube_size: float = 0.08,
    ) -> None:
        config = SceneConfig(
            name="dengyu_soft_grasp",
            size=(2.0, 2.0, 2.0),
            default_camera_pos=(1.0, 1.0, 1.0),
            default_camera_lookat=(0.45, 0.0, 0.1),
        )
        super().__init__(config)
        self.cube_size = cube_size
        cube_center = (0.45, 0.0, cube_size / 2)
        self.gripper_center = gripper_center or cube_center

    def _build_custom(self) -> None:
        """Add table and multiscale soft cube."""
        # Static table top.
        self.add_object(
            ObjectSpawn(
                name="table",
                shape_type="box",
                size=(1.0, 0.6, 0.02),
                position=(0.45, 0.0, 0.0),
                static=True,
                color=(0.7, 0.5, 0.3, 1.0),
                tags=["furniture", "table"],
            )
        )

        # Use the multiscale indicator to pick a resolution level for the soft
        # cube based on its distance from the expected gripper contact point.
        cube_center = (
            self.gripper_center[0],
            self.gripper_center[1],
            self.cube_size / 2,
        )
        half = self.cube_size / 2
        corners = np.array(
            [
                [cube_center[0] - half, cube_center[1] - half, cube_center[2] - half],
                [cube_center[0] - half, cube_center[1] + half, cube_center[2] - half],
                [cube_center[0] + half, cube_center[1] - half, cube_center[2] - half],
                [cube_center[0] + half, cube_center[1] + half, cube_center[2] - half],
                [cube_center[0] - half, cube_center[1] - half, cube_center[2] + half],
                [cube_center[0] - half, cube_center[1] + half, cube_center[2] + half],
                [cube_center[0] + half, cube_center[1] - half, cube_center[2] + half],
                [cube_center[0] + half, cube_center[1] + half, cube_center[2] + half],
            ],
            dtype=np.float64,
        )

        indicator = MultiscaleErrorIndicator(
            IndicatorConfig(
                roi_radius=0.08,
                transition_width=0.02,
                fine_threshold=0.5,
                medium_threshold=0.2,
            )
        )
        levels = indicator.evaluate(corners, roi_center=self.gripper_center)
        recommended_level = int(np.max(levels))  # Conservative: use finest recommended.

        print(f"Multiscale indicator recommended resolution level: {recommended_level}")
        print(f"  Corner levels: {levels.tolist()}")

        deformable_config = DeformableConfig(
            material=DeformableMaterialType.FEM_ELASTIC,
            youngs_modulus=1.0e4,
            poisson_ratio=0.45,
            density=1000.0,
            resolution_level=recommended_level,
            region_of_interest=(self.gripper_center, 0.06),
        )

        self.add_object(
            ObjectLibrary.deformable_soft_cube(
                name="soft_cube",
                position=cube_center,
                size=self.cube_size,
                color=(0.2, 0.7, 0.3, 1.0),
                deformable_config=deformable_config,
            )
        )


def create_predefined_grasp_action(step: int, total_steps: int) -> np.ndarray:
    """Return a simple open-loop grasp trajectory.

    The trajectory moves the arm above the cube, lowers the gripper, closes it,
    and lifts.  It is intentionally simple and only serves to demonstrate
    soft-body contact.
    """
    # Franka action: 7 joint targets + 1 gripper target, scaled to [-1, 1].
    # These are rough hand-tuned joint-angle targets, not dynamically computed.
    if step < total_steps * 0.3:
        # Approach: arm reaches forward, gripper opens.
        arm = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.5, 0.0])
        gripper = 1.0
    elif step < total_steps * 0.5:
        # Lower and close gripper.
        arm = np.array([0.0, -0.6, 0.0, -1.9, 0.0, 1.4, 0.0])
        gripper = -1.0
    elif step < total_steps * 0.7:
        # Grasp and lift slightly.
        arm = np.array([0.0, -0.6, 0.0, -1.9, 0.0, 1.5, 0.0])
        gripper = -1.0
    else:
        # Hold.
        arm = np.array([0.0, -0.6, 0.0, -1.9, 0.0, 1.5, 0.0])
        gripper = -1.0

    return np.concatenate([arm, np.array([gripper])])


def main() -> None:
    """Run the multiscale soft-body grasp example."""
    parser = argparse.ArgumentParser(
        description="Deng-Yu inspired multiscale soft-body grasping with Genesis"
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dengyu_multiscale_soft_grasp",
        help="Directory to save rendered frames",
    )
    args = parser.parse_args()

    # Detect pytest: shorten run for smoke tests.
    if "PYTEST_VERSION" in os.environ:
        args.steps = 5
        args.headless = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Deng-Yu Multiscale Soft-Body Grasp Example")
    print("=" * 60)

    # FEM solver options for the deformable body.
    fem_options = gs.options.FEMOptions(
        use_implicit_solver=True,
        enable_vertex_constraints=True,
    )

    composer = EnvironmentComposer(
        ComposerConfig(
            headless=args.headless,
            resolution=(800, 600),
            device="cpu" if args.cpu else "cuda",
            dt=2e-3,
            substeps=5,
            fem_options=fem_options,
        )
    )

    scene = SoftGraspScene()
    robot = FrankaPanda(
        EmbodimentConfig(
            name="franka_01",
            base_position=(0.0, 0.0, 0.0),
        )
    )
    task = PickPlaceTask(
        TaskConfig(max_episode_steps=args.steps),
        object_name="soft_cube",
        target_position=(0.5, 0.0, 0.2),
    )

    print("\nComposing environment...")
    env = composer.compose(scene, robot, task)

    print(f"\nRunning open-loop grasp trajectory for {args.steps} steps...")
    obs, info = env.reset(seed=0)

    for step in range(args.steps):
        action = create_predefined_grasp_action(step, args.steps)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                np.save(output_dir / f"frame_{step:04d}.npy", frame)

        if terminated or truncated:
            break

    print(f"\nSaved frames to {output_dir}")
    print("Example completed!")
    env.close()


if __name__ == "__main__":
    main()
