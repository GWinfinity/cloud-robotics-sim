"""CoStream peg-into-slot demonstration in Genesis."""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow importing the sibling costream package without installation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from costream import (
    ActionComposer,
    CartesianCompliantController,
    CoStreamBehaviors,
    CoStreamRuntime,
    ControllerCompiler,
    InsertionScene,
    ObjectInfo,
    PredictiveBehavior,
    ReactiveBehavior,
    SceneSummary,
    SemanticBehavior,
)
from costream.math_utils import pos_quat_to_matrix, translation_matrix
from costream.specs import ComposeSpec, StageSpec


def build_scene_summary(scene: InsertionScene) -> SceneSummary:
    """Ground-truth scene summary used as surrogate for LLM/VLM output."""
    return SceneSummary(
        objects={
            "hole": ObjectInfo(
                name="hole",
                pos=scene.hole_center.copy(),
                quat=scene.hole_quat.copy(),
                dims=(0.012, 0.100, 0.050),
            ),
            "peg": ObjectInfo(
                name="peg",
                pos=np.array(scene.peg.get_pos()).flatten(),
                quat=np.array(scene.peg.get_quat()).flatten(),
                dims=(0.005, 0.080),
            ),
        },
    )


def make_stages() -> list[tuple[StageSpec, ComposeSpec]]:
    """Define the approach-insert-home sequence."""
    # Task frame z points down.  Motions are expressed in that frame.
    approach_start = translation_matrix([0.0, 0.0, -0.10])  # 10 cm above hole
    approach_goal = translation_matrix([0.0, 0.0, 0.0])  # hole surface

    insert_start = translation_matrix([0.0, 0.0, 0.0])
    insert_goal = translation_matrix([0.0, 0.0, 0.05])  # 5 cm deep

    home_start = insert_goal
    home_goal = translation_matrix([0.0, 0.0, -0.12])

    stages = [
        (
            StageSpec(
                name="approach",
                objective="Move the peg to the top of the slot",
                task_frame="hole",
                motion="approach",
                duration=2.0,
                profile=ControllerCompiler.profile_library("free_space"),
                relative_start=approach_start,
                relative_goal=approach_goal,
            ),
            ComposeSpec(
                composition_frame="task",
                axis_ownership=np.ones(6, dtype=bool),
                residual_bounds=np.array([0.02, 0.02, 0.02]),
            ),
        ),
        (
            StageSpec(
                name="insert",
                objective="Insert the peg into the narrow slot",
                task_frame="hole",
                motion="insert",
                duration=4.0,
                profile=ControllerCompiler.profile_library("insertion"),
                relative_start=insert_start,
                relative_goal=insert_goal,
            ),
            ComposeSpec(
                composition_frame="task",
                axis_ownership=np.array([True, True, True, False, False, False]),
                residual_bounds=np.array([0.01, 0.01, 0.005]),
            ),
        ),
        (
            StageSpec(
                name="home",
                objective="Retract the peg",
                task_frame="hole",
                motion="home",
                duration=2.0,
                profile=ControllerCompiler.profile_library("free_space"),
                relative_start=home_start,
                relative_goal=home_goal,
            ),
            ComposeSpec(
                composition_frame="task",
                axis_ownership=np.ones(6, dtype=bool),
                residual_bounds=np.array([0.02, 0.02, 0.02]),
            ),
        ),
    ]
    return stages


def main() -> None:
    print("=" * 60)
    print("CoStream Genesis Demo: Peg Insertion")
    print("=" * 60)

    # Build scene.
    scene = InsertionScene(headless=True, dt=0.005, substeps=4)
    scene.build()

    # Define TCP: a cylinder extending 10 cm downward from the hand frame,
    # which is oriented with z pointing down for the insertion pose.
    tcp_offset = translation_matrix([0.0, 0.0, 0.10])
    scene.robot.set_tool(scene.peg, tcp_offset)

    # Move the arm to a reasonable starting configuration once.
    # Use IK to place the TCP above the hole before the behavior loop starts.
    # Initial TCP is 10 cm above the hole (world +z) before approach starts.
    hand_init_pose = (
        pos_quat_to_matrix(scene.hole_center + np.array([0.0, 0.0, +0.10]), scene.hole_quat)
        @ np.linalg.inv(tcp_offset)
    )
    q_init = scene.robot.ik(hand_init_pose)
    scene.robot.set_joint_positions(q_init)
    for _ in range(200):
        scene.scene.step()
    scene.robot.update_tool_pose()

    print(f"Initial TCP pos: {scene.robot.get_tcp_pose()[:3, 3]}")
    print(f"Hole center:     {scene.hole_center}")

    # Behaviors.
    behaviors = CoStreamBehaviors(
        semantic=SemanticBehavior(),
        predictive=PredictiveBehavior(),
        reactive=ReactiveBehavior(
            tactile_gain=0.8,
            force_gain=2e-4,
            max_lateral_correction=0.012,
            contact_force_threshold=1.5,
        ),
    )

    # Controller.
    controller = CartesianCompliantController(
        robot=scene.robot,
        tcp_offset=tcp_offset,
    )

    # Runtime.
    runtime = CoStreamRuntime(
        scene=scene,
        behaviors=behaviors,
        get_scene_summary=lambda: build_scene_summary(scene),
        dt=scene.dt,
    )

    task_z = np.array([0.0, 0.0, -1.0])
    result = runtime.run(make_stages(), controller, task_z_in_world=task_z)

    # Summary.
    print("\n" + "=" * 60)
    print("Result Summary")
    print("=" * 60)
    for name, r in result["results"]:
        status = "OK" if r["success"] else f"FAIL ({r['reason']})"
        print(f"  {name:10s}: {status}")

    log = result["log"]
    if log:
        final = log[-1]
        print(f"\nFinal TCP pos:  {final['W_T_cmd_pos']}")
        max_force = max(entry.get("contact_force_norm", 0.0) for entry in log)
        print(f"Max contact force observed: {max_force:.3f} N")

    scene.destroy()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
