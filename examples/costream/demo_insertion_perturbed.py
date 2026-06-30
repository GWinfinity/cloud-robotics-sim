"""CoStream peg-into-slot demo with intentional semantic misalignment.

The "hole" object in the scene summary is deliberately shifted by +6 mm in y.
This makes the semantic anchor guide the peg toward one wall.  The reactive
behavior senses the resulting lateral contact force and composes a corrective
residual that centers the peg in the slot.
"""

from __future__ import annotations

import os
import sys

import numpy as np

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


# Intentional lateral error: half the clearance, enough to brush the wall.
SEMANTIC_OFFSET = np.array([0.0, 0.006, 0.0])


def build_scene_summary(scene: InsertionScene) -> SceneSummary:
    """Return a *perturbed* scene summary: the hole is reported off-center."""
    return SceneSummary(
        objects={
            "hole": ObjectInfo(
                name="hole",
                pos=scene.hole_center.copy() + SEMANTIC_OFFSET,
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
    approach_start = translation_matrix([0.0, 0.0, -0.10])
    approach_goal = translation_matrix([0.0, 0.0, 0.0])
    insert_start = translation_matrix([0.0, 0.0, 0.0])
    insert_goal = translation_matrix([0.0, 0.0, 0.05])
    home_start = insert_goal
    home_goal = translation_matrix([0.0, 0.0, -0.12])

    return [
        (
            StageSpec(
                name="approach",
                objective="Move the peg to the (reported) top of the slot",
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
                objective="Insert the peg; reactive behavior must correct lateral error",
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


def main() -> None:
    print("=" * 60)
    print("CoStream Genesis Demo: Perturbed Peg Insertion")
    print(f"Semantic anchor offset: {SEMANTIC_OFFSET}")
    print("=" * 60)

    scene = InsertionScene(headless=True, dt=0.005, substeps=4)
    scene.build()

    tcp_offset = translation_matrix([0.0, 0.0, 0.10])
    scene.robot.set_tool(scene.peg, tcp_offset)

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
    print(f"True hole center: {scene.hole_center}")
    print(f"Reported hole center: {scene.hole_center + SEMANTIC_OFFSET}")

    behaviors = CoStreamBehaviors(
        semantic=SemanticBehavior(),
        predictive=PredictiveBehavior(),
        reactive=ReactiveBehavior(
            tactile_gain=0.8,
            force_gain=3e-4,
            max_lateral_correction=0.015,
            contact_force_threshold=1.0,
        ),
    )

    controller = CartesianCompliantController(
        robot=scene.robot,
        tcp_offset=tcp_offset,
    )

    runtime = CoStreamRuntime(
        scene=scene,
        behaviors=behaviors,
        get_scene_summary=lambda: build_scene_summary(scene),
        dt=scene.dt,
    )

    task_z = np.array([0.0, 0.0, -1.0])
    result = runtime.run(make_stages(), controller, task_z_in_world=task_z)

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
        corrections = [
            float(np.linalg.norm(entry.get("lateral_correction", np.zeros(3))))
            for entry in log
            if "lateral_correction" in entry
        ]
        if corrections:
            print(f"Max lateral reactive correction: {max(corrections)*1000:.2f} mm")
        else:
            print("No lateral reactive correction was triggered.")

    scene.destroy()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
