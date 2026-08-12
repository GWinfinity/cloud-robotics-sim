#!/usr/bin/env python3
"""Single-scene example using the Cloud Robotics Sim core API.

Shows how to build a single room, add furniture, and run a short
headless simulation. Use ``--headless`` to run without the viewer.
"""

from __future__ import annotations

import argparse

import genesis as gs

from cloud_robotics_sim import (
    ComposerConfig,
    EmbodimentConfig,
    EnvironmentComposer,
    FrankaPanda,
    ObjectLibrary,
    PickPlaceTask,
    TaskConfig,
)
from cloud_robotics_sim.core.scenes import EmptyRoom
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis Single Scene Example")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive viewer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps to run.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Genesis Single Scene Example")
    print("=" * 60)

    print("\n1. Initializing Genesis...")
    gs.init(backend=get_genesis_backend("cpu"))

    print("\n2. Creating scene...")
    scene = EmptyRoom(size=(5.0, 5.0, 3.0))
    scene.add_object(ObjectLibrary.sofa_three_seat(position=(0, -1.5, 0)))
    scene.add_object(ObjectLibrary.coffee_table(position=(0.5, 0, 0)))
    scene.add_object(
        ObjectLibrary.graspable_cube(
            name="red_cube",
            position=(0.5, 0, 0.6),
            color=(0.9, 0.2, 0.2, 1.0),
        )
    )

    robot = FrankaPanda(
        EmbodimentConfig(name="franka_01", base_position=(0.0, 1.0, 0.0))
    )
    task = PickPlaceTask(
        TaskConfig(max_episode_steps=200),
        object_name="red_cube",
        target_position=(0.5, 0.5, 0.05),
    )

    print("\n3. Composing environment...")
    composer = EnvironmentComposer(
        ComposerConfig(headless=args.headless, resolution=(800, 600))
    )
    env = composer.compose(scene, robot, task)
    print(f"   Created {len(scene.entities)} scene objects")

    print("\n4. Running simulation...")
    env.reset(seed=0)
    for i in range(args.steps):
        env.step(env.robot.action_space["low"] * 0.1)
        if i % 20 == 0:
            print(f"   Step {i}")

    print("\n" + "=" * 60)
    print("Simulation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
