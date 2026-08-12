#!/usr/bin/env python3
"""Multi-room apartment example using the Cloud Robotics Sim core API.

Builds several rooms in a single Genesis scene and runs a short simulation.
Use ``--headless`` to run without the viewer.
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


def create_room(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
) -> EmptyRoom:
    """Create a simple furnished room."""
    scene = EmptyRoom(size=size)

    if name == "living_room":
        scene.add_object(ObjectLibrary.sofa_three_seat(position=(0, -1.0, 0)))
        scene.add_object(ObjectLibrary.coffee_table(position=(0.5, 0, 0)))
        scene.add_object(
            ObjectLibrary.graspable_cube(
                name=f"{name}_cube",
                position=(0.5, 0, 0.6),
                color=(0.9, 0.2, 0.2, 1.0),
            )
        )
    elif name == "bedroom":
        scene.add_object(ObjectLibrary.bed_double(position=(0, 0, 0)))
    elif name == "kitchen":
        scene.add_object(ObjectLibrary.refrigerator(position=(-1.0, -1.0, 0)))
    elif name == "bathroom":
        scene.add_object(ObjectLibrary.obstacle_box(position=(0, 0, 0)))
    elif name == "laundry_room":
        scene.add_object(ObjectLibrary.obstacle_box(position=(-0.5, 0, 0)))

    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis Apartment Simulation Example")
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
    print("Genesis Apartment Simulation Example")
    print("=" * 60)

    print("\n1. Initializing Genesis...")
    gs.init(backend=get_genesis_backend("cpu"))

    print("\n2. Creating rooms...")
    rooms = [
        ("living_room", (5.0, 5.0, 3.0), (0, 0, 0)),
        ("bedroom", (4.0, 4.0, 3.0), (-8, 0, 0)),
        ("kitchen", (4.0, 4.0, 3.0), (8, 0, 0)),
        ("bathroom", (3.0, 3.0, 3.0), (0, -6, 0)),
        ("laundry_room", (3.0, 3.0, 3.0), (8, -4, 0)),
    ]

    scenes = [create_room(name, size, pos) for name, size, pos in rooms]
    total_objects = sum(len(s.entities) for s in scenes)

    print(f"\n   Total rooms: {len(rooms)}")
    print(f"   Total scene objects: {total_objects}")

    print("\n3. Composing environment...")
    # Use the first room as the primary scene for this example.
    primary_scene = scenes[0]
    for other in scenes[1:]:
        for spawn in other.object_spawns:
            primary_scene.add_object(spawn)

    robot = FrankaPanda(
        EmbodimentConfig(name="franka_01", base_position=(0.0, 1.0, 0.0))
    )
    task = PickPlaceTask(
        TaskConfig(max_episode_steps=200),
        object_name="living_room_cube",
        target_position=(0.5, 0.5, 0.05),
    )

    composer = EnvironmentComposer(
        ComposerConfig(headless=args.headless, resolution=(800, 600))
    )
    env = composer.compose(primary_scene, robot, task)

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
