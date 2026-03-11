"""Basic usage example for Cloud Robotics Simulation Platform.

This example demonstrates how to create a simple pick-and-place
environment and run a random policy.
"""

import numpy as np

from cloud_robotics_sim import (
    ComposerConfig,
    EmbodimentConfig,
    EnvironmentComposer,
    FrankaPanda,
    ObjectLibrary,
    PickPlaceTask,
    Scene,
    SceneConfig,
    TaskConfig,
)


def create_living_room_scene() -> Scene:
    """Create a simple living room scene."""
    config = SceneConfig(
        name="living_room",
        size=(5.0, 5.0, 3.0),
        default_camera_pos=(2.5, 2.5, 2.5),
    )
    
    scene = Scene(config)
    
    # Add furniture
    scene.add_object(ObjectLibrary.sofa_three_seat(position=(0, -1.5, 0)))
    scene.add_object(ObjectLibrary.coffee_table(position=(0.5, 0, 0)))
    
    # Add graspable object
    scene.add_object(ObjectLibrary.graspable_cube(
        name="red_cube",
        position=(0.5, 0, 0.6),
        color=(0.9, 0.2, 0.2, 1.0),
    ))
    
    return scene


def main():
    """Run the basic usage example."""
    print("=" * 60)
    print("Cloud Robotics Sim - Basic Usage Example")
    print("=" * 60)
    
    # Create composer with visualization
    composer = EnvironmentComposer(ComposerConfig(
        headless=False,
        resolution=(800, 600),
    ))
    
    # Create components
    scene = create_living_room_scene()
    
    robot = FrankaPanda(EmbodimentConfig(
        name="franka_01",
        base_position=(0.0, 1.0, 0.0),
    ))
    
    task = PickPlaceTask(
        TaskConfig(max_episode_steps=200),
        object_name="red_cube",
        target_position=(0.5, 0.5, 0.05),
    )
    
    # Compose environment
    print("\nComposing environment...")
    env = composer.compose(scene, robot, task)
    
    # Run random policy
    print("\nRunning random policy for 3 episodes...")
    
    for episode in range(3):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        
        for step in range(200):
            # Random action
            action = np.random.uniform(-1, 1, size=8)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Success = {info.get('success', False)}")
    
    print("\nExample completed!")
    env.close()


if __name__ == "__main__":
    main()
