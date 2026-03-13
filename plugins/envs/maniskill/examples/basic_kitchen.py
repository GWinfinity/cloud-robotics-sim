"""
Basic kitchen environment example.
"""

import genesis as gs
from genesis_maniskill.envs import KitchenEnv


def main():
    """Run basic kitchen environment."""
    print("Initializing Genesis...")
    gs.init(backend=gs.gpu)
    
    print("Creating kitchen environment...")
    env = KitchenEnv(
        num_envs=1,
        layout_id=0,  # G-shaped kitchen
        style_id=0,   # Modern style
        robot_uid="franka",
        task_type="pick_place",
        obs_mode="state",
        render_mode="human",
        sim_freq=100,
        control_freq=20,
    )
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    print("\nRunning simulation...")
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward.item():.3f}")
        
        # Check if done
        if terminated or truncated:
            print("Episode finished, resetting...")
            obs, info = env.reset()
    
    print("\nClosing environment...")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
