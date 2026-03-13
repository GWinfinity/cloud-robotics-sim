"""
Basic tabletop environment example.
"""

import genesis as gs
from genesis_maniskill.envs import TableTopEnv


def main():
    """Run basic tabletop environment."""
    print("Initializing Genesis...")
    gs.init(backend=gs.gpu)
    
    print("Creating tabletop environment...")
    env = TableTopEnv(
        num_envs=4,  # 4 parallel environments
        table_size=(1.0, 0.6, 0.05),
        num_objects=3,
        object_types=["cube", "sphere", "cylinder"],
        robot_uid="franka",
        task_type="pick_place",
        obs_mode="state",
        render_mode="human",
    )
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Number of environments: {env.num_envs}")
    
    print("\nRunning parallel simulation...")
    for step in range(50):
        # Random actions for all envs
        action = env.action_space.sample()
        
        # Step all environments
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render (only first env)
        env.render()
        
        if step % 10 == 0:
            print(f"Step {step}: reward={reward.mean().item():.3f}")
    
    print("\nClosing environment...")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
