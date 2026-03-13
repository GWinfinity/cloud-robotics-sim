"""
Migration examples showing how to port from RoboCasa and ManiSkill.
"""


def robocasa_migration():
    """
    Example: Migrating from RoboCasa to Genesis ManiSkill.
    """
    print("=" * 60)
    print("RoboCasa -> Genesis ManiSkill Migration Example")
    print("=" * 60)
    
    # --- RoboCasa (Old) ---
    print("\n# RoboCasa (Original)")
    print("""
    from robocasa.environments import KitchenEnv
    
    env = KitchenEnv(
        layout_id=0,
        style_id=0,
        robots="Panda",
        controller_configs={...},
    )
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    """)
    
    # --- Genesis ManiSkill (New) ---
    print("\n# Genesis ManiSkill (New)")
    print("""
    from genesis_maniskill.envs import KitchenEnv
    
    env = KitchenEnv(
        num_envs=16,           # NEW: GPU parallel support
        layout_id=0,           # Same as before
        style_id=0,            # Same as before
        robot_uid="franka",    # Changed from 'robots'
        task_type="pick_place", # NEW: explicit task
        obs_mode="state",      # NEW: flexible obs modes
        control_mode="pd_joint_pos",  # NEW: control mode
        sim_freq=100,          # NEW: simulation frequency
        control_freq=20,       # NEW: control frequency
    )
    
    obs, info = env.reset()   # NEW: returns info dict
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)  # NEW: terminated/truncated
    """)


def maniskill_migration():
    """
    Example: Migrating from ManiSkill to Genesis ManiSkill.
    """
    print("\n" + "=" * 60)
    print("ManiSkill -> Genesis ManiSkill Migration Example")
    print("=" * 60)
    
    # --- ManiSkill (Old) ---
    print("\n# ManiSkill (Original)")
    print("""
    import gymnasium as gym
    import mani_skill.envs
    
    env = gym.make(
        "PickCube-v1",
        num_envs=16,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    """)
    
    # --- Genesis ManiSkill (New) ---
    print("\n# Genesis ManiSkill (New)")
    print("""
    from genesis_maniskill.envs import TableTopEnv
    
    env = TableTopEnv(
        num_envs=16,
        table_size=(1.0, 0.6, 0.05),
        num_objects=1,
        object_types=["cube"],
        robot_uid="franka",
        task_type="pick_place",   # Maps to PickCube
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="human",
        sim_freq=100,
        control_freq=20,
    )
    
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    """)


def key_differences():
    """Print key differences between frameworks."""
    print("\n" + "=" * 60)
    print("Key Differences")
    print("=" * 60)
    
    differences = """
    1. Backend:
       - RoboCasa: MuJoCo (via RoboSuite)
       - ManiSkill: SAPIEN / PhysX
       - Genesis ManiSkill: Genesis (custom physics engine)
    
    2. GPU Parallel:
       - RoboCasa: Limited
       - ManiSkill: Full GPU parallel
       - Genesis ManiSkill: Full GPU parallel with Genesis backend
    
    3. Scene Support:
       - RoboCasa: Kitchen only
       - ManiSkill: Various scenes
       - Genesis ManiSkill: Kitchen + Tabletop + Custom
    
    4. API Design:
       - All use Gymnasium interface
       - Genesis ManiSkill adds explicit task_type parameter
       - Separate scene, robot, and task configurations
    
    5. Performance:
       - Genesis backend offers better performance for large parallel envs
       - Unified asset management across different scene types
    """
    print(differences)


if __name__ == "__main__":
    robocasa_migration()
    maniskill_migration()
    key_differences()
