"""
Examples of loading robots from URDF/MJCF with the new generic loader.
"""

import genesis as gs
from genesis_maniskill.agents import RobotLoader, RobotConfig, get_robot


def example_1_load_from_urdf():
    """Example 1: Load robot directly from URDF file."""
    print("\n" + "="*60)
    print("Example 1: Load from URDF")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Method 1: Direct URDF loading
    robot = RobotLoader.from_urdf(
        scene=scene,
        urdf_path="urdf/franka_emika_panda/panda.urdf",
        num_envs=1,
        name="MyFranka",
    )
    
    print(f"Loaded: {robot.config.name}")
    print(f"DOF: {robot.config.total_dof}")
    print(f"Action dim: {robot.action_dim}")
    
    scene.build()
    robot.reset()


def example_2_load_from_config():
    """Example 2: Load robot from configuration file."""
    print("\n" + "="*60)
    print("Example 2: Load from Config File")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Method 2: Load from YAML/JSON config
    robot = RobotLoader.from_config(
        scene=scene,
        config_path="configs/robots/ur5.yaml",
        num_envs=1,
    )
    
    print(f"Loaded: {robot.config.name}")
    print(f"Joint limits: {robot.config.joint_limits}")
    print(f"Home position: {robot.config.home_position}")
    
    scene.build()
    robot.reset()


def example_3_load_from_preset():
    """Example 3: Load from preset configuration."""
    print("\n" + "="*60)
    print("Example 3: Load from Preset")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Method 3: Use preset (looks in configs/robots/)
    robot = RobotLoader.from_preset(
        scene=scene,
        preset_name="kinova_gen3",
        num_envs=1,
    )
    
    print(f"Loaded: {robot.config.name}")
    print(f"EE link: {robot.config.ee_link_name}")
    
    scene.build()
    robot.reset()


def example_4_generic_get_robot():
    """Example 4: Use generic get_robot function."""
    print("\n" + "="*60)
    print("Example 4: Generic get_robot")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Method 4: Universal get_robot function
    # Tries config, then URDF, then preset
    robot = get_robot(scene, 'xarm7', num_envs=1)
    
    print(f"Loaded: {robot.config.name}")
    print(f"Control mode: {robot.control_mode}")
    
    scene.build()
    robot.reset()


def example_5_create_custom_config():
    """Example 5: Create custom robot configuration programmatically."""
    print("\n" + "="*60)
    print("Example 5: Custom Configuration")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Create config programmatically
    config = RobotConfig(
        name="MyCustomRobot",
        urdf_path="path/to/my_robot.urdf",
        arm_dof=6,
        gripper_dof=1,
        joint_limits={
            'lower': np.array([-3.14, -1.57, -1.57, -3.14, -1.57, -3.14]),
            'upper': np.array([3.14, 1.57, 1.57, 3.14, 1.57, 3.14]),
        },
        home_position=np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ee_link_name="tool0",
        control_mode="pd_joint_pos",
    )
    
    # Create loader with custom config
    robot = RobotLoader(scene, config, num_envs=1)
    
    print(f"Created: {robot.config.name}")
    print(f"Custom config: {config.to_dict()}")
    
    # Save config for later use
    config.save("my_robot_config.yaml")
    print("Config saved to: my_robot_config.yaml")


def example_6_auto_detect_params():
    """Example 6: Auto-detect parameters from URDF."""
    print("\n" + "="*60)
    print("Example 6: Auto-detect from URDF")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Load without specifying parameters - they will be auto-detected
    robot = RobotLoader.from_urdf(
        scene=scene,
        urdf_path="urdf/some_robot/robot.urdf",
        num_envs=1,
    )
    
    print(f"Auto-detected DOF: {robot.config.arm_dof}")
    print(f"Auto-detected joint limits: {robot.config.joint_limits}")


def example_7_control_modes():
    """Example 7: Different control modes."""
    print("\n" + "="*60)
    print("Example 7: Control Modes")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    scene = gs.Scene()
    
    # Joint position control (default)
    robot_pos = RobotLoader.from_preset(
        scene=scene,
        preset_name="franka",
        num_envs=1,
        control_mode="pd_joint_pos",
    )
    print(f"Joint position control - Action dim: {robot_pos.action_dim}")
    
    # Joint velocity control
    robot_vel = RobotLoader.from_preset(
        scene=scene,
        preset_name="franka",
        num_envs=1,
        control_mode="pd_joint_vel",
    )
    print(f"Joint velocity control - Action dim: {robot_vel.action_dim}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot loading examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_load_from_urdf,
        2: example_2_load_from_config,
        3: example_3_load_from_preset,
        4: example_4_generic_get_robot,
        5: example_5_create_custom_config,
        6: example_6_auto_detect_params,
        7: example_7_control_modes,
    }
    
    if args.example:
        examples[args.example]()
    else:
        # Run all examples (with placeholder paths)
        print("Run with --example N to run specific example")
        print("\nAvailable examples:")
        for i, func in examples.items():
            print(f"  {i}. {func.__doc__.strip()}")


if __name__ == '__main__':
    import numpy as np
    main()
