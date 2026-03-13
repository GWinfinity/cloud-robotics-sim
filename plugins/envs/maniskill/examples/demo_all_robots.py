"""
Demo all available robots in Genesis ManiSkill.

This script demonstrates each robot type with basic movements.
"""

import argparse
import time

import genesis as gs
from genesis_maniskill.envs import KitchenEnv, TableTopEnv
from genesis_maniskill.agents import list_available_agents, get_agent_info


def demo_robot(robot_name: str, num_steps: int = 100):
    """Demo a single robot."""
    print(f"\n{'='*60}")
    print(f"Demoing: {robot_name}")
    print('='*60)
    
    # Get robot info
    info = get_agent_info(robot_name)
    print(f"Name: {info.get('name', robot_name)}")
    print(f"Type: {info.get('type', 'unknown')}")
    print(f"Description: {info.get('description', '')}")
    
    # Determine environment type
    if info.get('type') == 'mobile_manipulator':
        env_class = KitchenEnv
        kwargs = {
            'num_envs': 1,
            'layout_id': 0,
            'robot_uid': robot_name,
            'task_type': 'pick_place',
            'render_mode': 'human',
        }
    elif info.get('type') == 'humanoid':
        env_class = KitchenEnv
        kwargs = {
            'num_envs': 1,
            'layout_id': 0,
            'robot_uid': robot_name,
            'task_type': 'pick_place',
            'render_mode': 'human',
        }
    else:  # manipulator
        env_class = TableTopEnv
        kwargs = {
            'num_envs': 1,
            'num_objects': 2,
            'robot_uid': robot_name,
            'task_type': 'pick_place',
            'render_mode': 'human',
        }
    
    # Create environment
    try:
        env = env_class(**kwargs)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Print robot-specific info
    if hasattr(env.robot, 'arm_dof'):
        print(f"Arm DOF: {env.robot.arm_dof}")
    if hasattr(env.robot, 'total_dof'):
        print(f"Total DOF: {env.robot.total_dof}")
    
    # Run
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Print progress
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward.item():.3f}")
        
        # Slow down for viewing
        time.sleep(0.02)
        
        if terminated or truncated:
            break
    
    env.close()


def demo_all_manipulators():
    """Demo all manipulator robots."""
    manipulators = ['franka', 'ur5', 'kinova', 'xarm7']
    
    for robot in manipulators:
        try:
            demo_robot(robot, num_steps=80)
        except Exception as e:
            print(f"Error demoing {robot}: {e}")
        
        input("\nPress Enter to continue to next robot...")


def demo_all_humanoids():
    """Demo all humanoid robots."""
    humanoids = ['g1', 'gr1']
    
    for robot in humanoids:
        try:
            demo_robot(robot, num_steps=80)
        except Exception as e:
            print(f"Error demoing {robot}: {e}")
        
        input("\nPress Enter to continue to next robot...")


def demo_all_mobile():
    """Demo all mobile manipulators."""
    mobile = ['fetch', 'tiago']
    
    for robot in mobile:
        try:
            demo_robot(robot, num_steps=80)
        except Exception as e:
            print(f"Error demoing {robot}: {e}")
        
        input("\nPress Enter to continue to next robot...")


def print_robot_catalog():
    """Print catalog of all available robots."""
    print("\n" + "="*60)
    print("Genesis ManiSkill Robot Catalog")
    print("="*60)
    
    robots = list_available_agents()
    
    for category, robot_list in robots.items():
        print(f"\n{category}:")
        print("-" * 40)
        for robot in robot_list:
            info = get_agent_info(robot)
            print(f"  • {robot:15} - {info.get('name', '')}")
            if 'dof' in info:
                print(f"    {' ' * 17} DOF: {info['dof']}, {info.get('description', '')}")
            else:
                print(f"    {' ' * 17} {info.get('description', '')}")
    
    total = sum(len(rl) for rl in robots.values())
    print(f"\nTotal robots: {total}")


def compare_robots():
    """Compare robot specifications."""
    print("\n" + "="*60)
    print("Robot Comparison")
    print("="*60)
    
    robots_to_compare = ['franka', 'ur5', 'kinova', 'xarm7']
    
    print(f"\n{'Robot':<15} {'DOF':<5} {'Type':<15} {'Payload':<10} {'Description'}")
    print("-" * 70)
    
    for robot in robots_to_compare:
        info = get_agent_info(robot)
        print(f"{info.get('name', robot):<15} "
              f"{info.get('dof', 'N/A'):<5} "
              f"{info.get('type', 'N/A'):<15} "
              f"{info.get('payload', 'N/A'):<10} "
              f"{info.get('description', '')}")


def main():
    parser = argparse.ArgumentParser(description='Demo all robots')
    parser.add_argument(
        '--category',
        choices=['manipulators', 'humanoids', 'mobile', 'all', 'list', 'compare'],
        default='list',
        help='Which category of robots to demo'
    )
    parser.add_argument(
        '--robot',
        help='Demo a specific robot'
    )
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(backend=gs.cpu)
    
    try:
        if args.robot:
            demo_robot(args.robot, num_steps=100)
        elif args.category == 'list':
            print_robot_catalog()
        elif args.category == 'compare':
            compare_robots()
        elif args.category == 'manipulators':
            demo_all_manipulators()
        elif args.category == 'humanoids':
            demo_all_humanoids()
        elif args.category == 'mobile':
            demo_all_mobile()
        elif args.category == 'all':
            print_robot_catalog()
            compare_robots()
            input("\nPress Enter to start manipulators demo...")
            demo_all_manipulators()
            input("\nPress Enter to start humanoids demo...")
            demo_all_humanoids()
            input("\nPress Enter to start mobile demo...")
            demo_all_mobile()
    
    finally:
        print("\nDone!")


if __name__ == '__main__':
    main()
