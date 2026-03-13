"""
Demo all available tasks in Genesis ManiSkill.

This script demonstrates each task type with random actions.
"""

import argparse
import time

import genesis as gs
from genesis_maniskill.envs import KitchenEnv, TableTopEnv
from genesis_maniskill.tasks import list_available_tasks, get_task


def demo_task(task_name: str, scene_type: str, num_steps: int = 100):
    """Demo a single task."""
    print(f"\n{'='*60}")
    print(f"Demoing: {task_name} ({scene_type})")
    print('='*60)
    
    # Create appropriate environment
    if scene_type == 'kitchen':
        env = KitchenEnv(
            num_envs=1,
            layout_id=0,
            robot_uid='franka',
            task_type=task_name,
            render_mode='human',
            sim_freq=100,
            control_freq=20,
        )
    else:  # tabletop
        env = TableTopEnv(
            num_envs=1,
            num_objects=3,
            robot_uid='franka',
            task_type=task_name,
            render_mode='human',
            sim_freq=100,
            control_freq=20,
        )
    
    # Reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run
    total_reward = 0
    success = False
    
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward.item()
        
        # Render
        env.render()
        
        # Check success
        if hasattr(env, 'task') and env.task.check_success().any():
            print(f"✓ Task completed at step {step}!")
            success = True
            break
        
        # Print progress
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward.item():.3f}, total={total_reward:.2f}")
        
        # Slow down for viewing
        time.sleep(0.02)
        
        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break
    
    print(f"Final: Total reward = {total_reward:.2f}, Success = {success}")
    
    env.close()


def demo_all_kitchen_tasks():
    """Demo all kitchen tasks."""
    kitchen_tasks = [
        'pick_place',
        'prepare_food',
        'cleanup',
        'organize_cabinet',
    ]
    
    for task in kitchen_tasks:
        try:
            demo_task(task, 'kitchen', num_steps=100)
        except Exception as e:
            print(f"Error demoing {task}: {e}")
        
        input("\nPress Enter to continue to next task...")


def demo_all_tabletop_tasks():
    """Demo all tabletop tasks."""
    tabletop_tasks = [
        'pick_place',
        'push',
        'stack',
        'insert',
        'sort',
        'assembly',
    ]
    
    for task in tabletop_tasks:
        try:
            demo_task(task, 'tabletop', num_steps=100)
        except Exception as e:
            print(f"Error demoing {task}: {e}")
        
        input("\nPress Enter to continue to next task...")


def print_task_catalog():
    """Print catalog of all available tasks."""
    print("\n" + "="*60)
    print("Genesis ManiSkill Task Catalog")
    print("="*60)
    
    tasks = list_available_tasks()
    
    for category, task_list in tasks.items():
        print(f"\n{category} Tasks:")
        print("-" * 40)
        for task in task_list:
            print(f"  • {task}")
    
    print(f"\nTotal tasks: {sum(len(tl) for tl in tasks.values())}")


def main():
    parser = argparse.ArgumentParser(description='Demo all tasks')
    parser.add_argument(
        '--category',
        choices=['kitchen', 'tabletop', 'all', 'list'],
        default='list',
        help='Which category of tasks to demo'
    )
    parser.add_argument(
        '--task',
        help='Demo a specific task'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of steps per task'
    )
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(backend=gs.cpu)
    
    try:
        if args.task:
            # Demo specific task
            scene_type = 'kitchen' if args.task in ['prepare_food', 'cleanup', 'organize_cabinet'] else 'tabletop'
            demo_task(args.task, scene_type, args.steps)
        elif args.category == 'list':
            print_task_catalog()
        elif args.category == 'kitchen':
            demo_all_kitchen_tasks()
        elif args.category == 'tabletop':
            demo_all_tabletop_tasks()
        elif args.category == 'all':
            print_task_catalog()
            input("\nPress Enter to start kitchen tasks demo...")
            demo_all_kitchen_tasks()
            input("\nPress Enter to start tabletop tasks demo...")
            demo_all_tabletop_tasks()
    
    finally:
        print("\nDone!")


if __name__ == '__main__':
    main()
