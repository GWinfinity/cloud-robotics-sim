"""
Advanced task examples showing multi-stage and mobile manipulation.

This script demonstrates:
1. Multi-stage task (prepare_food)
2. Mobile manipulation (mobile_manipulation)
3. Task progress tracking
4. Stage-specific behaviors
"""

import genesis as gs
import numpy as np
import torch

from genesis_maniskill.envs import KitchenEnv
from genesis_maniskill.tasks.prepare_food import PrepareFoodTask


def multi_stage_task_example():
    """Example of working with multi-stage tasks."""
    print("\n" + "="*60)
    print("Multi-Stage Task Example: Prepare Food")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    
    env = KitchenEnv(
        num_envs=1,
        layout_id=0,
        robot_uid='franka',
        task_type='prepare_food',
        render_mode='human',
    )
    
    obs, info = env.reset()
    
    # Get task info
    if hasattr(env.task, 'STAGES'):
        print(f"Task stages: {env.task.STAGES}")
    
    # Simulate with some logic
    for step in range(200):
        # Simple stage-based policy
        if hasattr(env.task, 'current_stage'):
            stage = env.task.current_stage
            
            if stage == 0:  # Pick ingredient
                # Move toward ingredient
                action = np.random.randn(*env.action_space.shape) * 0.3
            elif stage == 1:  # Place on board
                action = np.random.randn(*env.action_space.shape) * 0.2
            elif stage == 2:  # Cut
                # Cutting motion
                action = np.random.randn(*env.action_space.shape) * 0.5
            else:  # Transfer
                action = np.random.randn(*env.action_space.shape) * 0.3
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print stage info
        if hasattr(env.task, 'get_stage_info') and step % 20 == 0:
            stage_info = env.task.get_stage_info()
            print(f"Step {step}: Stage {stage_info['stage_name']}, "
                  f"Progress: {stage_info['stages_completed']}/{stage_info['total_stages']}")
        
        env.render()
        
        if env.task.check_success().any():
            print("✓ Task completed!")
            break
    
    env.close()


def mobile_manipulation_example():
    """Example of mobile manipulation task."""
    print("\n" + "="*60)
    print("Mobile Manipulation Example")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    
    # This would require a mobile manipulator robot
    # For now, we'll use Franka on a floating base
    env = KitchenEnv(
        num_envs=1,
        layout_id=0,
        robot_uid='franka',  # Would use mobile manipulator
        task_type='mobile_manipulation',
        render_mode='human',
    )
    
    obs, info = env.reset()
    
    print("Mobile manipulation requires a robot with mobile base.")
    print("This example shows the task structure.")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if hasattr(env.task, 'get_stage_info') and step % 10 == 0:
            stage_info = env.task.get_stage_info()
            print(f"Step {step}: {stage_info['stage_name']}")
        
        env.render()
    
    env.close()


def task_progress_tracking_example():
    """Example of tracking task progress."""
    print("\n" + "="*60)
    print("Task Progress Tracking Example")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    
    env = KitchenEnv(
        num_envs=1,
        layout_id=0,
        robot_uid='franka',
        task_type='cleanup',
        render_mode='human',
    )
    
    obs, info = env.reset()
    
    progress_history = []
    
    for step in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track progress
        if hasattr(env.task, 'get_progress'):
            progress = env.task.get_progress()
            progress_history.append(progress)
            
            if step % 20 == 0:
                print(f"Step {step}: Progress = {progress:.1%}")
        
        env.render()
        
        if env.task.check_success().any():
            print("✓ All items cleaned up!")
            break
    
    print(f"\nFinal progress: {progress_history[-1]:.1%}")
    
    env.close()


def custom_task_policy_example():
    """Example of writing a custom policy for a specific task."""
    print("\n" + "="*60)
    print("Custom Task Policy Example: Assembly")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    
    env = KitchenEnv(
        num_envs=1,
        layout_id=0,
        robot_uid='franka',
        task_type='assembly',
        render_mode='human',
    )
    
    obs, info = env.reset()
    
    # Get assembly info
    if hasattr(env.task, 'get_assembly_structure'):
        structure = env.task.get_assembly_structure()
        print(f"Assembly: {structure['num_parts']} parts")
        print(f"Order: {structure['assembly_order']}")
    
    # Simple policy: try to place parts in order
    for step in range(200):
        # Get current part to assemble
        if hasattr(env.task, 'get_current_part'):
            part, target = env.task.get_current_part()
            if part is not None and target is not None:
                # Simple proportional controller toward target
                # (In practice, you'd use proper motion planning)
                action = np.random.randn(*env.action_space.shape) * 0.2
            else:
                action = np.zeros(env.action_space.shape)
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if hasattr(env.task, 'get_progress') and step % 20 == 0:
            progress = env.task.get_progress()
            print(f"Step {step}: Assembly progress = {progress:.1%}")
        
        env.render()
        
        if env.task.check_success().any():
            print("✓ Assembly completed!")
            break
    
    env.close()


def compare_tasks():
    """Compare different task types."""
    print("\n" + "="*60)
    print("Task Comparison")
    print("="*60)
    
    gs.init(backend=gs.cpu)
    
    tasks_to_compare = [
        ('pick_place', 'Simple pick and place'),
        ('stack', 'Stack objects'),
        ('sort', 'Sort by type'),
        ('assembly', 'Assemble parts'),
    ]
    
    for task_name, description in tasks_to_compare:
        print(f"\n{task_name}: {description}")
        print("-" * 40)
        
        env = KitchenEnv(
            num_envs=1,
            layout_id=0,
            robot_uid='franka',
            task_type=task_name,
            render_mode=None,  # No rendering for speed
        )
        
        obs, info = env.reset()
        
        # Quick test
        total_reward = 0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.item()
        
        print(f"  State dim: {env.task.state_dim}")
        print(f"  Avg reward: {total_reward / 50:.3f}")
        
        env.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced task examples')
    parser.add_argument(
        '--example',
        choices=['multi_stage', 'mobile', 'progress', 'custom', 'compare', 'all'],
        default='all',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == 'multi_stage' or args.example == 'all':
            multi_stage_task_example()
            input("\nPress Enter to continue...")
        
        if args.example == 'mobile' or args.example == 'all':
            mobile_manipulation_example()
            input("\nPress Enter to continue...")
        
        if args.example == 'progress' or args.example == 'all':
            task_progress_tracking_example()
            input("\nPress Enter to continue...")
        
        if args.example == 'custom' or args.example == 'all':
            custom_task_policy_example()
            input("\nPress Enter to continue...")
        
        if args.example == 'compare' or args.example == 'all':
            compare_tasks()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
