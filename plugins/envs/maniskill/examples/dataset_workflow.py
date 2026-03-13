"""
Complete dataset workflow example.

This example demonstrates the full pipeline:
1. Load and convert data from RoboCasa/ManiSkill
2. Augment the dataset
3. Split into train/val/test
4. Visualize the data
5. Replay in simulation
6. Train a policy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from genesis_maniskill.datasets import (
    load_robocasa_dataset,
    load_maniskill_dataset,
    convert_robocasa_dataset,
    convert_maniskill_dataset,
    TrajectoryAugmenter,
    add_action_noise,
    perturb_states,
    split_dataset,
    merge_datasets,
    DatasetBalancer,
    TrajectoryVisualizer,
    visualize_dataset,
    replay_trajectory,
    validate_dataset,
)
from genesis_maniskill.datasets.formats import TrajectoryDataset


def step1_load_and_convert(args):
    """Step 1: Load or convert dataset."""
    print("\n" + "="*60)
    print("Step 1: Load/Convert Dataset")
    print("="*60)
    
    if args.converted_path and Path(args.converted_path).exists():
        print(f"Loading converted dataset from: {args.converted_path}")
        dataset = TrajectoryDataset.load(args.converted_path)
    else:
        if args.source_type == 'robocasa':
            print("Converting RoboCasa dataset...")
            output_path = convert_robocasa_dataset(
                source_path=args.source_path,
                target_path=args.converted_path or "./converted_dataset",
                max_demos=args.max_demos,
                verify=True,
            )
        elif args.source_type == 'maniskill':
            print("Converting ManiSkill dataset...")
            output_path = convert_maniskill_dataset(
                source_path=args.source_path,
                target_path=args.converted_path or "./converted_dataset",
                max_trajs=args.max_demos,
                verify=True,
            )
        else:
            raise ValueError(f"Unknown source type: {args.source_type}")
        
        dataset = TrajectoryDataset.load(output_path)
    
    print(f"Loaded {len(dataset)} trajectories")
    stats = dataset.get_statistics()
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Tasks: {stats['tasks']}")
    
    return dataset


def step2_augment(dataset, args):
    """Step 2: Augment dataset."""
    print("\n" + "="*60)
    print("Step 2: Data Augmentation")
    print("="*60)
    
    if not args.augment:
        print("Skipping augmentation (use --augment to enable)")
        return dataset
    
    # Create augmenter
    augmenter = TrajectoryAugmenter([
        lambda t: add_action_noise(t, noise_scale=0.01),
        lambda t: perturb_states(t, perturb_scale=0.005),
    ])
    
    # Augment
    print("Applying augmentations...")
    augmented_dataset = augmenter.augment_dataset(
        dataset.trajectories,
        n_copies=args.augment_copies,
        keep_original=True,
    )
    
    dataset = TrajectoryDataset(augmented_dataset)
    print(f"Augmented to {len(dataset)} trajectories")
    
    return dataset


def step3_split(dataset, args):
    """Step 3: Split dataset."""
    print("\n" + "="*60)
    print("Step 3: Split Dataset")
    print("="*60)
    
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
    )
    
    print(f"Train: {len(train_set)} trajectories")
    print(f"Val: {len(val_set)} trajectories")
    print(f"Test: {len(test_set)} trajectories")
    
    # Save splits
    if args.save_splits:
        output_dir = Path(args.converted_path or "./converted_dataset")
        train_set.save(output_dir / "train", format='hdf5')
        val_set.save(output_dir / "val", format='hdf5')
        test_set.save(output_dir / "test", format='hdf5')
        print(f"Saved splits to: {output_dir}")
    
    return train_set, val_set, test_set


def step4_visualize(train_set, args):
    """Step 4: Visualize dataset."""
    print("\n" + "="*60)
    print("Step 4: Visualization")
    print("="*60)
    
    if not args.visualize:
        print("Skipping visualization (use --visualize to enable)")
        return
    
    visualizer = TrajectoryVisualizer()
    
    # Visualize dataset statistics
    print("Creating dataset statistics plot...")
    output_dir = Path(args.converted_path or "./converted_dataset")
    visualizer.plot_dataset_statistics(
        train_set,
        save_path=output_dir / "dataset_stats.png",
        show=False,
    )
    
    # Visualize a sample trajectory
    if len(train_set) > 0:
        print("Creating sample trajectory plot...")
        visualizer.plot_trajectory(
            train_set[0],
            save_path=output_dir / "sample_trajectory.png",
            show=False,
        )
    
    print(f"Plots saved to: {output_dir}")


def step5_replay(val_set, args):
    """Step 5: Replay validation."""
    print("\n" + "="*60)
    print("Step 5: Replay Validation")
    print("="*60)
    
    if not args.replay:
        print("Skipping replay (use --replay to enable)")
        return
    
    print("Note: Replay requires Genesis environment setup")
    print("This step validates that actions can be replayed in simulation")
    
    # This would require setting up a Genesis environment
    # For demonstration, we just show the code
    print("""
    Example code for replay:
    
    from genesis_maniskill.envs import KitchenEnv
    from genesis_maniskill.datasets import validate_dataset
    
    def env_factory():
        return KitchenEnv(num_envs=1, robot_uid='franka')
    
    report = validate_dataset(env_factory, val_set, max_trajectories=10)
    print(f"Validity rate: {report['validity_rate']:.2%}")
    """)


def step6_train(train_set, val_set, args):
    """Step 6: Train a simple policy."""
    print("\n" + "="*60)
    print("Step 6: Train Policy")
    print("="*60)
    
    if not args.train:
        print("Skipping training (use --train to enable)")
        return
    
    # Convert to PyTorch dataset
    print("Converting to PyTorch dataset...")
    train_torch = train_set.to_torch_dataset(obs_key='state')
    val_torch = val_set.to_torch_dataset(obs_key='state')
    
    train_loader = DataLoader(train_torch, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_torch, batch_size=32)
    
    # Get dimensions
    sample = next(iter(train_loader))
    obs_dim = sample['obs'].shape[1]
    action_dim = sample['action'].shape[1]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create simple policy
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh(),
            )
        
        def forward(self, obs):
            return self.net(obs)
    
    policy = SimplePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        policy.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = batch['obs']
            action_target = batch['action']
            
            action_pred = policy(obs)
            loss = criterion(action_pred, action_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs']
                action_target = batch['action']
                
                action_pred = policy(obs)
                loss = criterion(action_pred, action_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Save model
    output_dir = Path(args.converted_path or "./converted_dataset")
    torch.save(policy.state_dict(), output_dir / "policy.pt")
    print(f"Model saved to: {output_dir / 'policy.pt'}")


def main():
    parser = argparse.ArgumentParser(description='Complete dataset workflow')
    
    # Source arguments
    parser.add_argument('--source-type', choices=['robocasa', 'maniskill'], required=True)
    parser.add_argument('--source-path', required=True, help='Path to source dataset')
    parser.add_argument('--converted-path', help='Path to save/load converted dataset')
    parser.add_argument('--max-demos', type=int, help='Maximum demos to load')
    
    # Workflow steps
    parser.add_argument('--augment', action='store_true', help='Enable augmentation')
    parser.add_argument('--augment-copies', type=int, default=1, help='Number of augmented copies')
    parser.add_argument('--save-splits', action='store_true', help='Save train/val/test splits')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--replay', action='store_true', help='Enable replay validation')
    parser.add_argument('--train', action='store_true', help='Enable training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Run workflow
    dataset = step1_load_and_convert(args)
    dataset = step2_augment(dataset, args)
    train_set, val_set, test_set = step3_split(dataset, args)
    step4_visualize(train_set, args)
    step5_replay(val_set, args)
    step6_train(train_set, val_set, args)
    
    print("\n" + "="*60)
    print("Workflow Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
