"""
Example: Train a policy using converted dataset.

This example shows how to:
1. Load converted dataset
2. Create a PyTorch DataLoader
3. Train a simple policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from genesis_maniskill.datasets.formats import TrajectoryDataset


class SimplePolicy(nn.Module):
    """Simple MLP policy."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)


def train_policy(
    dataset_path: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """Train policy on converted dataset."""
    
    print("Loading dataset...")
    dataset = TrajectoryDataset.load(dataset_path)
    print(f"Loaded {len(dataset)} trajectories")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Tasks: {stats['tasks']}")
    
    # Convert to PyTorch dataset
    print("\nConverting to PyTorch dataset...")
    torch_dataset = dataset.to_torch_dataset(obs_key="state")
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set >0 for parallel loading
    )
    
    # Get dimensions from first batch
    sample_batch = next(iter(dataloader))
    obs_dim = sample_batch['obs'].shape[1]
    action_dim = sample_batch['action'].shape[1]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    
    # Create model
    print("\nCreating model...")
    policy = SimplePolicy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            obs = batch['obs']
            action_target = batch['action']
            
            # Forward pass
            action_pred = policy(obs)
            loss = criterion(action_pred, action_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print("\nTraining complete!")
    
    # Save model
    torch.save(policy.state_dict(), "trained_policy.pt")
    print("Model saved to: trained_policy.pt")
    
    return policy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train policy on converted dataset'
    )
    parser.add_argument(
        'dataset_path',
        help='Path to converted dataset'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', '-l',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    train_policy(
        dataset_path=args.dataset_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == '__main__':
    main()
