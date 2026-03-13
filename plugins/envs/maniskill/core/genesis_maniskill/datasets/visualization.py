"""
Visualization tools for trajectory datasets.

Provides tools to visualize and debug trajectory data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from genesis_maniskill.datasets.formats.trajectory import Trajectory, TrajectoryDataset


class TrajectoryVisualizer:
    """
    Visualize trajectory data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
    
    def plot_trajectory(
        self,
        trajectory: Trajectory,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot a single trajectory with multiple subplots.
        
        Args:
            trajectory: Trajectory to plot
            save_path: Path to save figure
            show: Whether to show the plot
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 2, figure=fig)
        
        # Get data
        actions = trajectory.get_actions()
        rewards = trajectory.get_rewards()
        
        # Plot actions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_actions(ax1, actions)
        ax1.set_title('Actions over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Action Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot rewards
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(rewards, 'g-', linewidth=2)
        ax2.set_title('Rewards')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # Plot cumulative reward
        ax3 = fig.add_subplot(gs[1, 1])
        cumulative = np.cumsum(rewards)
        ax3.plot(cumulative, 'b-', linewidth=2)
        ax3.set_title('Cumulative Reward')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cumulative Reward')
        ax3.grid(True, alpha=0.3)
        
        # Plot state if available
        ax4 = fig.add_subplot(gs[2, :])
        if 'state' in trajectory.steps[0].obs:
            states = np.stack([s.obs['state'] for s in trajectory.steps])
            self._plot_states(ax4, states)
            ax4.set_title('State Trajectory (First 10 dims)')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('State Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No state observation available',
                    ha='center', va='center', transform=ax4.transAxes)
        
        # Add metadata as text
        metadata_text = f"Length: {len(trajectory)} steps\n"
        metadata_text += f"Total reward: {rewards.sum():.2f}\n"
        if trajectory.metadata:
            metadata_text += "\nMetadata:\n"
            for key, value in list(trajectory.metadata.items())[:5]:
                metadata_text += f"  {key}: {value}\n"
        
        fig.text(0.02, 0.98, metadata_text,
                transform=fig.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_actions(self, ax, actions: np.ndarray):
        """Plot action trajectories."""
        n_dims = min(actions.shape[1], 10)  # Plot first 10 dims
        
        for i in range(n_dims):
            ax.plot(actions[:, i], label=f'dim_{i}', alpha=0.7)
    
    def _plot_states(self, ax, states: np.ndarray):
        """Plot state trajectories."""
        n_dims = min(states.shape[1], 10)  # Plot first 10 dims
        
        for i in range(n_dims):
            ax.plot(states[:, i], label=f'dim_{i}', alpha=0.7)
    
    def plot_dataset_statistics(
        self,
        dataset: TrajectoryDataset,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot dataset statistics.
        
        Args:
            dataset: Dataset to analyze
            save_path: Path to save figure
            show: Whether to show the plot
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Get statistics
        stats = dataset.get_statistics()
        
        # Episode lengths
        lengths = [len(t) for t in dataset.trajectories]
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Episode Length Distribution')
        ax1.set_xlabel('Length (steps)')
        ax1.set_ylabel('Count')
        ax1.axvline(np.mean(lengths), color='r', linestyle='--',
                   label=f'Mean: {np.mean(lengths):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total reward distribution
        total_rewards = [t.get_rewards().sum() for t in dataset.trajectories]
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(total_rewards, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax2.set_title('Total Reward Distribution')
        ax2.set_xlabel('Total Reward')
        ax2.set_ylabel('Count')
        ax2.axvline(np.mean(total_rewards), color='r', linestyle='--',
                   label=f'Mean: {np.mean(total_rewards):.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Action statistics
        all_actions = np.concatenate([t.get_actions() for t in dataset.trajectories])
        action_dim = all_actions.shape[1]
        
        ax3 = fig.add_subplot(gs[0, 2])
        action_means = np.mean(all_actions, axis=0)
        action_stds = np.std(all_actions, axis=0)
        x = np.arange(min(action_dim, 10))
        ax3.bar(x, action_means[:10], yerr=action_stds[:10],
               capsize=5, alpha=0.7, edgecolor='black')
        ax3.set_title('Action Mean ± Std (First 10 dims)')
        ax3.set_xlabel('Action Dimension')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # Reward over time (average across episodes)
        ax4 = fig.add_subplot(gs[1, :])
        max_len = max(lengths)
        reward_curves = []
        
        for traj in dataset.trajectories:
            rewards = traj.get_rewards()
            # Pad to max length
            padded = np.pad(rewards, (0, max_len - len(rewards)),
                          mode='edge')
            reward_curves.append(padded)
        
        reward_curves = np.array(reward_curves)
        mean_rewards = np.mean(reward_curves, axis=0)
        std_rewards = np.std(reward_curves, axis=0)
        
        steps = np.arange(max_len)
        ax4.plot(steps, mean_rewards, 'b-', linewidth=2, label='Mean')
        ax4.fill_between(steps,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.3, label='±1 Std')
        ax4.set_title('Average Reward over Episode Steps')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text = f"""Dataset Summary:
Episodes: {stats['num_episodes']}
Total steps: {stats['total_steps']}
Avg length: {stats['avg_steps_per_episode']:.1f} steps
Tasks: {', '.join(stats['tasks']) if stats['tasks'] else 'N/A'}
"""
        
        fig.text(0.02, 0.98, summary_text,
                transform=fig.transAxes,
                verticalalignment='top',
                fontsize=10,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_trajectories(
        self,
        trajectories: List[Trajectory],
        labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Compare multiple trajectories side by side.
        
        Args:
            trajectories: List of trajectories to compare
            labels: Labels for each trajectory
            save_path: Path to save figure
            show: Whether to show the plot
        """
        n_trajs = len(trajectories)
        fig, axes = plt.subplots(2, n_trajs, figsize=(6*n_trajs, 10))
        
        if n_trajs == 1:
            axes = axes.reshape(2, 1)
        
        for i, traj in enumerate(trajectories):
            label = labels[i] if labels else f"Traj {i+1}"
            
            # Plot actions
            ax = axes[0, i]
            actions = traj.get_actions()
            for j in range(min(actions.shape[1], 5)):
                ax.plot(actions[:, j], label=f'dim_{j}', alpha=0.7)
            ax.set_title(f'{label}\nActions')
            ax.set_xlabel('Step')
            ax.set_ylabel('Action Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Plot rewards
            ax = axes[1, i]
            rewards = traj.get_rewards()
            ax.plot(rewards, 'g-', linewidth=2)
            ax.set_title('Rewards')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            
            # Add stats
            stats_text = f"Length: {len(traj)}\nTotal reward: {rewards.sum():.2f}"
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def visualize_trajectory(
    trajectory: Trajectory,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Convenience function to visualize a trajectory."""
    visualizer = TrajectoryVisualizer()
    visualizer.plot_trajectory(
        trajectory,
        save_path=Path(save_path) if save_path else None,
        show=show
    )


def visualize_dataset(
    dataset: TrajectoryDataset,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Convenience function to visualize dataset statistics."""
    visualizer = TrajectoryVisualizer()
    visualizer.plot_dataset_statistics(
        dataset,
        save_path=Path(save_path) if save_path else None,
        show=show
    )
