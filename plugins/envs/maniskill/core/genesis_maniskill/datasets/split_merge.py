"""
Dataset splitting and merging utilities.

Provides tools to:
- Split datasets into train/val/test sets
- Merge multiple datasets
- Balance datasets by task
- Filter datasets
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import copy

from genesis_maniskill.datasets.formats.trajectory import Trajectory, TrajectoryDataset


class DatasetSplitter:
    """
    Split dataset into train/val/test sets.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def random_split(
        self,
        dataset: TrajectoryDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        """
        Randomly split dataset into train/val/test.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        
        Returns:
            (train_set, val_set, test_set)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n = len(dataset)
        indices = np.arange(n)
        self.rng.shuffle(indices)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_set = TrajectoryDataset([dataset[i] for i in train_indices])
        val_set = TrajectoryDataset([dataset[i] for i in val_indices])
        test_set = TrajectoryDataset([dataset[i] for i in test_indices])
        
        return train_set, val_set, test_set
    
    def stratified_split(
        self,
        dataset: TrajectoryDataset,
        stratify_key: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        """
        Split dataset while maintaining class distribution.
        
        Args:
            dataset: Dataset to split
            stratify_key: Metadata key to stratify by (e.g., 'task_type')
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        
        Returns:
            (train_set, val_set, test_set)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Group by stratify key
        groups: Dict[str, List[int]] = {}
        for i, traj in enumerate(dataset.trajectories):
            key = str(traj.metadata.get(stratify_key, 'unknown'))
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for key, indices in groups.items():
            indices = np.array(indices)
            self.rng.shuffle(indices)
            
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        train_set = TrajectoryDataset([dataset[i] for i in train_indices])
        val_set = TrajectoryDataset([dataset[i] for i in val_indices])
        test_set = TrajectoryDataset([dataset[i] for i in test_indices])
        
        return train_set, val_set, test_set
    
    def k_fold_split(
        self,
        dataset: TrajectoryDataset,
        k: int = 5,
    ) -> List[Tuple[TrajectoryDataset, TrajectoryDataset]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            dataset: Dataset to split
            k: Number of folds
        
        Returns:
            List of (train_set, val_set) tuples
        """
        n = len(dataset)
        indices = np.arange(n)
        self.rng.shuffle(indices)
        
        fold_size = n // k
        folds = []
        
        for i in range(k):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k - 1 else n
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([
                indices[:val_start],
                indices[val_end:]
            ])
            
            train_set = TrajectoryDataset([dataset[int(i)] for i in train_indices])
            val_set = TrajectoryDataset([dataset[int(i)] for i in val_indices])
            
            folds.append((train_set, val_set))
        
        return folds
    
    def temporal_split(
        self,
        dataset: TrajectoryDataset,
        train_ratio: float = 0.8,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset]:
        """
        Split dataset temporally (first N for train, rest for test).
        
        Useful for evaluating on future episodes.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
        
        Returns:
            (train_set, test_set)
        """
        n_train = int(len(dataset) * train_ratio)
        
        train_set = TrajectoryDataset(dataset.trajectories[:n_train])
        test_set = TrajectoryDataset(dataset.trajectories[n_train:])
        
        return train_set, test_set


class DatasetMerger:
    """
    Merge multiple datasets.
    """
    
    @staticmethod
    def merge(
        datasets: List[TrajectoryDataset],
        deduplicate: bool = False,
    ) -> TrajectoryDataset:
        """
        Merge multiple datasets into one.
        
        Args:
            datasets: List of datasets to merge
            deduplicate: Whether to remove duplicate trajectories
        
        Returns:
            Merged dataset
        """
        all_trajectories = []
        
        for dataset in datasets:
            all_trajectories.extend(dataset.trajectories)
        
        if deduplicate:
            # Remove duplicates based on length and first/last states
            seen = set()
            unique_trajectories = []
            
            for traj in all_trajectories:
                # Create a fingerprint
                first_obs = tuple(traj.steps[0].action.flatten()[:5])
                last_obs = tuple(traj.steps[-1].action.flatten()[:5])
                fingerprint = (len(traj), first_obs, last_obs)
                
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    unique_trajectories.append(traj)
            
            all_trajectories = unique_trajectories
        
        return TrajectoryDataset(all_trajectories)
    
    @staticmethod
    def merge_by_task(
        datasets: List[TrajectoryDataset],
        task_weights: Optional[Dict[str, float]] = None,
    ) -> TrajectoryDataset:
        """
        Merge datasets with optional task weighting.
        
        Args:
            datasets: List of datasets to merge
            task_weights: Optional weights for each task (e.g., {'pick_place': 2.0})
        
        Returns:
            Merged dataset
        """
        all_trajectories = []
        
        for dataset in datasets:
            for traj in dataset.trajectories:
                task = traj.metadata.get('task_type', 'unknown')
                weight = task_weights.get(task, 1.0) if task_weights else 1.0
                
                # Add weighted copies
                n_copies = int(weight)
                remainder = weight - n_copies
                
                for _ in range(n_copies):
                    all_trajectories.append(copy.deepcopy(traj))
                
                if self.rng.random() < remainder:
                    all_trajectories.append(copy.deepcopy(traj))
        
        return TrajectoryDataset(all_trajectories)


class DatasetBalancer:
    """
    Balance dataset by task or other criteria.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def balance_by_task(
        self,
        dataset: TrajectoryDataset,
        strategy: str = 'oversample',
    ) -> TrajectoryDataset:
        """
        Balance dataset so each task has equal representation.
        
        Args:
            dataset: Dataset to balance
            strategy: 'oversample' or 'undersample'
        
        Returns:
            Balanced dataset
        """
        # Group by task
        task_groups: Dict[str, List[Trajectory]] = {}
        for traj in dataset.trajectories:
            task = str(traj.metadata.get('task_type', 'unknown'))
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(traj)
        
        # Find target count
        task_counts = {task: len(trajs) for task, trajs in task_groups.items()}
        
        if strategy == 'oversample':
            target_count = max(task_counts.values())
        elif strategy == 'undersample':
            target_count = min(task_counts.values())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Balance each task
        balanced_trajectories = []
        for task, trajs in task_groups.items():
            if strategy == 'oversample':
                # Oversample with replacement
                indices = self.rng.choice(len(trajs), target_count, replace=True)
                balanced_trajectories.extend([copy.deepcopy(trajs[i]) for i in indices])
            else:  # undersample
                # Randomly sample without replacement
                indices = self.rng.choice(len(trajs), target_count, replace=False)
                balanced_trajectories.extend([trajs[i] for i in indices])
        
        return TrajectoryDataset(balanced_trajectories)
    
    def balance_by_length(
        self,
        dataset: TrajectoryDataset,
        n_bins: int = 5,
        strategy: str = 'oversample',
    ) -> TrajectoryDataset:
        """
        Balance dataset by trajectory length.
        
        Args:
            dataset: Dataset to balance
            n_bins: Number of length bins
            strategy: 'oversample' or 'undersample'
        
        Returns:
            Balanced dataset
        """
        # Get lengths and create bins
        lengths = np.array([len(t) for t in dataset.trajectories])
        bins = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
        
        # Group by bin
        bin_groups: Dict[int, List[int]] = {i: [] for i in range(n_bins)}
        for i, length in enumerate(lengths):
            bin_idx = min(int((length - bins[0]) / (bins[1] - bins[0])), n_bins - 1)
            bin_groups[bin_idx].append(i)
        
        # Find target count
        bin_counts = {bin_idx: len(indices) for bin_idx, indices in bin_groups.items()}
        
        if strategy == 'oversample':
            target_count = max(bin_counts.values())
        else:
            target_count = min(bin_counts.values())
        
        # Balance
        balanced_indices = []
        for bin_idx, indices in bin_groups.items():
            if strategy == 'oversample':
                sampled = self.rng.choice(indices, target_count, replace=True)
            else:
                sampled = self.rng.choice(indices, target_count, replace=False)
            balanced_indices.extend(sampled)
        
        if strategy == 'oversample':
            # Deep copy for oversampled
            balanced_trajectories = [
                copy.deepcopy(dataset.trajectories[i]) for i in balanced_indices
            ]
        else:
            balanced_trajectories = [dataset.trajectories[i] for i in balanced_indices]
        
        return TrajectoryDataset(balanced_trajectories)


class DatasetFilter:
    """
    Filter dataset based on various criteria.
    """
    
    @staticmethod
    def filter_by_length(
        dataset: TrajectoryDataset,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> TrajectoryDataset:
        """
        Filter trajectories by length.
        
        Args:
            dataset: Dataset to filter
            min_length: Minimum trajectory length
            max_length: Maximum trajectory length
        
        Returns:
            Filtered dataset
        """
        filtered = []
        for traj in dataset.trajectories:
            length = len(traj)
            if min_length is not None and length < min_length:
                continue
            if max_length is not None and length > max_length:
                continue
            filtered.append(traj)
        
        return TrajectoryDataset(filtered)
    
    @staticmethod
    def filter_by_reward(
        dataset: TrajectoryDataset,
        min_total_reward: Optional[float] = None,
        max_total_reward: Optional[float] = None,
    ) -> TrajectoryDataset:
        """
        Filter trajectories by total reward.
        
        Args:
            dataset: Dataset to filter
            min_total_reward: Minimum total reward
            max_total_reward: Maximum total reward
        
        Returns:
            Filtered dataset
        """
        filtered = []
        for traj in dataset.trajectories:
            total_reward = traj.get_rewards().sum()
            if min_total_reward is not None and total_reward < min_total_reward:
                continue
            if max_total_reward is not None and total_reward > max_total_reward:
                continue
            filtered.append(traj)
        
        return TrajectoryDataset(filtered)
    
    @staticmethod
    def filter_by_task(
        dataset: TrajectoryDataset,
        tasks: List[str],
    ) -> TrajectoryDataset:
        """
        Filter trajectories by task type.
        
        Args:
            dataset: Dataset to filter
            tasks: List of task types to keep
        
        Returns:
            Filtered dataset
        """
        filtered = []
        for traj in dataset.trajectories:
            task = traj.metadata.get('task_type', 'unknown')
            if task in tasks:
                filtered.append(traj)
        
        return TrajectoryDataset(filtered)
    
    @staticmethod
    def filter_by_success(
        dataset: TrajectoryDataset,
        success_threshold: float = 0.8,
    ) -> TrajectoryDataset:
        """
        Filter trajectories by success rate.
        
        Args:
            dataset: Dataset to filter
            success_threshold: Minimum success rate (0-1)
        
        Returns:
            Filtered dataset with only successful trajectories
        """
        filtered = []
        for traj in dataset.trajectories:
            # Check if final reward indicates success
            final_reward = traj.get_rewards()[-1]
            # Or check if terminated with success
            if final_reward >= success_threshold or traj.steps[-1].terminated:
                filtered.append(traj)
        
        return TrajectoryDataset(filtered)
    
    @staticmethod
    def remove_outliers(
        dataset: TrajectoryDataset,
        z_threshold: float = 3.0,
    ) -> TrajectoryDataset:
        """
        Remove outlier trajectories based on action statistics.
        
        Args:
            dataset: Dataset to filter
            z_threshold: Z-score threshold for outliers
        
        Returns:
            Filtered dataset
        """
        # Compute action statistics
        all_actions = np.concatenate([t.get_actions() for t in dataset.trajectories])
        mean_action = np.mean(all_actions, axis=0)
        std_action = np.std(all_actions, axis=0)
        std_action = np.where(std_action == 0, 1, std_action)
        
        filtered = []
        for traj in dataset.trajectories:
            actions = traj.get_actions()
            # Check for any outlier actions
            z_scores = np.abs((actions - mean_action) / std_action)
            max_z = np.max(z_scores)
            
            if max_z < z_threshold:
                filtered.append(traj)
        
        return TrajectoryDataset(filtered)


# ==========================================
# Convenience Functions
# ==========================================

def split_dataset(
    dataset: TrajectoryDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
    """Convenience function for random split."""
    splitter = DatasetSplitter(seed=seed)
    return splitter.random_split(dataset, train_ratio, val_ratio, test_ratio)


def merge_datasets(datasets: List[TrajectoryDataset]) -> TrajectoryDataset:
    """Convenience function for merging datasets."""
    return DatasetMerger.merge(datasets)


def filter_by_task(dataset: TrajectoryDataset, tasks: List[str]) -> TrajectoryDataset:
    """Convenience function for filtering by task."""
    return DatasetFilter.filter_by_task(dataset, tasks)
