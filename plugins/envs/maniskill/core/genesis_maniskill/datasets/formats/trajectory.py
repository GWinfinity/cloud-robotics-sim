"""
Unified trajectory format for Genesis ManiSkill.

This format is designed to be:
- Compatible with both RoboCasa and ManiSkill data
- Efficient for GPU parallel training
- Easy to convert to/from various formats (HDF5, Zarr, TFRecord)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
import h5py
import json
from pathlib import Path


@dataclass
class Step:
    """
    Single step in a trajectory.
    
    Attributes:
        obs: Observation dict (can include 'state', 'rgb', 'depth', etc.)
        action: Action taken
        reward: Reward received
        terminated: Whether episode terminated
        truncated: Whether episode was truncated
        info: Additional info dict
    """
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Step":
        """Create Step from dictionary."""
        return cls(**data)


@dataclass
class Trajectory:
    """
    A complete trajectory (episode).
    
    Attributes:
        steps: List of steps
        metadata: Episode metadata (task, scene, robot, etc.)
    """
    steps: List[Step] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def add_step(self, step: Step):
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def get_observations(self, key: Optional[str] = None) -> np.ndarray:
        """
        Get all observations.
        
        Args:
            key: Observation key (e.g., 'state', 'rgb'). If None, returns all obs.
        
        Returns:
            Stacked observations
        """
        if key is None:
            # Return first observation key
            key = list(self.steps[0].obs.keys())[0]
        
        return np.stack([step.obs[key] for step in self.steps])
    
    def get_actions(self) -> np.ndarray:
        """Get all actions."""
        return np.stack([step.action for step in self.steps])
    
    def get_rewards(self) -> np.ndarray:
        """Get all rewards."""
        return np.array([step.reward for step in self.steps])
    
    def get_dones(self) -> np.ndarray:
        """Get done flags (terminated or truncated)."""
        return np.array([
            step.terminated or step.truncated for step in self.steps
        ])
    
    def to_dict(self) -> Dict:
        """Convert trajectory to dictionary."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Trajectory":
        """Create Trajectory from dictionary."""
        steps = [Step.from_dict(s) for s in data["steps"]]
        return cls(steps=steps, metadata=data.get("metadata", {}))
    
    def save_hdf5(self, path: Union[str, Path]):
        """Save trajectory to HDF5 file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['metadata'] = json.dumps(self.metadata)
            
            # Save steps
            for i, step in enumerate(self.steps):
                step_group = f.create_group(f'step_{i:06d}')
                
                # Save observations
                obs_group = step_group.create_group('obs')
                for key, value in step.obs.items():
                    obs_group.create_dataset(key, data=value)
                
                # Save action
                step_group.create_dataset('action', data=step.action)
                
                # Save reward
                step_group.attrs['reward'] = step.reward
                step_group.attrs['terminated'] = step.terminated
                step_group.attrs['truncated'] = step.truncated
                
                # Save info
                if step.info:
                    step_group.attrs['info'] = json.dumps(step.info)
    
    @classmethod
    def load_hdf5(cls, path: Union[str, Path]) -> "Trajectory":
        """Load trajectory from HDF5 file."""
        path = Path(path)
        
        with h5py.File(path, 'r') as f:
            # Load metadata
            metadata = json.loads(f.attrs['metadata'])
            
            # Load steps
            steps = []
            for key in sorted(f.keys()):
                if key.startswith('step_'):
                    step_group = f[key]
                    
                    # Load observations
                    obs = {}
                    if 'obs' in step_group:
                        for obs_key in step_group['obs'].keys():
                            obs[obs_key] = step_group['obs'][obs_key][:]
                    
                    # Load action
                    action = step_group['action'][:]
                    
                    # Load reward and done flags
                    reward = step_group.attrs['reward']
                    terminated = step_group.attrs['terminated']
                    truncated = step_group.attrs['truncated']
                    
                    # Load info
                    info = {}
                    if 'info' in step_group.attrs:
                        info = json.loads(step_group.attrs['info'])
                    
                    steps.append(Step(
                        obs=obs,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info
                    ))
            
            return cls(steps=steps, metadata=metadata)


class TrajectoryDataset:
    """
    Dataset of trajectories.
    
    Supports:
    - Loading from RoboCasa format
    - Loading from ManiSkill format
    - Saving/loading unified format
    - Conversion to PyTorch DataLoader
    """
    
    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        self.trajectories = trajectories or []
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the dataset."""
        self.trajectories.append(trajectory)
    
    def filter_by_metadata(self, key: str, value: Any) -> "TrajectoryDataset":
        """Filter trajectories by metadata."""
        filtered = [
            t for t in self.trajectories
            if t.metadata.get(key) == value
        ]
        return TrajectoryDataset(filtered)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_episodes = len(self.trajectories)
        total_steps = sum(len(t) for t in self.trajectories)
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0
        
        # Get unique tasks
        tasks = set()
        for t in self.trajectories:
            task = t.metadata.get('task')
            if task:
                tasks.add(task)
        
        return {
            'num_episodes': num_episodes,
            'total_steps': total_steps,
            'avg_steps_per_episode': avg_steps,
            'tasks': list(tasks),
        }
    
    def save(self, path: Union[str, Path], format: str = 'hdf5'):
        """
        Save dataset.
        
        Args:
            path: Save path
            format: 'hdf5' or 'zarr'
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            # Save each trajectory as separate file
            for i, traj in enumerate(self.trajectories):
                traj_path = path / f'trajectory_{i:06d}.hdf5'
                traj.save_hdf5(traj_path)
            
            # Save metadata
            metadata = {
                'num_trajectories': len(self.trajectories),
                'statistics': self.get_statistics(),
            }
            with open(path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        elif format == 'zarr':
            # TODO: Implement Zarr format
            raise NotImplementedError("Zarr format not yet implemented")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = 'hdf5') -> "TrajectoryDataset":
        """Load dataset."""
        path = Path(path)
        
        if format == 'hdf5':
            trajectories = []
            for traj_file in sorted(path.glob('trajectory_*.hdf5')):
                traj = Trajectory.load_hdf5(traj_file)
                trajectories.append(traj)
            
            return cls(trajectories)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def to_torch_dataset(self, obs_key: str = 'state'):
        """Convert to PyTorch dataset for training."""
        from torch.utils.data import Dataset
        
        class TorchTrajectoryDataset(Dataset):
            def __init__(self, trajectories, obs_key):
                self.trajectories = trajectories
                self.obs_key = obs_key
                
                # Flatten all steps
                self.steps = []
                for traj in trajectories:
                    for step in traj.steps:
                        self.steps.append((
                            step.obs.get(obs_key, np.zeros(0)),
                            step.action,
                            step.reward,
                        ))
            
            def __len__(self):
                return len(self.steps)
            
            def __getitem__(self, idx):
                obs, action, reward = self.steps[idx]
                return {
                    'obs': torch.from_numpy(obs).float(),
                    'action': torch.from_numpy(action).float(),
                    'reward': torch.tensor(reward).float(),
                }
        
        return TorchTrajectoryDataset(self.trajectories, obs_key)
