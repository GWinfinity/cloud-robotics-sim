"""
ManiSkill dataset loader.

Loads data from ManiSkill's format (HDF5 or pickled trajectories) and converts to unified format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import h5py
import pickle
import json

from genesis_maniskill.datasets.formats.trajectory import Trajectory, Step


class ManiSkillLoader:
    """
    Loader for ManiSkill datasets.
    
    ManiSkill stores data in HDF5 format with structure:
    trajectories/
    ├── trajectory_0.h5
    │   ├── observations/
    │   │   ├── agent/qpos (T, N)
    │   │   └── ...
    │   ├── actions (T, action_dim)
    │   ├── rewards (T,)
    │   ├── terminated (T,)
    │   └── truncated (T,)
    └── ...
    
    Or as trajectory pickles in replay buffer format.
    """
    
    # Observation key mapping from ManiSkill to unified format
    OBS_KEY_MAP = {
        # State observations
        'agent/qpos': 'joint_pos',
        'agent/qvel': 'joint_vel',
        'agent/tcp_pose': 'eef_pose',
        'extra/tcp_pose': 'eef_pose',
        # Images
        'sensor/camera/hand_camera/rgb': 'rgb_hand',
        'sensor/camera/base_camera/rgb': 'rgb_base',
        'sensor/camera/hand_camera/depth': 'depth_hand',
        'sensor/camera/base_camera/depth': 'depth_base',
        # Segmentation
        'sensor/camera/hand_camera/segmentation': 'seg_hand',
        'sensor/camera/base_camera/segmentation': 'seg_base',
    }
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        obs_mode: str = 'state',
        control_mode: str = 'pd_joint_pos',
        load_images: bool = True,
        load_depth: bool = False,
    ):
        """
        Initialize ManiSkill loader.
        
        Args:
            dataset_path: Path to dataset (file or directory)
            obs_mode: Observation mode used when collecting
            control_mode: Control mode used when collecting
            load_images: Whether to load image observations
            load_depth: Whether to load depth observations
        """
        self.dataset_path = Path(dataset_path)
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.load_images = load_images
        self.load_depth = load_depth
        
        # Detect format
        self.format = self._detect_format()
        
    def _detect_format(self) -> str:
        """Detect dataset format."""
        if self.dataset_path.is_file():
            if self.dataset_path.suffix == '.h5' or self.dataset_path.suffix == '.hdf5':
                return 'hdf5_single'
            elif self.dataset_path.suffix == '.pkl' or self.dataset_path.suffix == '.pickle':
                return 'pickle'
        elif self.dataset_path.is_dir():
            # Check for trajectory files
            h5_files = list(self.dataset_path.glob('*.h5'))
            if len(h5_files) > 0:
                return 'hdf5_dir'
            
            pkl_files = list(self.dataset_path.glob('*.pkl'))
            if len(pkl_files) > 0:
                return 'pickle_dir'
        
        raise ValueError(f"Could not detect format for {self.dataset_path}")
    
    def get_trajectory_files(self) -> List[Path]:
        """Get list of trajectory files."""
        if self.format == 'hdf5_single':
            return [self.dataset_path]
        elif self.format == 'hdf5_dir':
            return sorted(self.dataset_path.glob('*.h5'))
        elif self.format == 'pickle':
            return [self.dataset_path]
        elif self.format == 'pickle_dir':
            return sorted(self.dataset_path.glob('*.pkl'))
        else:
            return []
    
    def load_trajectory(self, traj_path: Path) -> Trajectory:
        """Load a single trajectory."""
        if traj_path.suffix in ['.h5', '.hdf5']:
            return self._load_hdf5_trajectory(traj_path)
        elif traj_path.suffix in ['.pkl', '.pickle']:
            return self._load_pickle_trajectory(traj_path)
        else:
            raise ValueError(f"Unknown file format: {traj_path}")
    
    def _load_hdf5_trajectory(self, traj_path: Path) -> Trajectory:
        """Load trajectory from HDF5."""
        with h5py.File(traj_path, 'r') as f:
            # Load actions
            actions = f['actions'][:]
            
            # Load rewards
            if 'rewards' in f:
                rewards = f['rewards'][:]
            else:
                rewards = np.zeros(len(actions))
            
            # Load terminated/truncated
            if 'terminated' in f:
                terminated = f['terminated'][:]
            else:
                terminated = np.zeros(len(actions), dtype=bool)
            
            if 'truncated' in f:
                truncated = f['truncated'][:]
            else:
                truncated = np.zeros(len(actions), dtype=bool)
            
            # Load observations
            obs_group = f['observations']
            obs_keys = list(obs_group.keys())
            
            # Build trajectory
            trajectory = Trajectory()
            trajectory.metadata = {
                'source': 'maniskill',
                'dataset_path': str(traj_path),
                'obs_mode': self.obs_mode,
                'control_mode': self.control_mode,
            }
            
            # Load each step
            for t in range(len(actions)):
                obs = {}
                
                # Load each observation key
                for ms_key in obs_keys:
                    unified_key = self.OBS_KEY_MAP.get(ms_key, ms_key)
                    
                    # Skip images if not loading
                    if 'rgb' in ms_key and not self.load_images:
                        continue
                    if 'depth' in ms_key and not self.load_depth:
                        continue
                    
                    # Handle different observation structures
                    if isinstance(obs_group[ms_key], h5py.Group):
                        # Nested structure (e.g., sensor/camera/...)
                        for sub_key in obs_group[ms_key].keys():
                            full_key = f"{unified_key}/{sub_key}"
                            obs[full_key] = obs_group[ms_key][sub_key][t]
                    else:
                        obs[unified_key] = obs_group[ms_key][t]
                
                step = Step(
                    obs=obs,
                    action=actions[t],
                    reward=float(rewards[t]) if t < len(rewards) else 0.0,
                    terminated=bool(terminated[t]) if t < len(terminated) else False,
                    truncated=bool(truncated[t]) if t < len(truncated) else False,
                )
                
                trajectory.add_step(step)
        
        return trajectory
    
    def _load_pickle_trajectory(self, traj_path: Path) -> Trajectory:
        """Load trajectory from pickle."""
        with open(traj_path, 'rb') as f:
            data = pickle.load(f)
        
        # ManiSkill pickle format varies, handle common cases
        trajectory = Trajectory()
        trajectory.metadata = {
            'source': 'maniskill',
            'dataset_path': str(traj_path),
        }
        
        if isinstance(data, dict):
            # Standard dict format
            observations = data.get('observations', [])
            actions = data.get('actions', [])
            rewards = data.get('rewards', [])
            terminated = data.get('terminated', [])
            truncated = data.get('truncated', [])
            
            for t in range(len(actions)):
                obs = observations[t] if t < len(observations) else {}
                # Convert observation keys
                unified_obs = {}
                for k, v in obs.items():
                    unified_key = self.OBS_KEY_MAP.get(k, k)
                    unified_obs[unified_key] = v
                
                step = Step(
                    obs=unified_obs,
                    action=actions[t],
                    reward=float(rewards[t]) if t < len(rewards) else 0.0,
                    terminated=bool(terminated[t]) if t < len(terminated) else False,
                    truncated=bool(truncated[t]) if t < len(truncated) else False,
                )
                trajectory.add_step(step)
        
        elif isinstance(data, list):
            # List of step dicts
            for step_data in data:
                obs = step_data.get('obs', {})
                unified_obs = {}
                for k, v in obs.items():
                    unified_key = self.OBS_KEY_MAP.get(k, k)
                    unified_obs[unified_key] = v
                
                step = Step(
                    obs=unified_obs,
                    action=step_data.get('action', np.zeros(0)),
                    reward=step_data.get('reward', 0.0),
                    terminated=step_data.get('terminated', False),
                    truncated=step_data.get('truncated', False),
                )
                trajectory.add_step(step)
        
        return trajectory
    
    def load_all(self, max_trajs: Optional[int] = None) -> List[Trajectory]:
        """
        Load all trajectories.
        
        Args:
            max_trajs: Maximum number of trajectories to load
        
        Returns:
            List of Trajectory objects
        """
        traj_files = self.get_trajectory_files()
        
        if max_trajs is not None:
            traj_files = traj_files[:max_trajs]
        
        trajectories = []
        for i, traj_path in enumerate(traj_files):
            if i % 100 == 0:
                print(f"Loading trajectory {i+1}/{len(traj_files)}: {traj_path.name}")
            
            traj = self.load_trajectory(traj_path)
            trajectories.append(traj)
        
        return trajectories
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        traj_files = self.get_trajectory_files()
        
        if len(traj_files) == 0:
            return {
                'num_trajectories': 0,
                'total_steps': 0,
            }
        
        # Load first trajectory to get shapes
        first_traj = self.load_trajectory(traj_files[0])
        
        # Count total steps across all files (expensive, sample a few)
        sample_size = min(10, len(traj_files))
        total_steps = 0
        for i in range(sample_size):
            traj = self.load_trajectory(traj_files[i])
            total_steps += len(traj)
        
        avg_steps = total_steps / sample_size
        estimated_total = avg_steps * len(traj_files)
        
        return {
            'num_trajectories': len(traj_files),
            'estimated_total_steps': int(estimated_total),
            'avg_steps_per_trajectory': avg_steps,
            'observation_keys': list(first_traj.steps[0].obs.keys()) if first_traj.steps else [],
            'action_dim': len(first_traj.steps[0].action) if first_traj.steps else 0,
            'format': self.format,
        }


def load_maniskill_dataset(
    dataset_path: Union[str, Path],
    max_trajs: Optional[int] = None,
    load_images: bool = True,
) -> List[Trajectory]:
    """
    Convenience function to load ManiSkill dataset.
    
    Args:
        dataset_path: Path to dataset
        max_trajs: Maximum trajectories to load
        load_images: Whether to load images
    
    Returns:
        List of trajectories
    """
    loader = ManiSkillLoader(dataset_path, load_images=load_images)
    return loader.load_all(max_trajs=max_trajs)
