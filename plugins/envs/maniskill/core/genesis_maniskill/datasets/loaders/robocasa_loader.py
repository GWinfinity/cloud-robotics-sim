"""
RoboCasa dataset loader.

Loads data from RoboCasa's HDF5 format and converts to unified trajectory format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import h5py
import json

from genesis_maniskill.datasets.formats.trajectory import Trajectory, Step


class RoboCasaLoader:
    """
    Loader for RoboCasa datasets.
    
    RoboCasa stores data in HDF5 format with structure:
    data/
    ├── demo_0/
    │   ├── obs/
    │   │   ├── agentview_image (T, H, W, 3)
    │   │   ├── robot0_eef_pos (T, 3)
    │   │   └── ...
    │   ├── actions (T, action_dim)
    │   ├── rewards (T,)
    │   └── dones (T,)
    ├── demo_1/
    └── ...
    """
    
    # Observation key mapping from RoboCasa to unified format
    OBS_KEY_MAP = {
        # Images
        'agentview_image': 'rgb_agentview',
        'robot0_eye_in_hand_image': 'rgb_hand',
        # Proprioception
        'robot0_eef_pos': 'eef_pos',
        'robot0_eef_quat': 'eef_quat',
        'robot0_joint_pos': 'joint_pos',
        'robot0_joint_vel': 'joint_vel',
        'robot0_gripper_qpos': 'gripper',
    }
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        demo_keys: Optional[List[str]] = None,
        load_images: bool = True,
        load_depth: bool = False,
    ):
        """
        Initialize RoboCasa loader.
        
        Args:
            dataset_path: Path to HDF5 file
            demo_keys: Specific demos to load (None = all)
            load_images: Whether to load image observations
            load_depth: Whether to load depth observations
        """
        self.dataset_path = Path(dataset_path)
        self.demo_keys = demo_keys
        self.load_images = load_images
        self.load_depth = load_depth
        
        # Open HDF5 file
        self.hdf5_file = None
        self.demo_keys_in_file = []
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open the HDF5 file."""
        self.hdf5_file = h5py.File(self.dataset_path, 'r')
        
        # Get available demo keys
        self.demo_keys_in_file = [
            key for key in self.hdf5_file['data'].keys()
            if key.startswith('demo_')
        ]
        
        if self.demo_keys is None:
            self.demo_keys = self.demo_keys_in_file
        else:
            # Filter to available demos
            self.demo_keys = [
                k for k in self.demo_keys if k in self.demo_keys_in_file
            ]
    
    def close(self):
        """Close the HDF5 file."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def get_demo_keys(self) -> List[str]:
        """Get list of available demo keys."""
        if self.hdf5_file is None:
            self.open()
        return self.demo_keys_in_file
    
    def get_demo_info(self, demo_key: str) -> Dict:
        """Get information about a demo."""
        if self.hdf5_file is None:
            self.open()
        
        demo_group = self.hdf5_file['data'][demo_key]
        
        # Get length from actions
        num_steps = demo_group['actions'].shape[0]
        
        # Get metadata
        info = {
            'num_steps': num_steps,
            'demo_key': demo_key,
        }
        
        # Add any stored metadata
        if 'metadata' in demo_group.attrs:
            metadata = json.loads(demo_group.attrs['metadata'])
            info.update(metadata)
        
        return info
    
    def load_demo(self, demo_key: str) -> Trajectory:
        """
        Load a single demo as Trajectory.
        
        Args:
            demo_key: Demo identifier (e.g., 'demo_0')
        
        Returns:
            Trajectory object
        """
        if self.hdf5_file is None:
            self.open()
        
        demo_group = self.hdf5_file['data'][demo_key]
        
        # Load actions
        actions = demo_group['actions'][:]
        
        # Load rewards if available
        if 'rewards' in demo_group:
            rewards = demo_group['rewards'][:]
        else:
            rewards = np.zeros(len(actions))
        
        # Load dones if available
        if 'dones' in demo_group:
            dones = demo_group['dones'][:]
        else:
            dones = np.zeros(len(actions), dtype=bool)
        
        # Load observations
        obs_group = demo_group['obs']
        obs_keys = list(obs_group.keys())
        
        # Build trajectory
        trajectory = Trajectory()
        trajectory.metadata = {
            'source': 'robocasa',
            'demo_key': demo_key,
            'dataset_path': str(self.dataset_path),
        }
        
        # Add task info if available
        if 'task' in demo_group.attrs:
            trajectory.metadata['task'] = demo_group.attrs['task']
        
        # Load each step
        for t in range(len(actions)):
            # Build observation dict
            obs = {}
            for robocasa_key in obs_keys:
                unified_key = self.OBS_KEY_MAP.get(robocasa_key, robocasa_key)
                
                # Skip images if not loading
                if 'image' in robocasa_key and not self.load_images:
                    continue
                if 'depth' in robocasa_key and not self.load_depth:
                    continue
                
                data = obs_group[robocasa_key][t]
                obs[unified_key] = data
            
            # Create step
            step = Step(
                obs=obs,
                action=actions[t],
                reward=float(rewards[t]) if t < len(rewards) else 0.0,
                terminated=bool(dones[t]) if t < len(dones) else False,
                truncated=False,  # RoboCasa doesn't have truncated flag
            )
            
            trajectory.add_step(step)
        
        return trajectory
    
    def load_all(self, max_demos: Optional[int] = None) -> List[Trajectory]:
        """
        Load all demos.
        
        Args:
            max_demos: Maximum number of demos to load
        
        Returns:
            List of Trajectory objects
        """
        if self.hdf5_file is None:
            self.open()
        
        demos_to_load = self.demo_keys
        if max_demos is not None:
            demos_to_load = demos_to_load[:max_demos]
        
        trajectories = []
        for i, demo_key in enumerate(demos_to_load):
            if i % 100 == 0:
                print(f"Loading demo {i+1}/{len(demos_to_load)}: {demo_key}")
            
            traj = self.load_demo(demo_key)
            trajectories.append(traj)
        
        return trajectories
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        if self.hdf5_file is None:
            self.open()
        
        num_demos = len(self.demo_keys_in_file)
        total_steps = 0
        
        for demo_key in self.demo_keys_in_file:
            demo_group = self.hdf5_file['data'][demo_key]
            num_steps = demo_group['actions'].shape[0]
            total_steps += num_steps
        
        # Get observation keys from first demo
        first_demo = self.hdf5_file['data'][self.demo_keys_in_file[0]]
        obs_keys = list(first_demo['obs'].keys())
        
        return {
            'num_demos': num_demos,
            'total_steps': total_steps,
            'avg_steps_per_demo': total_steps / num_demos if num_demos > 0 else 0,
            'observation_keys': obs_keys,
            'action_dim': first_demo['actions'].shape[1] if num_demos > 0 else 0,
        }


def load_robocasa_dataset(
    dataset_path: Union[str, Path],
    max_demos: Optional[int] = None,
    load_images: bool = True,
) -> List[Trajectory]:
    """
    Convenience function to load RoboCasa dataset.
    
    Args:
        dataset_path: Path to HDF5 file
        max_demos: Maximum demos to load
        load_images: Whether to load images
    
    Returns:
        List of trajectories
    """
    with RoboCasaLoader(dataset_path, load_images=load_images) as loader:
        return loader.load_all(max_demos=max_demos)
