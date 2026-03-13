"""
LeRobot dataset loader.

Loads data from LeRobot format (HuggingFace datasets / Parquet).
LeRobot format is becoming a standard for robot learning.

Reference: https://github.com/huggingface/lerobot
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from genesis_maniskill.datasets.formats.trajectory import Trajectory, Step


class LeRobotLoader:
    """
    Loader for LeRobot datasets.
    
    LeRobot stores data in:
    - data/ directory with Parquet files
    - videos/ directory with video files (if using video backend)
    - meta/ directory with metadata
    
    Each episode is a sequence of steps with observations and actions.
    """
    
    # Observation key mapping from LeRobot to unified format
    OBS_KEY_MAP = {
        # Images
        'observation.image': 'rgb_agentview',
        'observation.wrist_image': 'rgb_hand',
        'observation.left_image': 'rgb_left',
        'observation.right_image': 'rgb_right',
        # State
        'observation.state': 'state',
        'observation.position': 'joint_pos',
        'observation.velocity': 'joint_vel',
        'observation.effort': 'joint_effort',
        # End-effector
        'observation.ee_pos': 'eef_pos',
        'observation.ee_quat': 'eef_quat',
        # Gripper
        'observation.gripper': 'gripper',
        'observation.gripper_position': 'gripper',
    }
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        delta_timestamps: Optional[List[float]] = None,
    ):
        """
        Initialize LeRobot loader.
        
        Args:
            dataset_path: Path to LeRobot dataset root
            delta_timestamps: List of relative timestamps for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        self.delta_timestamps = delta_timestamps or [0.0]
        
        # Try to import LeRobot
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            self.lerobot_available = True
        except ImportError:
            self.lerobot_available = False
            print("Warning: LeRobot not installed. Install with: pip install lerobot")
    
    def _load_with_lerobot(self) -> List[Trajectory]:
        """Load dataset using LeRobot library."""
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        dataset = LeRobotDataset(
            repo_id=str(self.dataset_path),
            delta_timestamps=self.delta_timestamps,
        )
        
        trajectories = []
        
        # Get unique episode indices
        episode_indices = dataset.episode_indices.unique()
        
        for ep_idx in episode_indices:
            # Get all frames for this episode
            episode_mask = dataset.episode_indices == ep_idx
            episode_frames = [i for i, m in enumerate(episode_mask) if m]
            
            trajectory = Trajectory()
            trajectory.metadata = {
                'source': 'lerobot',
                'repo_id': str(self.dataset_path),
                'episode_index': int(ep_idx),
            }
            
            # Add any episode metadata
            if hasattr(dataset, 'meta') and 'episodes' in dataset.meta:
                ep_meta = dataset.meta['episodes'][ep_idx]
                trajectory.metadata.update(ep_meta)
            
            # Load each frame
            for frame_idx in episode_frames:
                item = dataset[frame_idx]
                
                # Extract observations
                obs = {}
                for lr_key, value in item.items():
                    if lr_key.startswith('observation.'):
                        unified_key = self.OBS_KEY_MAP.get(lr_key, lr_key)
                        obs[unified_key] = value.numpy() if hasattr(value, 'numpy') else value
                
                # Extract action
                action = item.get('action', np.zeros(0))
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                
                # Extract reward and done
                reward = item.get('reward', 0.0)
                if hasattr(reward, 'item'):
                    reward = reward.item()
                
                done = item.get('done', False)
                if hasattr(done, 'item'):
                    done = done.item()
                
                step = Step(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    terminated=bool(done),
                    truncated=False,
                )
                trajectory.add_step(step)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _load_without_lerobot(self) -> List[Trajectory]:
        """Load dataset manually without LeRobot library."""
        import pandas as pd
        
        data_dir = self.dataset_path / "data"
        
        # Find all parquet files
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        
        trajectories = []
        
        for parquet_file in parquet_files:
            # Read parquet
            df = pd.read_parquet(parquet_file)
            
            trajectory = Trajectory()
            trajectory.metadata = {
                'source': 'lerobot',
                'file': str(parquet_file.name),
            }
            
            # Process each row
            for _, row in df.iterrows():
                # Extract observations
                obs = {}
                for col in df.columns:
                    if col.startswith('observation.'):
                        unified_key = self.OBS_KEY_MAP.get(col, col)
                        value = row[col]
                        if isinstance(value, (list, np.ndarray)):
                            obs[unified_key] = np.array(value)
                        else:
                            obs[unified_key] = value
                
                # Extract action
                action_col = 'action' if 'action' in df.columns else 'action.action'
                if action_col in df.columns:
                    action = np.array(row[action_col])
                else:
                    action = np.zeros(0)
                
                # Extract reward and done
                reward = row.get('reward', 0.0)
                done = row.get('done', False)
                
                step = Step(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    terminated=bool(done),
                    truncated=False,
                )
                trajectory.add_step(step)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def load_all(self, max_episodes: Optional[int] = None) -> List[Trajectory]:
        """
        Load all episodes.
        
        Args:
            max_episodes: Maximum episodes to load
        
        Returns:
            List of Trajectory objects
        """
        if self.lerobot_available:
            print("Loading with LeRobot library...")
            trajectories = self._load_with_lerobot()
        else:
            print("Loading without LeRobot library...")
            trajectories = self._load_without_lerobot()
        
        if max_episodes is not None:
            trajectories = trajectories[:max_episodes]
        
        return trajectories
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        if self.lerobot_available:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            dataset = LeRobotDataset(repo_id=str(self.dataset_path))
            
            return {
                'num_episodes': dataset.num_episodes,
                'num_frames': dataset.num_frames,
                'avg_frames_per_episode': dataset.num_frames / dataset.num_episodes,
                'fps': dataset.fps if hasattr(dataset, 'fps') else 10,
            }
        else:
            # Manual stats
            data_dir = self.dataset_path / "data"
            parquet_files = list(data_dir.glob("episode_*.parquet"))
            
            import pandas as pd
            total_frames = 0
            for pf in parquet_files[:10]:  # Sample first 10
                df = pd.read_parquet(pf)
                total_frames += len(df)
            
            avg_frames = total_frames / min(10, len(parquet_files))
            
            return {
                'num_episodes': len(parquet_files),
                'estimated_total_frames': int(avg_frames * len(parquet_files)),
                'avg_frames_per_episode': avg_frames,
            }


def load_lerobot_dataset(
    dataset_path: Union[str, Path],
    max_episodes: Optional[int] = None,
) -> List[Trajectory]:
    """
    Convenience function to load LeRobot dataset.
    
    Args:
        dataset_path: Path to LeRobot dataset
        max_episodes: Maximum episodes to load
    
    Returns:
        List of trajectories
    """
    loader = LeRobotLoader(dataset_path)
    return loader.load_all(max_episodes=max_episodes)
