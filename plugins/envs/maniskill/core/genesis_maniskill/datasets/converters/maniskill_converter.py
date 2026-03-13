"""
ManiSkill to Genesis ManiSkill converter.

Converts ManiSkill datasets to the unified Genesis ManiSkill format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from genesis_maniskill.datasets.loaders.maniskill_loader import ManiSkillLoader
from genesis_maniskill.datasets.formats.trajectory import TrajectoryDataset


class ManiSkillConverter:
    """
    Converter from ManiSkill format to Genesis ManiSkill format.
    
    Handles:
    - Observation key remapping
    - Scene type mapping (tabletop -> TableTopEnv)
    - Task name mapping
    - Action space compatibility
    """
    
    # Task name mapping from ManiSkill to Genesis
    TASK_MAP = {
        'PickCube-v0': 'pick_place',
        'PickCube-v1': 'pick_place',
        'StackCube-v0': 'stack',
        'StackCube-v1': 'stack',
        'PushCube-v0': 'push',
        'PushCube-v1': 'push',
        'OpenCabinetDrawer-v1': 'open_drawer',
        'OpenCabinetDoor-v1': 'open_door',
        # Add more mappings as needed
    }
    
    # Scene type mapping
    SCENE_MAP = {
        'PickCube': 'tabletop',
        'StackCube': 'tabletop',
        'PushCube': 'tabletop',
        'OpenCabinet': 'kitchen',
        'OpenCabinetDrawer': 'kitchen',
        'OpenCabinetDoor': 'kitchen',
    }
    
    def __init__(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        task_map: Optional[Dict[str, str]] = None,
        scene_map: Optional[Dict[str, str]] = None,
        obs_mode: str = 'state',
    ):
        """
        Initialize converter.
        
        Args:
            source_path: Path to ManiSkill dataset
            target_path: Path to save converted dataset
            task_map: Custom task name mapping
            scene_map: Custom scene type mapping
            obs_mode: Observation mode used in source
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.task_map = task_map or self.TASK_MAP
        self.scene_map = scene_map or self.SCENE_MAP
        self.obs_mode = obs_mode
        
    def infer_task_and_scene(self, trajectory) -> tuple:
        """
        Infer task and scene type from trajectory metadata.
        
        Returns:
            (task_type, scene_type) tuple
        """
        # Try to get from metadata
        task = trajectory.metadata.get('task', '')
        
        # Map task name
        for ms_task, gs_task in self.task_map.items():
            if ms_task in task or task in ms_task:
                task = gs_task
                break
        
        # Infer scene type from task
        scene_type = 'tabletop'  # default
        for prefix, scene in self.scene_map.items():
            if prefix in task or prefix in trajectory.metadata.get('env_id', ''):
                scene_type = scene
                break
        
        return task, scene_type
    
    def convert_trajectory(self, trajectory) -> 'Trajectory':
        """
        Convert a single trajectory.
        
        Args:
            trajectory: Input trajectory
        
        Returns:
            Converted trajectory
        """
        # Infer task and scene
        task_type, scene_type = self.infer_task_and_scene(trajectory)
        
        # Update metadata
        trajectory.metadata['task_type'] = task_type
        trajectory.metadata['scene_type'] = scene_type
        trajectory.metadata['converted_from'] = 'maniskill'
        trajectory.metadata['conversion_version'] = '0.1.0'
        
        # Convert observation keys if needed
        for step in trajectory.steps:
            # Ensure state observation exists
            if 'state' not in step.obs and 'joint_pos' in step.obs:
                # Use joint_pos as main state
                step.obs['state'] = step.obs['joint_pos']
        
        return trajectory
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the source dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        loader = ManiSkillLoader(self.source_path, obs_mode=self.obs_mode)
        stats = loader.get_dataset_statistics()
        
        return stats
    
    def convert(
        self,
        max_trajs: Optional[int] = None,
        save_format: str = 'hdf5',
    ) -> Path:
        """
        Convert entire dataset.
        
        Args:
            max_trajs: Maximum trajectories to convert
            save_format: Format to save ('hdf5' or 'zarr')
        
        Returns:
            Path to converted dataset
        """
        print("=" * 60)
        print("ManiSkill to Genesis ManiSkill Converter")
        print("=" * 60)
        
        # Analyze dataset
        print("\n1. Analyzing source dataset...")
        stats = self.analyze_dataset()
        print(f"   Found {stats['num_trajectories']} trajectories")
        print(f"   Estimated total steps: {stats.get('estimated_total_steps', 'N/A')}")
        print(f"   Format: {stats.get('format', 'unknown')}")
        
        # Load trajectories
        print("\n2. Loading trajectories...")
        loader = ManiSkillLoader(self.source_path, obs_mode=self.obs_mode)
        trajectories = loader.load_all(max_trajs=max_trajs)
        print(f"   Loaded {len(trajectories)} trajectories")
        
        # Convert trajectories
        print("\n3. Converting trajectories...")
        converted_trajectories = []
        
        task_counts = {}
        for i, traj in enumerate(trajectories):
            if i % 100 == 0:
                print(f"   Converting {i+1}/{len(trajectories)}...")
            
            converted_traj = self.convert_trajectory(traj)
            converted_trajectories.append(converted_traj)
            
            # Count tasks
            task = converted_traj.metadata.get('task_type', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print(f"\n   Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"     {task}: {count}")
        
        # Create dataset
        print("\n4. Creating dataset...")
        dataset = TrajectoryDataset(converted_trajectories)
        
        # Save
        print(f"\n5. Saving to {save_format} format...")
        self.target_path.mkdir(parents=True, exist_ok=True)
        dataset.save(self.target_path, format=save_format)
        
        # Save metadata
        import json
        metadata = {
            'source': 'maniskill',
            'source_path': str(self.source_path),
            'num_trajectories': len(converted_trajectories),
            'statistics': dataset.get_statistics(),
            'task_distribution': task_counts,
            'conversion_config': {
                'task_map': self.task_map,
                'scene_map': self.scene_map,
                'obs_mode': self.obs_mode,
            }
        }
        
        with open(self.target_path / 'conversion_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Conversion complete!")
        print(f"   Saved to: {self.target_path}")
        print(f"   Format: {save_format}")
        
        return self.target_path
    
    def verify_conversion(self, num_samples: int = 5) -> bool:
        """
        Verify the converted dataset.
        
        Args:
            num_samples: Number of samples to verify
        
        Returns:
            True if verification passed
        """
        print("\n6. Verifying conversion...")
        
        # Load original and converted
        loader = ManiSkillLoader(self.source_path, obs_mode=self.obs_mode)
        orig_files = loader.get_trajectory_files()[:num_samples]
        orig_trajs = [loader.load_trajectory(f) for f in orig_files]
        
        converted_dataset = TrajectoryDataset.load(self.target_path)
        
        # Compare
        all_passed = True
        for i, (orig, conv) in enumerate(zip(orig_trajs, converted_dataset.trajectories)):
            # Check length
            if len(orig) != len(conv):
                print(f"   ❌ Trajectory {i}: Length mismatch ({len(orig)} vs {len(conv)})")
                all_passed = False
                continue
            
            # Check actions
            orig_actions = orig.get_actions()
            conv_actions = conv.get_actions()
            
            action_diff = np.abs(orig_actions - conv_actions).max()
            
            if action_diff > 1e-5:
                print(f"   ❌ Trajectory {i}: Action mismatch (diff={action_diff:.2e})")
                all_passed = False
            else:
                print(f"   ✅ Trajectory {i}: OK (len={len(orig)})")
        
        if all_passed:
            print("\n   ✅ All verifications passed!")
        else:
            print("\n   ⚠️  Some verifications failed")
        
        return all_passed


def convert_maniskill_dataset(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    max_trajs: Optional[int] = None,
    obs_mode: str = 'state',
    verify: bool = True,
) -> Path:
    """
    Convenience function to convert ManiSkill dataset.
    
    Args:
        source_path: Path to ManiSkill dataset
        target_path: Path to save converted dataset
        max_trajs: Maximum trajectories to convert
        obs_mode: Observation mode
        verify: Whether to verify conversion
    
    Returns:
        Path to converted dataset
    """
    converter = ManiSkillConverter(
        source_path, 
        target_path,
        obs_mode=obs_mode
    )
    output_path = converter.convert(max_trajs=max_trajs)
    
    if verify:
        converter.verify_conversion()
    
    return output_path
