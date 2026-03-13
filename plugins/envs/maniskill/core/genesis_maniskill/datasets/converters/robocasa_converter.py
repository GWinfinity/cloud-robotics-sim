"""
RoboCasa to Genesis ManiSkill converter.

Converts RoboCasa datasets to the unified Genesis ManiSkill format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from genesis_maniskill.datasets.loaders.robocasa_loader import RoboCasaLoader
from genesis_maniskill.datasets.formats.trajectory import TrajectoryDataset


class RoboCasaConverter:
    """
    Converter from RoboCasa format to Genesis ManiSkill format.
    
    Handles:
    - Observation key remapping
    - Coordinate system conversion
    - Action space normalization
    - Metadata preservation
    """
    
    def __init__(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        layout_map: Optional[Dict[int, int]] = None,
        action_normalization: bool = True,
    ):
        """
        Initialize converter.
        
        Args:
            source_path: Path to RoboCasa dataset
            target_path: Path to save converted dataset
            layout_map: Mapping from RoboCasa layout IDs to Genesis layout IDs
            action_normalization: Whether to normalize actions
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.layout_map = layout_map or {}
        self.action_normalization = action_normalization
        
        # Statistics for normalization
        self.action_mean = None
        self.action_std = None
        
    def analyze_dataset(self) -> Dict:
        """
        Analyze the source dataset before conversion.
        
        Returns:
            Dictionary with dataset statistics
        """
        with RoboCasaLoader(self.source_path) as loader:
            stats = loader.get_dataset_statistics()
            
            # Sample some trajectories to compute action statistics
            sample_trajs = loader.load_all(max_demos=min(100, stats['num_demos']))
            
            all_actions = []
            for traj in sample_trajs:
                actions = traj.get_actions()
                all_actions.append(actions)
            
            if all_actions:
                all_actions = np.concatenate(all_actions, axis=0)
                self.action_mean = np.mean(all_actions, axis=0)
                self.action_std = np.std(all_actions, axis=0)
                self.action_std = np.where(self.action_std == 0, 1.0, self.action_std)
            
            stats['action_mean'] = self.action_mean.tolist() if self.action_mean is not None else None
            stats['action_std'] = self.action_std.tolist() if self.action_std is not None else None
            
            return stats
    
    def convert_trajectory(self, trajectory) -> 'Trajectory':
        """
        Convert a single trajectory.
        
        Args:
            trajectory: Input trajectory
        
        Returns:
            Converted trajectory
        """
        # Remap layout ID if needed
        if 'layout_id' in trajectory.metadata:
            old_layout = trajectory.metadata['layout_id']
            if old_layout in self.layout_map:
                trajectory.metadata['layout_id'] = self.layout_map[old_layout]
        
        # Normalize actions if needed
        if self.action_normalization and self.action_mean is not None:
            for step in trajectory.steps:
                step.action = (step.action - self.action_mean) / self.action_std
        
        # Add conversion metadata
        trajectory.metadata['converted_from'] = 'robocasa'
        trajectory.metadata['conversion_version'] = '0.1.0'
        
        return trajectory
    
    def convert(
        self,
        max_demos: Optional[int] = None,
        save_format: str = 'hdf5',
    ) -> Path:
        """
        Convert entire dataset.
        
        Args:
            max_demos: Maximum demos to convert
            save_format: Format to save ('hdf5' or 'zarr')
        
        Returns:
            Path to converted dataset
        """
        print("=" * 60)
        print("RoboCasa to Genesis ManiSkill Converter")
        print("=" * 60)
        
        # Analyze dataset
        print("\n1. Analyzing source dataset...")
        stats = self.analyze_dataset()
        print(f"   Found {stats['num_demos']} demos")
        print(f"   Total steps: {stats['total_steps']}")
        print(f"   Avg steps per demo: {stats['avg_steps_per_demo']:.1f}")
        
        # Load trajectories
        print("\n2. Loading trajectories...")
        with RoboCasaLoader(self.source_path) as loader:
            trajectories = loader.load_all(max_demos=max_demos)
        
        print(f"   Loaded {len(trajectories)} trajectories")
        
        # Convert trajectories
        print("\n3. Converting trajectories...")
        converted_trajectories = []
        for i, traj in enumerate(trajectories):
            if i % 100 == 0:
                print(f"   Converting {i+1}/{len(trajectories)}...")
            
            converted_traj = self.convert_trajectory(traj)
            converted_trajectories.append(converted_traj)
        
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
            'source': 'robocasa',
            'source_path': str(self.source_path),
            'num_trajectories': len(converted_trajectories),
            'statistics': dataset.get_statistics(),
            'conversion_config': {
                'action_normalization': self.action_normalization,
                'layout_map': self.layout_map,
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
        with RoboCasaLoader(self.source_path) as loader:
            orig_trajs = loader.load_all(max_demos=num_samples)
        
        converted_dataset = TrajectoryDataset.load(self.target_path)
        
        # Compare
        all_passed = True
        for i, (orig, conv) in enumerate(zip(orig_trajs, converted_dataset.trajectories)):
            # Check length
            if len(orig) != len(conv):
                print(f"   ❌ Trajectory {i}: Length mismatch")
                all_passed = False
                continue
            
            # Check actions (allow for normalization difference)
            orig_actions = orig.get_actions()
            conv_actions = conv.get_actions()
            
            if self.action_normalization:
                # Denormalize converted actions for comparison
                conv_actions = conv_actions * self.action_std + self.action_mean
            
            action_diff = np.abs(orig_actions - conv_actions).max()
            
            if action_diff > 1e-5:
                print(f"   ❌ Trajectory {i}: Action mismatch (diff={action_diff:.2e})")
                all_passed = False
            else:
                print(f"   ✅ Trajectory {i}: OK")
        
        if all_passed:
            print("\n   ✅ All verifications passed!")
        else:
            print("\n   ⚠️  Some verifications failed")
        
        return all_passed


def convert_robocasa_dataset(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    max_demos: Optional[int] = None,
    verify: bool = True,
) -> Path:
    """
    Convenience function to convert RoboCasa dataset.
    
    Args:
        source_path: Path to RoboCasa dataset
        target_path: Path to save converted dataset
        max_demos: Maximum demos to convert
        verify: Whether to verify conversion
    
    Returns:
        Path to converted dataset
    """
    converter = RoboCasaConverter(source_path, target_path)
    output_path = converter.convert(max_demos=max_demos)
    
    if verify:
        converter.verify_conversion()
    
    return output_path
