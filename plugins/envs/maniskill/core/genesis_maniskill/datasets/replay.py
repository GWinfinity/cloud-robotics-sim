"""
Replay trajectories in Genesis environments.

Allows visualizing and validating converted trajectories by replaying them.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import time

import genesis as gs
from genesis_maniskill.datasets.formats.trajectory import Trajectory, TrajectoryDataset


class TrajectoryReplayer:
    """
    Replay trajectories in Genesis environments.
    
    Can be used to:
    - Visualize expert demonstrations
    - Validate converted datasets
    - Generate new observations from saved actions
    """
    
    def __init__(
        self,
        env,
        render_mode: Optional[str] = 'human',
        fps: int = 10,
    ):
        """
        Initialize replayer.
        
        Args:
            env: Genesis ManiSkill environment
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            fps: Playback frames per second
        """
        self.env = env
        self.render_mode = render_mode
        self.fps = fps
        self.dt = 1.0 / fps
    
    def replay(
        self,
        trajectory: Trajectory,
        record_video: bool = False,
        video_path: Optional[Path] = None,
    ) -> Dict:
        """
        Replay a single trajectory.
        
        Args:
            trajectory: Trajectory to replay
            record_video: Whether to record video
            video_path: Path to save video
        
        Returns:
            Dictionary with replay statistics
        """
        # Reset environment
        obs, info = self.env.reset()
        
        # If trajectory has initial state, try to set it
        if trajectory.steps and 'state' in trajectory.steps[0].obs:
            self._set_initial_state(trajectory.steps[0].obs['state'])
        
        # Setup video recording if needed
        frames = []
        
        # Replay steps
        replayed_observations = []
        action_errors = []
        rewards = []
        
        for i, step in enumerate(trajectory.steps):
            # Render
            if self.render_mode:
                frame = self.env.render()
                if record_video and frame is not None:
                    frames.append(frame)
            
            # Execute action
            action = step.action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Record results
            replayed_observations.append(obs)
            rewards.append(reward)
            
            # Compare with original observation if available
            if 'state' in step.obs and hasattr(obs, '__len__'):
                expected_obs = step.obs['state']
                if len(expected_obs) == len(obs):
                    error = np.abs(expected_obs - obs).mean()
                    action_errors.append(error)
            
            # Control playback speed
            if self.render_mode == 'human':
                time.sleep(self.dt)
        
        # Save video if recorded
        if record_video and frames and video_path:
            self._save_video(frames, video_path)
        
        # Return statistics
        stats = {
            'num_steps': len(trajectory.steps),
            'total_reward': sum(rewards),
            'avg_obs_error': np.mean(action_errors) if action_errors else None,
            'max_obs_error': np.max(action_errors) if action_errors else None,
        }
        
        return stats
    
    def replay_with_comparison(
        self,
        trajectory: Trajectory,
        save_comparison_video: bool = True,
        video_path: Optional[Path] = None,
    ) -> Dict:
        """
        Replay trajectory and compare with original observations.
        
        Creates a side-by-side comparison video showing original vs replayed.
        
        Args:
            trajectory: Trajectory to replay
            save_comparison_video: Whether to save comparison video
            video_path: Path to save video
        
        Returns:
            Dictionary with comparison statistics
        """
        import cv2
        
        # Reset environment
        obs, info = self.env.reset()
        
        frames_original = []
        frames_replayed = []
        errors = []
        
        for step in trajectory.steps:
            # Get original observation (if image available)
            if 'rgb_agentview' in step.obs:
                orig_frame = step.obs['rgb_agentview']
                frames_original.append(orig_frame)
            
            # Execute action
            action = step.action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Get replayed observation
            replayed_frame = self.env.render()
            if replayed_frame is not None:
                frames_replayed.append(replayed_frame)
            
            # Compute error
            if frames_original and frames_replayed:
                error = self._compute_frame_error(
                    frames_original[-1],
                    frames_replayed[-1]
                )
                errors.append(error)
        
        # Create comparison video
        if save_comparison_video and frames_original and frames_replayed:
            comparison_frames = self._create_comparison_frames(
                frames_original,
                frames_replayed,
                errors
            )
            
            if video_path:
                self._save_video(comparison_frames, video_path)
        
        return {
            'num_steps': len(trajectory.steps),
            'avg_frame_error': np.mean(errors) if errors else None,
            'max_frame_error': np.max(errors) if errors else None,
        }
    
    def batch_replay(
        self,
        dataset: TrajectoryDataset,
        max_trajectories: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Replay multiple trajectories.
        
        Args:
            dataset: Dataset to replay
            max_trajectories: Maximum trajectories to replay
            verbose: Whether to print progress
        
        Returns:
            List of statistics dictionaries
        """
        trajectories = dataset.trajectories
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]
        
        all_stats = []
        
        for i, traj in enumerate(trajectories):
            if verbose:
                print(f"Replaying trajectory {i+1}/{len(trajectories)}...")
            
            stats = self.replay(traj)
            all_stats.append(stats)
            
            if verbose:
                print(f"  Steps: {stats['num_steps']}, "
                      f"Reward: {stats['total_reward']:.2f}")
        
        return all_stats
    
    def _set_initial_state(self, state: np.ndarray):
        """Try to set environment to initial state."""
        # This depends on environment capabilities
        # Some environments support state setting, others don't
        if hasattr(self.env, 'set_state'):
            self.env.set_state(state)
    
    def _compute_frame_error(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """Compute error between two frames."""
        # Ensure same size
        if frame1.shape != frame2.shape:
            import cv2
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Compute MSE
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        return mse
    
    def _create_comparison_frames(
        self,
        frames_original: List[np.ndarray],
        frames_replayed: List[np.ndarray],
        errors: List[float],
    ) -> List[np.ndarray]:
        """Create side-by-side comparison frames."""
        import cv2
        
        comparison_frames = []
        
        for orig, replay, error in zip(frames_original, frames_replayed, errors):
            # Ensure same height
            h = max(orig.shape[0], replay.shape[0])
            w = orig.shape[1]
            
            if orig.shape[0] != h:
                orig = cv2.resize(orig, (w, h))
            if replay.shape[0] != h:
                replay = cv2.resize(replay, (w, h))
            
            # Concatenate horizontally
            combined = np.concatenate([orig, replay], axis=1)
            
            # Add labels
            cv2.putText(combined, 'Original', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, 'Replayed', (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add error text
            cv2.putText(combined, f'Error: {error:.2f}', (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            comparison_frames.append(combined)
        
        return comparison_frames
    
    def _save_video(self, frames: List[np.ndarray], path: Path):
        """Save frames as video."""
        import cv2
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        
        writer.release()
        print(f"Video saved to: {path}")


class DatasetValidator:
    """
    Validate converted datasets by replaying in simulation.
    
    This checks if the converted actions produce similar results
    when replayed in the Genesis environment.
    """
    
    def __init__(self, env_factory):
        """
        Initialize validator.
        
        Args:
            env_factory: Function that creates environment
        """
        self.env_factory = env_factory
    
    def validate_trajectory(
        self,
        trajectory: Trajectory,
        obs_tolerance: float = 0.1,
        reward_tolerance: float = 0.01,
    ) -> Tuple[bool, Dict]:
        """
        Validate a single trajectory.
        
        Args:
            trajectory: Trajectory to validate
            obs_tolerance: Tolerance for observation error
            reward_tolerance: Tolerance for reward error
        
        Returns:
            (is_valid, statistics)
        """
        env = self.env_factory()
        replayer = TrajectoryReplayer(env, render_mode=None)
        
        # Replay
        stats = replayer.replay(trajectory)
        
        # Check original vs replayed
        original_rewards = trajectory.get_rewards().sum()
        replayed_rewards = stats['total_reward']
        
        reward_error = abs(original_rewards - replayed_rewards)
        
        # Determine validity
        is_valid = True
        
        if reward_error > reward_tolerance:
            is_valid = False
        
        if stats['avg_obs_error'] is not None:
            if stats['avg_obs_error'] > obs_tolerance:
                is_valid = False
        
        stats['original_reward'] = original_rewards
        stats['reward_error'] = reward_error
        stats['is_valid'] = is_valid
        
        env.close()
        
        return is_valid, stats
    
    def validate_dataset(
        self,
        dataset: TrajectoryDataset,
        max_trajectories: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Validate entire dataset.
        
        Args:
            dataset: Dataset to validate
            max_trajectories: Maximum trajectories to validate
            verbose: Whether to print progress
        
        Returns:
            Validation report
        """
        trajectories = dataset.trajectories
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]
        
        valid_count = 0
        invalid_count = 0
        all_stats = []
        
        for i, traj in enumerate(trajectories):
            if verbose and i % 10 == 0:
                print(f"Validating {i+1}/{len(trajectories)}...")
            
            is_valid, stats = self.validate_trajectory(traj)
            all_stats.append(stats)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        # Compute aggregate statistics
        total_reward_errors = [s['reward_error'] for s in all_stats]
        avg_obs_errors = [s['avg_obs_error'] for s in all_stats if s['avg_obs_error'] is not None]
        
        report = {
            'total_trajectories': len(trajectories),
            'valid': valid_count,
            'invalid': invalid_count,
            'validity_rate': valid_count / len(trajectories) if trajectories else 0,
            'avg_reward_error': np.mean(total_reward_errors),
            'max_reward_error': np.max(total_reward_errors),
            'avg_obs_error': np.mean(avg_obs_errors) if avg_obs_errors else None,
            'max_obs_error': np.max(avg_obs_errors) if avg_obs_errors else None,
        }
        
        return report


def replay_trajectory(
    env,
    trajectory: Trajectory,
    fps: int = 10,
    record_video: bool = False,
    video_path: Optional[str] = None,
) -> Dict:
    """Convenience function to replay a trajectory."""
    replayer = TrajectoryReplayer(env, fps=fps)
    return replayer.replay(
        trajectory,
        record_video=record_video,
        video_path=Path(video_path) if video_path else None,
    )


def validate_dataset(
    env_factory,
    dataset: TrajectoryDataset,
    max_trajectories: Optional[int] = None,
) -> Dict:
    """Convenience function to validate a dataset."""
    validator = DatasetValidator(env_factory)
    return validator.validate_dataset(dataset, max_trajectories=max_trajectories)
