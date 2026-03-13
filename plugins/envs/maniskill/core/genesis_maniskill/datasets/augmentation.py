"""
Data augmentation for robot trajectories.

Provides various augmentation techniques for training data:
- Action noise injection
- State perturbation
- Observation augmentation (color jitter, etc.)
- Trajectory mixing
"""

from typing import Callable, List, Optional
import numpy as np
import copy

from genesis_maniskill.datasets.formats.trajectory import Trajectory, Step


class TrajectoryAugmenter:
    """
    Apply augmentations to trajectory datasets.
    """
    
    def __init__(self, augmentations: List[Callable]):
        """
        Initialize augmenter with list of augmentation functions.
        
        Args:
            augmentations: List of augmentation functions
        """
        self.augmentations = augmentations
    
    def augment(self, trajectory: Trajectory) -> Trajectory:
        """
        Apply all augmentations to a trajectory.
        
        Args:
            trajectory: Input trajectory
        
        Returns:
            Augmented trajectory
        """
        augmented = copy.deepcopy(trajectory)
        
        for aug_fn in self.augmentations:
            augmented = aug_fn(augmented)
        
        return augmented
    
    def augment_dataset(
        self,
        trajectories: List[Trajectory],
        n_copies: int = 1,
        keep_original: bool = True
    ) -> List[Trajectory]:
        """
        Augment entire dataset.
        
        Args:
            trajectories: List of trajectories
            n_copies: Number of augmented copies per trajectory
            keep_original: Whether to keep original trajectories
        
        Returns:
            List of (possibly augmented) trajectories
        """
        result = []
        
        if keep_original:
            result.extend(trajectories)
        
        for _ in range(n_copies):
            for traj in trajectories:
                augmented = self.augment(traj)
                result.append(augmented)
        
        return result


# ==========================================
# Individual Augmentation Functions
# ==========================================

def add_action_noise(
    trajectory: Trajectory,
    noise_scale: float = 0.01,
    noise_type: str = 'gaussian'
) -> Trajectory:
    """
    Add noise to actions.
    
    Args:
        trajectory: Input trajectory
        noise_scale: Scale of noise
        noise_type: 'gaussian' or 'uniform'
    
    Returns:
        Augmented trajectory
    """
    augmented = copy.deepcopy(trajectory)
    
    for step in augmented.steps:
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_scale, size=step.action.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_scale, noise_scale, size=step.action.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        step.action = step.action + noise
    
    return augmented


def perturb_states(
    trajectory: Trajectory,
    perturb_scale: float = 0.01,
    state_keys: Optional[List[str]] = None
) -> Trajectory:
    """
    Add small perturbations to state observations.
    
    Args:
        trajectory: Input trajectory
        perturb_scale: Scale of perturbation
        state_keys: Which state keys to perturb (None = all)
    
    Returns:
        Augmented trajectory
    """
    augmented = copy.deepcopy(trajectory)
    
    if state_keys is None:
        # Auto-detect state-like keys
        state_keys = ['state', 'joint_pos', 'joint_vel', 'eef_pos']
    
    for step in augmented.steps:
        for key in state_keys:
            if key in step.obs:
                noise = np.random.normal(0, perturb_scale, size=step.obs[key].shape)
                step.obs[key] = step.obs[key] + noise
    
    return augmented


def temporally_shift(
    trajectory: Trajectory,
    shift_range: int = 5
) -> Trajectory:
    """
    Randomly shift trajectory temporally.
    
    Args:
        trajectory: Input trajectory
        shift_range: Maximum shift in either direction
    
    Returns:
        Augmented trajectory
    """
    augmented = copy.deepcopy(trajectory)
    
    shift = np.random.randint(-shift_range, shift_range + 1)
    
    if shift == 0:
        return augmented
    
    if shift > 0:
        # Shift forward: repeat first frame, drop last frames
        augmented.steps = [augmented.steps[0]] * shift + augmented.steps[:-shift]
    else:
        # Shift backward: repeat last frame, drop first frames
        augmented.steps = augmented.steps[-shift:] + [augmented.steps[-1]] * (-shift)
    
    return augmented


def randomly_subsample(
    trajectory: Trajectory,
    keep_ratio: float = 0.9
) -> Trajectory:
    """
    Randomly subsample frames.
    
    Args:
        trajectory: Input trajectory
        keep_ratio: Ratio of frames to keep
    
    Returns:
        Augmented trajectory
    """
    augmented = copy.deepcopy(trajectory)
    
    n_frames = len(augmented.steps)
    n_keep = max(2, int(n_frames * keep_ratio))
    
    # Randomly select frames while maintaining order
    indices = sorted(np.random.choice(n_frames, n_keep, replace=False))
    augmented.steps = [augmented.steps[i] for i in indices]
    
    return augmented


def mix_trajectories(
    trajectory1: Trajectory,
    trajectory2: Trajectory,
    mix_ratio: float = 0.5
) -> Trajectory:
    """
    Mix two trajectories (for mixup augmentation).
    
    Args:
        trajectory1: First trajectory
        trajectory2: Second trajectory
        mix_ratio: Mixing ratio (0 = all traj1, 1 = all traj2)
    
    Returns:
        Mixed trajectory
    """
    # Take shorter length
    min_len = min(len(trajectory1), len(trajectory2))
    
    mixed = Trajectory()
    mixed.metadata = {
        'mixed_from': [trajectory1.metadata, trajectory2.metadata],
        'mix_ratio': mix_ratio,
    }
    
    for i in range(min_len):
        step1 = trajectory1.steps[i]
        step2 = trajectory2.steps[i]
        
        # Mix observations
        mixed_obs = {}
        for key in step1.obs.keys():
            if key in step2.obs:
                mixed_obs[key] = (
                    (1 - mix_ratio) * step1.obs[key] +
                    mix_ratio * step2.obs[key]
                )
            else:
                mixed_obs[key] = step1.obs[key]
        
        # Mix actions
        mixed_action = (
            (1 - mix_ratio) * step1.action +
            mix_ratio * step2.action
        )
        
        # Average rewards
        mixed_reward = (1 - mix_ratio) * step1.reward + mix_ratio * step2.reward
        
        mixed_step = Step(
            obs=mixed_obs,
            action=mixed_action,
            reward=mixed_reward,
            terminated=step1.terminated,
            truncated=step1.truncated,
        )
        
        mixed.add_step(mixed_step)
    
    return mixed


def reverse_trajectory(trajectory: Trajectory) -> Trajectory:
    """
    Reverse trajectory temporally.
    
    Note: This may not make sense for all tasks (actions may not be reversible).
    
    Args:
        trajectory: Input trajectory
    
    Returns:
        Reversed trajectory
    """
    augmented = copy.deepcopy(trajectory)
    augmented.steps = list(reversed(augmented.steps))
    augmented.metadata['reversed'] = True
    
    return augmented


def scale_temporal(
    trajectory: Trajectory,
    speed_factor: float = 1.0
) -> Trajectory:
    """
    Scale trajectory temporally by resampling.
    
    Args:
        trajectory: Input trajectory
        speed_factor: Speed factor (>1 = faster, <1 = slower)
    
    Returns:
        Temporally scaled trajectory
    """
    from scipy import interpolate
    
    augmented = copy.deepcopy(trajectory)
    
    if speed_factor == 1.0:
        return augmented
    
    original_len = len(augmented.steps)
    new_len = max(2, int(original_len / speed_factor))
    
    # Extract action sequence
    actions = np.stack([s.action for s in augmented.steps])
    
    # Create interpolator
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    
    interp = interpolate.interp1d(x_old, actions, axis=0, kind='linear')
    new_actions = interp(x_new)
    
    # Rebuild steps with new actions
    new_steps = []
    for i in range(new_len):
        # Use nearest observation
        obs_idx = min(int(i * speed_factor), original_len - 1)
        
        new_step = Step(
            obs=augmented.steps[obs_idx].obs,
            action=new_actions[i],
            reward=augmented.steps[obs_idx].reward,
            terminated=augmented.steps[obs_idx].terminated if i == new_len - 1 else False,
            truncated=augmented.steps[obs_idx].truncated if i == new_len - 1 else False,
        )
        new_steps.append(new_step)
    
    augmented.steps = new_steps
    augmented.metadata['temporal_scale'] = speed_factor
    
    return augmented


# ==========================================
# Predefined Augmentation Pipelines
# ==========================================

def get_standard_augmentation() -> TrajectoryAugmenter:
    """
    Get standard augmentation pipeline for robot learning.
    
    Includes:
    - Action noise
    - State perturbation
    - Temporal shift
    """
    return TrajectoryAugmenter([
        lambda t: add_action_noise(t, noise_scale=0.01),
        lambda t: perturb_states(t, perturb_scale=0.005),
    ])


def get_aggressive_augmentation() -> TrajectoryAugmenter:
    """
    Get aggressive augmentation pipeline.
    
    Includes:
    - Higher action noise
    - State perturbation
    - Temporal subsampling
    """
    return TrajectoryAugmenter([
        lambda t: add_action_noise(t, noise_scale=0.05),
        lambda t: perturb_states(t, perturb_scale=0.01),
        lambda t: randomly_subsample(t, keep_ratio=0.95),
    ])


def get_bc_augmentation() -> TrajectoryAugmenter:
    """
    Get augmentation pipeline suitable for behavior cloning.
    
    Includes:
    - Small action noise
    - State perturbation
    """
    return TrajectoryAugmenter([
        lambda t: add_action_noise(t, noise_scale=0.005),
        lambda t: perturb_states(t, perturb_scale=0.002),
    ])
