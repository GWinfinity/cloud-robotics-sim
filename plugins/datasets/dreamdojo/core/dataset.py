# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Genesis Dataset for DreamDojo.

This module provides dataset integration with Genesis physics simulator.
Adapted from DreamDojo's genesis_dreams module to work with 
genesis-cloud-sim's plugin architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Callable
import random

# Optional heavy dependencies
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    # Create a dummy Dataset base class if torch is not available
    class Dataset:
        pass
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .simulator import GenesisSimulator, GenesisSimulatorConfig, GenesisRobotType

# Optional Genesis import
try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False


class GenesisDataset(Dataset):
    """
    Dataset for Genesis physics simulator-generated data.
    
    This dataset generates synthetic robot trajectories using
    Genesis physics simulation on-the-fly or loads pre-generated data.
    
    Compatible with genesis-cloud-sim's dataset pipeline and
    maniskill's data converters.
    """
    
    def __init__(
        self,
        num_frames: int = 81,
        episode_length: int = 100,
        num_episodes: int = 1000,
        robot_type: str = "humanoid",
        simulator_config: Optional[dict[str, Any]] = None,
        pre_generated_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        seed: int = 0,
        use_online_sim: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            num_frames: Number of frames per sample
            episode_length: Length of each episode
            num_episodes: Number of episodes
            robot_type: Type of robot ("humanoid", "ur5", "franka", "g1", "gr1")
            simulator_config: Genesis simulator configuration
            pre_generated_path: Path to pre-generated data (if not using online sim)
            transforms: Data transforms to apply
            seed: Random seed
            use_online_sim: Whether to use online simulation (slow but diverse)
            device: Device to use for simulation ("cuda" or "cpu")
        """
        self.num_frames = num_frames
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.robot_type = robot_type
        self.transforms = transforms
        self.seed = seed
        self.use_online_sim = use_online_sim
        self.pre_generated_path = pre_generated_path
        self.device = device
        
        # Initialize simulator if using online simulation
        self.simulator: Optional[GenesisSimulator] = None
        if use_online_sim and GENESIS_AVAILABLE:
            config = GenesisSimulatorConfig(
                robot_type=GenesisRobotType(robot_type),
                headless=True,
                device=device,
                **(simulator_config or {})
            )
            self.simulator = GenesisSimulator(config)
            self.simulator.initialize()
        
        # Load pre-generated data if available
        self.pre_generated_data: Optional[Dict] = None
        if pre_generated_path is not None:
            self._load_pre_generated_data(pre_generated_path)
        
        if np is not None:
            self.rng = np.random.RandomState(seed)
        else:
            import random
            self.rng = random.Random(seed)
        
    def _load_pre_generated_data(self, path: str):
        """Load pre-generated simulation data."""
        import h5py
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pre-generated data not found: {path}")
            
        self.pre_generated_data = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                self.pre_generated_data[key] = {
                    "observations": f[key]["observations"][:],
                    "actions": f[key]["actions"][:],
                }
    
    def __len__(self) -> int:
        return self.num_episodes
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing video frames, actions, and metadata
        """
        if self.use_online_sim and self.simulator is not None:
            return self._get_online_sample(idx)
        elif self.pre_generated_data is not None:
            return self._get_pre_generated_sample(idx)
        else:
            # Generate synthetic data on-the-fly without simulator
            return self._get_synthetic_sample(idx)
    
    def _get_online_sample(self, idx: int) -> dict[str, Any]:
        """Generate sample using online Genesis simulation."""
        self.rng.seed(self.seed + idx)
        
        # Reset simulator
        self.simulator.reset(seed=self.seed + idx)
        
        # Generate episode
        frames = []
        actions = []
        states = []
        
        # Infer action dimension from robot type
        action_dim = self._get_action_dim()
        
        for t in range(self.num_frames):
            # Random action for data collection
            # In practice, this could be replaced with a policy
            action = self.rng.randn(action_dim).astype(np.float32) * 0.5
            actions.append(action)
            
            # Get state before stepping
            state = self.simulator.get_state()
            states.append(state)
            
            # Step simulator
            obs, _ = self.simulator.step(action)
            frames.append(obs)
        
        # Convert to tensors
        if torch is not None and np is not None:
            video = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
            video = video.float() / 255.0
        else:
            video = None
        
        # Process actions
        if torch is not None:
            action_tensor = torch.from_numpy(np.array(actions) if np is not None else actions).float()
        else:
            action_tensor = None
        
        data = {
            "video": video,
            "action": action_tensor,
            "fps": 20,
            "num_frames": self.num_frames,
        }
        
        if self.transforms is not None:
            data = self.transforms(data)
        
        return data
    
    def _get_pre_generated_sample(self, idx: int) -> dict[str, Any]:
        """Get sample from pre-generated data."""
        episode_key = f"episode_{idx % len(self.pre_generated_data)}"
        episode = self.pre_generated_data[episode_key]
        
        # Sample random start frame
        max_start = len(episode["observations"]) - self.num_frames
        if max_start <= 0:
            start = 0
        else:
            start = self.rng.randint(0, max_start)
        
        frames = episode["observations"][start:start + self.num_frames]
        actions = episode["actions"][start:start + self.num_frames]
        
        # Convert to tensors
        if torch is not None:
            video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
            if video.max() > 1.0:
                video = video.float() / 255.0
            action_tensor = torch.from_numpy(actions).float()
        else:
            video = None
            action_tensor = None
        
        data = {
            "video": video,
            "action": action_tensor,
            "fps": 20,
            "num_frames": self.num_frames,
        }
        
        if self.transforms is not None:
            data = self.transforms(data)
        
        return data
    
    def _get_synthetic_sample(self, idx: int) -> dict[str, Any]:
        """
        Generate synthetic sample without simulator.
        
        This is a placeholder for quick testing without Genesis installed.
        """
        self.rng.seed(self.seed + idx)
        
        # Generate synthetic video (random noise with temporal consistency)
        if np is not None and torch is not None:
            base_pattern = self.rng.rand(3, 480, 640)
            video = torch.zeros(self.num_frames, 3, 480, 640)
            for t in range(self.num_frames):
                noise = self.rng.rand(3, 480, 640) * 0.1
                video[t] = torch.from_numpy(base_pattern + noise).float()
            
            # Generate synthetic actions
            action_dim = self._get_action_dim()
            action = torch.randn(self.num_frames, action_dim) * 0.3
            
            # LAM video (lower resolution)
            lam_video = torch.rand(self.num_frames - 2, 240, 320, 3)
            
            # Key for conditioning
            key = torch.randn(1, 29)
            
            return {
                "video": video,
                "action": action,
                "lam_video": lam_video,
                "fps": 20,
                "num_frames": self.num_frames,
                "__key__": key,
                "padding_mask": torch.zeros(1, 256, 256),
                "image_size": torch.ones(4) * 256,
                "ai_caption": f"Genesis synthetic data ({self.robot_type})",
            }
        else:
            # Return dummy data when dependencies are not available
            # Create a simple object with shape attribute for compatibility
            class DummyTensor:
                def __init__(self, *shape):
                    self.shape = shape
            return {
                "video": DummyTensor(self.num_frames, 3, 480, 640),
                "action": DummyTensor(self.num_frames, self._get_action_dim()),
                "fps": 20,
                "num_frames": self.num_frames,
            }
    
    def _get_action_dim(self) -> int:
        """Get action dimension based on robot type."""
        action_dims = {
            "humanoid": 21,
            "ur5": 6,
            "franka": 7,
            "g1": 29,
            "gr1": 32,
            "custom": 7,
        }
        return action_dims.get(self.robot_type, 7)
    
    def close(self):
        """Clean up resources."""
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None


class GenesisRLDataset(Dataset):
    """
    Dataset for RL-generated data using Genesis.
    
    This dataset uses a trained policy to generate realistic
    robot trajectories in Genesis simulation.
    """
    
    def __init__(
        self,
        simulator: GenesisSimulator,
        policy: Optional[Callable] = None,
        num_episodes: int = 1000,
        episode_length: int = 100,
        num_frames: int = 81,
        seed: int = 0,
    ):
        """
        Args:
            simulator: Genesis simulator
            policy: Trained policy for action generation (optional)
            num_episodes: Number of episodes
            episode_length: Length of each episode
            num_frames: Number of frames per sample
            seed: Random seed
        """
        self.simulator = simulator
        self.policy = policy
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.num_frames = num_frames
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def __len__(self) -> int:
        return self.num_episodes
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate episode using RL policy."""
        self.rng.seed(self.seed + idx)
        self.simulator.reset(seed=self.seed + idx)
        
        frames = []
        actions = []
        
        action_dim = self._get_action_dim()
        
        for t in range(self.num_frames):
            # Get state
            state = self.simulator.get_state()
            
            # Get action from policy if available, otherwise random
            if self.policy is not None:
                action = self.policy(state)
            else:
                action = self.rng.randn(action_dim).astype(np.float32) * 0.5
            actions.append(action)
            
            # Step simulator
            obs, _ = self.simulator.step(action)
            frames.append(obs)
        
        if torch is not None:
            video = torch.from_numpy(np.array(frames) if np is not None else frames).permute(0, 3, 1, 2)
            video = video.float() / 255.0
            action_tensor = torch.from_numpy(np.array(actions) if np is not None else actions).float()
        else:
            video = None
            action_tensor = None
        
        return {
            "video": video,
            "action": action_tensor,
            "fps": 20,
            "num_frames": self.num_frames,
        }
    
    def _get_action_dim(self) -> int:
        """Get action dimension from simulator config."""
        robot_type = self.simulator.config.robot_type.value
        action_dims = {
            "humanoid": 21,
            "ur5": 6,
            "franka": 7,
            "g1": 29,
            "gr1": 32,
            "custom": 7,
        }
        return action_dims.get(robot_type, 7)


class GenesisDatasetWrapper:
    """
    Wrapper to use Genesis simulator as a dataset source.
    
    This generates synthetic data using Genesis physics simulation
    for training DreamDojo world models.
    
    Compatible with cloud-robotics-sim's dataset pipeline.
    """
    
    def __init__(
        self,
        simulator: GenesisSimulator,
        num_episodes: int = 1000,
        episode_length: int = 100,
        action_dim: Optional[int] = None,
        seed: int = 0,
    ):
        """
        Args:
            simulator: Genesis simulator instance
            num_episodes: Number of episodes to generate
            episode_length: Length of each episode
            action_dim: Dimension of action space (auto-detect if None)
            seed: Random seed
        """
        self.simulator = simulator
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Auto-detect action dimension
        if action_dim is None:
            self.action_dim = self._detect_action_dim()
        else:
            self.action_dim = action_dim
        
    def _detect_action_dim(self) -> int:
        """Detect action dimension from simulator."""
        robot_type = self.simulator.config.robot_type.value
        action_dims = {
            "humanoid": 21,
            "ur5": 6,
            "franka": 7,
            "g1": 29,
            "gr1": 32,
            "custom": 7,
        }
        return action_dims.get(robot_type, 7)
        
    def generate_episode(self, policy: Optional[Callable] = None) -> dict[str, Any]:
        """
        Generate a single episode using Genesis simulation.
        
        Args:
            policy: Optional policy for action generation
            
        Returns:
            Episode data dictionary
        """
        self.simulator.reset()
        
        observations = []
        actions = []
        states = []
        
        for t in range(self.episode_length):
            # Get current state
            state = self.simulator.get_state()
            states.append(state)
            
            # Generate or get action
            if policy is None:
                action = self.rng.randn(self.action_dim).astype(np.float32) * 0.5
            else:
                action = policy(state)
            actions.append(action)
            
            # Step simulation
            obs, info = self.simulator.step(action)
            observations.append(obs)
        
        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "states": states,
        }
    
    def generate_dataset(self, save_path: str, policy: Optional[Callable] = None):
        """
        Generate full dataset and save to disk.
        
        Args:
            save_path: Path to save dataset
            policy: Optional policy for action generation
        """
        import h5py
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, "w") as f:
            for i in tqdm(range(self.num_episodes), desc="Generating episodes"):
                episode = self.generate_episode(policy)
                grp = f.create_group(f"episode_{i}")
                grp.create_dataset("observations", data=episode["observations"])
                grp.create_dataset("actions", data=episode["actions"])


def create_genesis_dataset(
    dataset_path: Optional[str] = None,
    num_frames: int = 81,
    robot_type: str = "humanoid",
    use_online_sim: bool = False,
    **kwargs
) -> GenesisDataset:
    """
    Factory function to create Genesis dataset.
    
    Args:
        dataset_path: Path to pre-generated data
        num_frames: Number of frames per sample
        robot_type: Type of robot
        use_online_sim: Whether to use online simulation
        **kwargs: Additional arguments
        
    Returns:
        Genesis dataset instance
    """
    return GenesisDataset(
        num_frames=num_frames,
        pre_generated_path=dataset_path,
        robot_type=robot_type,
        use_online_sim=use_online_sim,
        **kwargs
    )


def is_genesis_dataset(dataset_path: str) -> bool:
    """Check if dataset path indicates Genesis simulator data."""
    return "genesis" in dataset_path.lower()
