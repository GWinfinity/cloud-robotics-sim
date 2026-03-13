"""
Open drawer task.
"""

import torch
import numpy as np


class OpenDrawerTask:
    """
    Open drawer task.
    
    Robot must open a cabinet drawer.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.target_drawer = None
        self.initial_drawer_pos = None
        
    def reset(self):
        """Reset task."""
        # Find drawer fixture
        if hasattr(self.env, 'fixtures'):
            for name, fixture in self.env.fixtures.items():
                if 'drawer' in name.lower() or 'cabinet' in name.lower():
                    self.target_drawer = fixture
                    self.initial_drawer_pos = fixture.get_pos().clone()
                    break
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 3  # Drawer position
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        if self.target_drawer is not None:
            return self.target_drawer.get_pos()
        return torch.zeros(self.env.num_envs, 3)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if self.target_drawer is None or self.initial_drawer_pos is None:
            return torch.zeros(self.env.num_envs)
        
        current_pos = self.target_drawer.get_pos()
        displacement = torch.norm(current_pos - self.initial_drawer_pos, dim=-1)
        
        # Reward based on how far drawer is opened
        reward = displacement
        
        # Success bonus
        success = displacement > 0.2
        reward += success.float() * 10.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        if self.target_drawer is None or self.initial_drawer_pos is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        
        current_pos = self.target_drawer.get_pos()
        displacement = torch.norm(current_pos - self.initial_drawer_pos, dim=-1)
        return displacement > 0.2
