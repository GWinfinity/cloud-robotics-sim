"""
Stack objects task.
"""

import torch
import numpy as np


class StackTask:
    """
    Stack objects on top of each other.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.objects_list = []
        self.target_height = None
        
    def reset(self):
        """Reset task."""
        if hasattr(self.env, 'objects'):
            self.objects_list = list(self.env.objects.values())
        
        # Calculate target height (sum of object heights)
        if len(self.objects_list) >= 2:
            self.target_height = 0.1 * len(self.objects_list)  # Approximate
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 3  # Top object position
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        if len(self.objects_list) > 0:
            # Return top object position
            top_obj = self.objects_list[-1]
            return top_obj.get_pos()
        return torch.zeros(self.env.num_envs, 3)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if len(self.objects_list) < 2:
            return torch.zeros(self.env.num_envs)
        
        # Check if objects are stacked
        base_pos = self.objects_list[0].get_pos()
        top_pos = self.objects_list[-1].get_pos()
        
        height_diff = top_pos[:, 2] - base_pos[:, 2]
        
        # Reward for stacking height
        reward = torch.clamp(height_diff, 0, 0.3)
        
        # Check if properly stacked
        success = self.check_success()
        reward += success.float() * 10.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if objects are stacked."""
        if len(self.objects_list) < 2:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        
        # Check vertical alignment and height
        base_pos = self.objects_list[0].get_pos()
        top_pos = self.objects_list[-1].get_pos()
        
        height_diff = top_pos[:, 2] - base_pos[:, 2]
        horizontal_dist = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=-1)
        
        # Stacked if height is sufficient and horizontally aligned
        return (height_diff > 0.08) & (horizontal_dist < 0.05)
