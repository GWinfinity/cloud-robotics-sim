"""
Push object task.
"""

import torch
import numpy as np


class PushTask:
    """
    Push object to target location.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.target_object = None
        self.target_pos = None
        
    def reset(self):
        """Reset task."""
        # Select object
        if hasattr(self.env, 'objects') and len(self.env.objects) > 0:
            obj_names = list(self.env.objects.keys())
            self.target_object = self.env.objects[obj_names[0]]
        
        # Set target position
        if self.target_object is not None:
            # Random target on table/surface
            self.target_pos = torch.tensor([
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.2, 0.2),
                0.05
            ])
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 6  # obj_pos (3) + target_pos (3)
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        if self.target_object is not None and self.target_pos is not None:
            obj_pos = self.target_object.get_pos()
            return torch.cat([obj_pos, self.target_pos], dim=-1)
        return torch.zeros(self.env.num_envs, 6)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if self.target_object is None or self.target_pos is None:
            return torch.zeros(self.env.num_envs)
        
        obj_pos = self.target_object.get_pos()
        distance = torch.norm(obj_pos - self.target_pos, dim=-1)
        
        reward = -distance
        success = distance < 0.05
        reward += success.float() * 10.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check success."""
        if self.target_object is None or self.target_pos is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        
        obj_pos = self.target_object.get_pos()
        distance = torch.norm(obj_pos - self.target_pos, dim=-1)
        return distance < 0.05
