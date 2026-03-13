"""
Pick and place task.
"""

import torch
import numpy as np


class PickPlaceTask:
    """
    Pick and place task.
    
    Robot must pick up an object and move it to a target location.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.target_object = None
        self.target_pos = None
        self.state_dim = 6  # object_pos (3) + target_pos (3)
        
    def reset(self):
        """Reset task."""
        # Select target object
        if hasattr(self.env, 'objects') and len(self.env.objects) > 0:
            obj_names = list(self.env.objects.keys())
            self.target_object = self.env.objects[obj_names[0]]
        
        # Set random target position
        if self.target_object is not None:
            # Random target location (on table or counter)
            self.target_pos = torch.tensor([
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.15, 0.15),
                0.05  # Height
            ])
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        if self.target_object is not None:
            obj_pos = self.target_object.get_pos()
            if self.target_pos is not None:
                return torch.cat([obj_pos, self.target_pos], dim=-1)
        
        return torch.zeros(self.env.num_envs, self.state_dim)
    
    @property
    def state_dim(self) -> int:
        """State dimension for task-specific info."""
        return 6
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if self.target_object is None:
            return torch.zeros(self.env.num_envs)
        
        # Get object position
        obj_pos = self.target_object.get_pos()
        
        # Distance to target
        if self.target_pos is not None:
            distance = torch.norm(obj_pos - self.target_pos, dim=-1)
            reward = -distance  # Negative distance as reward
            
            # Bonus for success
            success = distance < 0.05
            reward += success.float() * 10.0
            
            return reward
        
        return torch.zeros(self.env.num_envs)
    
    def check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        if self.target_object is None or self.target_pos is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        
        obj_pos = self.target_object.get_pos()
        distance = torch.norm(obj_pos - self.target_pos, dim=-1)
        return distance < 0.05
    
    def check_failure(self) -> torch.Tensor:
        """Check if task has failed."""
        # Task fails if object falls off table
        if self.target_object is not None:
            obj_pos = self.target_object.get_pos()
            # Check if object is too low (fell)
            return obj_pos[:, 2] < 0.1
        
        return torch.zeros(self.env.num_envs, dtype=torch.bool)
