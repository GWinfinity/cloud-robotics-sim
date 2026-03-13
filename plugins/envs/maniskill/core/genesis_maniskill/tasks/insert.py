"""
Insert object task - insert one object into another.

Examples:
- Insert peg into hole
- Place cap on bottle
- Put block into container
"""

import torch
import numpy as np
from typing import Tuple


class InsertTask:
    """
    Insertion task.
    
    Robot must pick up an object and insert it into a target container.
    Requires precise alignment and orientation control.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.object_to_insert = None
        self.container = None
        self.insertion_point = None
        
        self.inserted = False
        self.near_threshold = 0.05
        self.insertion_threshold = 0.02
        
    def reset(self):
        """Reset task."""
        # Find objects
        if hasattr(self.env, 'objects'):
            obj_names = list(self.env.objects.keys())
            
            # Object to insert (usually smaller one)
            for name in obj_names:
                if any(x in name.lower() for x in ['peg', 'cap', 'block', 'plug']):
                    self.object_to_insert = self.env.objects[name]
                    break
            
            # Container (usually larger one)
            for name in obj_names:
                if any(x in name.lower() for x in ['hole', 'bottle', 'container', 'socket']):
                    self.container = self.env.objects[name]
                    break
        
        # Calculate insertion point (top of container)
        if self.container is not None:
            container_pos = self.container.get_pos()
            # Assume container is upright, insertion at top
            self.insertion_point = container_pos.clone()
            self.insertion_point[2] += 0.05  # Top of container
        
        self.inserted = False
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 12  # object_pos(3) + container_pos(3) + insertion_point(3) + ee_pos(3)
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # Object position
        if self.object_to_insert is not None:
            state.extend(self.object_to_insert.get_pos().tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # Container position
        if self.container is not None:
            state.extend(self.container.get_pos().tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # Insertion point
        if self.insertion_point is not None:
            state.extend(self.insertion_point.tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # End-effector position
        ee_pos, _ = self.robot.get_ee_pose()
        state.extend(ee_pos.tolist())
        
        return torch.tensor(state, dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if self.object_to_insert is None or self.container is None:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Distance from object to insertion point
        obj_pos = self.object_to_insert.get_pos()
        
        if self.insertion_point is not None:
            dist_to_insertion = torch.norm(obj_pos - self.insertion_point, dim=-1)
            
            # Dense reward for approaching insertion point
            reward += 1.0 - torch.clamp(dist_to_insertion, 0, 1)
            
            # Check if inserted
            if dist_to_insertion < self.insertion_threshold:
                reward += 10.0
                self.inserted = True
        
        # Additional reward for maintaining insertion
        if self.inserted:
            reward += 5.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if object is successfully inserted."""
        if self.object_to_insert is None or self.insertion_point is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        
        obj_pos = self.object_to_insert.get_pos()
        dist = torch.norm(obj_pos - self.insertion_point, dim=-1)
        
        success = dist < self.insertion_threshold
        return success
    
    def get_insertion_depth(self) -> torch.Tensor:
        """Get how deep the object is inserted (0.0 to 1.0)."""
        if self.object_to_insert is None or self.insertion_point is None:
            return torch.zeros(self.env.num_envs)
        
        obj_pos = self.object_to_insert.get_pos()
        dist = torch.norm(obj_pos - self.insertion_point, dim=-1)
        
        # Normalized depth (0 = at insertion point, 1 = fully inserted)
        depth = 1.0 - torch.clamp(dist / self.insertion_threshold, 0, 1)
        return depth
    
    def get_alignment_error(self) -> torch.Tensor:
        """Get alignment error between object and container."""
        if self.object_to_insert is None or self.container is None:
            return torch.zeros(self.env.num_envs)
        
        # Get orientations
        obj_quat = self.object_to_insert.get_quat()
        container_quat = self.container.get_quat()
        
        # Compute angular distance (simplified)
        # In practice, you'd use proper quaternion distance
        return torch.zeros(self.env.num_envs)  # Placeholder
