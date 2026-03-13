"""
Sort objects task - arrange objects by type or color.

Examples:
- Sort blocks by color into different bins
- Arrange objects by size
- Group similar items together
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class SortTask:
    """
    Sorting task.
    
    Robot must classify objects and place them in appropriate containers
    based on their properties (color, size, type).
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.objects_to_sort: List = []
        self.sort_bins: Dict[str, object] = {}
        
        # Object -> target bin mapping
        self.object_targets: Dict[object, str] = {}
        self.objects_sorted: List[bool] = []
        
        # Classification criteria
        self.sort_by = 'color'  # 'color', 'size', 'type'
        
    def reset(self):
        """Reset task."""
        # Find objects to sort
        self.objects_to_sort = []
        if hasattr(self.env, 'objects'):
            for name, obj in self.env.objects.items():
                # Exclude non-sortable objects
                if not any(x in name.lower() for x in ['fixed', 'wall', 'floor', 'table']):
                    self.objects_to_sort.append(obj)
        
        # Find sort bins
        self.sort_bins = {}
        if hasattr(self.env, 'fixtures') or hasattr(self.env, 'objects'):
            all_objects = {}
            if hasattr(self.env, 'fixtures'):
                all_objects.update(self.env.fixtures)
            if hasattr(self.env, 'objects'):
                all_objects.update(self.env.objects)
            
            for name, obj in all_objects.items():
                if 'bin' in name.lower() or 'container' in name.lower() or 'zone' in name.lower():
                    # Determine bin type from name
                    if 'red' in name.lower():
                        self.sort_bins['red'] = obj
                    elif 'blue' in name.lower():
                        self.sort_bins['blue'] = obj
                    elif 'green' in name.lower():
                        self.sort_bins['green'] = obj
                    elif 'large' in name.lower():
                        self.sort_bins['large'] = obj
                    elif 'small' in name.lower():
                        self.sort_bins['small'] = obj
                    else:
                        self.sort_bins[f'bin_{name}'] = obj
        
        # Assign targets based on object properties
        self.object_targets = {}
        for i, obj in enumerate(self.objects_to_sort):
            # Simplified: assign based on index
            bin_names = list(self.sort_bins.keys())
            if bin_names:
                target_bin = bin_names[i % len(bin_names)]
                self.object_targets[obj] = target_bin
        
        self.objects_sorted = [False] * len(self.objects_to_sort)
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 3 + len(self.objects_to_sort) * 6  # ee_pos + (obj_pos + bin_pos) pairs
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # End-effector position
        ee_pos, _ = self.robot.get_ee_pose()
        state.extend(ee_pos.tolist())
        
        # Object and target bin positions
        for obj in self.objects_to_sort[:3]:  # Limit to 3 objects
            # Object position
            obj_pos = obj.get_pos()
            state.extend(obj_pos.tolist())
            
            # Target bin position
            target_bin = self.object_targets.get(obj)
            if target_bin and target_bin in self.sort_bins:
                bin_pos = self.sort_bins[target_bin].get_pos()
                state.extend(bin_pos.tolist())
            else:
                state.extend([0.0, 0.0, 0.0])
        
        # Pad to fixed size
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return torch.tensor(state[:self.state_dim], dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if not self.objects_to_sort:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Check and update sorted status
        self._check_sorted_status()
        
        # Reward for sorted objects
        for placed in self.objects_sorted:
            if placed:
                reward += 10.0
        
        # Distance-based reward for unsorted objects
        for obj, placed in zip(self.objects_to_sort, self.objects_sorted):
            if not placed:
                target_bin = self.object_targets.get(obj)
                if target_bin and target_bin in self.sort_bins:
                    obj_pos = obj.get_pos()
                    bin_pos = self.sort_bins[target_bin].get_pos()
                    
                    dist = torch.norm(obj_pos - bin_pos, dim=-1)
                    reward += 0.1 * (1.0 - torch.clamp(dist, 0, 1))
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if all objects are correctly sorted."""
        if not self.objects_to_sort:
            return torch.ones(self.env.num_envs, dtype=torch.bool)
        
        success = all(self.objects_sorted)
        return torch.tensor([success] * self.env.num_envs)
    
    def _check_sorted_status(self):
        """Update sorted status for each object."""
        for i, (obj, sorted_status) in enumerate(zip(self.objects_to_sort, self.objects_sorted)):
            if sorted_status:
                continue
            
            target_bin = self.object_targets.get(obj)
            if target_bin and target_bin in self.sort_bins:
                obj_pos = obj.get_pos()
                bin_pos = self.sort_bins[target_bin].get_pos()
                
                dist = torch.norm(obj_pos - bin_pos)
                if dist < 0.1:  # Within bin radius
                    self.objects_sorted[i] = True
    
    def get_progress(self) -> float:
        """Get sorting progress (0.0 to 1.0)."""
        if not self.objects_to_sort:
            return 1.0
        return sum(self.objects_sorted) / len(self.objects_to_sort)
    
    def get_next_object(self) -> Tuple[object, str]:
        """Get the next object to sort and its target bin."""
        for i, (obj, sorted_status) in enumerate(zip(self.objects_to_sort, self.objects_sorted)):
            if not sorted_status:
                target_bin = self.object_targets.get(obj, 'unknown')
                return obj, target_bin
        return None, None
    
    def get_classification_accuracy(self) -> float:
        """Get how many objects are in correct bins."""
        if not self.objects_to_sort:
            return 1.0
        return sum(self.objects_sorted) / len(self.objects_sorted)
