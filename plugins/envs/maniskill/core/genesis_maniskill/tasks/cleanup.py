"""
Cleanup task - clean up cluttered items.

Robot must pick up items from one location and place them
in their proper storage locations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class CleanupTask:
    """
    Cleanup task: Move items to their proper locations.
    
    Items start scattered on counter, must go to:
    - Cabinets (dishes, cups)
    - Fridge (perishables)
    - Trash (garbage)
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.items_to_clean: List = []
        self.storage_locations: Dict[str, object] = {}
        
        self.item_targets: Dict[object, str] = {}  # item -> target location name
        self.items_placed: List[bool] = []
        
    def reset(self):
        """Reset task."""
        # Find items to clean (scattered objects)
        self.items_to_clean = []
        if hasattr(self.env, 'objects'):
            for name, obj in self.env.objects.items():
                if not any(x in name.lower() for x in ['fixed', 'wall', 'floor']):
                    self.items_to_clean.append(obj)
        
        # Find storage locations
        self.storage_locations = {}
        if hasattr(self.env, 'fixtures'):
            for name, fixture in self.env.fixtures.items():
                if 'cabinet' in name.lower():
                    self.storage_locations[f'cabinet_{name}'] = fixture
                elif 'fridge' in name.lower():
                    self.storage_locations['fridge'] = fixture
                elif 'trash' in name.lower():
                    self.storage_locations['trash'] = fixture
        
        # Assign targets to items
        self.item_targets = {}
        for i, item in enumerate(self.items_to_clean):
            # Assign based on item type (simplified)
            target_name = list(self.storage_locations.keys())[i % len(self.storage_locations)]
            self.item_targets[item] = target_name
        
        self.items_placed = [False] * len(self.items_to_clean)
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 6 + len(self.items_to_clean) * 3  # ee_pos + item_positions
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # End-effector position
        ee_pos, _ = self.robot.get_ee_pose()
        state.extend(ee_pos.tolist())
        
        # Item positions
        for item in self.items_to_clean[:5]:  # Limit to 5 items
            pos = item.get_pos()
            state.extend(pos.tolist())
        
        # Pad if needed
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return torch.tensor(state[:self.state_dim], dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if not self.items_to_clean:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Reward for each correctly placed item
        for i, (item, placed) in enumerate(zip(self.items_to_clean, self.items_placed)):
            if placed:
                reward += 10.0  # Bonus for placing item
            else:
                # Reward for moving item toward target
                target_name = self.item_targets.get(item)
                if target_name and target_name in self.storage_locations:
                    target = self.storage_locations[target_name]
                    item_pos = item.get_pos()
                    target_pos = target.get_pos()
                    
                    dist = torch.norm(item_pos - target_pos, dim=-1)
                    reward += 0.1 * (1.0 - torch.clamp(dist, 0, 1))
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if all items are placed."""
        if not self.items_to_clean:
            return torch.ones(self.env.num_envs, dtype=torch.bool)
        
        success = all(self.items_placed)
        return torch.tensor([success] * self.env.num_envs)
    
    def check_item_placement(self):
        """Check and update item placements."""
        for i, item in enumerate(self.items_to_clean):
            if self.items_placed[i]:
                continue
            
            target_name = self.item_targets.get(item)
            if target_name and target_name in self.storage_locations:
                target = self.storage_locations[target_name]
                item_pos = item.get_pos()
                target_pos = target.get_pos()
                
                dist = torch.norm(item_pos - target_pos)
                if dist < 0.15:  # Within threshold
                    self.items_placed[i] = True
    
    def get_progress(self) -> float:
        """Get cleanup progress (0.0 to 1.0)."""
        if not self.items_to_clean:
            return 1.0
        return sum(self.items_placed) / len(self.items_placed)
    
    def get_next_target(self) -> Tuple[object, object]:
        """Get the next item to clean and its target location."""
        for i, (item, placed) in enumerate(zip(self.items_to_clean, self.items_placed)):
            if not placed:
                target_name = self.item_targets.get(item)
                target = self.storage_locations.get(target_name)
                return item, target
        return None, None
