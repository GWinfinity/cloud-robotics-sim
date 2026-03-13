"""
Organize cabinet task - arrange items in cabinet.

Robot must open cabinet, pick up items, and place them
in proper positions inside the cabinet.
"""

import torch
import numpy as np
from typing import List, Tuple


class OrganizeCabinetTask:
    """
    Organize cabinet task.
    
    Multi-step task:
    1. Open cabinet door
    2. Pick item from counter
    3. Place item in cabinet at target position
    4. Close cabinet door (optional)
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.cabinet = None
        self.items: List = []
        self.target_positions: List = []
        
        self.items_placed: List[bool] = []
        self.cabinet_opened = False
        
    def reset(self):
        """Reset task."""
        # Find cabinet
        self.cabinet = None
        if hasattr(self.env, 'fixtures'):
            for name, fixture in self.env.fixtures.items():
                if 'cabinet' in name.lower() or 'drawer' in name.lower():
                    self.cabinet = fixture
                    break
        
        # Find items to organize
        self.items = []
        if hasattr(self.env, 'objects'):
            # Get items that need to go in cabinet
            for name, obj in self.env.objects.items():
                if any(x in name.lower() for x in ['plate', 'cup', 'bowl', 'dish']):
                    self.items.append(obj)
        
        # Generate target positions inside cabinet
        self.target_positions = []
        if self.cabinet is not None:
            cabinet_pos = self.cabinet.get_pos()
            for i, item in enumerate(self.items):
                # Target positions on shelves
                offset = torch.tensor([
                    (i % 3 - 1) * 0.1,  # x spread
                    (i // 3) * 0.1,     # y spread  
                    0.05                # height
                ])
                target_pos = cabinet_pos + offset
                self.target_positions.append(target_pos)
        
        self.items_placed = [False] * len(self.items)
        self.cabinet_opened = False
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 4 + len(self.items) * 6  # cabinet_state + item_pos/target_pos
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # Cabinet state (0=closed, 1=open)
        state.append(1.0 if self.cabinet_opened else 0.0)
        
        # Number of items placed
        state.append(sum(self.items_placed) / max(len(self.items), 1))
        
        # Current item position and target
        current_idx = self._get_current_item_idx()
        if current_idx >= 0 and current_idx < len(self.items):
            item_pos = self.items[current_idx].get_pos()
            target_pos = self.target_positions[current_idx]
            state.extend(item_pos.tolist())
            state.extend(target_pos.tolist())
        else:
            state.extend([0.0] * 6)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def _get_current_item_idx(self) -> int:
        """Get index of current item to place."""
        for i, placed in enumerate(self.items_placed):
            if not placed:
                return i
        return -1
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if not self.items:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Reward for opening cabinet
        if self.cabinet_opened:
            reward += 1.0
        
        # Reward for placing items
        for i, (item, placed) in enumerate(zip(self.items, self.items_placed)):
            if placed:
                reward += 5.0
            else:
                # Distance to target
                item_pos = item.get_pos()
                target_pos = self.target_positions[i]
                dist = torch.norm(item_pos - target_pos, dim=-1)
                reward += 0.1 * (1.0 - torch.clamp(dist, 0, 1))
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if all items are placed in cabinet."""
        if not self.items:
            return torch.ones(self.env.num_envs, dtype=torch.bool)
        
        success = all(self.items_placed)
        return torch.tensor([success] * self.env.num_envs)
    
    def check_cabinet_opened(self):
        """Check if cabinet has been opened."""
        if self.cabinet is None:
            return
        
        # Check cabinet state (depends on cabinet implementation)
        # For now, assume it's opened if robot is near it
        ee_pos, _ = self.robot.get_ee_pose()
        cabinet_pos = self.cabinet.get_pos()
        dist = torch.norm(ee_pos - cabinet_pos)
        
        if dist < 0.2:
            self.cabinet_opened = True
    
    def check_item_placements(self):
        """Check if items are placed at target positions."""
        if not self.cabinet_opened:
            return
        
        for i, (item, placed) in enumerate(zip(self.items, self.items_placed)):
            if placed:
                continue
            
            item_pos = item.get_pos()
            target_pos = self.target_positions[i]
            dist = torch.norm(item_pos - target_pos)
            
            if dist < 0.05:  # Close enough
                self.items_placed[i] = True
    
    def get_progress(self) -> float:
        """Get task progress (0.0 to 1.0)."""
        if not self.items:
            return 1.0
        
        progress = sum(self.items_placed) / len(self.items)
        if self.cabinet_opened:
            progress += 0.1
        
        return min(progress, 1.0)
    
    def get_current_instruction(self) -> str:
        """Get instruction for current step."""
        if not self.cabinet_opened:
            return "Open the cabinet door"
        
        current_idx = self._get_current_item_idx()
        if current_idx >= 0:
            return f"Place {self.items[current_idx]} in cabinet"
        
        return "Task completed!"
