"""
Assembly task - assemble multiple parts together.

Examples:
- Stack blocks in specific order
- Assemble toy parts
- Build simple structures
"""

import torch
import numpy as np
from typing import List, Tuple


class AssemblyTask:
    """
    Assembly task.
    
    Multi-stage task where robot must pick up parts and assemble them
    in a specific order/configuration.
    """
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = robot
        self.robot = robot
        
        self.parts: List = []
        self.assembly_positions: List = []
        self.assembly_order: List[int] = []
        
        self.parts_assembled: List[bool] = []
        self.current_part_idx = 0
        
        # Tolerance for successful placement
        self.position_tolerance = 0.03
        self.orientation_tolerance = 0.1
        
    def reset(self):
        """Reset task."""
        # Find assembly parts
        self.parts = []
        if hasattr(self.env, 'objects'):
            for name, obj in self.env.objects.items():
                if any(x in name.lower() for x in ['part', 'block', 'piece', 'component']):
                    self.parts.append(obj)
        
        # Define assembly positions (relative to base)
        self.assembly_positions = []
        base_pos = torch.tensor([0.0, 0.0, 0.05])  # Default base position
        
        for i, part in enumerate(self.parts):
            # Calculate target position (e.g., stacked vertically)
            target_pos = base_pos.clone()
            target_pos[2] += i * 0.05  # Stack on top of each other
            self.assembly_positions.append(target_pos)
        
        # Define assembly order
        self.assembly_order = list(range(len(self.parts)))
        
        self.parts_assembled = [False] * len(self.parts)
        self.current_part_idx = 0
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 4 + len(self.parts) * 6  # progress + part_pos/target_pos pairs
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # Assembly progress
        progress = sum(self.parts_assembled) / max(len(self.parts), 1)
        state.append(progress)
        
        # Current part index
        state.append(self.current_part_idx / max(len(self.parts), 1))
        
        # Current part and target positions
        if self.current_part_idx < len(self.parts):
            current_part = self.parts[self.current_part_idx]
            current_target = self.assembly_positions[self.current_part_idx]
            
            state.extend(current_part.get_pos().tolist())
            state.extend(current_target.tolist())
        else:
            state.extend([0.0] * 6)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if not self.parts:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Check for newly assembled parts
        self._check_assembly()
        
        # Reward for assembled parts
        for assembled in self.parts_assembled:
            if assembled:
                reward += 10.0
        
        # Reward for approaching current part target
        if self.current_part_idx < len(self.parts):
            current_part = self.parts[self.current_part_idx]
            current_target = self.assembly_positions[self.current_part_idx]
            
            if not self.parts_assembled[self.current_part_idx]:
                part_pos = current_part.get_pos()
                dist = torch.norm(part_pos - current_target, dim=-1)
                reward += 0.5 * (1.0 - torch.clamp(dist, 0, 1))
        
        # Sequential bonus
        for i, assembled in enumerate(self.parts_assembled):
            if assembled and i == self.assembly_order[i]:
                reward += 1.0  # Bonus for correct order
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if all parts are assembled."""
        if not self.parts:
            return torch.ones(self.env.num_envs, dtype=torch.bool)
        
        success = all(self.parts_assembled)
        return torch.tensor([success] * self.env.num_envs)
    
    def _check_assembly(self):
        """Check and update assembly status."""
        for i, (part, assembled) in enumerate(zip(self.parts, self.parts_assembled)):
            if assembled:
                continue
            
            target_pos = self.assembly_positions[i]
            part_pos = part.get_pos()
            
            dist = torch.norm(part_pos - target_pos)
            
            if dist < self.position_tolerance:
                self.parts_assembled[i] = True
                
                # Update current part index
                if i == self.current_part_idx:
                    self.current_part_idx += 1
    
    def get_progress(self) -> float:
        """Get assembly progress (0.0 to 1.0)."""
        if not self.parts:
            return 1.0
        return sum(self.parts_assembled) / len(self.parts)
    
    def get_current_part(self) -> Tuple[object, torch.Tensor]:
        """Get the current part to assemble and its target position."""
        if self.current_part_idx < len(self.parts):
            part = self.parts[self.current_part_idx]
            target = self.assembly_positions[self.current_part_idx]
            return part, target
        return None, None
    
    def get_assembly_structure(self) -> dict:
        """Get the target assembly structure."""
        return {
            'num_parts': len(self.parts),
            'assembly_order': self.assembly_order,
            'target_positions': [p.tolist() for p in self.assembly_positions],
        }
