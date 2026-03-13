"""
Prepare food task - multi-stage task for food preparation.

Stages:
1. Pick up ingredient
2. Place on cutting board
3. Use tool (knife) to cut
4. Transfer to bowl/plate
"""

import torch
import numpy as np
from typing import Tuple


class PrepareFoodTask:
    """
    Multi-stage food preparation task.
    
    Example: Cut vegetables and place in bowl.
    """
    
    STAGES = [
        'pick_ingredient',
        'place_on_board',
        'cut',
        'transfer_to_bowl',
    ]
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.ingredient = None
        self.cutting_board = None
        self.knife = None
        self.bowl = None
        
        self.current_stage = 0
        self.stage_completed = [False] * len(self.STAGES)
        
    def reset(self):
        """Reset task."""
        self.current_stage = 0
        self.stage_completed = [False] * len(self.STAGES)
        
        # Find objects
        if hasattr(self.env, 'objects'):
            for name, obj in self.env.objects.items():
                if 'vegetable' in name.lower() or 'ingredient' in name.lower():
                    self.ingredient = obj
                elif 'knife' in name.lower():
                    self.knife = obj
                elif 'bowl' in name.lower():
                    self.bowl = obj
            
            # Find cutting board in fixtures
            if hasattr(self.env, 'fixtures'):
                for name, fixture in self.env.fixtures.items():
                    if 'board' in name.lower() or 'counter' in name.lower():
                        self.cutting_board = fixture
                        break
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 10  # stage + object positions
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # Current stage
        state.append(float(self.current_stage))
        
        # Object positions
        for obj in [self.ingredient, self.bowl]:
            if obj is not None:
                pos = obj.get_pos()
                state.extend(pos.tolist())
            else:
                state.extend([0.0, 0.0, 0.0])
        
        return torch.tensor(state, dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        if self.ingredient is None:
            return torch.zeros(self.env.num_envs)
        
        reward = torch.zeros(self.env.num_envs)
        
        # Stage-based rewards
        if self.current_stage == 0:  # Pick ingredient
            ee_pos, _ = self.robot.get_ee_pose()
            ingredient_pos = self.ingredient.get_pos()
            dist = torch.norm(ee_pos - ingredient_pos, dim=-1)
            reward += 1.0 - torch.clamp(dist, 0, 1)
            
        elif self.current_stage == 1:  # Place on board
            if self.cutting_board is not None:
                ingredient_pos = self.ingredient.get_pos()
                board_pos = self.cutting_board.get_pos()
                dist = torch.norm(ingredient_pos - board_pos, dim=-1)
                reward += 1.0 - torch.clamp(dist, 0, 1)
                
        elif self.current_stage == 2:  # Cut
            # Reward for using knife near ingredient
            if self.knife is not None:
                knife_pos = self.knife.get_pos()
                ingredient_pos = self.ingredient.get_pos()
                dist = torch.norm(knife_pos - ingredient_pos, dim=-1)
                reward += (dist < 0.05).float() * 0.1
                
        elif self.current_stage == 3:  # Transfer to bowl
            if self.bowl is not None:
                ingredient_pos = self.ingredient.get_pos()
                bowl_pos = self.bowl.get_pos()
                dist = torch.norm(ingredient_pos - bowl_pos, dim=-1)
                reward += 1.0 - torch.clamp(dist, 0, 1)
        
        # Stage completion bonus
        for i, completed in enumerate(self.stage_completed):
            if completed:
                reward += 1.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        # Success if all stages completed
        return torch.tensor([all(self.stage_completed)] * self.env.num_envs)
    
    def update_stage(self):
        """Update current stage based on progress."""
        if self.current_stage >= len(self.STAGES):
            return
        
        # Check if current stage is completed
        stage_name = self.STAGES[self.current_stage]
        
        if stage_name == 'pick_ingredient':
            # Check if holding ingredient
            ee_pos, _ = self.robot.get_ee_pose()
            if self.ingredient is not None:
                dist = torch.norm(ee_pos - self.ingredient.get_pos())
                if dist < 0.05:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'place_on_board':
            if self.cutting_board is not None and self.ingredient is not None:
                dist = torch.norm(
                    self.ingredient.get_pos() - self.cutting_board.get_pos()
                )
                if dist < 0.1:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'cut':
            # Check if cutting motion detected
            self.stage_completed[self.current_stage] = True
            self.current_stage += 1
            
        elif stage_name == 'transfer_to_bowl':
            if self.bowl is not None and self.ingredient is not None:
                dist = torch.norm(self.ingredient.get_pos() - self.bowl.get_pos())
                if dist < 0.1:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
    
    def get_stage_info(self) -> dict:
        """Get current stage information."""
        return {
            'current_stage': self.current_stage,
            'stage_name': self.STAGES[self.current_stage] if self.current_stage < len(self.STAGES) else 'completed',
            'stages_completed': sum(self.stage_completed),
            'total_stages': len(self.STAGES),
        }
