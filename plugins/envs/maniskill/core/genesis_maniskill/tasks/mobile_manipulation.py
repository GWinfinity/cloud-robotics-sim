"""
Mobile manipulation task - navigate and manipulate.

Robot must navigate to target location and perform manipulation.
Combines locomotion and manipulation skills.
"""

import torch
import numpy as np
from typing import Tuple


class MobileManipulationTask:
    """
    Mobile manipulation task.
    
    Multi-stage task:
    1. Navigate to target region
    2. Align with object
    3. Perform manipulation
    4. Navigate to delivery region
    """
    
    STAGES = [
        'navigate_to_object',
        'align',
        'manipulate',
        'navigate_to_delivery',
        'place',
    ]
    
    def __init__(self, env, scene, robot):
        self.env = env
        self.scene = scene
        self.robot = robot
        
        self.target_object = None
        self.start_position = None
        self.delivery_position = None
        
        self.current_stage = 0
        self.stage_completed = [False] * len(self.STAGES)
        
        # Navigation thresholds
        self.navigation_threshold = 0.5
        self.alignment_threshold = 0.2
        
    def reset(self):
        """Reset task."""
        # Save start position
        if hasattr(self.robot, 'get_pos'):
            self.start_position = self.robot.get_pos().clone()
        
        # Find target object
        if hasattr(self.env, 'objects'):
            obj_list = list(self.env.objects.values())
            if obj_list:
                self.target_object = obj_list[0]
        
        # Set delivery position (far from start)
        if self.start_position is not None:
            self.delivery_position = self.start_position.clone()
            self.delivery_position[0] += 1.0  # 1 meter away
        
        self.current_stage = 0
        self.stage_completed = [False] * len(self.STAGES)
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 12  # robot_pos(3) + target_pos(3) + delivery_pos(3) + stage(1) + progress(1) + ee_pos(2)
    
    def get_state(self) -> torch.Tensor:
        """Get task state."""
        state = []
        
        # Robot base position
        if hasattr(self.robot, 'get_pos'):
            state.extend(self.robot.get_pos().tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # Target object position
        if self.target_object is not None:
            state.extend(self.target_object.get_pos().tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # Delivery position
        if self.delivery_position is not None:
            state.extend(self.delivery_position.tolist())
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # Current stage
        state.append(float(self.current_stage))
        
        # Progress
        progress = sum(self.stage_completed) / len(self.STAGES)
        state.append(progress)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def compute_reward(self) -> torch.Tensor:
        """Compute reward."""
        reward = torch.zeros(self.env.num_envs)
        
        # Stage-specific rewards
        if self.current_stage == 0:  # Navigate to object
            if self.target_object is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                target_pos = self.target_object.get_pos()
                dist = torch.norm(robot_pos - target_pos, dim=-1)
                reward += 1.0 - torch.clamp(dist, 0, 1)
                
        elif self.current_stage == 1:  # Align
            if self.target_object is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                target_pos = self.target_object.get_pos()
                dist = torch.norm(robot_pos - target_pos, dim=-1)
                reward += 2.0 * (dist < self.alignment_threshold).float()
                
        elif self.current_stage == 2:  # Manipulate (pick up)
            if self.target_object is not None:
                # Check if holding object
                ee_pos, _ = self.robot.get_ee_pose()
                obj_pos = self.target_object.get_pos()
                dist = torch.norm(ee_pos - obj_pos, dim=-1)
                reward += (dist < 0.05).float()
                
        elif self.current_stage == 3:  # Navigate to delivery
            if self.delivery_position is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                dist = torch.norm(robot_pos - self.delivery_position, dim=-1)
                reward += 1.0 - torch.clamp(dist, 0, 1)
                
        elif self.current_stage == 4:  # Place
            if self.target_object is not None and self.delivery_position is not None:
                obj_pos = self.target_object.get_pos()
                dist = torch.norm(obj_pos - self.delivery_position, dim=-1)
                reward += 1.0 - torch.clamp(dist, 0, 1)
        
        # Stage completion bonus
        for i, completed in enumerate(self.stage_completed):
            if completed:
                reward += 5.0
        
        return reward
    
    def check_success(self) -> torch.Tensor:
        """Check if task is successful."""
        success = all(self.stage_completed)
        return torch.tensor([success] * self.env.num_envs)
    
    def update_stage(self):
        """Update current stage based on progress."""
        if self.current_stage >= len(self.STAGES):
            return
        
        stage_name = self.STAGES[self.current_stage]
        
        if stage_name == 'navigate_to_object':
            if self.target_object is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                target_pos = self.target_object.get_pos()
                dist = torch.norm(robot_pos - target_pos)
                if dist < self.navigation_threshold:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'align':
            if self.target_object is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                target_pos = self.target_object.get_pos()
                dist = torch.norm(robot_pos - target_pos)
                if dist < self.alignment_threshold:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'manipulate':
            # Check if picked up
            if self.target_object is not None:
                ee_pos, _ = self.robot.get_ee_pose()
                obj_pos = self.target_object.get_pos()
                dist = torch.norm(ee_pos - obj_pos)
                if dist < 0.05:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'navigate_to_delivery':
            if self.delivery_position is not None and hasattr(self.robot, 'get_pos'):
                robot_pos = self.robot.get_pos()
                dist = torch.norm(robot_pos - self.delivery_position)
                if dist < self.navigation_threshold:
                    self.stage_completed[self.current_stage] = True
                    self.current_stage += 1
                    
        elif stage_name == 'place':
            # Check if placed at delivery
            if self.target_object is not None and self.delivery_position is not None:
                obj_pos = self.target_object.get_pos()
                dist = torch.norm(obj_pos - self.delivery_position)
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
    
    def get_navigation_waypoints(self) -> list:
        """Get waypoints for navigation."""
        waypoints = []
        
        if self.current_stage <= 1:  # Going to object
            if self.target_object is not None:
                waypoints.append(self.target_object.get_pos())
        elif self.current_stage >= 3:  # Going to delivery
            if self.delivery_position is not None:
                waypoints.append(self.delivery_position)
        
        return waypoints
