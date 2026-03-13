"""
Table-top manipulation environment.
Similar to ManiSkill's tabletop tasks but using Genesis backend.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import genesis as gs
from gymnasium import spaces

from genesis_maniskill.envs.base_env import BaseEnv
from genesis_maniskill.scenes.tabletop_scene import TableTopSceneBuilder


class TableTopEnv(BaseEnv):
    """
    Table-top manipulation environment.
    
    Supports various tabletop manipulation tasks:
    - Pick and place
    - Stack objects
    - Open/close containers
    - Push objects
    
    Args:
        table_size: Size of the table (length, width, height)
        num_objects: Number of objects in the scene
        object_types: Types of objects to spawn
        **kwargs: Additional arguments passed to BaseEnv
    """
    
    def __init__(
        self,
        table_size: Tuple[float, float, float] = (1.0, 0.6, 0.05),
        num_objects: int = 1,
        object_types: Optional[List[str]] = None,
        **kwargs
    ):
        self.table_size = table_size
        self.num_objects = num_objects
        self.object_types = object_types or ["cube"]
        
        super().__init__(
            scene_type="tabletop",
            **kwargs
        )
    
    def _build_scene(self):
        """Build tabletop scene."""
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=1,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, 1.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
            ),
            show_viewer=self.render_mode == "human",
        )
        
        # Build table-top scene
        self.scene_builder = TableTopSceneBuilder(
            scene=self.scene,
            table_size=self.table_size,
            num_objects=self.num_objects,
            object_types=self.object_types,
            num_envs=self.num_envs,
            config=self.scene_config
        )
        self.scene_builder.build()
        
        # Store references
        self.table = self.scene_builder.table
        self.objects = self.scene_builder.objects
        
        # Build scene
        self.scene.build()
    
    def _setup_cameras(self):
        """Setup tabletop cameras."""
        # Base camera
        self.cameras["base_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(0.5, 0.5, 0.8),
            lookat=(0.0, 0.0, 0.0),
            fov=45,
        )
        
        # Hand camera (mounted on robot)
        self.cameras["hand_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, -0.3),
            fov=60,
        )
        
        # Top-down view
        self.cameras["top_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
        )
    
    def _get_state_obs_dim(self) -> int:
        """Get state observation dimension."""
        robot_dim = self.robot.state_dim
        # Object states: pos(3) + quat(4) + vel(3) + ang_vel(3) = 13 per object
        object_dim = self.num_objects * 13
        return robot_dim + object_dim
    
    def _get_state_obs(self) -> np.ndarray:
        """Get state observation."""
        robot_state = self.robot.get_state()
        
        # Get object states
        object_states = []
        for obj in self.objects.values():
            pos = obj.get_pos()
            quat = obj.get_quat()
            vel = obj.get_vel()
            ang_vel = obj.get_ang()
            obj_state = torch.cat([pos, quat, vel, ang_vel], dim=-1)
            object_states.append(obj_state)
        
        if object_states:
            object_states = torch.cat(object_states, dim=-1)
            obs = torch.cat([robot_state, object_states], dim=-1)
        else:
            obs = robot_state
            
        return obs.cpu().numpy()
    
    def _reset_scene(self):
        """Reset tabletop scene."""
        super()._reset_scene()
        
        # Randomize object positions on table
        self.scene_builder.randomize_object_placement()
    
    def get_object_position(self, object_name: str) -> torch.Tensor:
        """Get object position."""
        if object_name in self.objects:
            return self.objects[object_name].get_pos()
        raise ValueError(f"Object {object_name} not found")
    
    def get_object_pose(self, object_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get object pose (position, quaternion)."""
        if object_name in self.objects:
            obj = self.objects[object_name]
            return obj.get_pos(), obj.get_quat()
        raise ValueError(f"Object {object_name} not found")
    
    def get_table_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get table bounds (min, max)."""
        table_half_size = torch.tensor([
            self.table_size[0] / 2,
            self.table_size[1] / 2,
            self.table_size[2]
        ])
        table_pos = self.table.get_pos()
        min_bounds = table_pos - table_half_size
        max_bounds = table_pos + table_half_size
        return min_bounds, max_bounds
