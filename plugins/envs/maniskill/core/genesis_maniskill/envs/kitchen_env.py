"""
Kitchen environment based on RoboCasa design.
Adapted for Genesis backend.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import genesis as gs
from gymnasium import spaces

from genesis_maniskill.envs.base_env import BaseEnv
from genesis_maniskill.scenes.kitchen_scene import KitchenSceneBuilder


class KitchenEnv(BaseEnv):
    """
    Kitchen environment for everyday manipulation tasks.
    
    Supports:
    - Multiple kitchen layouts (G-shaped, U-shaped, L-shaped, etc.)
    - Multiple kitchen styles (modern, industrial, mediterranean, etc.)
    - 2500+ kitchen objects
    - Various manipulation tasks
    
    Args:
        layout_id: Kitchen layout ID (0-11)
        style_id: Kitchen style ID (0-7)
        layout_type: Optional layout type string
        style_type: Optional style type string
        **kwargs: Additional arguments passed to BaseEnv
    """
    
    # Layout types (same as RoboCasa)
    LAYOUTS = [
        "G-shaped", "U-shaped", "L-shaped", "single_wall",
        "island", "galley", "L-shaped_v2", "U-shaped_v2",
        "G-shaped_v2", "L-shaped_v3", "island_v2", "galley_v2"
    ]
    
    # Style types (same as RoboCasa)
    STYLES = [
        "modern", "industrial", "mediterranean", "transitional",
        "rustic", "farmhouse", "coastal", "traditional"
    ]
    
    def __init__(
        self,
        layout_id: int = 0,
        style_id: int = 0,
        layout_type: Optional[str] = None,
        style_type: Optional[str] = None,
        **kwargs
    ):
        # Resolve layout and style
        if layout_type is not None:
            layout_id = self.LAYOUTS.index(layout_type)
        if style_type is not None:
            style_id = self.STYLES.index(style_type)
            
        self.layout_id = layout_id
        self.style_id = style_id
        self.layout_type = self.LAYOUTS[layout_id]
        self.style_type = self.STYLES[style_id]
        
        # Initialize base environment
        super().__init__(
            scene_type="kitchen",
            **kwargs
        )
    
    def _build_scene(self):
        """Build kitchen scene."""
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=1,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            show_viewer=self.render_mode == "human",
        )
        
        # Build kitchen using scene builder
        self.scene_builder = KitchenSceneBuilder(
            scene=self.scene,
            layout_id=self.layout_id,
            style_id=self.style_id,
            num_envs=self.num_envs,
            config=self.scene_config
        )
        self.scene_builder.build()
        
        # Store references to fixtures and objects
        self.fixtures = self.scene_builder.fixtures
        self.objects = self.scene_builder.objects
        
        # Build scene
        self.scene.build()
    
    def _setup_cameras(self):
        """Setup kitchen cameras."""
        # Base camera - overhead view
        self.cameras["base_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(2.0, 2.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
        )
        
        # Additional cameras for different views
        self.cameras["robot_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(0.5, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
        )
        
        # Top-down view for planning
        self.cameras["top_camera"] = self.scene.add_camera(
            res=(128, 128),
            pos=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
        )
    
    def _get_state_obs_dim(self) -> int:
        """Get state observation dimension."""
        robot_dim = self.robot.state_dim
        # Add object states
        object_dim = len(self.objects) * 13  # pos(3) + rot(4) + vel(6)
        return robot_dim + object_dim
    
    def _get_state_obs(self) -> np.ndarray:
        """Get state observation."""
        robot_state = self.robot.get_state()
        
        # Get object states
        object_states = []
        for obj in self.objects.values():
            pos = obj.get_pos()
            rot = obj.get_quat()
            vel = obj.get_vel()
            ang_vel = obj.get_ang()
            obj_state = torch.cat([pos, rot, vel, ang_vel], dim=-1)
            object_states.append(obj_state)
        
        if object_states:
            object_states = torch.cat(object_states, dim=-1)
            obs = torch.cat([robot_state, object_states], dim=-1)
        else:
            obs = robot_state
            
        return obs.cpu().numpy()
    
    def _reset_scene(self):
        """Reset kitchen scene."""
        super()._reset_scene()
        
        # Randomize object positions
        self.scene_builder.randomize_object_placement()
        
        # Reset fixture states
        for fixture in self.fixtures.values():
            fixture.reset()
    
    def get_fixture(self, name: str):
        """Get a fixture by name."""
        return self.fixtures.get(name)
    
    def get_object(self, name: str):
        """Get an object by name."""
        return self.objects.get(name)
    
    def get_fixture_names(self) -> list:
        """Get list of fixture names."""
        return list(self.fixtures.keys())
    
    def get_object_names(self) -> list:
        """Get list of object names."""
        return list(self.objects.keys())
