"""
Fourier GR-1 humanoid robot agent.
"""

import torch
import genesis as gs
from gymnasium import spaces


class GR1Agent:
    """
    Fourier GR-1 humanoid robot.
    
    Bipedal humanoid robot designed for general purpose tasks.
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        num_envs: int = 1,
        control_mode: str = "pd_joint_pos",
        floating_base: bool = True,
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.floating_base = floating_base
        
        # DOF configuration
        if floating_base:
            self.base_dof = 6
        else:
            self.base_dof = 0
            
        self.body_dof = 29
        self.total_dof = self.base_dof + self.body_dof
        
        # Load robot
        self.robot = self._load_robot()
        
    def _load_robot(self):
        """Load GR-1 robot."""
        robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/gr1/gr1.urdf",
                pos=(0.0, 0.0, 0.97),
                fixed=(not self.floating_base),
            ),
        )
        return robot
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.total_dof * 2 + 3
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return self.total_dof
    
    @property
    def action_space(self) -> spaces.Space:
        """Action space."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=float
        )
    
    def apply_action(self, action: torch.Tensor):
        """Apply action to robot."""
        self.robot.control_dofs_position(action)
    
    def get_state(self) -> torch.Tensor:
        """Get robot state."""
        qpos = self.robot.get_dofs_position()
        qvel = self.robot.get_dofs_velocity()
        base_pos = self.robot.get_pos()
        return torch.cat([qpos, qvel, base_pos], dim=-1)
    
    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions."""
        return self.robot.get_dofs_position()
    
    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities."""
        return self.robot.get_dofs_velocity()
    
    def reset(self):
        """Reset robot."""
        standing_pos = torch.zeros((self.num_envs, self.total_dof))
        self.robot.set_dofs_position(standing_pos)
