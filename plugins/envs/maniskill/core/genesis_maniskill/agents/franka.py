"""
Franka Emika Panda robot agent.
"""

import numpy as np
import torch
import genesis as gs
from gymnasium import spaces


class FrankaAgent:
    """
    Franka Emika Panda robot.
    
    7-DOF arm with parallel jaw gripper.
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        num_envs: int = 1,
        control_mode: str = "pd_joint_pos",
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.control_mode = control_mode
        
        # Robot state dimensions
        self.arm_dof = 7
        self.gripper_dof = 2
        self.total_dof = self.arm_dof + self.gripper_dof
        
        # Load robot
        self.robot = self._load_robot()
        
        # Control parameters
        self.arm_qpos = torch.zeros((num_envs, self.arm_dof), dtype=torch.float32)
        self.gripper_qpos = torch.zeros((num_envs, self.gripper_dof), dtype=torch.float32)
        
    def _load_robot(self):
        """Load Franka robot."""
        # Use Genesis built-in Franka or load from MJCF/URDF
        robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.0, 0.0, 0.0),
            ),
        )
        return robot
    
    @property
    def state_dim(self) -> int:
        """State dimension (joint positions + velocities + gripper)."""
        return self.total_dof * 2 + 6  # pos + vel + ee_pos + ee_quat
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if self.control_mode == "pd_joint_pos":
            return self.total_dof
        elif self.control_mode == "pd_ee_pos":
            return 6  # dx, dy, dz, roll, pitch, yaw + gripper
        else:
            return self.total_dof
    
    @property
    def action_space(self) -> spaces.Space:
        """Action space."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
    
    def apply_action(self, action: torch.Tensor):
        """Apply action to robot."""
        if self.control_mode == "pd_joint_pos":
            self._apply_joint_pos_action(action)
        elif self.control_mode == "pd_ee_pos":
            self._apply_ee_pos_action(action)
    
    def _apply_joint_pos_action(self, action: torch.Tensor):
        """Apply joint position action."""
        # Scale action to joint limits
        arm_action = action[:, :self.arm_dof]
        gripper_action = action[:, self.arm_dof:self.arm_dof + 1]
        
        # Set control (Genesis API)
        self.robot.control_dofs_position(arm_action, dofs_idx_local=range(self.arm_dof))
        
        # Control gripper (map single action to both fingers)
        gripper_pos = gripper_action.repeat(1, self.gripper_dof)
        self.robot.control_dofs_position(
            gripper_pos, 
            dofs_idx_local=range(self.arm_dof, self.total_dof)
        )
    
    def _apply_ee_pos_action(self, action: torch.Tensor):
        """Apply end-effector position action."""
        # Get current EE pose
        ee_pos, ee_quat = self.get_ee_pose()
        
        # Apply delta
        delta_pos = action[:, :3] * 0.05  # Scale to 5cm max
        delta_rot = action[:, 3:6] * 0.1  # Scale rotation
        
        target_pos = ee_pos + delta_pos
        # Target rotation would need proper quaternion math
        
        # Inverse kinematics to get joint positions
        # This is a placeholder - real impl would use IK
        joint_pos = self._ik(target_pos, ee_quat)
        
        # Apply to robot
        self.robot.control_dofs_position(joint_pos, dofs_idx_local=range(self.arm_dof))
    
    def _ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Inverse kinematics (placeholder)."""
        # Return current joint positions as placeholder
        return self.get_joint_positions()[:, :self.arm_dof]
    
    def get_state(self) -> torch.Tensor:
        """Get robot state."""
        # Joint positions and velocities
        qpos = self.get_joint_positions()
        qvel = self.get_joint_velocities()
        
        # End-effector pose
        ee_pos, ee_quat = self.get_ee_pose()
        
        # Concatenate
        state = torch.cat([qpos, qvel, ee_pos, ee_quat], dim=-1)
        return state
    
    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions."""
        return self.robot.get_dofs_position()
    
    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities."""
        return self.robot.get_dofs_velocity()
    
    def get_ee_pose(self) -> tuple:
        """Get end-effector pose (position, quaternion)."""
        # Get link position (assuming last link is EE)
        link_idx = -1
        ee_pos = self.robot.get_link(link_idx).get_pos()
        ee_quat = self.robot.get_link(link_idx).get_quat()
        return ee_pos, ee_quat
    
    def reset(self):
        """Reset robot state."""
        # Reset to home position
        home_pos = torch.zeros((self.num_envs, self.total_dof), dtype=torch.float32)
        home_pos[:, 0] = 0.0
        home_pos[:, 1] = -0.785
        home_pos[:, 2] = 0.0
        home_pos[:, 3] = -2.356
        home_pos[:, 4] = 0.0
        home_pos[:, 5] = 1.571
        home_pos[:, 6] = 0.785
        # Gripper open
        home_pos[:, 7:] = 0.04
        
        self.robot.set_dofs_position(home_pos)
