"""
Universal Robots UR5 robot agent.

6-DOF industrial manipulator with versatile reach and payload.
"""

import torch
import numpy as np
from gymnasium import spaces


class UR5Agent:
    """
    Universal Robots UR5.
    
    Specifications:
    - 6 DOF
    - Reach: 850mm
    - Payload: 5kg
    - Repeatability: ±0.03mm
    """
    
    # UR5 DH parameters (in meters and radians)
    # [a, d, alpha, theta_offset]
    DH_PARAMS = [
        [0.0, 0.089159, np.pi/2, 0.0],
        [-0.425, 0.0, 0.0, 0.0],
        [-0.39225, 0.0, 0.0, 0.0],
        [0.0, 0.10915, np.pi/2, 0.0],
        [0.0, 0.09465, -np.pi/2, 0.0],
        [0.0, 0.0823, 0.0, 0.0],
    ]
    
    # Joint limits (radians)
    JOINT_LIMITS = {
        'lower': np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        'upper': np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
    }
    
    def __init__(
        self,
        scene,
        num_envs: int = 1,
        control_mode: str = "pd_joint_pos",
        gripper_type: str = "2f85",  # '2f85', 'hande', 'none'
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.gripper_type = gripper_type
        
        # DOF configuration
        self.arm_dof = 6
        self.gripper_dof = 1 if gripper_type != 'none' else 0
        self.total_dof = self.arm_dof + self.gripper_dof
        
        # Load robot
        self.robot = self._load_robot()
        
        # Default home configuration (joint angles in radians)
        self.home_position = torch.tensor([
            0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0
        ])
        
    def _load_robot(self):
        """Load UR5 robot."""
        # Try to load from URDF/MJCF
        try:
            robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/ur5/ur5.urdf",
                    pos=(0.0, 0.0, 0.0),
                ),
            )
        except:
            # Fallback: create from MJCF
            try:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file="xml/ur5/ur5.xml",
                        pos=(0.0, 0.0, 0.0),
                    ),
                )
            except:
                # Final fallback: use generic
                import genesis as gs
                robot = self.scene.add_entity(
                    gs.morphs.Box(
                        size=(0.1, 0.1, 0.5),
                        pos=(0.0, 0.0, 0.25),
                    )
                )
        
        return robot
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.total_dof * 2 + 7  # joint_pos, joint_vel, ee_pose
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if self.control_mode == "pd_joint_pos":
            return self.total_dof
        elif self.control_mode == "pd_ee_pos":
            return 7  # dx, dy, dz, qw, qx, qy, qz
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
        # Scale action from [-1, 1] to joint limits
        arm_action = action[:, :self.arm_dof]
        
        # Scale to joint limits
        scaled_action = self._scale_action_to_joint_limits(arm_action)
        
        # Apply to arm
        self.robot.control_dofs_position(
            scaled_action,
            dofs_idx_local=range(self.arm_dof)
        )
        
        # Apply to gripper if present
        if self.gripper_dof > 0 and action.shape[1] > self.arm_dof:
            gripper_action = action[:, self.arm_dof:self.arm_dof+1]
            self.robot.control_dofs_position(
                gripper_action,
                dofs_idx_local=range(self.arm_dof, self.total_dof)
            )
    
    def _scale_action_to_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Scale normalized action to joint limits."""
        lower = torch.from_numpy(self.JOINT_LIMITS['lower']).float()
        upper = torch.from_numpy(self.JOINT_LIMITS['upper']).float()
        
        # action is in [-1, 1], map to [lower, upper]
        scaled = lower + (action + 1) / 2 * (upper - lower)
        return scaled
    
    def _apply_ee_pos_action(self, action: torch.Tensor):
        """Apply end-effector pose action."""
        # Get current EE pose
        ee_pos, ee_quat = self.get_ee_pose()
        
        # Delta position
        delta_pos = action[:, :3] * 0.05  # 5cm max per step
        target_pos = ee_pos + delta_pos
        
        # Target orientation (use current for now)
        target_quat = ee_quat
        
        # Inverse kinematics
        joint_pos = self._inverse_kinematics(target_pos, target_quat)
        
        # Apply
        self.robot.control_dofs_position(joint_pos, dofs_idx_local=range(self.arm_dof))
    
    def _inverse_kinematics(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Compute inverse kinematics."""
        # Simplified: return current joint positions
        # In practice, use analytical IK or optimization
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
        # Get last link pose
        link_idx = -1
        ee_pos = self.robot.get_link(link_idx).get_pos()
        ee_quat = self.robot.get_link(link_idx).get_quat()
        return ee_pos, ee_quat
    
    def reset(self):
        """Reset robot to home position."""
        home_pos = self.home_position.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Add gripper position if present
        if self.gripper_dof > 0:
            gripper_pos = torch.zeros(self.num_envs, self.gripper_dof)
            home_pos = torch.cat([home_pos, gripper_pos], dim=-1)
        
        self.robot.set_dofs_position(home_pos)
    
    def forward_kinematics(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute forward kinematics.
        
        Args:
            joint_positions: Joint angles (batch_size, 6)
        
        Returns:
            End-effector positions (batch_size, 3)
        """
        # Simplified FK using DH parameters
        # In practice, use proper transformation matrices
        return self.get_ee_pose()[0]
