"""
Kinova Gen3 robot agent.

7-DOF collaborative manipulator designed for human-robot interaction.
"""

import torch
import numpy as np
from gymnasium import spaces


class KinovaGen3Agent:
    """
    Kinova Gen3 Ultra lightweight robot.
    
    Specifications:
    - 7 DOF (redundant)
    - Reach: 902mm
    - Payload: 4kg
    - Weight: 8.2kg (ultra lightweight)
    - Repeatability: ±0.1mm
    """
    
    # Joint limits (radians) - Kinova Gen3
    JOINT_LIMITS = {
        'lower': np.array([
            -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi
        ]),
        'upper': np.array([
            np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi
        ]),
    }
    
    # Default home position
    HOME_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def __init__(
        self,
        scene,
        num_envs: int = 1,
        control_mode: str = "pd_joint_pos",
        gripper_type: str = "2f85",  # '2f85', 'robotiq_2f_85', 'none'
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.gripper_type = gripper_type
        
        # DOF configuration
        self.arm_dof = 7  # 7 DOF redundant arm
        self.gripper_dof = 1 if gripper_type != 'none' else 0
        self.total_dof = self.arm_dof + self.gripper_dof
        
        # Load robot
        self.robot = self._load_robot()
        
        # Default home configuration
        self.home_position = torch.from_numpy(self.HOME_POSITION).float()
        
    def _load_robot(self):
        """Load Kinova Gen3 robot."""
        try:
            robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/kinova_gen3/gen3.urdf",
                    pos=(0.0, 0.0, 0.0),
                ),
            )
        except:
            try:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file="xml/kinova/gen3.xml",
                        pos=(0.0, 0.0, 0.0),
                    ),
                )
            except:
                import genesis as gs
                robot = self.scene.add_entity(
                    gs.morphs.Box(
                        size=(0.1, 0.1, 0.6),
                        pos=(0.0, 0.0, 0.3),
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
            return 7  # position + quaternion
        elif self.control_mode == "pd_joint_vel":
            return self.total_dof
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
        elif self.control_mode == "pd_joint_vel":
            self._apply_joint_vel_action(action)
        elif self.control_mode == "pd_ee_pos":
            self._apply_ee_pos_action(action)
    
    def _apply_joint_pos_action(self, action: torch.Tensor):
        """Apply joint position action."""
        # Scale from [-1, 1] to joint limits
        arm_action = action[:, :self.arm_dof]
        scaled_action = self._scale_action_to_joint_limits(arm_action)
        
        self.robot.control_dofs_position(
            scaled_action,
            dofs_idx_local=range(self.arm_dof)
        )
        
        # Gripper
        if self.gripper_dof > 0 and action.shape[1] > self.arm_dof:
            gripper_action = action[:, self.arm_dof:self.arm_dof+1]
            self.robot.control_dofs_position(
                gripper_action,
                dofs_idx_local=range(self.arm_dof, self.total_dof)
            )
    
    def _apply_joint_vel_action(self, action: torch.Tensor):
        """Apply joint velocity action."""
        arm_action = action[:, :self.arm_dof]
        # Scale velocities
        max_vel = 1.0  # rad/s
        scaled_action = arm_action * max_vel
        
        self.robot.control_dofs_velocity(
            scaled_action,
            dofs_idx_local=range(self.arm_dof)
        )
    
    def _scale_action_to_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Scale normalized action to joint limits."""
        lower = torch.from_numpy(self.JOINT_LIMITS['lower']).float()
        upper = torch.from_numpy(self.JOINT_LIMITS['upper']).float()
        
        scaled = lower + (action + 1) / 2 * (upper - lower)
        return scaled
    
    def _apply_ee_pos_action(self, action: torch.Tensor):
        """Apply end-effector pose action."""
        ee_pos, ee_quat = self.get_ee_pose()
        
        delta_pos = action[:, :3] * 0.05
        target_pos = ee_pos + delta_pos
        
        # For now, keep current orientation
        target_quat = ee_quat
        
        # Redundant IK for 7DOF
        joint_pos = self._inverse_kinematics_7dof(target_pos, target_quat)
        
        self.robot.control_dofs_position(joint_pos, dofs_idx_local=range(self.arm_dof))
    
    def _inverse_kinematics_7dof(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Compute inverse kinematics for 7DOF arm."""
        # 7DOF has infinite solutions, we can optimize for:
        # - Minimum joint displacement
        # - Singularity avoidance
        # - Elbow position preference
        
        # Simplified: return current position
        return self.get_joint_positions()[:, :self.arm_dof]
    
    def get_state(self) -> torch.Tensor:
        """Get robot state."""
        qpos = self.get_joint_positions()
        qvel = self.get_joint_velocities()
        ee_pos, ee_quat = self.get_ee_pose()
        
        return torch.cat([qpos, qvel, ee_pos, ee_quat], dim=-1)
    
    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions."""
        return self.robot.get_dofs_position()
    
    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities."""
        return self.robot.get_dofs_velocity()
    
    def get_ee_pose(self) -> tuple:
        """Get end-effector pose."""
        link_idx = -1
        ee_pos = self.robot.get_link(link_idx).get_pos()
        ee_quat = self.robot.get_link(link_idx).get_quat()
        return ee_pos, ee_quat
    
    def get_null_space(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        Get null space of the Jacobian.
        
        For 7DOF robots, this allows self-motion without affecting EE pose.
        
        Args:
            joint_positions: Current joint positions
        
        Returns:
            Null space projection matrix
        """
        # Compute Jacobian (simplified)
        # In practice, use proper Jacobian computation
        J = torch.zeros(6, 7)
        
        # Compute null space
        I = torch.eye(7)
        J_pinv = J.T @ torch.inverse(J @ J.T + 1e-4 * torch.eye(6))
        N = I - J_pinv @ J
        
        return N
    
    def reset(self):
        """Reset robot to home position."""
        home_pos = self.home_position.unsqueeze(0).repeat(self.num_envs, 1)
        
        if self.gripper_dof > 0:
            gripper_pos = torch.zeros(self.num_envs, self.gripper_dof)
            home_pos = torch.cat([home_pos, gripper_pos], dim=-1)
        
        self.robot.set_dofs_position(home_pos)
    
    def set_joint_configuration(self, joint_positions: torch.Tensor, null_space_objective: str = 'none'):
        """
        Set joint configuration with optional null space optimization.
        
        Args:
            joint_positions: Target joint positions
            null_space_objective: 'none', 'min_displacement', 'elbow_up', etc.
        """
        if null_space_objective == 'none':
            self.robot.control_dofs_position(joint_positions)
        else:
            # Use redundancy for optimization
            current = self.get_joint_positions()
            null_space_motion = self._compute_null_space_motion(null_space_objective)
            
            # Combine primary task with null space motion
            target = joint_positions + 0.1 * null_space_motion
            self.robot.control_dofs_position(target)
    
    def _compute_null_space_motion(self, objective: str) -> torch.Tensor:
        """Compute null space motion for secondary objectives."""
        if objective == 'elbow_up':
            # Prefer elbow up configuration
            current = self.get_joint_positions()
            preferred = torch.zeros_like(current)
            preferred[:, 2] = np.pi / 4  # Prefer elbow joint up
            return preferred - current
        
        return torch.zeros(self.num_envs, self.arm_dof)
