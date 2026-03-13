"""
Mobile manipulator agent - mobile base + robotic arm.

Combines locomotion and manipulation capabilities.
"""

import torch
import numpy as np
from gymnasium import spaces


class MobileManipulatorAgent:
    """
    Mobile manipulator combining a mobile base with a robotic arm.
    
    Common configurations:
    - Fetch: differential drive + 7DOF arm
    - Tiago: omnidirectional base + 7DOF arm
    - Stretch: lift + telescoping arm
    - RB-Kairos: omnidirectional + UR arm
    """
    
    # Configuration presets
    PRESETS = {
        'fetch': {
            'base_type': 'differential',
            'base_dof': 3,  # x, y, yaw
            'arm_type': 'fetch',
            'arm_dof': 7,
            'torso_dof': 1,  # prismatic lift
        },
        'tiago': {
            'base_type': 'omnidirectional',
            'base_dof': 4,  # x, y, yaw (holonomic)
            'arm_type': 'tiago',
            'arm_dof': 7,
            'torso_dof': 1,
        },
        'stretch': {
            'base_type': 'differential',
            'base_dof': 3,
            'arm_type': 'stretch',
            'arm_dof': 5,  # prismatic + revolute
            'lift_dof': 1,
        },
        'turtlebot_arm': {
            'base_type': 'differential',
            'base_dof': 3,
            'arm_type': 'pincher',
            'arm_dof': 5,
            'torso_dof': 0,
        },
    }
    
    def __init__(
        self,
        scene,
        num_envs: int = 1,
        preset: str = 'fetch',
        control_mode: str = "pd_joint_pos",
        arm_control_mode: str = "pd_joint_pos",
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.preset = preset.lower()
        self.control_mode = control_mode
        self.arm_control_mode = arm_control_mode
        
        # Get configuration
        config = self.PRESETS.get(self.preset, self.PRESETS['fetch'])
        self.base_type = config['base_type']
        self.base_dof = config['base_dof']
        self.arm_dof = config['arm_dof']
        self.torso_dof = config.get('torso_dof', 0)
        self.lift_dof = config.get('lift_dof', 0)
        
        self.arm_type = config['arm_type']
        
        # Total DOF
        self.total_dof = self.base_dof + self.torso_dof + self.lift_dof + self.arm_dof + 1  # +1 for gripper
        
        # Load robot
        self.robot = self._load_robot()
        
        # Home configuration
        self.home_position = self._get_home_position()
        
        # Navigation state
        self.base_position = torch.zeros(num_envs, 3)  # x, y, yaw
        self.target_position = None
        
    def _load_robot(self):
        """Load mobile manipulator."""
        # Try preset-specific URDF
        try:
            robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=f"urdf/{self.preset}/{self.preset}.urdf",
                    pos=(0.0, 0.0, 0.0),
                ),
            )
        except:
            try:
                # Try generic mobile manipulator
                robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=f"urdf/mobile_manipulator/{self.preset}.urdf",
                        pos=(0.0, 0.0, 0.0),
                    ),
                )
            except:
                import genesis as gs
                # Create simple representation
                robot = self.scene.add_entity(
                    gs.morphs.Box(
                        size=(0.3, 0.3, 0.8),
                        pos=(0.0, 0.0, 0.4),
                    )
                )
        
        return robot
    
    def _get_home_position(self) -> torch.Tensor:
        """Get default home configuration."""
        # Base: [x, y, yaw] or [vx, vy, omega]
        base_home = torch.zeros(self.base_dof)
        
        # Torso/lift
        torso_home = torch.zeros(self.torso_dof + self.lift_dof)
        
        # Arm
        arm_home = torch.zeros(self.arm_dof)
        
        # Gripper
        gripper_home = torch.zeros(1)
        
        return torch.cat([base_home, torso_home, arm_home, gripper_home])
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        # Base pose + arm joints + EE pose
        return 3 + self.arm_dof + 7  # base_xyz/base_vel + arm + ee_pose
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if self.control_mode == "pd_joint_pos":
            return self.total_dof
        elif self.control_mode == "base_vel_arm_pos":
            return self.base_dof + self.arm_dof + 1  # base vel + arm pos + gripper
        elif self.control_mode == "base_pos_arm_pos":
            return self.base_dof + self.arm_dof + 1  # base target + arm pos + gripper
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
            self._apply_full_joint_action(action)
        elif self.control_mode == "base_vel_arm_pos":
            self._apply_base_vel_arm_pos_action(action)
        elif self.control_mode == "base_pos_arm_pos":
            self._apply_base_pos_arm_pos_action(action)
    
    def _apply_full_joint_action(self, action: torch.Tensor):
        """Apply full joint position action."""
        # This assumes all DOFs are controlled
        self.robot.control_dofs_position(action)
    
    def _apply_base_vel_arm_pos_action(self, action: torch.Tensor):
        """Apply base velocity + arm position action."""
        # Split action
        base_action = action[:, :self.base_dof]
        arm_action = action[:, self.base_dof:self.base_dof + self.arm_dof]
        gripper_action = action[:, self.base_dof + self.arm_dof:self.base_dof + self.arm_dof + 1]
        
        # Scale base velocities
        max_linear_vel = 0.5  # m/s
        max_angular_vel = 1.0  # rad/s
        
        if self.base_type == 'differential':
            # [vx, vy, omega] -> [vx, 0, omega] for differential drive
            base_vel = torch.zeros_like(base_action)
            base_vel[:, 0] = base_action[:, 0] * max_linear_vel  # forward
            base_vel[:, 2] = base_action[:, 2] * max_angular_vel  # rotation
            
        elif self.base_type == 'omnidirectional':
            # [vx, vy, omega] all active
            base_vel = base_action.clone()
            base_vel[:, :2] *= max_linear_vel
            base_vel[:, 2] *= max_angular_vel
        
        # Apply base velocity
        if hasattr(self.robot, 'control_base_velocity'):
            self.robot.control_base_velocity(base_vel)
        
        # Apply arm position
        self.robot.control_dofs_position(
            arm_action,
            dofs_idx_local=range(self.base_dof + self.torso_dof + self.lift_dof,
                                self.base_dof + self.torso_dof + self.lift_dof + self.arm_dof)
        )
        
        # Apply gripper
        if gripper_action.shape[1] > 0:
            self.robot.control_dofs_position(
                gripper_action,
                dofs_idx_local=[-1]
            )
    
    def _apply_base_pos_arm_pos_action(self, action: torch.Tensor):
        """Apply base position target + arm position action."""
        # Base target position
        base_action = action[:, :self.base_dof]
        
        # Update target position
        if self.target_position is None:
            self.target_position = self.get_base_position()
        
        # Move toward target (simple proportional control)
        current_pos = self.get_base_position()
        target_delta = base_action * 0.1  # Scale to small steps
        
        # For position control, we'd use a controller to move to target
        # For now, use velocity control toward target
        direction = target_delta - current_pos
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)
        direction = direction / (direction_norm + 1e-6)
        
        velocity = direction * torch.min(direction_norm, torch.ones_like(direction_norm) * 0.5)
        
        if hasattr(self.robot, 'control_base_velocity'):
            self.robot.control_base_velocity(velocity)
        
        # Apply arm action
        arm_action = action[:, self.base_dof:self.base_dof + self.arm_dof]
        self.robot.control_dofs_position(
            arm_action,
            dofs_idx_local=range(self.base_dof + self.torso_dof + self.lift_dof,
                                self.base_dof + self.torso_dof + self.lift_dof + self.arm_dof)
        )
    
    def get_state(self) -> torch.Tensor:
        """Get robot state."""
        state = []
        
        # Base position
        base_pos = self.get_base_position()
        state.extend(base_pos.tolist())
        
        # Arm joint positions
        arm_qpos = self.get_arm_joint_positions()
        state.extend(arm_qpos.tolist())
        
        # End-effector pose
        ee_pos, ee_quat = self.get_ee_pose()
        state.extend(ee_pos.tolist())
        state.extend(ee_quat.tolist())
        
        return torch.tensor(state, dtype=torch.float32)
    
    def get_base_position(self) -> torch.Tensor:
        """Get base position [x, y, yaw]."""
        if hasattr(self.robot, 'get_pos'):
            pos = self.robot.get_pos()
            # For simplicity, assume base position is robot position
            # In practice, extract from appropriate link
            return pos[:, :2]  # x, y
        return torch.zeros(self.num_envs, 2)
    
    def get_arm_joint_positions(self) -> torch.Tensor:
        """Get arm joint positions."""
        all_qpos = self.robot.get_dofs_position()
        # Extract arm joints (skip base and torso)
        start_idx = self.base_dof + self.torso_dof + self.lift_dof
        end_idx = start_idx + self.arm_dof
        return all_qpos[:, start_idx:end_idx]
    
    def get_ee_pose(self) -> tuple:
        """Get end-effector pose."""
        link_idx = -1
        ee_pos = self.robot.get_link(link_idx).get_pos()
        ee_quat = self.robot.get_link(link_idx).get_quat()
        return ee_pos, ee_quat
    
    def reset(self):
        """Reset robot."""
        home_pos = self.home_position.unsqueeze(0).repeat(self.num_envs, 1)
        self.robot.set_dofs_position(home_pos)
        self.base_position = torch.zeros(self.num_envs, 3)
        self.target_position = None
    
    def navigate_to(self, target_position: torch.Tensor):
        """
        Set navigation target.
        
        Args:
            target_position: Target [x, y, yaw] or [x, y]
        """
        self.target_position = target_position
    
    def get_navigation_distance(self) -> torch.Tensor:
        """Get distance to navigation target."""
        if self.target_position is None:
            return torch.zeros(self.num_envs)
        
        current_pos = self.get_base_position()
        return torch.norm(current_pos - self.target_position[:, :2], dim=-1)
    
    def is_at_target(self, threshold: float = 0.1) -> torch.Tensor:
        """Check if robot is at navigation target."""
        return self.get_navigation_distance() < threshold
    
    def get_reachability(self, target_pos: torch.Tensor) -> torch.Tensor:
        """
        Check if target is reachable from current base position.
        
        Args:
            target_pos: Target position in world frame
        
        Returns:
            Boolean tensor indicating reachability
        """
        # Get base position
        base_pos = self.get_base_position()
        
        # Compute distance from base to target
        dist = torch.norm(target_pos[:, :2] - base_pos, dim=-1)
        
        # Check if within arm reach (simplified)
        arm_reach = 0.8  # meters
        return dist < arm_reach
    
    def compute_base_goal_for_target(self, target_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal base position to reach target.
        
        Args:
            target_pos: Target position in world frame
        
        Returns:
            Optimal base position [x, y, yaw]
        """
        # Simple heuristic: position base at optimal distance in front of target
        optimal_distance = 0.6  # meters
        
        # Direction to target
        base_pos = self.get_base_position()
        direction = target_pos[:, :2] - base_pos
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)
        
        # Position at optimal distance
        goal_pos = target_pos[:, :2] - direction * optimal_distance
        
        # Orient toward target
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        
        return torch.cat([goal_pos, yaw.unsqueeze(-1)], dim=-1)
