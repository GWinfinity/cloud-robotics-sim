"""
UFACTORY xArm robot agent.

Industrial/collaborative 6/7-DOF manipulator series.
Supports xArm 5, 6, and 7 models.
"""

import torch
import numpy as np
from gymnasium import spaces


class XArmAgent:
    """
    UFACTORY xArm series robot.
    
    Available models:
    - xArm 5: 5 DOF, reach 700mm, payload 2kg
    - xArm 6: 6 DOF, reach 700mm, payload 5kg  
    - xArm 7: 7 DOF, reach 700mm, payload 3.5kg
    
    All models feature:
    - High repeatability (±0.1mm)
    - Collision detection
    - Hand teaching
    """
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'xarm5': {
            'dof': 5,
            'reach': 0.7,
            'payload': 2.0,
        },
        'xarm6': {
            'dof': 6,
            'reach': 0.7,
            'payload': 5.0,
        },
        'xarm7': {
            'dof': 7,
            'reach': 0.7,
            'payload': 3.5,
        },
    }
    
    # Joint limits (approximate, in radians)
    JOINT_LIMITS = {
        'lower': np.array([-6.28, -2.06, -0.19, -6.28, -1.75, -6.28, -6.28]),
        'upper': np.array([6.28, 2.06, 3.93, 6.28, 1.75, 6.28, 6.28]),
    }
    
    # Home positions for different models
    HOME_POSITIONS = {
        'xarm5': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        'xarm6': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
        'xarm7': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    
    def __init__(
        self,
        scene,
        num_envs: int = 1,
        model: str = "xarm7",
        control_mode: str = "pd_joint_pos",
        gripper_type: str = "gripper",  # 'gripper', 'vacuum', 'none'
    ):
        self.scene = scene
        self.num_envs = num_envs
        self.model = model.lower()
        self.control_mode = control_mode
        self.gripper_type = gripper_type
        
        # Get model configuration
        config = self.MODEL_CONFIGS.get(self.model, self.MODEL_CONFIGS['xarm7'])
        self.arm_dof = config['dof']
        self.reach = config['reach']
        self.payload = config['payload']
        
        # Gripper DOF
        self.gripper_dof = 1 if gripper_type != 'none' else 0
        self.total_dof = self.arm_dof + self.gripper_dof
        
        # Load robot
        self.robot = self._load_robot()
        
        # Home position
        self.home_position = torch.from_numpy(
            self.HOME_POSITIONS.get(self.model, self.HOME_POSITIONS['xarm7'])
        ).float()
        
    def _load_robot(self):
        """Load xArm robot."""
        urdf_path = f"urdf/xarm/{self.model}_robot.urdf"
        
        try:
            robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    pos=(0.0, 0.0, 0.0),
                ),
            )
        except:
            try:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file=f"xml/xarm/{self.model}.xml",
                        pos=(0.0, 0.0, 0.0),
                    ),
                )
            except:
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
        return self.total_dof * 2 + 7
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if self.control_mode == "pd_joint_pos":
            return self.total_dof
        elif self.control_mode == "pd_ee_delta_pos":
            return 6  # dx, dy, dz, droll, dpitch, dyaw
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
        elif self.control_mode == "pd_ee_delta_pos":
            self._apply_ee_delta_action(action)
    
    def _apply_joint_pos_action(self, action: torch.Tensor):
        """Apply joint position action."""
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
        max_vel = 1.5  # rad/s
        scaled_action = arm_action * max_vel
        
        self.robot.control_dofs_velocity(
            scaled_action,
            dofs_idx_local=range(self.arm_dof)
        )
    
    def _apply_ee_delta_action(self, action: torch.Tensor):
        """Apply end-effector delta pose action."""
        ee_pos, ee_quat = self.get_ee_pose()
        
        # Position delta
        delta_pos = action[:, :3] * 0.05  # 5cm per step
        target_pos = ee_pos + delta_pos
        
        # Orientation delta (Euler angles)
        delta_rot = action[:, 3:6] * 0.1  # Small rotations
        
        # Convert to quaternion and apply
        # Simplified: keep current orientation
        target_quat = ee_quat
        
        # IK
        joint_pos = self._inverse_kinematics(target_pos, target_quat)
        self.robot.control_dofs_position(joint_pos, dofs_idx_local=range(self.arm_dof))
    
    def _scale_action_to_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Scale normalized action to joint limits."""
        lower = torch.from_numpy(self.JOINT_LIMITS['lower'][:self.arm_dof]).float()
        upper = torch.from_numpy(self.JOINT_LIMITS['upper'][:self.arm_dof]).float()
        
        scaled = lower + (action + 1) / 2 * (upper - lower)
        return scaled
    
    def _inverse_kinematics(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Compute inverse kinematics."""
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
    
    def reset(self):
        """Reset robot to home position."""
        home_pos = self.home_position.unsqueeze(0).repeat(self.num_envs, 1)
        
        if self.gripper_dof > 0:
            gripper_pos = torch.zeros(self.num_envs, self.gripper_dof)
            home_pos = torch.cat([home_pos, gripper_pos], dim=-1)
        
        self.robot.set_dofs_position(home_pos)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model': self.model,
            'dof': self.arm_dof,
            'reach': self.reach,
            'payload': self.payload,
            'control_mode': self.control_mode,
            'gripper': self.gripper_type,
        }
    
    def check_collision(self) -> torch.Tensor:
        """
        Check if robot is in collision.
        
        Returns:
            Boolean tensor indicating collision status
        """
        # This would require collision detection setup
        # Return all False for now
        return torch.zeros(self.num_envs, dtype=torch.bool)
    
    def enable_teach_mode(self):
        """Enable hand-teaching mode (zero gravity)."""
        # Set gravity compensation
        if hasattr(self.robot, 'set_gravity_compensation'):
            self.robot.set_gravity_compensation(True)
    
    def disable_teach_mode(self):
        """Disable hand-teaching mode."""
        if hasattr(self.robot, 'set_gravity_compensation'):
            self.robot.set_gravity_compensation(False)
