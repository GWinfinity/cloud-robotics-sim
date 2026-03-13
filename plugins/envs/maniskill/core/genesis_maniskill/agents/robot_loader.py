"""
Generic robot loader supporting URDF/MJCF formats.

This module provides flexible robot loading from standard description formats,
eliminating the need for hardcoded robot classes.

Usage:
    # Load from URDF with auto-detected parameters
    robot = RobotLoader.from_urdf(
        scene=scene,
        urdf_path="path/to/robot.urdf",
        num_envs=16,
    )
    
    # Load with configuration file
    robot = RobotLoader.from_config(
        scene=scene,
        config_path="configs/franka.yaml",
        num_envs=16,
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import xml.etree.ElementTree as ET
import numpy as np
import torch
import yaml
import json
from gymnasium import spaces
import genesis as gs


class RobotConfig:
    """
    Robot configuration container.
    
    Attributes:
        name: Robot name
        urdf_path: Path to URDF file
        mjcf_path: Path to MJCF file (alternative to URDF)
        arm_dof: Number of arm joints
        gripper_dof: Number of gripper joints
        joint_limits: Dict with 'lower' and 'upper' arrays
        velocity_limits: Max joint velocities
        effort_limits: Max joint efforts
        home_position: Default joint configuration
        ee_link_name: Name of end-effector link
        base_link_name: Name of base link
        gripper_names: List of gripper joint names
        control_mode: Default control mode
        scale_actions: Whether to scale actions to joint limits
    """
    
    def __init__(
        self,
        name: str = "robot",
        urdf_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        arm_dof: Optional[int] = None,
        gripper_dof: int = 0,
        joint_limits: Optional[Dict] = None,
        velocity_limits: Optional[np.ndarray] = None,
        effort_limits: Optional[np.ndarray] = None,
        home_position: Optional[np.ndarray] = None,
        ee_link_name: str = "",
        base_link_name: str = "",
        gripper_names: Optional[List[str]] = None,
        control_mode: str = "pd_joint_pos",
        scale_actions: bool = True,
        default_gripper: str = "none",
        **kwargs
    ):
        self.name = name
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        self.arm_dof = arm_dof
        self.gripper_dof = gripper_dof
        self.joint_limits = joint_limits or {}
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits
        self.home_position = home_position
        self.ee_link_name = ee_link_name
        self.base_link_name = base_link_name
        self.gripper_names = gripper_names or []
        self.control_mode = control_mode
        self.scale_actions = scale_actions
        self.default_gripper = default_gripper
        self.metadata = kwargs
    
    @property
    def total_dof(self) -> int:
        """Total degrees of freedom."""
        return (self.arm_dof or 0) + self.gripper_dof
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "RobotConfig":
        """Create config from dictionary."""
        # Convert lists to numpy arrays
        if 'joint_limits' in config_dict:
            limits = config_dict['joint_limits']
            if isinstance(limits, dict):
                limits = {
                    'lower': np.array(limits['lower']),
                    'upper': np.array(limits['upper']),
                }
                config_dict['joint_limits'] = limits
        
        for key in ['velocity_limits', 'effort_limits', 'home_position']:
            if key in config_dict and config_dict[key] is not None:
                config_dict[key] = np.array(config_dict[key])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RobotConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RobotConfig":
        """Load config from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'urdf_path': self.urdf_path,
            'mjcf_path': self.mjcf_path,
            'arm_dof': self.arm_dof,
            'gripper_dof': self.gripper_dof,
            'joint_limits': {
                'lower': self.joint_limits.get('lower', []).tolist() if hasattr(self.joint_limits.get('lower'), 'tolist') else self.joint_limits.get('lower', []),
                'upper': self.joint_limits.get('upper', []).tolist() if hasattr(self.joint_limits.get('upper'), 'tolist') else self.joint_limits.get('upper', []),
            } if self.joint_limits else None,
            'home_position': self.home_position.tolist() if self.home_position is not None else None,
            'ee_link_name': self.ee_link_name,
            'control_mode': self.control_mode,
        }
    
    def save(self, path: Union[str, Path]):
        """Save config to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)


class RobotLoader:
    """
    Generic robot loader from URDF/MJCF.
    
    This class loads robots from standard description formats and
    provides a unified interface for control.
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        config: RobotConfig,
        num_envs: int = 1,
        control_mode: Optional[str] = None,
    ):
        """
        Initialize robot loader.
        
        Args:
            scene: Genesis scene
            config: Robot configuration
            num_envs: Number of parallel environments
            control_mode: Override control mode from config
        """
        self.scene = scene
        self.config = config
        self.num_envs = num_envs
        self.control_mode = control_mode or config.control_mode
        
        # Load the robot
        self.robot = self._load_robot()
        
        # Auto-detect parameters if not specified
        if config.arm_dof is None:
            self._detect_dof()
        
        if not config.joint_limits:
            self._detect_joint_limits()
        
        # Initialize home position
        if config.home_position is not None:
            self.home_position = torch.from_numpy(config.home_position).float()
        else:
            self.home_position = torch.zeros(self.config.total_dof)
    
    def _load_robot(self) -> gs.Entity:
        """Load robot from URDF or MJCF."""
        # Try URDF first
        if self.config.urdf_path and Path(self.config.urdf_path).exists():
            return self._load_urdf(self.config.urdf_path)
        
        # Try MJCF
        if self.config.mjcf_path and Path(self.config.mjcf_path).exists():
            return self._load_mjcf(self.config.mjcf_path)
        
        # Try to find in standard locations
        robot_name = self.config.name
        search_paths = [
            f"urdf/{robot_name}/{robot_name}.urdf",
            f"urdf/{robot_name}/robot.urdf",
            f"xml/{robot_name}/{robot_name}.xml",
            f"xml/{robot_name}/robot.xml",
        ]
        
        for path in search_paths:
            if Path(path).exists():
                if path.endswith('.urdf'):
                    return self._load_urdf(path)
                else:
                    return self._load_mjcf(path)
        
        raise FileNotFoundError(
            f"Could not find robot description for {robot_name}. "
            f"Searched: {self.config.urdf_path}, {self.config.mjcf_path}"
        )
    
    def _load_urdf(self, path: str) -> gs.Entity:
        """Load from URDF."""
        return self.scene.add_entity(
            gs.morphs.URDF(
                file=path,
                pos=(0.0, 0.0, 0.0),
            ),
        )
    
    def _load_mjcf(self, path: str) -> gs.Entity:
        """Load from MJCF."""
        return self.scene.add_entity(
            gs.morphs.MJCF(
                file=path,
                pos=(0.0, 0.0, 0.0),
            ),
        )
    
    def _detect_dof(self):
        """Auto-detect DOF from loaded robot."""
        # Get number of joints from robot
        try:
            qpos = self.robot.get_dofs_position()
            total_dof = qpos.shape[1]
            
            # Assume last N DOF are gripper if gripper_dof specified
            if self.config.gripper_dof > 0:
                self.config.arm_dof = total_dof - self.config.gripper_dof
            else:
                self.config.arm_dof = total_dof
                
        except:
            # Default to 7 if can't detect
            self.config.arm_dof = 7
    
    def _detect_joint_limits(self):
        """Auto-detect joint limits from URDF/MJCF."""
        # Try to parse from URDF
        if self.config.urdf_path and Path(self.config.urdf_path).exists():
            limits = self._parse_urdf_limits(self.config.urdf_path)
            self.config.joint_limits = limits
        
        # Default limits if not found
        if not self.config.joint_limits:
            dof = self.config.arm_dof or 7
            self.config.joint_limits = {
                'lower': np.array([-np.pi] * dof),
                'upper': np.array([np.pi] * dof),
            }
    
    def _parse_urdf_limits(self, path: str) -> Dict:
        """Parse joint limits from URDF."""
        tree = ET.parse(path)
        root = tree.getroot()
        
        lower_limits = []
        upper_limits = []
        
        for joint in root.findall('.//joint'):
            joint_type = joint.get('type', 'revolute')
            if joint_type in ['revolute', 'prismatic']:
                limit = joint.find('limit')
                if limit is not None:
                    lower = float(limit.get('lower', -np.pi))
                    upper = float(limit.get('upper', np.pi))
                    lower_limits.append(lower)
                    upper_limits.append(upper)
        
        if lower_limits and upper_limits:
            return {
                'lower': np.array(lower_limits),
                'upper': np.array(upper_limits),
            }
        
        return {}
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.config.total_dof * 2 + 7  # joints + velocities + ee_pose
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if self.control_mode == "pd_joint_pos":
            return self.config.total_dof
        elif self.control_mode in ["pd_ee_pos", "pd_ee_delta_pos"]:
            return 6  # position + orientation (simplified)
        elif self.control_mode == "pd_joint_vel":
            return self.config.total_dof
        else:
            return self.config.total_dof
    
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
        elif self.control_mode in ["pd_ee_pos", "pd_ee_delta_pos"]:
            self._apply_ee_action(action)
    
    def _apply_joint_pos_action(self, action: torch.Tensor):
        """Apply joint position action."""
        if self.config.scale_actions and self.config.joint_limits:
            action = self._scale_to_joint_limits(action)
        
        self.robot.control_dofs_position(action)
    
    def _apply_joint_vel_action(self, action: torch.Tensor):
        """Apply joint velocity action."""
        # Scale from [-1, 1] to velocity limits
        if self.config.velocity_limits is not None:
            max_vel = self.config.velocity_limits.max()
            action = action * max_vel
        else:
            action = action * 1.0  # 1 rad/s default
        
        self.robot.control_dofs_velocity(action)
    
    def _apply_ee_action(self, action: torch.Tensor):
        """Apply end-effector action."""
        # Get current pose
        ee_pos, ee_quat = self.get_ee_pose()
        
        if self.control_mode == "pd_ee_delta_pos":
            # Delta position
            delta_pos = action[:, :3] * 0.05
            target_pos = ee_pos + delta_pos
            
            # Keep current orientation (simplified)
            target_quat = ee_quat
        else:
            # Absolute position (simplified)
            target_pos = action[:, :3]
            target_quat = ee_quat
        
        # Compute IK
        joint_pos = self._inverse_kinematics(target_pos, target_quat)
        self.robot.control_dofs_position(joint_pos)
    
    def _scale_to_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Scale normalized action to joint limits."""
        lower = torch.from_numpy(self.config.joint_limits['lower']).float()
        upper = torch.from_numpy(self.config.joint_limits['upper']).float()
        
        # Handle dimension mismatch
        if action.shape[1] != lower.shape[0]:
            # Pad or truncate
            min_dim = min(action.shape[1], lower.shape[0])
            action = action[:, :min_dim]
            lower = lower[:min_dim]
            upper = upper[:min_dim]
        
        # Map from [-1, 1] to [lower, upper]
        scaled = lower + (action + 1) / 2 * (upper - lower)
        return scaled
    
    def _inverse_kinematics(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor
    ) -> torch.Tensor:
        """Compute inverse kinematics (placeholder)."""
        # In practice, use proper IK solver
        # For now, return current position
        return self.get_joint_positions()
    
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
    
    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get end-effector pose."""
        # Try to find EE link
        if self.config.ee_link_name:
            try:
                link = self.robot.get_link(self.config.ee_link_name)
                return link.get_pos(), link.get_quat()
            except:
                pass
        
        # Default to last link
        try:
            link_idx = -1
            ee_pos = self.robot.get_link(link_idx).get_pos()
            ee_quat = self.robot.get_link(link_idx).get_quat()
            return ee_pos, ee_quat
        except:
            # Fallback
            return torch.zeros(self.num_envs, 3), torch.zeros(self.num_envs, 4)
    
    def reset(self):
        """Reset to home position."""
        home = self.home_position.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Ensure correct shape
        current_dof = self.get_joint_positions().shape[1]
        if home.shape[1] != current_dof:
            if home.shape[1] < current_dof:
                # Pad with zeros
                padding = torch.zeros(self.num_envs, current_dof - home.shape[1])
                home = torch.cat([home, padding], dim=-1)
            else:
                # Truncate
                home = home[:, :current_dof]
        
        self.robot.set_dofs_position(home)
    
    # Factory methods
    @classmethod
    def from_urdf(
        cls,
        scene: gs.Scene,
        urdf_path: str,
        num_envs: int = 1,
        name: Optional[str] = None,
        **kwargs
    ) -> "RobotLoader":
        """
        Create loader from URDF file.
        
        Args:
            scene: Genesis scene
            urdf_path: Path to URDF file
            num_envs: Number of environments
            name: Robot name (optional)
            **kwargs: Additional config parameters
        
        Returns:
            RobotLoader instance
        """
        config = RobotConfig(
            name=name or Path(urdf_path).stem,
            urdf_path=urdf_path,
            **kwargs
        )
        return cls(scene, config, num_envs)
    
    @classmethod
    def from_mjcf(
        cls,
        scene: gs.Scene,
        mjcf_path: str,
        num_envs: int = 1,
        name: Optional[str] = None,
        **kwargs
    ) -> "RobotLoader":
        """Create loader from MJCF file."""
        config = RobotConfig(
            name=name or Path(mjcf_path).stem,
            mjcf_path=mjcf_path,
            **kwargs
        )
        return cls(scene, config, num_envs)
    
    @classmethod
    def from_config(
        cls,
        scene: gs.Scene,
        config_path: str,
        num_envs: int = 1,
    ) -> "RobotLoader":
        """Create loader from config file."""
        path = Path(config_path)
        
        if path.suffix in ['.yaml', '.yml']:
            config = RobotConfig.from_yaml(path)
        elif path.suffix == '.json':
            config = RobotConfig.from_json(path)
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
        
        return cls(scene, config, num_envs)
    
    @classmethod
    def from_preset(
        cls,
        scene: gs.Scene,
        preset_name: str,
        num_envs: int = 1,
    ) -> "RobotLoader":
        """Load from preset configuration."""
        # Search for preset config
        preset_paths = [
            Path(__file__).parent.parent / "configs" / "robots" / f"{preset_name}.yaml",
            Path(__file__).parent.parent / "configs" / "robots" / f"{preset_name}.json",
            Path("configs/robots") / f"{preset_name}.yaml",
            Path("configs/robots") / f"{preset_name}.json",
        ]
        
        for path in preset_paths:
            if path.exists():
                return cls.from_config(scene, str(path), num_envs)
        
        raise FileNotFoundError(
            f"Could not find preset config for {preset_name}. "
            f"Searched: {[str(p) for p in preset_paths]}"
        )


def get_robot(
    scene: gs.Scene,
    robot_uid: str,
    num_envs: int = 1,
    **kwargs
) -> RobotLoader:
    """
    Get robot by UID.
    
    Args:
        scene: Genesis scene
        robot_uid: Robot identifier (preset name or config path)
        num_envs: Number of environments
        **kwargs: Additional parameters
    
    Returns:
        RobotLoader instance
    """
    # Check if it's a path
    if Path(robot_uid).exists():
        if Path(robot_uid).suffix in ['.yaml', '.yml', '.json']:
            return RobotLoader.from_config(scene, robot_uid, num_envs)
        elif Path(robot_uid).suffix == '.urdf':
            return RobotLoader.from_urdf(scene, robot_uid, num_envs, **kwargs)
        elif Path(robot_uid).suffix == '.xml':
            return RobotLoader.from_mjcf(scene, robot_uid, num_envs, **kwargs)
    
    # Try preset
    try:
        return RobotLoader.from_preset(scene, robot_uid, num_envs)
    except FileNotFoundError:
        pass
    
    # Try to find URDF/MJCF by name
    search_paths = [
        f"urdf/{robot_uid}/{robot_uid}.urdf",
        f"urdf/{robot_uid}/robot.urdf",
        f"xml/{robot_uid}/{robot_uid}.xml",
        f"xml/{robot_uid}/robot.xml",
    ]
    
    for path in search_paths:
        if Path(path).exists():
            if path.endswith('.urdf'):
                return RobotLoader.from_urdf(scene, path, num_envs, name=robot_uid, **kwargs)
            else:
                return RobotLoader.from_mjcf(scene, path, num_envs, name=robot_uid, **kwargs)
    
    raise ValueError(
        f"Could not find robot: {robot_uid}. "
        f"Provide a config file path, URDF path, or preset name."
    )
