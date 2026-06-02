# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Genesis Physics Simulator Integration for DreamDojo.

This module provides integration with the Genesis physics simulator
for robotics simulation and data generation. Adapted from DreamDojo's
GenesisSimulator to work with genesis-cloud-sim's plugin architecture.

Requirements:
    pip install genesis-world

References:
    - Genesis: https://github.com/Genesis-Embodied-AI/Genesis
    - Documentation: https://genesis-embodied-ai.github.io/
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

# Optional heavy dependencies
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None

# Genesis imports (optional - will fail gracefully if not installed)
try:
    import genesis as gs
    from genesis.engine.entities import RigidEntity
    from genesis.engine.scene import Scene
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    gs = None
    Scene = None
    RigidEntity = None


class GenesisRobotType(Enum):
    """Supported robot types in Genesis."""
    HUMANOID = "humanoid"
    UR5 = "ur5"
    FRANKA = "franka"
    BAXTER = "baxter"
    G1 = "g1"
    GR1 = "gr1"
    CUSTOM = "custom"


@dataclass
class GenesisSimulatorConfig:
    """Configuration for Genesis simulator."""
    
    scene_config: Optional[Dict[str, Any]] = None
    robot_type: GenesisRobotType = GenesisRobotType.HUMANOID
    robot_urdf_path: Optional[str] = None
    control_freq: int = 20
    sim_freq: int = 100
    render_camera: Optional[str] = None
    headless: bool = True
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        if self.scene_config is None:
            self.scene_config = {
                "sim_options": {
                    "dt": 0.01,
                    "substeps": 10,
                },
                "viewer_options": {
                    "res": (1280, 720),
                    "camera_pos": (3.5, 0.0, 2.5),
                    "camera_lookat": (0.0, 0.0, 0.5),
                },
                "show_viewer": not self.headless,
            }


class GenesisSimulator:
    """
    Genesis physics simulator interface for DreamDojo.
    
    This class wraps Genesis simulator functionality for use with
    DreamDojo's world model training and inference.
    
    Compatible with genesis-cloud-sim's EnvironmentComposer architecture.
    """
    
    def __init__(self, config: GenesisSimulatorConfig):
        """
        Initialize Genesis simulator.
        
        Args:
            config: Simulator configuration
        """
        if not GENESIS_AVAILABLE:
            raise ImportError(
                "Genesis is not installed. Please install it with: "
                "pip install genesis-world"
            )
        
        self.config = config
        self.scene: Optional[Scene] = None
        self.robot: Optional[RigidEntity] = None
        self.camera: Optional[Any] = None
        self._initialized = False
        self._episode_count = 0
        
    def initialize(self):
        """Initialize the Genesis scene and robot."""
        if self._initialized:
            return
            
        # Initialize Genesis
        backend = gs.gpu if self.config.device == "cuda" else gs.cpu
        gs.init(backend=backend)
        
        # Create scene
        self.scene = gs.Scene(**self.config.scene_config)
        
        # Add ground plane
        self.scene.add_entity(
            gs.morphs.Plane()
        )
        
        # Add robot based on type
        if self.config.robot_type == GenesisRobotType.HUMANOID:
            self.robot = self._add_humanoid()
        elif self.config.robot_type == GenesisRobotType.UR5:
            self.robot = self._add_ur5()
        elif self.config.robot_type == GenesisRobotType.FRANKA:
            self.robot = self._add_franka()
        elif self.config.robot_type == GenesisRobotType.G1:
            self.robot = self._add_g1()
        elif self.config.robot_type == GenesisRobotType.GR1:
            self.robot = self._add_gr1()
        elif self.config.robot_type == GenesisRobotType.CUSTOM:
            self.robot = self._add_custom_robot()
        else:
            raise ValueError(f"Unsupported robot type: {self.config.robot_type}")
        
        # Add camera for rendering
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(1.5, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )
        
        # Build scene
        self.scene.build()
        self._initialized = True
        
    def _add_humanoid(self) -> RigidEntity:
        """Add humanoid robot to scene."""
        return self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/humanoid/humanoid.xml",
                pos=(0.0, 0.0, 1.0),
            ),
        )
    
    def _add_ur5(self) -> RigidEntity:
        """Add UR5 robot arm to scene."""
        return self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/robots/ur5/ur5.urdf",
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 0),
            ),
        )
    
    def _add_franka(self) -> RigidEntity:
        """Add Franka Emika Panda robot to scene."""
        return self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/robots/franka_emika_panda/panda.urdf",
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 0),
            ),
        )
    
    def _add_g1(self) -> RigidEntity:
        """Add Unitree G1 humanoid robot to scene."""
        # Try to load from common paths
        possible_paths = [
            "urdf/robots/g1/g1.urdf",
            "urdf/g1/g1.urdf",
            "assets/g1/g1.urdf",
        ]
        for path in possible_paths:
            if Path(path).exists():
                return self.scene.add_entity(
                    gs.morphs.URDF(
                        file=path,
                        pos=(0.0, 0.0, 0.8),
                        euler=(0, 0, 0),
                    ),
                )
        # Fallback to MJCF humanoid with warning
        print("Warning: G1 URDF not found, falling back to humanoid")
        return self._add_humanoid()
    
    def _add_gr1(self) -> RigidEntity:
        """Add Fourier GR1 humanoid robot to scene."""
        # Try to load from common paths
        possible_paths = [
            "urdf/robots/gr1/gr1.urdf",
            "urdf/gr1/gr1.urdf",
            "assets/gr1/gr1.urdf",
        ]
        for path in possible_paths:
            if Path(path).exists():
                return self.scene.add_entity(
                    gs.morphs.URDF(
                        file=path,
                        pos=(0.0, 0.0, 0.8),
                        euler=(0, 0, 0),
                    ),
                )
        # Fallback to MJCF humanoid with warning
        print("Warning: GR1 URDF not found, falling back to humanoid")
        return self._add_humanoid()
    
    def _add_custom_robot(self) -> RigidEntity:
        """Add custom robot from URDF."""
        if self.config.robot_urdf_path is None:
            raise ValueError("robot_urdf_path must be provided for custom robot type")
        if not Path(self.config.robot_urdf_path).exists():
            raise FileNotFoundError(f"URDF file not found: {self.config.robot_urdf_path}")
        return self.scene.add_entity(
            gs.morphs.URDF(
                file=self.config.robot_urdf_path,
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 0),
            ),
        )
    
    def reset(self, seed: Optional[int] = None):
        """Reset the simulation."""
        if seed is not None and np is not None:
            np.random.seed(seed)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        self.scene.reset()
        self._episode_count += 1
        
    def step(self, action: Optional[Any] = None) -> Tuple[Any, dict]:
        """
        Step the simulation.
        
        Args:
            action: Robot control action
            
        Returns:
            observation: Current observation (RGB image)
            info: Additional information
        """
        if action is not None:
            self._apply_action(action)
            
        # Step simulation
        self.scene.step()
        
        # Get observation
        obs = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        return obs, info
    
    def _apply_action(self, action: Any):
        """Apply control action to robot."""
        # Implementation depends on specific robot
        # This is a placeholder for joint position control
        if self.robot is not None and hasattr(self.robot, 'control_dofs_position'):
            # Assuming action is joint positions
            try:
                self.robot.control_dofs_position(action)
            except Exception as e:
                # If control fails, just log it (for compatibility)
                pass
    
    def _get_observation(self) -> Any:
        """Get current observation (RGB image)."""
        if self.camera is not None:
            rgb, _, _, _ = self.camera.render(rgb=True, depth=False, seg=False, normal=False)
            return rgb
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _get_info(self) -> dict:
        """Get additional information."""
        info = {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "base_position": self.get_base_position(),
            "base_orientation": self.get_base_orientation(),
        }
        return info
    
    def get_joint_positions(self) -> Optional[Any]:
        """Get joint positions."""
        if self.robot is None:
            return None
        try:
            return self.robot.get_q()
        except:
            return None
    
    def get_joint_velocities(self) -> Optional[Any]:
        """Get joint velocities."""
        if self.robot is None:
            return None
        try:
            return self.robot.get_dq()
        except:
            return None
    
    def get_base_position(self) -> Optional[Any]:
        """Get base position."""
        if self.robot is None:
            return None
        try:
            links = self.robot.get_links()
            if links:
                return np.array(links[0].get_pos())
        except:
            pass
        return None
    
    def get_base_orientation(self) -> Optional[Any]:
        """Get base orientation (quaternion)."""
        if self.robot is None:
            return None
        try:
            links = self.robot.get_links()
            if links:
                return np.array(links[0].get_quat())
        except:
            pass
        return None
    
    def render(self, mode: str = "rgb_array") -> Any:
        """
        Render the current scene.
        
        Args:
            mode: Rendering mode ("rgb_array" or "human")
            
        Returns:
            Rendered image
        """
        if self.camera is not None:
            rgb, _, _, _ = self.camera.render(rgb=True)
            return rgb
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Close the simulator."""
        if self.scene is not None:
            self.scene = None
        try:
            gs.destroy()
        except:
            pass
        self._initialized = False
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get current robot state.
        
        Returns:
            Dictionary containing state information
        """
        if self.robot is None:
            return {}
        
        state = {
            "joint_position": self.get_joint_positions(),
            "joint_velocity": self.get_joint_velocities(),
        }
        
        # Try to get torque/force if available
        try:
            state["joint_torque"] = self.robot.get_force()
        except:
            pass
            
        # Add base pose
        base_pos = self.get_base_position()
        base_quat = self.get_base_orientation()
        if base_pos is not None:
            state["base_position"] = base_pos
        if base_quat is not None:
            state["base_orientation"] = base_quat
            
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """
        Set robot state.
        
        Args:
            state: Dictionary containing state information
        """
        if self.robot is None:
            return
        
        if "joint_position" in state and state["joint_position"] is not None:
            try:
                self.robot.set_q(state["joint_position"])
            except:
                pass
        if "joint_velocity" in state and state["joint_velocity"] is not None:
            try:
                self.robot.set_dq(state["joint_velocity"])
            except:
                pass


def create_genesis_simulator(
    robot_type: str = "humanoid",
    headless: bool = True,
    device: str = "cuda",
    **kwargs
) -> GenesisSimulator:
    """
    Factory function to create Genesis simulator.
    
    Args:
        robot_type: Type of robot ("humanoid", "ur5", "franka", "g1", "gr1", "custom")
        headless: Whether to run in headless mode
        device: Device to use ("cuda" or "cpu")
        **kwargs: Additional configuration options
        
    Returns:
        Configured Genesis simulator
    """
    config = GenesisSimulatorConfig(
        robot_type=GenesisRobotType(robot_type),
        headless=headless,
        device=device,
        **kwargs
    )
    simulator = GenesisSimulator(config)
    simulator.initialize()
    return simulator
