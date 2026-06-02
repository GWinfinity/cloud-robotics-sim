# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""
OpenLoong Walking Parameters

Walking control parameters for OpenLoong humanoid robot,
adapted from OpenEvolve's openloong_walking example.

These parameters can be evolved using evolutionary algorithms
to find optimal walking gaits in Genesis simulation.

References:
    - Original: openevolve/examples/openloong_walking/initial_program_genesis.py
    - OpenLoong: https://github.com/loongOpen/OpenLoong-Dyn-Control
"""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class WalkingParameters:
    """
    Walking control parameters for OpenLoong humanoid robot.
    
    These parameters control:
    - PD Controller Gains for joint control
    - MPC (Model Predictive Control) weights
    - Gait generation parameters
    - State estimation filters
    - Simulation settings
    
    The parameters can be evolved using OpenEvolve or other
    evolutionary algorithms to optimize walking performance.
    """
    
    # =========================================================================
    # PD Controller Gains (Float Controller)
    # =========================================================================
    
    # Base joints (pelvis, torso)
    kp_base: float = 40.0
    """Proportional gain for base joints"""
    
    kd_base: float = 4.0
    """Derivative gain for base joints"""
    
    # Leg joints (hip, thigh)
    kp_leg: float = 25.0
    """Proportional gain for leg joints"""
    
    kd_leg: float = 2.5
    """Derivative gain for leg joints"""
    
    # Knee joints
    kp_knee: float = 30.0
    """Proportional gain for knee joints"""
    
    kd_knee: float = 3.0
    """Derivative gain for knee joints"""
    
    # Ankle joints
    kp_ankle: float = 35.0
    """Proportional gain for ankle joints"""
    
    kd_ankle: float = 3.5
    """Derivative gain for ankle joints"""
    
    # =========================================================================
    # MPC (Model Predictive Control) Weights
    # =========================================================================
    
    # Orientation tracking weights
    mpc_weight_roll: float = 10.0
    """Weight for roll angle tracking"""
    
    mpc_weight_pitch: float = 10.0
    """Weight for pitch angle tracking"""
    
    mpc_weight_yaw: float = 5.0
    """Weight for yaw angle tracking"""
    
    # Position tracking weights
    mpc_weight_px: float = 100.0
    """Weight for X position tracking"""
    
    mpc_weight_py: float = 200.0
    """Weight for Y position tracking"""
    
    mpc_weight_pz: float = 50.0
    """Weight for Z position (height) tracking"""
    
    # Angular velocity tracking weights
    mpc_weight_wx: float = 1.0
    """Weight for roll rate tracking"""
    
    mpc_weight_wy: float = 1.0
    """Weight for pitch rate tracking"""
    
    mpc_weight_wz: float = 1.0
    """Weight for yaw rate tracking"""
    
    # Linear velocity tracking weights
    mpc_weight_vx: float = 10.0
    """Weight for X velocity tracking"""
    
    mpc_weight_vy: float = 10.0
    """Weight for Y velocity tracking"""
    
    mpc_weight_vz: float = 10.0
    """Weight for Z velocity tracking"""
    
    # Force and torque regularization weights
    mpc_weight_force: float = 1.0
    """Weight for contact force regularization"""
    
    mpc_weight_torque: float = 1.0
    """Weight for joint torque regularization"""
    
    mpc_weight_extra: float = 0.01
    """Extra regularization weight"""
    
    # =========================================================================
    # Gait Parameters
    # =========================================================================
    
    gait_period: float = 0.8
    """Duration of one complete walking cycle (seconds)"""
    
    swing_height: float = 0.08
    """Maximum foot lift height during swing phase (meters)"""
    
    stance_ratio: float = 0.5
    """Ratio of gait period spent in stance phase (0-1)"""
    
    # =========================================================================
    # State Estimation
    # =========================================================================
    
    filter_alpha: float = 0.3
    """Complementary filter coefficient (0-1, higher = more smoothing)"""
    
    # =========================================================================
    # Walking Task Parameters
    # =========================================================================
    
    desired_velocity: float = 0.2
    """Target forward walking speed (m/s)"""
    
    torque_limit: float = 60.0
    """Maximum joint torque limit (Nm)"""
    
    # =========================================================================
    # Simulation Settings
    # =========================================================================
    
    sim_duration: float = 10.0
    """Simulation duration for evaluation (seconds)"""
    
    enable_visualization: bool = False
    """Whether to enable Genesis viewer during simulation"""
    
    # =========================================================================
    # Robot Model Settings
    # =========================================================================
    
    robot_urdf_path: Optional[str] = None
    """Path to OpenLoong robot URDF file"""
    
    initial_height: float = 1.0
    """Initial robot height above ground (meters)"""
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate PD gains are positive
        for name in ['kp_base', 'kd_base', 'kp_leg', 'kd_leg', 
                     'kp_knee', 'kd_knee', 'kp_ankle', 'kd_ankle']:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        
        # Validate MPC weights are non-negative
        for name in ['mpc_weight_roll', 'mpc_weight_pitch', 'mpc_weight_yaw',
                     'mpc_weight_px', 'mpc_weight_py', 'mpc_weight_pz']:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        
        # Validate gait parameters
        if self.gait_period <= 0:
            raise ValueError(f"gait_period must be positive, got {self.gait_period}")
        if not 0 < self.stance_ratio < 1:
            raise ValueError(f"stance_ratio must be in (0, 1), got {self.stance_ratio}")
        
        # Set default URDF path if not provided
        if self.robot_urdf_path is None:
            # Try common paths
            possible_paths = [
                "urdf/robots/openloong/AzureLoong.urdf",
                "urdf/openloong/AzureLoong.urdf",
                "assets/openloong/AzureLoong.urdf",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    self.robot_urdf_path = path
                    break
    
    def get_pd_gains(self) -> Dict[str, Tuple[float, float]]:
        """Get PD gains as a dictionary.
        
        Returns:
            Dictionary mapping joint groups to (kp, kd) tuples
        """
        return {
            'base': (self.kp_base, self.kd_base),
            'leg': (self.kp_leg, self.kd_leg),
            'knee': (self.kp_knee, self.kd_knee),
            'ankle': (self.kp_ankle, self.kd_ankle),
        }
    
    def get_mpc_L_diag(self):
        """Get MPC state weighting matrix diagonal.
        
        Returns:
            12-element array for state weights [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz]
        """
        if not HAS_NUMPY:
            return [
                self.mpc_weight_roll, self.mpc_weight_pitch, self.mpc_weight_yaw,
                self.mpc_weight_px, self.mpc_weight_py, self.mpc_weight_pz,
                self.mpc_weight_wx, self.mpc_weight_wy, self.mpc_weight_wz,
                self.mpc_weight_vx, self.mpc_weight_vy, self.mpc_weight_vz
            ]
        return np.array([
            self.mpc_weight_roll, self.mpc_weight_pitch, self.mpc_weight_yaw,
            self.mpc_weight_px, self.mpc_weight_py, self.mpc_weight_pz,
            self.mpc_weight_wx, self.mpc_weight_wy, self.mpc_weight_wz,
            self.mpc_weight_vx, self.mpc_weight_vy, self.mpc_weight_vz
        ])
    
    def get_mpc_K_diag(self):
        """Get MPC control weighting matrix diagonal.
        
        Returns:
            13-element array for control weights
        """
        if not HAS_NUMPY:
            return [
                self.mpc_weight_force, self.mpc_weight_force, self.mpc_weight_force,
                self.mpc_weight_torque, self.mpc_weight_torque, self.mpc_weight_torque,
                self.mpc_weight_force, self.mpc_weight_force, self.mpc_weight_force,
                self.mpc_weight_torque, self.mpc_weight_torque, self.mpc_weight_torque,
                self.mpc_weight_extra
            ]
        return np.array([
            self.mpc_weight_force, self.mpc_weight_force, self.mpc_weight_force,
            self.mpc_weight_torque, self.mpc_weight_torque, self.mpc_weight_torque,
            self.mpc_weight_force, self.mpc_weight_force, self.mpc_weight_force,
            self.mpc_weight_torque, self.mpc_weight_torque, self.mpc_weight_torque,
            self.mpc_weight_extra
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary.
        
        Returns:
            Dictionary of all parameters
        """
        return {
            # PD gains
            'kp_base': self.kp_base,
            'kd_base': self.kd_base,
            'kp_leg': self.kp_leg,
            'kd_leg': self.kd_leg,
            'kp_knee': self.kp_knee,
            'kd_knee': self.kd_knee,
            'kp_ankle': self.kp_ankle,
            'kd_ankle': self.kd_ankle,
            # MPC weights
            'mpc_weight_roll': self.mpc_weight_roll,
            'mpc_weight_pitch': self.mpc_weight_pitch,
            'mpc_weight_yaw': self.mpc_weight_yaw,
            'mpc_weight_px': self.mpc_weight_px,
            'mpc_weight_py': self.mpc_weight_py,
            'mpc_weight_pz': self.mpc_weight_pz,
            'mpc_weight_wx': self.mpc_weight_wx,
            'mpc_weight_wy': self.mpc_weight_wy,
            'mpc_weight_wz': self.mpc_weight_wz,
            'mpc_weight_vx': self.mpc_weight_vx,
            'mpc_weight_vy': self.mpc_weight_vy,
            'mpc_weight_vz': self.mpc_weight_vz,
            'mpc_weight_force': self.mpc_weight_force,
            'mpc_weight_torque': self.mpc_weight_torque,
            'mpc_weight_extra': self.mpc_weight_extra,
            # Gait parameters
            'gait_period': self.gait_period,
            'swing_height': self.swing_height,
            'stance_ratio': self.stance_ratio,
            # State estimation
            'filter_alpha': self.filter_alpha,
            # Task parameters
            'desired_velocity': self.desired_velocity,
            'torque_limit': self.torque_limit,
            # Simulation
            'sim_duration': self.sim_duration,
            'enable_visualization': self.enable_visualization,
            # Robot
            'robot_urdf_path': self.robot_urdf_path,
            'initial_height': self.initial_height,
        }
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> "WalkingParameters":
        """Create WalkingParameters from dictionary.
        
        Args:
            params_dict: Dictionary of parameters
            
        Returns:
            WalkingParameters instance
        """
        return cls(**params_dict)
    
    def copy(self) -> "WalkingParameters":
        """Create a copy of these parameters.
        
        Returns:
            New WalkingParameters instance with same values
        """
        return self.from_dict(self.to_dict())
    
    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1) -> "WalkingParameters":
        """Create a mutated copy of parameters.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_scale: Relative scale of mutations
            
        Returns:
            New WalkingParameters with mutations applied
        """
        if not HAS_NUMPY:
            import random
            mutated = self.copy()
            for key in ['kp_base', 'kd_base', 'kp_leg', 'kd_leg', 
                       'kp_knee', 'kd_knee', 'kp_ankle', 'kd_ankle',
                       'gait_period', 'swing_height', 'desired_velocity']:
                if random.random() < mutation_rate:
                    current = getattr(mutated, key)
                    delta = current * mutation_scale * (random.random() - 0.5) * 2
                    setattr(mutated, key, current + delta)
            return mutated
        
        mutated = self.copy()
        param_names = [
            'kp_base', 'kd_base', 'kp_leg', 'kd_leg',
            'kp_knee', 'kd_knee', 'kp_ankle', 'kd_ankle',
            'mpc_weight_roll', 'mpc_weight_pitch', 'mpc_weight_pz',
            'gait_period', 'swing_height', 'desired_velocity'
        ]
        
        for name in param_names:
            if np.random.random() < mutation_rate:
                current = getattr(mutated, name)
                delta = current * mutation_scale * np.random.randn()
                setattr(mutated, name, max(0, current + delta))
        
        return mutated


# Preset parameter sets
WALKING_PRESETS = {
    'default': WalkingParameters(),
    
    'stable_walk': WalkingParameters(
        kp_base=50.0, kd_base=5.0,
        kp_leg=30.0, kd_leg=3.0,
        kp_knee=35.0, kd_knee=3.5,
        kp_ankle=40.0, kd_ankle=4.0,
        mpc_weight_pitch=15.0,
        mpc_weight_pz=80.0,
        gait_period=0.9,
        swing_height=0.06,
    ),
    
    'fast_walk': WalkingParameters(
        kp_base=35.0, kd_base=3.5,
        kp_leg=20.0, kd_leg=2.0,
        gait_period=0.6,
        swing_height=0.10,
        desired_velocity=0.4,
    ),
    
    'cautious_walk': WalkingParameters(
        kp_base=60.0, kd_base=6.0,
        kp_leg=35.0, kd_leg=3.5,
        gait_period=1.0,
        swing_height=0.05,
        desired_velocity=0.1,
    ),
}


def get_preset(name: str) -> WalkingParameters:
    """Get a preset parameter set.
    
    Args:
        name: Preset name ('default', 'stable_walk', 'fast_walk', 'cautious_walk')
        
    Returns:
        WalkingParameters preset
    """
    if name not in WALKING_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(WALKING_PRESETS.keys())}")
    return WALKING_PRESETS[name].copy()


__all__ = [
    'WalkingParameters',
    'WALKING_PRESETS',
    'get_preset',
]
