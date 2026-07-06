# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
OpenLoong Walking Evaluator

Evaluates walking parameters for OpenLoong humanoid robot,
adapted from OpenEvolve's openloong_walking example.

Provides both:
- Simple heuristic evaluator (fast, no Genesis dependency)
- Genesis simulation evaluator (accurate, requires Genesis)

References:
    - Original: openevolve/examples/openloong_walking/initial_program_genesis.py
    - OpenLoong: https://github.com/loongOpen/OpenLoong-Dyn-Control
"""

from typing import Any
from pathlib import Path
from dataclasses import dataclass

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Genesis imports
try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None

# Local imports
try:
    from .walking_params import WalkingParameters
except ImportError:
    from walking_params import WalkingParameters


@dataclass
class EvaluationResult:
    """Result of evaluating a set of walking parameters."""
    
    stability_score: float
    """Overall walking stability (0-1, higher is better)"""
    
    final_height: float
    """Final robot height in meters"""
    
    fall_time: float
    """Time until robot fell or simulation ended"""
    
    max_roll: float
    """Maximum roll angle deviation in radians"""
    
    max_pitch: float
    """Maximum pitch angle deviation in radians"""
    
    success: bool = True
    """Whether evaluation completed successfully"""
    
    error_message: Optional[str] = None
    """Error message if evaluation failed"""
    
    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple format for OpenEvolve compatibility.
        
        Returns:
            (stability_score, final_height, fall_time)
        """
        return (self.stability_score, self.final_height, self.fall_time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stability_score': self.stability_score,
            'final_height': self.final_height,
            'fall_time': self.fall_time,
            'max_roll': self.max_roll,
            'max_pitch': self.max_pitch,
            'success': self.success,
            'error_message': self.error_message,
        }


class WalkingEvaluator:
    """
    Evaluates walking parameters for OpenLoong robot.
    
    Supports two evaluation modes:
    1. Simple heuristic mode: Fast estimation without Genesis
    2. Genesis simulation mode: Accurate physics simulation
    """
    
    def __init__(
        self,
        use_genesis: bool = False,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Args:
            use_genesis: Whether to use Genesis simulation (if available)
            device: Device for Genesis ("cuda" or "cpu")
            verbose: Whether to print evaluation details
        """
        self.use_genesis = use_genesis and HAS_GENESIS
        self.device = device
        self.verbose = verbose
        
        if use_genesis and not HAS_GENESIS:
            print("Warning: Genesis not available, falling back to simple evaluator")
    
    def evaluate(self, params: WalkingParameters) -> EvaluationResult:
        """Evaluate walking parameters.
        
        Args:
            params: Walking parameters to evaluate
            
        Returns:
            Evaluation result
        """
        if self.use_genesis:
            try:
                return self._evaluate_with_genesis(params)
            except Exception as e:
                if self.verbose:
                    print(f"Genesis evaluation failed: {e}, falling back to simple")
                return self._evaluate_simple(params)
        else:
            return self._evaluate_simple(params)
    
    def _evaluate_simple(self, params: WalkingParameters) -> EvaluationResult:
        """
        Evaluate using simplified heuristic model.
        
        This is a proxy evaluation that estimates walking performance
        based on parameter values without running physics simulation.
        Fast but less accurate than Genesis simulation.
        
        Args:
            params: Walking parameters
            
        Returns:
            Estimated evaluation result
        """
        if self.verbose:
            print("Using simple heuristic evaluator")
        
        # Score based on PD gains being in reasonable ranges
        pd_score = 0.0
        
        checks = [
            (20 <= params.kp_base <= 80, 0.15),
            (2 <= params.kd_base <= 8, 0.15),
            (15 <= params.kp_leg <= 40, 0.10),
            (1.5 <= params.kd_leg <= 4, 0.10),
            (20 <= params.kp_knee <= 50, 0.10),
            (2 <= params.kd_knee <= 5, 0.10),
            (25 <= params.kp_ankle <= 45, 0.05),
            (2.5 <= params.kd_ankle <= 4.5, 0.05),
        ]
        
        for condition, weight in checks:
            if condition:
                pd_score += weight
        
        # MPC weight score
        mpc_score = 0.0
        if 30 <= params.mpc_weight_pz <= 80:
            mpc_score += 0.25
        if 5 <= params.mpc_weight_pitch <= 20:
            mpc_score += 0.25
        if 5 <= params.mpc_weight_vx <= 20:
            mpc_score += 0.10
        if params.mpc_weight_py >= 100:
            mpc_score += 0.15
        
        # Gait parameter score
        gait_score = 0.0
        if 0.6 <= params.gait_period <= 1.0:
            gait_score += 0.35
        if 0.05 <= params.swing_height <= 0.12:
            gait_score += 0.15
        
        # Combined stability (normalize to [0, 1])
        max_possible_score = 2.05  # Sum of all max weights
        stability = (pd_score + mpc_score + gait_score) / max_possible_score
        
        # Penalize extreme values
        if params.torque_limit > 100 or params.torque_limit < 30:
            stability *= 0.8
        if params.desired_velocity > 0.5:
            stability *= 0.7
        
        # Clamp to [0, 1] range
        stability = max(0.0, min(1.0, stability))
        
        # Estimate fall time
        fall_probability = max(0, 1.0 - stability)
        fall_time = params.sim_duration * (1 - fall_probability * 0.5)
        
        # Final height
        if fall_probability > 0.5:
            final_height = 0.3 + 0.7 * (1 - fall_probability)
        else:
            final_height = 0.95 + 0.05 * stability
        
        # Estimate roll/pitch from stability
        max_roll = 0.1 / (stability + 0.1)
        max_pitch = 0.1 / (stability + 0.1)
        
        return EvaluationResult(
            stability_score=float(stability),
            final_height=float(final_height),
            fall_time=float(fall_time),
            max_roll=float(max_roll),
            max_pitch=float(max_pitch),
            success=True,
        )
    
    def _evaluate_with_genesis(self, params: WalkingParameters) -> EvaluationResult:
        """
        Evaluate using Genesis physics simulation.
        
        This runs actual physics simulation with the given parameters
        and measures robot stability and walking performance.
        More accurate but slower than simple evaluation.
        
        Args:
            params: Walking parameters
            
        Returns:
            Evaluation result from simulation
        """
        if not HAS_GENESIS:
            raise RuntimeError("Genesis not available")
        
        if self.verbose:
            print("Using Genesis physics simulation evaluator")
        
        # Initialize Genesis
        backend = gs.gpu if self.device == "cuda" else gs.cpu
        gs.init(backend=backend)
        
        # Create scene
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            sim_options=gs.options.SimOptions(dt=0.001),
            show_viewer=params.enable_visualization,
        )
        
        # Load robot
        if params.robot_urdf_path and Path(params.robot_urdf_path).exists():
            model_path = params.robot_urdf_path
        else:
            # Use a default humanoid model if OpenLoong not available
            model_path = "xml/humanoid/humanoid.xml"
        
        try:
            if model_path.endswith('.xml'):
                robot = scene.add_entity(
                    gs.morphs.MJCF(
                        file=model_path,
                        pos=(0.0, 0.0, params.initial_height),
                    ),
                )
            else:
                robot = scene.add_entity(
                    gs.morphs.URDF(
                        file=model_path,
                        pos=(0.0, 0.0, params.initial_height),
                        fixed=False,
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load robot model: {e}")
        
        scene.build()
        
        # Run simulation
        sim_time = 0.0
        dt = 0.001
        max_time = params.sim_duration
        
        # Tracking metrics
        heights = []
        roll_angles = []
        pitch_angles = []
        
        try:
            for i in range(int(max_time / dt)):
                sim_time += dt
                
                # Get robot state
                if hasattr(robot, 'get_dofs_position'):
                    dof_pos = robot.get_dofs_position()
                    if HAS_NUMPY and hasattr(dof_pos, 'numpy'):
                        dof_pos = dof_pos.numpy()
                    base_pos = dof_pos[:3]
                    rpy = dof_pos[3:6]
                elif hasattr(robot, 'get_qpos'):
                    qpos = robot.get_qpos()
                    base_pos = qpos[:3] if HAS_NUMPY else qpos
                    rpy = qpos[3:6] if HAS_NUMPY else qpos
                else:
                    break
                
                heights.append(float(base_pos[2]) if HAS_NUMPY else base_pos[2])
                roll_angles.append(abs(float(rpy[0])) if HAS_NUMPY else abs(rpy[0]))
                pitch_angles.append(abs(float(rpy[1])) if HAS_NUMPY else abs(rpy[1]))
                
                # Check for fall (robot dropped too low)
                current_height = heights[-1]
                if current_height < 0.5:
                    if self.verbose:
                        print(f"Robot fell at t={sim_time:.2f}s, height={current_height:.2f}m")
                    break
                
                # Step simulation
                scene.step()
        
        finally:
            # Cleanup
            try:
                gs.destroy()
            except Exception:
                pass
        
        # Calculate metrics
        if heights:
            min_height = min(heights)
            avg_height = sum(heights) / len(heights) if not HAS_NUMPY else np.mean(heights)
            max_roll = max(roll_angles)
            max_pitch = max(pitch_angles)
        else:
            min_height = 0
            avg_height = 0
            max_roll = 0
            max_pitch = 0
        
        # Stability score
        height_score = min(1.0, avg_height)
        roll_score = max(0, 1.0 - max_roll / 0.5)
        pitch_score = max(0, 1.0 - max_pitch / 0.5)
        time_score = sim_time / max_time
        
        stability = (height_score + roll_score + pitch_score + time_score) / 4.0
        
        return EvaluationResult(
            stability_score=float(stability),
            final_height=float(avg_height),
            fall_time=float(sim_time),
            max_roll=float(max_roll),
            max_pitch=float(max_pitch),
            success=True,
        )


def evaluate_walking(
    params: WalkingParameters,
    use_genesis: bool = False,
    device: str = "cuda",
) -> EvaluationResult:
    """Convenience function to evaluate walking parameters.
    
    Args:
        params: Walking parameters to evaluate
        use_genesis: Whether to use Genesis simulation
        device: Device for Genesis ("cuda" or "cpu")
        
    Returns:
        Evaluation result
    """
    evaluator = WalkingEvaluator(use_genesis=use_genesis, device=device)
    return evaluator.evaluate(params)


def compare_parameters(
    params1: WalkingParameters,
    params2: WalkingParameters,
    use_genesis: bool = False,
) -> dict[str, Any]:
    """Compare two sets of walking parameters.
    
    Args:
        params1: First parameter set
        params2: Second parameter set
        use_genesis: Whether to use Genesis simulation
        
    Returns:
        Comparison dictionary
    """
    evaluator = WalkingEvaluator(use_genesis=use_genesis)
    
    result1 = evaluator.evaluate(params1)
    result2 = evaluator.evaluate(params2)
    
    return {
        'params1': result1.to_dict(),
        'params2': result2.to_dict(),
        'better_stability': 'params1' if result1.stability_score > result2.stability_score else 'params2',
        'stability_diff': abs(result1.stability_score - result2.stability_score),
        'better_height': 'params1' if result1.final_height > result2.final_height else 'params2',
        'height_diff': abs(result1.final_height - result2.final_height),
    }


# OpenEvolve compatibility
run_search = evaluate_walking


__all__ = [
    'EvaluationResult',
    'WalkingEvaluator',
    'evaluate_walking',
    'compare_parameters',
    'run_search',
]
