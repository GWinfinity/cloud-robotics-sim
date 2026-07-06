# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
OpenLoong Walking Environment

Gymnasium-compatible environment for OpenLoong humanoid walking,
adapted from OpenEvolve's openloong_walking example.

This environment provides:
- Standard Gymnasium API (reset, step, render)
- Support for both simple and Genesis simulation modes
- Reward function based on walking stability
- Observation space for robot state

References:
    - Original: openevolve/examples/openloong_walking/initial_program_genesis.py
    - Gymnasium: https://gymnasium.farama.org/
"""

from typing import Any, Union
from pathlib import Path

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Gymnasium imports (optional)
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None
    spaces = None

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
    from .evaluator import WalkingEvaluator, EvaluationResult
except ImportError:
    from walking_params import WalkingParameters
    from evaluator import WalkingEvaluator, EvaluationResult


# Fallback for gym spaces if not available
if not HAS_GYM:
    class DummySpace:
        def __init__(self, shape, low=0, high=1):
            self.shape = shape
            self.low = low
            self.high = high
    
    class DummySpaces:
        Box = DummySpace
        Discrete = type('Discrete', (), {'__init__': lambda self, n: setattr(self, 'n', n)})
    
    spaces = DummySpaces()


class OpenLoongWalkingEnv:
    """
    OpenLoong humanoid robot walking environment.
    
    This environment implements a Gymnasium-like interface for training
    walking policies on the OpenLoong robot. Supports both simplified
    evaluation (fast) and Genesis simulation (accurate).
    
    Example:
        ```python
        env = OpenLoongWalkingEnv(use_genesis=True)
        obs, info = env.reset()
        
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        ```
    """
    
    def __init__(
        self,
        params: Optional[WalkingParameters] = None,
        use_genesis: bool = False,
        device: str = "cuda",
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            params: Walking parameters. Uses defaults if None.
            use_genesis: Whether to use Genesis simulation
            device: Device for Genesis ("cuda" or "cpu")
            max_episode_steps: Maximum steps per episode
            render_mode: Rendering mode ("human" or None)
        """
        self.params = params or WalkingParameters()
        self.use_genesis = use_genesis and HAS_GENESIS
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Initialize evaluator
        self.evaluator = WalkingEvaluator(
            use_genesis=self.use_genesis,
            device=device,
            verbose=False,
        )
        
        # Define spaces
        self._setup_spaces()
        
        # Episode state
        self.current_step = 0
        self.current_obs = None
        self.last_result = None
        
        # Genesis scene (if used)
        self.scene = None
        self.robot = None
        
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: modifications to walking parameters
        # [delta_kp_base, delta_kd_base, delta_kp_leg, ..., delta_gait_period, ...]
        num_actions = 12  # Key parameters that can be adjusted
        
        if HAS_GYM:
            self.action_space = spaces.Box(
                low=-0.1, high=0.1, shape=(num_actions,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-0.1, high=0.1, shape=(num_actions,)
            )
        
        # Observation space: robot state
        # [height, roll, pitch, roll_rate, pitch_rate, vx, vy, vz, ...]
        obs_dim = 12
        
        if HAS_GYM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(obs_dim,)
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (e.g., initial parameters)
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            if HAS_NUMPY:
                np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Use provided parameters or defaults
        if options and 'params' in options:
            self.params = options['params']
        else:
            self.params = WalkingParameters()
        
        # Initialize Genesis scene if needed
        if self.use_genesis and self.scene is None:
            self._init_genesis_scene()
        
        # Get initial observation
        obs = self._get_observation()
        info = {'params': self.params.to_dict()}
        
        self.current_obs = obs
        return obs, info
    
    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take (parameter modifications)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Apply action (modify parameters)
        self._apply_action(action)
        
        # Evaluate current parameters
        result = self.evaluator.evaluate(self.params)
        self.last_result = result
        
        # Get observation
        obs = self._get_observation()
        self.current_obs = obs
        
        # Compute reward
        reward = self._compute_reward(result)
        
        # Check termination
        terminated = result.stability_score < 0.2 or result.final_height < 0.5
        truncated = self.current_step >= self.max_episode_steps
        
        # Info
        info = {
            'stability_score': result.stability_score,
            'final_height': result.final_height,
            'fall_time': result.fall_time,
            'params': self.params.to_dict(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply action to modify parameters.
        
        Args:
            action: Parameter modifications
        """
        if not HAS_NUMPY:
            # Fallback without numpy
            action = list(action) if hasattr(action, '__iter__') else [0] * 12
        
        # Map action to parameter modifications
        # Action indices:
        # 0-1: base PD gains
        # 2-3: leg PD gains
        # 4-5: knee PD gains
        # 6-7: ankle PD gains
        # 8: gait_period
        # 9: swing_height
        # 10: desired_velocity
        # 11: mpc_weight_pitch
        
        self.params.kp_base *= (1 + action[0])
        self.params.kd_base *= (1 + action[1])
        self.params.kp_leg *= (1 + action[2])
        self.params.kd_leg *= (1 + action[3])
        self.params.kp_knee *= (1 + action[4])
        self.params.kd_knee *= (1 + action[5])
        self.params.kp_ankle *= (1 + action[6])
        self.params.kd_ankle *= (1 + action[7])
        self.params.gait_period *= (1 + action[8])
        self.params.swing_height *= (1 + action[9])
        self.params.desired_velocity *= (1 + action[10])
        self.params.mpc_weight_pitch *= (1 + action[11])
        
        # Clamp to valid ranges
        self.params.kp_base = max(10, min(100, self.params.kp_base))
        self.params.kd_base = max(1, min(10, self.params.kd_base))
        self.params.gait_period = max(0.4, min(1.5, self.params.gait_period))
        self.params.swing_height = max(0.02, min(0.2, self.params.swing_height))
        self.params.desired_velocity = max(0.05, min(0.8, self.params.desired_velocity))
    
    def _get_observation(self):
        """Get current observation.
        
        Returns:
            Observation array
        """
        if self.last_result:
            # Use evaluation result as observation
            obs = [
                self.last_result.final_height,
                self.last_result.max_roll,
                self.last_result.max_pitch,
                self.last_result.fall_time / self.params.sim_duration,
                self.params.kp_base / 50.0,
                self.params.kd_base / 5.0,
                self.params.gait_period,
                self.params.swing_height / 0.1,
                self.params.desired_velocity / 0.3,
                self.params.mpc_weight_pitch / 10.0,
                self.params.mpc_weight_pz / 50.0,
                self.current_step / self.max_episode_steps,
            ]
        else:
            # Initial observation
            obs = [1.0, 0.0, 0.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.67, 1.0, 1.0, 0.0]
        
        if HAS_NUMPY:
            return np.array(obs, dtype=np.float32)
        return obs
    
    def _compute_reward(self, result: EvaluationResult) -> float:
        """Compute reward from evaluation result.
        
        Args:
            result: Evaluation result
            
        Returns:
            Reward value
        """
        # Reward components
        stability_reward = result.stability_score
        height_reward = min(1.0, result.final_height)
        time_reward = result.fall_time / self.params.sim_duration
        
        # Combined reward
        reward = 0.5 * stability_reward + 0.3 * height_reward + 0.2 * time_reward
        
        return reward
    
    def _init_genesis_scene(self):
        """Initialize Genesis scene."""
        if not HAS_GENESIS:
            return
        
        backend = gs.gpu if self.device == "cuda" else gs.cpu
        gs.init(backend=backend)
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            sim_options=gs.options.SimOptions(dt=0.001),
            show_viewer=(self.render_mode == "human"),
        )
        
        # Load robot
        model_path = self.params.robot_urdf_path or "xml/humanoid/humanoid.xml"
        
        try:
            if model_path.endswith('.xml'):
                self.robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file=model_path,
                        pos=(0.0, 0.0, self.params.initial_height),
                    ),
                )
            else:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=model_path,
                        pos=(0.0, 0.0, self.params.initial_height),
                        fixed=False,
                    ),
                )
            
            self.scene.build()
        except Exception as e:
            print(f"Warning: Failed to initialize Genesis scene: {e}")
            self.scene = None
            self.robot = None
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.scene is not None:
            # Rendering is handled automatically by Genesis viewer
            pass
    
    def close(self):
        """Close the environment."""
        if self.scene is not None:
            try:
                gs.destroy()
            except Exception:
                pass
            self.scene = None
            self.robot = None


# Gymnasium compatibility wrapper
if HAS_GYM:
    class OpenLoongWalkingGymEnv(gym.Env):
        """
        Gymnasium wrapper for OpenLoongWalkingEnv.
        
        This class wraps the base environment to provide full
        Gymnasium compatibility.
        """
        
        metadata = {'render_modes': ['human']}
        
        def __init__(self, **kwargs):
            super().__init__()
            self.env = OpenLoongWalkingEnv(**kwargs)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
        
        def reset(self, seed=None, options=None):
            return self.env.reset(seed=seed, options=options)
        
        def step(self, action):
            return self.env.step(action)
        
        def render(self):
            return self.env.render()
        
        def close(self):
            return self.env.close()
else:
    OpenLoongWalkingGymEnv = OpenLoongWalkingEnv


def make_env(
    use_genesis: bool = False,
    device: str = "cuda",
    **kwargs
) -> Union[OpenLoongWalkingEnv, Any]:
    """Create an OpenLoong walking environment.
    
    Args:
        use_genesis: Whether to use Genesis simulation
        device: Device for Genesis ("cuda" or "cpu")
        **kwargs: Additional arguments for the environment
        
    Returns:
        Environment instance
    """
    if HAS_GYM:
        return OpenLoongWalkingGymEnv(
            use_genesis=use_genesis,
            device=device,
            **kwargs
        )
    else:
        return OpenLoongWalkingEnv(
            use_genesis=use_genesis,
            device=device,
            **kwargs
        )


__all__ = [
    'OpenLoongWalkingEnv',
    'OpenLoongWalkingGymEnv',
    'make_env',
]
