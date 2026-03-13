"""
Flatten observation and action wrappers.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlattenObservation(gym.ObservationWrapper):
    """
    Flatten observation to 1D vector.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Update observation space
        obs_shape = env.observation_space.shape
        flat_dim = np.prod(obs_shape)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        """Flatten observation."""
        return obs.flatten()


class FlattenAction(gym.ActionWrapper):
    """
    Flatten action to 1D vector.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Update action space
        action_shape = env.action_space.shape
        flat_dim = np.prod(action_shape)
        self.action_space = spaces.Box(
            low=env.action_space.low.flatten()[0],
            high=env.action_space.high.flatten()[0],
            shape=(flat_dim,),
            dtype=env.action_space.dtype
        )
    
    def action(self, act):
        """Flatten action."""
        return act.flatten()
