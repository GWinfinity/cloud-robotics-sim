"""
Base environment class for Genesis ManiSkill.
Inspired by ManiSkill's BaseEnv but adapted for Genesis backend.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env):
    """
    Base environment for Genesis ManiSkill.
    
    Args:
        num_envs: Number of parallel environments. If > 1, uses GPU parallel simulation.
        scene_type: Type of scene ('kitchen', 'tabletop', etc.)
        robot_uid: Robot agent ID (e.g., 'franka', 'g1', 'gr1')
        task_type: Task type (e.g., 'pick_place', 'open_drawer')
        obs_mode: Observation mode ('state', 'rgb', 'depth', 'rgbd', 'pointcloud')
        control_mode: Control mode ('pd_joint_pos', 'pd_joint_vel', 'pd_ee_pos')
        render_mode: Render mode ('human', 'rgb_array')
        sim_freq: Simulation frequency (Hz)
        control_freq: Control frequency (Hz)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "sim_backend": "genesis",
    }
    
    def __init__(
        self,
        num_envs: int = 1,
        scene_type: str = "kitchen",
        robot_uid: str = "franka",
        task_type: str = "pick_place",
        obs_mode: str = "state",
        control_mode: str = "pd_joint_pos",
        render_mode: Optional[str] = None,
        sim_freq: int = 100,
        control_freq: int = 20,
        scene_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        
        self.num_envs = num_envs
        self.scene_type = scene_type
        self.robot_uid = robot_uid
        self.task_type = task_type
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.sim_freq = sim_freq
        self.control_freq = control_freq
        self.scene_config = scene_config or {}
        
        # Calculate control decimation
        self.control_dt = 1.0 / control_freq
        self.sim_dt = 1.0 / sim_freq
        self.control_decimation = int(sim_freq / control_freq)
        
        # Genesis backend
        self.scene = None
        self.robot = None
        self.task = None
        self.cameras = {}
        
        # Initialize Genesis
        self._init_genesis()
        
        # Build scene
        self._build_scene()
        self._build_agent()
        self._build_task()
        self._build_sensors()
        
        # Setup action and observation spaces
        self._setup_spaces()
        
        # Episode management
        self.episode_steps = torch.zeros(num_envs, dtype=torch.int32)
        self.max_episode_steps = 200
        
    def _init_genesis(self):
        """Initialize Genesis physics engine."""
        if not gs._initialized:
            gs.init(backend=gs.gpu if self.num_envs > 1 else gs.cpu)
    
    def _build_scene(self):
        """Build the simulation scene. Override in subclasses."""
        raise NotImplementedError
    
    def _build_agent(self):
        """Build the robot agent."""
        from genesis_maniskill.agents import get_agent
        self.robot = get_agent(
            self.robot_uid,
            scene=self.scene,
            num_envs=self.num_envs
        )
    
    def _build_task(self):
        """Build the task."""
        from genesis_maniskill.tasks import get_task
        self.task = get_task(
            self.task_type,
            env=self,
            scene=self.scene,
            robot=self.robot
        )
    
    def _build_sensors(self):
        """Build sensors (cameras, etc.)."""
        if "rgb" in self.obs_mode or "depth" in self.obs_mode:
            self._setup_cameras()
    
    def _setup_cameras(self):
        """Setup default cameras."""
        # Base camera setup - override in subclasses
        pass
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space from robot
        self.action_space = self.robot.action_space
        
        # Observation space based on obs_mode
        self.observation_space = self._get_obs_space()
    
    def _get_obs_space(self) -> spaces.Space:
        """Get observation space based on obs_mode."""
        if self.obs_mode == "state":
            obs_dim = self._get_state_obs_dim()
            return spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(obs_dim,),
                dtype=np.float32
            )
        elif self.obs_mode == "rgb":
            # Assuming 128x128 RGB image
            return spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 3),
                dtype=np.uint8
            )
        elif self.obs_mode == "rgbd":
            return spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 4),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _get_state_obs_dim(self) -> int:
        """Get state observation dimension. Override in subclasses."""
        # Robot state + task-specific state
        return self.robot.state_dim + self.task.state_dim
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reconfigure scene if needed
        if options and options.get("reconfigure", False):
            self._reconfigure()
        
        # Reset scene and task
        self._reset_scene()
        self.task.reset()
        
        # Reset episode counters
        self.episode_steps[:] = 0
        
        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _reset_scene(self):
        """Reset the scene."""
        if self.scene is not None:
            self.scene.reset()
    
    def _reconfigure(self):
        """Reconfigure the environment."""
        # Rebuild scene, agent, task
        self._build_scene()
        self._build_agent()
        self._build_task()
    
    def step(
        self,
        action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, Dict], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        for _ in range(self.control_decimation):
            self.scene.step()
        
        # Update episode counters
        self.episode_steps += 1
        
        # Get observation, reward, done
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: torch.Tensor):
        """Apply action to robot."""
        self.robot.apply_action(action)
    
    def _get_obs(self) -> Union[np.ndarray, Dict]:
        """Get observation."""
        if self.obs_mode == "state":
            return self._get_state_obs()
        elif self.obs_mode == "rgb":
            return self._get_rgb_obs()
        elif self.obs_mode == "rgbd":
            return self._get_rgbd_obs()
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _get_state_obs(self) -> np.ndarray:
        """Get state observation."""
        robot_state = self.robot.get_state()
        task_state = self.task.get_state()
        obs = torch.cat([robot_state, task_state], dim=-1)
        return obs.cpu().numpy()
    
    def _get_rgb_obs(self) -> np.ndarray:
        """Get RGB observation."""
        # Get camera images
        obs = {}
        for name, camera in self.cameras.items():
            obs[name] = camera.get_rgb()
        return obs
    
    def _get_rgbd_obs(self) -> np.ndarray:
        """Get RGBD observation."""
        obs = {}
        for name, camera in self.cameras.items():
            obs[name] = camera.get_rgbd()
        return obs
    
    def _get_reward(self) -> torch.Tensor:
        """Get reward from task."""
        return self.task.compute_reward()
    
    def _get_terminated(self) -> torch.Tensor:
        """Get terminated flags."""
        return self.task.check_success()
    
    def _get_truncated(self) -> torch.Tensor:
        """Get truncated flags (max episode length)."""
        return self.episode_steps >= self.max_episode_steps
    
    def _get_info(self) -> Dict:
        """Get info dict."""
        return {
            "episode_steps": self.episode_steps.clone(),
            "success": self.task.check_success(),
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return camera image
            if "base_camera" in self.cameras:
                return self.cameras["base_camera"].get_rgb()
        elif self.render_mode == "human":
            # Genesis viewer
            self.scene.viewer.update()
        return None
    
    def close(self):
        """Close the environment."""
        if self.scene is not None:
            self.scene = None
