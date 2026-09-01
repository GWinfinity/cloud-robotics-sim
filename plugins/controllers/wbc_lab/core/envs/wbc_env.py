"""WBC Motion Tracking Environment for Genesis.

Ported from wbc_lab/env/wbc_env_cfg.py  -the robot-agnostic WBC MDP
with multi-clip motion tracking, adaptive RSI, and assistive wrench.

This environment implements the shared MDP from WBC-Lab using Genesis
physics instead of MuJoCo/mjlab.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import genesis as gs
    import torch

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

from ..motion.motion_command import MotionCommand, MotionCommandCfg
from ..motion.sampling import AdaptiveRsiSampler, RsiCfg
from ..robots.g1_config import G1RobotConfig
from .rewards import RewardComputer, RewardConfig
from .terminations import TerminationChecker, TerminationConfig

logger = logging.getLogger(__name__)


@dataclass
class WbcEnvConfig:
    """Configuration for the WBC Genesis environment."""

    # Physics
    dt: float = 0.005
    substeps: int = 20
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Scene
    headless: bool = False
    show_viewer: bool = True
    camera_pos: tuple[float, float, float] = (3.0, 3.0, 2.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_fov: int = 45

    # Environment
    num_envs: int = 1
    max_episode_steps: int = 1000
    device: str = "cuda"

    # Robot
    robot: G1RobotConfig = field(default_factory=G1RobotConfig)

    # Motion
    motion_cfg: MotionCommandCfg = field(default_factory=MotionCommandCfg)

    # RSI
    rsi_cfg: RsiCfg = field(default_factory=RsiCfg)

    # Rewards
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)

    # Terminations
    term_cfg: TerminationConfig = field(default_factory=TerminationConfig)

    # Action control
    action_scale: float = 0.25
    clip_actions: float = 1.0


class WbcGenesisEnv:
    """WBC Motion Tracking Environment on Genesis physics engine.

    Implements the shared WBC MDP from WBC-Lab:
      - Multi-clip motion library playback (NPZ format)
      - Adaptive RSI (Reference State Initialization)
      - Tracking rewards: joint pos/vel, anchor pos/ori, keybody
      - Regularization rewards: action rate, torque limits, collision
      - Assistive wrench curriculum (optional)
      - Deploy export (ONNX policy + tracking params YAML)

    Usage:
        >>> from plugins.controllers.wbc_lab import WbcGenesisEnv
        >>> env = WbcGenesisEnv(config)
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = policy(obs)
        ...     obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, cfg: WbcEnvConfig | None = None):
        if not HAS_GENESIS:
            raise ImportError(
                "Genesis is required. Install with: pip install genesis-world"
            )

        self.cfg = cfg or WbcEnvConfig()
        self.num_envs = self.cfg.num_envs
        self.device = self.cfg.device

        # Initialize Genesis
        backend = (
            gs.cuda
            if self.device == "cuda" and torch.cuda.is_available()
            else gs.cpu
        )
        if not gs._initialized:
            gs.init(backend=backend)

        # Create scene
        self.scene = self._create_scene()

        # Add ground plane
        self.ground = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0), roughness=0.8),
        )

        # Add robot
        self.robot = self._create_robot()

        # Build scene
        self.scene.build(n_envs=self.num_envs)

        # Robot dimensions
        self.n_dofs = self.robot.n_dofs
        self.num_actions = self.n_dofs

        # Load motion library
        if self.cfg.motion_cfg.motion_path:
            self.motion_cmd = MotionCommand(
                self.cfg.motion_cfg,
                num_envs=self.num_envs,
                device=self.device,
            )
            # RSI sampler
            if self.motion_cmd.clips:
                clip0 = self.motion_cmd.clips[0]
                self.rsi_sampler = AdaptiveRsiSampler(
                    self.cfg.rsi_cfg,
                    clip_duration_s=clip0.duration,
                    fps=clip0.fps,
                )
            else:
                self.rsi_sampler = None
        else:
            self.motion_cmd = None
            self.rsi_sampler = None

        # Reward / termination
        self.reward_computer = RewardComputer(self.cfg.reward_cfg, self.num_envs)
        self.termination_checker = TerminationChecker(self.cfg.term_cfg, self.num_envs)

        # Default joint positions
        self.default_joint_pos = np.array(self.cfg.robot.home_qpos, dtype=np.float32)
        if len(self.default_joint_pos) < self.n_dofs:
            self.default_joint_pos = np.zeros(self.n_dofs, dtype=np.float32)

        # Episode state
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        # Observation dimensions
        self._obs_dim = self._compute_obs_dim()

        logger.info(
            "WbcGenesisEnv: %d envs, %d DOFs, %d obs, motion=%s",
            self.num_envs,
            self.n_dofs,
            self._obs_dim,
            self.cfg.motion_cfg.motion_path or "none",
        )

    @property
    def observation_space_dim(self) -> int:
        return self._obs_dim

    @property
    def action_space_dim(self) -> int:
        return self.num_actions

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension."""
        dim = 0
        # Joint state
        dim += self.n_dofs * 2  # qpos + qvel
        # Last action
        dim += self.n_dofs
        # IMU: base angular velocity + projected gravity
        dim += 3 + 3
        # Gait phase (sin/cos)
        dim += 2
        # Reference features (if motion tracking)
        if self.motion_cmd is not None:
            dim += self.n_dofs  # ref_joint_pos
            dim += self.n_dofs  # ref_joint_vel
            dim += 3  # ref_base_height (1) + ref_base_lin_vel (2)
            dim += 3  # ref_anchor_pos
        return dim

    def _create_scene(self):
        """Create Genesis scene."""
        return gs.Scene(
            viewer_options=(
                gs.options.ViewerOptions(
                    camera_pos=self.cfg.camera_pos,
                    camera_lookat=self.cfg.camera_lookat,
                    camera_fov=self.cfg.camera_fov,
                    res=(1280, 720),
                    max_FPS=60,
                )
                if not self.cfg.headless
                else None
            ),
            sim_options=gs.options.SimOptions(
                dt=self.cfg.dt,
                substeps=self.cfg.substeps,
            ),
            show_viewer=not self.cfg.headless and self.cfg.show_viewer,
        )

    def _create_robot(self):
        """Load G1 robot into Genesis scene."""
        robot_cfg = self.cfg.robot

        # Try to find G1 URDF/MJCF
        urdf_path = robot_cfg.urdf_path
        if urdf_path is None:
            # Search common locations
            candidates = [
                Path("assets/g1/g1.urdf"),
                Path("assets/g1/g1.xml"),
                Path(__file__).parent.parent.parent / "assets" / "g1" / "g1.urdf",
            ]
            for c in candidates:
                if c.exists():
                    urdf_path = str(c)
                    break

        if urdf_path is not None and Path(urdf_path).exists():
            if urdf_path.endswith(".xml"):
                return self.scene.add_entity(
                    morph=gs.morphs.MJCF(file=urdf_path),
                    surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)),
                )
            else:
                return self.scene.add_entity(
                    morph=gs.morphs.URDF(file=urdf_path),
                    surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)),
                )
        else:
            # Fallback: Genesis built-in humanoid
            logger.warning("G1 URDF not found, using Genesis built-in humanoid")
            import genesis as gs

            humanoid_path = os.path.join(gs.__path__[0], "assets", "xml", "humanoid.xml")
            return self.scene.add_entity(
                morph=gs.morphs.MJCF(file=humanoid_path),
                surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)),
            )

    def _to_numpy(self, tensor: Any, max_len: int | None = None) -> np.ndarray:
        """Convert tensor to numpy array."""
        if hasattr(tensor, "cpu"):
            arr = tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            arr = np.array(tensor)
        return arr[:max_len] if max_len else arr

    def reset(
        self,
        env_ids: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environments.

        Args:
            env_ids: Environment indices to reset. None = all.
            seed: Random seed for reproducibility.

        Returns:
            (observations, info_dict)
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.step_count[env_ids] = 0
        self.last_actions[env_ids] = 0.0

        # Reset motion command (new clip assignment)
        if self.motion_cmd is not None:
            self.motion_cmd.reset(env_ids)

        # Reset termination checker
        self.termination_checker.reset(env_ids)

        # Reset reward accumulators
        self.reward_computer.reset_episode(env_ids)

        # RSI: sample start frame and set robot state
        if self.rsi_sampler is not None and self.motion_cmd is not None:
            rng = np.random.default_rng(seed)
            for i in env_ids:
                start_frame = self.rsi_sampler.sample_start_frame(rng)
                self._set_robot_to_reference(int(i), start_frame)
        else:
            # Default reset
            self._reset_robot_default(env_ids)

        # Get observations
        obs = self._get_observations(env_ids)

        info: dict[str, Any] = {
            "env_ids": env_ids,
            "step_count": self.step_count[env_ids].copy(),
        }
        if self.motion_cmd is not None:
            info["clip_idx"] = self.motion_cmd.get_clip_idx()[env_ids]

        return obs, info

    def _reset_robot_default(self, env_ids: np.ndarray) -> None:
        """Reset robot to default home position."""
        qpos = self.default_joint_pos.copy()
        for i in env_ids:
            self.robot.set_dofs_position(
                torch.tensor(qpos, dtype=torch.float32).unsqueeze(0),
                indices=[int(i)],
            )

    def _set_robot_to_reference(self, env_idx: int, frame: int) -> None:
        """Set robot state to match motion reference at given frame."""
        if self.motion_cmd is None:
            return

        clip_idx = int(self.motion_cmd._current_clip_idx[env_idx])
        clip = self.motion_cmd.clips[clip_idx]
        frame = frame % clip.num_frames

        # Set joint positions from reference
        qpos = clip.joint_pos[frame].astype(np.float32)
        if len(qpos) < self.n_dofs:
            qpos = np.pad(qpos, (0, self.n_dofs - len(qpos)))
        qpos = qpos[: self.n_dofs]

        self.robot.set_dofs_position(
            torch.tensor(qpos, dtype=torch.float32).unsqueeze(0),
            indices=[env_idx],
        )

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: (num_envs, num_actions) or (num_actions,) action array.

        Returns:
            (obs, rewards, terminated, truncated, info)
        """
        actions = np.atleast_2d(actions).astype(np.float32)
        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = np.tile(actions, (self.num_envs, 1))
        actions = np.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions)

        # Apply actions: PD position control
        target_pos = self.default_joint_pos + actions * self.cfg.action_scale
        self.robot.control_dofs_position(
            torch.tensor(target_pos, dtype=torch.float32),
        )

        # Step physics
        self.scene.step()

        # Advance motion playback
        if self.motion_cmd is not None:
            self.motion_cmd.advance(self.cfg.dt)

        # Get current state
        joint_pos = self._to_numpy(self.robot.get_dofs_position(), self.n_dofs)
        joint_vel = self._to_numpy(self.robot.get_dofs_velocity(), self.n_dofs)
        base_pos = self._to_numpy(self.robot.get_pos(), 3)
        base_quat = self._to_numpy(self.robot.get_quat(), 4)
        base_vel = self._to_numpy(self.robot.get_vel(), 3)
        base_ang_vel = self._to_numpy(self.robot.get_ang(), 3)

        # Compute rewards
        if self.motion_cmd is not None:
            ref_jp = self.motion_cmd.ref_joint_pos()
            ref_jv = self.motion_cmd.ref_joint_vel()
            ref_bp = self.motion_cmd.ref_base_pos()

            jp_error = joint_pos - ref_jp
            jv_error = joint_vel - ref_jv
            anchor_pos_error = base_pos - ref_bp
            anchor_ori_error = np.linalg.norm(anchor_pos_error[:, :2], axis=-1)

            rewards, reward_terms = self.reward_computer.compute_all(
                joint_pos_error=jp_error,
                joint_vel_error=jv_error,
                anchor_pos_error=anchor_pos_error,
                anchor_ori_error=anchor_ori_error,
                anchor_lin_vel_error=np.abs(base_vel[:, 0] - 0.0),
                anchor_ang_vel_error=np.abs(base_ang_vel[:, 2] - 0.0),
                actions=actions,
                last_actions=self.last_actions,
                joint_vel=joint_vel,
            )

            # Update RSI failure levels
            if self.rsi_sampler is not None:
                for i in range(self.num_envs):
                    clip_idx = int(self.motion_cmd._current_clip_idx[i])
                    frame = int(self.motion_cmd._current_frame[i])
                    bin_idx = self.rsi_sampler.bin_for_frame(frame)
                    self.rsi_sampler.step_tracking_similarity(
                        {k: float(v[i]) for k, v in reward_terms.items()},
                        bin_idx,
                    )
        else:
            rewards = np.zeros(self.num_envs)

        # Check terminations
        base_height = base_pos[:, 2] if base_pos.ndim > 1 else np.array([base_pos[2]])
        terminated, truncated, term_info = self.termination_checker.check(
            base_pos=base_pos,
            ref_base_pos=ref_bp if self.motion_cmd is not None else None,
            base_height=base_height,
            anchor_ori_error=anchor_ori_error if self.motion_cmd is not None else None,
        )

        # Add termination penalty
        rewards += self.reward_computer.termination(terminated)

        # Update state
        self.last_actions = actions.copy()
        self.step_count += 1

        # Get new observations
        obs = self._get_observations()

        # Auto-reset terminated/truncated environments
        done_mask = terminated | truncated
        done_ids = np.where(done_mask)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        info: dict[str, Any] = {
            "step_count": self.step_count.copy(),
            "terminated": terminated.copy(),
            "truncated": truncated.copy(),
            "term_info": term_info,
        }
        if self.motion_cmd is not None:
            info["clip_idx"] = self.motion_cmd.get_clip_idx()

        return obs, rewards, terminated, truncated, info

    def _get_observations(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Compute observations for specified environments."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        n = len(env_ids)
        parts: list[np.ndarray] = []

        # Joint state
        qpos = self._to_numpy(self.robot.get_dofs_position(), self.n_dofs)
        qvel = self._to_numpy(self.robot.get_dofs_velocity(), self.n_dofs)
        parts.append(qpos[:n])
        parts.append(qvel[:n])

        # Last action
        parts.append(self.last_actions[env_ids])

        # IMU
        base_ang_vel = self._to_numpy(self.robot.get_ang(), 3)[:n]
        base_quat = self._to_numpy(self.robot.get_quat(), 4)[:n]
        projected_gravity = self._quat_rotate_inverse(base_quat, np.array([0, 0, -1]))
        parts.append(base_ang_vel)
        parts.append(projected_gravity)

        # Gait phase (dummy if no motion)
        if self.motion_cmd is not None:
            phase = (
                self.motion_cmd._current_frame[env_ids]
                / (self.motion_cmd.clips[0].num_frames if self.motion_cmd.clips else 100)
            ) % 1.0
        else:
            phase = np.zeros(n)
        parts.append(np.stack([np.sin(phase * 2 * np.pi), np.cos(phase * 2 * np.pi)], axis=-1))

        # Reference features
        if self.motion_cmd is not None:
            ref_jp = self.motion_cmd.ref_joint_pos(env_ids)
            ref_jv = self.motion_cmd.ref_joint_vel(env_ids)
            ref_height = self.motion_cmd.ref_base_height(env_ids)
            ref_bp = self.motion_cmd.ref_base_pos(env_ids)
            parts.append(ref_jp[:n])
            parts.append(ref_jv[:n])
            parts.append(ref_height[:n])
            parts.append(ref_bp[:n, :2])  # xy only

        obs = np.concatenate(parts, axis=-1).astype(np.float32)
        return obs

    def _quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by inverse of quaternion q."""
        # q = (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # v' = q^-1 * v * q
        t = 2.0 * np.cross(q[:, 1:4], v[np.newaxis, :].repeat(len(q), axis=0))
        result = v[np.newaxis, :].repeat(len(q), axis=0) + w[:, np.newaxis] * t + np.cross(q[:, 1:4], t)
        return result

    def get_observation_dim(self) -> int:
        """Get observation dimension."""
        return self._obs_dim

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.num_actions

    def close(self) -> None:
        """Clean up Genesis resources."""
        try:
            gs.destroy()
        except Exception:
            pass
