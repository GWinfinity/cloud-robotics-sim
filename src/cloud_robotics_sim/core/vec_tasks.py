"""Real :class:`VecTask` implementations for batched Genesis scenes.

:mod:`cloud_robotics_sim.core.vectorized` provides the batched environment
machinery (``scene.build(n_envs=...)`` + tensor-in/tensor-out interface); this
module provides the first *real* task running on it — a Franka Panda
pick-cube task matching the semantics of ``configs/tasks/pick_place_cube.yaml``
(franka panda, 0.05 m red cube, target ``[0.5, 0.0, 0.1]``, cube initialized
over x∈[0.4, 0.6], y∈[-0.2, 0.2]).

Design notes (Genesis 1.4 quirks, cf. AGENTS.md):

- Entities are unbuilt at spawn time: the asset-default qpos is captured and
  PD gains (``set_dofs_kp``/``set_dofs_kv`` on the raw entity — backend
  *wrappers* expose ``set_dofs_gains`` but raw entities do not) are applied on
  the **first** ``reset()``, not in ``build_scene``.
- GPU-backed scenes return CUDA tensors from ``get_qpos``/``get_pos`` — all
  buffers are preallocated on the task device and getters use ``envs_idx``
  subset reads, so nothing crosses the host in the step hot path.
- v1 intentionally has no cameras: this measures physics/control throughput.
  Batched rendering (Madrona) is the L2 follow-up.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from cloud_robotics_sim.core.robot_assets import resolve_robot_model
from cloud_robotics_sim.core.vectorized import VecTask

logger = logging.getLogger(__name__)

#: Franka panda URDF: 7 arm dofs + 2 finger dofs (no mimic tendon in URDF).
ARM_DOFS = 7
GRIPPER_DOFS = 2
NUM_ROBOT_QS = ARM_DOFS + GRIPPER_DOFS

#: Observation: qpos(9) + qvel(9) + (cube_pos - target)(3) + cube_vel(3).
NUM_OBSERVATIONS = 2 * NUM_ROBOT_QS + 6
#: Action: delta joint-position targets for all 9 dofs.
NUM_ACTIONS = NUM_ROBOT_QS

#: PD gains applied at first reset (matches EmbodimentConfig defaults).
_JOINT_STIFFNESS = 100.0
_JOINT_DAMPING = 10.0
#: Delta-position action scales (arm / gripper), aligned with ManiSkill's
#: pd_joint_delta_pos controller semantics.
_ARM_ACTION_SCALE = 0.1
_GRIPPER_ACTION_SCALE = 0.04
#: Joint-position target safety clamp (rad).
_TARGET_CLAMP = 3.0

#: Task geometry (consistent with configs/tasks/pick_place_cube.yaml).
CUBE_SIZE = 0.05
CUBE_RHO = 800.0  # 0.1 kg for a 0.05 m cube (rho=, not density=, in 1.4)
CUBE_FRICTION = 0.5
DEFAULT_TARGET = (0.5, 0.0, 0.1)
INIT_X_RANGE = (0.4, 0.6)
INIT_Y_RANGE = (-0.2, 0.2)
SUCCESS_THRESHOLD = 0.05
DEFAULT_MAX_EPISODE_STEPS = 200
_SUCCESS_BONUS = 1.0


def cube_xy_from_uniform(u: torch.Tensor) -> torch.Tensor:
    """Map uniform samples ``(n, 2)`` in ``[0, 1)`` to the cube init region.

    Pure function (no Genesis, no global RNG) so the reset distribution is
    unit-testable and deterministic per seed.
    """
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"expected (n, 2) uniform samples, got {tuple(u.shape)}")
    lo_x, hi_x = INIT_X_RANGE
    lo_y, hi_y = INIT_Y_RANGE
    xy = torch.empty_like(u)
    xy[:, 0] = lo_x + (hi_x - lo_x) * u[:, 0]
    xy[:, 1] = lo_y + (hi_y - lo_y) * u[:, 1]
    return xy


def pick_cube_reward(
    cube_pos: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shaped reward and success mask from cube positions ``(n, 3)``.

    Returns ``(-distance, distance < threshold)`` — pure and device-agnostic
    so it is unit-testable without Genesis.
    """
    dist = torch.linalg.norm(cube_pos - target, dim=-1)
    return -dist, dist < SUCCESS_THRESHOLD


class FrankaPickCubeVecTask(VecTask):
    """Batched Franka pick-cube task on a single ``gs.Scene(n_envs=...)``.

    Class attributes:
        num_observations: 24 (see module docstring).
        num_actions: 9 delta joint-position targets.
    """

    num_observations = NUM_OBSERVATIONS
    num_actions = NUM_ACTIONS

    def __init__(
        self,
        target: tuple[float, float, float] = DEFAULT_TARGET,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        urdf_path: str | None = None,
        add_camera: bool = False,
        camera_res: tuple[int, int] = (256, 256),
        camera_fov: float = 60.0,
    ) -> None:
        self.target = torch.tensor(target, dtype=torch.float32)
        self.max_episode_steps = int(max_episode_steps)
        self.urdf_path = urdf_path
        # Optional batched third-person camera (L2 render path). Renders every
        # step when present — pair with VecEnvConfig.render_config for the
        # Madrona batch renderer; without a batch renderer Genesis falls back
        # to the (slow, per-env) rasterizer.
        self.add_camera = bool(add_camera)
        self.camera_res = tuple(camera_res)
        self.camera_fov = float(camera_fov)

        self.robot: Any = None
        self.cube: Any = None
        self.camera: Any = None
        self.last_rgb: torch.Tensor | None = None
        self.scene: Any = None
        self.device: torch.device | None = None

        # Captured on first reset (entity is unbuilt before scene.build()).
        self._home_qpos: torch.Tensor | None = None
        self._gains_applied = False
        self._episode_steps: torch.Tensor | None = None
        self._reset_epoch = 0
        self._action_scale: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def build_scene(self, scene: Any) -> None:
        """Add ground, cube and Franka to the scene before ``scene.build``."""
        import genesis as gs

        self.scene = scene
        scene.add_entity(gs.morphs.Plane())
        self.cube = scene.add_entity(
            gs.morphs.Box(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                pos=(DEFAULT_TARGET[0], 0.0, CUBE_SIZE / 2.0),
            ),
            gs.materials.Rigid(rho=CUBE_RHO, friction=CUBE_FRICTION),
        )

        model = resolve_robot_model("franka_panda", self.urdf_path)
        if model is not None:
            if model.format == "mjcf":
                morph = gs.morphs.MJCF(file=str(model.path), pos=(0.0, 0.0, 0.0))
            else:
                # fixed=True welds the base link (floating-base impact NaNs
                # the constraint solver on CPU — same rationale as
                # core/embodiment.py::FrankaPanda.spawn).
                morph = gs.morphs.URDF(
                    file=str(model.path), pos=(0.0, 0.0, 0.0), fixed=True
                )
            self.robot = scene.add_entity(morph)
            logger.info("vec task robot asset: %s:%s", model.format, model.path)
        else:
            # Last resort: Genesis' built-in model lookup.
            self.robot = scene.add_entity(
                gs.morphs.MJCF(file="franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0))
            )
            logger.warning("vec task robot: genesis built-in franka fallback")

        if self.add_camera:
            # Cameras must join before scene.build(). Same third-person pose
            # family as configs/render/batch_madrona.yaml consumers.
            self.camera = scene.add_camera(
                res=self.camera_res,
                pos=(0.8, 0.0, 0.5),
                lookat=(DEFAULT_TARGET[0], 0.0, 0.1),
                fov=self.camera_fov,
            )

    def setup(self, scene: Any) -> None:
        """Cache handles and preallocate buffers after ``scene.build``."""
        if self.seeds is None:
            raise RuntimeError("seeds not injected — build the env first")
        self.device = self.seeds.device
        self.target = self.target.to(self.device)
        self._episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Cached once — creating this per step costs a host->device copy in
        # the hot path.
        self._action_scale = torch.tensor(
            [_ARM_ACTION_SCALE] * ARM_DOFS + [_GRIPPER_ACTION_SCALE] * GRIPPER_DOFS,
            device=self.device,
        )
        # Preallocated obs buffer: the step hot path writes into it instead of
        # allocating a fresh (num_envs, 24) tensor every step (launch-bound
        # GPUs care more about kernel count than allocations, but cat's
        # output allocation is avoidable). Callers must not hold the returned
        # tensor across steps; the rsl-rl adapter copies into its rollout
        # storage, so the standard training loop is safe.
        self._obs_buf = torch.zeros(self.num_envs, NUM_OBSERVATIONS, device=self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batched(self, tensor: torch.Tensor, width: int) -> torch.Tensor:
        """Normalize a getter result to ``(n, width)`` on the device.

        ``n`` follows the getter result (``num_envs`` for full reads, the
        subset size for ``envs_idx`` reads — Genesis squeezes neither).
        """
        out = torch.as_tensor(tensor, device=self.device, dtype=torch.float32)
        return out.reshape(-1, width)

    def _read_state(
        self, envs_idx: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """qpos/qvel/cube_pos/cube_vel for a subset (or all) envs."""
        idx = envs_idx
        qpos = self._batched(self.robot.get_qpos(envs_idx=idx), NUM_ROBOT_QS)
        qvel = self._batched(self.robot.get_dofs_velocity(envs_idx=idx), NUM_ROBOT_QS)
        cube_pos = self._batched(self.cube.get_pos(envs_idx=idx), 3)
        cube_vel = self._batched(self.cube.get_vel(envs_idx=idx), 3)
        return qpos, qvel, cube_pos, cube_vel

    def _write_obs(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        cube_pos: torch.Tensor,
        cube_vel: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the 24-dim observation into ``out`` (default: the shared
        preallocated buffer).

        Kernel-count-optimized for the launch-bound hot path: one ``cat``
        into ``out=`` plus one in-place subtract, instead of ``cat`` +
        separate ``sub`` with temporaries. The default buffer is reused
        across steps — do not hold the returned tensor across steps.
        Partial resets (subset width) must pass their own ``out``.
        """
        buf = self._obs_buf if out is None else out
        torch.cat(
            [qpos, qvel, cube_pos, cube_vel],
            dim=-1,
            out=buf,
        )
        buf[:, 2 * NUM_ROBOT_QS : 2 * NUM_ROBOT_QS + 3].sub_(self.target)
        return buf

    def _ensure_ready(self) -> None:
        """First-reset one-time work: capture home qpos, apply PD gains.

        Genesis entities are unbuilt at spawn time, so the asset-default qpos
        and the PD gains can only be handled here (AGENTS.md, W4 notes).
        """
        if self._gains_applied:
            return
        qpos0 = self._batched(self.robot.get_qpos(), NUM_ROBOT_QS)
        if qpos0.shape[0] != self.num_envs:
            raise RuntimeError(
                f"expected qpos for {self.num_envs} envs, "
                f"got shape {tuple(qpos0.shape)}"
            )
        self._home_qpos = qpos0[0].clone()
        self.robot.set_dofs_kp(
            torch.full((NUM_ROBOT_QS,), _JOINT_STIFFNESS, device=self.device)
        )
        self.robot.set_dofs_kv(
            torch.full((NUM_ROBOT_QS,), _JOINT_DAMPING, device=self.device)
        )
        self._gains_applied = True

    # ------------------------------------------------------------------
    # VecTask interface
    # ------------------------------------------------------------------

    def reset(self, envs_idx: Any = None) -> torch.Tensor:
        """Reset the given envs (or all) and return their observations."""
        if self.device is None or self._episode_steps is None or self.seeds is None:
            raise RuntimeError("setup() has not run — build the env first")
        self._ensure_ready()
        idx = (
            torch.arange(self.num_envs, device=self.device)
            if envs_idx is None
            else torch.as_tensor(
                envs_idx, device=self.device, dtype=torch.long
            ).reshape(-1)
        )
        n = idx.numel()

        # Robot back to the captured asset-default pose (zeroes velocity).
        assert self._home_qpos is not None
        self.robot.set_qpos(self._home_qpos.unsqueeze(0).expand(n, -1), envs_idx=idx)

        # Per-env seeded cube placement: uniforms from a CPU generator keyed
        # on the env seeds + reset epoch, then a pure region mapping.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(
            (int(self.seeds[idx].sum()) + self._reset_epoch * 7919) % (2**31 - 1)
        )
        # Genesis sets torch's default device to the scene device inside its
        # device context, so the CPU-generator draw must pin device="cpu".
        u = torch.rand(n, 2, generator=gen, device="cpu")
        xy = cube_xy_from_uniform(u).to(self.device)
        cube_pos = torch.cat(
            [xy, torch.full((n, 1), CUBE_SIZE / 2.0, device=self.device)], dim=-1
        )
        self.cube.set_pos(cube_pos, envs_idx=idx)

        self._episode_steps[idx] = 0
        self._reset_epoch += 1

        qpos, qvel, pos, vel = self._read_state(idx)
        if n == self.num_envs:
            return self._write_obs(qpos, qvel, pos, vel)
        # partial reset: subset-width obs cannot land in the shared buffer
        subset_buf = torch.empty(
            n, NUM_OBSERVATIONS, dtype=torch.float32, device=self.device
        )
        return self._write_obs(qpos, qvel, pos, vel, out=subset_buf)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply delta-position actions, advance physics, compute quantities."""
        if self.device is None or self._episode_steps is None:
            raise RuntimeError("setup() has not run — build the env first")
        if self._home_qpos is None:
            raise RuntimeError("reset() must run before step()")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if self._action_scale is None:
            raise RuntimeError("setup() has not run — build the env first")
        # addcmul fuses home + action*scale into one kernel; clamp_ is in-place
        # on the fresh result (launch-bound GPUs: fewer launches > fewer allocs).
        targets = torch.addcmul(self._home_qpos, actions, self._action_scale).clamp_(
            -_TARGET_CLAMP, _TARGET_CLAMP
        )
        self.robot.control_dofs_position(targets)

        self.scene.step()

        qpos, qvel, cube_pos, cube_vel = self._read_state(None)
        obs = self._write_obs(qpos, qvel, cube_pos, cube_vel)
        # reward = -dist + 1.0 * success in as few launches as possible
        # (semantic match with pick_cube_reward + _SUCCESS_BONUS).
        dist = torch.linalg.vector_norm(cube_pos - self.target, dim=-1)
        success = dist < SUCCESS_THRESHOLD
        reward = torch.sub(success.float(), dist)

        if self.camera is not None:
            # Batched render; last frame kept on device for training loops.
            self.last_rgb = self.camera.render()

        self._episode_steps += 1
        terminated = success
        truncated = self._episode_steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, {"success": success}


__all__ = [
    "ARM_DOFS",
    "GRIPPER_DOFS",
    "NUM_ACTIONS",
    "NUM_OBSERVATIONS",
    "FrankaPickCubeVecTask",
    "cube_xy_from_uniform",
    "pick_cube_reward",
]
