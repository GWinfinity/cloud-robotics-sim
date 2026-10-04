"""Vectorized environment support for parallel training.

Enables running thousands of simulation environments in parallel
for efficient reinforcement learning.

The batched training entry point follows the official Genesis pattern
(``examples/manipulation/grasp_env.py`` + rsl-rl): a single ``gs.Scene``
built with ``scene.build(n_envs=...)`` drives all environments, and a
:class:`VecTask` implements observation, reward, and termination logic with
``(num_envs, ...)`` torch tensors that stay resident on the compute device.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

try:  # torch is a hard dependency of genesis-world, guard for import hygiene
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]


@dataclass
class VecEnvConfig:
    """Configuration for vectorized environments.

    Attributes:
        num_envs: Number of parallel environments.
        num_scenes_per_env: Number of scenes per environment.
        max_parallel: Maximum parallel workers.
        use_cuda: Whether to use CUDA acceleration.
        sim_dt: Physics timestep (seconds).
        sim_substeps: Simulation substeps per step.
        integrator: Rigid-body integrator (e.g. ``implicitfast``).
        solver_iterations: Newton constraint-solver iterations (fidelity knob).
        ls_iterations: Newton line-search iterations (fidelity knob).
        noslip_iterations: Contact fidelity knob; Genesis' own default is 0
            (the project's earlier default of 5 made the noslip kernel ~80%
            of CUDA step time — see AGENTS.md Vectorized Throughput Baseline).
        self_collision: Whether links of the same articulated entity collide
            with each other (False saves broadphase/narrowphase work).
        hibernation: Park envs whose bodies all move slower than the solver
            threshold (skips their dynamics until re-awakened by contact).
        cache_dir: Genesis simulation cache directory.
        dataset_pipeline: Stage cache for a downstream dataset pipeline.
        dataset_pipeline_dir: Staging directory for the dataset pipeline.
    """

    num_envs: int = 128
    num_scenes_per_env: int = 1
    max_parallel: int = 32
    use_cuda: bool = True
    # Physics / contact fidelity (exposed per the improvement plan so tasks
    # can tune them without touching engine internals).
    sim_dt: float = 0.02
    sim_substeps: int = 2
    integrator: str = "implicitfast"
    solver_iterations: int = 50
    ls_iterations: int = 50
    # Genesis' own RigidOptions default is 0, and profiling on 2026-10-04
    # (outputs/benchmarks/profile_vec_step_20261004) showed the noslip kernel
    # alone was ~80% of CUDA step time at noslip=5 while the within-round
    # A/B measurement showed noslip=0 consistently ~35-45% faster in wall
    # clock. Fidelity trade-off is the caller's: bump this for sticky-contact
    # tasks.
    noslip_iterations: int = 0
    self_collision: bool = True
    hibernation: bool = False
    # Throughput: gs.init(performance_mode=True) switches Genesis to static
    # arrays (no per-step ndarray conversion). It does not imply
    # non-determinism (that is gs.init's separate
    # ``use_deterministic_algorithms`` flag), but it does disable runtime
    # scene editing/rebuild — fine for batched data production.
    performance_mode: bool = False
    # Cache cleanup / dataset pipeline hand-off.
    cache_dir: str | Path = "outputs/sim_cache"
    dataset_pipeline: bool = False
    dataset_pipeline_dir: str | Path = "outputs/dataset_pipeline/staging"
    # Batched rendering (L2). ``render_config`` is a path to a yaml like
    # ``configs/render/batch_madrona.yaml`` or an equivalent dict; ``mode:
    # batch`` selects Genesis' BatchRenderer (Madrona). The optional package
    # ``gs-madrona`` is Linux-only — on other platforms building a scene with
    # a batch render config raises an actionable error.
    render_config: "str | Path | dict[str, Any] | None" = None


def load_render_config(
    config: "str | Path | dict[str, Any] | None",
) -> dict[str, Any] | None:
    """Normalize a render config (yaml path or dict) to a plain dict.

    Returns ``None`` when no rendering is configured. Raises ``ValueError``
    for unknown modes or malformed configs.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        data = dict(config)
    else:
        import yaml

        path = Path(config)
        if not path.exists():
            raise ValueError(f"render config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mode = data.get("mode", "batch")
    if mode != "batch":
        raise ValueError(
            f"unsupported render mode {mode!r} (only 'batch' is wired; "
            f"got keys {sorted(data)})"
        )
    data["mode"] = mode
    return data


def batch_renderer_available() -> bool:
    """Whether the optional ``gs-madrona`` batch renderer is importable."""
    return importlib.util.find_spec("gs_madrona") is not None


class VectorizedEnvironment:
    """Base class for vectorized environments.

    Manages multiple simulation instances running in parallel,
    providing a unified interface for batched step and reset.

    Attributes:
        config: Vectorized environment configuration.
        num_envs: Number of parallel environments.
        observation_space: Shared observation space specification.
        action_space: Shared action space specification.
    """

    def __init__(self, config: VecEnvConfig | None = None) -> None:
        self.config = config or VecEnvConfig()
        self.num_envs = self.config.num_envs
        self._envs: list[Any] = []

    def reset(self, seeds: list[int] | None = None) -> tuple[Any, list[dict]]:
        """Reset all environments.

        Args:
            seeds: Optional seeds for each environment.

        Returns:
            Tuple of (observations, infos). Observations are batched
            ``(num_envs, obs_dim)`` arrays (torch tensors on the compute
            device for Genesis-backed environments).
        """
        raise NotImplementedError

    def step(
        self,
        actions: Any,
    ) -> tuple[Any, Any, Any, Any, list[dict]]:
        """Step all environments with batched actions.

        Args:
            actions: Batched actions of shape (num_envs, action_dim).

        Returns:
            Tuple of (obs, reward, terminated, truncated, infos).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Clean up all environments."""
        pass


class VecTask(ABC):
    """Batched task interface for :class:`GenesisVectorizedEnv`.

    A ``VecTask`` owns everything that defines a learning task on top of a
    single batched Genesis scene: scene population, observation computation,
    reward, and termination. All tensors are ``(num_envs, ...)`` and live on
    the environment's compute device.

    The environment injects ``num_envs`` and ``seeds`` (a ``(num_envs,)``
    long tensor) before :meth:`setup` is called; tasks should sample reset
    randomness from ``self.seeds`` so per-environment seeds stay meaningful.

    Class attributes:
        num_observations: Dimension of the observation vector per env.
        num_actions: Dimension of the action vector per env.
    """

    num_observations: int = 0
    num_actions: int = 0
    num_envs: int = 0
    seeds: "torch.Tensor | None" = None

    @abstractmethod
    def build_scene(self, scene: Any) -> None:
        """Add entities (robot, objects, cameras) to the scene before build."""

    @abstractmethod
    def setup(self, scene: Any) -> None:
        """Post-build setup: cache entity handles, preallocate buffers.

        Called exactly once after ``scene.build(n_envs=...)``.
        """

    @abstractmethod
    def reset(self, envs_idx: Any = None) -> "torch.Tensor":
        """Reset the given environments and return their observations.

        Args:
            envs_idx: Optional tensor of environment indices to reset.
                ``None`` resets all environments.

        Returns:
            Observations of shape ``(len(envs_idx), num_observations)`` (or
            ``(num_envs, num_observations)`` when ``envs_idx`` is None).
        """

    @abstractmethod
    def step(
        self, actions: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", dict]:
        """Advance physics by one control step and compute task quantities.

        Implementations are responsible for calling ``scene.step()`` (with
        any desired substeps) and applying the actions to the robot.

        Args:
            actions: ``(num_envs, num_actions)`` action tensor.

        Returns:
            Tuple of (obs, reward, terminated, truncated, extras) where the
            first four are ``(num_envs, ...)`` tensors and ``extras`` is a
            dict of auxiliary per-env tensors.
        """


class GenesisVectorizedEnv(VectorizedEnvironment):
    """Genesis-based vectorized environment.

    Uses a single batched Genesis scene (``scene.build(n_envs=...)``) to
    drive all environments on GPU, following the official Genesis training
    pattern. Supports up to 4096 parallel environments on high-end GPUs.

    Example:
        >>> config = VecEnvConfig(num_envs=1024, use_cuda=True)
        >>> vec_env = GenesisVectorizedEnv(config, task=MyVecTask())
        >>> obs, info = vec_env.reset()
        >>> actions = torch.randn(1024, vec_env.num_actions)
        >>> obs, reward, done, trunc, info = vec_env.step(actions)
    """

    def __init__(
        self,
        config: VecEnvConfig | None = None,
        task: VecTask | None = None,
        scene_fn: Any = None,
    ) -> None:
        super().__init__(config)
        self.task = task
        self.scene_fn = scene_fn
        self.scene: Any = None
        self.device: "torch.device | None" = None
        self.episode_length: "torch.Tensor | None" = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @property
    def num_actions(self) -> int:
        """Action dimension defined by the task."""
        if self.task is None:
            raise RuntimeError("No task configured for this environment")
        return self.task.num_actions

    @property
    def num_observations(self) -> int:
        """Observation dimension defined by the task."""
        if self.task is None:
            raise RuntimeError("No task configured for this environment")
        return self.task.num_observations

    def initialize(self) -> None:
        """Initialize Genesis and build the batched scene.

        Idempotent. Requires a task; raises ``RuntimeError`` otherwise.
        """
        if self._initialized:
            return
        if self.task is None:
            raise RuntimeError(
                "GenesisVectorizedEnv requires a VecTask to initialize; "
                "pass task=... to the constructor"
            )
        if _torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for the vectorized environment")

        from cloud_robotics_sim.utils.genesis_compat import ensure_genesis_initialized

        ensure_genesis_initialized(
            use_cuda=self.config.use_cuda,
            performance_mode=self.config.performance_mode,
        )
        self.device = _torch.device(
            "cuda" if self.config.use_cuda and _torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Building batched scene with %d envs on %s", self.num_envs, self.device
        )

        scene = self.scene_fn() if self.scene_fn is not None else self._default_scene()

        self.task.num_envs = self.num_envs
        self.task.seeds = _torch.arange(self.num_envs, device=self.device)
        self.task.build_scene(scene)
        scene.build(n_envs=self.num_envs)
        self.task.setup(scene)

        self.scene = scene
        self.episode_length = _torch.zeros(
            self.num_envs, dtype=_torch.long, device=self.device
        )
        self._initialized = True

    def _default_scene(self) -> Any:
        """Create a default Genesis scene with contact options from config."""
        import genesis as gs

        # Genesis 1.4 takes the integrator as the ``gs.integrator`` enum, not
        # a string (``RigidOptions`` validates the instance type).
        integrator = {
            "euler": gs.integrator.Euler,
            "implicitfast": gs.integrator.implicitfast,
            "approximate_implicitfast": gs.integrator.approximate_implicitfast,
        }.get(self.config.integrator.lower())
        if integrator is None:
            raise ValueError(f"unknown integrator {self.config.integrator!r}")

        sim_options = gs.options.SimOptions(
            dt=self.config.sim_dt,
            substeps=self.config.sim_substeps,
        )
        # RigidOptions.dt defaults to the scene dt; passing it explicitly
        # alongside SimOptions.substeps makes the solver reject the pair
        # ("dt implies N substeps, conflicting with the requested substeps").
        rigid_options = gs.options.RigidOptions(
            integrator=integrator,
            iterations=self.config.solver_iterations,
            ls_iterations=self.config.ls_iterations,
            noslip_iterations=self.config.noslip_iterations,
            enable_self_collision=self.config.self_collision,
            use_hibernation=self.config.hibernation,
        )

        renderer = None
        render_cfg = load_render_config(self.config.render_config)
        if render_cfg is not None:
            if not batch_renderer_available():
                raise RuntimeError(
                    "render_config requests the Genesis batch renderer, but "
                    "the optional 'gs-madrona' package is not importable. "
                    "It is distributed as Linux-only wheels "
                    "(pip install gs-madrona); run batched rendering on a "
                    "Linux host or drop render_config for physics-only runs."
                )
            renderer = gs.renderers.BatchRenderer(
                use_rasterizer=bool(render_cfg.get("batch_use_rasterizer", True))
            )

        return gs.Scene(
            sim_options=sim_options,
            rigid_options=rigid_options,
            renderer=renderer,
            show_viewer=False,
        )

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def _resolve_envs_idx(self, envs_idx: Any) -> "torch.Tensor":
        """Normalize an optional index subset to a tensor on the device."""
        if envs_idx is None:
            return _torch.arange(self.num_envs, device=self.device)
        idx = _torch.as_tensor(envs_idx, device=self.device, dtype=_torch.long)
        if idx.numel() == 0:
            raise ValueError("envs_idx must not be empty")
        return idx.reshape(-1)

    def reset(
        self,
        seeds: list[int] | None = None,
        envs_idx: Any = None,
    ) -> tuple["torch.Tensor", list[dict]]:
        """Reset environments (all, or a subset via ``envs_idx``).

        Args:
            seeds: Optional per-environment seeds. When given, seeds for the
                reset environments are updated before ``task.reset`` runs.
            envs_idx: Optional subset of environment indices (mask-style
                partial reset). ``None`` resets all environments.

        Returns:
            Tuple of (obs, infos). ``obs`` is ``(len(envs_idx),
            num_observations)``; ``infos`` carries one dict per environment
            with at least the ``seed`` key.
        """
        if not self._initialized:
            self.initialize()
        task = self.task
        episode_length = self.episode_length
        task_seeds = task.seeds if task is not None else None
        if task is None or episode_length is None or task_seeds is None:
            # pragma: no cover - initialize() guards this
            raise RuntimeError("Environment is not properly initialized")

        idx = self._resolve_envs_idx(envs_idx)
        if seeds is not None:
            if len(seeds) != idx.numel():
                raise ValueError(
                    f"Expected {idx.numel()} seeds for the reset subset, "
                    f"got {len(seeds)}"
                )
            seed_tensor = _torch.as_tensor(
                seeds, device=self.device, dtype=task_seeds.dtype
            )
            task_seeds[idx] = seed_tensor

        obs = task.reset(idx)
        episode_length[idx] = 0

        infos = [{"seed": int(task_seeds[i])} for i in range(self.num_envs)]
        return obs, infos

    def reset_idx(self, envs_idx: Any) -> "torch.Tensor":
        """Reset a subset of environments (mask-style, rsl-rl convention).

        Args:
            envs_idx: Tensor of environment indices to reset.

        Returns:
            Observations of shape ``(len(envs_idx), num_observations)``.
        """
        obs, _ = self.reset(envs_idx=envs_idx)
        return obs

    def step(
        self,
        actions: Any,
    ) -> tuple[
        "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", list[dict]
    ]:
        """Execute one batched step across all environments.

        Args:
            actions: ``(num_envs, num_actions)`` actions; numpy arrays or
                tensors on any device are accepted and moved to the compute
                device.

        Returns:
            Tuple of (obs, reward, terminated, truncated, infos) with
            ``(num_envs, ...)`` torch tensors.
        """
        if not self._initialized:
            self.initialize()
        task = self.task
        episode_length = self.episode_length
        if task is None or episode_length is None:  # pragma: no cover
            raise RuntimeError("No task configured for this environment")

        actions = _torch.as_tensor(actions, device=self.device, dtype=_torch.float32)
        if actions.shape != (self.num_envs, task.num_actions):
            raise ValueError(
                f"Expected actions of shape ({self.num_envs}, "
                f"{task.num_actions}), got {tuple(actions.shape)}"
            )

        obs, rewards, terminated, truncated, _extras = task.step(actions)
        episode_length += 1

        expected = (self.num_envs,)
        for name, tensor in (
            ("reward", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"task.step returned {name} of shape {tuple(tensor.shape)}, "
                    f"expected {expected}"
                )

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up Genesis resources.

        If a downstream dataset pipeline is enabled, the simulation cache is
        staged for later processing.  Otherwise the cache is deleted and the
        Genesis runtime is released.
        """
        from cloud_robotics_sim.utils.cache_cleanup import cleanup_after_simulation

        cleanup_after_simulation(
            cache_dir=self.config.cache_dir,
            pipeline_dir=self.config.dataset_pipeline_dir,
            dataset_pipeline=self.config.dataset_pipeline,
        )
        logger.info("Vectorized environment closed")


# Type alias for backward compatibility
VectorizedEnv = VectorizedEnvironment
GenesisVecEnv = GenesisVectorizedEnv
