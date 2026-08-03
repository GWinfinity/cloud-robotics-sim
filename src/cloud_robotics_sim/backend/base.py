"""Backend abstract base classes.

Defines the interface contract between the core simulation logic and
concrete physics/rendering backends (Genesis, MT Lambda, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cloud_robotics_sim.backend.types import (
    ArticulationState,
    BackendName,
    DeformableConfig,
    DeformableState,
    LightDescription,
    PhysicsState,
    Pose,
    RenderOutput,
    ViewerOptions,
)


class EntityBackend(ABC):
    """Backend-agnostic interface for a rigid body entity.

    Entities are static or dynamic objects in the scene (boxes, spheres,
    meshes, etc.). They expose pose/velocity queries and basic force
    application without requiring joint-space state.
    """

    @property
    @abstractmethod
    def name(self) -> str | None:
        """Return the entity name, if any."""
        ...

    @abstractmethod
    def get_pos(self) -> np.ndarray:
        """Return world-space position as a length-3 float64 array."""
        ...

    @abstractmethod
    def set_pos(self, pos: np.ndarray) -> None:
        """Set world-space position."""
        ...

    @abstractmethod
    def get_quat(self) -> np.ndarray:
        """Return orientation as a length-4 (w, x, y, z) float64 array."""
        ...

    @abstractmethod
    def set_quat(self, quat: np.ndarray) -> None:
        """Set orientation quaternion (w, x, y, z)."""
        ...

    @abstractmethod
    def set_color(self, color: tuple[float, float, float, float]) -> None:
        """Update entity color for visualization."""
        ...

    @abstractmethod
    def apply_force(
        self,
        force: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> None:
        """Apply a world-space force (optionally at a specific point)."""
        ...


class ArticulationBackend(EntityBackend):
    """Backend-agnostic interface for articulated robots.

    In addition to the pose interface inherited from EntityBackend, an
    articulation exposes joint-space state and control methods.
    """

    @property
    @abstractmethod
    def n_dofs(self) -> int:
        """Number of actuated degrees of freedom."""
        ...

    @property
    @abstractmethod
    def n_qs(self) -> int:
        """Number of generalized coordinates (qpos size)."""
        ...

    @abstractmethod
    def get_qpos(self) -> np.ndarray:
        """Return generalized positions."""
        ...

    @abstractmethod
    def set_qpos(self, qpos: np.ndarray) -> None:
        """Set generalized positions."""
        ...

    @abstractmethod
    def get_qvel(self) -> np.ndarray:
        """Return generalized velocities."""
        ...

    @abstractmethod
    def set_qvel(self, qvel: np.ndarray) -> None:
        """Set generalized velocities."""
        ...

    @abstractmethod
    def control_dofs_position(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        """Set joint position targets for internal PD controllers."""
        ...

    @abstractmethod
    def control_dofs_velocity(self, targets: np.ndarray) -> None:
        """Set joint velocity targets."""
        ...

    @abstractmethod
    def control_dofs_force(self, targets: np.ndarray) -> None:
        """Set joint force/torque targets."""
        ...

    @abstractmethod
    def get_state_batch(
        self,
        env_ids: list[int] | None = None,
    ) -> ArticulationState:
        """Batch-read qpos/qvel/pos/quat in a single backend call.

        This is the primary optimization path for vectorized environments
        and tight observation loops.
        """
        ...

    @abstractmethod
    def set_state_batch(
        self,
        state: ArticulationState,
        env_ids: list[int] | None = None,
    ) -> None:
        """Batch-write qpos/qvel/pos/quat in a single backend call."""
        ...

    @abstractmethod
    def control_position_batch(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        """Batch position control for vectorized environments.

        Args:
            targets: Array of shape (num_envs, n_dofs) or (n_dofs,).
            stiffness: Optional per-joint PD stiffness.
            damping: Optional per-joint PD damping.
        """
        ...

    @abstractmethod
    def get_end_effector_pose(self) -> Pose:
        """Return the current end-effector pose."""
        ...

    @abstractmethod
    def get_joint_names(self) -> list[str]:
        """Return the names of all joints in the articulation.

        Returns:
            List of joint names in the order reported by the backend.
        """
        ...

    @abstractmethod
    def get_joint_dofs_idx_local(self, joint_name: str) -> list[int]:
        """Return the local DoF indices for a given joint.

        Args:
            joint_name: Name of the joint to query.

        Returns:
            List of local DoF indices. Empty if the joint is not found.
        """
        ...

    @abstractmethod
    def get_joint_qs_idx_local(self, joint_name: str) -> list[int]:
        """Return the local generalized-position indices for a given joint.

        Args:
            joint_name: Name of the joint to query.

        Returns:
            List of local qpos indices. Empty if the joint is not found.
        """
        ...

    @abstractmethod
    def is_fixed_base(self) -> bool:
        """Return True if the articulation has a fixed (non-floating) base."""
        ...

    # ------------------------------------------------------------------
    # Extended interface (RoboTwin->Genesis migration skeleton).
    #
    # The following methods are intentionally *non-abstract*: backends that
    # do not support IK / motion planning / batched domain randomization keep
    # working unchanged and raise NotImplementedError on use. Concrete
    # backends with native support (e.g. Genesis) override them.
    # ------------------------------------------------------------------

    def set_dofs_gains(
        self,
        kp: np.ndarray,
        kv: np.ndarray | None = None,
        force_range: tuple[np.ndarray, np.ndarray] | None = None,
        armature: np.ndarray | None = None,
        dofs_idx: list[int] | None = None,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Set per-DoF PD gains for the internal position controller.

        Maps the embodiment ``config.yml`` stiffness/damping fields onto the
        backend's PD controller (RoboTwin migration section 2).

        Args:
            kp: Stiffness per DoF, shape ``(n_dofs,)`` or ``(n_envs, n_dofs)``.
            kv: Optional damping per DoF, same shape as ``kp``.
            force_range: Optional ``(lower, upper)`` force limits per DoF.
            armature: Optional per-DoF armature (rotor inertia).
            dofs_idx: Optional subset of local DoF indices to configure.
            envs_idx: Optional subset of parallel envs to configure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_dofs_gains"
        )

    def inverse_kinematics(
        self,
        link_name: str,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
        dofs_idx: list[int] | None = None,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Solve IK for a single end-effector link.

        Args:
            link_name: Name of the target link.
            pos: Target world-space position, shape ``(3,)``.
            quat: Optional target orientation quaternion ``(w, x, y, z)``.
            dofs_idx: Optional subset of DoFs allowed to move.
            envs_idx: Optional subset of parallel envs.

        Returns:
            Joint positions achieving the target pose, shape ``(n_dofs,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support inverse_kinematics"
        )

    def inverse_kinematics_multilink(
        self,
        link_names: list[str],
        poss: list[np.ndarray] | np.ndarray,
        quats: list[np.ndarray] | np.ndarray | None = None,
        rot_mask: tuple[bool, bool, bool] | None = None,
        pos_mask: tuple[bool, bool, bool] | None = None,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Solve IK for multiple end-effector links simultaneously.

        Used for dual-arm embodiments (e.g. aloha-agilex loaded as a single
        URDF containing both arms).

        Returns:
            Joint positions achieving the target poses, shape ``(n_dofs,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support inverse_kinematics_multilink"
        )

    def plan_path(
        self,
        qpos_goal: np.ndarray,
        qpos_start: np.ndarray | None = None,
        num_waypoints: int = 50,
        envs_idx: list[int] | None = None,
    ) -> np.ndarray:
        """Plan a collision-free joint-space path to ``qpos_goal``.

        Replaces mplib RRT in the RoboTwin pipeline with the backend's
        native planner (OMPL on Genesis).

        Returns:
            Waypoint trajectory of shape ``(T, n_dofs)`` including the goal.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support plan_path")

    def get_link_pose(self, link_name: str) -> Pose:
        """Return the world-space pose of a link (e.g. an end-effector)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_link_pose"
        )

    def set_friction_ratio(
        self,
        ratios: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env friction domain randomization, shape ``(n_envs, n_links)``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_friction_ratio"
        )

    def set_mass_shift(
        self,
        shifts: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env link-mass domain randomization, shape ``(n_envs, n_links)``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_mass_shift"
        )

    def set_com_shift(
        self,
        shifts: np.ndarray,
        envs_idx: list[int] | None = None,
    ) -> None:
        """Per-env center-of-mass domain randomization, shape ``(n_envs, n_links, 3)``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_com_shift"
        )


class DeformableEntityBackend(EntityBackend):
    """Backend-agnostic interface for deformable entities.

    Deformable entities include soft bodies (FEM/PBD elastic solids), cloth,
    liquids (SPH/PBD), and other continuum materials. They expose particle or
    vertex state in addition to the coarse rigid-body pose inherited from
    EntityBackend.
    """

    @abstractmethod
    def get_particle_positions(self) -> np.ndarray:
        """Return particle/vertex positions as an (N, 3) float64 array."""
        ...

    @abstractmethod
    def get_particle_velocities(self) -> np.ndarray | None:
        """Return particle/vertex velocities as an (N, 3) array, if available."""
        ...

    @abstractmethod
    def set_vertex_constraints(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
        *,
        soft: bool = False,
        stiffness: float = 1.0e4,
    ) -> None:
        """Pin or drag a subset of vertices.

        Args:
            indices: Vertex indices to constrain.
            positions: Target positions for the constrained vertices.
            soft: If True, use a soft (spring-like) constraint.
            stiffness: Constraint stiffness for soft constraints.
        """
        ...

    @abstractmethod
    def update_constraint_targets(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """Update target positions for previously constrained vertices."""
        ...

    @abstractmethod
    def get_deformable_state(self) -> DeformableState:
        """Return a backend-agnostic deformable state snapshot."""
        ...

    @abstractmethod
    def set_deformable_state(self, state: DeformableState) -> None:
        """Restore a deformable state snapshot."""
        ...


class CameraBackend(ABC):
    """Backend-agnostic camera sensor interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def render(
        self,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]: ...

    def get_camera_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(intrinsic, extrinsic)`` for this camera.

        The intrinsic matrix is the standard 3x3 pinhole model. The extrinsic
        is a 4x4 camera-to-world transform (RoboTwin migration section 7:
        consistent conventions are critical for cross-simulator data reuse).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_camera_params"
        )


class RendererBackend(ABC):
    """Backend-agnostic rendering interface."""

    @abstractmethod
    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float],
        resolution: tuple[int, int],
        fov: float = 60.0,
    ) -> CameraBackend:
        """Add a camera to the renderer."""
        ...

    @abstractmethod
    def render(
        self,
        camera_name: str | None = None,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> RenderOutput:
        """Render from the specified camera (or default camera)."""
        ...

    @abstractmethod
    def render_async(
        self,
        camera_names: list[str],
        *,
        rgb: bool = True,
        depth: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously submit render jobs for the listed cameras.

        Returns a mapping from camera name to a future-like handle.
        Concrete backends may return NumPy arrays directly if async
        execution is not supported.
        """
        ...


class SceneBackend(ABC):
    """Backend-agnostic scene/simulation world interface."""

    @property
    @abstractmethod
    def backend(self) -> "SimulatorBackend":
        """Return the simulator backend that created this scene."""
        ...

    @abstractmethod
    def add_entity(self, entity: EntityBackend) -> None:
        """Add a generic entity to the scene."""
        ...

    @abstractmethod
    def add_articulation(self, articulation: ArticulationBackend) -> None:
        """Add an articulated robot to the scene."""
        ...

    @abstractmethod
    def add_light(self, light: LightDescription) -> None:
        """Add a light source to the scene."""
        ...

    @abstractmethod
    def build(
        self, n_envs: int = 1, env_spacing: tuple[float, float] | None = None
    ) -> None:
        """Finalize scene construction and compile the physics model.

        Args:
            n_envs: Number of parallel environments (batched rollout /
                parallel seed search). Backends without batching support may
                ignore values > 1.
            env_spacing: Optional spacing between parallel env copies.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one step."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to the initial state."""
        ...

    @abstractmethod
    def get_physics_state(self) -> PhysicsState:
        """Return a snapshot of the full physics state."""
        ...

    @abstractmethod
    def set_physics_state(self, state: PhysicsState) -> None:
        """Restore the physics state from a snapshot."""
        ...

    @property
    @abstractmethod
    def renderer(self) -> RendererBackend | None:
        """Return the renderer attached to this scene, if any."""
        ...


class SimulatorBackend(ABC):
    """Backend-agnostic physics engine factory.

    A SimulatorBackend is responsible for engine initialization and for
    creating scene/entity/articulation primitives. It is intentionally
    stateless with respect to the simulation world; the world is owned by
    SceneBackend instances.
    """

    @property
    @abstractmethod
    def name(self) -> BackendName:
        """Return the backend identifier."""
        ...

    @abstractmethod
    def initialize(
        self,
        *,
        headless: bool = True,
        device: str = "musa",
        **kwargs: Any,
    ) -> None:
        """Initialize the underlying physics engine and compute runtime."""
        ...

    @abstractmethod
    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
        fem_options: Any | None = None,
        pbd_options: Any | None = None,
        sph_options: Any | None = None,
        mpm_options: Any | None = None,
        sf_options: Any | None = None,
        renderer: Any | None = None,
    ) -> SceneBackend:
        """Create a new simulation scene.

        The deformable-solver options (fem_options, pbd_options, etc.) are
        backend-specific and may be ignored by backends that do not support
        soft bodies or fluids.

        Args:
            dt: Simulation timestep in seconds.
            substeps: Physics substeps per step.
            headless: Run without the interactive viewer.
            viewer_options: Optional viewer camera configuration (used only
                when ``headless`` is False).
            fem_options: Backend-specific FEM solver options.
            pbd_options: Backend-specific PBD solver options.
            sph_options: Backend-specific SPH solver options.
            mpm_options: Backend-specific MPM solver options.
            sf_options: Backend-specific SPH-fluid options.
            renderer: Optional backend-specific renderer selection (e.g. a
                Genesis ``gs.renderers.Rasterizer/RayTracer/BatchRenderer``
                instance). ``None`` uses the backend default (rasterizer).
        """
        ...

    @abstractmethod
    def create_box(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def create_sphere(
        self,
        radius: float,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def create_cylinder(
        self,
        radius: float,
        height: float,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        density: float | None = None,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def create_mesh(
        self,
        file: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        static: bool = True,
        friction: float = 0.5,
        name: str | None = None,
    ) -> EntityBackend: ...

    @abstractmethod
    def create_deformable(
        self,
        config: DeformableConfig,
        shape: str,
        *,
        size: tuple[float, float, float] | None = None,
        radius: float | None = None,
        file: str | None = None,
        scale: tuple[float, float, float] | None = None,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] | None = None,
        color: tuple[float, float, float, float] | None = None,
        name: str | None = None,
    ) -> DeformableEntityBackend:
        """Create a deformable entity (soft body, cloth, liquid, etc.).

        Args:
            config: Material and multiscale configuration.
            shape: Underlying shape primitive ('box', 'sphere', 'mesh').
            size: Box dimensions for shape='box'.
            radius: Sphere radius for shape='sphere'.
            file: Mesh file path for shape='mesh'.
            scale: Mesh scale for shape='mesh'.
            pos: Initial position.
            quat: Initial orientation quaternion.
            color: RGBA color tuple.
            name: Optional entity name.
        """
        ...

    @abstractmethod
    def load_mjcf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        """Load an MJCF model as an articulation."""
        ...

    @abstractmethod
    def load_urdf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        """Load a URDF model as an articulation."""
        ...

    @abstractmethod
    def create_light(self, description: LightDescription) -> LightDescription:
        """Validate and optionally preprocess a light description."""
        ...
