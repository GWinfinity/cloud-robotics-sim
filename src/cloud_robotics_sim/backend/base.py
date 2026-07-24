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
    def build(self) -> None:
        """Finalize scene construction and compile the physics model."""
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
    ) -> SceneBackend:
        """Create a new simulation scene."""
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
