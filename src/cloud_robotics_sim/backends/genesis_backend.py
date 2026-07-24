"""Genesis (Taichi/CUDA) backend implementation.

This backend preserves the existing Genesis-based behavior by adapting the
new Backend ABC to the legacy ``gs.*`` API surface.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cloud_robotics_sim.backend.base import (
    ArticulationBackend,
    CameraBackend,
    EntityBackend,
    RendererBackend,
    SceneBackend,
    SimulatorBackend,
)
from cloud_robotics_sim.backend.types import (
    ArticulationState,
    BackendName,
    LightDescription,
    PhysicsState,
    Pose,
    RenderOutput,
    ViewerOptions,
)
from cloud_robotics_sim.utils.genesis_compat import ensure_genesis_initialized

logger = logging.getLogger(__name__)

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None


def _require_genesis() -> None:
    if not HAS_GENESIS or gs is None:
        raise RuntimeError(
            "Genesis backend requested but 'genesis-world' is not installed."
        )


class GenesisEntityBackend(EntityBackend):
    """Genesis implementation of a generic rigid body entity."""

    def __init__(
        self,
        morph: Any,
        surface: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        _require_genesis()
        self._morph = morph
        self._surface = surface
        self._name = name
        self._entity: Any = None

    @property
    def name(self) -> str | None:
        return self._name

    def _resolve_entity(self) -> Any:
        if self._entity is None:
            raise RuntimeError("Entity has not been added to a Genesis scene yet.")
        return self._entity

    def bind(self, entity: Any) -> None:
        """Called by GenesisSceneBackend after add_entity."""
        self._entity = entity

    def get_pos(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_pos(), dtype=np.float64)

    def set_pos(self, pos: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.set_pos(np.asarray(pos, dtype=np.float64))

    def get_quat(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_quat(), dtype=np.float64)

    def set_quat(self, quat: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.set_quat(np.asarray(quat, dtype=np.float64))

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "set_color"):
            entity.set_color(color)
        else:
            logger.debug("Genesis entity does not support set_color")

    def apply_force(
        self,
        force: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> None:
        entity = self._resolve_entity()
        if hasattr(entity, "apply_force"):
            entity.apply_force(
                np.asarray(force), pos=np.asarray(pos) if pos is not None else None
            )
        else:
            logger.debug("Genesis entity does not support apply_force")


class GenesisArticulationBackend(GenesisEntityBackend, ArticulationBackend):
    """Genesis implementation of an articulated robot."""

    @property
    def n_dofs(self) -> int:
        entity = self._resolve_entity()
        return int(getattr(entity, "n_dofs", 0) or getattr(entity, "n_qs", 0))

    @property
    def n_qs(self) -> int:
        entity = self._resolve_entity()
        return int(getattr(entity, "n_qs", 0) or getattr(entity, "n_dofs", 0))

    def get_qpos(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_qpos(), dtype=np.float64)

    def set_qpos(self, qpos: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.set_qpos(np.asarray(qpos, dtype=np.float64))

    def get_qvel(self) -> np.ndarray:
        entity = self._resolve_entity()
        return np.asarray(entity.get_qvel(), dtype=np.float64)

    def set_qvel(self, qvel: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.set_qvel(np.asarray(qvel, dtype=np.float64))

    def control_dofs_position(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        entity = self._resolve_entity()
        kwargs: dict[str, Any] = {}
        if stiffness is not None:
            kwargs["stiffness"] = np.asarray(stiffness)
        if damping is not None:
            kwargs["damping"] = np.asarray(damping)
        entity.control_dofs_position(np.asarray(targets), **kwargs)

    def control_dofs_velocity(self, targets: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.control_dofs_velocity(np.asarray(targets))

    def control_dofs_force(self, targets: np.ndarray) -> None:
        entity = self._resolve_entity()
        entity.control_dofs_force(np.asarray(targets))

    def get_state_batch(
        self,
        env_ids: list[int] | None = None,
    ) -> ArticulationState:
        # Genesis single-env path: env_ids is ignored.
        return ArticulationState(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            pos=self.get_pos(),
            quat=self.get_quat(),
        )

    def set_state_batch(
        self,
        state: ArticulationState,
        env_ids: list[int] | None = None,
    ) -> None:
        if state.qpos is not None:
            self.set_qpos(state.qpos)
        if state.qvel is not None:
            self.set_qvel(state.qvel)
        if state.pos is not None:
            self.set_pos(state.pos)
        if state.quat is not None:
            self.set_quat(state.quat)

    def control_position_batch(
        self,
        targets: np.ndarray,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
    ) -> None:
        # Genesis single-env path: squeeze leading batch dim if present.
        targets_arr = np.asarray(targets)
        if targets_arr.ndim == 2 and targets_arr.shape[0] == 1:
            targets_arr = targets_arr[0]
        self.control_dofs_position(targets_arr, stiffness=stiffness, damping=damping)

    def get_end_effector_pose(self) -> Pose:
        entity = self._resolve_entity()
        if hasattr(entity, "get_end_effector_pose"):
            pose = entity.get_end_effector_pose()
            return Pose(pos=pose[:3], quat=pose[3:])
        # Fallback: return base pose.
        return Pose(pos=self.get_pos(), quat=self.get_quat())


class GenesisCameraBackend(CameraBackend):
    """Genesis camera wrapper."""

    def __init__(self, name: str, camera: Any) -> None:
        self._name = name
        self._camera = camera

    @property
    def name(self) -> str:
        return self._name

    def render(
        self,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        result = self._camera.render(rgb=rgb, depth=depth, segmentation=segmentation)
        if isinstance(result, tuple):
            return tuple(np.asarray(r) for r in result)
        return np.asarray(result)


class GenesisRendererBackend(RendererBackend):
    """Genesis renderer adapter.

    Genesis ties rendering closely to the scene/viewer; this adapter provides
    a thin compatibility layer on top of the scene's camera entities.
    """

    def __init__(self, gs_scene: Any) -> None:
        self._gs_scene = gs_scene
        self._cameras: dict[str, GenesisCameraBackend] = {}

    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float],
        resolution: tuple[int, int],
        fov: float = 60.0,
    ) -> CameraBackend:
        cam = self._gs_scene.add_camera(
            pos=pos,
            lookat=lookat,
            res=resolution,
            fov=fov,
            GUI=False,
        )
        backend = GenesisCameraBackend(name, cam)
        self._cameras[name] = backend
        return backend

    def render(
        self,
        camera_name: str | None = None,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> RenderOutput:
        if camera_name is None:
            camera_name = next(iter(self._cameras), None)
        if camera_name is None or camera_name not in self._cameras:
            raise ValueError(f"Camera '{camera_name}' not found")
        rgb_arr, depth_arr, seg_arr = None, None, None
        result = self._cameras[camera_name].render(
            rgb=rgb, depth=depth, segmentation=segmentation
        )
        if isinstance(result, tuple):
            parts = list(result)
            rgb_arr = parts.pop(0) if rgb else None
            depth_arr = parts.pop(0) if depth else None
            seg_arr = parts.pop(0) if segmentation else None
        else:
            rgb_arr = np.asarray(result) if rgb else None
        return RenderOutput(rgb=rgb_arr, depth=depth_arr, segmentation=seg_arr)

    def render_async(
        self,
        camera_names: list[str],
        *,
        rgb: bool = True,
        depth: bool = False,
    ) -> dict[str, Any]:
        # Genesis does not support async rendering; return synchronous results.
        return {
            name: self.render(name, rgb=rgb, depth=depth)
            for name in camera_names
            if name in self._cameras
        }


class GenesisSceneBackend(SceneBackend):
    """Genesis scene/world adapter."""

    def __init__(self, backend: "GenesisBackend", gs_scene: Any) -> None:
        _require_genesis()
        self._backend = backend
        self._gs_scene = gs_scene
        self._renderer = GenesisRendererBackend(gs_scene)
        self._built = False

    @property
    def backend(self) -> "GenesisBackend":
        return self._backend

    @property
    def renderer(self) -> RendererBackend | None:
        return self._renderer

    def add_entity(self, entity: EntityBackend) -> None:
        if not isinstance(entity, GenesisEntityBackend):
            raise TypeError("Genesis scene only accepts GenesisEntityBackend instances")
        kwargs: dict[str, Any] = {"morph": entity._morph}
        if entity._surface is not None:
            kwargs["surface"] = entity._surface
        gs_entity = self._gs_scene.add_entity(**kwargs)
        entity.bind(gs_entity)

    def add_articulation(self, articulation: ArticulationBackend) -> None:
        # In Genesis, articulations are added the same way as entities.
        self.add_entity(articulation)

    def add_light(self, light: LightDescription) -> None:
        from cloud_robotics_sim.utils.genesis_compat import get_genesis_lights

        lights = get_genesis_lights()
        if lights is None:
            logger.debug("Genesis version does not expose gs.lights; skipping light")
            return
        if light.light_type.name == "AMBIENT":
            self._gs_scene.add_light(
                lights.Ambient(color=light.color, intensity=light.intensity)
            )
        elif light.light_type.name == "DIRECTIONAL":
            direction = light.direction or (0.0, 0.3, -1.0)
            self._gs_scene.add_light(
                lights.Directional(
                    pos=light.pos or (0.0, 0.0, 5.0),
                    direction=direction,
                    color=light.color,
                    intensity=light.intensity,
                    cast_shadow=light.cast_shadow,
                )
            )
        else:
            logger.warning("Unsupported Genesis light type: %s", light.light_type)

    def build(self) -> None:
        self._gs_scene.build()
        self._built = True

    def step(self) -> None:
        self._gs_scene.step()

    def reset(self) -> None:
        # Genesis resets via state restore; core logic handles recomposition.
        logger.debug("GenesisSceneBackend.reset is a no-op; rely on state rebuild")

    def get_physics_state(self) -> PhysicsState:
        return PhysicsState()

    def set_physics_state(self, state: PhysicsState) -> None:
        logger.debug("Genesis physics state restore not yet implemented")


class GenesisBackend(SimulatorBackend):
    """Genesis (Taichi/CUDA) simulator backend."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> BackendName:
        return BackendName.GENESIS

    def initialize(
        self,
        *,
        headless: bool = True,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        _require_genesis()
        use_cuda = device.lower() in {"cuda", "gpu", "auto"}
        ensure_genesis_initialized(headless=headless, use_cuda=use_cuda, **kwargs)
        self._initialized = True

    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
    ) -> SceneBackend:
        _require_genesis()
        gs_viewer_options = None
        if not headless and viewer_options is not None:
            gs_viewer_options = gs.options.ViewerOptions(
                camera_pos=viewer_options.camera_pos,
                camera_lookat=viewer_options.camera_lookat,
                res=viewer_options.resolution,
                max_FPS=viewer_options.max_fps,
            )
        gs_scene = gs.Scene(
            viewer_options=gs_viewer_options,
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            show_viewer=not headless,
        )
        return GenesisSceneBackend(self, gs_scene)

    def _create_morph_kwargs(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None,
        color: tuple[float, float, float, float] | None,
        static: bool,
        friction: float,
        density: float | None,
        name: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"pos": pos, "fixed": static}
        if quat is not None:
            kwargs["quat"] = quat
        if density is not None:
            kwargs["density"] = density
        return kwargs

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
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Box(size=size, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

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
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Sphere(radius=radius, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

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
    ) -> EntityBackend:
        _require_genesis()
        kwargs = self._create_morph_kwargs(
            pos, quat, color, static, friction, density, name
        )
        morph = gs.morphs.Cylinder(radius=radius, height=height, **kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

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
    ) -> EntityBackend:
        _require_genesis()
        kwargs: dict[str, Any] = {"file": file, "pos": pos, "fixed": static}
        if quat is not None:
            kwargs["quat"] = quat
        if scale is not None:
            kwargs["scale"] = scale
        morph = gs.morphs.Mesh(**kwargs)
        surface = gs.surfaces.Default(color=color, roughness=0.8) if color else None
        return GenesisEntityBackend(morph, surface, name=name)

    def load_mjcf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        _require_genesis()
        morph = gs.morphs.MJCF(file=file, pos=pos, **kwargs)
        return GenesisArticulationBackend(morph, name=file)

    def load_urdf(
        self,
        file: str,
        pos: tuple[float, float, float],
        **kwargs: Any,
    ) -> ArticulationBackend:
        _require_genesis()
        morph = gs.morphs.URDF(file=file, pos=pos, **kwargs)
        return GenesisArticulationBackend(morph, name=file)

    def create_light(self, description: LightDescription) -> LightDescription:
        return description
