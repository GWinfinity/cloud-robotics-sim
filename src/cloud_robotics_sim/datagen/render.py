"""Genesis-based episode renderer for synthetic data generation.

This is the genesis-world replacement for the reference pipeline's
Isaac-Sim render step: it stages one :class:`~.specs.EpisodeSpec` per scene,
steps the physics, captures every declared camera, and returns stacked
uint8 frames plus JSON-safe metadata. Genesis is imported lazily so the
module stays importable (and testable) without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary
from cloud_robotics_sim.utils.genesis_compat import genesis_init

from .specs import DrSample, EpisodeSpec

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Frames and metadata produced for one episode."""

    episode_id: str
    frames: dict[str, np.ndarray]  # camera name -> (T, H, W, 3) uint8
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        """Number of captured frames (all cameras share the timeline)."""
        if not self.frames:
            return 0
        return int(next(iter(self.frames.values())).shape[0])


def _host_xyz(value: Any) -> np.ndarray:
    """Normalize an entity position getter result to a (3,) host array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"expected at least 3 position components, got {arr}")
    return arr[:3]


class GenesisEpisodeRenderer:
    """Render episodes from object placements with per-episode DR.

    Args:
        objects_dir: RoboTwin-OD object library root used to resolve
            placement class names. May be ``None`` when every episode
            places nothing.
        backend: ``"cpu"`` or ``"cuda"``; forwarded to
            :func:`~cloud_robotics_sim.utils.genesis_compat.genesis_init`.
        dt: physics timestep in seconds.
        substeps: physics substeps per step.
        add_lights: attempt per-episode randomized point lights. Only the
            BatchRenderer supports scene lights in genesis-world 1.4.x;
            other renderers skip this silently.
    """

    def __init__(
        self,
        objects_dir: str | Path | None = None,
        *,
        backend: str = "cuda",
        dt: float = 0.004,
        substeps: int = 10,
        add_lights: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self._objects_dir = Path(objects_dir) if objects_dir is not None else None
        self._backend = backend
        self._dt = dt
        self._substeps = substeps
        self._add_lights = add_lights
        self._library: Any | None = None

    def _get_library(self) -> Any | None:
        """Lazily build the object library once per renderer (batch reuse)."""
        if self._objects_dir is None:
            return None
        if self._library is None:
            self._library = RoboTwinObjectLibrary(self._objects_dir)
        return self._library

    def render_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        """Build one scene, run ``spec.steps`` steps, capture all cameras."""
        import genesis as gs

        dr = spec.randomization.sample()
        genesis_init(
            use_cuda=self._backend != "cpu",
            logging_level="warning",
        )
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self._dt, substeps=self._substeps),
            vis_options=gs.options.VisOptions(background_color=dr.background),
            show_viewer=False,
        )
        entities: dict[str, Any] = {}
        try:
            scene.add_entity(
                gs.morphs.Plane(),
                surface=gs.surfaces.Default(color=(*dr.floor, 1.0)),
            )
            library = self._get_library()
            if library is not None and spec.objects:
                for placement in spec.objects:
                    resolved = self._resolve_class(library, placement.class_name)
                    entities[placement.class_name] = library.spawn_in_scene(
                        scene,
                        resolved,
                        index=placement.instance,
                        pos=placement.pos,
                        quat=placement.quat,
                    )
            if self._add_lights:
                self._add_randomized_lights(scene, dr)
            cameras = self._add_cameras(scene, spec)

            scene.build()

            buffers: dict[str, list[np.ndarray]] = {
                camera_spec.name: [] for camera_spec in spec.cameras
            }
            for _ in range(spec.steps):
                scene.step()
                for camera_spec, camera in zip(spec.cameras, cameras, strict=True):
                    if camera_spec.follow is not None:
                        self._follow(camera_spec, camera, entities)
                    frame = camera.render()
                    if isinstance(frame, tuple):
                        frame = frame[0]
                    buffers[camera_spec.name].append(np.asarray(frame, dtype=np.uint8))
        finally:
            scene.destroy()

        frames = {
            name: (
                np.stack(buffer, axis=0)
                if buffer
                else np.zeros((0, *self._frame_shape(spec, name)), dtype=np.uint8)
            )
            for name, buffer in buffers.items()
        }
        return EpisodeResult(
            episode_id=spec.episode_id,
            frames=frames,
            metadata={
                "episode_id": spec.episode_id,
                "steps": spec.steps,
                "objects": [
                    {
                        "class_name": p.class_name,
                        "instance": p.instance,
                        "pos": list(p.pos),
                        "quat": list(p.quat) if p.quat is not None else None,
                    }
                    for p in spec.objects
                ],
                "cameras": [
                    {
                        "name": camera.name,
                        "model": camera.model,
                        "res": list(camera.res),
                        "fov": camera.fov,
                        "follow": camera.follow,
                    }
                    for camera in spec.cameras
                ],
                "domain_randomization": dr.to_metadata(),
            },
        )

    @staticmethod
    def _resolve_class(library: Any, class_name: str) -> str:
        """Resolve a bare or prefixed class name to the library's class dir.

        The object library stores classes as ``NNN_name``; specs use the bare
        ``name`` (recipe item ids), so match by unique ``_name`` suffix.
        """
        classes: list[str] = list(library.list_classes())
        if class_name in classes:
            return class_name
        matches = [c for c in classes if c.endswith(f"_{class_name}")]
        if len(matches) != 1:
            raise ValueError(
                f"class {class_name!r} resolves to {matches} "
                f"in library with classes {classes}"
            )
        return matches[0]

    def _frame_shape(self, spec: EpisodeSpec, camera_name: str) -> tuple[int, int, int]:
        for camera in spec.cameras:
            if camera.name == camera_name:
                width, height = camera.res
                return (height, width, 3)
        raise KeyError(camera_name)  # pragma: no cover - guarded by spec

    def _add_cameras(self, scene: Any, spec: EpisodeSpec) -> list[Any]:
        cameras = []
        for camera in spec.cameras:
            cameras.append(
                scene.add_camera(
                    model=camera.model,
                    res=camera.res,
                    pos=camera.pos,
                    lookat=camera.lookat,
                    fov=camera.fov,
                    GUI=False,
                )
            )
        return cameras

    def _follow(
        self,
        camera_spec: Any,
        camera: Any,
        entities: dict[str, Any],
    ) -> None:
        """Re-pose a follow camera from its target's current base position."""
        target = _host_xyz(entities[camera_spec.follow].get_pos())
        pos = tuple(target + np.asarray(camera_spec.pos, dtype=np.float64))
        lookat = tuple(target + np.asarray(camera_spec.lookat, dtype=np.float64))
        camera.set_pose(pos=pos, lookat=lookat)

    def _add_randomized_lights(self, scene: Any, dr: DrSample) -> None:
        """Add sampled point lights; unsupported renderers degrade silently."""
        for index, light in enumerate(dr.lights):
            angle = 2.0 * np.pi * index / max(len(dr.lights), 1)
            pos = (1.5 * np.cos(angle), 1.5 * np.sin(angle), 2.5)
            try:
                scene.add_light(
                    pos=pos,
                    dir=(0.0, 0.0, -1.0),
                    color=light.color,
                    intensity=light.intensity,
                )
            except Exception as exc:  # noqa: BLE001 - renderer capability probe
                logger.debug("light DR skipped: %s", exc)
                return


__all__ = ["EpisodeResult", "GenesisEpisodeRenderer"]
