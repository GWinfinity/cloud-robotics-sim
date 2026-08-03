"""Replay scene for RoboTwin demonstrations."""

from __future__ import annotations

import logging
from typing import Any

from cloud_robotics_sim.backend import SceneBackend
from cloud_robotics_sim.core.scene import ObjectSpawn, Scene, SceneConfig
from cloud_robotics_sim.robotwin.bridge import RobotwinBridge
from cloud_robotics_sim.utils.genesis_compat import is_genesis_scene

logger = logging.getLogger(__name__)


class RobotwinReplayScene(Scene):
    """A scene reconstructed from a RoboTwin bridge episode.

    This scene loads the static room structure, a tabletop, and all objects
    referenced by the bridge. Object poses are later overwritten by the
    replay task on a per-frame basis (kinematic replay).
    """

    def __init__(
        self,
        bridge: RobotwinBridge,
        config: SceneConfig | None = None,
    ) -> None:
        super().__init__(config or SceneConfig(name=f"robotwin_{bridge.task_name}"))
        self.bridge = bridge
        self.object_assets = bridge.object_assets
        self.table_height = bridge.table_height
        self._articulations: dict[str, Any] = {}
        self._register_object_spawns()

    def _register_object_spawns(self) -> None:
        """Register ObjectSpawn entries for the table and mesh objects."""
        self.add_object(
            ObjectSpawn(
                name="table",
                shape_type="box",
                size=(1.2, 0.7, 0.05),
                position=(0.0, 0.0, self.table_height),
                static=True,
                color=(0.6, 0.5, 0.4, 1.0),
                tags=["furniture", "table"],
            )
        )

        for obj_name, asset in self.object_assets.items():
            if asset.asset_type == "mesh":
                self.add_object(
                    ObjectSpawn(
                        name=obj_name,
                        shape_type="mesh",
                        mesh_path=asset.path,
                        scale=asset.scale,
                        position=(0.0, 0.0, self.table_height + 0.05),
                        static=False,
                        color=(0.9, 0.9, 0.9, 1.0),
                        tags=["object"],
                    )
                )

    def _build_custom(self) -> None:
        """Load articulated objects directly via the backend."""
        if is_genesis_scene(self.scene):
            raise RuntimeError(
                "RobotwinReplayScene must be built through the backend abstraction"
            )

        scene_backend = self.scene
        if not isinstance(scene_backend, SceneBackend):
            raise RuntimeError("Invalid scene backend")

        backend = scene_backend.backend

        # Load articulated objects directly through the backend.
        for obj_name, asset in self.object_assets.items():
            if asset.asset_type != "urdf":
                continue
            try:
                articulation = backend.load_urdf(
                    file=asset.path,
                    pos=(0.0, 0.0, self.table_height + 0.05),
                    fixed=False,
                    scale=_normalize_scale(asset.scale),
                )
                scene_backend.add_articulation(articulation)
                self._articulations[obj_name] = articulation
                self.entities[obj_name] = articulation
            except Exception as e:
                logger.warning(f"Failed to load URDF object '{obj_name}': {e}")

    def get_articulation(self, name: str) -> Any:
        """Return an articulated object by name, if any."""
        return self._articulations.get(name)


def _normalize_scale(scale: tuple[float, float, float] | float) -> tuple[float, float, float]:
    """Normalize a uniform or per-axis scale to a 3-tuple."""
    if isinstance(scale, (int, float)):
        return (float(scale), float(scale), float(scale))
    values = [float(v) for v in scale]
    while len(values) < 3:
        values.append(values[-1] if values else 1.0)
    return (values[0], values[1], values[2])
