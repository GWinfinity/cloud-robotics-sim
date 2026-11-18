"""EFE (Embodied First Exam) suite registrations for the runtime registry.

P1 single-arm suites (MetaWorld+ / LIBERO-MAX / VLABench / RoboPaint / KinDER)
run on the generic stack: a neutral base task whose success comes entirely from
the configured :class:`~cloud_robotics_sim.core.success.SuccessCondition`
(EFE's graders are final-state predicates, not reward-shaping tasks), the
suite's robot (Sawyer / Franka Panda / TidyBot++), and a tabletop scene whose
objects the task YAML declares as primitive assets.

Importing this module registers the factories; ``runtime.main`` imports it so
the built-ins exist once the improvement-loop entry point is loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.core.embodiment import EmbodimentConfig, FrankaPanda
from cloud_robotics_sim.core.registry import (
    register_robot,
    register_scene,
    register_task,
)
from cloud_robotics_sim.core.scene import Scene, SceneConfig
from cloud_robotics_sim.core.task import Task, TaskConfig

#: Repo-rooted default model paths (upstream mirrors under assets/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
SAWYER_MJCF = (
    _REPO_ROOT / "assets" / "metaworld" / "envs" / "assets" / "objects" / "xyz_base.xml"
)
TIDYBOT_URDF_DIR = _REPO_ROOT / "assets" / "kindergarden"
VLB_PANDA_MJCF = (
    _REPO_ROOT / "assets" / "vlabench" / "robots" / "franka_emika_panda" / "panda.xml"
)


class EfeBaseTask(Task):
    """Neutral base task for EFE ports: no shaping, success via condition.

    EFE grades the final state with a boolean predicate; there is no reward
    curriculum. ``ConfigurableTask`` (task_loader) wraps this base with the
    YAML-configured success condition, which is where every EFE goal check
    lives. The base task itself never terminates and never rewards.
    """

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        """Reset episode counters; nothing else to initialise."""
        self.step_count = 0
        self.succeeded = False
        return {"task": "efe", "seed": seed}

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        """Advance one step with zero reward and no self-termination."""
        self.step_count += 1
        info: dict = {"task": "efe"}
        return 0.0, False, False, info


class EfeTabletopScene(Scene):
    """Tabletop workspace for EFE single-arm suites.

    A small room whose floor plane stands in for the benchmark's table top
    (MetaWorld: table top at z = 0; VLABench/LIBERO: their own tables arrive
    as static assets in the task YAML). The robot mounts beside the workspace.
    """

    def __init__(self, size: tuple[float, float, float] = (3.0, 3.0, 2.0)):
        config = SceneConfig(
            name="efe_tabletop",
            size=size,
            default_camera_pos=(1.2, -1.2, 1.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """No built-in furniture; task YAMLs declare the workspace assets."""


@register_scene("efe_tabletop")
def _make_efe_tabletop(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    return EfeTabletopScene(**kwargs)


class EfeLiberoMaxTask(EfeBaseTask):
    """LIBERO-MAX base task: applies the frozen mid-episode event once.

    EFE's LIBERO-MAX cases carry one online change with an
    ``on_proximity`` trigger (gripper within ``distance_m`` of an entity,
    checked every control step like LIBERO-MAX's runtime). Physical
    operations (``move_object`` / ``insert_obstacle`` /
    ``insert_distractors`` / ``remove_object``) teleport scene entities via
    the entity backend, mirroring ``libero_backend.LiberoMujocoBackend``
    position semantics (``position_m`` / ``delta_position_m`` /
    ``relative_to``+``offset_m`` / eef-path placement). Observation-space
    operations (``set_visual_theme`` / ``set_lighting`` /
    ``set_sensor_corruption`` / ``shift_camera``) change no physics and are
    recorded in ``info`` only — Genesis configs grade final-state physics.
    """

    _PHYSICAL_OPS = {
        "move_object",
        "insert_obstacle",
        "insert_distractors",
        "remove_object",
    }
    _OBSERVATION_OPS = {
        "set_visual_theme",
        "set_lighting",
        "set_sensor_corruption",
        "shift_camera",
    }
    _EE_LINK_CANDIDATES = (
        "franka_hand",
        "panda_hand",
        "hand",
        "ee_link",
        "link7",
        "link_7",
        "wrist",
    )

    def __init__(self, config: TaskConfig, event: dict | None = None):
        super().__init__(config)
        if event is not None and not isinstance(event, dict):
            raise TypeError("efe_liberomax event must be a mapping or None")
        self._event = event
        self._fired = False
        self._parked: dict[str, np.ndarray] = {}

    # -- helpers -----------------------------------------------------------

    def _entity(self, scene: Any, name: str) -> Any:
        entities = getattr(scene, "entities", None) or {}
        if name not in entities:
            raise KeyError(
                f"efe_liberomax event references unknown entity {name!r}; "
                f"available: {sorted(entities)}"
            )
        return entities[name]

    @staticmethod
    def _entity_pos(entity: Any) -> np.ndarray:
        get_pos = getattr(entity, "get_pos", None)
        if get_pos is None:
            raise TypeError("event entity does not expose get_pos()")
        return np.asarray(get_pos(), dtype=float)

    def _ee_pos(self, robot: Any) -> np.ndarray:
        entity = getattr(robot, "entity", None) or robot
        hook = getattr(entity, "get_end_effector_pose", None)
        if callable(hook):
            pose = hook()
            pos = getattr(pose, "pos", pose)
            if isinstance(pose, tuple):
                pos = pose[0]
            return np.asarray(pos, dtype=float).reshape(3)
        get_link_pos = getattr(entity, "get_link_pos", None)
        if callable(get_link_pos):
            for link in self._EE_LINK_CANDIDATES:
                try:
                    return np.asarray(get_link_pos(link), dtype=float).reshape(3)
                except Exception:  # noqa: BLE001 - candidate links differ per model
                    continue
        raise TypeError(
            "efe_liberomax trigger needs an end-effector pose: robot entity "
            "exposes neither get_end_effector_pose() nor get_link_pos()"
        )

    def _requested_position(
        self, scene: Any, change: dict, entity_name: str, robot: Any
    ) -> np.ndarray:
        if "position_m" in change:
            position: np.ndarray = np.asarray(
                change["position_m"], dtype=float
            ).reshape(3)
            return position
        before = self._entity_pos(self._entity(scene, entity_name))
        if "delta_position_m" in change:
            position = before + np.asarray(change["delta_position_m"], dtype=float)
            return position
        if "relative_to" in change:
            anchor = self._entity_pos(self._entity(scene, change["relative_to"]))
            offset: np.ndarray = np.asarray(
                change.get("offset_m", [0.0, 0.0, 0.0]), dtype=float
            )
            position = anchor + offset
            if change.get("preserve_initial_z", False):
                position[2] = float(self._parked[entity_name][2])
            return position
        if "path_target" in change:
            target = self._entity_pos(self._entity(scene, change["path_target"]))
            eef = self._ee_pos(robot)
            fraction = float(change.get("path_fraction", 0.5))
            lateral = float(change.get("lateral_offset_m", 0.0))
            direction_xy = target[:2] - eef[:2]
            norm = float(np.linalg.norm(direction_xy))
            if norm <= 1e-9:
                raise ValueError("eef and path target share an XY position")
            perpendicular: np.ndarray = (
                np.array([-direction_xy[1], direction_xy[0]]) / norm
            )
            position = eef + fraction * (target - eef)
            position[:2] += lateral * perpendicular
            position[2] = float(self._parked[entity_name][2])
            return position
        raise ValueError(
            f"event change for {entity_name!r} needs position_m, "
            f"delta_position_m, relative_to or path_target"
        )

    def _set_entity_position(self, scene: Any, name: str, position: np.ndarray) -> None:
        entity = self._entity(scene, name)
        set_pos = getattr(entity, "set_pos", None)
        if set_pos is None:
            raise TypeError(f"entity {name!r} does not expose set_pos()")
        set_pos(np.asarray(position, dtype=float).reshape(3))

    def _apply_change(self, scene: Any, change: dict, robot: Any) -> dict:
        operation = change.get("operation")
        if operation in self._OBSERVATION_OPS:
            return {"operation": operation, "simulator_changed": False}
        if operation == "remove_object":
            name = change["object"]
            self._parked[name] = self._entity_pos(self._entity(scene, name))
            offworld = np.asarray(
                change.get("offworld_position_m", [0.0, 0.0, -5.0]), dtype=float
            )
            self._set_entity_position(scene, name, offworld)
            return {"operation": operation, "object": name, "parked": True}
        if operation == "insert_distractors":
            placements = []
            for placement in change.get("placements") or []:
                name = placement["object"]
                position = self._requested_position(scene, placement, name, robot)
                self._set_entity_position(scene, name, position)
                placements.append({"object": name, "position": position.tolist()})
            return {"operation": operation, "placements": placements}
        if operation in {"move_object", "insert_obstacle"}:
            name = change["object"]
            position = self._requested_position(scene, change, name, robot)
            self._set_entity_position(scene, name, position)
            return {
                "operation": operation,
                "object": name,
                "position": position.tolist(),
            }
        raise ValueError(f"unsupported efe_liberomax operation {operation!r}")

    # -- task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._fired = False
        self._parked = {}
        for change in (self._event or {}).get("setup") or []:
            self._apply_change(scene, change, robot)
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        event = self._event or {}
        if self._fired or not event:
            return reward, terminated, truncated, info
        trigger = event.get("trigger") or {}
        if trigger.get("type") != "on_proximity":
            raise ValueError(
                f"efe_liberomax supports on_proximity triggers only, "
                f"got {trigger.get('type')!r}"
            )
        distance = float(trigger.get("distance_m", 0.18))
        ee = self._ee_pos(robot)
        target = self._entity_pos(self._entity(scene, trigger["value"]))
        if float(np.linalg.norm(ee - target)) >= distance:
            return reward, terminated, truncated, info
        self._fired = True
        result = self._apply_change(scene, event["change"], robot)
        info = dict(info)
        info["libero_max_event"] = {
            "step": self.step_count,
            "trigger": trigger,
            "result": result,
        }
        return reward, terminated, truncated, info


@register_task("efe_liberomax")
def _make_efe_liberomax(**kwargs: Any) -> EfeLiberoMaxTask:
    """LIBERO-MAX base task with the frozen online event from the YAML block."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    event = kwargs.pop("event", None)
    return EfeLiberoMaxTask(
        TaskConfig(name="efe", max_episode_steps=max_steps), event=event
    )


@register_task("efe_static")
def _make_efe_static(**kwargs: Any) -> EfeBaseTask:
    """Neutral EFE base task; success comes from the YAML success block."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    return EfeBaseTask(TaskConfig(name="efe", max_episode_steps=max_steps))


@register_robot("sawyer")
def _make_sawyer(**kwargs: Any) -> Any:
    """Sawyer arm from MetaWorld's MJCF (MIT), via the MJCF morph path."""
    kwargs.pop("type", None)
    base_position = kwargs.pop("base_position", (0.0, -0.3, 0.0))
    model_path = kwargs.pop("urdf_path", None) or kwargs.pop("model_path", None)
    if model_path is None:
        model_path = str(SAWYER_MJCF)
    elif model_path.startswith("mjcf:"):
        # resolve_robot_model infers the format from the file suffix; a
        # "mjcf:" scheme prefix would be treated as part of the path.
        model_path = model_path.removeprefix("mjcf:")
    return FrankaPanda(
        EmbodimentConfig(
            name="sawyer",
            base_position=base_position,
            urdf_path=model_path,
        )
    )


@register_robot("franka_panda")
def _make_franka_panda(**kwargs: Any) -> Any:
    """Franka Panda from VLABench's MJCF (MIT), the EFE single-arm robot."""
    kwargs.pop("type", None)
    base_position = kwargs.pop("base_position", (0.0, -0.4, 0.78))
    model_path = kwargs.pop("urdf_path", None) or kwargs.pop("model_path", None)
    if model_path is None:
        model_path = str(VLB_PANDA_MJCF)
    elif model_path.startswith("mjcf:"):
        model_path = model_path.removeprefix("mjcf:")
    return FrankaPanda(
        EmbodimentConfig(
            name="franka_panda",
            base_position=base_position,
            urdf_path=model_path,
        )
    )
