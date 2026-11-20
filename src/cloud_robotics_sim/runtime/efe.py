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

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.core.embodiment import (
    EmbodimentConfig,
    FrankaPanda,
    RobotEmbodiment,
)
from cloud_robotics_sim.core.registry import (
    register_robot,
    register_scene,
    register_task,
)
from cloud_robotics_sim.core.scene import Scene, SceneConfig
from cloud_robotics_sim.core.task import Task, TaskConfig

logger = logging.getLogger(__name__)

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


class EfeWorldScene(Scene):
    """Bare container for whole-world EFE suites (RoboCasa).

    The frozen kitchen world asset carries its own room shell (walls,
    floor), robot and objects, so this scene spawns no structure at all —
    unlike ``efe_tabletop`` whose floor plane would double the kitchen.
    """

    def __init__(self, size: tuple[float, float, float] = (12.0, 12.0, 4.0)):
        config = SceneConfig(
            name="efe_world",
            size=size,
            default_camera_pos=(2.5, -2.5, 2.0),
        )
        super().__init__(config)

    def _build_room_structure(self) -> None:
        """No shell: the world MJCF provides walls, floor and lighting."""

    def _build_custom(self) -> None:
        """Nothing beyond the task-declared world asset."""


@register_scene("efe_world")
def _make_efe_world(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    return EfeWorldScene(**kwargs)


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


class _WorldEmbeddedRobot(RobotEmbodiment):
    """Placeholder embodiment for robots that live inside a whole-world MJCF.

    The RoboCasa suites spawn the robot (Franka on an Omron base / Fourier
    GR1) as part of the frozen kitchen world asset; the composer still wants
    a robot component, so this embodiment spawns nothing and exposes no
    controls. Success predicates reach bodies/sites through the world
    entity registered by the asset spawn.
    """

    def __init__(self, name: str) -> None:
        super().__init__(EmbodimentConfig(name=name, base_position=(0.0, 0.0, 0.0)))
        self._obs_dim = 0
        self._action_dim = 0

    def spawn(self, scene: Any, position: tuple | None = None) -> RobotEmbodiment:
        """No-op: the robot is already inside the spawned world."""
        self.scene = scene
        logger.info(f"{self.config.name} embedded in the world asset; no spawn")
        return self

    def reset(self) -> None:
        """No-op: the frozen world state is applied by the task."""

    def apply_action(self, action: np.ndarray) -> None:
        """No-op: EFE grades final states, not controls."""

    def get_observation(self) -> dict:
        """Empty observation; the world task carries its own state."""
        return {}


@register_robot("franka_panda_omron")
def _make_franka_panda_omron(**kwargs: Any) -> Any:
    """Franka Panda on an Omron mobile base (RoboCasa, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("franka_panda_omron")


@register_robot("fourier_gr1")
def _make_fourier_gr1(**kwargs: Any) -> Any:
    """Fourier GR1 humanoid (RoboCasa-GR1, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("fourier_gr1")


@register_robot("kuka_iiwa_sharpa")
def _make_kuka_iiwa_sharpa(**kwargs: Any) -> Any:
    """KUKA iiwa 14 with a Sharpa HA4 hand (DexToolBench, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("kuka_iiwa_sharpa")


class EfeDtbTask(EfeBaseTask):
    """DexToolBench frozen-scene task (KUKA iiwa 14 + Sharpa HA4 + tool).

    The task YAML carries the composed world (``world`` block, see
    ``tools/dtb_worlds.py``) and a ``dtb`` block with the ordered goal
    trajectory and the grasp-box constants each task's ``instruction.md``
    states verbatim. Each step the task transforms the tool's 8 grasp-box
    corners by its live pose and by the current goal pose; the goal counts
    when every corner pair is within the tolerance (1.5 cm), the latch
    advances to the next goal, and ``info['dtb_goals_reached']`` is
    published — the outer :class:`DtbAllGoalsCondition` ends the episode
    the moment every goal is reached (the EFE suite's rule; no hold).
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        dtb: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_dtb world must be a mapping or None")
        if dtb is not None and not isinstance(dtb, dict):
            raise TypeError("efe_dtb dtb must be a mapping or None")
        self._world_cfg = world or {}
        self._dtb = dtb or {}
        self._goals_reached = 0
        self._servo_warned = False

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_dtb expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    # -- grasp-box corner test (instruction.md verbatim) --------------------

    def _corners(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """8 grasp-box corners (world frame): pose frame +- dims/2, all signs."""
        dims = np.asarray(self._dtb["grasp_box_dims_m"], dtype=float) * float(
            self._dtb.get("grasp_box_scale", 1.0)
        )
        half = dims / 2.0
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        local = signs * half[None, :]
        from cloud_robotics_sim.core.success import _quat_to_mat

        corners: np.ndarray = np.asarray(pos)[None, :] + local @ _quat_to_mat(quat).T
        return corners

    def _goal_reached(self, scene: Any) -> bool:
        from cloud_robotics_sim.core.success import _abc_pos, _abc_quat

        goals = self._dtb.get("goals") or []
        if self._goals_reached >= len(goals):
            return False
        goal = goals[self._goals_reached]
        tol = float(self._dtb.get("corner_tolerance_m", 0.015))
        object_pos = _abc_pos(scene, self._dtb.get("object_body", "object"))
        object_quat = _abc_quat(scene, self._dtb.get("object_body", "object"))
        goal_pos = np.asarray(goal[:3], dtype=float)
        goal_quat = np.asarray(goal[3:7], dtype=float)  # stored wxyz by resolver
        corners_tool = self._corners(object_pos, object_quat)
        corners_goal = self._corners(goal_pos, goal_quat)
        return bool(np.all(np.linalg.norm(corners_tool - corners_goal, axis=1) <= tol))

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._apply_frozen_state(scene)
        self._goals_reached = 0
        info["dtb_goals_reached"] = 0
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        while self._goals_reached < len(
            self._dtb.get("goals") or []
        ) and self._goal_reached(scene):
            self._goals_reached += 1
        info["dtb_goals_reached"] = self._goals_reached
        return reward, terminated, truncated, info


@register_robot("franka_allegro")
def _make_franka_allegro(**kwargs: Any) -> Any:
    """Franka Panda + 16-DoF Allegro hand (DexJoCo single arm, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("franka_allegro")


@register_robot("franka_allegro_bimanual")
def _make_franka_allegro_bimanual(**kwargs: Any) -> Any:
    """Twin Franka Panda + Allegro hands (DexJoCo bimanual, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("franka_allegro_bimanual")


class EfeDjxTask(EfeBaseTask):
    """DexJoCo frozen-scene task (Franka Panda + Allegro hand, 1-2 arms).

    The world block carries the replayed instance (env XML + model-array
    overlays + frozen qpos/qvel); the ``djx`` block carries the family name
    and per-family constants. Each step the task evaluates the family's
    ``_compute_success`` port (verbatim from the upstream env class) and
    publishes ``info['djx_success']``; the episode ends at the first success
    or on the family's failure latch, exactly like DexJoCo's own loop.

    Ported families (pose/joint/contact readable): ``water_plant``
    (ref_point site in the plant cylinder + spray-trigger joint hysteresis,
    30-step hold, trigger-outside fails), ``pick_bucket`` (boxed food inside
    the bucket-ref site AABB + bucket bottom lifted 15 cm),
    ``pinch_tongs`` (tongs-joint pinch counting + 10 cm lift, 30-step
    hold), ``fold_glasses`` (both glass joints > 1.1 and the glass inside
    the open-box cavity, 50-step hold), ``microwave_cook`` (door joint
    closed, hot dog inside the interior-site box, start-button contact),
    ``bimanual_assembly`` (peg-to-socket-bottom contact, 30-step hold) and
    ``click_mouse`` (mouse on the pad + joint-0 click past the frozen
    initial value, display latches blue, 10-step hold). Families needing
    engine quantities this runtime does not expose (pre-impact velocities,
    static display-geom poses, tower reconstruction, digit-sequence
    material state) stay ``pending`` in the compile KB with the reason
    recorded.
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        djx: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_djx world must be a mapping or None")
        if djx is not None and not isinstance(djx, dict):
            raise TypeError("efe_djx djx must be a mapping or None")
        self._world_cfg = world or {}
        self._djx = djx or {}
        self._meta_cache: dict | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        """Per-episode latches (the env classes' Python-side state)."""
        self._success_count = 0
        self._failure = False
        self._trigger_pulled = False
        self._pinch_count = 0
        self._pinch_in_progress = False
        self._display_blue = False
        self._nail_depth = 0.0
        self._prev_face_z: float | None = None
        self._vz_buf: list[float] = []
        self._unlock_index = 0
        self._unlock_failed = False
        self._screen_unlocked = False
        self._pressed_last: set[str] = set()
        self._illegal_trigger_reason = ""
        self._illegal_trigger_count = 0
        family = self._djx.get("family", "")
        target = {
            "water_plant": 30,
            "click_mouse": 10,
            "fold_glasses": 50,
            "pinch_tongs": 30,
            "bimanual_assembly": 30,
            "bimanual_hanoi": 20,
            "bimanual_unlock_ipad": 10,
        }.get(family, 1)
        self._success_target = int(self._djx.get("success_target") or target)

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_djx expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    # -- accessors ---------------------------------------------------------

    def _pos(self, scene: Any, body: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_pos

        return _abc_pos(scene, body)

    def _quat(self, scene: Any, body: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_quat

        return _abc_quat(scene, body)

    def _site_pos(self, scene: Any, site: str) -> np.ndarray:
        """Current world position of a site from the extraction meta."""
        meta = self._meta(scene)
        record = (meta.get("sites") or {})[site]
        body_pos = self._pos(scene, record["body"])
        body_quat = self._quat(scene, record["body"])
        from cloud_robotics_sim.core.success import _quat_to_mat

        local = np.asarray(record["local"], dtype=float)
        world_point: np.ndarray = body_pos + _quat_to_mat(body_quat) @ local
        return world_point

    def _qpos(self, scene: Any, joint: str) -> float:
        world = self._world(scene)
        for record in self._meta(scene).get("joints") or []:
            if record.get("name") == joint:
                qpos = np.asarray(world.get_qpos(), dtype=float)
                return float(qpos[int(record["qposadr"])])
        raise KeyError(joint)

    def _touch(self, scene: Any, a: str, b: str) -> bool:
        from cloud_robotics_sim.core.success import _world_bodies_touch

        return bool(_world_bodies_touch(self._world(scene), a, b))

    def _geom_pose(self, scene: Any, geom: str) -> tuple[np.ndarray, np.ndarray]:
        """World (pos, wxyz quat) of one geom via meta local pose + body pose."""
        record = (self._meta(scene).get("geoms") or {})[geom]
        body_pos = self._pos(scene, record["body"])
        body_quat = self._quat(scene, record["body"])
        from cloud_robotics_sim.core.success import _quat_to_mat

        local = np.asarray(record["local"], dtype=float)
        local_quat = np.asarray(record["local_quat"], dtype=float)
        rot = _quat_to_mat(body_quat) @ _quat_to_mat(local_quat)
        # rotation matrix -> wxyz quat (active rotation, column-major safe)
        qw = float(np.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0)
        qx = float((rot[2, 1] - rot[1, 2]) / (4.0 * qw)) if qw > 1e-9 else 0.0
        qy = float((rot[0, 2] - rot[2, 0]) / (4.0 * qw)) if qw > 1e-9 else 0.0
        qz = float((rot[1, 0] - rot[0, 1]) / (4.0 * qw)) if qw > 1e-9 else 0.0
        quat = np.array([qw, qx, qy, qz], dtype=float)
        return body_pos + _quat_to_mat(body_quat) @ local, quat

    def _dofs_velocity(self, scene: Any) -> np.ndarray | None:
        """Full DOF velocity vector when the wrapped entity exposes it."""
        world = self._world(scene)
        resolver = getattr(world, "_resolve_entity", None)
        if resolver is None:
            return None
        try:
            get_vel = getattr(resolver(), "get_dofs_velocity", None)
            if get_vel is None:
                return None
            return np.asarray(get_vel(), dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001 - velocity is best effort
            return None

    def _joint_vel_norm(self, scene: Any, joint: str) -> float | None:
        """Norm of one joint's DOF velocities (free=6, else 1)."""
        qvel = self._dofs_velocity(scene)
        if qvel is None:
            return None
        for record in self._meta(scene).get("joints") or []:
            if record.get("name") == joint:
                width = 6 if record.get("type") == "free" else 1
                adr = int(record["qveladr"])
                if adr + width <= qvel.size:
                    return float(np.linalg.norm(qvel[adr : adr + width]))
                return None
        return None

    def _geom_any_contact(self, scene: Any, geom: str) -> bool:
        """True when the named geom appears in any live contact pair."""
        world = self._world(scene)
        contacts = world.get_self_contacts()
        geom_names = world.geom_names()
        link_names = world.link_names()
        wanted = {geom}
        pairs: list[tuple[str, str]] = []
        ga = np.asarray(contacts.get("geom_a", [])).reshape(-1)
        gb = np.asarray(contacts.get("geom_b", [])).reshape(-1)
        la = np.asarray(contacts.get("link_a", [])).reshape(-1)
        lb = np.asarray(contacts.get("link_b", [])).reshape(-1)
        for i in range(la.size):
            a = geom_names[ga[i]] if i < ga.size else link_names[la[i]]
            b = geom_names[gb[i]] if i < gb.size else link_names[lb[i]]
            pairs.append((a, b))
        return any(a in wanted or b in wanted for a, b in pairs)

    def _meta(self, scene: Any) -> dict:
        if self._meta_cache is None:
            import json

            meta_path = self._world_cfg.get("meta_path")
            if not meta_path:
                raise KeyError("efe_djx world.meta_path missing")
            self._meta_cache = json.loads(
                Path(str(meta_path)).read_text(encoding="utf-8")
            )
        return self._meta_cache

    # -- family success ports (verbatim from the env classes) --------------

    def _step_family(self, scene: Any) -> bool:
        family = self._djx.get("family", "")
        trigger = False
        if family == "water_plant":
            ref = self._site_pos(scene, "ref_point")
            plant = self._pos(scene, "plant")
            dx, dy, dz = ref - plant
            inside = (dx * dx + dy * dy <= 0.2 * 0.2) and (-0.2 <= dz <= 0.2)
            joint = self._qpos(scene, "joint_0")
            if joint < 0.25:
                self._trigger_pulled = False
            elif joint > 0.34:
                self._trigger_pulled = True
            if self._trigger_pulled and not inside:
                self._failure = True
            trigger = inside and self._trigger_pulled
        elif family == "pick_bucket":
            box_pos = self._pos(scene, self._djx.get("object_body", "boxed_food_0"))
            corners = np.array(
                [self._site_pos(scene, f"bucket_ref_{i}") for i in range(8)]
            )
            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            inside = bool(np.all(box_pos >= mins) and np.all(box_pos <= maxs))
            bottom = np.array(
                [
                    self._site_pos(scene, name)[2]
                    for name in self._djx.get("bucket_bottom_sites", [])
                ]
            )
            lifted = bool(np.all(bottom - float(self._djx["bucket_bottom_z0"]) >= 0.15))
            return inside and lifted
        elif family == "fold_glasses":
            glass = self._pos(scene, "glass")
            box = self._pos(scene, "open_box")
            local = glass - box
            inside = (
                -0.145 <= local[0] <= 0.145
                and -0.145 <= local[1] <= 0.145
                and -0.045 <= local[2] <= 0.0475 - 0.02
            )
            j0 = self._qpos(scene, "glass_joint_0")
            j1 = self._qpos(scene, "glass_joint_1")
            trigger = j0 > 1.1 and j1 > 1.1 and inside
        elif family == "bimanual_microwave_cook":
            closed = abs(self._qpos(scene, "microjoint")) < 1e-2
            hot = self._pos(scene, "hot_dog")
            micro = self._pos(scene, "microwave_object")
            micro_quat = self._quat(scene, "microwave_object")
            from cloud_robotics_sim.core.success import _quat_to_mat

            local = _quat_to_mat(micro_quat).T @ (hot - micro)
            sites = self._meta(scene).get("sites") or {}
            p0 = np.asarray(sites["int_p0"]["local"], dtype=float)
            px = np.asarray(sites["int_px"]["local"], dtype=float)
            py = np.asarray(sites["int_py"]["local"], dtype=float)
            pz = np.asarray(sites["int_pz"]["local"], dtype=float)
            margin = 0.01
            inside = (
                p0[0] - margin <= local[0] <= px[0] + margin
                and p0[1] - margin <= local[1] <= py[1] + margin
                and p0[2] - margin <= local[2] <= pz[2] + margin
            )
            pressed = self._geom_any_contact(scene, "start_button")
            return bool(inside and closed and pressed)
        elif family == "bimanual_assembly":
            trigger = self._touch(scene, "industreal_round_peg_8mm", "socket_bottom")
        elif family == "click_mouse":
            mouse = self._pos(scene, "mouse_root")
            pad = self._pos(scene, "mousepad")
            radius = float(self._djx["mousepad_radius"])
            dx, dy, dz = mouse - pad
            in_pad = (dx * dx + dy * dy <= radius * radius) and abs(dz) <= 0.05
            clicked = (
                self._qpos(scene, "mouse_joint_0")
                - float(self._djx["mouse_joint0_init"])
            ) > 0.001
            if in_pad and clicked and not self._display_blue:
                self._display_blue = True
            trigger = in_pad and self._display_blue
        elif family == "pinch_tongs":
            joint = self._qpos(scene, "tongs_joint_0")
            if joint <= -0.07 and self._pinch_in_progress:
                self._pinch_in_progress = False
                self._pinch_count += 1
            elif joint >= 0.1 and not self._pinch_in_progress:
                self._pinch_in_progress = True
            tongs_geom = (self._meta(scene).get("sensors") or {}).get("tongs_pos")
            tongs_z = float(
                self._geom_pose(scene, tongs_geom["objname"])[0][2]
                if tongs_geom and tongs_geom.get("objkind") == "geom"
                else self._pos(scene, self._djx.get("tongs_body", "tongs"))[2]
            )
            trigger = tongs_z >= float(self._djx["lift_z"]) and self._pinch_count >= 3
        elif family == "hammer_nail":
            face_pos, _ = self._geom_pose(scene, "face")
            face_z = float(face_pos[2])
            if self._prev_face_z is None:
                self._prev_face_z = face_z
            vz = (face_z - self._prev_face_z) / 0.02
            self._prev_face_z = face_z
            self._vz_buf.append(vz)
            if len(self._vz_buf) > 12:
                self._vz_buf.pop(0)
            hit = self._touch(scene, "face", "nail_head") or self._touch(
                scene, "face", "nail_shaft"
            )
            for head in ("head", "neck", "claw"):
                if hit:
                    break
                hit = self._touch(scene, head, "nail_head") or self._touch(
                    scene, head, "nail_shaft"
                )
            if hit:
                preimpact_vz = min(self._vz_buf) if self._vz_buf else 0.0
                threshold = float(self._djx.get("impact_vel_threshold", 0.02))
                if preimpact_vz < -threshold:
                    scale = min(3.0, abs(preimpact_vz) / max(threshold, 1e-6))
                    delta = float(self._djx.get("impact_insert_step", 0.008)) * scale
                    new_depth = min(
                        float(self._djx.get("max_insert_depth", 0.0726)),
                        self._nail_depth + delta,
                    )
                    if new_depth > self._nail_depth:
                        self._nail_depth = new_depth
            return self._nail_depth >= float(self._djx.get("success_depth", 0.04))
        elif family == "bimanual_hanoi":
            return self._hanoi_step(scene)
        elif family == "bimanual_unlock_ipad":
            return self._unlock_step(scene)
        elif family == "bimanual_photograph":
            return self._photograph_step(scene)
        else:
            return False

        # Held families: count consecutive trigger steps.
        if self._success_target <= 1:
            return bool(trigger)
        self._success_count = self._success_count + 1 if trigger else 0
        return self._success_count >= self._success_target

    # -- bimanual family ports ---------------------------------------------

    def _hanoi_step(self, scene: Any) -> bool:
        meta = self._meta(scene)
        disks = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        posts = {
            "A": self._site_pos(scene, "hanoi_post_a_site"),
            "B": self._site_pos(scene, "hanoi_post_b_site"),
            "C": self._site_pos(scene, "hanoi_post_c_site"),
        }
        base_top_z = float(meta["bodies"]["hanoi_base"]["world"][2]) + float(
            self._djx["base_upper_top_local_z"]
        )
        disk_pos = {}
        disk_static = {}
        for disk in disks:
            disk_pos[disk] = self._pos(scene, disk)
            vel = self._joint_vel_norm(scene, f"{disk}_joint")
            disk_static[disk] = True if vel is None else vel <= 0.35
        assign_tol = float(self._djx.get("post_assignment_tol", 0.04))
        towers: dict[str, list[str]] = {"A": [], "B": [], "C": []}
        assignment: dict[str, str] = {}
        for disk in disks:
            pos = disk_pos[disk]
            closest = min(
                posts, key=lambda k: float(np.linalg.norm(pos[:2] - posts[k][:2]))
            )
            if float(np.linalg.norm(pos[:2] - posts[closest][:2])) <= assign_tol:
                assignment[disk] = closest
                towers[closest].append(disk)
            elif pos[2] <= base_top_z + 0.06:
                assignment[disk] = "table"
            else:
                assignment[disk] = "floating"
        for post in towers:
            towers[post].sort(key=lambda d: disk_pos[d][2])
        rank = {"hanoi_disk_large": 3, "hanoi_disk_medium": 2, "hanoi_disk_small": 1}
        height_tol = float(self._djx.get("placement_height_tol", 0.012))

        def errors(stack: list[str], post: str) -> tuple[float, float]:
            running = base_top_z
            xy_max = height_max = 0.0
            half = self._djx["disk_half_height"]
            for disk in stack:
                expected = running + float(half[disk])
                running += 2.0 * float(half[disk])
                xy = float(np.linalg.norm(disk_pos[disk][:2] - posts[post][:2]))
                hz = abs(float(disk_pos[disk][2]) - expected)
                xy_max = max(xy_max, xy)
                height_max = max(height_max, hz)
            return xy_max, height_max

        illegal = False
        reason = ""
        for post, stack in towers.items():
            ranks = [rank[d] for d in stack]
            if any(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)):
                if all(disk_static[d] for d in stack):
                    xy_e, h_e = errors(stack, post)
                    if xy_e <= assign_tol and h_e <= height_tol:
                        illegal = True
                        reason = f"illegal_stack_{post}"
                        break
        if not illegal and all(disk_static.values()):
            if any(a == "table" for a in assignment.values()):
                illegal = True
                reason = "disk_on_table"
        if illegal:
            if reason == self._illegal_trigger_reason:
                self._illegal_trigger_count += 1
            else:
                self._illegal_trigger_reason = reason
                self._illegal_trigger_count = 1
        else:
            self._illegal_trigger_reason = ""
            self._illegal_trigger_count = 0
        if self._illegal_trigger_count >= int(
            self._djx.get("illegal_trigger_target", 10)
        ):
            self._failure = True
        target = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        geom_ok = False
        if towers["C"] == target and not towers["A"] and not towers["B"]:
            xy_e, h_e = errors(target, "C")
            geom_ok = (
                xy_e <= assign_tol and h_e <= height_tol and all(disk_static.values())
            )
        trigger = geom_ok and not self._failure
        self._success_count = self._success_count + 1 if trigger else 0
        return self._success_count >= self._success_target

    def _unlock_step(self, scene: Any) -> bool:
        if self._unlock_failed:
            return False
        geoms = self._meta(scene).get("geoms") or {}
        pressed = {
            name
            for name in geoms
            if name.startswith("btn")
            and name.endswith("_cyl")
            and self._geom_any_contact(scene, name)
        }
        newly = pressed - self._pressed_last
        self._pressed_last = pressed
        sequence = [int(v) for v in self._djx.get("unlock_sequence", [1, 2, 3])]
        for name in sorted(newly):
            digit = int(name.replace("btn", "").replace("_cyl", ""))
            expected = sequence[self._unlock_index]
            if digit == expected:
                self._unlock_index += 1
                if self._unlock_index >= len(sequence):
                    self._screen_unlocked = True
            else:
                self._unlock_failed = True
                self._unlock_index = 0
                self._screen_unlocked = False
        if self._screen_unlocked:
            self._success_count += 1
        else:
            self._success_count = 0
        if self._unlock_failed:
            self._failure = True
        return self._success_count >= self._success_target

    def _photograph_step(self, scene: Any) -> bool:
        region_pos, _ = self._geom_pose(scene, "photograph_target_region_visual")
        view = self._site_pos(scene, "view_direction_site")
        delta = view - region_pos
        radial = float(np.linalg.norm(delta[:2]))
        height = float(abs(delta[2]))
        region_pass = radial <= 0.16 and height <= 0.10
        axis_pos, axis_quat = self._geom_pose(scene, "camera_axis_z")
        from cloud_robotics_sim.core.success import _quat_to_mat

        axis_z = _quat_to_mat(axis_quat)[:, 2]
        norm = float(np.linalg.norm(axis_z))
        axis_z = axis_z / norm if norm >= 1e-8 else np.array([0.0, 0.0, 1.0])
        cos_val = float(np.clip(np.dot(axis_z, np.array([1.0, 0.0, 0.0])), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        angle_pass = angle_deg <= 10.0
        shutter = self._geom_any_contact(scene, "camera_body_top_cap")
        return bool(region_pass and angle_pass and shutter)

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._meta_cache = None
        self._reset_state()
        self._apply_frozen_state(scene)
        if self._djx.get("family") == "click_mouse":
            world = self._world(scene)
            for record in self._meta(scene).get("joints") or []:
                if record.get("name") == "mouse_joint_0":
                    qpos = np.asarray(world.get_qpos(), dtype=float)
                    self._djx["mouse_joint0_init"] = float(qpos[int(record["qposadr"])])
        info["djx_success"] = False
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        success = self._step_family(scene)
        info["djx_success"] = success
        if self._failure:
            return reward, True, truncated, info
        return reward, terminated, truncated, info


@register_task("efe_djx")
def _make_efe_djx(**kwargs: Any) -> EfeDjxTask:
    """DexJoCo frozen-scene task with world/djx YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    world = kwargs.pop("world", None)
    djx = kwargs.pop("djx", None)
    return EfeDjxTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        djx=djx,
    )


@register_task("efe_dtb")
def _make_efe_dtb(**kwargs: Any) -> EfeDtbTask:
    """DexToolBench frozen-scene task with world/dtb YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    world = kwargs.pop("world", None)
    dtb = kwargs.pop("dtb", None)
    return EfeDtbTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        dtb=dtb,
    )


@register_robot("i2rt_yam_bimanual")
def _make_i2rt_yam_bimanual(**kwargs: Any) -> Any:
    """Twin I2RT YAM arms (ABC Sim / YAMLab, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("i2rt_yam_bimanual")


class EfeAbcTask(EfeBaseTask):
    """ABC Sim frozen-world task (two I2RT YAM arms at a table).

    The task YAML carries the whole frozen scene as one ``world`` MJCF asset
    (robot + table + objects at the frozen poses), plus an ``abc`` block the
    resolver bakes from the upstream probe:

    * ``step_check`` — the verbatim per-control-step success predicate
      (an ``abc_*`` success spec; latched families see below);
    * ``hold_steps`` — the EFE suite's 30-consecutive-step rule (the task
      counts and publishes ``info['abc_success_run']``; the outer success
      condition is :class:`AbcHoldCondition`);
    * ``runtime`` — the three Python-side upstream runtimes, ported at
      control-step granularity (one task step = 17 upstream physics steps):
      the ball-tray countdown + perturbation pulses, the conveyor belt's
      kinematic carry, and the drawer-search ordered target latch.

    The ball-tray family instead publishes ``info['abc_tray_complete']`` and
    terminates the episode itself on upstream's failure conditions (ball
    leaves the tray / handle let go during the countdown); drawer search
    publishes ``info['abc_drawer_completed']``; sweep publishes the active
    scrap mask ``info['abc_sweep_active']`` (upstream's evaluator-side latch
    ``active |= z > active_z_min``).

    Actions are the 14 upstream joint-position targets
    ``[left_j1..6, left_grip, right_j1..6, right_grip]``; when the world
    entity exposes Genesis dof control they are float32-rounded, clamped to
    the actuator ctrlranges (upstream servo semantics) and applied as
    position targets — otherwise control is a documented no-op, matching
    the RoboCasa world-embedded precedent (EFE grades final states).
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        abc: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_abc world must be a mapping or None")
        if abc is not None and not isinstance(abc, dict):
            raise TypeError("efe_abc abc must be a mapping or None")
        self._world_cfg = world or {}
        self._abc = abc or {}
        self._step_condition: Any = None
        self._drawer_condition: Any = None
        self._servo_warned = False
        self._reset_state()

    # -- state -------------------------------------------------------------

    def _reset_state(self) -> None:
        """Per-episode latches (upstream evaluator/runtime instance state)."""
        self._success_run = 0
        self._step_condition = None
        self._drawer_condition = None
        tray = self._abc.get("tray") or {}
        self._tray: dict[str, Any] = {
            "started": False,
            "start_time": None,
            "deadline": None,
            "grip_since": None,
            "last_both_contact_time": None,
            "next_pulse_at": None,
            "pulse_until": None,
            "direction": 0.0,
            "run_pulses": 0,
            "complete": False,
            "sim_time": 0.0,
        }
        if tray:
            seed = int((self._abc.get("randomization_seed") or 0))
            import numpy as _np

            rng = _np.random.default_rng(seed)
            self._tray["direction"] = float(rng.uniform(0.0, 2.0 * _np.pi))
            self._tray["run_pulses"] = int(
                rng.integers(
                    int(tray.get("run_length_min", 3)),
                    int(tray.get("run_length_max", 5)) + 1,
                )
            )
        self._drawer_completed = 0
        sweep = self._abc.get("sweep") or {}
        self._sweep_active = [False] * len(sweep.get("scraps", []))

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_abc expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    # -- helpers -----------------------------------------------------------

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))
        conveyor = self._abc.get("conveyor") or {}
        if conveyor:
            self._open_idle_grippers(world)

    def _open_idle_grippers(self, world: Any) -> None:
        """conveyor_pick runtime: start each gripper parked open.

        The whole-world entity backend exposes ``get_qpos``/``set_qpos``
        but no ``get_qvel`` (Genesis 1.4 RigidEntity has no velocity
        getter), so this is a qpos read-modify-write; the frozen qvel is
        zeros and stays zeros for the parked fingers.
        """
        qpos = np.asarray(world.get_qpos(), dtype=np.float64).copy()
        joints = self._abc.get("joints") or {}
        for side in ("left", "right"):
            for finger, target in (
                (f"{side}_left_finger", "open_hi"),
                (f"{side}_right_finger", "open_lo"),
            ):
                record = joints.get(finger)
                if not record:
                    continue
                adr = int(record["qposadr"])
                lo, hi = (float(v) for v in record.get("range") or (0.0, 0.0))
                qpos[adr] = hi if target == "open_hi" else lo
        world.set_qpos(qpos)

    def _apply_action_targets(self, scene: Any, action: np.ndarray) -> None:
        """Servo the 14 upstream joint targets (best effort; see class doc)."""
        control = self._abc.get("control") or {}
        joints = control.get("joints") or []
        if not joints or action is None or len(action) < len(joints):
            return
        world = self._world(scene)
        control_fn = getattr(world, "control_dofs_position", None)
        idx_fn = getattr(world, "get_joint_dofs_idx_local", None)
        if control_fn is None or idx_fn is None:
            if not self._servo_warned:
                logger.info(
                    "efe_abc: world entity exposes no dof position control; "
                    "joint-target actions are not applied"
                )
                self._servo_warned = True
            return
        rows = np.asarray(action, dtype=np.float32).reshape(-1)[: len(joints)]
        targets = rows.astype(np.float64)
        for i, joint in enumerate(joints):
            lo, hi = (
                float(v)
                for v in (
                    joint.get("ctrlrange") or joint.get("range") or (-np.inf, np.inf)
                )
            )
            targets[i] = min(max(targets[i], lo), hi)
        try:
            dofs_idx = [idx_fn(j["name"]) for j in joints]
            flat = [
                i for sub in dofs_idx for i in (sub if isinstance(sub, list) else [sub])
            ]
            control_fn(targets=np.asarray(targets), dofs_idx=flat)
        except Exception as exc:  # noqa: BLE001 - servo is best effort
            if not self._servo_warned:
                logger.info(f"efe_abc: dof position control unavailable: {exc}")
                self._servo_warned = True

    # -- runtimes (control-step granularity; see class doc) -----------------

    def _update_conveyor(self, scene: Any, dt: float) -> None:
        from cloud_robotics_sim.core.success import _world_bodies_touch

        conveyor = self._abc.get("conveyor") or {}
        if not conveyor:
            return
        world = self._world(scene)
        speed = float(conveyor.get("belt_speed_y", -0.20))
        joints = self._abc.get("joints") or {}
        belt = conveyor.get("belt_geom")
        # One read-modify-write per step (the entity backend has no qvel
        # getter; the kinematic carry upstream pairs with qvel[y] = 0, which
        # the frozen zero-qvel state already satisfies between carries).
        qpos = np.asarray(world.get_qpos(), dtype=np.float64).copy()
        dirty = False
        for obj in conveyor.get("objects", []):
            if not _world_bodies_touch(world, obj, belt):
                continue
            record = joints.get(obj)
            if not record:
                continue
            adr = int(record["qposadr"])
            qpos[adr + 1] += speed * dt  # upstream: qpos[y] += speed*dt
            dirty = True
        if dirty:
            world.set_qpos(qpos)

    def _update_tray(self, scene: Any, dt: float) -> bool:
        """Ball-tray runtime + Session.step rule. Returns True on episode end."""
        from cloud_robotics_sim.core.success import (
            _abc_local,
            _abc_pos,
            _abc_quat,
            _world_bodies_touch,
        )

        tray = self._abc.get("tray") or {}
        if not tray:
            return False
        world = self._world(scene)
        t = self._tray
        t["sim_time"] = float(t.get("sim_time", 0.0)) + dt
        now = t["sim_time"]

        ball, tray_body = tray["ball"], tray["tray"]
        local = _abc_local(
            _abc_pos(scene, tray_body),
            _abc_quat(scene, tray_body),
            _abc_pos(scene, ball),
        )
        dropped = local[2] < float(tray["dropped_local_z"])
        inside = abs(local[0]) <= float(tray["usable_half_extents"][0]) and abs(
            local[1]
        ) <= float(tray["usable_half_extents"][1])
        alive = inside and not dropped
        if not alive:
            return True  # failure: the ball left the tray

        left = tray.get("left_handle_geom")
        right = tray.get("right_handle_geom")
        sides = tray.get("finger_bodies") or {}
        left_held = any(
            _world_bodies_touch(world, b, left) for b in sides.get("left", [])
        )
        right_held = any(
            _world_bodies_touch(world, b, right) for b in sides.get("right", [])
        )
        if left_held and right_held:
            t["last_both_contact_time"] = now
        held = (
            t["last_both_contact_time"] is not None
            and now - float(t["last_both_contact_time"])
            <= float(tray.get("grace_s", 0.1)) + 1e-9
        )
        if not t["started"]:
            if left_held and right_held:
                if t["grip_since"] is None:
                    t["grip_since"] = now
                t["started"] = True
                t["start_time"] = now
                t["deadline"] = now + float(tray["duration_s"])
                t["next_pulse_at"] = now + float(tray.get("pulse_interval_s", 0.5))
        elif not held:
            return True  # failure: a handle was let go during the countdown

        if t["started"] and not t["complete"]:
            if t["next_pulse_at"] is not None and now >= float(t["next_pulse_at"]):
                self._fire_pulse(scene, tray)
                t["next_pulse_at"] = now + float(tray.get("pulse_interval_s", 0.5))
            if t["deadline"] is not None and now >= float(t["deadline"]):
                t["complete"] = True
        return False

    def _fire_pulse(self, scene: Any, tray: dict) -> None:
        """Upstream perturbation: a 0.1 s force pulse in the tray's plane."""
        import numpy as _np

        from cloud_robotics_sim.core.success import (
            _abc_quat,
            _quat_to_mat,
        )

        t = self._tray
        lo, hi = (float(v) for v in tray.get("force_range_n", [0.024, 0.034]))
        force_n = (lo + hi) / 2.0  # magnitude is sampled per pulse upstream
        direction = float(t["direction"])
        rot = _quat_to_mat(_abc_quat(scene, tray["tray"]))
        axis = rot[:, 0] * _np.cos(direction) + rot[:, 1] * _np.sin(direction)
        force = axis * force_n
        world = self._world(scene)
        resolver = getattr(world, "_resolve_entity", None)
        if resolver is None:
            return
        try:
            link = resolver().get_link(tray["ball"])
            apply = getattr(link, "apply_external_force", None)
            if apply is not None:
                apply(_np.asarray(force, dtype=_np.float64))
        except Exception as exc:  # noqa: BLE001 - force application is best effort
            if not self._servo_warned:
                logger.info(f"efe_abc: tray perturbation force unavailable: {exc}")
                self._servo_warned = True

    def _update_drawer(self, scene: Any) -> None:
        drawer = self._abc.get("drawer") or {}
        if not drawer:
            return
        from cloud_robotics_sim.core.success import create_success_condition

        if self._drawer_condition is None:
            self._drawer_condition = create_success_condition(
                {
                    "type": "abc_in_local_box",
                    "params": {
                        "objects": [drawer["targets"][self._drawer_completed]],
                        "frame": drawer["bin"],
                        "local_center": drawer["local_center"],
                        "half_size": drawer["half_size"],
                    },
                }
            )
        while self._drawer_completed < len(drawer["targets"]):
            if not self._drawer_condition.evaluate(scene, None, {}):
                break
            self._drawer_completed += 1
            if self._drawer_completed < len(drawer["targets"]):
                self._drawer_condition = create_success_condition(
                    {
                        "type": "abc_in_local_box",
                        "params": {
                            "objects": [drawer["targets"][self._drawer_completed]],
                            "frame": drawer["bin"],
                            "local_center": drawer["local_center"],
                            "half_size": drawer["half_size"],
                        },
                    }
                )

    def _update_sweep(self, scene: Any) -> None:
        sweep = self._abc.get("sweep") or {}
        if not sweep:
            return
        from cloud_robotics_sim.core.success import _abc_pos

        z_min = float(sweep["active_z_min"])
        for index, scrap in enumerate(sweep.get("scraps", [])):
            if not self._sweep_active[index] and _abc_pos(scene, scrap)[2] > z_min:
                self._sweep_active[index] = True

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._reset_state()
        self._apply_frozen_state(scene)
        info["abc_success_run"] = 0
        info["abc_tray_complete"] = False
        info["abc_drawer_completed"] = 0
        info["abc_sweep_active"] = list(self._sweep_active)
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        dt = float(self._abc.get("control_dt", 0.034))
        self._apply_action_targets(scene, action)
        self._update_conveyor(scene, dt)
        tray_over = self._update_tray(scene, dt)
        self._update_drawer(scene)
        self._update_sweep(scene)

        if tray_over:
            info["abc_tray_complete"] = False
            return reward, True, truncated, info
        if self._abc.get("tray"):
            info["abc_tray_complete"] = bool(self._tray["complete"])
            self._success_run = 0
        else:
            step_check = self._abc.get("step_check")
            if step_check and self._step_condition is None:
                from cloud_robotics_sim.core.success import create_success_condition

                self._step_condition = create_success_condition(step_check)
            ok = (
                bool(self._step_condition.evaluate(scene, robot, info))
                if self._step_condition
                else False
            )
            self._success_run = self._success_run + 1 if ok else 0
            info["abc_success_run"] = self._success_run
        info["abc_drawer_completed"] = self._drawer_completed
        info["abc_sweep_active"] = list(self._sweep_active)
        return reward, terminated, truncated, info


@register_task("efe_abc")
def _make_efe_abc(**kwargs: Any) -> EfeAbcTask:
    """ABC Sim frozen-world task with world/abc YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    world = kwargs.pop("world", None)
    abc = kwargs.pop("abc", None)
    return EfeAbcTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        abc=abc,
    )


class EfeRobocasaTask(EfeBaseTask):
    """RoboCasa frozen-world task.

    The task YAML carries the whole kitchen as one ``world`` MJCF asset
    (robot + fixtures + objects at the frozen poses compiled into the
    extracted XML); this task applies the frozen ``qpos``/``qvel`` at reset,
    registers the baked sites (eef, buttons) on the world entity, and steps
    the upstream appliance toggle state machines (microwave / coffee
    machine / toaster oven: door-open forces off, gripper-button contact
    toggles) into ``info['rc_appliance']`` for :class:`RcApplianceCondition`.
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        appliances: list[dict] | None = None,
        timers: list[dict] | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_robocasa world must be a mapping or None")
        if appliances is not None and not isinstance(appliances, list):
            raise TypeError("efe_robocasa appliances must be a list or None")
        if timers is not None and not isinstance(timers, list):
            raise TypeError("efe_robocasa timers must be a list or None")
        self._world_cfg = world or {}
        self._appliance_cfgs = appliances or []
        self._timer_cfgs = timers or []
        self._appliance_states: dict[str, bool] = {}
        self._timer_counts: dict[str, int] = {}
        self._timer_conditions: dict[str, Any] = {}
        self._sites_registered = False

    # -- helpers -----------------------------------------------------------

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_robocasa expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _register_sites(self, scene: Any) -> None:
        """Bake site (body, local offset) pairs onto the world entity."""
        if self._sites_registered:
            return
        meta_path = self._world_cfg.get("meta_path")
        sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
        if meta_path:
            import json

            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            for name, record in (meta.get("sites") or {}).items():
                try:
                    local = tuple(float(v) for v in str(record["pos"]).split())
                except (KeyError, ValueError):
                    continue
                sites[name] = (record["body"], local)  # type: ignore[assignment]
        for name, record in (self._world_cfg.get("extra_sites") or {}).items():
            body = str(record[0])
            local = tuple(float(v) for v in record[1])
            sites[name] = (body, local)  # type: ignore[assignment]
        world = self._world(scene)
        world._rc_sites = sites
        self._sites_registered = True

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        overrides = self._world_cfg.get("qpos_overrides") or {}
        if qpos_path:
            qpos = np.load(qpos_path)
            if overrides:
                qpos = qpos.copy()
                for joint, value in overrides.items():
                    indices = self._qpos_indices(scene, joint)
                    if indices:
                        qpos[indices[0]] = float(value)
            world.set_qpos(qpos)
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    def _qpos_indices(self, scene: Any, joint: str) -> list[int]:
        """Qpos segment of one named joint (document-order width walk)."""
        meta_path = self._world_cfg.get("meta_path")
        if not meta_path:
            return []
        import json

        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        sizes = {"free": 7, "ball": 4}
        cursor = 0
        for record in meta.get("joints") or []:
            width = sizes.get(record.get("type"), 1)
            if record.get("name") == joint:
                return [cursor]
            cursor += width
        return []

    def _update_appliances(self, scene: Any) -> None:
        """Upstream ``Fixture.update_state`` toggle semantics, per step."""
        if not self._appliance_cfgs:
            return
        world = self._world(scene)
        from cloud_robotics_sim.core.success import (  # local: circular-safe
            _world_bodies_touch,
        )

        for cfg in self._appliance_cfgs:
            name = cfg["name"]
            door_joint = cfg.get("door_joint")
            door_open = False
            if door_joint:
                value = float(world.get_joint_position(door_joint))
                lo, hi = (float(v) for v in cfg.get("door_range", [0.0, 1.5707963]))
                sign = 1.0
                if lo < 0.0 <= hi:
                    # drawer/door convention: range [-d, 0] with upstream
                    # sign -1 (closed at q=0 must read 0, not 1)
                    sign, lo, hi = -1.0, 0.0, -lo
                door_open = (sign * value - lo) / (hi - lo) > 0.005
            gripper = cfg.get("gripper_body", "gripper0_right")
            if door_open:
                self._appliance_states[name] = False
                continue
            start = cfg.get("start_button")
            stop = cfg.get("stop_button")
            start_pressed = (
                _world_bodies_touch(world, gripper, start) if start else False
            )
            stop_pressed = _world_bodies_touch(world, gripper, stop) if stop else False
            if self._appliance_states.get(name, False) and stop_pressed:
                self._appliance_states[name] = False
            elif not self._appliance_states.get(name, False) and start_pressed:
                self._appliance_states[name] = True

    def _update_timers(self, scene: Any, robot: Any, info: dict) -> None:
        """Consecutive-step success counters (washed/broiled/scrubbed).

        Mirrors upstream ``_check_success`` call-pattern state machines:
        the condition block is evaluated every step; holding it increments
        the counter, losing it resets to zero.
        """
        if not self._timer_cfgs:
            return
        from cloud_robotics_sim.core.success import create_success_condition

        for cfg in self._timer_cfgs:
            name = cfg["name"]
            if name not in self._timer_conditions:
                self._timer_conditions[name] = create_success_condition(
                    {"type": "all", "params": {"conditions": cfg["conditions"]}}
                )
            condition = self._timer_conditions[name]
            if condition.evaluate(scene, robot, info):
                self._timer_counts[name] = self._timer_counts.get(name, 0) + 1
            else:
                self._timer_counts[name] = 0

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        """Apply the frozen world state and reset appliance flags."""
        info = super().reset(scene, robot, seed)
        self._register_sites(scene)
        self._apply_frozen_state(scene)
        self._appliance_states = {
            cfg["name"]: bool(cfg.get("initial_on", False))
            for cfg in self._appliance_cfgs
        }
        self._timer_counts = {}
        self._update_appliances(scene)
        info["rc_appliance"] = dict(self._appliance_states)
        info["rc_timer"] = dict(self._timer_counts)
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        """Advance one step; mirror upstream appliance/toggle checks."""
        reward, terminated, truncated, info = super().step(scene, robot, action)
        self._update_appliances(scene)
        self._update_timers(scene, robot, info)
        info["rc_appliance"] = dict(self._appliance_states)
        info["rc_timer"] = dict(self._timer_counts)
        return reward, terminated, truncated, info


@register_task("efe_robocasa")
def _make_efe_robocasa(**kwargs: Any) -> EfeRobocasaTask:
    """RoboCasa frozen-world task with world/appliances YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    world = kwargs.pop("world", None)
    appliances = kwargs.pop("appliances", None)
    timers = kwargs.pop("timers", None)
    return EfeRobocasaTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        appliances=appliances,
        timers=timers,
    )


class EfePgTask(EfeBaseTask):
    """MuJoCo Playground frozen-scene task (9 quadruped + 4 manipulation).

    The task YAML carries the composed world (``world`` block, see
    ``tools/pg_worlds.py``) and a ``pg`` block with the check constants the
    EFE graders' sessions define (``rcb_quadruped.py`` / ``rcb_aloha.py`` /
    ``rcb_panda.py``, mirrored under ``data/efe/upstream/pg-src/efe-benchmark``).
    Success formulas are ported verbatim at control-step granularity:

    * joystick — every command segment (after its first 25 control steps) has
      mean ``|v_xy - cmd| <= 0.2 m/s`` and ``|w_z - cmd_wz| <= 0.3 rad/s`` (+
      gait RMS height error <= 0.05 m for gait tracking), the episode reaches
      its horizon and Playground's termination rule never fired (trunk up axis
      below ``up_z_min``, Barkour's joint box / torso height);
    * getup — the last 25 control steps all have trunk tilt <= 10 deg and IMU
      height >= the env's minimum (0.22 m Go1 / 0.40 m Spot);
    * stand_on_two — the last 100 control steps all have the forward axis
      within 30 deg of straight down (handstand) or up (footstand), IMU height
      >= 0.40 m and the lifted feet off the floor, with no fall before;
    * ALOHA peg insertion — the tip is within 5 mm of the socket axis with the
      axial parameter t in ``[t_min, 0.75]`` (``t_min`` read from the model's
      rear wall minus a 2 mm contact margin);
    * ALOHA handover — the box is within 3 cm of the target, touches a right
      gripper finger, does not touch the left arm and is at least 5 cm high;
    * Panda pick — the box is within 2 cm and 15 deg of the target pose;
    * Panda cabinet — the handle's x is within 1.5 cm of the target x.

    Controls are best-effort: action rows are clipped to the actuator ranges
    and servoed with the world entity's dof-position control when it exposes
    one (same contract as ``EfeAbcTask``); IMU/gyro/velocimeter readings are
    derived from the root body's world velocity rotated into the IMU frame.
    Barkour's random pushes are external disturbances and are not reproduced
    (documented; the success formula is unaffected). The verdict is published
    as ``info['pg']['success']`` — the outer :class:`PgSuccessCondition`
    forwards it; missing info is a failure (nop-safe).
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        pg: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_pg world must be a mapping or None")
        if pg is not None and not isinstance(pg, dict):
            raise TypeError("efe_pg pg must be a mapping or None")
        self._world_cfg = world or {}
        self._pg = pg or {}
        self._servo_warned = False
        self._steps = 0
        self._fell_at: int | None = None
        self._log: list[tuple] = []
        self._gait_log: list[float] = []
        self._pose_log: list[tuple] = []

    # -- world access -------------------------------------------------------

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_pg expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    # -- shared accessors ----------------------------------------------------

    def _pos(self, scene: Any, body: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_pos

        return _abc_pos(scene, body)

    def _quat(self, scene: Any, body: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_quat

        return _abc_quat(scene, body)

    def _site_pose(self, scene: Any, site: str) -> tuple[np.ndarray, np.ndarray]:
        """World (pos, rotation matrix) of a site from the pg sites table."""
        record = (self._pg.get("sites") or {})[site]
        body_pos = self._pos(scene, record["body"])
        body_quat = self._quat(scene, record["body"])
        from cloud_robotics_sim.core.success import _quat_to_mat

        rot = _quat_to_mat(body_quat) @ _quat_to_mat(
            np.asarray(record["local_quat"], dtype=float)
        )
        pos = body_pos + _quat_to_mat(body_quat) @ np.asarray(
            record["local"], dtype=float
        )
        return pos, rot

    def _pair_contact(self, scene: Any, geom_a: str, geom_b: str) -> bool:
        """True when the two named geoms appear in one live contact pair."""
        world = self._world(scene)
        contacts = world.get_self_contacts()
        geom_names = world.geom_names()
        link_names = world.link_names()
        ga = np.asarray(contacts.get("geom_a", [])).reshape(-1)
        gb = np.asarray(contacts.get("geom_b", [])).reshape(-1)
        la = np.asarray(contacts.get("link_a", [])).reshape(-1)
        lb = np.asarray(contacts.get("link_b", [])).reshape(-1)
        for i in range(la.size):
            a = geom_names[ga[i]] if i < ga.size else link_names[la[i]]
            b = geom_names[gb[i]] if i < gb.size else link_names[lb[i]]
            if (a == geom_a and b == geom_b) or (a == geom_b and b == geom_a):
                return True
        return False

    def _any_contact(self, scene: Any, geom: str, others: set[str]) -> bool:
        return any(self._pair_contact(scene, geom, o) for o in others)

    def _root_velocity(self, scene: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """Root body (world linear, world angular) velocity, best effort."""
        world = self._world(scene)
        resolver = getattr(world, "_resolve_entity", None)
        if resolver is None:
            return None
        try:
            get_vel = getattr(resolver(), "get_dofs_velocity", None)
            if get_vel is None:
                return None
            vel = np.asarray(get_vel(), dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001 - velocity is best effort
            return None
        if vel.size < 6:
            return None
        return vel[0:3].copy(), vel[3:6].copy()

    # -- controls ------------------------------------------------------------

    def _apply_targets(self, scene: Any, action: np.ndarray) -> None:
        """Servo the upstream actuator targets (best effort; see class doc)."""
        joints = self._pg.get("joint_names") or self._pg.get("actuators") or []
        if not joints or action is None or len(action) < len(joints):
            return
        world = self._world(scene)
        control_fn = getattr(world, "control_dofs_position", None)
        idx_fn = getattr(world, "get_joint_dofs_idx_local", None)
        if control_fn is None or idx_fn is None:
            if not self._servo_warned:
                logger.info(
                    "efe_pg: world entity exposes no dof position control; "
                    "joint-target actions are not applied"
                )
                self._servo_warned = True
            return
        ctrlrange = self._pg.get("ctrlrange") or []
        targets = np.asarray(action, dtype=np.float64).reshape(-1)[: len(joints)]
        for i in range(len(targets)):
            if i < len(ctrlrange) and ctrlrange[i]:
                lo, hi = (float(v) for v in ctrlrange[i])
                targets[i] = min(max(targets[i], lo), hi)
        try:
            dofs_idx = [idx_fn(j) for j in joints]
            flat = [
                i for sub in dofs_idx for i in (sub if isinstance(sub, list) else [sub])
            ]
            control_fn(targets=targets, dofs_idx=flat)
        except Exception as exc:  # noqa: BLE001 - servo is best effort
            if not self._servo_warned:
                logger.info(f"efe_pg: dof position control unavailable: {exc}")
                self._servo_warned = True

    # -- quadruped runtime (rcb_quadruped.Session, control-step granularity) --

    def _gait_rz(self, phi: np.ndarray, swing_height: float) -> np.ndarray:
        """mujoco_playground/_src/gait.py get_rz (cubic Bezier up then down)."""
        x = (np.asarray(phi) + np.pi) / (2 * np.pi)

        def bez(y0: float, y1: float, t: np.ndarray) -> np.ndarray:
            return np.asarray(
                y0 + (y1 - y0) * (t**3 + 3 * (t**2 * (1 - t))), dtype=float
            )

        curve = np.where(
            x <= 0.5, bez(0.0, swing_height, 2 * x), bez(swing_height, 0.0, 2 * x - 1)
        )
        return np.asarray(curve, dtype=float)

    def _command_at(self, step: int) -> np.ndarray:
        commands = self._pg.get("commands") or []
        horizon = int(self._pg.get("horizon", 0))
        cmd = np.zeros(3)
        for i, seg in enumerate(commands):
            if int(seg["start_step"]) <= step:
                cmd = np.array([seg["vx"], seg["vy"], seg["wz"]], dtype=float)
        if horizon and step >= horizon:
            return cmd
        return cmd

    def _quadruped_log(self, scene: Any) -> None:
        """Append one control-step row (lin_err, yaw_err, tilt, imu_h, up_z)."""
        imu = self._pg.get("imu_site")
        if not imu:
            self._log.append((float("inf"), float("inf"), 180.0, 0.0, 0.0))
            return
        pos, rot = self._site_pose(scene, str(imu))
        up_z = float(rot[2, 2])
        tilt = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
        cmd = self._command_at(self._steps)
        vel = self._root_velocity(scene)
        if vel is not None:
            lin_w, ang_w = vel
            lin_err = float(np.linalg.norm(rot.T @ lin_w - cmd[:2]))
            yaw_err = float(abs(float((rot.T @ ang_w)[2]) - cmd[2]))
        else:
            lin_err, yaw_err = float("inf"), float("inf")
        self._log.append((lin_err, yaw_err, tilt, float(pos[2]), up_z))

        kind = self._pg.get("task_kind")
        consts = self._pg.get("constants") or {}
        if kind == "stand_on_two":
            fwd = rot[:, 0]
            forward = np.asarray(consts.get("forward"), dtype=float)
            ang = float(np.degrees(np.arccos(np.clip(fwd @ forward, -1.0, 1.0))))
            air_down = any(
                self._pair_contact(scene, f, self._pg.get("floor_geom", "floor"))
                for f in (consts.get("air_feet") or [])
            )
            self._pose_log.append((ang, air_down))
            bad = up_z < -0.25 or self._any_contact(
                scene,
                self._pg.get("floor_geom", "floor"),
                set(consts.get("front_leg_geoms") or []),
            )
            if consts.get("hind_leg_contact") and not bad:
                bad = self._any_contact(
                    scene,
                    self._pg.get("floor_geom", "floor"),
                    set(consts.get("hind_leg_geoms") or []),
                )
            if bad and self._fell_at is None:
                self._fell_at = self._steps
        elif self._fell_at is None and self._terminated(scene, up_z):
            self._fell_at = self._steps

        gait = self._pg.get("gait")
        if gait:
            ph = (
                np.fmod(
                    np.asarray(gait["phase0"], dtype=float)
                    + (self._steps - 1) * float(gait["phase_dt"])
                    + np.pi,
                    2 * np.pi,
                )
                - np.pi
            )
            feet_z = []
            for foot in self._pg.get("feet") or []:
                fpos, _ = self._site_pose(scene, foot)
                feet_z.append(fpos[2])
            err = np.asarray(feet_z, dtype=float) - self._gait_rz(
                ph, float(gait["foot_height_m"])
            )
            self._gait_log.append(float(np.sqrt(np.mean(err**2))))

    def _terminated(self, scene: Any, up_z: float) -> bool:
        """Playground's joystick termination (rcb_quadruped.terminated)."""
        consts = self._pg.get("constants") or {}
        if up_z < float(consts.get("up_z_min", 0.0)):
            return True
        joint_box = consts.get("joint_box")
        if joint_box:
            lo = np.tile(np.asarray(joint_box[0], dtype=float), 4)
            hi = np.tile(np.asarray(joint_box[1], dtype=float), 4)
            qpos = np.asarray(self._world(scene).get_qpos(), dtype=float)
            if np.any(qpos[7:] < lo) or np.any(qpos[7:] > hi):
                return True
        torso_min_z = consts.get("torso_min_z")
        if torso_min_z is not None:
            root_z = float(self._pos(scene, self._pg["root_body"])[2])
            if root_z < float(torso_min_z):
                return True
        return False

    def _quadruped_check(self) -> dict:
        """rcb_quadruped.Session.check on the steps executed so far."""
        consts = self._pg.get("constants") or {}
        horizon = int(self._pg.get("horizon", 0))
        kind = self._pg.get("task_kind")
        log_arr = np.array(self._log).reshape(-1, 5)
        out: dict[str, Any] = {
            "steps": self._steps,
            "horizon": horizon,
            "fell_at": self._fell_at,
        }
        if kind == "joystick":
            segs: list[dict[str, Any]] = []
            commands = self._pg.get("commands") or []
            for i, s in enumerate(commands):
                a = int(s["start_step"]) + int(consts.get("settle_steps", 25))
                b = min(
                    (
                        int(commands[i + 1]["start_step"])
                        if i + 1 < len(commands)
                        else horizon
                    ),
                    self._steps,
                )
                if b > a:
                    segs.append(
                        {
                            "segment": i,
                            "lin_err": round(float(log_arr[a:b, 0].mean()), 4),
                            "yaw_err": round(float(log_arr[a:b, 1].mean()), 4),
                        }
                    )
            gait = self._pg.get("gait")
            if gait:
                gait_arr = np.asarray(self._gait_log)
                for g in segs:
                    s0 = commands[g["segment"]]
                    a = int(s0["start_step"]) + int(consts.get("settle_steps", 25))
                    nxt = (
                        commands[g["segment"] + 1]["start_step"]
                        if g["segment"] + 1 < len(commands)
                        else horizon
                    )
                    g["gait_err"] = round(
                        float(gait_arr[a : min(int(nxt), self._steps)].mean()), 4
                    )
            ok_track = all(
                g["lin_err"] <= float(consts.get("lin_tol", 0.2))
                and g["yaw_err"] <= float(consts.get("yaw_tol", 0.3))
                and g.get("gait_err", 0.0) <= float(consts.get("gait_tol", 0.05))
                for g in segs
            )
            out.update(
                success=bool(
                    self._steps == horizon and self._fell_at is None and ok_track
                ),
                tracking_ok=bool(ok_track),
                segments=segs,
            )
        elif kind == "stand_on_two":
            pose_arr = np.array(self._pose_log, dtype=float).reshape(-1, 2)
            n = int(consts.get("pose_hold_steps", 100))
            held = (
                len(pose_arr) >= n
                and self._fell_at is None
                and bool(
                    np.all(pose_arr[-n:, 0] <= float(consts.get("fwd_tol_deg", 30.0)))
                )
                and bool(
                    np.all(log_arr[-n:, 3] >= float(consts.get("min_imu_height", 0.40)))
                )
                and not bool(pose_arr[-n:, 1].any())
            )
            out.update(success=bool(held))
        else:  # getup
            n = int(consts.get("hold_steps", 25))
            last = log_arr[-n:]
            held = (
                len(last) == n
                and bool(np.all(last[:, 2] <= float(consts.get("tilt_max_deg", 10.0))))
                and bool(
                    np.all(last[:, 3] >= float(consts.get("min_imu_height", 0.22)))
                )
            )
            out.update(success=bool(held))
        return out

    # -- manipulation checks (rcb_aloha / rcb_panda, verbatim) ---------------

    def _manip_check(self, scene: Any) -> dict:
        kind = self._pg.get("task_kind")
        check = self._pg.get("check") or {}
        if kind == "AlohaSinglePegInsertion":
            rear, _ = self._site_pose(scene, check["sites"]["socket_rear"])
            ent, _ = self._site_pose(scene, check["sites"]["socket_entrance"])
            tip, _ = self._site_pose(scene, check["sites"]["peg_tip"])
            ab = ent - rear
            t = float(np.dot(tip - rear, ab) / (np.dot(ab, ab) + 1e-12))
            dist = float(np.linalg.norm(tip - (rear + t * ab)))
            inserted = dist < float(check["line_tol_m"]) and float(
                check["t_min"]
            ) <= t <= float(check["depth_t_max"])
            return {
                "success": bool(inserted),
                "tip_to_axis_m": round(dist, 5),
                "axial_t": round(t, 4),
            }
        if kind == "AlohaHandOver":
            box = self._pos(scene, check["box_body"])
            target = np.asarray(check["target_pos"], dtype=float)
            dist = float(np.linalg.norm(box - target))
            right = self._any_contact(
                scene, check["box_geom"], set(check.get("right_finger_geoms") or [])
            )
            left = self._any_contact(
                scene, check["box_geom"], set(check.get("left_geoms") or [])
            )
            ok = (
                dist < float(check["tol_m"])
                and right
                and not left
                and box[2] >= float(check["min_z"])
            )
            return {
                "success": bool(ok),
                "box_to_target_m": round(dist, 4),
                "right_holds": bool(right),
                "left_touches": bool(left),
                "box_z": round(float(box[2]), 4),
            }
        obj = self._pos(scene, self._pg.get("object_body", "box"))
        target = np.asarray(self._pg.get("target_pos"), dtype=float)
        if kind == "PandaPickCubeOrientation":
            err = float(np.linalg.norm(obj - target))
            obj_quat = self._quat(scene, self._pg.get("object_body", "box"))
            tgt_quat = np.asarray(self._pg.get("target_quat_wxyz"), dtype=float)
            d = abs(
                float(
                    np.dot(
                        obj_quat / np.linalg.norm(obj_quat),
                        tgt_quat / np.linalg.norm(tgt_quat),
                    )
                )
            )
            rot = float(np.degrees(2 * np.arccos(min(1.0, d))))
            ok = err < float(check["pos_tol_m"]) and rot < float(check["rot_tol_deg"])
            return {
                "success": bool(ok),
                "box_to_target_m": round(err, 4),
                "rot_err_deg": round(rot, 2),
            }
        err = float(abs(obj[0] - target[0]))
        return {
            "success": bool(err < float(check["tol_m"])),
            "handle_x": round(float(obj[0]), 4),
        }

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._apply_frozen_state(scene)
        self._steps = 0
        self._fell_at = None
        self._log = []
        self._gait_log = []
        self._pose_log = []
        self._servo_warned = False
        info["pg"] = {"success": False, "steps": 0}
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        self._apply_targets(scene, action)
        self._steps += 1
        if self._pg.get("kind_group") == "quadruped":
            self._quadruped_log(scene)
            info["pg"] = self._quadruped_check()
        else:
            info["pg"] = self._manip_check(scene)
        return reward, terminated, truncated, info


@register_task("efe_pg")
def _make_efe_pg(**kwargs: Any) -> EfePgTask:
    """MuJoCo Playground frozen-scene task with world/pg YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 500))
    world = kwargs.pop("world", None)
    pg = kwargs.pop("pg", None)
    return EfePgTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        pg=pg,
    )


@register_robot("aloha_2")
def _make_aloha_2(**kwargs: Any) -> Any:
    """Mobile ALOHA 2 (VX300S bimanual, MuJoCo Playground, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("aloha_2")


@register_robot("unitree_go1")
def _make_unitree_go1(**kwargs: Any) -> Any:
    """Unitree Go1 quadruped (MuJoCo Playground, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("unitree_go1")


@register_robot("boston_dynamics_spot")
def _make_boston_dynamics_spot(**kwargs: Any) -> Any:
    """Boston Dynamics Spot quadruped (MuJoCo Playground, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("boston_dynamics_spot")


@register_robot("google_barkour_vb")
def _make_google_barkour_vb(**kwargs: Any) -> Any:
    """Google Barkour vB quadruped (MuJoCo Playground, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("google_barkour_vb")


@register_robot("franka_panda_embedded")
def _make_franka_panda_embedded(**kwargs: Any) -> Any:
    """Franka Panda inside a whole-world MJCF (MuJoCo Playground, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("franka_panda_embedded")


def _hb_tolerance(
    x: Any,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> Any:
    """Verbatim port of dm_control.utils.rewards.tolerance (rewards.py).

    Returns 1 inside the inclusive bounds, decreasing sigmoidally with the
    distance to the nearest bound outside them. Used by every HumanoidBench
    reward term.
    """
    x = np.asarray(x, dtype=float)
    lower, upper = bounds
    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        if sigmoid == "gaussian":
            scale = float(np.sqrt(-2 * np.log(value_at_margin)))
            s = np.exp(-0.5 * (d * scale) ** 2)
        elif sigmoid == "linear":
            scaled = d * (1 - value_at_margin)
            s = np.where(np.abs(scaled) < 1, 1 - scaled, 0.0)
        elif sigmoid == "quadratic":
            scaled = d * float(np.sqrt(1 - value_at_margin))
            s = np.where(np.abs(scaled) < 1, 1 - scaled**2, 0.0)
        else:  # pragma: no cover - the ported tasks only use the three above
            raise ValueError(f"hb_tolerance: unported sigmoid {sigmoid!r}")
        value = np.where(in_bounds, 1.0, s)
    return float(value) if value.ndim == 0 else value


class EfeHbTask(EfeBaseTask):
    """HumanoidBench frozen-scene task (Unitree H1 / H1+Shadow hands).

    The task YAML carries the composed world (``world`` block, see
    ``tools/hb_worlds.py``) and an ``hb`` block with the frozen-instance
    constants (success bar, episode cap, termination spec, class/module
    constants, task state, site table, robot link/mass table). Success is
    HumanoidBench's own rule: the episode's summed dense reward reaches the
    task's success bar (``episode_return >= success_bar``), with the episode
    ending on the task's ported ``get_terminated``. The ten reward functions
    (``rewards.json`` / upstream ``envs/*.py``) are ported verbatim at
    control-step granularity, including the dm_control ``rewards.tolerance``
    semantics, and accumulated as ``reward_sum``; ``info['hb']['success']``
    is published each step and the outer :class:`HbSuccessCondition`
    forwards it (missing info is a failure, nop-safe).

    Sensor reads map onto the Genesis world entity: head/IMU/hand sites via
    the recorded local poses, actuator forces via per-DoF actuator forces,
    centre-of-mass velocity as the mass-weighted link-COM velocity over the
    robot's links. Controls are best-effort: normalised [-1, 1] rows are
    unnormalised to the actuator ranges and servoed with the entity's
    dof-position control (same contract as ``EfeAbcTask``).
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        hb: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_hb world must be a mapping or None")
        if hb is not None and not isinstance(hb, dict):
            raise TypeError("efe_hb hb must be a mapping or None")
        self._world_cfg = world or {}
        self._hb = hb or {}
        self._servo_warned = False
        self._ret = 0.0
        self._hb_steps = 0
        self._terminated = False
        self._task_index = 0
        self._link_idx: dict[str, int] = {}
        self._ts: dict[str, Any] = {}

    @staticmethod
    def _decode_state(value: Any) -> Any:
        """Undo the hb_worlds __ndarray__ JSON tagging."""
        if isinstance(value, dict) and "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=value.get("dtype", float))
        if isinstance(value, list):
            return [EfeHbTask._decode_state(v) for v in value]
        return value

    # -- world access -------------------------------------------------------

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_hb expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    # -- accessors -----------------------------------------------------------

    def _qpos(self, scene: Any) -> np.ndarray:
        return np.asarray(self._world(scene).get_qpos(), dtype=float)

    def _body_pos(self, scene: Any, name: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_pos

        return _abc_pos(scene, name)

    def _body_quat(self, scene: Any, name: str) -> np.ndarray:
        from cloud_robotics_sim.core.success import _abc_quat

        return _abc_quat(scene, name)

    def _site_pos(self, scene: Any, name: str) -> np.ndarray:
        record = (self._hb.get("sites") or {})[name]
        body_pos = self._body_pos(scene, record["body"])
        body_quat = self._body_quat(scene, record["body"])
        from cloud_robotics_sim.core.success import _quat_to_mat

        offset = np.asarray(record["local"], dtype=float)
        return np.asarray(body_pos + _quat_to_mat(body_quat) @ offset, dtype=float)

    def _entity(self, scene: Any) -> Any:
        resolver = getattr(self._world(scene), "_resolve_entity", None)
        return None if resolver is None else resolver()

    def _link_indices(self, scene: Any, names: list[str]) -> list[int]:
        entity = self._entity(scene)
        if entity is None or not hasattr(entity, "links"):
            return []
        if not self._link_idx:
            self._link_idx = {
                link.name: i for i, link in enumerate(entity.links) if link.name
            }
        return [self._link_idx[n] for n in names if n in self._link_idx]

    def _subtree_linvel(self, scene: Any, names: list[str]) -> np.ndarray:
        """MuJoCo ``subtree_linvel`` over the named links (verbatim formula).

        mj_subtreeVel expresses each body's com-based velocity at the
        subtree's com (mj_objectVelocity: ``v_at = v_com + w x (xipos -
        subtree_com)``) and mass-averages; for a single-body subtree this
        reduces to the link's com velocity (window tool case).
        """
        import genesis as gs

        entity = self._entity(scene)
        idx = self._link_indices(scene, names)
        if entity is None or not idx:
            return np.zeros(3)
        vel = np.asarray(
            entity.get_links_vel(links_idx_local=idx, ref=gs.link_ref_frame.link_COM),
            dtype=float,
        ).reshape(len(idx), 3)
        com = np.asarray(
            entity.get_links_COM(links_idx_local=idx), dtype=float
        ).reshape(len(idx), 3)
        ang = np.asarray(
            entity.get_links_ang(links_idx_local=idx), dtype=float
        ).reshape(len(idx), 3)
        mass = np.asarray(entity.get_links_mass(links_idx_local=idx), dtype=float)
        total = float(mass.sum())
        if total <= 0.0:
            return np.zeros(3)
        com_sub = (com * mass[:, None]).sum(axis=0) / total
        v_at = vel + np.cross(ang, com - com_sub[None, :])
        linvel = (mass[:, None] * v_at).sum(axis=0) / total
        return np.asarray(linvel, dtype=float)

    def _actuator_forces(self, scene: Any) -> np.ndarray:
        """Upstream per-actuator forces, recovered from the dof forces.

        Upstream reads ``data.actuator_force`` (nu,). Tendon-routed hands
        map one actuator to several dofs, so the dof forces are the moment
        matrix times the actuator forces; the moment rows recorded in the
        world pack make the recovery an exact lstsq while the mapping has
        full column rank (it does for the H1 / H1+Shadow models).
        """
        entity = self._entity(scene)
        actuated = self._hb.get("actuated") or []
        if entity is None or not hasattr(entity, "get_dofs_actuator_force"):
            return np.zeros(int(self._hb.get("nu", 0)))
        qfrc = np.asarray(entity.get_dofs_actuator_force(), dtype=float)
        if not actuated:
            return qfrc.reshape(-1)
        nv = int(qfrc.size)
        moment = np.zeros((nv, len(actuated)), dtype=float)
        for col, record in enumerate(actuated):
            for dof, value in record.get("moment") or []:
                if 0 <= int(dof) < nv:
                    moment[int(dof), col] = float(value)
        try:
            recovered, *_ = np.linalg.lstsq(moment, qfrc, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - fall back to dof slice
            adrs = [int(a["dofadr"]) for a in actuated]
            return qfrc[adrs] if qfrc.size > max(adrs) else qfrc
        return np.asarray(recovered, dtype=float)

    # -- robot helpers (humanoid_bench/robots.py) ----------------------------

    def _head_height(self, scene: Any) -> float:
        return float(self._site_pos(scene, "head")[2])

    def _torso_upright(self, scene: Any) -> float:
        from cloud_robotics_sim.core.success import _quat_to_mat

        quat = self._body_quat(scene, "torso_link")
        return float(_quat_to_mat(quat)[2, 2])

    def _com_velocity(self, scene: Any) -> np.ndarray:
        """Robot pelvis-subtree linear velocity (the pelvis_subtreelinvel sensor)."""
        names = [r["name"] for r in self._hb.get("robot_links") or []]
        return self._subtree_linvel(scene, names)

    # -- shared terms --------------------------------------------------------

    def _stand_height(self) -> float:
        for expr in (self._hb.get("module_globals") or {}).values():
            if "_STAND_HEIGHT" in str(expr):
                try:
                    return float(str(expr).split("=")[1].strip())
                except (IndexError, ValueError):
                    break
        return 1.65

    def _standing(self, scene: Any, margin_div: float = 4.0) -> Any:
        h = self._stand_height()
        return _hb_tolerance(
            self._head_height(scene), bounds=(h, float("inf")), margin=h / margin_div
        )

    def _upright(
        self,
        scene: Any,
        bound: float,
        margin: float,
        value_at_margin: float = 0.0,
    ) -> Any:
        return _hb_tolerance(
            self._torso_upright(scene),
            bounds=(bound, float("inf")),
            sigmoid="linear",
            margin=margin,
            value_at_margin=value_at_margin,
        )

    def _small_control(self, scene: Any) -> Any:
        t = _hb_tolerance(
            self._actuator_forces(scene),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        return (4 + t) / 5

    def _dont_move(self, scene: Any) -> Any:
        v = self._com_velocity(scene)[[0, 1]]
        return _hb_tolerance(v, margin=2).mean()

    def _pair_contact(self, scene: Any, geom_a: str, geom_b: str) -> bool:
        world = self._world(scene)
        contacts = world.get_self_contacts()
        geom_names = world.geom_names()
        link_names = world.link_names()
        ga = np.asarray(contacts.get("geom_a", [])).reshape(-1)
        gb = np.asarray(contacts.get("geom_b", [])).reshape(-1)
        la = np.asarray(contacts.get("link_a", [])).reshape(-1)
        lb = np.asarray(contacts.get("link_b", [])).reshape(-1)
        for i in range(la.size):
            a = geom_names[ga[i]] if i < ga.size else link_names[la[i]]
            b = geom_names[gb[i]] if i < gb.size else link_names[lb[i]]
            if (a == geom_a and b == geom_b) or (a == geom_b and b == geom_a):
                return True
        return False

    def _any_geom_contact(self, scene: Any, geom: str) -> bool:
        world = self._world(scene)
        contacts = world.get_self_contacts()
        geom_names = world.geom_names()
        link_names = world.link_names()
        ga = np.asarray(contacts.get("geom_a", [])).reshape(-1)
        gb = np.asarray(contacts.get("geom_b", [])).reshape(-1)
        la = np.asarray(contacts.get("link_a", [])).reshape(-1)
        lb = np.asarray(contacts.get("link_b", [])).reshape(-1)
        for i in range(la.size):
            a = geom_names[ga[i]] if i < ga.size else link_names[la[i]]
            b = geom_names[gb[i]] if i < gb.size else link_names[lb[i]]
            if a == geom or b == geom:
                return True
        return False

    # -- reward ports (rewards.json get_reward, verbatim formulas) -----------

    def _reward(self, scene: Any) -> tuple[float, dict]:
        short = self._hb.get("task_short")
        method = getattr(self, f"_reward_{short}", None)
        if method is None:  # pragma: no cover - resolver binds only the 10
            raise KeyError(f"efe_hb: unported task {short!r}")
        reward, detail = method(scene)
        return float(reward), detail

    def _reward_walk(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        move_speed = float((self._hb.get("class_consts") or {}).get("_move_speed", 1))
        if move_speed == 0:
            dont_move = self._dont_move(scene)
            return small_control * stand_reward * dont_move, {
                "standing": standing,
                "upright": upright,
                "dont_move": dont_move,
            }
        move = _hb_tolerance(
            self._com_velocity(scene)[0],
            bounds=(move_speed, float("inf")),
            margin=move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return small_control * stand_reward * move, {
            "standing": standing,
            "upright": upright,
            "move": move,
        }

    def _reward_hurdle(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.8, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        move_speed = float((self._hb.get("class_consts") or {}).get("_move_speed", 5))
        move = _hb_tolerance(
            self._com_velocity(scene)[0],
            bounds=(move_speed, float("inf")),
            margin=move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        discount = 1.0
        for wall in (
            "left_barrier_collision",
            "right_barrier_collision",
            "behind_barrier_collision",
        ):
            if self._any_geom_contact(scene, wall):
                discount = 0.1
                break
        return small_control * stand_reward * move * discount, {
            "standing": standing,
            "upright": upright,
            "move": move,
            "wall_collision_discount": discount,
        }

    def _reward_balance_hard(self, scene: Any) -> tuple[float, dict]:
        h = self._stand_height()
        standing = _hb_tolerance(
            self._head_height(scene),
            bounds=(h + 0.37, float("inf")),
            margin=h / 4,
        )
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        dont_move = self._dont_move(scene)
        return small_control * stand_reward * dont_move, {
            "standing": standing,
            "upright": upright,
            "dont_move": dont_move,
        }

    def _reward_sit_simple(self, scene: Any) -> tuple[float, dict]:
        qpos = self._qpos(scene)
        sitting = _hb_tolerance(qpos[2], bounds=(0.68, 0.72), margin=0.2)
        chair = self._body_pos(scene, "chair")
        on_chair = _hb_tolerance(
            qpos[0] - chair[0], bounds=(-0.19, 0.19), margin=0.2
        ) * _hb_tolerance(qpos[1] - chair[1], margin=0.1)
        sitting_posture = _hb_tolerance(
            self._head_height(scene) - self._site_pos(scene, "imu")[2],
            bounds=(0.35, 0.45),
            margin=0.3,
        )
        upright = self._upright(scene, 0.95, 0.9, 0.0)
        sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
        small_control = self._small_control(scene)
        dont_move = self._dont_move(scene)
        return small_control * sit_reward * dont_move, {
            "sitting": sitting,
            "on_chair": on_chair,
            "sitting_posture": sitting_posture,
            "upright": upright,
        }

    def _reward_door(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 0.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        qpos = self._qpos(scene)
        door_openness = min(1.0, (qpos[-2] / 1.0) * abs(qpos[-2] / 1.0))
        door_hatch = _hb_tolerance(
            qpos[-1], bounds=(0.75, 2), margin=0.75, sigmoid="linear"
        )
        hatch_pos = self._body_pos(scene, "door_hatch")
        left_d = float(np.linalg.norm(hatch_pos - self._site_pos(scene, "left_hand")))
        right_d = float(np.linalg.norm(hatch_pos - self._site_pos(scene, "right_hand")))
        proximity = _hb_tolerance(
            min(right_d, left_d), bounds=(0, 0.25), margin=1, sigmoid="linear"
        )
        passage = _hb_tolerance(
            self._site_pos(scene, "imu")[0],
            bounds=(1.2, float("inf")),
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )
        reward = (
            0.1 * stand_reward * small_control
            + 0.45 * door_openness
            + 0.05 * door_hatch
            + 0.05 * proximity
            + 0.35 * passage
        )
        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "door_openness": door_openness,
            "passage": passage,
        }

    def _reward_cube(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        dont_move = self._dont_move(scene)
        target_quat = self._body_quat(scene, "target_cube")
        orientations = []
        for name in ("left_cube_to_rotate", "right_cube_to_rotate"):
            quat = self._body_quat(scene, name)
            orientations.append(
                _hb_tolerance(np.linalg.norm(quat - target_quat), margin=0.3)
            )
        orientation_reward = (orientations[0] + orientations[1]) / 2
        proximities = []
        for name in ("left_cube_to_rotate", "right_cube_to_rotate"):
            site = "left_hand" if name.startswith("left") else "right_hand"
            dist = float(
                np.linalg.norm(
                    self._site_pos(scene, site) - self._body_pos(scene, name)
                )
            )
            proximities.append(_hb_tolerance(dist, bounds=(0, 0.1), margin=0.5))
        closeness = (proximities[0] + proximities[1]) / 2
        reward = (
            0.2 * (small_control * stand_reward * dont_move)
            + 0.5 * orientation_reward
            + 0.3 * closeness
        )
        return reward, {
            "stand_reward": stand_reward,
            "orientation_alignment_reward": orientation_reward,
            "cube_closeness_reward": closeness,
        }

    def _reward_powerlift(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene, margin_div=3.0)
        upright = self._upright(scene, 0.8, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        dumbbell_h = self._body_pos(scene, "dumbbell")[2]
        lifted = _hb_tolerance(dumbbell_h, bounds=(1.9, 2.1), margin=2)
        return 0.2 * (small_control * stand_reward) + 0.8 * lifted, {
            "stand_reward": stand_reward,
            "reward_dumbbell_lifted": lifted,
        }

    def _reward_window(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        contact_reward = float(
            min(
                _hb_tolerance(
                    self._site_pos(scene, f"wipe_contact_site_{s}")[0],
                    bounds=(0.92, 0.92),
                    margin=0.4,
                    sigmoid="linear",
                )
                for s in "abcde"
            )
        )
        contact_filter = (
            1.0
            if self._pair_contact(
                scene, "window_pane_collision", "window_wipe_collision"
            )
            else 0.0
        )
        tool_pos = self._body_pos(scene, "window_wiping_tool")
        left_d = float(np.linalg.norm(self._site_pos(scene, "left_hand") - tool_pos))
        right_d = float(np.linalg.norm(self._site_pos(scene, "right_hand") - tool_pos))
        hand_tool = min(
            _hb_tolerance(left_d, bounds=(0, 0.2), margin=0.5),
            _hb_tolerance(right_d, bounds=(0, 0.2), margin=0.5),
        )
        tool_names = self._hb.get("tool_subtree") or ["window_wiping_tool"]
        tool_vel_z = float(self._subtree_linvel(scene, list(tool_names))[2])
        moving_wipe = _hb_tolerance(abs(tool_vel_z), bounds=(0.5, 0.5), margin=0.5)
        head_pos0 = np.asarray(self._ts.get("head_pos0", [0, 0, 0]), dtype=float)
        head_dist = _hb_tolerance(
            float(np.linalg.norm(self._site_pos(scene, "head") - head_pos0)),
            bounds=(0.4, 0.4),
            margin=0.1,
        )
        manipulation = (
            0.2 * (stand_reward * small_control * head_dist)
            + 0.4 * moving_wipe
            + 0.4 * hand_tool
        )
        total = 0.5 * manipulation + 0.5 * (contact_filter * contact_reward)
        return total, {
            "stand_reward": stand_reward,
            "moving_wipe_reward": moving_wipe,
            "hand_tool_proximity_reward": hand_tool,
            "window_contact_total_reward": contact_filter * contact_reward,
        }

    def _reward_room(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        positions = np.vstack(
            [self._body_pos(scene, n) for n in self._hb.get("room_objects") or []]
        )
        entropies = np.array([float(np.var(positions[:, col])) for col in range(2)])
        organized = _hb_tolerance(np.max(entropies), margin=3)
        return 0.2 * (small_control * stand_reward) + 0.8 * organized, {
            "stand_reward": stand_reward,
            "room_object_organized": organized,
        }

    def _reward_bookshelf_simple(self, scene: Any) -> tuple[float, dict]:
        standing = self._standing(scene)
        upright = self._upright(scene, 0.9, 1.9, 0.0)
        stand_reward = standing * upright
        small_control = self._small_control(scene)
        objects = self._ts.get("bookshelf_objects") or []
        goals = self._ts.get("placement_goals") or []
        obj_pos = self._body_pos(scene, objects[self._task_index])
        goal = np.asarray(goals[self._task_index], dtype=float)
        dist = float(np.linalg.norm(obj_pos - goal))
        proximity = _hb_tolerance(dist, bounds=(0, 0.15), margin=1, sigmoid="linear")
        left_d = float(np.linalg.norm(self._site_pos(scene, "left_hand") - obj_pos))
        right_d = float(np.linalg.norm(self._site_pos(scene, "right_hand") - obj_pos))
        hand_proximity = float(np.exp(-min(left_d, right_d)))
        reward = (
            0.2 * (stand_reward * small_control)
            + 0.4 * proximity
            + 0.4 * hand_proximity
        )
        if dist < 0.15:
            self._task_index += 1
            reward += 100 * self._task_index
        return reward, {
            "stand_reward": stand_reward,
            "reward_proximity": proximity,
            "reward_hand_proximity": hand_proximity,
            "success_subtasks": self._task_index,
        }

    # -- termination ports ----------------------------------------------------

    def _check_terminated(self, scene: Any) -> bool:
        spec = self._hb.get("termination") or {}
        kind = spec.get("kind")
        if kind in (None, "never"):
            return False
        qpos_z = float(self._qpos(scene)[2])
        threshold = spec.get("qpos_z_lt")
        if threshold is not None and qpos_z < float(threshold):
            return True
        if kind == "balance_hard":
            # upstream: geom id 0 is the floor; any contact between the
            # pivot sphere and something that is neither floor nor board,
            # or between the floor and anything but the sphere, terminates
            world = self._world(scene)
            names = world.geom_names()
            try:
                sphere_id = names.index(str(spec.get("sphere_geom")))
                board_id = names.index(str(spec.get("board_geom")))
            except ValueError:
                return False
            contacts = world.get_self_contacts()
            ga = np.asarray(contacts.get("geom_a", [])).reshape(-1)
            gb = np.asarray(contacts.get("geom_b", [])).reshape(-1)
            floor_id = 0
            for i in range(min(ga.size, gb.size)):
                pair = {int(ga[i]), int(gb[i])}
                if sphere_id in pair and not ({floor_id, board_id} & pair):
                    return True
                if floor_id in pair and sphere_id not in pair:
                    return True
            return False
        if kind == "cube":
            for name in ("left_cube_to_rotate", "right_cube_to_rotate"):
                if self._body_pos(scene, name)[2] < float(spec["min_obj_z_lt"]):
                    return True
            return False
        if kind == "window":
            return bool(
                self._body_pos(scene, "window_wiping_tool")[2]
                < float(spec["tool_z_lt"])
            )
        if kind == "bookshelf":
            if self._task_index == 5:
                return True
            objects = self._ts.get("bookshelf_objects") or []
            return bool(
                self._body_pos(scene, objects[self._task_index])[2]
                < float(spec["min_obj_z_lt"])
            )
        return False

    # -- controls -------------------------------------------------------------

    def _apply_targets(self, scene: Any, action: np.ndarray) -> None:
        """Unnormalise [-1, 1] rows to actuator targets and servo (best effort)."""
        actuated = self._hb.get("actuated") or []
        if not actuated or action is None or len(action) < len(actuated):
            return
        rows = np.clip(
            np.asarray(action, dtype=float).reshape(-1)[: len(actuated)], -1.0, 1.0
        )
        targets = np.array(
            [
                (r + 1) / 2 * (hi - lo) + lo
                for r, (lo, hi) in (
                    (rows[i], a["ctrlrange"]) for i, a in enumerate(actuated)
                )
            ],
            dtype=float,
        )
        world = self._world(scene)
        control_fn = getattr(world, "control_dofs_position", None)
        if control_fn is None:
            if not self._servo_warned:
                logger.info("efe_hb: no dof position control; actions not applied")
                self._servo_warned = True
            return
        try:
            control_fn(
                targets=targets,
                dofs_idx_local=[int(a["dofadr"]) for a in actuated],
            )
        except Exception as exc:  # noqa: BLE001 - servo is best effort
            if not self._servo_warned:
                logger.info(f"efe_hb: dof position control unavailable: {exc}")
                self._servo_warned = True

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._apply_frozen_state(scene)
        self._link_idx = {}
        self._ret = 0.0
        self._hb_steps = 0
        self._terminated = False
        state = self._hb.get("task_state") or {}
        self._ts = {k: self._decode_state(v) for k, v in state.items()}
        self._task_index = int(self._ts.get("task_index", 0) or 0)
        info["hb"] = {"success": False, "reward_sum": 0.0, "steps": 0}
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        self._apply_targets(scene, action)
        self._hb_steps += 1
        step_reward, detail = self._reward(scene)
        self._ret += step_reward
        terminated = self._check_terminated(scene)
        self._terminated = bool(terminated)
        info["hb"] = {
            "success": bool(
                self._ret >= float(self._hb.get("success_bar", float("inf")))
            ),
            "reward_sum": float(self._ret),
            "terminated": self._terminated,
            "steps": self._hb_steps,
            **detail,
        }
        return float(step_reward), terminated, truncated, info


@register_task("efe_hb")
def _make_efe_hb(**kwargs: Any) -> EfeHbTask:
    """HumanoidBench frozen-scene task with world/hb YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 1000))
    world = kwargs.pop("world", None)
    hb = kwargs.pop("hb", None)
    return EfeHbTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        hb=hb,
    )


@register_robot("unitree_h1_embedded")
def _make_unitree_h1_embedded(**kwargs: Any) -> Any:
    """Unitree H1 inside a whole-world MJCF (HumanoidBench, world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("unitree_h1_embedded")


@register_robot("unitree_h1hand_embedded")
def _make_unitree_h1hand_embedded(**kwargs: Any) -> Any:
    """Unitree H1 + Shadow hands inside a whole-world MJCF (world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("unitree_h1hand_embedded")


def _rw_box_aabb(
    position: np.ndarray, quat: np.ndarray, size: np.ndarray
) -> np.ndarray:
    """World AABB of a box at (pos, wxyz quat): 8 corners -> min/max."""
    from cloud_robotics_sim.core.success import _quat_to_mat

    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    corners = np.asarray(position) + signs * half @ _quat_to_mat(quat).T
    return np.array([corners.min(axis=0), corners.max(axis=0)])


def _rw_footprint_poly(
    position: np.ndarray, quat: np.ndarray, size: np.ndarray
) -> list[tuple[float, float]]:
    """2D footprint of a box: its 4 base corners projected to x-y."""
    from cloud_robotics_sim.core.success import _quat_to_mat

    half = np.asarray(size, dtype=float) / 2.0
    corners = np.array(
        [[sx * half[0], sy * half[1], -half[2]] for sx in (-1, 1) for sy in (-1, 1)]
    ) @ _quat_to_mat(quat).T + np.asarray(position)
    pts = [(float(c[0]), float(c[1])) for c in corners]
    # unique corners (a flat box projects to 2 points; keep all unique)
    seen: list[tuple[float, float]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12 for q in seen):
            seen.append(p)
    return seen


def _rw_poly_area(poly: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _rw_clip_polygon(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip (06_dominos.py, verbatim algorithm)."""
    if len(subject) < 3 or len(clip) < 3:
        return []

    def signed_area(poly: list[tuple[float, float]]) -> float:
        s = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    def line_intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return (px, py)

    ori_sign = 1.0 if signed_area(clip) >= 0.0 else -1.0

    def inside(
        pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        return ori_sign * cross >= -1e-9

    output = list(subject)
    cl = list(clip)
    for i in range(len(cl)):
        a = cl[i]
        b = cl[(i + 1) % len(cl)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(line_intersection(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(line_intersection(s, e, a, b))
            s = e
    return output


class EfeRwTask(EfeBaseTask):
    """RoboWits frozen-scene task (Tianji Marvin bimanual, Genesis upstream).

    The task YAML carries the composed world (``world`` block, see
    ``tools/rw_worlds.py``) and an ``rw`` block with the frozen-instance
    constants (object table, graded lists, settle-window constants, action
    layout). Success is the EFE grader's verdict (``rcb_robowits``): the
    family's ``_check_success`` predicate (ported verbatim from the RoboWits
    gs_gym sources) must hold, no robot part may touch a graded object, and
    the goal must persist through the settle window (SETTLE_STEPS hold with
    nothing checked, then CHECK_STEPS with checks every CHECK_STRIDE-th step,
    every graded object within REST_M of its window-start pose).

    The closed-loop runtime mirrors the open-loop settle replay: while the
    goal holds, a hold counter accumulates; success publishes once the counter
    reaches settle+check steps with the rest constraint satisfied inside the
    window (positions captured at window start). Predicates operate on object
    poses/sizes from the Genesis world entity; footprints are the 2D convex
    hulls of the boxes' projected corners (the ported families are all-box).
    Controls are best-effort: the 16-vector action is expanded to the 18 dofs
    (gripper width split equally between the two prismatic finger joints) and
    servoed with the entity's dof-position control.
    """

    def __init__(
        self,
        config: TaskConfig,
        world: dict | None = None,
        rw: dict | None = None,
    ):
        super().__init__(config)
        if world is not None and not isinstance(world, dict):
            raise TypeError("efe_rw world must be a mapping or None")
        if rw is not None and not isinstance(rw, dict):
            raise TypeError("efe_rw rw must be a mapping or None")
        self._world_cfg = world or {}
        self._rw = rw or {}
        self._servo_warned = False
        self._hold_count = 0
        self._window_pos: dict[str, np.ndarray] = {}
        self._rest_violation = False

    # -- world access -------------------------------------------------------

    def _world(self, scene: Any) -> Any:
        entities = getattr(scene, "entities", None) or {}
        entity = entities.get("world")
        if entity is None:
            raise KeyError(
                "efe_rw expects the 'world' asset in scene.entities; "
                f"available: {sorted(entities)}"
            )
        return entity

    def _apply_frozen_state(self, scene: Any) -> None:
        world = self._world(scene)
        qpos_path = self._world_cfg.get("qpos_path")
        qvel_path = self._world_cfg.get("qvel_path")
        if qpos_path:
            world.set_qpos(np.load(qpos_path))
        if qvel_path:
            set_qvel = getattr(world, "set_qvel", None)
            if set_qvel is not None:
                set_qvel(np.load(qvel_path))

    # -- accessors -----------------------------------------------------------

    def _object_state(
        self, scene: Any, name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, wxyz quat, size) of one frozen object."""
        entry = (self._rw.get("objects") or {})[name]
        from cloud_robotics_sim.core.success import _abc_pos, _abc_quat

        return (
            _abc_pos(scene, name),
            _abc_quat(scene, name),
            np.asarray(entry["size"], dtype=float),
        )

    def _object_aabb(self, scene: Any, name: str) -> np.ndarray:
        pos, quat, size = self._object_state(scene, name)
        return _rw_box_aabb(pos, quat, size)

    def _ee_pos(self, scene: Any, arm: str) -> np.ndarray:
        link = (self._rw.get("ee_links") or {}).get(arm)
        from cloud_robotics_sim.core.success import _abc_pos

        return _abc_pos(scene, str(link)) if link else np.zeros(3)

    def _being_held(self, scene: Any, name: str, tol: float = 0.03) -> bool:
        """Either gripper tip inside the object's AABB grown by `tol` (utils.check_being_held)."""
        bounds = self._object_aabb(scene, name)
        for arm in ("right", "left"):
            p = self._ee_pos(scene, arm)
            if np.all(p >= bounds[0] - tol) and np.all(p <= bounds[1] + tol):
                return True
        return False

    def _touching_robot(self, scene: Any, names: list[str]) -> list[str]:
        """Graded objects some robot link touches now (contact pairs)."""
        world = self._world(scene)
        contacts = world.get_self_contacts()
        link_names = world.link_names()
        la = np.asarray(contacts.get("link_a", [])).reshape(-1)
        lb = np.asarray(contacts.get("link_b", [])).reshape(-1)
        robot_joints = set(self._rw.get("robot_joints") or [])
        out = []
        for name in names:
            try:
                idx = link_names.index(name)
            except ValueError:
                continue
            for i in range(la.size):
                a, b = int(la[i]), int(lb[i])
                if (a == idx and link_names[b] not in (name,)) or (
                    b == idx and link_names[a] != name
                ):
                    other = link_names[a] if b == idx else link_names[b]
                    if other not in (name, "worktable", "world") and other not in [
                        n for n in (self._rw.get("objects") or {})
                    ]:
                        out.append(name)
                        break
        return sorted(set(out))

    # -- predicates (gs_gym _check_success ports) -----------------------------

    def _predicate(self, scene: Any) -> bool:
        family = self._rw.get("family")
        method = getattr(self, f"_predicate_{family}", None)
        if method is None:  # pragma: no cover - resolver binds only the 3
            raise KeyError(f"efe_rw: unported family {family!r}")
        return bool(method(scene))

    def _predicate_dominos(self, scene: Any) -> bool:
        """06_dominos._check_success: >= 50 % of the red block's footprint over the target."""
        red_pos, red_quat, red_size = self._object_state(scene, "red block")
        tg_pos, tg_quat, tg_size = self._object_state(scene, "target area")
        red_poly = _rw_footprint_poly(red_pos, red_quat, red_size)
        target_poly = _rw_footprint_poly(tg_pos, tg_quat, tg_size)
        if len(red_poly) < 3 or len(target_poly) < 3:
            return False
        inter = _rw_clip_polygon(red_poly, target_poly)
        red_area = _rw_poly_area(red_poly)
        if red_area <= 0.0:
            return False
        return _rw_poly_area(inter) / red_area >= 0.5

    def _predicate_stand_pages(self, scene: Any) -> bool:
        """07_stand_pages._check_success: both pages upright, on the table, in bounds, not held."""
        angle_tol = 15.0
        z_tol = 0.03
        bounds = self._rw.get("table_bounds") or {}

        def angle_close(angle_deg: float) -> bool:
            a = angle_deg
            if a < 0:
                a += 360.0
            if a > 360:
                a -= 360.0
            d = min(abs(a), abs(180 - a), abs(360 - a))
            return d <= angle_tol

        for name in ("page A with eyelets", "page B with eyelets"):
            pos, quat, size = self._object_state(scene, name)
            import math

            from cloud_robotics_sim.core.success import _quat_to_mat

            r = _quat_to_mat(quat)
            ex = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            ey = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
            if not (angle_close(ex) and angle_close(ey)):
                return False
            hull = _rw_footprint_poly(pos, quat, size)
            for x, y in hull:
                if not (
                    float(bounds.get("x", [0.21, 1.0])[0])
                    <= x
                    <= float(bounds.get("x", [0.21, 1.0])[1])
                    and float(bounds.get("y", [-0.69, 0.69])[0])
                    <= y
                    <= float(bounds.get("y", [-0.69, 0.69])[1])
                ):
                    return False
            aabb = _rw_box_aabb(pos, quat, size)
            if abs(float(aabb[0][2]) - float(self._rw.get("table_z", 0.76))) > z_tol:
                return False
            if self._being_held(scene, name):
                return False
        return True

    def _predicate_stack_cubes(self, scene: Any) -> bool:
        """14_stack_cubes._check_success: bases on the table touching, apex on a base, over the dot, not held."""
        table_z = float(self._rw.get("table_z", 0.76))
        aabb = {
            n: self._object_aabb(scene, n)
            for n in ("base cube 1", "base cube 2", "apex cube")
        }
        apex_pos, _, _ = self._object_state(scene, "apex cube")
        target_pos, _, _ = self._object_state(scene, "apex target")

        def on_table(b: np.ndarray, tol: float = 0.015) -> bool:
            return abs(float(b[0][2]) - table_z) <= tol

        def touching(b1: np.ndarray, b2: np.ndarray, tol: float = 0.01) -> bool:
            return all(
                not (b1[1][i] + tol < b2[0][i] or b2[1][i] + tol < b1[0][i])
                for i in range(3)
            )

        if not on_table(aabb["base cube 1"]) or not on_table(aabb["base cube 2"]):
            return False
        if not touching(aabb["base cube 1"], aabb["base cube 2"]):
            return False
        if float(aabb["apex cube"][0][2]) < float(aabb["base cube 1"][1][2]) - 0.01:
            return False
        dist = float(np.linalg.norm(apex_pos[:2] - np.asarray(target_pos)[:2]))
        if dist > 0.025:
            return False
        if self._being_held(scene, "apex cube"):
            return False
        return True

    # -- settle-window verdict -------------------------------------------------

    def _goal(self, scene: Any) -> bool:
        graded = self._rw.get("graded") or {}
        rest = list(graded.get("rest") or [])
        release_also = list(graded.get("release_also") or [])
        if not self._predicate(scene):
            return False
        touching = self._touching_robot(scene, rest + release_also)
        return not touching

    def _apply_targets(self, scene: Any, action: np.ndarray) -> None:
        """Expand the 16-vector to 18 dofs and servo (best effort)."""
        joints = self._rw.get("robot_joints") or []
        if not joints or action is None or len(action) < 16:
            return
        a = np.asarray(action, dtype=float).reshape(-1)[:16]
        targets18 = np.concatenate(
            [a[0:7], [a[7] / 2.0, a[7] / 2.0], a[8:15], [a[15] / 2.0, a[15] / 2.0]]
        )
        world = self._world(scene)
        control_fn = getattr(world, "control_dofs_position", None)
        idx_fn = getattr(world, "get_joint_dofs_idx_local", None)
        if control_fn is None or idx_fn is None:
            if not self._servo_warned:
                logger.info("efe_rw: no dof position control; actions not applied")
                self._servo_warned = True
            return
        try:
            dofs_idx = [i for j in joints for i in idx_fn(j)]
            control_fn(targets=targets18, dofs_idx=dofs_idx)
        except Exception as exc:  # noqa: BLE001 - servo is best effort
            if not self._servo_warned:
                logger.info(f"efe_rw: dof position control unavailable: {exc}")
                self._servo_warned = True

    # -- Task API ----------------------------------------------------------

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        info = super().reset(scene, robot, seed)
        self._apply_frozen_state(scene)
        self._hold_count = 0
        self._window_pos = {}
        self._rest_violation = False
        self._servo_warned = False
        info["rw"] = {"success": False, "goal": False, "hold_steps": 0}
        return info

    def step(self, scene: Any, robot: Any, action: np.ndarray):
        reward, terminated, truncated, info = super().step(scene, robot, action)
        self._apply_targets(scene, action)
        settle = self._rw.get("settle") or {}
        settle_steps = int(settle.get("settle_steps", 30))
        check_steps = int(settle.get("check_steps", 60))
        rest_m = float(settle.get("rest_m", 0.002))
        graded = list((self._rw.get("graded") or {}).get("rest") or [])

        goal = self._goal(scene)
        if goal:
            self._hold_count += 1
        else:
            self._hold_count = 0
            self._window_pos = {}
            self._rest_violation = False
        # window positions captured when the hold reaches the settle phase end
        if self._hold_count == settle_steps + 1 and not self._window_pos:
            for name in graded:
                try:
                    pos, _, _ = self._object_state(scene, name)
                    self._window_pos[name] = pos.copy()
                except KeyError:
                    pass
        if self._window_pos:
            for name, p0 in self._window_pos.items():
                try:
                    pos, _, _ = self._object_state(scene, name)
                except KeyError:
                    continue
                if float(np.linalg.norm(pos - p0)) > rest_m:
                    self._rest_violation = True
        success = bool(
            self._hold_count >= settle_steps + check_steps and not self._rest_violation
        )
        info["rw"] = {
            "success": success,
            "goal": goal,
            "hold_steps": self._hold_count,
        }
        return reward, terminated, truncated, info


@register_task("efe_rw")
def _make_efe_rw(**kwargs: Any) -> EfeRwTask:
    """RoboWits frozen-scene task with world/rw YAML blocks."""
    kwargs.pop("type", None)
    max_steps = int(kwargs.pop("max_episode_steps", 200_000))
    world = kwargs.pop("world", None)
    rw = kwargs.pop("rw", None)
    return EfeRwTask(
        TaskConfig(name="efe", max_episode_steps=max_steps),
        world=world,
        rw=rw,
    )


@register_robot("marvin_bimanual")
def _make_marvin_bimanual(**kwargs: Any) -> Any:
    """Tianji Marvin bimanual robot inside a whole-world MJCF (world-embedded)."""
    kwargs.pop("type", None)
    return _WorldEmbeddedRobot("marvin_bimanual")
