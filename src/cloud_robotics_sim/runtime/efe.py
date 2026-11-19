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
