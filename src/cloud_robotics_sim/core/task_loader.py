"""Task-level YAML loader: define evaluation tasks without writing Python.

This module implements W1 of ``docs/ROBODOJO_P0_PLAN.md`` — the task-level
counterpart of the environment-level ``core/config_loader.py``. A task YAML
(see ``configs/tasks/pick_place_cube.yaml`` for the schema v1 reference)
declares which registry scene/robot/task factories to use, the objects that
populate the scene, the initial-pose distribution, cameras, a configurable
success condition, and the evaluation protocol (seeds x episodes).

Loading is layered so each layer stays testable without Genesis::

    load_task_spec(path)                # YAML -> validated TaskSpec (pure)
    sample_layout(spec, seed)           # deterministic layout sampling (pure)
    build_task_components(spec, reg)    # registry factories + ConfigurableTask
    compose_task(spec, reg, composer)   # full ComposedEnvironment
    load_task(path, ...)                # spec -> compose_task shortcut
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, overload

import numpy as np
import yaml

from cloud_robotics_sim.core.composer import (
    ComposedEnvironment,
    ComposerConfig,
    EnvironmentComposer,
)
from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.registry import AssetRegistry, default_registry
from cloud_robotics_sim.core.scene import ObjectSpawn
from cloud_robotics_sim.core.success import SuccessCondition, create_success_condition
from cloud_robotics_sim.core.task import Task, TaskConfig
from cloud_robotics_sim.robotwin.bridge import CameraConfig
from cloud_robotics_sim.robotwin.seed_search import ApplySeedFn

logger = logging.getLogger(__name__)

_TASK_KEYS = ("name", "type", "robot", "scene", "max_episode_steps")
_SPAWN_FIELD_NAMES = {f.name for f in dataclass_fields(ObjectSpawn)}
_SPAWN_SUPPORTED = _SPAWN_FIELD_NAMES - {"deformable_config"}
_SHAPE_TYPES = ("box", "sphere", "cylinder", "mesh", "deformable")
_COMPOSER_FIELD_NAMES = {f.name for f in dataclass_fields(ComposerConfig)}
_CAMERA_REQUIRED = ("name", "pos", "look_at", "resolution")


@dataclass
class EvaluationSpec:
    """Evaluation protocol: which seeds, how many episodes per seed."""

    seeds: list[int]
    episodes_per_seed: int = 1

    def episodes(self) -> list[tuple[int, int]]:
        """Expand to the full (seed, episode_index) evaluation grid."""
        return [
            (seed, episode)
            for seed in self.seeds
            for episode in range(self.episodes_per_seed)
        ]


@dataclass
class TaskSpec:
    """Validated, parsed task-level YAML (schema v1).

    ``scene_kwargs``/``robot_kwargs``/``task_kwargs`` carry the optional
    top-level ``scene:``/``robot:`` mappings (plus leftover ``task:`` keys),
    which are forwarded verbatim to the registry factories. ``simulation``
    holds optional ``ComposerConfig`` overrides (dt, substeps, headless, ...).
    """

    name: str
    scene_name: str
    robot_name: str
    task_name: str
    max_episode_steps: int
    scene_kwargs: dict[str, Any] = field(default_factory=dict)
    robot_kwargs: dict[str, Any] = field(default_factory=dict)
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    object_spawns: list[ObjectSpawn] = field(default_factory=list)
    distractor_names: list[str] = field(default_factory=list)
    init_distribution: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)
    cameras: list[CameraConfig] = field(default_factory=list)
    success: dict[str, Any] = field(default_factory=dict)
    evaluation: EvaluationSpec = field(
        default_factory=lambda: EvaluationSpec(seeds=[0])
    )
    simulation: dict[str, Any] = field(default_factory=dict)

    def success_condition(self) -> SuccessCondition:
        """Build the configured success condition."""
        return create_success_condition(self.success)


# ---------------------------------------------------------------------------
# YAML parsing / validation
# ---------------------------------------------------------------------------


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load and validate a task-level YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"task config file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse task config {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"task config {config_path} must be a YAML mapping")
    return parse_task_spec(data, source=str(config_path))


def parse_task_spec(data: Mapping[str, Any], source: str = "task config") -> TaskSpec:
    """Validate a parsed task YAML mapping and build a TaskSpec.

    Raises:
        ConfigError: Aggregated list of every problem found.
    """
    problems: list[str] = []

    task_section = data.get("task")
    if not isinstance(task_section, dict):
        problems.append("missing 'task' mapping")
        task_section = {}

    name = task_section.get("name")
    if not isinstance(name, str) or not name:
        problems.append("task.name is required")

    task_type = task_section.get("type", name)
    if not isinstance(task_type, str) or not task_type:
        problems.append("task.type (registry task factory name) is required")
        task_type = name

    robot_name = task_section.get("robot")
    if not isinstance(robot_name, str) or not robot_name:
        problems.append("task.robot is required")

    scene_name = task_section.get("scene")
    if not isinstance(scene_name, str) or not scene_name:
        problems.append("task.scene is required (registry scene factory name)")

    max_episode_steps = task_section.get("max_episode_steps", 500)
    if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
        problems.append("task.max_episode_steps must be a positive integer")
        max_episode_steps = 500

    task_kwargs = {
        key: value for key, value in task_section.items() if key not in _TASK_KEYS
    }

    scene_kwargs = _plain_mapping(data.get("scene"), "scene", problems)
    robot_kwargs = _plain_mapping(data.get("robot"), "robot", problems)
    init_distribution = _plain_mapping(
        data.get("init_distribution"), "init_distribution", problems
    )
    randomization = _plain_mapping(data.get("randomization"), "randomization", problems)
    simulation = _parse_simulation(data.get("simulation"), problems)

    object_spawns, distractor_names = _parse_assets(data.get("assets"), problems)
    cameras = _parse_cameras(data.get("cameras"), problems)
    evaluation = _parse_evaluation(data.get("evaluation"), problems)

    success = data.get("success")
    if not isinstance(success, dict):
        problems.append("missing 'success' mapping ({type, params})")
        success = {}
    else:
        try:
            create_success_condition(success)
        except ConfigError as exc:
            problems.append(f"invalid success block: {exc}")

    if problems:
        raise ConfigError(f"invalid {source}: " + "; ".join(problems))

    return TaskSpec(
        name=name,  # type: ignore[arg-type]
        scene_name=scene_name,  # type: ignore[arg-type]
        robot_name=robot_name,  # type: ignore[arg-type]
        task_name=task_type,  # type: ignore[arg-type]
        max_episode_steps=max_episode_steps,
        scene_kwargs=scene_kwargs,
        robot_kwargs=robot_kwargs,
        task_kwargs=task_kwargs,
        object_spawns=object_spawns,
        distractor_names=distractor_names,
        init_distribution=init_distribution,
        randomization=randomization,
        cameras=cameras,
        success=success,
        evaluation=evaluation,
        simulation=simulation,
    )


def _plain_mapping(value: Any, name: str, problems: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"'{name}' must be a mapping")
        return {}
    return dict(value)


def _parse_simulation(value: Any, problems: list[str]) -> dict[str, Any]:
    section = _plain_mapping(value, "simulation", problems)
    for key in section:
        if key not in _COMPOSER_FIELD_NAMES:
            problems.append(
                f"simulation.{key} is not a ComposerConfig field "
                f"(available: {sorted(_COMPOSER_FIELD_NAMES)})"
            )
    if "resolution" in section and isinstance(section["resolution"], list):
        section["resolution"] = tuple(section["resolution"])
    return section


def _parse_evaluation(value: Any, problems: list[str]) -> EvaluationSpec:
    section = _plain_mapping(value, "evaluation", problems)
    seeds = section.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(s, int) for s in seeds)
    ):
        problems.append("evaluation.seeds must be a non-empty list of integers")
        seeds = [0]
    episodes_per_seed = section.get("episodes_per_seed", 1)
    if not isinstance(episodes_per_seed, int) or episodes_per_seed < 1:
        problems.append("evaluation.episodes_per_seed must be a positive integer")
        episodes_per_seed = 1
    return EvaluationSpec(seeds=list(seeds), episodes_per_seed=episodes_per_seed)


def _parse_cameras(value: Any, problems: list[str]) -> list[CameraConfig]:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append("'cameras' must be a list of camera mappings")
        return []
    cameras: list[CameraConfig] = []
    for index, entry in enumerate(value):
        label = f"cameras[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{label} must be a mapping")
            continue
        missing = [key for key in _CAMERA_REQUIRED if key not in entry]
        if missing:
            problems.append(f"{label} missing required keys: {missing}")
            continue
        try:
            cameras.append(
                CameraConfig(
                    name=str(entry["name"]),
                    pos=_float_tuple(entry["pos"], 3, f"{label}.pos"),
                    look_at=_float_tuple(entry["look_at"], 3, f"{label}.look_at"),
                    resolution=(
                        int(entry["resolution"][0]),
                        int(entry["resolution"][1]),
                    ),
                    fov=float(entry.get("fov", 60.0)),
                    near=float(entry.get("near", 0.01)),
                    far=float(entry.get("far", 100.0)),
                )
            )
        except (ConfigError, TypeError, ValueError, IndexError) as exc:
            problems.append(f"{label}: {exc}")
    return cameras


def _parse_assets(
    value: Any, problems: list[str]
) -> tuple[list[ObjectSpawn], list[str]]:
    if value is None:
        return [], []
    if not isinstance(value, dict):
        problems.append("'assets' must be a mapping with task_relevant/distractors")
        return [], []

    spawns: list[ObjectSpawn] = []
    distractor_names: list[str] = []
    for group in ("task_relevant", "distractors"):
        entries = value.get(group) or []
        if not isinstance(entries, list):
            problems.append(f"assets.{group} must be a list")
            continue
        for index, entry in enumerate(entries):
            label = f"assets.{group}[{index}]"
            spawn = _parse_object_spawn(entry, label, problems)
            if spawn is None:
                continue
            if group == "distractors":
                distractor_names.append(spawn.name)
            spawns.append(spawn)
    return spawns, distractor_names


def _parse_object_spawn(
    entry: Any, label: str, problems: list[str]
) -> ObjectSpawn | None:
    if not isinstance(entry, dict):
        problems.append(f"{label} must be a mapping")
        return None
    unknown = sorted(set(entry) - _SPAWN_SUPPORTED)
    if unknown:
        problems.append(f"{label} has unknown keys: {unknown}")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        problems.append(f"{label}.name is required")
        return None

    shape_type = entry.get("shape_type", "box")
    if shape_type not in _SHAPE_TYPES:
        problems.append(f"{label}.shape_type must be one of {list(_SHAPE_TYPES)}")
        return None
    mesh_path = entry.get("mesh_path")
    if shape_type == "mesh" and not mesh_path:
        problems.append(f"{label}.mesh_path is required for shape_type='mesh'")
        return None

    try:
        size = _float_tuple(entry.get("size", (1.0, 1.0, 1.0)), None, f"{label}.size")
        color = _color_tuple(entry.get("color"), f"{label}.color")
        scale_value = entry.get("scale", (1.0, 1.0, 1.0))
        if isinstance(scale_value, (int, float)):
            scale: tuple[float, float, float] | float = float(scale_value)
        else:
            scale = _float_tuple(scale_value, 3, f"{label}.scale")
        tags = entry.get("tags", [])
        if not isinstance(tags, list):
            raise ConfigError(f"{label}.tags must be a list")
        properties = entry.get("properties", {})
        if not isinstance(properties, dict):
            raise ConfigError(f"{label}.properties must be a mapping")
        return ObjectSpawn(
            name=name,
            shape_type=shape_type,
            size=size,
            scale=scale,
            mesh_path=mesh_path,
            position=_float_tuple(
                entry.get("position", (0.0, 0.0, 0.0)), 3, f"{label}.position"
            ),
            orientation=_float_tuple(
                entry.get("orientation", (1.0, 0.0, 0.0, 0.0)),
                4,
                f"{label}.orientation",
            ),
            mass=float(entry.get("mass", 1.0)),
            static=bool(entry.get("static", True)),
            friction=float(entry.get("friction", 0.5)),
            material=str(entry.get("material", "default")),
            color=color,
            tags=[str(tag) for tag in tags],
            properties=dict(properties),
        )
    except (ConfigError, TypeError, ValueError) as exc:
        problems.append(str(exc))
        return None


@overload
def _float_tuple(
    value: Any, n: Literal[3], name: str
) -> tuple[float, float, float]: ...


@overload
def _float_tuple(
    value: Any, n: Literal[4], name: str
) -> tuple[float, float, float, float]: ...


@overload
def _float_tuple(value: Any, n: None, name: str) -> tuple[float, ...]: ...


def _float_tuple(value: Any, n: int | None, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or (n is not None and len(value) != n):
        expected = f"{n} numbers" if n is not None else "a list of numbers"
        raise ConfigError(f"{name} must be {expected}, got {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must contain only numbers, got {value!r}") from exc


def _color_tuple(value: Any, name: str) -> tuple[float, float, float, float]:
    if value is None:
        return (0.8, 0.8, 0.8, 1.0)
    rgb = _float_tuple(value, None, name)
    if len(rgb) not in (3, 4):
        raise ConfigError(f"{name} must have 3 or 4 components, got {value!r}")
    if len(rgb) == 3:
        return (rgb[0], rgb[1], rgb[2], 1.0)
    return (rgb[0], rgb[1], rgb[2], rgb[3])


# ---------------------------------------------------------------------------
# Deterministic reset sampling
# ---------------------------------------------------------------------------


def _sample_axis(value: Any, rng: np.random.Generator) -> float:
    """Sample one axis: a fixed number or a [lo, hi] range."""
    if isinstance(value, (int, float)):
        return float(value)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
    ):
        return float(rng.uniform(float(value[0]), float(value[1])))
    raise ConfigError(f"pose axis must be a number or [lo, hi] range, got {value!r}")


def _sample_position(
    spec: Any, default: Sequence[float], rng: np.random.Generator
) -> list[float]:
    if spec is None:
        return [float(v) for v in default]
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        return [_sample_axis(axis, rng) for axis in spec]
    raise ConfigError(f"position must be [x, y, z] or per-axis ranges, got {spec!r}")


def _sample_orientation(
    spec: Any, default: Sequence[float], rng: np.random.Generator
) -> list[float]:
    if spec is None:
        return [float(v) for v in default]
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return [float(v) for v in spec]
    if isinstance(spec, dict) and isinstance(spec.get("yaw_deg"), (list, tuple)):
        yaw = np.radians(_sample_axis(spec["yaw_deg"], rng))
        return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]
    raise ConfigError(
        f"orientation must be [w, x, y, z] or {{yaw_deg: [lo, hi]}}, got {spec!r}"
    )


def sample_layout(spec: TaskSpec, seed: int) -> dict[str, dict[str, Any]]:
    """Sample an object layout deterministically from ``init_distribution``.

    The same seed always produces the same layout (``numpy.default_rng``).
    Objects without an ``object_poses`` entry keep the fixed pose from their
    ``assets`` block unless a ``clutter_layout`` scatters them.
    """
    rng = np.random.default_rng(seed)
    pose_specs = spec.init_distribution.get("object_poses") or {}
    articulation = spec.init_distribution.get("articulation_states") or {}

    layout: dict[str, dict[str, Any]] = {}
    for spawn in spec.object_spawns:
        entry = pose_specs.get(spawn.name) or {}
        record: dict[str, Any] = {
            "position": _sample_position(entry.get("position"), spawn.position, rng),
            "orientation": _sample_orientation(
                entry.get("orientation"), spawn.orientation, rng
            ),
        }
        if spawn.name in articulation:
            joints = articulation[spawn.name] or {}
            if not isinstance(joints, dict):
                raise ConfigError(
                    f"articulation_states.{spawn.name} must be a mapping of "
                    f"joint -> value or [lo, hi] range"
                )
            record["articulation"] = {
                joint: _sample_axis(value, rng) for joint, value in joints.items()
            }
        layout[spawn.name] = record

    clutter = spec.init_distribution.get("clutter_layout")
    if clutter:
        layout.update(_sample_clutter(clutter, pose_specs, spec.distractor_names, rng))
    return layout


def _sample_clutter(
    clutter: Any,
    pose_specs: Mapping[str, Any],
    distractor_names: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    if not isinstance(clutter, dict):
        raise ConfigError("clutter_layout must be a mapping with a 'region'")
    region = clutter.get("region")
    if (
        not isinstance(region, (list, tuple))
        or len(region) != 5
        or any(not isinstance(v, (int, float)) for v in region)
    ):
        raise ConfigError(
            f"clutter_layout.region must be [xmin, xmax, ymin, ymax, z], got {region!r}"
        )
    names = clutter.get("objects") or [
        name for name in distractor_names if name not in pose_specs
    ]
    xmin, xmax, ymin, ymax, z = (float(v) for v in region)
    return {
        name: {
            "position": [
                float(rng.uniform(xmin, xmax)),
                float(rng.uniform(ymin, ymax)),
                z,
            ],
            "orientation": [1.0, 0.0, 0.0, 0.0],
        }
        for name in names
    }


def apply_layout(scene: Any, layout: Mapping[str, Mapping[str, Any]]) -> None:
    """Apply a sampled layout to a built scene's entities (in place)."""
    entities = getattr(scene, "entities", None) or {}
    for name, record in layout.items():
        entity = entities.get(name)
        if entity is None:
            logger.warning(
                "layout references object '%s' but the scene has no such entity", name
            )
            continue
        set_pos = getattr(entity, "set_pos", None)
        if set_pos is not None:
            set_pos(np.asarray(record["position"], dtype=float))
        orientation = record.get("orientation")
        set_quat = getattr(entity, "set_quat", None)
        if orientation is not None and set_quat is not None:
            set_quat(np.asarray(orientation, dtype=float))


def make_apply_seed_fn(spec: TaskSpec, scenes: Sequence[Any]) -> ApplySeedFn:
    """Build an ``apply_seed`` callback for ``robotwin.seed_search.batched_seed_search``.

    Layouts are cached per seed, so the same seed always produces the same
    layout in every env regardless of evaluation order.
    """
    cache: dict[int, dict[str, dict[str, Any]]] = {}

    def apply_seed(seed: int, env_idx: int) -> None:
        if not 0 <= env_idx < len(scenes):
            raise IndexError(f"env_idx {env_idx} out of range for {len(scenes)} scenes")
        if seed not in cache:
            cache[seed] = sample_layout(spec, seed)
        apply_layout(scenes[env_idx], cache[seed])

    return apply_seed


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------


class ConfigurableTask(Task):
    """Wrap a registry-built base task with a configured success condition.

    The base task keeps its reward shaping, reset behavior, and optional own
    success check; the configured :class:`SuccessCondition` overrides success
    attribution: it drives ``terminated``, reports ``failure_stage`` in info
    (FAIL_STAGES vocabulary), and enforces ``max_episode_steps`` from the YAML
    (registry factories may hardcode their own episode budget).
    """

    def __init__(
        self,
        base_task: Task,
        success_condition: SuccessCondition | None | None = None,
        *,
        name: str | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        base_config = base_task.config
        super().__init__(
            TaskConfig(
                name=name or base_config.name,
                max_episode_steps=max_episode_steps or base_config.max_episode_steps,
                success_reward=base_config.success_reward,
                timeout_penalty=base_config.timeout_penalty,
                step_penalty=base_config.step_penalty,
            )
        )
        self.base_task = base_task
        self.success_condition = success_condition

    def reset(self, scene: Any, robot: Any, seed: int) -> dict:
        """Reset the wrapped base task."""
        self.step_count = 0
        self.succeeded = False
        return self.base_task.reset(scene, robot, seed)

    def step(
        self, scene: Any, robot: Any, action: np.ndarray
    ) -> tuple[float, bool, bool, dict]:
        """Delegate to the base task, then apply the configured condition."""
        reward, terminated, truncated, info = self.base_task.step(scene, robot, action)
        self.step_count = self.base_task.step_count
        self.succeeded = self.base_task.succeeded

        if self.success_condition is not None and not terminated:
            success = bool(self.success_condition.evaluate(scene, robot, info))
            info = {**info, "success": success}
            if success:
                terminated = True
                self.succeeded = True
                reward += self.config.success_reward
                info["failure_stage"] = None
            else:
                info["failure_stage"] = self.success_condition.last_failure_stage()

        if (
            not terminated
            and not truncated
            and self.step_count >= self.config.max_episode_steps
        ):
            truncated = True
            if not self.succeeded:
                reward += self.config.timeout_penalty

        return reward, terminated, truncated, info


def _resolve_registry(registry: Any) -> AssetRegistry:
    if registry is None:
        resolved = default_registry()
    elif isinstance(registry, AssetRegistry):
        resolved = registry
    elif callable(registry):
        resolved = registry()
    else:
        resolved = registry
    if not isinstance(resolved, AssetRegistry):
        raise TypeError(
            f"registry must be an AssetRegistry or a callable returning one, "
            f"got {type(resolved).__name__}"
        )
    return resolved


def _create_from(
    component_registry: Any, kind: str, name: str, kwargs: dict[str, Any]
) -> Any:
    try:
        return component_registry.create(name, **kwargs)
    except KeyError as exc:
        available = component_registry.list_components()
        raise ConfigError(
            f"{kind} '{name}' is not registered (available: {available}). "
            f"Import 'cloud_robotics_sim.runtime.main' to register the built-in "
            f"components, or pass a populated registry=..."
        ) from exc


def build_task_components(
    spec: TaskSpec, registry: Any = None
) -> tuple[Any, Any, ConfigurableTask]:
    """Instantiate scene, robot, and wrapped task from registry factories.

    Object spawns are injected via ``scene.add_object`` between scene creation
    and build — this is why the loader uses the registry factories directly
    instead of ``compose_from_registry``.
    """
    reg = _resolve_registry(registry)
    scene = _create_from(reg.scenes, "scene", spec.scene_name, spec.scene_kwargs)
    for spawn in spec.object_spawns:
        scene.add_object(spawn)
    robot = _create_from(reg.robots, "robot", spec.robot_name, spec.robot_kwargs)
    base_task = _create_from(reg.tasks, "task", spec.task_name, spec.task_kwargs)
    if not isinstance(base_task, Task):
        raise TypeError(
            f"task factory '{spec.task_name}' returned {type(base_task).__name__}, "
            f"expected a Task subclass"
        )
    task = ConfigurableTask(
        base_task,
        spec.success_condition(),
        name=spec.name,
        max_episode_steps=spec.max_episode_steps,
    )
    return scene, robot, task


@dataclass
class LoadedTask:
    """A composed task environment plus its spec and success condition."""

    spec: TaskSpec
    env: ComposedEnvironment
    success_condition: SuccessCondition | None

    def reset_episode(self, seed: int) -> tuple[dict, dict]:
        """Sample the seed's layout, apply it, and reset the environment."""
        apply_layout(self.env.scene, sample_layout(self.spec, seed))
        return self.env.reset(seed=seed)

    def evaluation_episodes(self) -> list[tuple[int, int]]:
        """Expand the evaluation protocol to (seed, episode_index) pairs."""
        return self.spec.evaluation.episodes()


def compose_task(
    spec: TaskSpec,
    registry: Any = None,
    composer: EnvironmentComposer | None = None,
) -> LoadedTask:
    """Build a full ComposedEnvironment from a validated TaskSpec."""
    scene, robot, task = build_task_components(spec, registry)
    if composer is None:
        composer = EnvironmentComposer(ComposerConfig(**spec.simulation))
    env = composer.compose(scene, robot, task)
    return LoadedTask(spec=spec, env=env, success_condition=task.success_condition)


def load_task(
    path: str | Path,
    registry: Any = None,
    composer: EnvironmentComposer | None = None,
) -> LoadedTask:
    """Load a task YAML and compose its environment (spec -> compose_task)."""
    return compose_task(load_task_spec(path), registry=registry, composer=composer)
