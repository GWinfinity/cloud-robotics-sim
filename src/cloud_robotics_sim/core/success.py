"""Configurable success conditions for task-level YAML definitions.

Success conditions decouple "what counts as success" from ``Task`` subclasses:
a task YAML declares a ``success: {type, params}`` block and the loader builds
a :class:`SuccessCondition` that the evaluation runner (or the loader's
``ConfigurableTask`` wrapper) evaluates against the built scene. The staged
condition reuses the seven-level ``FAIL_STAGES`` vocabulary from
``robotwin.grasp_report`` so failure reports stay comparable with grasp
pipeline reports.

All conditions are engine-agnostic: they only require scene entities to
expose ``get_pos()`` (and ``get_quat()`` for pose checks), so they can be
unit-tested without Genesis.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np

from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.robotwin.grasp_report import FAIL_STAGES

logger = logging.getLogger(__name__)

SUCCESS_TYPES = (
    "distance_threshold",
    "pose_window",
    "staged",
    "axis_window",
    "in_region",
    "count_in_region",
    "body_point_near",
    "ee_near",
    "above_region",
    "tipped",
    "contains",
    "lift",
    "ordered",
    "in_contact",
    "joint_window",
    "libero_on",
    "libero_in",
    "libero_joint_state",
    "rc_contact",
    "rc_in_receptacle",
    "rc_gripper_far",
    "rc_joint_state",
    "rc_appliance",
    "rc_inside",
    "rc_upright",
    "rc_at_spawn",
    "rc_near",
    "rc_contact_any",
    "rc_timer",
    "rc_basin",
    "rc_on_burner",
    "rc_closer",
    "all",
    "any",
    "not",
)


class SuccessCondition(ABC):
    """A boolean goal predicate evaluated against a built scene."""

    @abstractmethod
    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        """Return True when the configured goal is reached.

        Args:
            scene: Built scene with an ``entities`` mapping.
            robot: Robot embodiment (reserved for end-effector checks).
            info: Optional task info dict from the current step.
        """

    def last_failure_stage(self) -> str:
        """Return the FAIL_STAGES label for the most recent failed evaluation."""
        return "error"


def _entity(scene: Any, name: str) -> Any:
    entities = getattr(scene, "entities", None) or {}
    if name not in entities:
        raise KeyError(
            f"success condition references unknown object '{name}'; "
            f"available entities: {sorted(entities)}"
        )
    return entities[name]


def _entity_position(scene: Any, name: str) -> np.ndarray:
    entity = _entity(scene, name)
    get_pos = getattr(entity, "get_pos", None)
    if get_pos is None:
        raise TypeError(f"entity '{name}' does not expose get_pos()")
    return np.asarray(get_pos(), dtype=float)


def _entity_quaternion(scene: Any, name: str) -> np.ndarray:
    entity = _entity(scene, name)
    get_quat = getattr(entity, "get_quat", None)
    if get_quat is None:
        raise TypeError(
            f"entity '{name}' does not expose get_quat() "
            f"required by a pose_window success condition"
        )
    return np.asarray(get_quat(), dtype=float)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle in degrees between two wxyz quaternions."""
    d = float(np.dot(q1, q2))
    return float(np.degrees(2.0 * np.arccos(min(abs(d), 1.0))))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the wxyz quaternion ``q``."""
    w, x, y, z = (float(c) for c in q)
    u = np.array([x, y, z], dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


@overload
def _vec(value: Any, n: Literal[3], name: str) -> tuple[float, float, float]: ...


@overload
def _vec(value: Any, n: Literal[4], name: str) -> tuple[float, float, float, float]: ...


@overload
def _vec(value: Any, n: int, name: str) -> tuple[float, ...]: ...


def _vec(value: Any, n: int | None, name: str) -> tuple[float, ...]:
    """Coerce a YAML sequence to an n-tuple of floats."""
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ConfigError(f"{name} must be a sequence of {n} numbers, got {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} must be a sequence of {n} numbers, got {value!r}"
        ) from exc


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive number, got {value!r}") from exc
    if result <= 0:
        raise ConfigError(f"{name} must be a positive number, got {value!r}")
    return result


def _failure_stage(params: dict, default: str = "error") -> str:
    stage = params.get("failure_stage", default)
    if stage not in FAIL_STAGES:
        raise ConfigError(
            f"failure_stage must be one of {list(FAIL_STAGES)}, got {stage!r}"
        )
    return str(stage)


@dataclass
class DistanceThresholdCondition(SuccessCondition):
    """Success when an object rests within ``threshold`` of a target position.

    ``axis_weights`` scales each coordinate difference before the norm —
    MetaWorld's basketball counts the height difference double. Weights
    default to 1 (plain Euclidean distance).
    """

    object_name: str
    target_position: tuple[float, float, float]
    threshold: float
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        delta = (pos - np.asarray(self.target_position)) * np.asarray(self.axis_weights)
        dist = float(np.linalg.norm(delta))
        return dist <= self.threshold

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class PoseWindowCondition(SuccessCondition):
    """Success when an object pose is within position and rotation windows.

    ``target_orientation`` is a wxyz quaternion; when omitted, only the
    position window is checked.
    """

    object_name: str
    target_position: tuple[float, float, float]
    target_orientation: tuple[float, float, float, float] | None = None
    pos_threshold: float = 0.02
    rot_threshold_deg: float = 10.0
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        if (
            float(np.linalg.norm(pos - np.asarray(self.target_position)))
            > self.pos_threshold
        ):
            return False
        if self.target_orientation is None:
            return True
        quat = _entity_quaternion(scene, self.object_name)
        return (
            _quat_angle_deg(quat, np.asarray(self.target_orientation))
            <= self.rot_threshold_deg
        )

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class AxisWindowCondition(SuccessCondition):
    """Success when selected position coordinates sit inside per-axis windows.

    Each window is ``[lo, hi]``; a ``None`` bound leaves that side
    unbounded. With ``target_position`` set, the windows are *offsets*
    around the target (MetaWorld's "within 2 cm of the target horizontally
    and lower than it"); without it they are absolute world-frame ranges.
    Only the listed axes are checked — the classic MetaWorld "door-open
    compares x only" family of predicates.
    """

    object_name: str
    windows: dict[str, tuple[float | None, float | None]]
    target_position: tuple[float, float, float] | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        axis_index = {"x": 0, "y": 1, "z": 2}
        anchor = (
            np.asarray(self.target_position, dtype=float)
            if self.target_position is not None
            else None
        )
        for axis, (lo, hi) in self.windows.items():
            idx = axis_index[axis]
            value = float(pos[idx])
            if anchor is not None:
                value -= float(anchor[idx])
            if lo is not None and value < lo:
                return False
            if hi is not None and value > hi:
                return False
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class BodyPointNearCondition(SuccessCondition):
    """Success when a body-fixed point is near a target position.

    The point is ``offset`` (body frame, metres) rotated by the entity's
    current orientation and added to its position — MetaWorld's rim markers
    (dial-turn), handle ends (faucet) and peg heads (peg-insertion), whose
    XML tracks a *site* on a moving body rather than the body origin.
    """

    object_name: str
    offset: tuple[float, float, float]
    target_position: tuple[float, float, float]
    threshold: float
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        quat = _entity_quaternion(scene, self.object_name)
        # wxyz quaternion rotation, matching _quat_rotate in the compiler.
        w, x, y, z = quat
        u = np.array([x, y, z])
        rotated = self.offset + 2.0 * np.cross(
            u, np.cross(u, self.offset) + w * np.asarray(self.offset)
        )
        point = pos + rotated
        delta = (point - np.asarray(self.target_position)) * np.asarray(
            self.axis_weights
        )
        return float(np.linalg.norm(delta)) <= self.threshold

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class InRegionCondition(SuccessCondition):
    """Success when an entity position is inside an axis-aligned box.

    ``region`` is ``[xmin, xmax, ymin, ymax, zmin, zmax]`` in metres — goal
    areas, containers, hoops (VLABench/MetaWorld/KinDER goal regions).
    """

    object_name: str
    region: tuple[float, float, float, float, float, float]
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        xmin, xmax, ymin, ymax, zmin, zmax = self.region
        x, y, z = (float(v) for v in pos)
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class CountInRegionCondition(SuccessCondition):
    """Success when at least ``min_count`` named entities are inside a box.

    Entities are matched either by an explicit ``objects`` list or by a
    ``prefix`` of their scene name (KinDER's ``o50`` cube families). The
    count includes only entities that exist in the scene.
    """

    objects: tuple[str, ...] | None = None
    region: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    min_count: int = 1
    prefix: str | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        entities = getattr(scene, "entities", None) or {}
        xmin, xmax, ymin, ymax, zmin, zmax = self.region
        count = 0
        for name, entity in entities.items():
            if self.prefix is not None and not name.startswith(self.prefix):
                continue
            if self.objects is not None and name not in self.objects:
                continue
            get_pos = getattr(entity, "get_pos", None)
            if get_pos is None:
                continue
            x, y, z = (float(v) for v in np.asarray(get_pos(), dtype=float))
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                count += 1
        return count >= self.min_count

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class EeNearCondition(SuccessCondition):
    """Success when a robot link (default: end effector) is near a target.

    MetaWorld's ``reach`` family grades the point between the fingertips
    (the welded ``hand``/mocap link), not a scene object. The robot's
    articulation backend exposes ``get_link_pose``; when ``link`` is unset
    the articulation root position is used (documented fallback).
    """

    target_position: tuple[float, float, float]
    threshold: float
    link: str | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        entity = getattr(robot, "entity", None) or robot
        pos: np.ndarray | None = None
        get_link_pose = getattr(entity, "get_link_pose", None)
        if self.link is not None and callable(get_link_pose):
            try:
                pos = np.asarray(get_link_pose(self.link).pos, dtype=float)
            except Exception:  # noqa: BLE001 - unknown link names vary by backend
                pos = None
        if pos is None:
            get_pos = getattr(entity, "get_pos", None)
            if get_pos is None:
                raise TypeError(
                    "ee_near needs a robot exposing get_link_pose() or get_pos()"
                )
            pos = np.asarray(get_pos(), dtype=float)
        dist = float(np.linalg.norm(pos - np.asarray(self.target_position)))
        return dist <= self.threshold

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class AboveRegionCondition(SuccessCondition):
    """Success when an entity hovers over a platform's footprint.

    VLABench's ``AboveCondition``: the entity's root must be at or above the
    platform's origin and its (x, y) must fall inside the platform footprint
    (``FlatContainer.contain`` at the platform's own height). The footprint is
    either a ``circle`` (radius in metres, centred on the platform — pans and
    dishes carry a ``horizontal_radius_site`` upstream) or an absolute
    ``region`` box ``[xmin, xmax, ymin, ymax]``.
    """

    object_name: str
    platform: str
    circle: float | None = None
    region: tuple[float, float, float, float] | None = None
    z_offset: float = 0.0
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        platform_pos = _entity_position(scene, self.platform)
        if pos[2] < platform_pos[2] + self.z_offset:
            return False
        x, y = float(pos[0]), float(pos[1])
        if self.circle is not None:
            cx, cy = float(platform_pos[0]), float(platform_pos[1])
            return float(np.hypot(x - cx, y - cy)) < self.circle
        assert self.region is not None
        xmin, xmax, ymin, ymax = self.region
        return xmin <= x <= xmax and ymin <= y <= ymax

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class TippedCondition(SuccessCondition):
    """Success when an entity is tipped past horizontal (VLABench ``pour``).

    MuJoCo cannot simulate the liquid, so upstream judges pouring by the
    entity's authored sites: success when the world-frame ``bottom_offset``
    point sits above the ``top_offset`` point by more than ``threshold``
    (``PourCondition``: ``bottom.z - top.z > threshold``, default 0). The
    offsets are in the entity's local frame — the compile knowledge base
    extracts them from the object MJCF's ``top_site``/``bottom_site``.
    """

    object_name: str
    top_offset: tuple[float, float, float]
    bottom_offset: tuple[float, float, float]
    threshold: float = 0.0
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        quat = _entity_quaternion(scene, self.object_name)
        top = pos + _quat_rotate(quat, np.asarray(self.top_offset, dtype=float))
        bottom = pos + _quat_rotate(quat, np.asarray(self.bottom_offset, dtype=float))
        return bool((bottom[2] - top[2]) > self.threshold)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class ContainsCondition(SuccessCondition):
    """Success when entities are contained by a container (VLABench family).

    Faithful port of ``ContainCondition`` + the upstream container mixins,
    evaluated in the container's local frame (all VLABench containers are
    static, so this equals the world-frame semantics):

    * ``mode="circle"`` (``FlatContainer``: plate/placemat/tray/pan): the
      entity root's xy-distance to the container origin is < ``circle`` and
      its relative height is inside the *exclusive* ``z_band`` (upstream
      ``0 < point.z - center.z < z_threshold``, default 0.1 m).
    * ``mode="boxes"`` (``CommonContainer``: giftbox/shelf/microwave/vase,
      and billiard pockets): the entity root lies in at least one box
      (any-of), bounds *inclusive* per upstream. Each region is
      ``{center, min, max}`` in the container frame — the compile knowledge
      base extracts them from the container MJCF (radius/keypoint sites,
      pocket bodies).
    * ``negate=True`` implements ``NotContainCondition`` (every entity must
      be outside).
    """

    container: str
    objects: tuple[str, ...]
    mode: str = "circle"
    circle: float = 0.0
    z_band: tuple[float, float] = (0.0, 0.1)
    regions: tuple[dict, ...] = ()
    negate: bool = False
    failure_stage: str = "error"

    def _inside(self, scene: Any, name: str) -> bool:
        pos = _entity_position(scene, name)
        center = _entity_position(scene, self.container)
        rel = pos - center
        if self.mode == "circle":
            z_lo, z_hi = self.z_band
            if not (z_lo < rel[2] < z_hi):
                return False
            return bool(np.hypot(rel[0], rel[1]) < self.circle)
        for region in self.regions:
            lo = np.asarray(region["min"], dtype=float)
            hi = np.asarray(region["max"], dtype=float)
            off = np.asarray(region.get("center", (0.0, 0.0, 0.0)), dtype=float)
            r = rel - off
            if bool(np.all(r >= lo) and np.all(r <= hi)):
                return True
        return False

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        if self.negate:
            return not any(self._inside(scene, name) for name in self.objects)
        return all(self._inside(scene, name) for name in self.objects)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class LiftCondition(SuccessCondition):
    """Success when every named entity is above ``min_height`` (VLABench lift)."""

    objects: tuple[str, ...]
    min_height: float
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        for name in self.objects:
            if _entity_position(scene, name)[2] < self.min_height:
                return False
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class OrderedCondition(SuccessCondition):
    """Success when entities are ordered along axes (VLABench ``order``).

    For each axis in ``axes`` the consecutive entities must be strictly
    increasing; on all other axes ``(p[i] - p[i+1]) < offset`` must hold
    (upstream's one-sided tolerance, transcribed verbatim).
    """

    objects: tuple[str, ...]
    axes: tuple[int, ...] = (0,)
    offset: float = 0.1
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        points = [_entity_position(scene, name) for name in self.objects]
        for axis in range(3):
            for i in range(len(points) - 1):
                if axis in self.axes:
                    if not points[i][axis] < points[i + 1][axis]:
                        return False
                elif not (points[i][axis] - points[i + 1][axis]) < self.offset:
                    return False
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class JointWindowCondition(SuccessCondition):
    """Success when an entity's joint position sits inside a window.

    VLABench ``joint_in_range`` (hammer-nail: nail slide qpos in
    ``target_pos_range``) and door-state checks derived from upstream
    ``ContainerWithDoor.is_closed`` (|qpos| < closed_threshold → window
    ``[-threshold, +threshold]``). ``dof`` indexes the entity's DOF vector;
    for articulated spawns (joints kept, no freejoint) dof 0 is the model's
    first joint.
    """

    object_name: str
    window: tuple[float, float]
    dof: int = 0
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        entity = _entity(scene, self.object_name)
        get_joints = getattr(entity, "get_joint_positions", None)
        if get_joints is None:
            raise TypeError(
                f"entity '{self.object_name}' does not expose "
                f"get_joint_positions() required by joint_window"
            )
        qpos = np.asarray(get_joints(), dtype=float)
        if qpos.size <= self.dof:
            raise IndexError(
                f"joint_window dof {self.dof} out of range for "
                f"'{self.object_name}' ({qpos.size} dofs)"
            )
        lo, hi = self.window
        return bool(lo <= qpos[self.dof] <= hi)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class InContactCondition(SuccessCondition):
    """Success when every named object contacts the counterpart set.

    VLABench ``is_grasped`` (entity ↔ robot gripper geoms) and
    ``press_button`` (upstream: touch sensor force in band sustained >10
    substeps; transcribed as contact at the graded final state — the force
    band 0.01–100 N is permissive, spec note). ``with_entity: "@robot"``
    resolves to the robot passed to ``evaluate``; otherwise an entity name
    in the scene (e.g. an object pressing another).
    """

    objects: tuple[str, ...]
    with_entity: str = "@robot"
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        if self.with_entity == "@robot":
            other = getattr(robot, "entity", None) or robot
        else:
            other = _entity(scene, self.with_entity)
        for name in self.objects:
            target = _entity(scene, name)
            if not _in_contact(target, other):
                return False
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


def _in_contact(entity_a: Any, entity_b: Any) -> bool:
    """True when two entities (backend wrappers or raw engine objects) touch."""
    if hasattr(entity_a, "in_contact"):
        return bool(entity_a.in_contact(entity_b))
    get_contacts = getattr(entity_a, "get_contacts", None)
    if get_contacts is None:
        raise TypeError(
            f"entity {getattr(entity_a, 'name', entity_a)!r} exposes neither "
            f"in_contact() nor get_contacts() — contact predicates need an "
            f"engine hook"
        )
    data = get_contacts(with_entity=entity_b)
    if not data:
        return False
    geom_a = data.get("geom_a") if isinstance(data, dict) else None
    if geom_a is None:
        return bool(len(data))
    shape = getattr(geom_a, "shape", None)
    return bool(int(shape[0]) > 0) if shape else bool(geom_a)


@dataclass
class StagedCondition(SuccessCondition):
    """Success when every sub-condition holds; reports the earliest failing stage.

    Each entry is a ``(FAIL_STAGES label, SuccessCondition)`` pair. Evaluation
    short-circuits at the first failing condition and remembers its stage so
    evaluation reports can attribute failures the same way the grasp pipeline
    does (load_fail ... place_fail / error).
    """

    stages: list[tuple[str, SuccessCondition]]
    _last_failure_stage: str = field(default="error", init=False)

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        for stage, condition in self.stages:
            if not condition.evaluate(scene, robot, info):
                self._last_failure_stage = stage
                return False
        self._last_failure_stage = "error"
        return True

    def last_failure_stage(self) -> str:
        return self._last_failure_stage


def _joint_qpos(scene: Any, entity_name: str, joint: str) -> float:
    """Read one named joint position from a scene entity.

    Prefers an explicit ``get_joint_position(name)`` hook, then the
    articulation backend pair ``get_joint_names`` + ``get_joint_qs_idx_local``
    + ``get_qpos``. Articulated names are matched verbatim first, then by
    unambiguous suffix: LIBERO composes models with an instance prefix
    (``flat_stove_1_button``) while component-level spawns carry the bare
    MJCF joint name (``button``); spawn-time duplicate-name repairs add a
    ``__dupN`` suffix that is ignored for the comparison.
    """
    import re

    def _base(name: str) -> str:
        return re.sub(r"__dup\d+$", "", name)

    entity = _entity(scene, entity_name)
    hook = getattr(entity, "get_joint_position", None)
    if hook is not None:
        return float(hook(joint))
    names = getattr(entity, "get_joint_names", None)
    idx_for = getattr(entity, "get_joint_qs_idx_local", None)
    get_qpos = getattr(entity, "get_qpos", None)
    if names is not None and idx_for is not None and get_qpos is not None:
        available = names()
        resolved = joint
        if joint not in available:
            candidates = [
                name
                for name in available
                if (
                    _base(name) == joint
                    or joint.endswith(_base(name))
                    or _base(name).endswith(joint)
                )
            ]
            if len(candidates) == 1:
                resolved = candidates[0]
            else:
                raise KeyError(
                    f"entity '{entity_name}' has no joint '{joint}' "
                    f"(available: {available})"
                )
        indices = idx_for(resolved)
        qpos = np.asarray(get_qpos(), dtype=float).reshape(-1)
        if not indices:
            raise KeyError(f"entity '{entity_name}' joint '{resolved}' has no qpos")
        return float(qpos[indices[0]])
    raise TypeError(
        f"entity '{entity_name}' exposes neither get_joint_position() nor "
        f"get_joint_names()/get_joint_qs_idx_local()/get_qpos() — "
        f"libero_joint_state needs a named-joint hook"
    )


@dataclass
class LiberoOnCondition(SuccessCondition):
    """LIBERO's ``On`` goal predicate (BDDL goal_state ``on``).

    Object support — ``arg2.check_ontop(arg1)``: the support's z is at or
    below the object's, the two touch, and their body origins are within
    0.03 m horizontally. Site support (a BDDL region site): the object sits
    in the site's ``under`` slab — site-local delta z within
    ``[halfsize.z - 0.005, halfsize.z + 0.10]`` and |dx|, |dy| within the
    site's xy half extents — plus contact with the site's parent fixture
    when the site has a parent (static fixtures: compiled as a baked world
    box + an explicit contact entity).
    """

    object_name: str
    support: str | None = None
    site_center: tuple[float, float, float] | None = None
    site_xmat: tuple[float, ...] | None = None
    site_halfsize: tuple[float, float, float] | None = None
    site_contact: str | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _entity_position(scene, self.object_name)
        if self.site_center is not None:
            center = np.asarray(self.site_center, dtype=float)
            xmat = np.asarray(self.site_xmat, dtype=float).reshape(3, 3)
            half = np.asarray(self.site_halfsize, dtype=float)
            delta = xmat @ (obj_pos - center)
            if not (
                half[2] - 0.005 < delta[2] < half[2] + 0.10
                and abs(delta[0]) < half[0]
                and abs(delta[1]) < half[1]
            ):
                return False
            if self.site_contact is not None:
                return bool(
                    _in_contact(
                        _entity(scene, self.object_name),
                        _entity(scene, self.site_contact),
                    )
                )
            return True
        if self.support is None:
            raise ValueError("libero_on needs either support or site_* params")
        support_pos = _entity_position(scene, self.support)
        if support_pos[2] > obj_pos[2]:
            return False
        if np.linalg.norm(support_pos[:2] - obj_pos[:2]) >= 0.03:
            return False
        return bool(
            _in_contact(_entity(scene, self.object_name), _entity(scene, self.support))
        )

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class LiberoInCondition(SuccessCondition):
    """LIBERO's ``In`` goal predicate against a BDDL region site.

    ``SiteObjectState.check_contain`` is contact-free (sites have no
    dynamics); the object's body origin must lie strictly inside the site's
    world box ``|xmat @ halfsize|``, with the lower z bound relaxed by
    0.01 m (LIBERO ``SiteObject.in_box``).
    """

    object_name: str
    site_center: tuple[float, float, float]
    site_xmat: tuple[float, ...]
    site_halfsize: tuple[float, float, float]
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _entity_position(scene, self.object_name)
        center = np.asarray(self.site_center, dtype=float)
        xmat = np.asarray(self.site_xmat, dtype=float).reshape(3, 3)
        half = np.asarray(self.site_halfsize, dtype=float)
        total = np.abs(xmat @ half)
        ub = center + total
        lb = center - total
        lb[2] -= 0.01
        return bool(np.all(obj_pos > lb) and np.all(obj_pos < ub))

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class LiberoJointStateCondition(SuccessCondition):
    """LIBERO's unary articulated-fixture predicates: ``Close``/``TurnOn``/``TurnOff``.

    Evaluated on the fixture's named joints against the object class's
    ``articulation`` ranges transcribed from the mirrored LIBERO source:
    ``close`` — every joint qpos above ``min(close_range)``;
    ``turn_on`` — any joint qpos at or above ``min(turnon_range)``;
    ``turn_off`` — every joint qpos below ``max(turnoff_range)``.
    """

    object_name: str
    joints: tuple[str, ...]
    mode: str
    range: tuple[float, float]
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        values = [_joint_qpos(scene, self.object_name, joint) for joint in self.joints]
        lo, hi = self.range
        if self.mode == "close":
            return all(v > lo for v in values)
        if self.mode == "turn_on":
            return any(v >= lo for v in values)
        if self.mode == "turn_off":
            return all(v < hi for v in values)
        raise ValueError(f"unknown libero_joint_state mode {self.mode!r}")

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class AllCondition(SuccessCondition):
    """Success when every sub-condition holds (LIBERO ``And`` goal trees)."""

    conditions: tuple[SuccessCondition, ...]
    _last_failure_stage: str = field(default="error", init=False)

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        for condition in self.conditions:
            if not condition.evaluate(scene, robot, info):
                self._last_failure_stage = condition.last_failure_stage()
                return False
        self._last_failure_stage = "error"
        return True

    def last_failure_stage(self) -> str:
        return self._last_failure_stage


# ---------------------------------------------------------------------------
# RoboCasa predicates (EFE robocasa / robocasa365 / robocasa-gr1)
# ---------------------------------------------------------------------------


def _world_entity_key(scene: Any) -> str:
    """Registry key of the whole-world entity backing a RoboCasa task."""
    entities = getattr(scene, "entities", None) or {}
    for key in ("world", "kitchen", "scene"):
        if key in entities:
            return key
    raise KeyError(
        "rc_* predicates need a whole-world entity ('world') in scene.entities; "
        f"available: {sorted(entities)}"
    )


def _world_entity(scene: Any) -> Any:
    """The whole-world entity backing a RoboCasa task (asset name ``world``)."""
    entities = getattr(scene, "entities", None) or {}
    return entities[_world_entity_key(scene)]


def _body_entity(scene: Any, name: str) -> Any:
    """Resolve a body name to an entity exposing position hooks.

    Whole-world spawns register one entity ('world'); bodies inside it are
    addressed by their MuJoCo body name through the link hooks. Tests may
    instead register one lightweight fake entity per body name.
    """
    entities = getattr(scene, "entities", None) or {}
    if name in entities:
        return entities[name]
    world = _world_entity(scene)
    if name in world.link_names():
        return _WorldBodyView(world, name)
    raise KeyError(
        f"no entity or world link named {name!r} " f"(entities: {sorted(entities)})"
    )


class _WorldBodyView:
    """Positional view of one body inside a whole-world entity."""

    def __init__(self, world: Any, name: str) -> None:
        self._world = world
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_pos(self) -> np.ndarray:
        return np.asarray(self._world.get_link_pos(self._name), dtype=float)

    def get_quat(self) -> np.ndarray:
        return np.asarray(self._world.get_link_quat(self._name), dtype=float)

    def in_contact(self, other: Any) -> bool:
        return _world_bodies_touch(self._world, self._name, other)


def _body_prefixes(world: Any, name: str) -> set[str]:
    """Link names belonging to one body (the body and its sub-bodies)."""
    prefix = name.rstrip("_") + "_"
    return {n for n in world.link_names() if n == name or n.startswith(prefix)}


def _world_bodies_touch(world: Any, name_a: str, other: Any) -> bool:
    """Contact between body/geoms ``name_a`` and ``other`` (name or entity).

    Link names match by prefix (a body and its sub-bodies — the gripper
    touches through its finger links); names that are geoms rather than
    bodies (microwave buttons are geoms on the door body) filter the
    contact pairs by geom name instead.
    """
    links = world.link_names()

    def identifiers(name: str | None) -> set[str] | None:
        if not name:
            return None
        if name in links:
            prefix = name.rstrip("_") + "_"
            return {n for n in links if n == name or n.startswith(prefix)}
        try:
            geoms = world.geom_names()
        except TypeError:
            geoms = []
        if name in geoms:
            return {name}
        return set()

    names_a = identifiers(name_a)
    if names_a is None:
        return False
    if isinstance(other, _WorldBodyView):
        other_name: str | None = other._name
        entity_b: Any = None
    elif isinstance(other, str):
        other_name, entity_b = other, None
    else:
        other_name, entity_b = None, other
    if entity_b is not None:
        return bool(world.in_contact(entity_b))
    names_b = identifiers(other_name)
    if not names_b:
        return False
    contacts = world.get_self_contacts()
    link_a = np.asarray(contacts.get("link_a", [])).reshape(-1)
    link_b = np.asarray(contacts.get("link_b", [])).reshape(-1)
    geom_a = np.asarray(contacts.get("geom_a", [])).reshape(-1)
    geom_b = np.asarray(contacts.get("geom_b", [])).reshape(-1)
    if link_a.size == 0:
        return False
    link_names = world.link_names()
    geom_names = world.geom_names()
    pairs: set[tuple[str, str]] = set()
    for i in range(link_a.size):
        a_link, b_link = link_names[link_a[i]], link_names[link_b[i]]
        a_geom = geom_names[geom_a[i]] if i < geom_a.size else ""
        b_geom = geom_names[geom_b[i]] if i < geom_b.size else ""
        for a in (a_link, a_geom):
            for b in (b_link, b_geom):
                pairs.add((a, b))
                pairs.add((b, a))
    return any(a in names_a and b in names_b for a, b in pairs)


def _site_world_pos(scene: Any, site: str) -> np.ndarray:
    """World position of a baked site (eef / button / marker).

    Sites do not exist in Genesis; the RoboCasa task records each needed
    site as (body, local offset) at reset time and serves it through
    ``scene.entities['world']._sites`` — see ``EfeRobocasaTask``.
    """
    world = _world_entity(scene)
    sites = getattr(world, "_rc_sites", None) or {}
    if site not in sites:
        raise KeyError(
            f"site {site!r} not registered on the world entity; "
            f"available: {sorted(sites)}"
        )
    body, local = sites[site]
    pos = np.asarray(world.get_link_pos(body), dtype=float)
    quat = np.asarray(world.get_link_quat(body), dtype=float)
    offset = _quat_to_mat(quat) @ np.asarray(local, dtype=float)
    return np.asarray(pos + offset, dtype=float)


@dataclass
class RcContactCondition(SuccessCondition):
    """RoboCasa ``env.check_contact(obj, fixture)`` — any touching geom pair."""

    object_name: str
    fixture: str
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        entities = getattr(scene, "entities", None) or {}
        if self.object_name in entities and self.fixture in entities:
            return bool(_in_contact(entities[self.object_name], entities[self.fixture]))
        return _world_bodies_touch(_world_entity(scene), self.object_name, self.fixture)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcInReceptacleCondition(SuccessCondition):
    """RoboCasa ``OU.check_obj_in_receptacle`` (upstream-exact).

    Contact between the object and the receptacle, and the horizontal
    distance between their body origins below the threshold — upstream
    defaults the threshold to ``0.7 * receptacle.horizontal_radius``; the
    resolver bakes the radius (and the explicit ``th`` where the env passes
    one) so no geometry is needed at runtime.
    """

    object_name: str
    receptacle: str
    radius: float | None = None
    factor: float = 0.7
    threshold: float | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        if not RcContactCondition(self.object_name, self.receptacle).evaluate(
            scene, robot, info
        ):
            return False
        obj_pos = _body_entity(scene, self.object_name).get_pos()
        rec_pos = _body_entity(scene, self.receptacle).get_pos()
        th = (
            self.threshold
            if self.threshold is not None
            else self.factor * float(self.radius or 0.0)
        )
        return bool(np.linalg.norm(obj_pos[:2] - rec_pos[:2]) < th)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcGripperFarCondition(SuccessCondition):
    """RoboCasa ``OU.gripper_obj_far`` / ``OU.any_gripper_obj_far``.

    The gripper site (baked; both sites for the GR1 two-arm variant) must
    be farther than ``threshold`` from the target body origin. With two
    sites, ANY site satisfying the distance counts (upstream ``any``).
    """

    target: str
    sites: tuple[str, ...]
    threshold: float = 0.25
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        world = _world_entity(scene)
        sites = getattr(world, "_rc_sites", None) or {}
        target_pos = (
            _site_world_pos(scene, self.target)
            if self.target in sites
            else _body_entity(scene, self.target).get_pos()
        )
        distances = [
            float(np.linalg.norm(_site_world_pos(scene, s) - target_pos))
            for s in self.sites
        ]
        return any(d > self.threshold for d in distances)

    def last_failure_stage(self) -> str:
        return self.failure_stage


_RC_JOINT_MODES = ("normalized", "abs", "mod2pi", "spout_ori")


@dataclass
class RcJointStateCondition(SuccessCondition):
    """RoboCasa articulated-fixture joint checks (upstream-exact).

    ``normalized`` — cabinet doors and drawers: the joint qpos is mapped
    through its baked ``range`` to [0, 1] and compared against
    ``[min, max]`` (open >= 0.90 / close <= 0.05 style).
    ``abs`` — stove knobs: ``min <= |qpos| <= max`` (on-window rad).
    ``mod2pi`` — sink handle: ``(qpos mod 2π)`` in the open min/max window.
    ``spout_ori`` — sink spout orientation bucket (left/right/center).
    """

    joint: str
    mode: str
    range: tuple[float, float] = (0.0, 1.0)
    min: float = 0.0
    max: float = 1.0
    sign: float = 1.0
    orientation: str = ""
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        world_key = _world_entity_key(scene)
        qpos = float(_joint_qpos(scene, world_key, self.joint))
        q = qpos * self.sign
        if self.mode == "normalized":
            lo, hi = self.range
            v = (q - lo) / (hi - lo) if hi != lo else 0.0
            return bool(self.min <= v <= self.max)
        if self.mode == "abs":
            return bool(self.min <= abs(q) <= self.max)
        if self.mode == "mod2pi":
            v = q % (2.0 * np.pi)
            return bool(self.min < v < self.max)
        if self.mode == "spout_ori":
            v = q % (2.0 * np.pi)
            if np.pi <= v <= 2.0 * np.pi - np.pi / 6.0:
                ori = "left"
            elif np.pi / 6.0 <= v <= np.pi:
                ori = "right"
            else:
                ori = "center"
            return ori == self.orientation
        raise ValueError(f"unknown rc_joint_state mode {self.mode!r}")

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcApplianceCondition(SuccessCondition):
    """Toggle-appliance state (microwave / coffee machine / toaster oven).

    The upstream fixture flips its ``turned_on`` flag from gripper-button
    contact each step; :class:`EfeRobocasaTask` mirrors that state machine
    and publishes the flags in ``info['rc_appliance']``. Missing info is a
    failure (nop-safe).
    """

    name: str
    state: str = "on"
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        states = (info or {}).get("rc_appliance") or {}
        return bool(states.get(self.name, False)) == (self.state == "on")

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcInsideCondition(SuccessCondition):
    """RoboCasa ``OU.obj_inside_of`` (upstream-exact, both site and region).

    v0.2 / GR1 fixtures expose four ``int_p*`` sites; robocasa 1.0.1
    (RoboCasa365) instead defines ``reg_*`` region geoms and ``obj_inside_of``
    succeeds when ANY region holds every checked point. The object's body
    origin (``partial_check``) or all eight oriented bounding-box corners
    (full check, 0.05 m tolerance) must project inside the box. The resolver
    bakes fixture sites / region boxes and the object bbox from the library
    XML, so the runtime only transforms baked locals into the world.
    """

    object_name: str
    fixture: str
    fixture_sites: tuple[str, ...] = ()  # p0, px, py, pz site names
    regions: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ] = ()
    object_bbox: tuple[tuple[float, float, float], ...] | None = None  # 8 corners
    tolerance: float = 0.0
    failure_stage: str = "error"

    def _points_inside(
        self,
        p0: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
        points: list[np.ndarray],
        th: float,
    ) -> bool:
        u, v, w = px - p0, py - p0, pz - p0
        for point in points:
            checks = (
                np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th,
                np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th,
                np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th,
            )
            if not all(checks):
                return False
        return True

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj = _body_entity(scene, self.object_name)
        obj_pos = obj.get_pos()
        if self.object_bbox is None:
            points = [obj_pos]
            th = 0.0
        else:
            rot = _quat_to_mat(obj.get_quat())
            points = [
                obj_pos + rot @ np.asarray(corner, dtype=float)
                for corner in self.object_bbox
            ]
            th = self.tolerance
        if self.regions:
            rot = _quat_to_mat(
                np.asarray(_body_entity(scene, self.fixture).get_quat(), dtype=float)
            )
            base = np.asarray(_body_entity(scene, self.fixture).get_pos(), dtype=float)
            for center, half in self.regions:
                c = np.asarray(center, dtype=float)
                h = np.asarray(half, dtype=float)

                def corner(signs: tuple[float, float, float]) -> np.ndarray:
                    delta = c + h * np.asarray(signs, dtype=float)
                    return np.asarray(base + rot @ delta, dtype=float)

                p0 = corner((-1.0, -1.0, -1.0))
                px = corner((1.0, -1.0, -1.0))
                py = corner((-1.0, 1.0, -1.0))
                pz = corner((-1.0, -1.0, 1.0))
                if self._points_inside(p0, px, py, pz, points, th):
                    return True
            return False
        if not self.fixture_sites:
            raise ValueError("rc_inside needs either fixture_sites or regions")
        p0 = _site_world_pos(scene, self.fixture_sites[0])
        px = _site_world_pos(scene, self.fixture_sites[1])
        py = _site_world_pos(scene, self.fixture_sites[2])
        pz = _site_world_pos(scene, self.fixture_sites[3])
        return self._points_inside(p0, px, py, pz, points, th)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcUprightCondition(SuccessCondition):
    """RoboCasa ``OU.check_obj_upright``: body z-axis cosine above threshold."""

    object_name: str
    threshold: float = 0.8
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        rot = _quat_to_mat(_body_entity(scene, self.object_name).get_quat())
        return bool(rot[2, 2] > self.threshold)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcNearCondition(SuccessCondition):
    """Horizontal-distance checks used by several composites.

    ``target`` may be a world body or a baked site name (coffee machine
    pour site). Upstream compares ``body_xpos`` components against a baked
    radius (e.g. ``bowl_radius + 0.25``, ``<= 0.3``); ``axes`` selects the
    xy plane (upstream ``[:2]`` norms) or full 3D.
    """

    object_name: str
    target: str
    radius: float
    axes: int = 2
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _body_entity(scene, self.object_name).get_pos()
        world = _world_entity(scene)
        sites = getattr(world, "_rc_sites", None) or {}
        if self.target in sites:
            target_pos = _site_world_pos(scene, self.target)
        else:
            target_pos = _body_entity(scene, self.target).get_pos()
        delta = obj_pos - target_pos
        return bool(
            np.linalg.norm(delta if self.axes == 3 else delta[:2]) <= self.radius
        )

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcCloserCondition(SuccessCondition):
    """Comparative distance (RestockPantry ``_close_to_cab_cans``).

    ``|object − anchor| * ratio < |other − object|`` — the object sits
    closer to the anchor than the other object is (3D, upstream norm).
    """

    object_name: str
    anchor: str
    other: str
    ratio: float = 2.0
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _body_entity(scene, self.object_name).get_pos()
        anchor_pos = _body_entity(scene, self.anchor).get_pos()
        other_pos = _body_entity(scene, self.other).get_pos()
        return bool(
            np.linalg.norm(obj_pos - anchor_pos) * self.ratio
            < np.linalg.norm(other_pos - obj_pos)
        )

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcContactAnyCondition(SuccessCondition):
    """``OU.check_obj_any_counter_contact``: obj touches ANY of the fixtures."""

    object_name: str
    fixtures: tuple[str, ...]
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        for fixture in self.fixtures:
            if RcContactCondition(self.object_name, fixture).evaluate(
                scene, robot, info
            ):
                return True
        return False

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcTimerCondition(SuccessCondition):
    """Step-counting upstream checks (washed > 10, broiled >= 10, scrub >= 15).

    :class:`EfeRobocasaTask` evaluates each timer's condition block every
    step (same call pattern as upstream, twice per step), counts
    consecutive successes and publishes ``info['rc_timer']``. Missing info
    is a failure (nop-safe).
    """

    name: str
    count: int
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        timers = (info or {}).get("rc_timer") or {}
        return int(timers.get(self.name, 0)) >= self.count

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcBasinCondition(SuccessCondition):
    """Double-basin sink side check (``get_obj_basin_loc``).

    All eight oriented bbox corners of the object (0.05 m tolerance, as
    upstream) must fall inside the named basin's region box, expressed in
    the sink fixture frame. The region center/halfsize and the object bbox
    are baked at resolve time.
    """

    object_name: str
    fixture_body: str
    side: str  # "left" | "right"
    center_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    halfsize_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_bbox: tuple[tuple[float, float, float], ...] | None = None
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        fixture = _body_entity(scene, self.fixture_body)
        rot_t = _quat_to_mat(fixture.get_quat()).T
        center_w = fixture.get_pos() + _quat_to_mat(fixture.get_quat()) @ np.asarray(
            self.center_local, dtype=float
        )
        obj = _body_entity(scene, self.object_name)
        obj_pos = obj.get_pos()
        points = (
            [obj_pos]
            if self.object_bbox is None
            else [
                obj_pos + _quat_to_mat(obj.get_quat()) @ np.asarray(c, dtype=float)
                for c in self.object_bbox
            ]
        )
        half = np.asarray(self.halfsize_local, dtype=float)
        th = 0.05
        for point in points:
            local = rot_t @ (point - center_w)
            if not np.all(np.abs(local) <= half + th):
                return False
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcOnBurnerCondition(SuccessCondition):
    """``check_obj_location_on_stove``: object sits on a burner.

    The object origin must be within ``threshold`` (baked) of the burner's
    world xy; with ``require_knob_on`` the burner's knob joint must also be
    inside the upstream on-window.
    """

    object_name: str
    burner_site: str
    threshold: float
    knob_joint: str = ""
    require_knob_on: bool = False
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _body_entity(scene, self.object_name).get_pos()
        burner_pos = _site_world_pos(scene, self.burner_site)
        if np.linalg.norm(obj_pos[:2] - burner_pos[:2]) > self.threshold:
            return False
        if self.require_knob_on and self.knob_joint:
            world = _world_entity(scene)
            value = abs(float(world.get_joint_position(self.knob_joint)))
            return bool(0.35 <= value <= 2.0 * np.pi - 0.35)
        return True

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class AnyCondition(SuccessCondition):
    """Success when ANY sub-condition holds (upstream ``or`` branches)."""

    conditions: tuple[SuccessCondition, ...]
    _last_failure_stage: str = field(default="error", init=False)

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        for condition in self.conditions:
            if condition.evaluate(scene, robot, info):
                self._last_failure_stage = "error"
                return True
        self._last_failure_stage = self.conditions[-1].last_failure_stage()
        return False

    def last_failure_stage(self) -> str:
        return self._last_failure_stage


@dataclass
class NotCondition(SuccessCondition):
    """Success when the sub-condition does NOT hold (upstream ``not``)."""

    condition: SuccessCondition
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        return not self.condition.evaluate(scene, robot, info)

    def last_failure_stage(self) -> str:
        return self.failure_stage


@dataclass
class RcAtSpawnCondition(SuccessCondition):
    """GR1 tiered containers: the object sits in the target spawn region.

    Upstream compares ``container.closest_spawn_id(obj)`` with the task's
    target site — i.e. the container spawn site nearest the object origin
    must be one of the flagged target sites. Spawn sites are baked
    (body, local) pairs.
    """

    object_name: str
    spawn_sites: tuple[tuple[str, bool], ...]  # (site name, is_target)
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        obj_pos = _body_entity(scene, self.object_name).get_pos()
        nearest: tuple[float, bool] | None = None
        for site_name, is_target in self.spawn_sites:
            dist = float(np.linalg.norm(_site_world_pos(scene, site_name) - obj_pos))
            if nearest is None or dist < nearest[0]:
                nearest = (dist, is_target)
        return bool(nearest is not None and nearest[1])

    def last_failure_stage(self) -> str:
        return self.failure_stage


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _require(params: dict, key: str, success_type: str) -> Any:
    if key not in params:
        raise ConfigError(f"success type '{success_type}' requires params.{key}")
    return params[key]


def _build_distance(params: dict) -> DistanceThresholdCondition:
    object_name = _require(params, "object", "distance_threshold")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    weights = params.get("axis_weights", [1.0, 1.0, 1.0])
    if (
        not isinstance(weights, (list, tuple))
        or len(weights) != 3
        or any(not isinstance(w, (int, float)) or w <= 0 for w in weights)
    ):
        raise ConfigError(
            f"params.axis_weights must be three positive numbers, got {weights!r}"
        )
    return DistanceThresholdCondition(
        object_name=object_name,
        target_position=_vec(
            _require(params, "target_position", "distance_threshold"),
            3,
            "params.target_position",
        ),
        threshold=_positive_float(
            _require(params, "threshold", "distance_threshold"), "params.threshold"
        ),
        axis_weights=(float(weights[0]), float(weights[1]), float(weights[2])),
        failure_stage=_failure_stage(params),
    )


def _build_pose_window(params: dict) -> PoseWindowCondition:
    object_name = _require(params, "object", "pose_window")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    orientation = params.get("target_orientation")
    return PoseWindowCondition(
        object_name=object_name,
        target_position=_vec(
            _require(params, "target_position", "pose_window"),
            3,
            "params.target_position",
        ),
        target_orientation=(
            _vec(orientation, 4, "params.target_orientation")
            if orientation is not None
            else None
        ),
        pos_threshold=_positive_float(
            params.get("pos_threshold", 0.02), "params.pos_threshold"
        ),
        rot_threshold_deg=_positive_float(
            params.get("rot_threshold_deg", 10.0), "params.rot_threshold_deg"
        ),
        failure_stage=_failure_stage(params),
    )


def _build_axis_window(params: dict) -> AxisWindowCondition:
    object_name = _require(params, "object", "axis_window")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    raw_windows = _require(params, "windows", "axis_window")
    if not isinstance(raw_windows, dict) or not raw_windows:
        raise ConfigError(
            "params.windows must be a non-empty mapping of axis -> [lo, hi]"
        )
    windows: dict[str, tuple[float | None, float | None]] = {}
    for axis, bounds in raw_windows.items():
        if axis not in ("x", "y", "z"):
            raise ConfigError(
                f"params.windows keys must be one of ['x', 'y', 'z'], got {axis!r}"
            )
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or bounds[0] is None
            and bounds[1] is None
        ):
            raise ConfigError(
                f"params.windows[{axis!r}] must be [lo, hi] with at least one "
                f"finite bound, got {bounds!r}"
            )
        parsed: list[float | None] = []
        for bound in bounds:
            if bound is None:
                parsed.append(None)
                continue
            try:
                value = float(bound)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"params.windows[{axis!r}] bounds must be numbers or null, "
                    f"got {bounds!r}"
                ) from exc
            if not np.isfinite(value):
                raise ConfigError(
                    f"params.windows[{axis!r}] bounds must be finite, got {bounds!r}"
                )
            parsed.append(value)
        lo, hi = parsed
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(
                f"params.windows[{axis!r}] must have lo <= hi, got {bounds!r}"
            )
        windows[axis] = (lo, hi)
    anchor = params.get("target_position")
    return AxisWindowCondition(
        object_name=object_name,
        windows=windows,
        target_position=(
            _vec(anchor, 3, "params.target_position") if anchor is not None else None
        ),
        failure_stage=_failure_stage(params),
    )


def _region_tuple(value: Any, name: str = "params.region") -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ConfigError(f"{name} must be a sequence of 6 numbers, got {value!r}")
    try:
        region = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} must be a sequence of 6 numbers, got {value!r}"
        ) from exc
    if not all(np.isfinite(v) for v in region):
        raise ConfigError(f"{name} must contain only finite numbers, got {value!r}")
    return region


def _build_body_point_near(params: dict) -> BodyPointNearCondition:
    object_name = _require(params, "object", "body_point_near")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    weights = params.get("axis_weights", [1.0, 1.0, 1.0])
    if (
        not isinstance(weights, (list, tuple))
        or len(weights) != 3
        or any(not isinstance(w, (int, float)) or w <= 0 for w in weights)
    ):
        raise ConfigError(
            f"params.axis_weights must be three positive numbers, got {weights!r}"
        )
    return BodyPointNearCondition(
        object_name=object_name,
        offset=_vec(_require(params, "offset", "body_point_near"), 3, "params.offset"),
        target_position=_vec(
            _require(params, "target_position", "body_point_near"),
            3,
            "params.target_position",
        ),
        threshold=_positive_float(
            _require(params, "threshold", "body_point_near"), "params.threshold"
        ),
        axis_weights=(float(weights[0]), float(weights[1]), float(weights[2])),
        failure_stage=_failure_stage(params),
    )


def _build_in_region(params: dict) -> InRegionCondition:
    object_name = _require(params, "object", "in_region")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    return InRegionCondition(
        object_name=object_name,
        region=_region_tuple(_require(params, "region", "in_region")),  # type: ignore[arg-type]
        failure_stage=_failure_stage(params),
    )


def _build_count_in_region(params: dict) -> CountInRegionCondition:
    objects = params.get("objects")
    prefix = params.get("prefix")
    if objects is None and prefix is None:
        raise ConfigError(
            "success type 'count_in_region' requires params.objects or params.prefix"
        )
    if objects is not None:
        if (
            not isinstance(objects, list)
            or not objects
            or any(not isinstance(name, str) for name in objects)
        ):
            raise ConfigError(
                f"params.objects must be a non-empty list of names, got {objects!r}"
            )
    if prefix is not None and not isinstance(prefix, str):
        raise ConfigError(f"params.prefix must be a string, got {prefix!r}")
    min_count = _require(params, "min_count", "count_in_region")
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ConfigError(
            f"params.min_count must be a positive integer, got {min_count!r}"
        )
    return CountInRegionCondition(
        objects=tuple(objects) if objects is not None else None,
        prefix=prefix,
        region=_region_tuple(_require(params, "region", "count_in_region")),  # type: ignore[arg-type]
        min_count=min_count,
        failure_stage=_failure_stage(params),
    )


def _build_ee_near(params: dict) -> EeNearCondition:
    target = params.get("target_position")
    if target == "from_site":
        raise ConfigError(
            "ee_near requires an explicit params.target_position "
            "(there is no scene object to anchor to)"
        )
    link = params.get("link")
    if link is not None and not isinstance(link, str):
        raise ConfigError(f"params.link must be a string, got {link!r}")
    return EeNearCondition(
        target_position=_vec(
            _require(params, "target_position", "ee_near"),
            3,
            "params.target_position",
        ),
        threshold=_positive_float(
            _require(params, "threshold", "ee_near"), "params.threshold"
        ),
        link=link,
        failure_stage=_failure_stage(params),
    )


def _build_above_region(params: dict) -> AboveRegionCondition:
    object_name = _require(params, "object", "above_region")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    platform = _require(params, "platform", "above_region")
    if not isinstance(platform, str):
        raise ConfigError("params.platform must be a string")
    circle = params.get("circle")
    region = params.get("region")
    if (circle is None) == (region is None):
        raise ConfigError(
            "success type 'above_region' requires exactly one of "
            "params.circle (radius, platform-centred) or params.region "
            "([xmin, xmax, ymin, ymax])"
        )
    z_offset = params.get("z_offset", 0.0)
    try:
        z_offset = float(z_offset)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"params.z_offset must be a number, got {z_offset!r}"
        ) from exc
    circle_f: float | None = None
    if circle is not None:
        circle_f = _positive_float(circle, "params.circle")
    region_t: tuple[float, float, float, float] | None = None
    if region is not None:
        region_t = _vec(region, 4, "params.region")
    return AboveRegionCondition(
        object_name=object_name,
        platform=platform,
        circle=circle_f,
        region=region_t,
        z_offset=z_offset,
        failure_stage=_failure_stage(params),
    )


def _build_tipped(params: dict) -> TippedCondition:
    object_name = _require(params, "object", "tipped")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    top_offset = _vec(_require(params, "top_offset", "tipped"), 3, "params.top_offset")
    bottom_offset = _vec(
        _require(params, "bottom_offset", "tipped"), 3, "params.bottom_offset"
    )
    threshold = params.get("threshold", 0.0)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"params.threshold must be a number, got {threshold!r}"
        ) from exc
    if threshold < 0:
        raise ConfigError(f"params.threshold must be >= 0, got {threshold!r}")
    return TippedCondition(
        object_name=object_name,
        top_offset=top_offset,
        bottom_offset=bottom_offset,
        threshold=threshold,
        failure_stage=_failure_stage(params),
    )


def _build_contains(params: dict) -> ContainsCondition:
    container = _require(params, "container", "contains")
    if not isinstance(container, str):
        raise ConfigError("params.container must be a string")
    objects = _require(params, "objects", "contains")
    if (
        not isinstance(objects, list)
        or not objects
        or any(not isinstance(name, str) for name in objects)
    ):
        raise ConfigError(
            f"params.objects must be a non-empty list of names, got {objects!r}"
        )
    mode = params.get("mode", "circle")
    if mode not in ("circle", "boxes"):
        raise ConfigError(f"params.mode must be 'circle' or 'boxes', got {mode!r}")
    circle: float = 0.0
    z_band: tuple[float, float] = (0.0, 0.1)
    regions: tuple[dict, ...] = ()
    if mode == "circle":
        circle = _positive_float(params.get("circle", 0.0), "params.circle")
        z_vec = _vec(params.get("z_band", (0.0, 0.1)), 2, "params.z_band")
        z_band = (float(z_vec[0]), float(z_vec[1]))
    else:
        raw_regions = params.get("regions")
        if (
            not isinstance(raw_regions, list)
            or not raw_regions
            or any(
                not isinstance(r, dict) or "min" not in r or "max" not in r
                for r in raw_regions
            )
        ):
            raise ConfigError(
                "params.regions must be a non-empty list of {center, min, max}"
            )
        regions = tuple(raw_regions)
    return ContainsCondition(
        container=container,
        objects=tuple(objects),
        mode=mode,
        circle=circle,
        z_band=(float(z_band[0]), float(z_band[1])),
        regions=regions,
        negate=bool(params.get("negate", False)),
        failure_stage=_failure_stage(params),
    )


def _build_lift(params: dict) -> LiftCondition:
    objects = _require(params, "objects", "lift")
    if (
        not isinstance(objects, list)
        or not objects
        or any(not isinstance(name, str) for name in objects)
    ):
        raise ConfigError(
            f"params.objects must be a non-empty list of names, got {objects!r}"
        )
    min_height = params.get("min_height", params.get("target_height"))
    if min_height is None:
        raise ConfigError("success type 'lift' requires params.min_height")
    return LiftCondition(
        objects=tuple(objects),
        min_height=float(min_height),
        failure_stage=_failure_stage(params),
    )


def _build_ordered(params: dict) -> OrderedCondition:
    objects = _require(params, "objects", "ordered")
    if (
        not isinstance(objects, list)
        or len(objects) < 2
        or any(not isinstance(name, str) for name in objects)
    ):
        raise ConfigError(
            f"params.objects must be a list of >= 2 names, got {objects!r}"
        )
    axes = params.get("axes", (0,))
    if (
        not isinstance(axes, list)
        or not axes
        or any(not isinstance(a, int) or a not in (0, 1, 2) for a in axes)
    ):
        raise ConfigError(
            f"params.axes must be a non-empty list of 0/1/2, got {axes!r}"
        )
    offset = params.get("offset", 0.1)
    try:
        offset = float(offset)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"params.offset must be a number, got {offset!r}") from exc
    return OrderedCondition(
        objects=tuple(objects),
        axes=tuple(axes),
        offset=offset,
        failure_stage=_failure_stage(params),
    )


def _build_joint_window(params: dict) -> JointWindowCondition:
    object_name = _require(params, "object", "joint_window")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    window = _vec(_require(params, "window", "joint_window"), 2, "params.window")
    dof = params.get("dof", 0)
    if not isinstance(dof, int) or isinstance(dof, bool) or dof < 0:
        raise ConfigError(f"params.dof must be a non-negative integer, got {dof!r}")
    return JointWindowCondition(
        object_name=object_name,
        window=(float(window[0]), float(window[1])),
        dof=dof,
        failure_stage=_failure_stage(params),
    )


def _build_in_contact(params: dict) -> InContactCondition:
    objects = _require(params, "objects", "in_contact")
    if (
        not isinstance(objects, list)
        or not objects
        or any(not isinstance(name, str) for name in objects)
    ):
        raise ConfigError(
            f"params.objects must be a non-empty list of names, got {objects!r}"
        )
    with_entity = params.get("with_entity", "@robot")
    if not isinstance(with_entity, str):
        raise ConfigError(f"params.with_entity must be a string, got {with_entity!r}")
    return InContactCondition(
        objects=tuple(objects),
        with_entity=with_entity,
        failure_stage=_failure_stage(params),
    )


def _build_staged(params: dict) -> StagedCondition:
    raw_stages = _require(params, "stages", "staged")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ConfigError(
            "params.stages must be a non-empty list of {stage, condition}"
        )
    stages: list[tuple[str, SuccessCondition]] = []
    for index, entry in enumerate(raw_stages):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"params.stages[{index}] must be a mapping, got {entry!r}"
            )
        stage = entry.get("stage")
        if stage not in FAIL_STAGES:
            raise ConfigError(
                f"params.stages[{index}].stage must be one of {list(FAIL_STAGES)}, "
                f"got {stage!r}"
            )
        stages.append((stage, create_success_condition(entry.get("condition") or {})))
    return StagedCondition(stages=stages)


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    """Coerce a validated 3-sequence to a fixed-length float tuple."""
    return (float(value[0]), float(value[1]), float(value[2]))


def _build_libero_on(params: dict) -> LiberoOnCondition:
    object_name = _require(params, "object", "libero_on")
    if not isinstance(object_name, str):
        raise ConfigError("params.object must be a string")
    support = params.get("support")
    if support is not None and not isinstance(support, str):
        raise ConfigError("params.support must be a string")
    site_center = params.get("site_center")
    if site_center is None and support is None:
        raise ConfigError("libero_on needs params.support or params.site_center")
    site_xmat = params.get("site_xmat")
    site_halfsize = params.get("site_halfsize")
    if site_center is not None:
        for key, value in (("site_xmat", site_xmat), ("site_halfsize", site_halfsize)):
            if value is None:
                raise ConfigError(f"libero_on site mode needs params.{key}")
        if len(site_xmat) != 9:  # type: ignore[arg-type]
            raise ConfigError("params.site_xmat must have 9 entries")
        if len(site_halfsize) != 3:  # type: ignore[arg-type]
            raise ConfigError("params.site_halfsize must have 3 entries")
    site_contact = params.get("site_contact")
    if site_contact is not None and not isinstance(site_contact, str):
        raise ConfigError("params.site_contact must be a string")
    return LiberoOnCondition(
        object_name=object_name,
        support=support,
        site_center=_vec3(site_center, "site_center") if site_center else None,
        site_xmat=tuple(float(v) for v in site_xmat) if site_xmat else None,
        site_halfsize=_vec3(site_halfsize, "site_halfsize") if site_halfsize else None,
        site_contact=site_contact,
        failure_stage=_failure_stage(params),
    )


def _build_libero_in(params: dict) -> LiberoInCondition:
    object_name = _require(params, "object", "libero_in")
    site_center = _require(params, "site_center", "libero_in")
    site_xmat = _require(params, "site_xmat", "libero_in")
    site_halfsize = _require(params, "site_halfsize", "libero_in")
    if len(site_xmat) != 9:
        raise ConfigError("params.site_xmat must have 9 entries")
    if len(site_halfsize) != 3:
        raise ConfigError("params.site_halfsize must have 3 entries")
    return LiberoInCondition(
        object_name=object_name,
        site_center=_vec3(site_center, "site_center"),
        site_xmat=tuple(float(v) for v in site_xmat),
        site_halfsize=_vec3(site_halfsize, "site_halfsize"),
        failure_stage=_failure_stage(params),
    )


_LIBERO_JOINT_MODES = ("close", "turn_on", "turn_off")


def _build_libero_joint_state(params: dict) -> LiberoJointStateCondition:
    object_name = _require(params, "entity", "libero_joint_state")
    joints = _require(params, "joints", "libero_joint_state")
    if not isinstance(joints, (list, tuple)) or not joints:
        raise ConfigError("params.joints must be a non-empty list of joint names")
    mode = _require(params, "mode", "libero_joint_state")
    if mode not in _LIBERO_JOINT_MODES:
        raise ConfigError(
            f"params.mode must be one of {list(_LIBERO_JOINT_MODES)}, got {mode!r}"
        )
    window = _require(params, "range", "libero_joint_state")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ConfigError("params.range must be a [lo, hi] pair")
    return LiberoJointStateCondition(
        object_name=object_name,
        joints=tuple(str(j) for j in joints),
        mode=mode,
        range=(float(window[0]), float(window[1])),
        failure_stage=_failure_stage(params),
    )


def _build_rc_contact(params: dict) -> RcContactCondition:
    return RcContactCondition(
        object_name=str(_require(params, "object", "rc_contact")),
        fixture=str(_require(params, "fixture", "rc_contact")),
        failure_stage=_failure_stage(params),
    )


def _build_rc_in_receptacle(params: dict) -> RcInReceptacleCondition:
    radius = params.get("radius")
    threshold = params.get("threshold")
    return RcInReceptacleCondition(
        object_name=str(_require(params, "object", "rc_in_receptacle")),
        receptacle=str(_require(params, "receptacle", "rc_in_receptacle")),
        radius=None if radius is None else float(radius),
        factor=float(params.get("factor", 0.7)),
        threshold=None if threshold is None else float(threshold),
        failure_stage=_failure_stage(params),
    )


def _build_rc_gripper_far(params: dict) -> RcGripperFarCondition:
    sites = params.get("sites")
    if (
        not isinstance(sites, list)
        or not sites
        or not all(isinstance(s, str) for s in sites)
    ):
        raise ConfigError("rc_gripper_far needs params.sites (non-empty list of names)")
    return RcGripperFarCondition(
        target=str(_require(params, "target", "rc_gripper_far")),
        sites=tuple(sites),
        threshold=float(params.get("threshold", 0.25)),
        failure_stage=_failure_stage(params),
    )


def _build_rc_joint_state(params: dict) -> RcJointStateCondition:
    mode = params.get("mode", "normalized")
    if mode not in _RC_JOINT_MODES:
        raise ConfigError(
            f"params.mode must be one of {list(_RC_JOINT_MODES)}, got {mode!r}"
        )
    window = params.get("range", [0.0, 1.0])
    if (
        not isinstance(window, (list, tuple))
        or len(window) != 2
        or any(not isinstance(v, (int, float)) for v in window)
    ):
        raise ConfigError(
            f"rc_joint_state params.range must be two numbers, got {window!r}"
        )
    orientation = params.get("orientation", "")
    if mode == "spout_ori" and orientation not in ("left", "right", "center"):
        raise ConfigError(
            f"rc_joint_state spout_ori needs params.orientation left/right/center, got {orientation!r}"
        )
    return RcJointStateCondition(
        joint=str(_require(params, "joint", "rc_joint_state")),
        mode=mode,
        range=(float(window[0]), float(window[1])),
        min=float(params.get("min", 0.0)),
        max=float(params.get("max", 1.0)),
        sign=float(params.get("sign", 1.0)),
        orientation=str(orientation),
        failure_stage=_failure_stage(params),
    )


def _build_rc_appliance(params: dict) -> RcApplianceCondition:
    state = params.get("state", "on")
    if state not in ("on", "off"):
        raise ConfigError(f"rc_appliance params.state must be on/off, got {state!r}")
    return RcApplianceCondition(
        name=str(_require(params, "name", "rc_appliance")),
        state=str(state),
        failure_stage=_failure_stage(params),
    )


def _build_rc_inside(params: dict) -> RcInsideCondition:
    sites = params.get("fixture_sites")
    regions_raw = params.get("regions")
    regions: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    if regions_raw is not None:
        if not isinstance(regions_raw, list) or not regions_raw:
            raise ConfigError("rc_inside params.regions must be a non-empty list")
        for entry in regions_raw:
            if not isinstance(entry, dict):
                raise ConfigError("rc_inside region entries must be mappings")
            regions.append(
                (
                    _vec(_require(entry, "center", "rc_inside.regions"), 3, "center"),
                    _vec(
                        _require(entry, "halfsize", "rc_inside.regions"), 3, "halfsize"
                    ),
                )
            )
    fixture_sites: tuple[str, ...] = ()
    if sites is not None:
        if (
            not isinstance(sites, list)
            or len(sites) != 4
            or not all(isinstance(s, str) for s in sites)
        ):
            raise ConfigError(
                "rc_inside needs params.fixture_sites (4 int-site names p0/px/py/pz)"
            )
        fixture_sites = (sites[0], sites[1], sites[2], sites[3])
    if not fixture_sites and not regions:
        raise ConfigError(
            "rc_inside needs either params.fixture_sites or params.regions"
        )
    bbox = params.get("object_bbox")
    corners = None
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 8:
            raise ConfigError("rc_inside params.object_bbox must be 8 corners")
        corners = tuple(
            _vec(corner, 3, "rc_inside params.object_bbox[]") for corner in bbox
        )
    return RcInsideCondition(
        object_name=str(_require(params, "object", "rc_inside")),
        fixture=str(_require(params, "fixture", "rc_inside")),
        fixture_sites=fixture_sites,
        regions=tuple(regions),
        object_bbox=corners,
        tolerance=float(params.get("tolerance", 0.0)),
        failure_stage=_failure_stage(params),
    )


def _build_rc_upright(params: dict) -> RcUprightCondition:
    return RcUprightCondition(
        object_name=str(_require(params, "object", "rc_upright")),
        threshold=float(params.get("threshold", 0.8)),
        failure_stage=_failure_stage(params),
    )


def _build_rc_at_spawn(params: dict) -> RcAtSpawnCondition:
    raw = params.get("spawn_sites")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            "rc_at_spawn needs params.spawn_sites (list of {site, target})"
        )
    pairs: list[tuple[str, bool]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ConfigError("rc_at_spawn spawn_sites entries must be mappings")
        site = entry.get("site")
        if not isinstance(site, str):
            raise ConfigError("rc_at_spawn spawn_sites entries need a site name")
        target = entry.get("target", False)
        pairs.append((site, bool(target)))
    return RcAtSpawnCondition(
        object_name=str(_require(params, "object", "rc_at_spawn")),
        spawn_sites=tuple(pairs),
        failure_stage=_failure_stage(params),
    )


def _build_rc_near(params: dict) -> RcNearCondition:
    axes = int(params.get("axes", 2))
    if axes not in (2, 3):
        raise ConfigError(f"rc_near params.axes must be 2 or 3, got {axes!r}")
    return RcNearCondition(
        object_name=str(_require(params, "object", "rc_near")),
        target=str(_require(params, "target", "rc_near")),
        radius=_positive_float(_require(params, "radius", "rc_near"), "params.radius"),
        axes=axes,
        failure_stage=_failure_stage(params),
    )


def _build_rc_contact_any(params: dict) -> RcContactAnyCondition:
    fixtures = params.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ConfigError("rc_contact_any needs params.fixtures (non-empty list)")
    return RcContactAnyCondition(
        object_name=str(_require(params, "object", "rc_contact_any")),
        fixtures=tuple(str(f) for f in fixtures),
        failure_stage=_failure_stage(params),
    )


def _build_rc_timer(params: dict) -> RcTimerCondition:
    return RcTimerCondition(
        name=str(_require(params, "name", "rc_timer")),
        count=int(_require(params, "count", "rc_timer")),
        failure_stage=_failure_stage(params),
    )


def _build_rc_basin(params: dict) -> RcBasinCondition:
    side = params.get("side", "left")
    if side not in ("left", "right"):
        raise ConfigError(f"rc_basin params.side must be left/right, got {side!r}")
    bbox = params.get("object_bbox")
    corners = None
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 8:
            raise ConfigError("rc_basin params.object_bbox must be 8 corners")
        corners = tuple(
            _vec(corner, 3, "rc_basin params.object_bbox[]") for corner in bbox
        )
    return RcBasinCondition(
        object_name=str(_require(params, "object", "rc_basin")),
        fixture_body=str(_require(params, "fixture_body", "rc_basin")),
        side=str(side),
        center_local=_vec(
            _require(params, "center_local", "rc_basin"), 3, "params.center_local"
        ),
        halfsize_local=_vec(
            _require(params, "halfsize_local", "rc_basin"), 3, "params.halfsize_local"
        ),
        object_bbox=corners,
        failure_stage=_failure_stage(params),
    )


def _build_rc_closer(params: dict) -> RcCloserCondition:
    return RcCloserCondition(
        object_name=str(_require(params, "object", "rc_closer")),
        anchor=str(_require(params, "anchor", "rc_closer")),
        other=str(_require(params, "other", "rc_closer")),
        ratio=float(params.get("ratio", 2.0)),
        failure_stage=_failure_stage(params),
    )


def _build_rc_on_burner(params: dict) -> RcOnBurnerCondition:
    return RcOnBurnerCondition(
        object_name=str(_require(params, "object", "rc_on_burner")),
        burner_site=str(_require(params, "burner_site", "rc_on_burner")),
        threshold=_positive_float(
            _require(params, "threshold", "rc_on_burner"), "params.threshold"
        ),
        knob_joint=str(params.get("knob_joint", "")),
        require_knob_on=bool(params.get("require_knob_on", False)),
        failure_stage=_failure_stage(params),
    )


def _build_any(params: dict) -> AnyCondition:
    raw = _require(params, "conditions", "any")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("params.conditions must be a non-empty list of conditions")
    return AnyCondition(
        conditions=tuple(create_success_condition(spec) for spec in raw)
    )


def _build_not(params: dict) -> NotCondition:
    raw = _require(params, "condition", "not")
    if not isinstance(raw, dict):
        raise ConfigError("params.condition must be a condition mapping")
    return NotCondition(condition=create_success_condition(raw))


def _build_all(params: dict) -> AllCondition:
    raw = _require(params, "conditions", "all")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("params.conditions must be a non-empty list of conditions")
    return AllCondition(
        conditions=tuple(create_success_condition(spec) for spec in raw)
    )


def create_success_condition(spec: dict | None) -> SuccessCondition:
    """Build a SuccessCondition from a ``success:`` YAML block.

    Args:
        spec: Mapping with ``type`` (distance_threshold / pose_window / staged)
            and a ``params`` mapping. See ``configs/tasks/pick_place_cube.yaml``
            for examples.

    Raises:
        ConfigError: If the spec is malformed or the type is unknown.
    """
    if not isinstance(spec, dict):
        raise ConfigError(f"success spec must be a mapping, got {spec!r}")
    success_type = spec.get("type")
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError(f"success params must be a mapping, got {params!r}")

    if success_type == "distance_threshold":
        return _build_distance(params)
    if success_type == "pose_window":
        return _build_pose_window(params)
    if success_type == "staged":
        return _build_staged(params)
    if success_type == "axis_window":
        return _build_axis_window(params)
    if success_type == "in_region":
        return _build_in_region(params)
    if success_type == "count_in_region":
        return _build_count_in_region(params)
    if success_type == "body_point_near":
        return _build_body_point_near(params)
    if success_type == "ee_near":
        return _build_ee_near(params)
    if success_type == "above_region":
        return _build_above_region(params)
    if success_type == "tipped":
        return _build_tipped(params)
    if success_type == "contains":
        return _build_contains(params)
    if success_type == "lift":
        return _build_lift(params)
    if success_type == "ordered":
        return _build_ordered(params)
    if success_type == "in_contact":
        return _build_in_contact(params)
    if success_type == "joint_window":
        return _build_joint_window(params)
    if success_type == "libero_on":
        return _build_libero_on(params)
    if success_type == "libero_in":
        return _build_libero_in(params)
    if success_type == "libero_joint_state":
        return _build_libero_joint_state(params)
    if success_type == "rc_contact":
        return _build_rc_contact(params)
    if success_type == "rc_in_receptacle":
        return _build_rc_in_receptacle(params)
    if success_type == "rc_gripper_far":
        return _build_rc_gripper_far(params)
    if success_type == "rc_joint_state":
        return _build_rc_joint_state(params)
    if success_type == "rc_appliance":
        return _build_rc_appliance(params)
    if success_type == "rc_inside":
        return _build_rc_inside(params)
    if success_type == "rc_upright":
        return _build_rc_upright(params)
    if success_type == "rc_at_spawn":
        return _build_rc_at_spawn(params)
    if success_type == "rc_near":
        return _build_rc_near(params)
    if success_type == "rc_contact_any":
        return _build_rc_contact_any(params)
    if success_type == "rc_timer":
        return _build_rc_timer(params)
    if success_type == "rc_basin":
        return _build_rc_basin(params)
    if success_type == "rc_on_burner":
        return _build_rc_on_burner(params)
    if success_type == "rc_closer":
        return _build_rc_closer(params)
    if success_type == "all":
        return _build_all(params)
    if success_type == "any":
        return _build_any(params)
    if success_type == "not":
        return _build_not(params)
    raise ConfigError(
        f"unknown success type {success_type!r}; available types: {list(SUCCESS_TYPES)}"
    )
