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
    raise ConfigError(
        f"unknown success type {success_type!r}; available types: {list(SUCCESS_TYPES)}"
    )
