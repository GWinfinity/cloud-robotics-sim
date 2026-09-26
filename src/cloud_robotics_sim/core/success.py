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

SUCCESS_TYPES = ("distance_threshold", "pose_window", "staged")


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
    """Success when an object rests within ``threshold`` of a target position."""

    object_name: str
    target_position: tuple[float, float, float]
    threshold: float
    failure_stage: str = "error"

    def evaluate(self, scene: Any, robot: Any, info: dict | None = None) -> bool:
        pos = _entity_position(scene, self.object_name)
        dist = float(np.linalg.norm(pos - np.asarray(self.target_position)))
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
    raise ConfigError(
        f"unknown success type {success_type!r}; available types: {list(SUCCESS_TYPES)}"
    )
