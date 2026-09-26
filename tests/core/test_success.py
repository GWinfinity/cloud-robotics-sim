"""Tests for configurable success conditions (core/success.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.success import (
    DistanceThresholdCondition,
    PoseWindowCondition,
    StagedCondition,
    create_success_condition,
)
from cloud_robotics_sim.robotwin.grasp_report import FAIL_STAGES


def _scene_with(pos, quat=None):
    """Build a minimal scene-like object with one entity at the given pose."""
    entity = MagicMock()
    entity.get_pos.return_value = np.asarray(pos, dtype=float)
    if quat is not None:
        entity.get_quat.return_value = np.asarray(quat, dtype=float)
    return SimpleNamespace(entities={"obj": entity})


class TestDistanceThreshold:
    """Tests for DistanceThresholdCondition."""

    def test_within_threshold(self):
        """Object at the target succeeds."""
        condition = DistanceThresholdCondition(
            object_name="obj", target_position=(0.5, 0.0, 0.1), threshold=0.05
        )
        assert condition.evaluate(_scene_with((0.52, 0.0, 0.1)), None) is True

    def test_beyond_threshold(self):
        """Object far from the target fails."""
        condition = DistanceThresholdCondition(
            object_name="obj", target_position=(0.5, 0.0, 0.1), threshold=0.05
        )
        assert condition.evaluate(_scene_with((0.8, 0.0, 0.1)), None) is False

    def test_failure_stage_override(self):
        """A configured failure stage is reported on failure."""
        condition = DistanceThresholdCondition(
            object_name="obj",
            target_position=(0.5, 0.0, 0.1),
            threshold=0.05,
            failure_stage="place_fail",
        )
        condition.evaluate(_scene_with((0.8, 0.0, 0.1)), None)
        assert condition.last_failure_stage() == "place_fail"

    def test_unknown_object_raises(self):
        """Referencing a missing entity raises a descriptive KeyError."""
        condition = DistanceThresholdCondition(
            object_name="missing", target_position=(0, 0, 0), threshold=0.05
        )
        with pytest.raises(KeyError, match="unknown object 'missing'"):
            condition.evaluate(_scene_with((0, 0, 0)), None)


class TestPoseWindow:
    """Tests for PoseWindowCondition."""

    IDENTITY = (1.0, 0.0, 0.0, 0.0)

    def test_position_and_orientation_pass(self):
        """Object within both windows succeeds."""
        condition = PoseWindowCondition(
            object_name="obj",
            target_position=(0.5, 0.0, 0.1),
            target_orientation=self.IDENTITY,
        )
        scene = _scene_with((0.51, 0.0, 0.1), quat=self.IDENTITY)
        assert condition.evaluate(scene, None) is True

    def test_orientation_beyond_window(self):
        """A 90-degree yaw exceeds the default rotation window."""
        condition = PoseWindowCondition(
            object_name="obj",
            target_position=(0.5, 0.0, 0.1),
            target_orientation=self.IDENTITY,
        )
        yaw90 = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
        scene = _scene_with((0.5, 0.0, 0.1), quat=yaw90)
        assert condition.evaluate(scene, None) is False

    def test_position_only_when_orientation_omitted(self):
        """Without a target orientation only the position window is checked."""
        condition = PoseWindowCondition(
            object_name="obj", target_position=(0.5, 0.0, 0.1)
        )
        scene = _scene_with((0.5, 0.0, 0.1), quat=(0.0, 1.0, 0.0, 0.0))
        assert condition.evaluate(scene, None) is True

    def test_position_beyond_window(self):
        """Position outside the window fails even with matching orientation."""
        condition = PoseWindowCondition(
            object_name="obj",
            target_position=(0.5, 0.0, 0.1),
            target_orientation=self.IDENTITY,
            pos_threshold=0.02,
        )
        scene = _scene_with((0.5, 0.1, 0.1), quat=self.IDENTITY)
        assert condition.evaluate(scene, None) is False


class TestStaged:
    """Tests for StagedCondition."""

    def _stage(self, name, target):
        return (
            name,
            DistanceThresholdCondition(
                object_name="obj", target_position=target, threshold=0.05
            ),
        )

    def test_all_stages_pass(self):
        """All sub-conditions holding means success."""
        condition = StagedCondition(
            stages=[
                self._stage("lift_fail", (0.5, 0, 0.1)),
                self._stage("place_fail", (0.5, 0, 0.1)),
            ]
        )
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.1)), None) is True
        assert condition.last_failure_stage() == "error"

    def test_first_failing_stage_reported(self):
        """The earliest failing stage is remembered."""
        condition = StagedCondition(
            stages=[
                self._stage("lift_fail", (0.5, 0, 0.5)),
                self._stage("place_fail", (0.5, 0, 0.1)),
            ]
        )
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.1)), None) is False
        assert condition.last_failure_stage() == "lift_fail"


class TestCreateSuccessCondition:
    """Tests for the success spec factory."""

    def test_distance_threshold(self):
        """A YAML-style spec builds the right condition type."""
        condition = create_success_condition(
            {
                "type": "distance_threshold",
                "params": {
                    "object": "obj",
                    "target_position": [0.5, 0.0, 0.1],
                    "threshold": 0.05,
                    "failure_stage": "place_fail",
                },
            }
        )
        assert isinstance(condition, DistanceThresholdCondition)
        assert condition.last_failure_stage() == "place_fail"
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.12)), None) is True

    def test_pose_window(self):
        """pose_window spec with explicit thresholds."""
        condition = create_success_condition(
            {
                "type": "pose_window",
                "params": {
                    "object": "obj",
                    "target_position": [0.5, 0.0, 0.1],
                    "target_orientation": [1.0, 0.0, 0.0, 0.0],
                    "pos_threshold": 0.02,
                    "rot_threshold_deg": 5.0,
                },
            }
        )
        assert isinstance(condition, PoseWindowCondition)

    def test_staged(self):
        """Staged spec recurses into nested condition specs."""
        condition = create_success_condition(
            {
                "type": "staged",
                "params": {
                    "stages": [
                        {
                            "stage": "grasp_fail",
                            "condition": {
                                "type": "distance_threshold",
                                "params": {
                                    "object": "obj",
                                    "target_position": [0.5, 0.0, 0.1],
                                    "threshold": 0.05,
                                },
                            },
                        }
                    ]
                },
            }
        )
        assert isinstance(condition, StagedCondition)
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.1)), None) is True

    @pytest.mark.parametrize("stage", list(FAIL_STAGES))
    def test_staged_accepts_all_fail_stages(self, stage):
        """Every FAIL_STAGES label is accepted as a stage name."""
        condition = create_success_condition(
            {
                "type": "staged",
                "params": {
                    "stages": [
                        {
                            "stage": stage,
                            "condition": {
                                "type": "distance_threshold",
                                "params": {
                                    "object": "obj",
                                    "target_position": [0, 0, 0],
                                    "threshold": 1.0,
                                },
                            },
                        }
                    ]
                },
            }
        )
        assert isinstance(condition, StagedCondition)

    def test_unknown_type(self):
        """Unknown success types are rejected with available types listed."""
        with pytest.raises(ConfigError, match="unknown success type"):
            create_success_condition({"type": "telepathy", "params": {}})

    def test_missing_required_param(self):
        """Missing required params are reported."""
        with pytest.raises(ConfigError, match="requires params.threshold"):
            create_success_condition(
                {
                    "type": "distance_threshold",
                    "params": {"object": "obj", "target_position": [0, 0, 0]},
                }
            )

    def test_invalid_failure_stage(self):
        """failure_stage outside FAIL_STAGES is rejected."""
        with pytest.raises(ConfigError, match="failure_stage"):
            create_success_condition(
                {
                    "type": "distance_threshold",
                    "params": {
                        "object": "obj",
                        "target_position": [0, 0, 0],
                        "threshold": 0.05,
                        "failure_stage": "exploded",
                    },
                }
            )

    def test_staged_rejects_bad_stage(self):
        """A staged entry with an unknown stage label is rejected."""
        with pytest.raises(ConfigError, match="params.stages\\[0\\].stage"):
            create_success_condition(
                {
                    "type": "staged",
                    "params": {
                        "stages": [
                            {
                                "stage": "nope",
                                "condition": {
                                    "type": "distance_threshold",
                                    "params": {
                                        "object": "obj",
                                        "target_position": [0, 0, 0],
                                        "threshold": 1.0,
                                    },
                                },
                            }
                        ]
                    },
                }
            )

    def test_non_mapping_spec(self):
        """A non-mapping spec is rejected."""
        with pytest.raises(ConfigError, match="must be a mapping"):
            create_success_condition(["distance_threshold"])  # type: ignore[arg-type]
