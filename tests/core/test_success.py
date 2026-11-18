"""Tests for configurable success conditions (core/success.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.success import (
    AboveRegionCondition,
    AllCondition,
    AxisWindowCondition,
    BodyPointNearCondition,
    ContainsCondition,
    CountInRegionCondition,
    DistanceThresholdCondition,
    EeNearCondition,
    InContactCondition,
    InRegionCondition,
    LiberoInCondition,
    LiberoJointStateCondition,
    LiberoOnCondition,
    LiftCondition,
    OrderedCondition,
    PoseWindowCondition,
    StagedCondition,
    TippedCondition,
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

    def test_axis_weights_scale_height(self):
        """Weighted norms count an axis double (MetaWorld basketball)."""
        condition = DistanceThresholdCondition(
            object_name="obj",
            target_position=(0.0, 0.0, 0.3),
            threshold=0.08,
            axis_weights=(1.0, 1.0, 2.0),
        )
        # dy=0.05, dz=+0.05 -> sqrt(0.05^2 + (2*0.05)^2) ~ 0.112 > 0.08.
        assert condition.evaluate(_scene_with((0.0, 0.05, 0.35)), None) is False
        # Unweighted this would pass (norm ~ 0.071 < 0.08).
        plain = DistanceThresholdCondition(
            object_name="obj",
            target_position=(0.0, 0.0, 0.3),
            threshold=0.08,
        )
        assert plain.evaluate(_scene_with((0.0, 0.05, 0.35)), None) is True

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


class TestAxisWindow:
    """Tests for AxisWindowCondition (per-axis range predicates)."""

    def test_all_windows_pass(self):
        """Object inside every axis window succeeds."""
        condition = AxisWindowCondition(
            object_name="obj",
            windows={"x": [0.4, 0.6], "y": [-0.1, 0.1], "z": [0.05, 0.2]},
        )
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.1)), None) is True

    def test_one_axis_outside_fails(self):
        """A single axis outside its window fails (MetaWorld x-only checks)."""
        condition = AxisWindowCondition(object_name="obj", windows={"x": [0.4, 0.6]})
        assert condition.evaluate(_scene_with((0.7, 9.0, 9.0)), None) is False
        assert condition.evaluate(_scene_with((0.5, 9.0, 9.0)), None) is True

    def test_open_bounds(self):
        """A null bound means unbounded on that side (e.g. z >= min)."""
        condition = AxisWindowCondition(object_name="obj", windows={"z": [0.1, None]})
        assert condition.evaluate(_scene_with((0.0, 0.0, 5.0)), None) is True
        assert condition.evaluate(_scene_with((0.0, 0.0, 0.05)), None) is False

    def test_relative_windows_around_target(self):
        """With a target anchor, windows are offsets (MetaWorld phrasing)."""
        condition = AxisWindowCondition(
            object_name="obj",
            windows={"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [None, 0.0]},
            target_position=(0.5, 0.8, 0.1),
        )
        assert condition.evaluate(_scene_with((0.51, 0.79, 0.08)), None) is True
        assert condition.evaluate(_scene_with((0.51, 0.79, 0.12)), None) is False
        assert condition.evaluate(_scene_with((0.54, 0.79, 0.08)), None) is False

    def test_unknown_axis_rejected_by_factory(self):
        """Windows on axes other than x/y/z are a config error."""
        with pytest.raises(ConfigError, match="params.windows"):
            create_success_condition(
                {
                    "type": "axis_window",
                    "params": {"object": "obj", "windows": {"w": [0, 1]}},
                }
            )


class TestBodyPointNear:
    """Tests for BodyPointNearCondition (body-frame offset points)."""

    IDENTITY = (1.0, 0.0, 0.0, 0.0)
    YAW90 = (
        float(np.cos(np.pi / 4)),
        0.0,
        0.0,
        float(np.sin(np.pi / 4)),
    )

    def _scene(self, pos, quat):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(pos, dtype=float)
        entity.get_quat.return_value = np.asarray(quat, dtype=float)
        return SimpleNamespace(entities={"obj": entity})

    def test_identity_orientation_uses_offset_directly(self):
        """With identity quat the point is pos + offset."""
        condition = BodyPointNearCondition(
            object_name="obj",
            offset=(0.1, 0.0, 0.0),
            target_position=(0.6, 0.0, 0.0),
            threshold=0.05,
        )
        assert condition.evaluate(self._scene((0.5, 0.0, 0.0), self.IDENTITY), None)

    def test_rotation_moves_the_point(self):
        """A 90-degree yaw rotates the body-frame offset."""
        condition = BodyPointNearCondition(
            object_name="obj",
            offset=(0.1, 0.0, 0.0),
            target_position=(0.5, 0.1, 0.0),
            threshold=0.02,
        )
        # Identity: point at 0.6,0 -> fails. Yaw90: point at 0.5,0.1 -> passes.
        assert (
            condition.evaluate(self._scene((0.5, 0.0, 0.0), self.IDENTITY), None)
            is False
        )
        assert (
            condition.evaluate(self._scene((0.5, 0.0, 0.0), self.YAW90), None) is True
        )

    def test_factory_builds_condition(self):
        """The YAML factory accepts offset/weights."""
        condition = create_success_condition(
            {
                "type": "body_point_near",
                "params": {
                    "object": "obj",
                    "offset": [0.0, -0.05, 0.035],
                    "target_position": [0.0, 0.65, 0.035],
                    "threshold": 0.07,
                    "axis_weights": [1.0, 2.0, 2.0],
                },
            }
        )
        assert isinstance(condition, BodyPointNearCondition)


class TestEeNear:
    """Tests for EeNearCondition (end-effector position checks)."""

    def test_link_pose_used_when_available(self):
        """A robot with get_link_pose(link) is checked at that link."""

        class FakePose:
            pos = np.array([0.5, 0.0, 0.1])

        class FakeEntity:
            def get_link_pose(self, link):
                assert link == "hand"
                return FakePose()

        robot = SimpleNamespace(entity=FakeEntity())
        condition = create_success_condition(
            {
                "type": "ee_near",
                "params": {
                    "target_position": [0.5, 0.0, 0.1],
                    "threshold": 0.05,
                    "link": "hand",
                },
            }
        )
        assert condition.evaluate(None, robot) is True

    def test_root_position_fallback(self):
        """Without the link accessor the articulation root is used."""
        entity = MagicMock()
        entity.get_link_pose.return_value = None
        entity.get_pos.return_value = np.array([0.0, 0.0, 0.0])
        robot = SimpleNamespace(entity=entity)
        condition = EeNearCondition(target_position=(1.0, 0.0, 0.0), threshold=0.05)
        assert condition.evaluate(None, robot) is False

    def test_missing_target_rejected(self):
        """from_site has no meaning for an ee check."""
        with pytest.raises(ConfigError, match="explicit"):
            create_success_condition(
                {
                    "type": "ee_near",
                    "params": {"target_position": "from_site", "threshold": 0.05},
                }
            )


class TestInRegion:
    """Tests for InRegionCondition (axis-aligned box containment)."""

    def test_inside_region(self):
        """Object inside the box succeeds (goal areas, containers)."""
        condition = InRegionCondition(
            object_name="obj", region=[0.0, 0.2, -0.1, 0.1, 0.0, 0.1]
        )
        assert condition.evaluate(_scene_with((0.1, 0.0, 0.05)), None) is True

    def test_outside_region(self):
        """Object outside any face fails."""
        condition = InRegionCondition(
            object_name="obj", region=[0.0, 0.2, -0.1, 0.1, 0.0, 0.1]
        )
        assert condition.evaluate(_scene_with((0.1, 0.0, 0.2)), None) is False
        assert condition.evaluate(_scene_with((-0.05, 0.0, 0.05)), None) is False

    def test_bad_region_shape(self):
        """A region that is not 6 finite numbers is rejected."""
        with pytest.raises(ConfigError, match="params.region"):
            create_success_condition(
                {
                    "type": "in_region",
                    "params": {"object": "obj", "region": [0, 1, 2]},
                }
            )


class TestCountInRegion:
    """Tests for CountInRegionCondition (N objects inside a box)."""

    def _scene_many(self, positions):
        entities = {}
        for name, pos in positions.items():
            entity = MagicMock()
            entity.get_pos.return_value = np.asarray(pos, dtype=float)
            entities[name] = entity
        return SimpleNamespace(entities=entities)

    def test_min_count_reached(self):
        """Success once enough listed objects are inside the region."""
        condition = CountInRegionCondition(
            objects=["cube_1", "cube_2", "cube_3"],
            region=[0.0, 0.3, -0.1, 0.1, 0.0, 0.1],
            min_count=2,
        )
        scene = self._scene_many(
            {
                "cube_1": (0.1, 0.0, 0.05),
                "cube_2": (0.2, 0.0, 0.05),
                "cube_3": (5, 5, 5),
            }
        )
        assert condition.evaluate(scene, None) is True

    def test_min_count_missed(self):
        """Failure while fewer than min_count objects are inside."""
        condition = CountInRegionCondition(
            objects=["cube_1", "cube_2", "cube_3"],
            region=[0.0, 0.3, -0.1, 0.1, 0.0, 0.1],
            min_count=2,
        )
        scene = self._scene_many(
            {"cube_1": (0.1, 0.0, 0.05), "cube_2": (5, 5, 5), "cube_3": (5, 5, 5)}
        )
        assert condition.evaluate(scene, None) is False

    def test_prefix_matching(self):
        """A name prefix counts every matching entity (KinDER o50 cubes)."""
        condition = CountInRegionCondition(
            prefix="cube_",
            region=[0.0, 0.3, -0.1, 0.1, 0.0, 0.1],
            min_count=50,
        )
        scene = self._scene_many(
            {f"cube_{i}": (0.05 + 0.001 * i, 0.0, 0.05) for i in range(50)}
        )
        assert condition.evaluate(scene, None) is True

    def test_requires_objects_or_prefix(self):
        """Without an explicit list or prefix the spec is rejected."""
        with pytest.raises(ConfigError, match="params.objects"):
            create_success_condition(
                {
                    "type": "count_in_region",
                    "params": {"region": [0, 1, 0, 1, 0, 1], "min_count": 1},
                }
            )


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

    def test_axis_window(self):
        """axis_window spec builds an AxisWindowCondition."""
        condition = create_success_condition(
            {
                "type": "axis_window",
                "params": {
                    "object": "obj",
                    "windows": {"x": [0.4, 0.6], "z": [0.05, None]},
                },
            }
        )
        assert isinstance(condition, AxisWindowCondition)
        assert condition.evaluate(_scene_with((0.5, 0.0, 0.2)), None) is True

    def test_in_region(self):
        """in_region spec builds an InRegionCondition."""
        condition = create_success_condition(
            {
                "type": "in_region",
                "params": {
                    "object": "obj",
                    "region": [0.0, 0.2, -0.1, 0.1, 0.0, 0.1],
                },
            }
        )
        assert isinstance(condition, InRegionCondition)
        assert condition.evaluate(_scene_with((0.1, 0.0, 0.05)), None) is True

    def test_count_in_region(self):
        """count_in_region spec builds a CountInRegionCondition."""
        condition = create_success_condition(
            {
                "type": "count_in_region",
                "params": {
                    "prefix": "cube_",
                    "region": [0.0, 0.3, -0.1, 0.1, 0.0, 0.1],
                    "min_count": 2,
                },
            }
        )
        assert isinstance(condition, CountInRegionCondition)

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


class TestAboveRegion:
    """Tests for AboveRegionCondition (VLABench ``above``)."""

    def _scene(self, obj_pos, platform_pos=(0.5, 0.0, 0.8)):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(obj_pos, dtype=float)
        platform = MagicMock()
        platform.get_pos.return_value = np.asarray(platform_pos, dtype=float)
        return SimpleNamespace(entities={"obj": entity, "pan": platform})

    def test_over_circle_footprint(self):
        """Object above the platform and inside the radius succeeds."""
        condition = AboveRegionCondition(object_name="obj", platform="pan", circle=0.1)
        assert condition.evaluate(self._scene((0.53, 0.0, 0.85)), None) is True

    def test_outside_circle_footprint(self):
        """Object above but beyond the radius fails."""
        condition = AboveRegionCondition(object_name="obj", platform="pan", circle=0.1)
        assert condition.evaluate(self._scene((0.65, 0.0, 0.85)), None) is False

    def test_below_platform_height(self):
        """Object lower than the platform origin fails (upstream z >= rule)."""
        condition = AboveRegionCondition(object_name="obj", platform="pan", circle=0.1)
        assert condition.evaluate(self._scene((0.5, 0.0, 0.7)), None) is False

    def test_equal_height_passes(self):
        """Upstream uses strict ``<`` on z, so equality succeeds."""
        condition = AboveRegionCondition(object_name="obj", platform="pan", circle=0.1)
        assert condition.evaluate(self._scene((0.5, 0.0, 0.8)), None) is True

    def test_region_box_footprint(self):
        """Absolute box footprint works without a platform-centred circle."""
        condition = AboveRegionCondition(
            object_name="obj",
            platform="pan",
            region=(0.4, 0.6, -0.1, 0.1),
        )
        assert condition.evaluate(self._scene((0.45, 0.05, 0.9)), None) is True
        assert condition.evaluate(self._scene((0.7, 0.05, 0.9)), None) is False

    def test_z_offset_raises_the_bar(self):
        """z_offset adds headroom above the platform origin."""
        condition = AboveRegionCondition(
            object_name="obj", platform="pan", circle=0.1, z_offset=0.1
        )
        assert condition.evaluate(self._scene((0.5, 0.0, 0.85)), None) is False
        assert condition.evaluate(self._scene((0.5, 0.0, 0.91)), None) is True

    def test_failure_stage_override(self):
        condition = AboveRegionCondition(
            object_name="obj",
            platform="pan",
            circle=0.1,
            failure_stage="place_fail",
        )
        condition.evaluate(self._scene((0.65, 0.0, 0.85)), None)
        assert condition.last_failure_stage() == "place_fail"


class TestTipped:
    """Tests for TippedCondition (VLABench ``pour``)."""

    TOP = (0.0, 0.0, 0.047)
    BOTTOM = (0.0, 0.0, -0.047)

    def _scene(self, quat):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray((0.2, -0.1, 0.9), dtype=float)
        entity.get_quat.return_value = np.asarray(quat, dtype=float)
        return SimpleNamespace(entities={"obj": entity})

    def test_upright_fails(self):
        """Upright bottle (sites aligned with +z) is not poured."""
        condition = TippedCondition(
            object_name="obj", top_offset=self.TOP, bottom_offset=self.BOTTOM
        )
        assert condition.evaluate(self._scene((1.0, 0.0, 0.0, 0.0)), None) is False

    def test_past_horizontal_passes(self):
        """180° flip puts the bottom site above the top site."""
        condition = TippedCondition(
            object_name="obj", top_offset=self.TOP, bottom_offset=self.BOTTOM
        )
        assert condition.evaluate(self._scene((0.0, 1.0, 0.0, 0.0)), None) is True

    def test_exactly_horizontal_fails(self):
        """90° tip gives bottom.z == top.z; upstream needs strict >."""
        q = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
        condition = TippedCondition(
            object_name="obj", top_offset=self.TOP, bottom_offset=self.BOTTOM
        )
        assert condition.evaluate(self._scene(q), None) is False

    def test_threshold_requires_deeper_tip(self):
        """A positive threshold needs bottom.z - top.z > threshold."""
        condition = TippedCondition(
            object_name="obj",
            top_offset=self.TOP,
            bottom_offset=self.BOTTOM,
            threshold=0.05,
        )
        q = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
        # 90°: dz == 0 -> fails even though the bottle is fully horizontal.
        assert condition.evaluate(self._scene(q), None) is False
        # 180°: dz == 0.094 > 0.05 -> passes.
        assert condition.evaluate(self._scene((0.0, 1.0, 0.0, 0.0)), None) is True

    def test_rotation_about_unrelated_axis(self):
        """Yaw does not count as tipping."""
        condition = TippedCondition(
            object_name="obj", top_offset=self.TOP, bottom_offset=self.BOTTOM
        )
        q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # 90° about z
        assert condition.evaluate(self._scene(q), None) is False

    def test_failure_stage_override(self):
        condition = TippedCondition(
            object_name="obj",
            top_offset=self.TOP,
            bottom_offset=self.BOTTOM,
            failure_stage="place_fail",
        )
        condition.evaluate(self._scene((1.0, 0.0, 0.0, 0.0)), None)
        assert condition.last_failure_stage() == "place_fail"


class TestCreateVlabenchConditions:
    """Factory coverage for the VLABench predicate types."""

    def test_above_region_circle(self):
        condition = create_success_condition(
            {
                "type": "above_region",
                "params": {"object": "obj", "platform": "pan", "circle": 0.1},
            }
        )
        assert isinstance(condition, AboveRegionCondition)

    def test_above_region_requires_exactly_one_footprint(self):
        with pytest.raises(ConfigError, match="exactly one of"):
            create_success_condition(
                {
                    "type": "above_region",
                    "params": {"object": "obj", "platform": "pan"},
                }
            )
        with pytest.raises(ConfigError, match="exactly one of"):
            create_success_condition(
                {
                    "type": "above_region",
                    "params": {
                        "object": "obj",
                        "platform": "pan",
                        "circle": 0.1,
                        "region": [0.0, 0.2, -0.1, 0.1],
                    },
                }
            )

    def test_tipped(self):
        condition = create_success_condition(
            {
                "type": "tipped",
                "params": {
                    "object": "obj",
                    "top_offset": [0.0, 0.0, 0.047],
                    "bottom_offset": [0.0, 0.0, -0.047],
                },
            }
        )
        assert isinstance(condition, TippedCondition)

    def test_tipped_requires_sites(self):
        with pytest.raises(ConfigError, match="requires params.top_offset"):
            create_success_condition(
                {
                    "type": "tipped",
                    "params": {"object": "obj", "bottom_offset": [0, 0, -0.05]},
                }
            )

    def test_tipped_rejects_negative_threshold(self):
        with pytest.raises(ConfigError, match=">= 0"):
            create_success_condition(
                {
                    "type": "tipped",
                    "params": {
                        "object": "obj",
                        "top_offset": [0, 0, 0.05],
                        "bottom_offset": [0, 0, -0.05],
                        "threshold": -0.1,
                    },
                }
            )


class TestContains:
    """Tests for ContainsCondition (VLABench contain / not_contain)."""

    def _scene(self, obj_pos, container_pos=(0.5, 0.0, 0.8)):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(obj_pos, dtype=float)
        container = MagicMock()
        container.get_pos.return_value = np.asarray(container_pos, dtype=float)
        return SimpleNamespace(entities={"obj": entity, "box": container})

    def test_circle_inside_zband(self):
        condition = ContainsCondition(
            container="box", objects=("obj",), mode="circle", circle=0.1
        )
        assert condition.evaluate(self._scene((0.53, 0.0, 0.85)), None) is True

    def test_circle_zband_exclusive(self):
        """Upstream: 0 < p.z - c.z < z_threshold (strict)."""
        condition = ContainsCondition(
            container="box", objects=("obj",), mode="circle", circle=0.1
        )
        assert condition.evaluate(self._scene((0.5, 0.0, 0.8)), None) is False
        assert condition.evaluate(self._scene((0.5, 0.0, 0.95)), None) is False

    def test_circle_above_band_fails(self):
        condition = ContainsCondition(
            container="box", objects=("obj",), mode="circle", circle=0.1
        )
        assert condition.evaluate(self._scene((0.5, 0.0, 1.0)), None) is False

    def test_boxes_any_of_inclusive(self):
        condition = ContainsCondition(
            container="box",
            objects=("obj",),
            mode="boxes",
            regions=(
                {
                    "center": [0.0, 0.0, 0.05],
                    "min": [-0.05, -0.05, 0.0],
                    "max": [0.05, 0.05, 0.1],
                },
                {
                    "center": [0.2, 0.0, 0.05],
                    "min": [-0.02, -0.02, 0.0],
                    "max": [0.02, 0.02, 0.1],
                },
            ),
        )
        assert condition.evaluate(self._scene((0.54, 0.0, 0.87)), None) is True
        assert condition.evaluate(self._scene((0.70, 0.0, 0.87)), None) is True
        assert condition.evaluate(self._scene((0.60, 0.0, 0.87)), None) is False

    def test_multi_object_all_required(self):
        condition = ContainsCondition(
            container="box",
            objects=("obj", "obj2"),
            mode="circle",
            circle=0.2,
        )
        entity2 = MagicMock()
        entity2.get_pos.return_value = np.asarray((0.6, 0.0, 0.85), dtype=float)
        scene = self._scene((0.4, 0.0, 0.85))
        scene.entities["obj2"] = entity2
        assert condition.evaluate(scene, None) is True
        entity2.get_pos.return_value = np.asarray((0.9, 0.0, 0.85), dtype=float)
        assert condition.evaluate(scene, None) is False

    def test_negate_not_contain(self):
        condition = ContainsCondition(
            container="box", objects=("obj",), mode="circle", circle=0.1, negate=True
        )
        assert condition.evaluate(self._scene((0.9, 0.0, 0.85)), None) is True
        assert condition.evaluate(self._scene((0.53, 0.0, 0.85)), None) is False


class TestLift:
    """Tests for LiftCondition (VLABench lift)."""

    def test_above_height(self):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray((0.2, 0.0, 0.95), dtype=float)
        scene = SimpleNamespace(entities={"card": entity})
        condition = LiftCondition(objects=("card",), min_height=0.9)
        assert condition.evaluate(scene, None) is True
        entity.get_pos.return_value = np.asarray((0.2, 0.0, 0.8), dtype=float)
        assert condition.evaluate(scene, None) is False


class TestOrdered:
    """Tests for OrderedCondition (VLABench order)."""

    def _scene(self, positions):
        entities = {}
        for name, pos in positions.items():
            entity = MagicMock()
            entity.get_pos.return_value = np.asarray(pos, dtype=float)
            entities[name] = entity
        return SimpleNamespace(entities=entities)

    def test_increasing_on_axis(self):
        scene = self._scene({"a": (0.1, 0.0, 0.8), "b": (0.2, 0.0, 0.8)})
        condition = OrderedCondition(objects=("a", "b"), axes=(0,))
        assert condition.evaluate(scene, None) is True
        scene = self._scene({"a": (0.2, 0.0, 0.8), "b": (0.1, 0.0, 0.8)})
        assert condition.evaluate(scene, None) is False

    def test_off_axis_tolerance_one_sided(self):
        """Upstream: (p[i] - p[i+1]) < offset on non-order axes."""
        scene = self._scene({"a": (0.1, 0.05, 0.8), "b": (0.2, 0.0, 0.8)})
        condition = OrderedCondition(objects=("a", "b"), axes=(0,), offset=0.1)
        assert condition.evaluate(scene, None) is True
        # a.y - b.y == 0.05 is fine, 0.15 exceeds the one-sided tolerance.
        scene = self._scene({"a": (0.1, 0.15, 0.8), "b": (0.2, 0.0, 0.8)})
        assert condition.evaluate(scene, None) is False
        # Negative deltas always satisfy the one-sided check.
        scene = self._scene({"a": (0.1, -0.5, 0.8), "b": (0.2, 0.0, 0.8)})
        assert condition.evaluate(scene, None) is True


class TestCreateContainLiftOrder:
    """Factory coverage for contain / lift / ordered."""

    def test_contains_circle(self):
        condition = create_success_condition(
            {
                "type": "contains",
                "params": {"container": "box", "objects": ["obj"], "circle": 0.1},
            }
        )
        assert isinstance(condition, ContainsCondition)

    def test_contains_requires_valid_regions(self):
        with pytest.raises(ConfigError, match="regions"):
            create_success_condition(
                {
                    "type": "contains",
                    "params": {"container": "box", "objects": ["obj"], "mode": "boxes"},
                }
            )

    def test_lift(self):
        condition = create_success_condition(
            {
                "type": "lift",
                "params": {"objects": ["card"], "min_height": 0.9},
            }
        )
        assert isinstance(condition, LiftCondition)

    def test_ordered(self):
        condition = create_success_condition(
            {
                "type": "ordered",
                "params": {"objects": ["a", "b"], "axes": [0]},
            }
        )
        assert isinstance(condition, OrderedCondition)


class TestInContact:
    """Tests for InContactCondition (VLABench press_button / is_grasped)."""

    def _entity(self, contacts):
        entity = MagicMock(spec=["get_contacts"])
        entity.get_contacts.return_value = contacts
        return entity

    def test_no_contact_fails(self):
        scene = SimpleNamespace(entities={"button": self._entity({"geom_a": []})})
        robot = SimpleNamespace(entity=self._entity({"geom_a": []}))
        condition = InContactCondition(objects=("button",))
        assert condition.evaluate(scene, robot) is False

    def test_contact_passes(self):
        import numpy as np

        scene = SimpleNamespace(
            entities={"button": self._entity({"geom_a": np.zeros((1, 2))})}
        )
        robot = SimpleNamespace(entity=MagicMock())
        condition = InContactCondition(objects=("button",))
        assert condition.evaluate(scene, robot) is True

    def test_multi_object_all_required(self):
        import numpy as np

        ok = self._entity({"geom_a": np.zeros((1, 2))})
        empty = self._entity({"geom_a": []})
        scene = SimpleNamespace(entities={"card_a": ok, "card_b": empty})
        robot = SimpleNamespace(entity=MagicMock())
        condition = InContactCondition(objects=("card_a", "card_b"))
        assert condition.evaluate(scene, robot) is False
        scene.entities["card_b"] = ok
        assert condition.evaluate(scene, robot) is True

    def test_named_counterpart(self):
        import numpy as np

        scene = SimpleNamespace(
            entities={
                "hammer": self._entity({"geom_a": np.zeros((1, 2))}),
                "nail": MagicMock(),
            }
        )
        condition = InContactCondition(objects=("hammer",), with_entity="nail")
        assert condition.evaluate(scene, None) is True
        scene.entities["hammer"].get_contacts.return_value = {"geom_a": []}
        assert condition.evaluate(scene, None) is False

    def test_missing_hook_raises(self):
        scene = SimpleNamespace(entities={"button": MagicMock(spec=[])})
        condition = InContactCondition(objects=("button",))
        with pytest.raises(TypeError, match="contact"):
            condition.evaluate(scene, SimpleNamespace(entity=MagicMock()))


class TestCreateInContact:
    """Factory coverage for in_contact."""

    def test_in_contact(self):
        condition = create_success_condition(
            {
                "type": "in_contact",
                "params": {"objects": ["button"]},
            }
        )
        assert isinstance(condition, InContactCondition)
        assert condition.with_entity == "@robot"

    def test_in_contact_requires_objects(self):
        with pytest.raises(ConfigError, match="objects"):
            create_success_condition({"type": "in_contact", "params": {}})


class TestLiberoOn:
    """Tests for LiberoOnCondition (LIBERO-MAX ``on`` goals)."""

    def _entity(self, pos, touching=True):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(pos, dtype=float)
        entity.in_contact.return_value = touching
        return entity

    def _scene(self, obj_pos, support_pos, touching=True):
        return SimpleNamespace(
            entities={
                "bowl": self._entity(obj_pos, touching),
                "plate": self._entity(support_pos, touching),
            }
        )

    def test_object_support_passes(self):
        condition = LiberoOnCondition(object_name="bowl", support="plate")
        # support below, xy within 0.03, contact
        assert condition.evaluate(self._scene((0.5, 0.0, 0.95), (0.51, 0.0, 0.9)), None)

    def test_object_support_requires_contact(self):
        condition = LiberoOnCondition(object_name="bowl", support="plate")
        assert not condition.evaluate(
            self._scene((0.5, 0.0, 0.95), (0.51, 0.0, 0.9), touching=False), None
        )

    def test_object_support_rejects_wide_xy(self):
        condition = LiberoOnCondition(object_name="bowl", support="plate")
        # exactly 0.03 m off -> upstream uses strict <, so this fails
        assert not condition.evaluate(
            self._scene((0.53, 0.0, 0.95), (0.5, 0.0, 0.9)), None
        )

    def test_object_support_rejects_below_support(self):
        condition = LiberoOnCondition(object_name="bowl", support="plate")
        assert not condition.evaluate(
            self._scene((0.5, 0.0, 0.85), (0.5, 0.0, 0.9)), None
        )

    def test_site_support_under_slab(self):
        # identity xmat, half extents (0.1, 0.1, 0.005): slab z in
        # [0.0, 0.105] relative to center, |dx|,|dy| < 0.1
        condition = LiberoOnCondition(
            object_name="bowl",
            site_center=(0.0, 0.0, 0.9),
            site_xmat=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            site_halfsize=(0.1, 0.1, 0.005),
        )
        scene = SimpleNamespace(entities={"bowl": self._entity((0.05, -0.05, 0.905))})
        assert condition.evaluate(scene, None)

    def test_site_support_slab_bounds(self):
        condition = LiberoOnCondition(
            object_name="bowl",
            site_center=(0.0, 0.0, 0.9),
            site_xmat=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            site_halfsize=(0.1, 0.1, 0.005),
        )
        too_high = SimpleNamespace(entities={"bowl": self._entity((0.0, 0.0, 1.1))})
        outside_xy = SimpleNamespace(entities={"bowl": self._entity((0.1, 0.0, 0.905))})
        assert not condition.evaluate(too_high, None)
        assert not condition.evaluate(outside_xy, None)

    def test_site_support_requires_parent_contact(self):
        bowl = self._entity((0.0, 0.0, 0.905))
        cabinet = self._entity((0.0, 0.0, 0.8))
        bowl.in_contact.return_value = True
        condition = LiberoOnCondition(
            object_name="bowl",
            site_center=(0.0, 0.0, 0.9),
            site_xmat=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            site_halfsize=(0.1, 0.1, 0.005),
            site_contact="cabinet",
        )
        scene = SimpleNamespace(entities={"bowl": bowl, "cabinet": cabinet})
        assert condition.evaluate(scene, None)
        bowl.in_contact.return_value = False
        assert not condition.evaluate(scene, None)


class TestLiberoIn:
    """Tests for LiberoInCondition (LIBERO-MAX ``in`` goals, site boxes)."""

    def _condition(self):
        return LiberoInCondition(
            object_name="soup",
            site_center=(0.0, 0.0, 0.9),
            site_xmat=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            site_halfsize=(0.05, 0.05, 0.02),
        )

    def _scene(self, pos):
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(pos, dtype=float)
        return SimpleNamespace(entities={"soup": entity})

    def test_inside_passes(self):
        assert self._condition().evaluate(self._scene((0.0, 0.0, 0.9)), None)

    def test_strict_inequalities(self):
        # on the x boundary -> fails (upstream uses strict >/<)
        assert not self._condition().evaluate(self._scene((0.05, 0.0, 0.9)), None)

    def test_lower_z_margin(self):
        # box z in [0.88, 0.92]; lb.z is relaxed by 0.01 m -> [0.87, 0.92]
        assert self._condition().evaluate(self._scene((0.0, 0.0, 0.871)), None)
        assert not self._condition().evaluate(self._scene((0.0, 0.0, 0.87)), None)

    def test_rotated_box_uses_abs_xmat_size(self):
        # 90 deg yaw swaps x/y extents: |xmat @ half| = (0.02, 0.05, 0.01)
        condition = LiberoInCondition(
            object_name="soup",
            site_center=(0.0, 0.0, 0.9),
            site_xmat=(0, -1, 0, 1, 0, 0, 0, 0, 1),
            site_halfsize=(0.05, 0.02, 0.01),
        )
        entity = MagicMock()
        entity.get_pos.return_value = np.array([0.015, 0.04, 0.9])
        scene = SimpleNamespace(entities={"soup": entity})
        assert condition.evaluate(scene, None)


class TestLiberoJointState:
    """Tests for LiberoJointStateCondition (Close/TurnOn/TurnOff)."""

    def _entity(self, positions: dict):
        entity = MagicMock()
        entity.get_joint_position = lambda name: positions[name]
        return entity

    def test_close_all_joints_above_min(self):
        scene = SimpleNamespace(
            entities={"cabinet": self._entity({"j1": 0.001, "j2": 0.002})}
        )
        condition = LiberoJointStateCondition(
            object_name="cabinet", joints=("j1", "j2"), mode="close", range=(0.0, 0.005)
        )
        assert condition.evaluate(scene, None)
        scene.entities["cabinet"].get_joint_position = lambda name: -0.15
        assert not condition.evaluate(scene, None)

    def test_turn_on_any_joint(self):
        scene = SimpleNamespace(entities={"stove": self._entity({"b1": 0.0})})
        condition = LiberoJointStateCondition(
            object_name="stove", joints=("b1",), mode="turn_on", range=(0.5, 2.1)
        )
        assert not condition.evaluate(scene, None)
        scene.entities["stove"].get_joint_position = lambda name: 0.5
        assert condition.evaluate(scene, None)

    def test_turn_off_all_joints_below_max(self):
        scene = SimpleNamespace(entities={"stove": self._entity({"b1": -0.001})})
        condition = LiberoJointStateCondition(
            object_name="stove", joints=("b1",), mode="turn_off", range=(-0.005, 0.0)
        )
        # upstream: qpos < max(turnoff_range) — exactly 0.0 is NOT "off"
        assert condition.evaluate(scene, None)
        scene.entities["stove"].get_joint_position = lambda name: 0.0
        assert not condition.evaluate(scene, None)

    def test_articulation_backend_hook(self):
        entity = MagicMock()
        entity.get_joint_position = None
        entity.get_joint_names = lambda: ["j1"]
        entity.get_joint_qs_idx_local = lambda name: [0]
        entity.get_qpos = lambda: np.array([0.003])
        scene = SimpleNamespace(entities={"cabinet": entity})
        condition = LiberoJointStateCondition(
            object_name="cabinet", joints=("j1",), mode="close", range=(0.0, 0.005)
        )
        assert condition.evaluate(scene, None)

    def test_spawn_renamed_joint_matches_by_suffix(self):
        """Component spawns carry bare MJCF names + __dupN repair suffixes.

        LIBERO composes with an instance prefix (``flat_stove_1_button``);
        the component-level spawn exposes ``button__dup2`` after the
        duplicate-name repair — the comparison ignores the repair suffix.
        """
        entity = MagicMock()
        entity.get_joint_position = None
        entity.get_joint_names = lambda: ["button__dup2"]
        entity.get_joint_qs_idx_local = lambda name: [0]
        entity.get_qpos = lambda: np.array([-0.001])
        scene = SimpleNamespace(entities={"stove": entity})
        condition = LiberoJointStateCondition(
            object_name="stove",
            joints=("flat_stove_1_button",),
            mode="turn_off",
            range=(-0.005, 0.0),
        )
        assert condition.evaluate(scene, None) is True

    def test_unknown_mode_raises(self):
        condition = LiberoJointStateCondition(
            object_name="cabinet", joints=("j1",), mode="tilt", range=(0.0, 1.0)
        )
        with pytest.raises(ValueError, match="mode"):
            condition.evaluate(
                SimpleNamespace(entities={"cabinet": self._entity({"j1": 0.0})}), None
            )


class TestAllCondition:
    """Tests for AllCondition (LIBERO ``And`` goal trees)."""

    def test_all_must_hold(self):
        inner_a = LiberoInCondition(
            object_name="soup",
            site_center=(0, 0, 0.9),
            site_xmat=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            site_halfsize=(0.1, 0.1, 0.1),
        )
        inner_b = LiberoJointStateCondition(
            object_name="cabinet", joints=("j1",), mode="close", range=(0.0, 0.005)
        )
        condition = AllCondition(conditions=(inner_a, inner_b))
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray([0.0, 0.0, 0.9])
        entity.get_joint_position = lambda name: 0.001
        scene = SimpleNamespace(entities={"soup": entity, "cabinet": entity})
        assert condition.evaluate(scene, None)
        entity.get_joint_position = lambda name: -0.2
        assert not condition.evaluate(scene, None)

    def test_factory_recursion(self):
        condition = create_success_condition(
            {
                "type": "all",
                "params": {
                    "conditions": [
                        {
                            "type": "libero_in",
                            "params": {
                                "object": "soup",
                                "site_center": [0, 0, 0.9],
                                "site_xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                "site_halfsize": [0.1, 0.1, 0.1],
                            },
                        },
                        {
                            "type": "libero_joint_state",
                            "params": {
                                "entity": "cabinet",
                                "joints": ["j1"],
                                "mode": "close",
                                "range": [0.0, 0.005],
                            },
                        },
                    ]
                },
            }
        )
        assert isinstance(condition, AllCondition)
        assert len(condition.conditions) == 2

    def test_factory_requires_conditions(self):
        with pytest.raises(ConfigError, match="conditions"):
            create_success_condition({"type": "all", "params": {}})


class TestCreateLibero:
    """Factory coverage for the LIBERO-MAX success types."""

    def test_libero_on_object_mode(self):
        condition = create_success_condition(
            {
                "type": "libero_on",
                "params": {"object": "bowl", "support": "plate"},
            }
        )
        assert isinstance(condition, LiberoOnCondition)

    def test_libero_on_site_mode(self):
        condition = create_success_condition(
            {
                "type": "libero_on",
                "params": {
                    "object": "bowl",
                    "site_center": [0, 0, 0.9],
                    "site_xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "site_halfsize": [0.1, 0.1, 0.005],
                    "site_contact": "cabinet",
                },
            }
        )
        assert isinstance(condition, LiberoOnCondition)
        assert condition.site_contact == "cabinet"

    def test_libero_on_requires_support_or_site(self):
        with pytest.raises(ConfigError, match="support"):
            create_success_condition(
                {"type": "libero_on", "params": {"object": "bowl"}}
            )

    def test_libero_joint_state_bad_mode(self):
        with pytest.raises(ConfigError, match="mode"):
            create_success_condition(
                {
                    "type": "libero_joint_state",
                    "params": {
                        "entity": "stove",
                        "joints": ["b1"],
                        "mode": "tilt",
                        "range": [0.0, 1.0],
                    },
                }
            )

    def test_libero_in_requires_full_site(self):
        with pytest.raises(ConfigError, match="site_halfsize"):
            create_success_condition(
                {
                    "type": "libero_in",
                    "params": {
                        "object": "soup",
                        "site_center": [0, 0, 0.9],
                        "site_xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    },
                }
            )
