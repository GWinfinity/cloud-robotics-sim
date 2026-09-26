"""Tests for the robot skill primitives (core/robot_skills).

All tests are Genesis-free: the robot backend and simulator hooks are
recorded mocks; planning runs through the OMPL branch of
``plan_with_fallback`` (planner=None).

Python 3.9 compatible (``from __future__ import annotations``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cloud_robotics_sim.core.robot_skills import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GraspSkill,
    PlaceSkill,
    SkillContext,
    SkillExecutionError,
    SkillSequence,
    candidate_in_world,
    first_model_data,
    load_grasp_candidates,
    score_candidates,
)


class _MockRobot:
    """ArticulationBackend stand-in for the OMPL fallback path."""

    n_dofs = 7

    def __init__(self, fail_methods=()):
        self.fail_methods = set(fail_methods)
        self.calls: list[str] = []

    def get_qpos(self):
        self.calls.append("get_qpos")
        return np.zeros(7)

    def inverse_kinematics(self, link, pos=None, quat=None):
        self.calls.append("ik")
        if "ik" in self.fail_methods:
            raise RuntimeError("mock IK failure")
        return np.zeros(7)

    def plan_path(self, q_goal, num_waypoints=20):
        self.calls.append("plan_path")
        if "plan_path" in self.fail_methods:
            raise RuntimeError("mock OMPL failure")
        return np.zeros((num_waypoints, 7))


class _RecordingHooks:
    """SkillContext hooks that record invocations."""

    def __init__(self, fail=None):
        self.executed: list[np.ndarray] = []
        self.gripper: list[float] = []
        self.attached: list[str] = []
        self.detached: list[str] = []
        self.fail = fail  # hook name to raise SkillExecutionError

    def get_entity_pose(self, name):
        return np.array([0.5, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])

    def execute_trajectory(self, traj):
        if self.fail == "execute_trajectory":
            raise SkillExecutionError("mock execution failure")
        self.executed.append(np.asarray(traj))

    def set_gripper(self, value):
        if self.fail == "set_gripper":
            raise SkillExecutionError("mock gripper failure")
        self.gripper.append(value)

    def attach(self, name):
        if self.fail == "attach":
            raise SkillExecutionError("mock attach failure")
        self.attached.append(name)

    def detach(self, name):
        if self.fail == "detach":
            raise SkillExecutionError("mock detach failure")
        self.detached.append(name)

    def context(self, robot=None) -> SkillContext:
        return SkillContext(
            robot=robot or _MockRobot(),
            planner=None,
            ee_link="ee_link",
            get_entity_pose=self.get_entity_pose,
            execute_trajectory=self.execute_trajectory,
            set_gripper=self.set_gripper,
            attach=self.attach,
            detach=self.detach,
        )


class TestGraspSkill:
    """Grasp skill phase flow and stage mapping."""

    def test_happy_path_phase_order(self):
        hooks = _RecordingHooks()
        result = GraspSkill().execute(hooks.context(), object_name="cube")

        assert result.success is True
        assert result.stage == "success"
        # pre-grasp, descend, lift: three trajectories
        assert len(hooks.executed) == 3
        assert hooks.gripper == [GRIPPER_CLOSED]
        assert hooks.attached == ["cube"]

    def test_default_grasp_point_from_entity_pose(self):
        hooks = _RecordingHooks()
        GraspSkill().execute(hooks.context(), object_name="cube")
        assert all(np.all(np.isfinite(t)) for t in hooks.executed)

    def test_plan_failure_maps_to_plan_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("plan_path",))
        result = GraspSkill().execute(hooks.context(robot), object_name="cube")

        assert result.success is False
        assert result.stage == "plan_fail"
        assert "pre-grasp" in result.message
        assert hooks.attached == []

    def test_attach_failure_maps_to_grasp_fail(self):
        hooks = _RecordingHooks(fail="attach")
        result = GraspSkill().execute(hooks.context(), object_name="cube")

        assert result.success is False
        assert result.stage == "grasp_fail"

    def test_execution_failure_in_lift_maps_to_lift_fail(self):
        hooks = _RecordingHooks()

        original = hooks.execute_trajectory

        def _fail_on_third(traj):
            if len(hooks.executed) == 2:
                raise SkillExecutionError("lift blocked")
            original(traj)

        hooks.execute_trajectory = _fail_on_third
        context = hooks.context()
        result = GraspSkill().execute(context, object_name="cube")
        assert result.success is False
        assert result.stage == "lift_fail"

    def test_missing_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = GraspSkill().execute(context, object_name="ghost")
        assert result.success is False
        assert result.stage == "load_fail"


class TestPlaceSkill:
    """Place skill phase flow and stage mapping."""

    def test_happy_path(self):
        hooks = _RecordingHooks()
        result = PlaceSkill().execute(
            hooks.context(), object_name="cube", place_pos=np.array([0.5, 0.2, 0.05])
        )

        assert result.success is True
        assert result.stage == "success"
        assert len(hooks.executed) == 3  # approach, descend, retreat
        assert hooks.gripper == [GRIPPER_OPEN]
        assert hooks.detached == ["cube"]

    def test_motion_failure_maps_to_transport_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("ik",))
        result = PlaceSkill().execute(
            hooks.context(robot),
            object_name="cube",
            place_pos=np.array([0.5, 0.2, 0.05]),
        )
        assert result.success is False
        assert result.stage == "transport_fail"

    def test_detach_failure_maps_to_place_fail(self):
        hooks = _RecordingHooks(fail="detach")
        result = PlaceSkill().execute(
            hooks.context(), object_name="cube", place_pos=np.array([0.5, 0.2, 0.05])
        )
        assert result.success is False
        assert result.stage == "place_fail"


class TestSkillSequence:
    """Orchestrator ordering and failure short-circuit."""

    def test_grasp_then_place_succeeds(self):
        hooks = _RecordingHooks()
        sequence = SkillSequence(
            [
                ("grasp", {"object_name": "cube"}),
                (
                    "place",
                    {"object_name": "cube", "place_pos": np.array([0.5, 0.0, 0.1])},
                ),
            ]
        )
        result = sequence.execute(hooks.context())

        assert result.success is True
        assert result.stage == "success"
        assert hooks.attached == ["cube"] and hooks.detached == ["cube"]

    def test_stops_at_first_failure_with_stage(self):
        hooks = _RecordingHooks(fail="attach")
        sequence = SkillSequence(
            [
                ("grasp", {"object_name": "cube"}),
                (
                    "place",
                    {"object_name": "cube", "place_pos": np.array([0.5, 0.0, 0.1])},
                ),
            ]
        )
        result = sequence.execute(hooks.context())

        assert result.success is False
        assert result.stage == "grasp_fail"
        assert hooks.detached == []  # place never ran

    def test_unknown_skill_rejected(self):
        with pytest.raises(ValueError, match="unknown skill"):
            SkillSequence([("teleport", {})])


def _write_model_data(tmp_path: Path, poses) -> Path:
    path = tmp_path / "model_data0.json"
    path.write_text(
        json.dumps(
            {
                "center": [0, 0, 0],
                "extents": [1, 1, 1],
                "scale": [0.5, 0.5, 0.5],
                "target_pose": [],
                "contact_points_pose": poses,
            }
        ),
        encoding="utf-8",
    )
    return path


# Identity 4x4 with a z-pointing approach column.
POSE_Z = [
    [1.0, 0.0, 0.0, 0.1],
    [0.0, 1.0, 0.0, 0.2],
    [0.0, 0.0, 1.0, 0.3],
    [0.0, 0.0, 0.0, 1.0],
]
# Rotated 90 deg about y: approach column (z) becomes +x in object frame.
POSE_X = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class TestGraspPoints:
    """model_data grasp candidate loading and scoring."""

    def test_load_candidates(self, tmp_path):
        path = _write_model_data(tmp_path, [POSE_Z, POSE_X])
        candidates = load_grasp_candidates(path)

        assert len(candidates) == 2
        np.testing.assert_allclose(candidates[0].position, [0.1, 0.2, 0.3], atol=1e-9)
        np.testing.assert_allclose(candidates[0].approach, [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(candidates[1].approach, [1.0, 0.0, 0.0], atol=1e-9)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_grasp_candidates(tmp_path / "nope.json")

    def test_empty_poses(self, tmp_path):
        path = _write_model_data(tmp_path, [])
        with pytest.raises(ValueError, match="contact_points_pose"):
            load_grasp_candidates(path)

    def test_first_model_data_picks_lowest_index(self, tmp_path):
        _write_model_data(tmp_path, [POSE_Z])
        (tmp_path / "model_data3.json").write_text("{}", encoding="utf-8")
        found = first_model_data(tmp_path)
        assert found is not None and found.name == "model_data0.json"

    def test_candidate_to_world_applies_scale_and_rotation(self):
        from cloud_robotics_sim.core.robot_skills import GraspCandidate

        candidate = GraspCandidate(
            position=np.array([1.0, 0.0, 0.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            matrix=np.eye(4),
        )
        # 90 deg about z maps +x -> +y
        quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        pos, approach = candidate_in_world(candidate, [1, 2, 3], quat, scale=0.5)
        np.testing.assert_allclose(pos, [1.0, 2.5, 3.0], atol=1e-9)
        np.testing.assert_allclose(approach, [0.0, 0.0, 1.0], atol=1e-9)

    def test_scoring_prefers_top_down(self):
        top_down = (np.array([0.4, 0.0, 0.3]), np.array([0.0, 0.0, 1.0]))
        sideways = (np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]))
        order = score_candidates([sideways, top_down], ee_pos=np.array([0.2, 0.0, 0.4]))
        assert order[0] == 1
