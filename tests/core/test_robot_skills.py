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
    CloseSkill,
    GraspSkill,
    HandoverSkill,
    InsertSkill,
    JointInfo,
    OpenSkill,
    PlaceSkill,
    PushUpSkill,
    SkillContext,
    SkillExecutionError,
    SkillSequence,
    StackSkill,
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
        # EE goals of every plan() call, in order (plan_with_fallback plans
        # via inverse_kinematics) — enables geometric assertions.
        self.ik_goals: list[np.ndarray] = []

    def get_qpos(self):
        self.calls.append("get_qpos")
        return np.zeros(7)

    def inverse_kinematics(self, link, pos=None, quat=None):
        self.calls.append("ik")
        if pos is not None:
            self.ik_goals.append(np.asarray(pos, dtype=np.float64).reshape(3))
        if "ik" in self.fail_methods:
            raise RuntimeError("mock IK failure")
        return np.zeros(7)

    def plan_path(self, q_goal, num_waypoints=20):
        self.calls.append("plan_path")
        if "plan_path" in self.fail_methods:
            raise RuntimeError("mock OMPL failure")
        # plan_with_fallback rejects all-zeros paths (real OMPL failure
        # signature), so the mock returns a valid non-zero trajectory.
        return np.linspace(0.1, 0.2, num=num_waypoints * 7).reshape(num_waypoints, 7)


class _RecordingHooks:
    """SkillContext hooks that record invocations."""

    def __init__(self, fail=None):
        self.executed: list[np.ndarray] = []
        self.gripper: list[float] = []
        self.attached: list[str] = []
        self.detached: list[str] = []
        self.fail = fail  # hook name to raise SkillExecutionError
        # P1 optional-hook recordings.
        self.extents_calls: list[str] = []
        self.joint_calls: list[tuple[str, str]] = []
        self.driven: list[tuple[str, str, float]] = []
        self.extents: dict[str, np.ndarray] = {
            "cube": np.array([0.025, 0.025, 0.025]),
            "blue_cube": np.array([0.02, 0.02, 0.02]),
        }
        self.joint_info = JointInfo(
            name="drawer_joint",
            axis=np.array([0.0, 0.0, 1.0]),
            anchor=np.array([0.0, 0.0, 0.0]),
            dof_index=0,
            position=0.0,
            lower=-3.0,
            upper=3.0,
        )

    def get_entity_pose(self, name):
        return np.array([0.5, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])

    def get_entity_extents(self, name):
        if self.fail == "get_entity_extents":
            raise SkillExecutionError("mock extents failure")
        self.extents_calls.append(name)
        if name not in self.extents:
            raise KeyError(name)
        return self.extents[name]

    def get_joint_info(self, object_name, joint_name):
        if self.fail == "get_joint_info":
            raise SkillExecutionError("mock joint info failure")
        self.joint_calls.append((object_name, joint_name))
        return self.joint_info

    def drive_joint(self, object_name, joint_name, target):
        if self.fail == "drive_joint":
            raise SkillExecutionError("mock joint drive failure")
        self.driven.append((object_name, joint_name, float(target)))

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
            get_entity_extents=self.get_entity_extents,
            get_joint_info=self.get_joint_info,
            drive_joint=self.drive_joint,
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


class TestHandoverSkill:
    """Handover skill phase flow and stage mapping."""

    def test_happy_path(self):
        hooks = _RecordingHooks()
        result = HandoverSkill().execute(
            hooks.context(),
            object_name="cube",
            transfer_pos=np.array([0.5, 0.2, 0.05]),
        )

        assert result.success is True
        assert result.stage == "success"
        assert len(hooks.executed) == 3  # approach, transfer, retreat
        assert hooks.gripper == [GRIPPER_OPEN]
        assert hooks.detached == ["cube"]

    def test_motion_failure_maps_to_transport_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("ik",))
        result = HandoverSkill().execute(
            hooks.context(robot),
            object_name="cube",
            transfer_pos=np.array([0.5, 0.2, 0.05]),
        )
        assert result.success is False
        assert result.stage == "transport_fail"
        assert hooks.detached == []

    def test_detach_failure_maps_to_place_fail(self):
        hooks = _RecordingHooks(fail="detach")
        result = HandoverSkill().execute(
            hooks.context(),
            object_name="cube",
            transfer_pos=np.array([0.5, 0.2, 0.05]),
        )
        assert result.success is False
        assert result.stage == "place_fail"

    def test_missing_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = HandoverSkill().execute(
            context, object_name="ghost", transfer_pos=np.array([0.5, 0.0, 0.1])
        )
        assert result.success is False
        assert result.stage == "load_fail"
        assert hooks.executed == []


class TestInsertSkill:
    """Insert skill phase flow, geometry and stage mapping."""

    def test_happy_path_downward_axis(self):
        hooks = _RecordingHooks()
        result = InsertSkill().execute(
            hooks.context(), object_name="cube", insert_pos=np.array([0.5, 0.0, 0.05])
        )

        assert result.success is True
        assert result.stage == "success"
        assert len(hooks.executed) == 3  # align, insert, retreat
        assert hooks.gripper == [GRIPPER_OPEN]
        assert hooks.detached == ["cube"]

    def test_approach_goal_above_mouth(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        InsertSkill().execute(
            context,
            object_name="cube",
            insert_pos=np.array([0.5, 0.0, 0.05]),
            mouth_pos=np.array([0.5, 0.0, 0.2]),
            standoff=0.08,
        )
        np.testing.assert_allclose(robot.ik_goals[0], [0.5, 0.0, 0.28], atol=1e-9)
        np.testing.assert_allclose(robot.ik_goals[1], [0.5, 0.0, 0.05], atol=1e-9)

    def test_custom_axis_horizontal(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        InsertSkill().execute(
            context,
            object_name="cube",
            insert_pos=np.array([0.5, 0.0, 0.05]),
            axis=np.array([1.0, 0.0, 0.0]),
            standoff=0.08,
        )
        # approach = insert - axis * standoff
        np.testing.assert_allclose(robot.ik_goals[0], [0.42, 0.0, 0.05], atol=1e-9)

    def test_zero_norm_axis_maps_to_error(self):
        hooks = _RecordingHooks()
        result = InsertSkill().execute(
            hooks.context(),
            object_name="cube",
            insert_pos=np.array([0.5, 0.0, 0.05]),
            axis=np.array([0.0, 0.0, 0.0]),
        )
        assert result.success is False
        assert result.stage == "error"
        assert hooks.executed == []
        assert hooks.gripper == []

    def test_motion_failure_maps_to_transport_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("ik",))
        result = InsertSkill().execute(
            hooks.context(robot),
            object_name="cube",
            insert_pos=np.array([0.5, 0.0, 0.05]),
        )
        assert result.success is False
        assert result.stage == "transport_fail"

    def test_detach_failure_maps_to_place_fail(self):
        hooks = _RecordingHooks(fail="detach")
        result = InsertSkill().execute(
            hooks.context(), object_name="cube", insert_pos=np.array([0.5, 0.0, 0.05])
        )
        assert result.success is False
        assert result.stage == "place_fail"

    def test_missing_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = InsertSkill().execute(
            context, object_name="ghost", insert_pos=np.array([0.5, 0.0, 0.05])
        )
        assert result.success is False
        assert result.stage == "load_fail"


class TestSwingSkills:
    """open/close phase flow, arc geometry and stage mapping."""

    def test_open_happy_path(self):
        hooks = _RecordingHooks()
        result = OpenSkill().execute(
            hooks.context(),
            object_name="cube",
            joint_name="drawer_joint",
            swing_steps=5,
        )

        assert result.success is True
        assert result.stage == "success"
        # approach, descend, 5 swing steps, retreat
        assert len(hooks.executed) == 8
        assert hooks.gripper == [GRIPPER_CLOSED, GRIPPER_OPEN]
        assert hooks.attached == ["cube"] and hooks.detached == ["cube"]
        assert hooks.joint_calls == [("cube", "drawer_joint")]
        targets = [t for _, _, t in hooks.driven]
        expected = np.linspace(0.0, np.pi / 3.0, 6)[1:]
        np.testing.assert_allclose(targets, expected, atol=1e-9)

    def test_close_default_angle_negative(self):
        hooks = _RecordingHooks()
        result = CloseSkill().execute(
            hooks.context(),
            object_name="cube",
            joint_name="drawer_joint",
            swing_steps=3,
        )
        assert result.success is True
        targets = [t for _, _, t in hooks.driven]
        assert targets == sorted(targets, reverse=True)
        np.testing.assert_allclose(targets[-1], -np.pi / 3.0, atol=1e-9)

    def test_arc_rotates_about_joint_axis(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        result = OpenSkill().execute(
            context,
            object_name="cube",
            joint_name="drawer_joint",
            angle=np.pi / 2.0,
            grasp_pos=np.array([0.2, 0.0, 0.05]),
            swing_steps=1,
        )
        assert result.success is True
        # approach, descend, arc (90 deg about +Z maps +X -> +Y), retreat
        np.testing.assert_allclose(robot.ik_goals[0], [0.2, 0.0, 0.15], atol=1e-9)
        np.testing.assert_allclose(robot.ik_goals[1], [0.2, 0.0, 0.05], atol=1e-9)
        np.testing.assert_allclose(robot.ik_goals[2], [0.0, 0.2, 0.05], atol=1e-9)
        np.testing.assert_allclose(robot.ik_goals[3], [0.0, 0.2, 0.15], atol=1e-9)

    def test_missing_joint_info_hook_maps_to_error(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        context.get_joint_info = None
        result = OpenSkill().execute(
            context, object_name="cube", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "error"
        assert "get_joint_info" in result.message
        assert hooks.executed == [] and hooks.gripper == []

    def test_missing_drive_hook_maps_to_error(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        context.drive_joint = None
        result = CloseSkill().execute(
            context, object_name="cube", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "error"
        assert "drive_joint" in result.message

    def test_joint_lookup_failure_maps_to_load_fail(self):
        hooks = _RecordingHooks(fail="get_joint_info")
        result = OpenSkill().execute(
            hooks.context(), object_name="cube", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "load_fail"

    def test_attach_failure_maps_to_grasp_fail(self):
        hooks = _RecordingHooks(fail="attach")
        result = OpenSkill().execute(
            hooks.context(), object_name="cube", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "grasp_fail"
        assert hooks.driven == []  # swing never started

    def test_drive_failure_maps_to_place_fail(self):
        hooks = _RecordingHooks(fail="drive_joint")
        result = OpenSkill().execute(
            hooks.context(), object_name="cube", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "place_fail"
        assert "swing" in result.message

    def test_zero_swing_steps_maps_to_error(self):
        hooks = _RecordingHooks()
        result = OpenSkill().execute(
            hooks.context(),
            object_name="cube",
            joint_name="drawer_joint",
            swing_steps=0,
        )
        assert result.success is False
        assert result.stage == "error"

    def test_missing_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = OpenSkill().execute(
            context, object_name="ghost", joint_name="drawer_joint"
        )
        assert result.success is False
        assert result.stage == "load_fail"


class TestStackSkill:
    """Stack skill target resolution, phase flow and stage mapping."""

    def test_happy_path_computes_target_from_extents(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        result = StackSkill().execute(
            context, object_name="cube", support_name="blue_cube"
        )

        assert result.success is True
        assert result.stage == "success"
        assert len(hooks.executed) == 3
        assert hooks.gripper == [GRIPPER_OPEN]
        assert hooks.detached == ["cube"]
        assert hooks.extents_calls == ["blue_cube", "cube"]
        # support top 0.05 + 0.02 = 0.07; carried half-height 0.025 -> 0.095
        np.testing.assert_allclose(robot.ik_goals[1], [0.5, 0.0, 0.095], atol=1e-9)

    def test_explicit_params_bypass_extents_hook(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        context.get_entity_extents = None
        result = StackSkill().execute(
            context,
            object_name="cube",
            support_name="blue_cube",
            support_top_z=0.07,
            carried_half_height=0.025,
        )
        assert result.success is True
        assert hooks.extents_calls == []

    def test_missing_extents_without_params_maps_to_error(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        context.get_entity_extents = None
        result = StackSkill().execute(
            context, object_name="cube", support_name="blue_cube"
        )
        assert result.success is False
        assert result.stage == "error"
        assert hooks.executed == []

    def test_explicit_target_pos_overrides(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        result = StackSkill().execute(
            context,
            object_name="cube",
            support_name="blue_cube",
            target_pos=np.array([0.6, 0.1, 0.2]),
        )
        assert result.success is True
        np.testing.assert_allclose(robot.ik_goals[1], [0.6, 0.1, 0.2], atol=1e-9)
        assert hooks.extents_calls == []

    def test_motion_failure_maps_to_transport_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("ik",))
        result = StackSkill().execute(
            hooks.context(robot), object_name="cube", support_name="blue_cube"
        )
        assert result.success is False
        assert result.stage == "transport_fail"

    def test_detach_failure_maps_to_place_fail(self):
        hooks = _RecordingHooks(fail="detach")
        result = StackSkill().execute(
            hooks.context(), object_name="cube", support_name="blue_cube"
        )
        assert result.success is False
        assert result.stage == "place_fail"

    def test_missing_support_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = StackSkill().execute(context, object_name="cube", support_name="ghost")
        assert result.success is False
        assert result.stage == "load_fail"


class TestPushUpSkill:
    """Push skill contact geometry, phase flow and stage mapping."""

    def test_happy_path(self):
        hooks = _RecordingHooks()
        result = PushUpSkill().execute(hooks.context(), object_name="cube")

        assert result.success is True
        assert result.stage == "success"
        assert len(hooks.executed) == 4  # approach, descend, push, retreat
        assert hooks.gripper == [GRIPPER_OPEN]
        assert hooks.attached == [] and hooks.detached == []

    def test_contact_point_behind_surface(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        PushUpSkill().execute(
            context,
            object_name="cube",
            direction=np.array([1.0, 0.0, 0.0]),
        )
        # contact = obj (0.5) - (0.025 + 0.01) = 0.465; push end = +0.15
        np.testing.assert_allclose(robot.ik_goals[1], [0.465, 0.0, 0.05], atol=1e-9)
        np.testing.assert_allclose(robot.ik_goals[2], [0.615, 0.0, 0.05], atol=1e-9)

    def test_zero_norm_direction_maps_to_error(self):
        hooks = _RecordingHooks()
        result = PushUpSkill().execute(
            hooks.context(),
            object_name="cube",
            direction=np.array([0.0, 0.0, 0.0]),
        )
        assert result.success is False
        assert result.stage == "error"
        assert hooks.executed == []

    def test_missing_extents_without_param_maps_to_error(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        context.get_entity_extents = None
        result = PushUpSkill().execute(context, object_name="cube")
        assert result.success is False
        assert result.stage == "error"

    def test_explicit_half_extent_bypasses_hook(self):
        hooks = _RecordingHooks()
        robot = _MockRobot()
        context = hooks.context(robot)
        context.get_entity_extents = None
        result = PushUpSkill().execute(
            context,
            object_name="cube",
            direction=np.array([1.0, 0.0, 0.0]),
            half_extent=0.03,
        )
        assert result.success is True
        # contact = 0.5 - (0.03 + 0.01) = 0.46
        np.testing.assert_allclose(robot.ik_goals[1], [0.46, 0.0, 0.05], atol=1e-9)

    def test_motion_failure_maps_to_transport_fail(self):
        hooks = _RecordingHooks()
        robot = _MockRobot(fail_methods=("ik",))
        result = PushUpSkill().execute(hooks.context(robot), object_name="cube")
        assert result.success is False
        assert result.stage == "transport_fail"

    def test_missing_entity_maps_to_load_fail(self):
        hooks = _RecordingHooks()

        def _boom(name):
            raise KeyError(name)

        context = hooks.context()
        context.get_entity_pose = _boom
        result = PushUpSkill().execute(context, object_name="ghost")
        assert result.success is False
        assert result.stage == "load_fail"


class TestSkillSequenceP1:
    """Orchestrator registration of the six P1 primitives."""

    def test_new_primitive_names_accepted(self):
        sequence = SkillSequence(
            [
                ("handover", {"object_name": "cube", "transfer_pos": np.zeros(3)}),
                ("insert", {"object_name": "cube", "insert_pos": np.zeros(3)}),
                ("open", {"object_name": "cube", "joint_name": "j"}),
                ("close", {"object_name": "cube", "joint_name": "j"}),
                ("stack", {"object_name": "cube", "support_name": "blue_cube"}),
                ("push_up", {"object_name": "cube"}),
            ]
        )
        assert [s.skill for s in sequence.steps] == [
            "handover",
            "insert",
            "open",
            "close",
            "stack",
            "push_up",
        ]

    def test_grasp_then_handover_succeeds(self):
        hooks = _RecordingHooks()
        sequence = SkillSequence(
            [
                ("grasp", {"object_name": "cube"}),
                (
                    "handover",
                    {"object_name": "cube", "transfer_pos": np.array([0.5, 0.0, 0.2])},
                ),
            ]
        )
        result = sequence.execute(hooks.context())

        assert result.success is True
        assert hooks.attached == ["cube"] and hooks.detached == ["cube"]
        assert len(result.trajectories) == 6  # grasp 3 + handover 3

    def test_base_class_name_not_registered(self):
        # "swing" is the shared base class; only open/close are registered.
        with pytest.raises(ValueError, match="unknown skill 'swing'"):
            SkillSequence([("swing", {"object_name": "cube", "joint_name": "j"})])


class TestSkillContextDefaults:
    """Backward compatibility of the optional P1 hooks."""

    def test_new_hooks_default_to_none(self):
        context = SkillContext(
            robot=None,
            planner=None,
            ee_link="ee_link",
            get_entity_pose=lambda name: (np.zeros(3), np.zeros(4)),
            execute_trajectory=lambda traj: None,
            set_gripper=lambda value: None,
            attach=lambda name: None,
            detach=lambda name: None,
        )
        assert context.get_entity_extents is None
        assert context.get_joint_info is None
        assert context.drive_joint is None

    def test_legacy_construction_and_plan_unchanged(self):
        hooks = _RecordingHooks()
        context = hooks.context()
        traj = context.plan(np.array([0.4, 0.0, 0.3]), None)
        assert traj.ndim == 2 and traj.shape[1] == 7
