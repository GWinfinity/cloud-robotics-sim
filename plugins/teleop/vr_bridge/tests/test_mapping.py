"""Semantic mapping tests: YAML-driven controller -> robot intent mapping."""

from __future__ import annotations

import pytest
import yaml
from conftest import MAPPINGS_DIR, make_hand, make_state_dict
from vr_bridge.core.mapping import SemanticMapper
from vr_bridge.core.messages import ControllerState, EventMsg


@pytest.fixture()
def mapper() -> SemanticMapper:
    """The franka_single mapping table."""
    return SemanticMapper.from_yaml(MAPPINGS_DIR / "franka_single.yaml")


def _state(right_grip=0.0, right_trigger=0.0, **kwargs) -> ControllerState:
    right = make_hand(grip=right_grip, trigger=right_trigger)
    return ControllerState.from_dict(make_state_dict(right=right, **kwargs))


def test_franka_single_config_parsed(mapper):
    """The franka_single table parses into the expected arms/grippers/buttons."""
    assert len(mapper.arms) == 1
    arm = mapper.arms[0]
    assert arm.name == "right_arm"
    assert arm.source == "right"
    assert arm.ee_link == "hand"
    assert arm.dofs == [0, 1, 2, 3, 4, 5, 6]
    assert arm.pos_scale == pytest.approx(1.0)

    assert len(mapper.grippers) == 1
    gripper = mapper.grippers[0]
    assert gripper.name == "gripper"
    assert gripper.source == "right"
    assert gripper.dofs == [7, 8]
    assert gripper.open_value == pytest.approx(0.04)
    assert gripper.close_value == pytest.approx(0.0)

    assert mapper.grip_threshold == pytest.approx(0.5)
    assert mapper.buttons == {
        "a": "reset_episode",
        "b": "record_toggle",
        "menu": "emergency_stop",
    }


def test_right_hand_maps_to_ee_intent(mapper):
    """The right hand pose flows into the right_arm intent unchanged."""
    state = _state()
    action = mapper.map(state)
    intent = action.arm_intents["right_arm"]
    assert intent.pose.pos == pytest.approx([0.3, 0.0, 0.5])
    assert intent.pose.quat == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert intent.mapping.ee_link == "hand"


def test_clutch_engagement_threshold(mapper):
    """Grip >= grip_threshold engages the clutch (deadman semantics)."""
    assert mapper.map(_state(right_grip=0.0)).arm_intents["right_arm"].engaged is False
    assert mapper.map(_state(right_grip=0.49)).arm_intents["right_arm"].engaged is False
    assert mapper.map(_state(right_grip=0.5)).arm_intents["right_arm"].engaged is True
    assert mapper.map(_state(right_grip=1.0)).arm_intents["right_arm"].engaged is True


def test_trigger_maps_to_gripper_command(mapper):
    """The right trigger drives the gripper close fraction directly."""
    action = mapper.map(_state(right_trigger=0.7))
    assert action.gripper_cmds["gripper"] == pytest.approx(0.7)


def test_gripper_joint_value_interpolation(mapper):
    """to_joint_value interpolates open->close and clamps out-of-range input."""
    gripper = mapper.grippers[0]
    assert gripper.to_joint_value(0.0) == pytest.approx(0.04)  # fully open
    assert gripper.to_joint_value(1.0) == pytest.approx(0.0)  # fully closed
    assert gripper.to_joint_value(0.5) == pytest.approx(0.02)
    assert gripper.to_joint_value(-0.5) == pytest.approx(0.04)  # clamped
    assert gripper.to_joint_value(1.5) == pytest.approx(0.0)  # clamped


def test_button_events_mapped_to_semantics(mapper):
    """Raw button names are translated; unknown buttons are ignored."""
    events = [
        EventMsg(event="menu", pressed=True),
        EventMsg(event="x", pressed=True),  # not in the mapping table
        EventMsg(event="a", pressed=True),
    ]
    action = mapper.map(_state(), events)
    assert action.events == ["emergency_stop", "reset_episode"]


def test_unknown_controller_source_raises(mapper):
    """An arm wired to a non-existent controller source is an error."""
    bad = yaml.safe_load(
        (MAPPINGS_DIR / "franka_single.yaml").read_text(encoding="utf-8")
    )
    bad["arms"][0]["source"] = "middle"
    bad_mapper = SemanticMapper(bad)
    with pytest.raises(ValueError, match="unknown controller source"):
        bad_mapper.map(_state())


def test_invalid_config_missing_keys_raises():
    """A mapping entry without the required keys fails fast at load time."""
    with pytest.raises(KeyError):
        SemanticMapper({"arms": [{"name": "arm"}]})


def test_humanoid_base_mapping():
    """Thumbsticks map to planar velocity / yaw with configured scales."""
    mapper = SemanticMapper.from_yaml(MAPPINGS_DIR / "humanoid_dexhand.yaml")
    left = make_hand(thumbstick=(0.4, -0.2))
    right = make_hand(thumbstick=(0.5, 0.0))
    state = ControllerState.from_dict(make_state_dict(left=left, right=right))
    action = mapper.map(state)
    assert action.base_velocity == pytest.approx((0.2, -0.1))  # scale 0.5
    assert action.base_yaw == pytest.approx(0.5)  # scale 1.0


def test_dual_arm_mapping_sources():
    """Dual-arm table wires left/right hands to their respective arms."""
    mapper = SemanticMapper.from_yaml(MAPPINGS_DIR / "franka_dual.yaml")
    left = make_hand(pos=(0.1, 0.2, 0.3), grip=1.0)
    right = make_hand(pos=(0.4, 0.5, 0.6), grip=0.0, trigger=0.9)
    state = ControllerState.from_dict(make_state_dict(left=left, right=right))
    action = mapper.map(state)
    assert action.arm_intents["left_arm"].pose.pos == pytest.approx([0.1, 0.2, 0.3])
    assert action.arm_intents["left_arm"].engaged is True
    assert action.arm_intents["right_arm"].pose.pos == pytest.approx([0.4, 0.5, 0.6])
    assert action.arm_intents["right_arm"].engaged is False
    assert action.gripper_cmds["left_gripper"] == pytest.approx(0.0)
    assert action.gripper_cmds["right_gripper"] == pytest.approx(0.9)
