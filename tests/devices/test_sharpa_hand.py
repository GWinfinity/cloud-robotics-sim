"""Tests for the Genesis-backed Sharpa Wave dexterous hand device.

Genesis runs headless on CPU; the scene is built once per test module via a
module-scoped fixture.
"""

from __future__ import annotations

import pytest

from cloud_robotics_sim.devices import SafetyViolationError, SharpaHandDevice


@pytest.fixture(scope="module")
def hand() -> SharpaHandDevice:
    """Build one shared hand device for the whole module."""
    dev = SharpaHandDevice("hand_test", with_grasp_object=True)
    dev.build()
    yield dev
    dev.close()


def test_build_discovers_22_actuated_joints(hand: SharpaHandDevice) -> None:
    """Build discovers 22 actuated joints."""
    assert hand.n_actuated == 22
    assert len(hand.joint_names) == 22
    assert hand.joint_names[0].startswith("right_thumb_")


def test_joint_target_within_limits_accepted(hand: SharpaHandDevice) -> None:
    """Joint target within limits accepted."""
    name = "right_index_MCP_FE"
    hand.write("grasp_primitive", "open")
    i = hand.joint_names.index(name)
    mid = 0.5 * (hand._lower[i] + hand._upper[i])
    hand.write("joint_targets", {name: float(mid)})
    assert hand._targets[i] == pytest.approx(float(mid))


def test_joint_target_beyond_urdf_limit_rejected(hand: SharpaHandDevice) -> None:
    """Joint target beyond urdf limit rejected."""
    name = "right_index_MCP_FE"
    i = hand.joint_names.index(name)
    with pytest.raises(SafetyViolationError):
        hand.write("joint_targets", {name: float(hand._upper[i] + 1.0)})
    with pytest.raises(SafetyViolationError):
        hand.write("joint_targets", {"not_a_joint": 0.0})


def test_power_grasp_closes_fingers(hand: SharpaHandDevice) -> None:
    """Power grasp closes fingers."""
    hand.write("force_limit_n", 400.0)  # allow stiff position-control spikes
    hand.write("grasp_primitive", "open")
    for _ in range(100):
        hand.step(0.01)
    before = hand.read("joint_positions")
    hand.write("grasp_primitive", "power_grasp")
    for _ in range(200):
        hand.step(0.01)
    after = hand.read("joint_positions")
    flexion = [k for k in after if k.endswith(("_FE", "_PIP", "_DIP", "_IP"))]
    moved = sum(abs(after[k] - before[k]) for k in flexion)
    assert moved > 1.0  # rad, cumulative over flexion joints


def test_contact_and_grasp_state(hand: SharpaHandDevice) -> None:
    """Contact and grasp state."""
    hand.write("grasp_primitive", "power_grasp")
    for _ in range(300):
        hand.step(0.01)
    forces = hand.read("fingertip_forces_n")
    assert set(forces) == {"thumb", "index", "middle", "ring", "pinky"}
    assert hand.read("contact") is True
    assert hand.read("grasp_state") == "grasping"
    assert forces["thumb"] > 0.0


def test_force_limit_guard_trips(hand: SharpaHandDevice) -> None:
    """Force limit guard trips."""
    hand.write("grasp_primitive", "open")
    for _ in range(100):
        hand.step(0.01)
    hand.write("reset_force_limit", True)
    hand.write("force_limit_n", 5.0)
    hand.write("grasp_primitive", "power_grasp")
    tripped = False
    for _ in range(400):
        hand.step(0.01)
        if hand.read("force_limit_tripped"):
            tripped = True
            break
    assert tripped
    hand.write("reset_force_limit", True)
    hand.write("force_limit_n", 400.0)


def test_reference_file_contents(hand: SharpaHandDevice) -> None:
    """Reference file contents."""
    ref = hand.reference_file()
    assert ref["device_class"] == "dexterous_hand"
    names = {c["name"] for c in ref["compliance"]}
    assert any("灵巧手" in n for n in names)
    assert any("协作机器人" in n for n in names)
    assert ref["safety_limits"]["pinch_force_reference_n"] == 140.0


def test_mounted_on_parent_device() -> None:
    """Mounted on parent device."""
    dev = SharpaHandDevice("hand_mounted", mounted_on="franka_lab_01")
    assert dev.mounted_on() == "franka_lab_01"
    assert dev.reference_file()["mounted_on"] == "franka_lab_01"
