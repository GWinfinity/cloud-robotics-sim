"""Clutch retargeting tests: incremental controller -> EE delta mapping."""

from __future__ import annotations

import numpy as np
import pytest
from vr_bridge.core.messages import PoseMsg
from vr_bridge.core.quat_utils import qmul
from vr_bridge.core.retargeting import ClutchRetargeter

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _pose(pos, quat=None) -> PoseMsg:
    return PoseMsg(
        pos=np.asarray(pos, dtype=np.float64),
        quat=IDENTITY.copy() if quat is None else np.asarray(quat, dtype=np.float64),
    )


def test_disengaged_returns_none():
    """Without the clutch held there is no EE target at all."""
    rt = ClutchRetargeter()
    target = rt.update("arm", _pose([0, 0, 0]), _pose([1, 1, 1]), engaged=False)
    assert target is None
    assert rt.is_engaged("arm") is False


def test_engage_anchors_and_holds_ee_pose():
    """The engagement tick anchors both poses and holds the current EE pose."""
    rt = ClutchRetargeter()
    ee = _pose([0.3, 0.0, 0.5])
    target = rt.update("arm", _pose([9.0, 9.0, 9.0]), ee, engaged=True)
    assert rt.is_engaged("arm") is True
    assert target is not None
    assert target.pos == pytest.approx(ee.pos)
    assert target.quat == pytest.approx(ee.quat)


def test_position_delta_maps_with_scale():
    """While engaged, controller displacement * pos_scale becomes EE delta."""
    rt = ClutchRetargeter()
    ctrl0 = _pose([0.5, 0.5, 0.5])
    ee0 = _pose([0.3, 0.0, 0.5])
    rt.update("arm", ctrl0, ee0, engaged=True)

    ctrl1 = _pose([0.6, 0.4, 0.5])  # +0.1 x, -0.1 y
    target = rt.update("arm", ctrl1, ee0, engaged=True, pos_scale=2.0)
    assert target.pos == pytest.approx([0.5, -0.2, 0.5])
    assert target.quat == pytest.approx(IDENTITY)


def test_release_freezes_and_reengage_reanchors():
    """Releasing the clutch freezes; re-engaging re-anchors without a jump."""
    rt = ClutchRetargeter()
    ctrl0 = _pose([0.0, 0.0, 0.0])
    ee0 = _pose([0.3, 0.0, 0.5])
    rt.update("arm", ctrl0, ee0, engaged=True)
    moved = rt.update("arm", _pose([0.1, 0.0, 0.0]), ee0, engaged=True)
    assert moved.pos == pytest.approx([0.4, 0.0, 0.5])

    # Release: operator repositions their hand freely.
    assert rt.update("arm", _pose([5.0, 5.0, 5.0]), ee0, engaged=False) is None

    # Re-engage elsewhere: target starts from the *current* EE pose again,
    # with no accumulated jump from the hand repositioning.
    ee1 = _pose([0.4, 0.0, 0.5])
    target = rt.update("arm", _pose([5.0, 5.0, 5.0]), ee1, engaged=True)
    assert target.pos == pytest.approx(ee1.pos)
    # Small motion from the new anchor maps 1:1 again.
    target = rt.update("arm", _pose([5.05, 5.0, 5.0]), ee1, engaged=True)
    assert target.pos == pytest.approx([0.45, 0.0, 0.5])


def test_rotation_delta_composes_with_ee_anchor():
    """Controller rotation deltas left-compose onto the EE anchor quat."""
    rt = ClutchRetargeter()
    z90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    ee_quat = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # y90

    rt.update("arm", _pose([0, 0, 0], IDENTITY), _pose([0.3, 0, 0.5], ee_quat), True)
    target = rt.update(
        "arm", _pose([0, 0, 0], z90), _pose([0.3, 0, 0.5], ee_quat), True
    )
    expected = qmul(z90, ee_quat)
    # Quaternion sign is irrelevant: compare on the same hemisphere.
    sign = 1.0 if float(np.dot(target.quat, expected)) >= 0 else -1.0
    assert sign * target.quat == pytest.approx(expected, abs=1e-9)
    assert np.linalg.norm(target.quat) == pytest.approx(1.0)


def test_quaternion_double_cover_does_not_jump():
    """A sign-flipped controller quat yields the same target (hemisphere fix)."""
    rt = ClutchRetargeter()
    q = np.array([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    rt.update("arm", _pose([0, 0, 0], IDENTITY), _pose([0.3, 0, 0.5]), True)
    t1 = rt.update("arm", _pose([0, 0, 0], q), _pose([0.3, 0, 0.5]), True)
    rt.reset()
    rt.update("arm", _pose([0, 0, 0], IDENTITY), _pose([0.3, 0, 0.5]), True)
    t2 = rt.update("arm", _pose([0, 0, 0], -q), _pose([0.3, 0, 0.5]), True)
    assert float(np.dot(t1.quat, t2.quat)) == pytest.approx(1.0, abs=1e-9)


def test_per_arm_state_is_independent():
    """Each arm keeps its own clutch anchors."""
    rt = ClutchRetargeter()
    rt.update("left", _pose([0, 0, 0]), _pose([0.1, 0, 0]), engaged=True)
    assert rt.is_engaged("right") is False
    rt.update("right", _pose([1, 1, 1]), _pose([0.2, 0, 0]), engaged=True)
    rt.disengage("left")
    assert rt.is_engaged("left") is False
    assert rt.is_engaged("right") is True


def test_reset_clears_all_arms():
    """reset() disengages every arm and drops last targets."""
    rt = ClutchRetargeter()
    rt.update("arm", _pose([0, 0, 0]), _pose([0.1, 0, 0]), engaged=True)
    rt.reset()
    assert rt.is_engaged("arm") is False
    assert rt.last_target("arm") is None
