"""Unit tests for the CoStream simulation reproduction."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from costream import (
    ActionComposer,
    ControllerCompiler,
    PredictiveBehavior,
    ReactiveBehavior,
    SemanticBehavior,
)
from costream.math_utils import (
    compose,
    identity,
    interpolate,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
    translation_matrix,
)
from costream.specs import ComposeSpec, ControllerProfile, ObjectInfo, SceneSummary, StageSpec


def test_pos_quat_round_trip() -> None:
    pos = np.array([0.1, 0.2, 0.3])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    T = pos_quat_to_matrix(pos, quat)
    p2, q2 = matrix_to_pos_quat(T)
    assert np.allclose(p2, pos)
    assert np.allclose(np.abs(q2), np.abs(quat))


def test_interpolation_identity() -> None:
    T = translation_matrix([1.0, 2.0, 3.0])
    mid = interpolate(T, T, 0.5)
    assert np.allclose(mid[:3, 3], T[:3, 3])


def test_action_composer() -> None:
    composer = ActionComposer()
    anchor = translation_matrix([0.5, 0.0, 0.85])
    nominal = translation_matrix([0.0, 0.0, -0.05])
    residual = translation_matrix([0.002, 0.0, 0.0])
    spec = ComposeSpec()
    cmd = composer.compose(anchor, nominal, residual, spec)
    expected = anchor @ nominal @ residual
    assert np.allclose(cmd, expected)


def test_semantic_anchor() -> None:
    scene = SceneSummary(
        objects={
            "hole": ObjectInfo(
                name="hole",
                pos=np.array([0.4, 0.0, 0.85]),
                quat=np.array([0.0, 1.0, 0.0, 0.0]),
            )
        }
    )
    stage = StageSpec(
        name="insert",
        objective="insert peg",
        task_frame="hole",
        motion="insert",
        duration=1.0,
    )
    WTI = SemanticBehavior().compute_anchor(stage, scene)
    assert np.allclose(WTI[:3, 3], scene.objects["hole"].pos)


def test_predictive_nominal() -> None:
    stage = StageSpec(
        name="insert",
        objective="insert peg",
        task_frame="hole",
        motion="insert",
        duration=2.0,
        relative_start=translation_matrix([0.0, 0.0, 0.0]),
        relative_goal=translation_matrix([0.0, 0.0, 0.05]),
    )
    pred = PredictiveBehavior()
    WTI = identity()
    T0 = pred.compute_nominal(stage, WTI, 0.0)
    T1 = pred.compute_nominal(stage, WTI, 2.0)
    assert np.allclose(T0[:3, 3], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(T1[:3, 3], np.array([0.0, 0.0, 0.05]), atol=1e-6)


def test_reactive_force_correction() -> None:
    reactive = ReactiveBehavior(
        force_gain=1e-3,
        max_lateral_correction=0.01,
        contact_force_threshold=1.0,
    )
    stage = StageSpec(
        name="insert",
        objective="insert peg",
        task_frame="hole",
        motion="insert",
        duration=1.0,
    )
    # Pure lateral force along +x should push correction along -x.
    obs = {
        "object_in_hand_pose": identity(),
        "contact_force": np.array([10.0, 0.0, 0.0]),
    }
    T_tact, info = reactive.compute_residual(stage, obs)
    assert info["contact"]
    assert T_tact[0, 3] < -1e-4


def test_controller_profile_library() -> None:
    p = ControllerCompiler.profile_library("insertion")
    assert p.name == "insertion"
    assert p.force_axis.any()


def test_compose_missing_streams() -> None:
    composer = ActionComposer()
    anchor = translation_matrix([1.0, 0.0, 0.0])
    spec = ComposeSpec()
    cmd = composer.compose(anchor, None, None, spec)
    assert np.allclose(cmd, anchor)
